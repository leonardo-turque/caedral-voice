"""CV-G1-E0027-memory Part B: ONNX Runtime session memory configs.

One fresh subprocess per config (session options are fixed at creation).
Each worker builds the session with its config's SessionOptions, loads the
full engine stack (kokoro_onnx wrapper + Caedral TextFrontend), pays
warmup(), then renders one frozen sentence 5x and reports:

    rss_after_load_mb   steady RSS after load+warmup, BEFORE timed reps
    warm_rss_mb         steady RSS after the 5 reps
    peak_rss_mb         peak during reps (10 ms sampler)
    rtf_median          median wall / audio over the 5 reps

Usage:
    python config_matrix.py --worker CONFIG_KEY   # internal child mode
    python config_matrix.py                       # parent: all configs
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"
OUT = Path(__file__).resolve().parent / "results.json"

SENTENCE = "Hello. This is first generation of Caedral Voice."
VOICE = "af_heart"
# Primary matrix runs at the production default (4). Set E0027_THREADS to
# run an appendix config under a different thread count.
THREADS = int(os.environ.get("E0027_THREADS", "4"))
REPS = 5

CONFIGS: dict[str, dict] = {
    # Production baseline: ORT defaults for every knob.
    "base": {},
    "mem_pattern_off": {"enable_mem_pattern": False},
    "cpu_arena_off": {"enable_cpu_mem_arena": False},
    "arena_same": {"arena_extend_strategy": "kSameAsRequested"},
    "opt_disabled": {"graph_optimization_level": "ORT_DISABLE_ALL"},
    "combo_a": {"enable_mem_pattern": False,
                "arena_extend_strategy": "kSameAsRequested"},
    "combo_b": {"enable_mem_pattern": False, "enable_cpu_mem_arena": False,
                "arena_extend_strategy": "kSameAsRequested"},
}


def build_options(spec: dict):
    import onnxruntime as rt
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    if spec.get("enable_mem_pattern") is False:
        opts.enable_mem_pattern = False
    if spec.get("enable_cpu_mem_arena") is False:
        opts.enable_cpu_mem_arena = False
    if spec.get("arena_extend_strategy") == "kSameAsRequested":
        opts.add_session_config_entry("arena.extend_strategy",
                                      "kSameAsRequested")
    if spec.get("graph_optimization_level") == "ORT_DISABLE_ALL":
        opts.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return opts


def mem_mb() -> tuple[float, float]:
    import psutil
    mi = psutil.Process().memory_info()
    return mi.rss / 2**20, mi.private / 2**20


class RssSampler:
    def __init__(self):
        import psutil
        self._proc = psutil.Process()
        mi = self._proc.memory_info()
        self.peak_rss = mi.rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                mi = self._proc.memory_info()
                if mi.rss > self.peak_rss:
                    self.peak_rss = mi.rss
            except Exception:
                pass
            self._stop.wait(0.01)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def worker(config_key: str) -> dict:
    import numpy as np  # noqa: F401
    import onnxruntime as rt
    import psutil
    ambient_cpu_pct = psutil.cpu_percent(interval=0.5)
    from kokoro_onnx import Kokoro
    from caedral_voice.text import TextFrontend
    from caedral_voice.text.kokoro_map import (
        caedral_tokens_to_kokoro,
        kokoro_vocab_set,
    )

    spec = CONFIGS[config_key]
    opts = build_options(spec)
    t0 = time.perf_counter()
    sess = rt.InferenceSession(str(MODEL), sess_options=opts,
                               providers=["CPUExecutionProvider"])
    create_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    k = Kokoro.from_session(sess, str(VOICES))
    wrapper_s = time.perf_counter() - t0
    fe = TextFrontend("lexicon+g2pen")
    vocab_set = kokoro_vocab_set(k.tokenizer.vocab)

    def phonemize(text: str) -> str:
        res = fe.process(text)
        parts = [caedral_tokens_to_kokoro(list(w.selected), vocab_set)[0]
                 for w in res.words]
        out = " ".join(parts)
        if text.rstrip().endswith("."):
            out += "."
        return out

    # Warmup: pay NLTK g2p construction + ORT graph init before measuring.
    t0 = time.perf_counter()
    phonemes = phonemize(SENTENCE)
    style = k.get_voice_style(VOICE)
    audio0, _ = k.create(phonemes, voice=VOICE, speed=1.0,
                         lang="en-us", is_phonemes=True)
    warmup_s = time.perf_counter() - t0
    rss_after_load, commit_after_load = mem_mb()

    walls, audio_s_list = [], []
    with RssSampler() as sampler:
        for _ in range(REPS):
            t0 = time.perf_counter()
            audio, _ = k.create(phonemes, voice=VOICE, speed=1.0,
                                lang="en-us", is_phonemes=True)
            walls.append(time.perf_counter() - t0)
            audio_s_list.append(len(audio) / 24000)
    warm_rss, warm_commit = mem_mb()

    med_wall = statistics.median(walls)
    audio_s = statistics.median(audio_s_list)
    return {
        "config": config_key,
        "spec": spec,
        "threads_intra_op": THREADS,
        "ambient_cpu_pct": round(ambient_cpu_pct, 1),
        "session_create_s": round(create_s, 2),
        "wrapper_load_s": round(wrapper_s, 2),
        "warmup_first_synth_s": round(warmup_s, 3),
        "audio_s": round(audio_s, 3),
        "walls_s": [round(w, 4) for w in walls],
        "wall_median_s": round(med_wall, 4),
        "rtf_median": round(med_wall / max(audio_s, 1e-9), 4),
        "rss_after_load_mb": round(rss_after_load, 1),
        "commit_after_load_mb": round(commit_after_load, 1),
        "warm_rss_mb": round(warm_rss, 1),
        "warm_commit_mb": round(warm_commit, 1),
        "peak_rss_during_reps_mb": round(sampler.peak_rss / 2**20, 1),
    }


def parent() -> dict:
    import platform

    result = {
        "experiment": "CV-G1-E0027-memory",
        "part": "B-session-memory-configs",
        "date": time.strftime("%Y-%m-%d"),
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor(),
                    "logical_cpus": __import__("os").cpu_count()},
        "sentence": SENTENCE,
        "voice": VOICE,
        "threads_intra_op": THREADS,
        "reps_per_config": REPS,
        "method_note": (
            "Fresh subprocess per config; SessionOptions fixed at creation. "
            "RTF = median of 5 reps on frozen sentence after warmup synth. "
            "Warm RSS = steady-state working set after reps; peak sampled "
            "at 10 ms."),
        "rtf_gate": 0.55,
        "configs": [],
    }
    try:
        import onnxruntime
        import importlib.metadata
        result["versions"] = {
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "kokoro_onnx": importlib.metadata.version("kokoro-onnx"),
        }
    except Exception:
        pass

    for key in CONFIGS:
        print(f"[parent] config {key} ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", key]
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=900, env={**__import__("os").environ,
                                                "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"worker {key} failed rc={proc.returncode}")
        row = json.loads(_extract_json_line(proc.stdout))
        row["passes_rtf_gate"] = row["rtf_median"] <= 0.55
        result["configs"].append(row)
        print(f"[parent] {key}: warm_rss {row['warm_rss_mb']}MB "
              f"(load {row['rss_after_load_mb']}MB), "
              f"RTF {row['rtf_median']}, peak {row['peak_rss_during_reps_mb']}MB",
              flush=True)

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[parent] wrote {OUT}", flush=True)
    return result


def _extract_json_line(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith("MATRIX_JSON:"):
            return line[len("MATRIX_JSON:"):]
    raise RuntimeError("no MATRIX_JSON line in stdout:\n" + stdout[-2000:])


def main():
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=str, default=None, metavar="KEY")
    args = ap.parse_args()
    if args.worker is not None:
        res = worker(args.worker)
        print("MATRIX_JSON:" + json.dumps(res, ensure_ascii=False),
              flush=True)
    else:
        parent()


if __name__ == "__main__":
    main()
