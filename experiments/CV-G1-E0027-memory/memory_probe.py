"""CV-G1-E0027-memory Part A: decompose the warm RSS of the Kokoro engine.

Ladder design: each rung N is a FRESH subprocess that loads stages 0..N
cumulatively, so the marginal cost of stage N is the RSS delta between
consecutive fresh processes (no allocator-history contamination between
stages). One extra single-process walk records how the production engine
accumulates memory stage by stage inside one heap.

Stages:
    0 imports        python + numpy + onnxruntime + psutil
    1 session        + rt.InferenceSession(kokoro-v1.0.onnx, threads=4)
    2 wrapper        + kokoro_onnx.Kokoro.from_session (voices np.load +
                     tokenizer vocab)
    3 frontend       + caedral TextFrontend ctor (cmudict lexicon JSON)
    4 g2p_warmup     + first phonemize -> g1p_en NLTK LSTM construction
    5 synth          + ONE short synthesis, peak-sampled at 10 ms

Usage:
    python memory_probe.py --worker N          # internal child mode
    python memory_probe.py                     # parent: ladder + walk
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"
OUT = Path(__file__).resolve().parent / "memory-breakdown.json"
SENTENCE = "Hello. This is the first generation of Caedral Voice."
VOICE = "af_heart"
THREADS = 4  # production CLI default


def rss_mb() -> tuple[float, float]:
    import psutil
    mi = psutil.Process().memory_info()
    return mi.rss / 2**20, mi.private / 2**20  # rss, commit charge


class RssSampler:
    """Sample this process's RSS/commit every ~10 ms until stopped."""

    def __init__(self):
        import psutil
        self._proc = psutil.Process()
        mi = self._proc.memory_info()
        self.peak_rss = mi.rss
        self.peak_commit = mi.private
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                mi = self._proc.memory_info()
                if mi.rss > self.peak_rss:
                    self.peak_rss = mi.rss
                if mi.private > self.peak_commit:
                    self.peak_commit = mi.private
            except Exception:
                pass  # keep sampler alive on transient psutil errors
            self._stop.wait(0.01)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def worker(max_stage: int) -> dict:
    """Fresh process: load stages 0..max_stage cumulatively."""
    import numpy as np  # noqa: F401
    import onnxruntime as rt
    import psutil

    stages = []
    r0, c0 = rss_mb()
    stages.append({"stage": "imports", "rss_mb": round(r0, 1),
                   "commit_mb": round(c0, 1)})

    sess = None
    if max_stage >= 1:
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        t0 = time.perf_counter()
        sess = rt.InferenceSession(str(MODEL), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
        create_s = time.perf_counter() - t0
        r, c = rss_mb()
        stages.append({"stage": "session", "rss_mb": round(r, 1),
                       "commit_mb": round(c, 1),
                       "create_s": round(create_s, 2)})

    if max_stage >= 2:
        from kokoro_onnx import Kokoro
        t0 = time.perf_counter()
        k = Kokoro.from_session(sess, str(VOICES))
        load_s = time.perf_counter() - t0
        r_lazy, c_lazy = rss_mb()
        # Production always materializes the requested voice before inference
        # (NpzFile is lazy: each getitem decompresses that voice's 0.5 MB
        # slice from disk).
        style = k.get_voice_style(VOICE)
        r_mat, c_mat = rss_mb()
        stages.append({
            "stage": "wrapper_voices", "rss_mb": round(r_mat, 1),
            "commit_mb": round(c_mat, 1), "load_s": round(load_s, 2),
            "rss_after_from_session_lazy_mb": round(r_lazy, 1),
            "rss_after_voice_style_mb": round(r_mat, 1),
            "voices_type": type(k.voices).__name__,
            "n_voices": len(k.voices.files),
            "one_voice_shape": list(style.shape),
            "one_voice_mb": round(style.nbytes / 2**20, 2),
            "all_voices_disk_mb": round(VOICES.stat().st_size / 2**20, 1),
        })

    if max_stage >= 3:
        from caedral_voice.text import TextFrontend
        t0 = time.perf_counter()
        fe = TextFrontend("lexicon+g2pen")
        ctor_s = time.perf_counter() - t0
        r, c = rss_mb()
        stages.append({"stage": "frontend_ctor", "rss_mb": round(r, 1),
                       "commit_mb": round(c, 1),
                       "ctor_s": round(ctor_s, 2)})

    if max_stage >= 4:
        from caedral_voice.text.kokoro_map import (
            caedral_tokens_to_kokoro,
            kokoro_vocab_set,
        )
        vocab_set = kokoro_vocab_set(k.tokenizer.vocab)

        def map_sentence(text: str) -> str:
            res = fe.process(text)
            parts = [caedral_tokens_to_kokoro(list(w.selected), vocab_set)[0]
                     for w in res.words]
            out = " ".join(parts)
            if text.rstrip().endswith("."):
                out += "."
            return out

        # The frozen sentence is fully in-lexicon, so it never touches the
        # g2p_en NLTK fallback. Production warmup() deliberately uses
        # out-of-lexicon "Hello caedral." to pay that one-time construction.
        t0 = time.perf_counter()
        _ = map_sentence(SENTENCE)
        inlex_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _oov = map_sentence("Hello caedral.")   # forces g2p_en NLTK build
        oov_s = time.perf_counter() - t0
        r, c = rss_mb()
        stages.append({"stage": "g2p_warmup", "rss_mb": round(r, 1),
                       "commit_mb": round(c, 1),
                       "in_lexicon_phonemize_s": round(inlex_s, 3),
                       "oov_g2pen_warmup_s": round(oov_s, 3),
                       "note": "OOV word forces g2p_en NLTK LSTM build"})

    if max_stage >= 5:
        style = k.get_voice_style(VOICE)
        phonemes = map_sentence(SENTENCE)
        tokens = k.tokenizer.tokenize(phonemes)
        with RssSampler() as sampler:
            t0 = time.perf_counter()
            audio, _ = k.create(phonemes, voice=VOICE, speed=1.0,
                                lang="en-us", is_phonemes=True)
            synth_s = time.perf_counter() - t0
        r, c = rss_mb()
        audio_s = len(audio) / 24000
        stages.append({
            "stage": "synth", "rss_mb": round(r, 1), "commit_mb": round(c, 1),
            "synth_s": round(synth_s, 3), "audio_s": round(audio_s, 3),
            "rtf_one_shot": round(synth_s / audio_s, 3),
            "peak_rss_during_synth_mb": round(sampler.peak_rss / 2**20, 1),
            "peak_commit_during_synth_mb":
                round(sampler.peak_commit / 2**20, 1),
        })

    return {"worker_max_stage": max_stage, "threads": THREADS,
            "stages": stages}


def parent(ladder: bool, walk: bool) -> dict:
    import platform

    result = {
        "experiment": "CV-G1-E0027-memory",
        "part": "A-memory-decomposition",
        "date": time.strftime("%Y-%m-%d"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": __import__("os").cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
        },
        "model": {
            "path": "models/kokoro/kokoro-v1.0.onnx",
            "size_mb": round(MODEL.stat().st_size / 2**20, 1),
            "voices_size_mb": round(VOICES.stat().st_size / 2**20, 1),
        },
        "sentence": SENTENCE,
        "voice": VOICE,
        "method_note": (
            "Ladder: fresh subprocess per rung loads stages 0..N; marginal "
            "cost of stage N = RSS(N) - RSS(N-1) across clean processes. "
            "Walk: one process accumulates all stages, recording RSS after "
            "each — mirrors production long-lived engine heap."),
        "ladder": [],
        "walk": None,
    }

    try:
        import onnxruntime
        result["versions"]["onnxruntime"] = onnxruntime.__version__
        import importlib.metadata
        result["versions"]["kokoro_onnx"] = importlib.metadata.version(
            "kokoro-onnx")
    except Exception:
        pass

    def run_worker(max_stage: int) -> dict:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", str(max_stage)]
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=900,
                              env={**sys_orig_env(), "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            raise SystemExit(f"worker stage {max_stage} failed:\n"
                             f"{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
        return json.loads(_extract_json_line(proc.stdout))

    if ladder:
        for n in range(0, 6):
            print(f"[parent] ladder rung {n} ...", flush=True)
            result["ladder"].append(run_worker(n))

    if walk:
        print("[parent] cumulative walk ...", flush=True)
        # The walk IS worker(max_stage=5); its per-stage rows already record
        # RSS after every boundary in ONE process.
        result["walk"] = run_worker(5)

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[parent] wrote {OUT}", flush=True)
    return result


def sys_orig_env():
    import os
    return dict(os.environ)


def _extract_json_line(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROBE_JSON:"):
            return line[len("PROBE_JSON:"):]
    raise RuntimeError("no PROBE_JSON line in stdout:\n" + stdout[-2000:])


def main():
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=int, default=None, metavar="N")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--no-walk", action="store_true")
    args = ap.parse_args()

    if args.worker is not None:
        res = worker(args.worker)
        print("PROBE_JSON:" + json.dumps(res, ensure_ascii=False),
              flush=True)
    else:
        parent(ladder=not args.no_ladder, walk=not args.no_walk)


if __name__ == "__main__":
    main()
