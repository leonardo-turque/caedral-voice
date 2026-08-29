"""CV-G1-E0021-engine-profile: phase-level profiler for the winning
production path (kokoro_onnx + Caedral frontend).

Phases probed for one synthesis call:
    frontend_phonemize      Caedral TextFrontend (TN+G2P) -> vocab mapping
    frontend_cache_lookup   dict hit on the text-hash cache (engine fast path)
    tokenize                Kokoro char tokenizer -> ids
    session_run_acoustic    pure sess.run on precomputed ids/style/speed
    vocoder_decode_if_any   None for this export: v1.0 fuses acoustic+vocoder
                            into ONE output ("audio"); verified at runtime
    wav_serialize           soundfile WAV encode to an in-memory buffer

Method:
    - One fresh subprocess per thread count {2,4}: true cold first-call per
      config, no cross-contamination, thermal order fixed (2 then 4).
    - COLD = first engine.synth() after construction + NLTK/g2p warmup
      (i.e. ORT graph init/allocations are the cold cost, not the one-time
      lexicon/LSTM build, which is measured separately as g2p_warmup_s).
    - WARM = median of --reps (default 10) synth() calls per sentence.
    - Peak RSS sampled at 10 ms by a psutil thread during the whole block.
    - Process startup->first-audio: the worker itself reports monotonic
      deltas from interpreter start; the parent additionally wall-clocks
      the whole spawn.

Usage:
    python scripts/profile_kokoro.py                 # parent, writes profile.json
    python scripts/profile_kokoro.py --worker 2      # internal child mode
"""
from __future__ import annotations

_T0 = __import__("time").perf_counter()   # earliest timestamp: before imports

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
}
VOICE = "af_heart"
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES_BIN = REPO / "models/kokoro/voices-v1.0.bin"
OUT_DIR = REPO / "experiments/CV-G1-E0021-engine-profile"


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


class RssSampler:
    """Sample this process's RSS every ~10 ms until stopped."""

    def __init__(self):
        import psutil
        self._proc = psutil.Process()
        self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak:
                    self.peak = rss
            except Exception:
                pass
            self._stop.wait(0.01)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def worker(threads: int, reps: int, startup_probe_only: bool) -> dict:
    """Run in a fresh process; returns the result dict for one thread config."""
    t_imports = time.perf_counter() - _T0

    import numpy as np
    import soundfile as sf
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine

    engine = CaedralVoiceKokoroEngine(MODEL, VOICES_BIN, threads=threads)
    k = engine._kokoro
    output_names = [o.name for o in engine._kokoro.sess.get_outputs()]
    baseline_rss = __import__("psutil").Process().memory_info().rss

    if startup_probe_only:
        audio = engine.synth("Hello.", VOICE)
        return {
            "mode": "startup_probe",
            "threads": threads,
            "internal": {
                "imports_s": t_imports,
                "session_create_s": engine.session_create_seconds,
                "wrapper_setup_s": engine.wrapper_setup_seconds,
                "frontend_ctor_s": engine.frontend_ctor_seconds,
                "total_load_s": engine.total_load_seconds,
                "t_after_load": time.perf_counter() - _T0,
                "t_first_audio": time.perf_counter() - _T0,
            },
            "audio_samples": int(len(audio)),
        }

    # Pay the one-time NLTK g2p_en construction up front so COLD isolates
    # ORT graph init instead of mixing in a ~12 s lexicon build.
    g2p_warmup_s = engine.warmup()

    with RssSampler() as rss:
        # ---- COLD: very first real synth of this session ----
        cold_name = next(iter(SENTENCES))
        t0 = time.perf_counter()
        audio = engine.synth(SENTENCES[cold_name], VOICE)
        cold_wall = time.perf_counter() - t0
        cold_audio_s = len(audio) / engine.sample_rate

        # ---- WARM: reps x sentences ----
        warm = {}
        for name, text in SENTENCES.items():
            walls, audios = [], []
            for _ in range(reps):
                t0 = time.perf_counter()
                audio = engine.synth(text, VOICE)
                walls.append(time.perf_counter() - t0)
                audios.append(len(audio) / engine.sample_rate)
            warm[name] = {"walls": walls, "audio_s": median(audios)}

        # ---- Phase breakdown probes (uncached frontend each rep) ----
        phases = {p: [] for p in (
            "frontend_phonemize", "frontend_cache_lookup", "tokenize",
            "session_run_acoustic", "wav_serialize", "create_total_is_phonemes",
        )}
        style_vec = k.get_voice_style(VOICE)
        for name, text in SENTENCES.items():
            for _ in range(5):
                t0 = time.perf_counter()
                phonemes, _skipped = engine.phonemize(text, use_cache=False)
                phases["frontend_phonemize"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                engine.phonemize(text, use_cache=True)
                phases["frontend_cache_lookup"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                tokens = k.tokenizer.tokenize(phonemes)
                phases["tokenize"].append(time.perf_counter() - t0)

                style = style_vec[min(len(tokens), len(style_vec)) - 1]
                t0 = time.perf_counter()
                k._infer(tokens, style, 1.0)
                phases["session_run_acoustic"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                audio2, _ = k.create(phonemes, voice=VOICE, speed=1.0,
                                     lang="en-us", is_phonemes=True)
                phases["create_total_is_phonemes"].append(
                    time.perf_counter() - t0
                )

                buf = io.BytesIO()
                t0 = time.perf_counter()
                sf.write(buf, audio2, engine.sample_rate,
                         format="WAV", subtype="FLOAT")
                phases["wav_serialize"].append(time.perf_counter() - t0)

    peak_rss = rss.peak
    per_sentence = []
    for name in SENTENCES:
        w = warm[name]
        med = median(w["walls"])
        per_sentence.append({
            "sentence": name, "reps": reps,
            "median_wall_s": round(med, 4),
            "audio_s": round(w["audio_s"], 4),
            "rtf_median": round(med / w["audio_s"], 4),
        })
    all_walls = [w for name in SENTENCES for w in warm[name]["walls"]]
    all_audio = [warm[n]["audio_s"] for n in SENTENCES]
    head_wall, head_audio = median(all_walls), median(all_audio)

    return {
        "mode": "worker",
        "threads": threads,
        "internal_startup": {
            "imports_s": round(t_imports, 3),
            "session_create_s": round(engine.session_create_seconds, 3),
            "wrapper_setup_s": round(engine.wrapper_setup_seconds, 3),
            "frontend_ctor_s": round(engine.frontend_ctor_seconds, 3),
            "g2p_warmup_s": round(g2p_warmup_s, 3),
            "t_first_audio_cold_call_done": round(
                time.perf_counter() - _T0, 3),
        },
        "cold": {
            "sentence": cold_name,
            "wall_s": round(cold_wall, 4),
            "audio_s": round(cold_audio_s, 4),
            "rtf": round(cold_wall / cold_audio_s, 4),
        },
        "warm": {
            "per_sentence": per_sentence,
            "headline_median_wall_s": round(head_wall, 4),
            "headline_audio_s": round(head_audio, 4),
            "headline_rtf": round(head_wall / head_audio, 4),
        },
        "phases_median_s": {
            k2: round(median(v), 6) for k2, v in phases.items()
        },
        "vocoder_decode_if_any": None if output_names == ["audio"]
        else "separate-phase-not-implemented",
        "model_outputs": output_names,
        "rss": {
            "baseline_after_load_mb": round(baseline_rss / 2**20, 1),
            "peak_during_synth_mb": round(peak_rss / 2**20, 1),
            "delta_mb": round((peak_rss - baseline_rss) / 2**20, 1),
        },
    }


def parent(thread_counts, reps, out_path: Path) -> dict:
    import kokoro_onnx
    import onnxruntime

    profile = {
        "experiment": "CV-G1-E0021-engine-profile",
        "date": time.strftime("%Y-%m-%d"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cpus": os.cpu_count(),
        },
        "versions": {
            "kokoro_onnx": importlib_metadata_version("kokoro-onnx"),
            "onnxruntime": onnxruntime.__version__,
            "python": platform.python_version(),
        },
        "model": {
            "path": str(MODEL.relative_to(REPO)),
            "size_mb": round(MODEL.stat().st_size / 2**20, 1),
            "voices": str(VOICES_BIN.relative_to(REPO)),
            "voice_used": VOICE,
            "fused_acoustic_vocoder": True,
        },
        "sentences": SENTENCES,
        "method": {
            "cold_definition":
                "first engine.synth() after construction + NLTK g2p warmup "
                "(isolates ORT graph init); g2p_warmup_s reported separately",
            "warm_definition": f"median of {reps} synth() calls per sentence",
            "phase_probes_per_sentence": 5,
            "thread_order_note":
                "configs run sequentially in fresh subprocesses, order "
                + "->".join(str(t) for t in thread_counts)
                + "; RTF variance across runs is thermal/contention-sensitive",
        },
        "thread_configs": [],
    }

    for threads in thread_counts:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", str(threads), "--reps", str(reps)]
        print(f"[parent] spawning worker threads={threads} ...", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=600,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        spawn_wall = time.perf_counter() - t0
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"worker threads={threads} failed "
                             f"(rc={proc.returncode})")
        result = json.loads(_extract_json_line(proc.stdout))
        result["parent_spawn_wall_total_s"] = round(spawn_wall, 3)
        profile["thread_configs"].append(result)
        w = result["warm"]["headline_rtf"]
        print(f"[parent] threads={threads}: cold {result['cold']['wall_s']}s, "
              f"warm median {result['warm']['headline_median_wall_s']}s, "
              f"RTF {w}, peak RSS "
              f"{result['rss']['peak_during_synth_mb']}MB", flush=True)

    # Dedicated startup->first-audio probe: fresh process renders 1 short
    # sentence; parent wall-clocks the entire spawn.
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", str(thread_counts[-1]), "--startup-probe"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=600,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit("startup probe failed")
    probe_internal = json.loads(_extract_json_line(proc.stdout))["internal"]
    profile["startup_to_first_audio"] = {
        "spawn_wall_total_s": round(time.perf_counter() - t0, 3),
        "child_imports_s": probe_internal["imports_s"],
        "child_model_load_s": probe_internal["total_load_s"],
        "child_t_first_audio_s": probe_internal["t_first_audio"],
        "note": "short sentence 'Hello.' rendered once, then exit",
    }
    print(f"[parent] startup->first-audio: "
          f"{profile['startup_to_first_audio']['spawn_wall_total_s']}s wall",
          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[parent] wrote {out_path}", flush=True)
    return profile


def importlib_metadata_version(dist):
    import importlib.metadata
    return importlib.metadata.version(dist)


def _extract_json_line(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROFILE_JSON:"):
            return line[len("PROFILE_JSON:"):]
    raise RuntimeError("no PROFILE_JSON line in worker stdout:\n"
                       + stdout[-2000:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=int, default=None, metavar="THREADS",
                    help="internal: run one worker pass with THREADS threads")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", default="2,4",
                    help="comma list of thread counts (parent mode)")
    ap.add_argument("--startup-probe", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "profile.json"))
    args = ap.parse_args()

    if args.worker is not None:
        result = worker(args.worker, args.reps, args.startup_probe)
        # Single tagged line so the parent can parse reliably amid logs.
        print("PROFILE_JSON:" + json.dumps(result, ensure_ascii=False),
              flush=True)
    else:
        counts = [int(t) for t in args.threads.split(",")]
        profile = parent(counts, args.reps, Path(args.out))
        print(json.dumps({
            t["threads"]: {
                "cold_rtf": t["cold"]["rtf"],
                "warm_rtf": t["warm"]["headline_rtf"],
                "peak_rss_mb": t["rss"]["peak_during_synth_mb"],
            } for t in profile["thread_configs"]
        }, indent=2))


if __name__ == "__main__":
    main()
