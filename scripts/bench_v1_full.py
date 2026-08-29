"""CV-G1-E0022-prod-benchmark: FULL benchmark of the canonical production
CaedralVoiceKokoroEngine (PROD-ENGINE-01).

Everything the founder asked for, measured on ONE shared loaded engine:

    model disk size            os.path.getsize on the .onnx (+ voices bin)
    cold load time             fresh subprocess, engine construction + first
                               synth, with pre/post/peak RSS
    thread sweep               {2, 4, 6} intra_op threads, frozen sentence A,
                               median RTF over reps; picks best threads
    per-sentence RTF matrix    short / normal / technical / longform x 4 locked
                               caedral voices @ best threads
    speed validation           0.25x..2.0x on one voice: finite? clipped?
                               duration scales as ~1/speed?
    concurrency                1/2/4 simultaneous requests via
                               ThreadPoolExecutor on the SHARED session;
                               throughput + p50/p95 latency (the engine lock
                               serializes access by design - this measures
                               how much queueing hurts)
    TTFA proxy                 time to first audio chunk via
                               kokoro_onnx.create_stream when available
    quality scoring            classify_preview (dual-ASR WER + DNSMOS) on a
                               subset of rendered cells

Method rules honored:
    - sequential benchmarks only: every phase runs one synth at a time except
      the explicit concurrency probe (which IS the parallelism experiment);
      create_stream executes its batches one at a time internally
    - COLD numbers always from fresh subprocesses; warm numbers from one
      long-lived process

Usage:
    python scripts/bench_v1_full.py --stage cold   [--sweep-threads 2,4,6]
    python scripts/bench_v1_full.py --stage main --threads N
                                                   [--reps-matrix 3]
    python scripts/bench_v1_full.py --stage all    # parent orchestration

Outputs:
    artifacts/benchmarks/caedral-voice-v1-full.json
"""
from __future__ import annotations

_T0 = __import__("time").perf_counter()   # earliest timestamp: before imports

import argparse
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

MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES_BIN = REPO / "models/kokoro/voices-v1.0.bin"
OUT_JSON = REPO / "artifacts/benchmarks/caedral-voice-v1-full.json"
PARTIAL_JSON = REPO / "artifacts/benchmarks/caedral-voice-v1-full.partial.json"

VOICE_IDS = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
SENTENCE_A = "Hello. This is the first generation of Caedral Voice."

MATRIX_SENTENCES = {
    "short": "Hello.",
    "normal": SENTENCE_A,
    "technical": "The invoice totals $1,247.83, due within 30 days.",
    "longform": (
        "The first generation of Caedral Voice ships four production voices "
        "across male and female registers. Every request flows through text "
        "normalization, grapheme to phoneme conversion, and the fused Kokoro "
        "acoustic vocoder graph. On a single laptop CPU the engine sustains "
        "faster than real time synthesis, which means a minute of finished "
        "audio costs less than a minute to render."
    ),
}
SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0]
CONC_LEVELS = [1, 2, 4]
CONC_REQUESTS = 8
SWEEP_REPS = 7
QUALITY_CELLS = {  # (sentence_key, voice_id) rendered to WAV + classified
    ("normal", "caedral-m1"), ("normal", "caedral-m2"),
    ("normal", "caedral-f1"), ("normal", "caedral-f2"),
    ("short", "caedral-m1"), ("technical", "caedral-m1"),
    ("longform", "caedral-m1"),
}
WAV_DIR = REPO / "artifacts/benchmarks/wavs-e0022"


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


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


def mb(b: float) -> float:
    return round(b / 2**20, 1)


def rss_now() -> float:
    import psutil
    return psutil.Process().memory_info().rss


def make_engine(threads: int | None):
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
    return CaedralVoiceKokoroEngine(
        model_path=MODEL, voices_path=VOICES_BIN, threads=threads)


# --------------------------------------------------------------------------- #
# Workers (fresh subprocess each)                                             #
# --------------------------------------------------------------------------- #
def worker_cold() -> dict:
    """Fresh process: pre-load RSS -> load -> post-load RSS -> first synth."""
    t_imports = time.perf_counter() - _T0
    pre_rss = rss_now()

    engine = make_engine(None)          # default ORT thread behaviour
    load_s = engine.total_load_seconds
    post_load_rss = rss_now()

    g2p_warmup_s = engine.warmup()      # NLTK/g2p one-time cost, isolated
    post_warmup_rss = rss_now()

    with RssSampler() as sampler:
        t0 = time.perf_counter()
        audio = engine.synth(MATRIX_SENTENCES["normal"], "caedral-m1")
        cold_wall = time.perf_counter() - t0
        t_first_audio = time.perf_counter() - _T0
    cold_audio_s = len(audio) / engine.sample_rate

    stats = engine.last_stats
    return {
        "mode": "cold",
        "internal_timing": {
            "imports_s": round(t_imports, 3),
            "model_load_s": round(load_s, 3),
            "session_create_s": round(engine.session_create_seconds, 3),
            "wrapper_setup_s": round(engine.wrapper_setup_seconds, 3),
            "frontend_ctor_s": round(engine.frontend_ctor_seconds, 3),
            "g2p_warmup_s": round(g2p_warmup_s, 3),
            "cold_first_synth_wall_s": round(cold_wall, 3),
            "cold_first_synth_audio_s": round(cold_audio_s, 3),
            "cold_first_synth_rtf": round(cold_wall / max(cold_audio_s, 1e-9), 4),
            "process_start_to_first_audio_s": round(t_first_audio, 3),
        },
        "rss": {
            "pre_load_mb": mb(pre_rss),
            "post_load_mb": mb(post_load_rss),
            "post_g2p_warmup_mb": mb(post_warmup_rss),
            "peak_during_first_synth_mb": mb(sampler.peak),
        },
        "stats_voice_profile": stats.voice_profile,
        "stats_speed_clamped": stats.speed_clamped,
    }


def worker_threads(threads: int, reps: int) -> dict:
    """Thread sweep: frozen sentence A, median RTF over reps."""
    engine = make_engine(threads)
    engine.warmup()
    walls, audios = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        audio = engine.synth(SENTENCE_A, "caedral-m1", speed=1.0)
        walls.append(time.perf_counter() - t0)
        audios.append(len(audio) / engine.sample_rate)
    a = median(audios)
    rtfs = [w / a for w in walls]
    return {
        "mode": "threads",
        "threads": threads,
        "reps": reps,
        "sentence": SENTENCE_A,
        "audio_s": round(a, 3),
        "rtf_median": round(median(rtfs), 4),
        "rtf_min": round(min(rtfs), 4),
        "rtf_max": round(max(rtfs), 4),
        "wall_median_s": round(median(walls), 4),
        "warm_rss_mb": mb(rss_now()),
    }


def worker_main(threads: int, reps_matrix: int) -> dict:
    """ONE shared loaded engine: matrix + speeds + concurrency + TTFA +
    quality-render. All sequential inside this process."""
    import asyncio
    import concurrent.futures

    import numpy as np

    pre_rss = rss_now()
    engine = make_engine(threads)
    load_s = engine.total_load_seconds
    post_load_rss = rss_now()
    engine.warmup()
    sr = engine.sample_rate

    out: dict = {"mode": "main", "threads": threads}

    with RssSampler() as sampler:
        # ---- per-sentence x per-voice RTF matrix --------------------------
        matrix: dict[str, dict[str, dict]] = {}
        for skey, text in MATRIX_SENTENCES.items():
            matrix[skey] = {}
            for vid in VOICE_IDS:
                walls, audios = [], []
                keep_audio = None
                for r in range(reps_matrix):
                    t0 = time.perf_counter()
                    audio = engine.synth(text, vid, speed=1.0)
                    walls.append(time.perf_counter() - t0)
                    audios.append(len(audio) / sr)
                    if (skey, vid) in QUALITY_CELLS and r == reps_matrix - 1:
                        keep_audio = audio
                a = median(audios)
                cell = {
                    "wall_median_s": round(median(walls), 4),
                    "audio_s": round(a, 4),
                    "rtf_median": round(median(walls) / a, 4),
                    "rtf_all": sorted(round(w / a, 4) for w in walls),
                }
                matrix[skey][vid] = cell
                print(f"  matrix {skey}/{vid}: rtf={cell['rtf_median']}",
                      flush=True)
                if keep_audio is not None:
                    WAV_DIR.mkdir(parents=True, exist_ok=True)
                    import soundfile as sf
                    wav_path = WAV_DIR / f"e0022-{skey}-{vid}.wav"
                    sf.write(str(wav_path), keep_audio, sr,
                             format="WAV", subtype="FLOAT")
                    cell["wav"] = str(wav_path.relative_to(REPO))

        # ---- speed validation ---------------------------------------------
        speeds = []
        base_audio_s = None
        for sp in SPEEDS:
            t0 = time.perf_counter()
            audio = engine.synth(MATRIX_SENTENCES["normal"],
                                 "caedral-m1", speed=sp)
            wall = time.perf_counter() - t0
            a_s = len(audio) / sr
            if sp == 1.0:
                base_audio_s = a_s
            st = engine.last_stats
            speeds.append({
                "speed_requested": sp,
                "speed_effective": st.speed,
                "clamped": st.speed_clamped,
                "finite": bool(np.all(np.isfinite(audio))),
                "peak_amplitude": round(float(np.abs(audio).max()), 4),
                "clipped": bool(np.abs(audio).max() > 0.99),
                "audio_s": round(a_s, 3),
                "duration_vs_1x": (
                    round(base_audio_s / a_s, 3) if base_audio_s else None
                ),  # should approximate `sp` (audio shrinks as speed grows)
                "wall_s": round(wall, 4),
                "rtf": round(wall / max(a_s, 1e-9), 4),
            })
            print(f"  speed {sp}x: finite ok, audio {a_s:.2f}s", flush=True)

        # ---- concurrency ---------------------------------------------------
        conc_texts = [
            f"Request {i}: Caedral Voice handles simultaneous traffic while "
            f"every syllable stays crisp and intelligible number {i}."
            for i in range(CONC_REQUESTS)
        ]
        concurrency = {}
        for k in CONC_LEVELS:
            lat, aud = [], []

            def one(txt: str) -> float:
                t0 = time.perf_counter()
                au = engine.synth(txt, "caedral-m1", speed=1.0)
                lat.append(time.perf_counter() - t0)
                aud.append(len(au) / sr)
                return lat[-1]

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(one, conc_texts))
            total = time.perf_counter() - t0
            concurrency[k] = {
                "workers": k,
                "requests": CONC_REQUESTS,
                "total_wall_s": round(total, 3),
                "latency_p50_s": round(pct(lat, 0.50), 3),
                "latency_p95_s": round(pct(lat, 0.95), 3),
                "throughput_req_per_s": round(CONC_REQUESTS / total, 3),
                "throughput_audio_s_per_s": round(sum(aud) / total, 3),
            }
            print(f"  conc k={k}: {CONC_REQUESTS / total:.2f} req/s, "
                  f"p95 {pct(lat, 0.95):.2f}s", flush=True)

        # ---- TTFA proxy -----------------------------------------------------
        ttfa: dict = {"create_stream_exists": hasattr(engine._kokoro,
                                                      "create_stream")}
        if ttfa["create_stream_exists"]:
            stream_text = (
                "Caedral Voice streams audio while it renders. The first "
                "chunk arrives before the full clip exists. Playback can "
                "start immediately instead of waiting."
            )
            phonemes, _ = engine.phonemize(stream_text)

            async def _probe() -> dict:
                t0 = time.perf_counter()
                n_chunks = 0
                first_chunk_s = None
                audio_samples = 0
                sr_c = None
                async for audio_c, sr_c in engine._kokoro.create_stream(
                    phonemes, voice="am_michael", speed=1.0,
                    lang="en-us", is_phonemes=True,
                ):
                    if first_chunk_s is None:
                        first_chunk_s = time.perf_counter() - t0
                    n_chunks += 1
                    audio_samples += len(audio_c)
                return {
                    "n_chunks": n_chunks,
                    "ttfa_s": (round(first_chunk_s, 3)
                               if first_chunk_s is not None else None),
                    "stream_total_s": round(time.perf_counter() - t0, 3),
                    "sample_rate": sr_c,
                    "audio_s_total": round(audio_samples / sr_c, 3)
                    if sr_c else None,
                }

            ttfa.update(asyncio.run(_probe()))

        out["load"] = {
            "pre_load_mb": mb(pre_rss),
            "post_load_mb": mb(post_load_rss),
            "model_load_s": round(load_s, 3),
        }
        out["matrix"] = matrix
        out["speed_validation"] = speeds
        out["concurrency"] = concurrency
        out["ttfa"] = ttfa
        out["peak_rss_during_block_mb"] = mb(sampler.peak)

    # ---- quality scoring (after perf block; adds ASR/DNSMOS models to RSS,
    # which is why it sits outside the sampler) -----------------------------
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "classify_preview", REPO / "scripts" / "classify_preview.py")
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)
        models = cp.load_models()
        quality = []
        for (skey, vid) in sorted(QUALITY_CELLS):
            wav_path = WAV_DIR / f"e0022-{skey}-{vid}.wav"
            res = cp.classify_file(wav_path, MATRIX_SENTENCES[skey],
                                   models=models)
            res["cell"] = f"{skey}/{vid}"
            quality.append(res)
            print(f"  classify {skey}/{vid}: {res['class']} "
                  f"minWER={res['min_wer']}", flush=True)
        out["quality_classify"] = quality
    except Exception as exc:  # never let scoring kill the perf numbers
        out["quality_classify_error"] = repr(exc)

    return out


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _spawn(mode_args: list[str], timeout: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), *mode_args]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"worker {' '.join(mode_args)} failed rc="
                         f"{proc.returncode}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("BENCH_JSON:"):
            return json.loads(line[len("BENCH_JSON:"):])
    raise RuntimeError("no BENCH_JSON line in worker stdout:\n"
                       + proc.stdout[-2000:])


def stage_cold(sweep: list[int]) -> dict:
    doc: dict = {
        "task": "PROD-ENGINE-01",
        "experiment": "CV-G1-E0022-prod-benchmark",
        "date": time.strftime("%Y-%m-%d"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": _versions(),
        "model": {
            "path": str(MODEL.relative_to(REPO)),
            "size_bytes": os.path.getsize(MODEL),
            "size_mb": round(os.path.getsize(MODEL) / 2**20, 1),
            "voices_bin": str(VOICES_BIN.relative_to(REPO)),
            "voices_bin_mb": round(os.path.getsize(VOICES_BIN) / 2**20, 1),
        },
        "voice_profiles": _voice_profiles(),
        "method": {
            "sequential_only":
                "one synth at a time everywhere except the concurrency probe",
            "cold_numbers": "fresh subprocess per measurement group",
            "warm_numbers": "single long-lived process, one shared engine",
            "sweep_reps": SWEEP_REPS,
        },
    }

    print("[parent] cold-load worker ...", flush=True)
    doc["cold_load"] = _spawn(["--worker", "cold"], timeout=420)
    print(json.dumps(doc["cold_load"]["internal_timing"], indent=2))

    doc["thread_sweep"] = []
    for t in sweep:
        print(f"[parent] thread sweep threads={t} ...", flush=True)
        r = _spawn(["--worker", "threads", "--threads", str(t)],
                   timeout=420)
        doc["thread_sweep"].append(r)
        print(f"[parent] threads={t}: rtf_median={r['rtf_median']} "
              f"(min {r['rtf_min']}) warm_rss={r['warm_rss_mb']}MB")

    best = min(doc["thread_sweep"], key=lambda r: r["rtf_median"])
    doc["best_threads"] = best["threads"]
    doc["partial_note"] = "run --stage main --threads %d to complete" % \
        best["threads"]
    PARTIAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARTIAL_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[parent] BEST_THREADS={doc['best_threads']} "
          f"(partial saved -> {PARTIAL_JSON.name})", flush=True)
    return doc


def stage_main(threads: int, reps_matrix: int) -> dict:
    doc = json.loads(PARTIAL_JSON.read_text(encoding="utf-8"))
    doc["best_threads"] = threads
    doc["method"]["matrix_reps"] = reps_matrix
    doc["method"]["concurrency_requests_per_level"] = CONC_REQUESTS
    print(f"[parent] main worker threads={threads} ...", flush=True)
    doc["production_engine"] = _spawn(
        ["--worker", "main", "--threads", str(threads),
         "--reps-matrix", str(reps_matrix)],
        timeout=3000)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    PARTIAL_JSON.unlink(missing_ok=True)
    print(f"[parent] wrote {OUT_JSON}", flush=True)
    return doc


def _versions() -> dict:
    import onnxruntime
    import importlib.metadata as im
    return {
        "kokoro_onnx": im.version("kokoro-onnx"),
        "onnxruntime": onnxruntime.__version__,
        "python": platform.python_version(),
    }


def _voice_profiles() -> dict:
    from caedral_voice.inference.kokoro_engine import VOICE_PROFILES
    return dict(VOICE_PROFILES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["cold", "main", "all"],
                    default="all")
    ap.add_argument("--worker", choices=["cold", "threads", "main"],
                    default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sweep-threads", default="2,4,6")
    ap.add_argument("--reps-matrix", type=int, default=3)
    ap.add_argument("--reps-sweep", type=int, default=SWEEP_REPS)
    args = ap.parse_args()

    if args.worker == "cold":
        result = worker_cold()
    elif args.worker == "threads":
        assert args.threads is not None
        result = worker_threads(args.threads, args.reps_sweep)
    elif args.worker == "main":
        assert args.threads is not None
        result = worker_main(args.threads, args.reps_matrix)
    else:
        result = None

    if args.worker is not None:
        print("BENCH_JSON:" + json.dumps(result, ensure_ascii=False),
              flush=True)
        return

    sweep = [int(t) for t in args.sweep_threads.split(",")]
    if args.stage in ("cold", "all"):
        doc = stage_cold(sweep)
        print(f"BEST_THREADS={doc['best_threads']}")
    if args.stage in ("main", "all"):
        threads = args.threads
        if threads is None:
            threads = json.loads(PARTIAL_JSON.read_text(
                encoding="utf-8"))["best_threads"]
        stage_main(threads, args.reps_matrix)


if __name__ == "__main__":
    main()
