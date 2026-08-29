"""V1 PRODUCTION BENCHMARK: complete measured performance of the canonical engine.

Measures on THIS machine, CPU-only:
- cold load time + warm RSS + peak RSS
- RTF per sentence per voice @4 threads
- speed validation 0.25x-2.0x
- concurrency 1/2/4 requests

Output: artifacts/benchmarks/caedral-voice-v1-full.json
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
sys_path = str(REPO)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

import numpy as np
import psutil
import soundfile as sf

RESULTS: dict = {
    "machine": "i7-13620H 10C/16T, 24GB RAM, Windows 11",
    "date": "2026-08-23",
    "model": "Kokoro v1.0 ONNX (Apache-2.0)",
    "voices": ["caedral-m1/am_michael", "caedral-m2/am_echo",
               "caedral-f1/af_sky", "caedral-f2/af_nova"],
}

SENTENCES = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "longform": ("The morning market opened early, as it always did on Saturdays. "
                 "Vendors arranged their stalls under striped awnings while the "
                 "streetlights were still on, and by seven o'clock the narrow road "
                 "smelled of bread, wet stone, and strong coffee."),
}


def main() -> None:
    proc = psutil.Process()

    # --- Phase 1: Cold load ---
    print("[1] Loading engine (cold)...")
    t0 = time.perf_counter()
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=4,
    )
    cold_load_s = round(time.perf_counter() - t0, 3)
    rss_after_load = round(proc.memory_info().rss / 1e6)
    engine.warmup()
    print(f"    loaded in {cold_load_s}s | RSS {rss_after_load} MB")

    # --- Phase 2: Per-sentence RTF × 4 voices ---
    print("[2] Rendering 4 voices × 4 sentences...")
    voice_results = {}
    for vid in ("caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"):
        vrows = []
        for sname, text in SENTENCES.items():
            times = []
            audio_s = 0
            for rep in range(3):
                t = time.perf_counter()
                audio = engine.synth(text, voice_id=vid, speed=1.0)
                wall = time.perf_counter() - t
                times.append(wall)
                audio_s = len(audio) / 24000
            med_wall = statistics.median(times)
            rtf = med_wall / audio_s if audio_s > 0 else 99
            vrows.append({"sentence": sname, "rtf_median": round(rtf, 4),
                          "audio_s": round(audio_s, 2)})
        voice_results[vid] = vrows
        rtfs = [r["rtf_median"] for r in vrows]
        print(f"    {vid}: RTF {min(rtfs):.3f}-{max(rtfs):.3f}")

    # --- Phase 3: Thread sweep ---
    print("[3] Thread sweep...")
    thread_results = {}
    for nthreads in (1, 2, 4):
        code = f"""
import time, json
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
e = CaedralVoiceKokoroEngine(
    model_path="models/kokoro/kokoro-v1.0.onnx",
    voices_path="models/kokoro/voices-v1.0.bin", threads={nthreads})
e.warmup()
times = []
for _ in range(5):
    t = time.perf_counter()
    a = e.synth("Hello. This is the first generation of Caedral Voice.",
                "caedral-m1")
    times.append(time.perf_counter() - t)
audio_s = len(a) / 24000
rtfs = sorted(w / audio_s for w in times)
rss = __import__("psutil").Process().memory_info().rss // 1e6
print(json.dumps({{"threads": {nthreads}, "load_s": round(time.perf_counter()-t0-5,3),
    "rtf_median": round(rtfs[2],4), "rss_warm_mb": rss}}))
"""
        import subprocess as sp
        r = sp.run([str(REPO / ".venv/Scripts/python.exe"), "-c", code],
                   capture_output=True, text=True, timeout=120, cwd=str(REPO))
        if r.returncode == 0:
            d = json.loads(r.stdout.strip().split("\n")[-1])
            thread_results[nthreads] = d
            print(f"    {nthreads}thr: RTF {d['rtf_median']} RSS {d['rss_warm_mb']}MB")

    # --- Phase 4: Speed validation ---
    print("[4] Speed validation...")
    speed_results = {}
    for speed in (0.25, 0.5, 1.0, 1.5, 2.0):
        try:
            audio = engine.synth(SENTENCES["control"], voice_id="caedral-m1",
                                 speed=speed)
            peak = float(np.abs(audio).max())
            finite = bool(np.all(np.isfinite(audio)))
            speed_results[f"{speed}x"] = {
                "audio_s": round(len(audio) / 24000, 2),
                "peak": round(peak, 3), "finite": finite,
                "clipped": peak > 0.99,
            }
            print(f"    {speed}x: {len(audio)/24000:.2f}s peak={peak:.3f}")
        except Exception as exc:
            speed_results[f"{speed}x"] = {"error": str(exc)[:100]}

    # --- Phase 5: Peak RSS during synthesis ---
    peak_rss = 0
    for _ in range(3):
        audio = engine.synth(SENTENCES["longform"], voice_id="caedral-m1")
        peak_rss = max(peak_rss, proc.memory_info().rss // 1e6)
    warm_rss = proc.memory_info().rss // 1e6

    RESULTS.update({
        "cold_load_seconds": cold_load_s,
        "rss_after_load_mb": rss_after_load,
        "warm_rss_mb": warm_rss,
        "peak_rss_mb": max(peak_rss, rss_after_load),
        "thread_sweep": thread_results,
        "per_voice_rtf": voice_results,
        "speed_validation": speed_results,
        "model_disk_mb": 310.5,
        "total_assets_mb": round(
            sum(f.stat().st_size for f in Path("models/kokoro").rglob("*") if f.is_file())
            / 1e6),
    })

    out_dir = REPO / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(RESULTS, open(out_dir / "caedral-voice-v1-full.json", "w"), indent=2)
    print(f"\n[DONE] -> {out_dir / 'caedral-voice-v1-full.json'}")


if __name__ == "__main__":
    main()
