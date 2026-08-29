"""Final CPU benchmark: threads sweep + RSS + cold load + speed sanity."""
from __future__ import annotations

import json
import os
import subprocess as sp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
results: dict = {
    "machine": "i7-13620H 10C/16T, 24GB RAM, Windows 11, CPU-only",
    "date": "2026-08-23",
}


def subprocess_run(code: str, env_over: dict | None = None) -> dict | None:
    env = {**os.environ, **(env_over or {})}
    try:
        r = sp.run([str(REPO / ".venv/Scripts/python.exe"), "-c", code],
                   capture_output=True, text=True, timeout=120,
                   cwd=str(REPO), env=env)
        return json.loads(r.stdout.strip().split("\n")[-1])
    except Exception:
        return None


# Thread sweep on ONE frozen sentence
for threads in (1, 2, 4):
    code = f"""
import time, json
t0 = time.perf_counter()
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
e = CaedralVoiceKokoroEngine(
    model_path="models/kokoro/kokoro-v1.0.onnx",
    voices_path="models/kokoro/voices-v1.0.bin", threads={threads})
load_s = time.perf_counter() - t0
e.synth("Warm.", "am_michael")
times = []
for _ in range(5):
    t = time.perf_counter()
    a = e.synth("Hello. This is the first generation of Caedral Voice.",
                "am_michael", speed=1.0)
    times.append(time.perf_counter() - t)
audio_s = len(a) / 24000
rtfs = sorted(w / audio_s for w in times)
import psutil
rss_mb = round(psutil.Process().memory_info().rss / 1e6)
print(json.dumps({{"threads": {threads}, "load_s": round(load_s,3),
    "rtf_median": round(rtfs[2],4), "rtf_min": round(rtfs[0],4),
    "audio_s": round(audio_s,2), "rss_warm_mb": rss_mb}}))
"""
    r = subprocess_run(code, {"OMP_NUM_THREADS": str(threads),
                              "MKL_NUM_THREADS": str(threads)})
    if r:
        results[f"threads_{threads}"] = r
        print(f"threads={threads}: RTF median {r['rtf_median']} "
              f"load {r['load_s']}s RSS {r.get('rss_warm_mb')}MB")

# Speed sanity
for speed in (0.5, 1.0, 1.5, 2.0):
    code = f"""
import json
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
e = CaedralVoiceKokoroEngine(
    model_path="models/kokoro/kokoro-v1.0.onnx",
    voices_path="models/kokoro/voices-v1.0.bin", threads=4)
a = e.synth("Hello world test.", "am_michael", speed={speed})
peak = float(abs(a).max())
import numpy as np
finite = bool(np.all(np.isfinite(a)))
clip = bool(peak > 0.99)
print(json.dumps({{"speed": {speed}, "audio_s": round(len(a)/24000,2),
    "peak": round(peak,3), "finite": finite, "clipped": clip}}))
"""
    r = subprocess_run(code, {})
    if r:
        results[f"speed_{speed}x"] = r
        print(f"speed {speed}x: {r}")

out_dir = REPO / "artifacts" / "benchmarks"
out_dir.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out_dir / "caedral-voice-cpu.json", "w"), indent=2)
print("\nDONE -> artifacts/benchmarks/caedral-voice-cpu.json")
