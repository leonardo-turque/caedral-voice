"""In-container FIRE TEST probe: cold load + N sequential ~30 s generations.

Runs INSIDE a cgroup-constrained container. Reads cgroup v2 enforcement values
from /sys/fs/cgroup (never trusts CLI flags), measures cold-start timeline,
RSS ladder, kernel VmHWM peaks, CPU time, throttling, and memory events, then
runs SEQUENTIAL generations of the canonical frozen input through the frozen
engine's public API (phonemize + infer_from_phonemes — same frontend + fused
graph the production service uses; no ASR/DNSMOS/acceptance tooling).

Output: JSON to /out/probe-results.json (mounted volume).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

T_SCRIPT = time.perf_counter()

CG = Path("/sys/fs/cgroup")
OUT = Path(os.environ.get("FIRE_OUT", "/out"))
CANONICAL = Path(os.environ.get("FIRE_INPUT", "/app/canonical-input.txt"))
REPS = int(os.environ.get("FIRE_REPS", "10"))


def read_cg(name: str) -> str:
    try:
        return (CG / name).read_text().strip()
    except OSError:
        return "unavailable"


def cg_mem_events() -> dict:
    ev = {}
    for line in read_cg("memory.events").splitlines():
        parts = line.split()
        if len(parts) == 2:
            ev[parts[0]] = int(parts[1])
    return ev


def cg_cpu_stat() -> dict:
    st = {}
    for line in read_cg("cpu.stat").splitlines():
        parts = line.split()
        if len(parts) == 2:
            st[parts[0]] = int(parts[1])
    return st


def status_mb(field: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) / 1024
    return -1.0


def main() -> int:
    result: dict = {"profile_env": {}, "cold": {}, "reps": [], "summary": {}}

    # --- environment truth from inside ---------------------------------------
    result["profile_env"] = {
        "memory.max": read_cg("memory.max"),
        "memory.current": read_cg("memory.current"),
        "memory.swap.max": read_cg("memory.swap.max"),
        "memory.swap.events": read_cg("memory.swap.events"),
        "cpu.max": read_cg("cpu.max"),
        "cpu.stat_initial": cg_cpu_stat(),
        "memory.events_initial": cg_mem_events(),
        "cgroup_controllers": read_cg("cgroup.controllers"),
        "swaptotal_kb": next(
            (int(l.split()[1]) for l in
             Path("/proc/meminfo").read_text().splitlines()
             if l.startswith("SwapTotal")), 0),
        "nproc_visible": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    print("[ENV] " + json.dumps(result["profile_env"]), flush=True)

    # --- imports + model load timeline ---------------------------------------
    t_imp0 = time.perf_counter()
    import psutil  # noqa: F401
    from caedral_voice.inference.kokoro_engine import (
        CaedralVoiceKokoroEngine, VOICE_PROFILES)
    t_imports_done = time.perf_counter() - T_SCRIPT

    text = CANONICAL.read_text(encoding="utf-8")
    t_load0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(
        model_path=os.environ.get("CV_MODEL_PATH",
                                  "/app/models/kokoro-v1.0.onnx"),
        voices_path=os.environ.get("CV_VOICES_PATH",
                                   "/app/models/voices-v1.0.bin"),
        threads=int(os.environ.get("CV_THREADS", "6")),
    )
    t_load_s = time.perf_counter() - t_load0
    rss_after_load = status_mb("VmRSS:")

    t_warm0 = time.perf_counter()
    try:
        engine.warmup()
        warm_ok = True
    except MemoryError:
        warm_ok = False
    t_warm_s = time.perf_counter() - t_warm0

    result["cold"] = {
        "script_to_imports_s": round(t_imports_done, 3),
        "engine_ctor_s": round(t_load_s, 3),
        "warmup_s": round(t_warm_s, 3),
        "warmup_success": warm_ok,
        "rss_after_load_mb": round(rss_after_load),
        "rss_after_warmup_mb": round(status_mb("VmRSS:")),
        "vmhwm_peak_mb": round(status_mb("VmHWM:")),
        "load_succeeded": True,
    }
    print("[COLD] " + json.dumps(result["cold"]), flush=True)

    # --- sequential generations ----------------------------------------------
    voice_name = VOICE_PROFILES["caedral-f1"]["kokoro_voice"]
    phonemes, _skipped = engine.phonemize(text)

    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(REPS):
        cpu0 = psutil.Process().cpu_times()
        cg_cpu0 = cg_cpu_stat()
        mem_ev0 = cg_mem_events()
        rss_before = status_mb("VmRSS:")
        t0 = time.perf_counter()
        try:
            audio = engine.infer_from_phonemes(
                phonemes, voice_name, speed=1.0)
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001 — record honestly
            audio, ok, err = None, False, f"{type(exc).__name__}: {exc}"[:200]
        wall = time.perf_counter() - t0
        cpu1 = psutil.Process().cpu_times()
        cg_cpu1 = cg_cpu_stat()
        mem_ev1 = cg_mem_events()

        audio_s = (len(audio) / 24000.0) if ok else 0.0
        row = {
            "rep": i + 1,
            "success": ok,
            "error": err,
            "wall_s": round(wall, 3),
            "audio_s": round(audio_s, 3),
            "rtf": round(wall / audio_s, 4) if ok else None,
            "realtime_multiplier": round(audio_s / wall, 3) if ok else None,
            "cpu_time_s": round((cpu1.user - cpu0.user)
                                + (cpu1.system - cpu0.system), 3),
            "throttled_usec_delta": cg_cpu1.get("throttled_usec", 0)
                                    - cg_cpu0.get("throttled_usec", 0),
            "nr_throttled_delta": cg_cpu1.get("nr_throttled", 0)
                                  - cg_cpu0.get("nr_throttled", 0),
            "oom_events_delta": mem_ev1.get("oom", 0) - mem_ev0.get("oom", 0),
            "rss_before_mb": round(rss_before),
            "rss_after_mb": round(status_mb("VmRSS:")),
            "vmhwm_peak_mb": round(status_mb("VmHWM:")),
            "memory_current_mb": round(int(read_cg("memory.current")) / 1e6),
        }
        result["reps"].append(row)
        print(f"[REP {i+1}/{REPS}] " + json.dumps(row), flush=True)
        # persist incrementally so a kill mid-run still leaves data
        (OUT / "probe-results.partial.json").write_text(
            json.dumps(result, indent=2))

    oks = [r for r in result["reps"] if r["success"]]
    walls = sorted(r["wall_s"] for r in oks)
    rtfs = sorted(r["rtf"] for r in oks)

    def pct(v, p):
        return v[min(len(v) - 1, int(p * len(v)))] if v else None

    result["summary"] = {
        "succeeded": len(oks), "failed": REPS - len(oks),
        "gen_p50_s": pct(walls, 0.50), "gen_p95_s": pct(walls, 0.95),
        "gen_min_s": walls[0] if walls else None,
        "gen_max_s": walls[-1] if walls else None,
        "rtf_p50": pct(rtfs, 0.50), "rtf_p95": pct(rtfs, 0.95),
        "peak_rss_max_mb": max((r["vmhwm_peak_mb"] for r in result["reps"]),
                               default=None),
        "final_memory_events": cg_mem_events(),
        "final_cpu_stat": cg_cpu_stat(),
    }

    (OUT / "probe-results.json").write_text(json.dumps(result, indent=2))
    (OUT / "probe-results.partial.json").unlink(missing_ok=True)
    print("[SUMMARY] " + json.dumps(result["summary"], indent=2), flush=True)
    return 0 if oks else 2


if __name__ == "__main__":
    sys.exit(main())
