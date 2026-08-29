"""V1 RTF/RAM RECHECK (read-only diagnostics) — controlled re-measurement at the
frozen production config (threads=6, ENABLE_ALL, cpu_arena_off).

Explains the two open reconciliation items with fresh MEASURED data:
  A. Warm-RSS ladder (imports -> session -> frontend -> warmup) and TRUE
     engine-only sampled peak RSS during short + long-form synthesis
     (10 ms background sampler), i.e. without any ASR stack resident.
  B. Same process then loads the acceptance-run audit stack
     (faster-whisper small.en+base.en int8 via classify_preview.load_models)
     and repeats one long-form render — reproducing the acceptance-run
     memory profile that produced "~1765 MB peak".
  C. Controlled warm RTF medians per voice x sentence (5 reps) so the
     acceptance "0.59-1.10" scatter and the historical "~0.46" figure can be
     reconciled against current idle-machine numbers.

Output: artifacts/benchmarks/v1-rtf-ram-recheck.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import psutil

SENTENCES = {
    "control": ("Hello. This is the first generation of Caedral Voice."),
    "technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "longform": ("The morning market opened early, as it always did on "
                 "Saturdays. Vendors arranged their stalls under striped "
                 "awnings while the streetlights were still on, and by seven "
                 "o'clock the narrow road smelled of bread, wet stone, and "
                 "strong coffee."),
}
VOICES = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
THREADS = 6
REPS = 5


def main() -> None:
    proc = psutil.Process()
    out: dict = {
        "experiment": "v1-rtf-ram-recheck",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"backend": "ORT CPU", "graph_opt": "ENABLE_ALL",
                   "intra_op_threads": THREADS, "cpu_arena": False},
        "machine": {"cpu": __import__("platform").processor(),
                    "logical_cpus": psutil.cpu_count(logical=True)},
    }

    # --- Stage A0: RSS ladder -------------------------------------------------
    rss_after_imports = proc.memory_info().rss / 1e6

    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
    t0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=THREADS,
    )
    load_s = time.perf_counter() - t0
    rss_after_load = proc.memory_info().rss / 1e6
    engine.warmup()
    rss_warm = proc.memory_info().rss / 1e6

    out["rss_ladder_mb"] = {
        "after_imports": round(rss_after_imports),
        "after_engine_load": round(rss_after_load),
        "after_warmup_warm": round(rss_warm),
        "load_seconds": round(load_s, 2),
    }

    # --- Sampler helper -------------------------------------------------------
    samples: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            samples.append(proc.memory_info().rss / 1e6)
            time.sleep(0.01)

    def peak_of(fn):
        stop.clear()  # reset between calls (Event stays set otherwise)
        samples.clear()
        t = threading.Thread(target=sample, daemon=True)
        t.start()
        t_start = time.perf_counter()
        result = fn()
        wall = time.perf_counter() - t_start
        stop.set()
        t.join()
        return result, wall, max(samples) if samples else None

    peaks_only = bool(os.environ.get("PEAKS_ONLY"))
    grid_only = bool(os.environ.get("GRID_ONLY"))

    # --- Stage A1: engine-only peaks ------------------------------------------
    if grid_only:
        out["engine_only_peaks"] = {"skipped": "GRID_ONLY run"}
    elif peaks_only:
        ctrl_peak = None
        lf_audio, lf_wall, lf_peak = peak_of(
            lambda: engine.synth(SENTENCES["longform"], voice_id="caedral-m1"))
        out["engine_only_peaks"] = {
            "longform_16s": {
                "peak_rss_mb": round(lf_peak or 0),
                "wall_s": round(lf_wall, 2),
                "audio_s": round(len(lf_audio) / 24000, 2),
                "rtf_one_shot": round(lf_wall / (len(lf_audio) / 24000), 3)},
            "post_render_settled_rss_mb": round(proc.memory_info().rss / 1e6),
        }
    else:
        ctrl_audio, _, ctrl_peak = peak_of(
            lambda: engine.synth(SENTENCES["control"], voice_id="caedral-m1"))
        lf_audio, lf_wall, lf_peak = peak_of(
            lambda: engine.synth(SENTENCES["longform"], voice_id="caedral-m1"))
        out["engine_only_peaks"] = {
            "control_short_sentence": {
                "peak_rss_mb": round(ctrl_peak or 0), "audio_s":
                round(len(ctrl_audio) / 24000, 2)},
            "longform_16s": {
                "peak_rss_mb": round(lf_peak or 0),
                "wall_s": round(lf_wall, 2),
                "audio_s": round(len(lf_audio) / 24000, 2),
                "rtf_one_shot": round(lf_wall / (len(lf_audio) / 24000), 3)},
            "post_render_settled_rss_mb": round(proc.memory_info().rss / 1e6),
        }

    # --- Stage C: controlled warm RTF grid ------------------------------------
    rtf_rows = []
    if not peaks_only:
        for vid in VOICES:
            for sname, text in SENTENCES.items():
                walls = []
                audio_s = 0.0
                for _ in range(REPS):
                    t = time.perf_counter()
                    a = engine.synth(text, voice_id=vid, speed=1.0)
                    walls.append(time.perf_counter() - t)
                    audio_s = len(a) / 24000
                med = statistics.median(walls)
                rtf_rows.append({
                    "voice": vid, "sentence": sname,
                    "audio_s": round(audio_s, 2),
                    "wall_median_s": round(med, 3),
                    "rtf_median": round(med / audio_s, 3),
                })
    out["rtf_grid"] = rtf_rows
    rtfs_all = [r["rtf_median"] for r in rtf_rows]
    out["rtf_summary"] = (
        {"min": min(rtfs_all), "max": max(rtfs_all),
         "median_across_combos": statistics.median(rtfs_all),
         "note": "warm median of 5 reps per voice x sentence, idle machine"}
        if rtfs_all else {"skipped": "PEAKS_ONLY run"})

    if grid_only:
        dest = REPO / "artifacts" / "benchmarks" / "v1-rtf-grid-recheck.json"
        json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)
        print(json.dumps({"rss_ladder_mb": out["rss_ladder_mb"],
                          "rtf_grid": out["rtf_grid"],
                          "rtf_summary": out["rtf_summary"]}, indent=2))
        print(f"\nwrote {dest}")
        return

    # --- Stage B: load acceptance-audit stack (ASR + DNSMOS deps) --------------
    import classify_preview as cp
    t_models = time.perf_counter()
    cp.load_models()
    asr_load_s = time.perf_counter() - t_models
    rss_with_asr = proc.memory_info().rss / 1e6

    import torch  # noqa: F401  (speechmos/torchaudio path used by DNSMOS)
    import torchaudio  # noqa: F401
    from speechmos import dnsmos  # noqa: F401
    rss_with_dnsmos_deps = proc.memory_info().rss / 1e6

    lf_audio2, lf_wall2, lf_peak2 = peak_of(
        lambda: engine.synth(SENTENCES["longform"], voice_id="caedral-m1"))
    out["with_acceptance_audit_stack"] = {
        "asr_model_load_seconds": round(asr_load_s, 1),
        "rss_after_asr_loaded_mb": round(rss_with_asr),
        "rss_after_dnsmos_deps_imported_mb": round(rss_with_dnsmos_deps),
        "delta_vs_warm_engine_mb": round(rss_with_asr - rss_warm),
        "longform_rerender_peak_rss_mb": round(lf_peak2 or 0),
        "longform_rerender_wall_s": round(lf_wall2, 2),
    }

    out["conclusion"] = (
        f"Engine-only warm RSS {round(rss_warm)} MB; engine-only long-form "
        f"peak {round(lf_peak or 0)} MB; loading the dual-ASR audit stack adds "
        f"~{round(rss_with_asr - rss_warm)} MB in the SAME process, putting "
        f"combined peak near {round(max(lf_peak2 or 0, rss_with_asr))} MB."
    )

    dest = REPO / "artifacts" / "benchmarks" / "v1-rtf-ram-recheck.json"
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("rss_ladder_mb", "engine_only_peaks", "rtf_summary",
                       "with_acceptance_audit_stack", "conclusion")}, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
