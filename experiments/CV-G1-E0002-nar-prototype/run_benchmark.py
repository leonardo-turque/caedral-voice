"""CV-G1-E0002 — CAEDRAL-NAR prototype measurement driver (Phase F).

Protocol (per config):
  1. params      — count_params() + per-component breakdown vs ±15% target.
  2. rss         — baseline RSS, then model build + batch-1 forward
                   (T=200 phonemes, 1 thread) under benchmarks.harness.measure;
                   native Win32 PeakWorkingSetSize is process-lifetime, so each
                   config runs in a FRESH child process (parent mode spawns one
                   python per config) and reports native peak at that point.
  3. latency     — batch-1 forward, median-of-20 after 5 warmups, for phoneme
                   lengths {50, 200, 800} x intraop threads {1, 4}, at the
                   model's own predicted durations (untrained stub -> ~1 frame
                   per phoneme), plus a 4-frames-per-phoneme anchor at T=200
                   (-> 800 mel frames) to anchor production-expansion scaling.
                   Anchor cells adapt reps down if a single forward is slow
                   (recorded per row); spec cells always use 20.

Usage:
  parent:  python run_benchmark.py --parent
  child:   python run_benchmark.py --config tiny --results results.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from benchmarks.harness.measure import measure, win_process_peaks  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402
from caedral_voice.model import load_model  # noqa: E402

EXP_ID = "CV-G1-E0002"
TARGETS = {"tiny": 8_000_000, "lite": 30_000_000, "quality": 80_000_000}
LENGTHS = [50, 200, 800]
THREAD_SWEEP = [1, 4]
WARMUPS = 5
REPS = 20
ANCHOR_FPP = 4  # anchor: 4 mel frames per phoneme (realistic ~60ms phonemes @ 93.75Hz)


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def make_ids(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 200, (1, n), generator=g)


def time_forward(fn, warmups: int, reps: int) -> dict:
    """Median-of-N wall time via the shared harness (5 warmups excluded)."""
    for _ in range(warmups):
        fn()
    walls = []
    for _ in range(reps):
        _, perf = measure(fn, collect_gc=False)
        walls.append(perf.wall_seconds)
    walls.sort()
    med = statistics.median(walls)
    return {
        "median_s": med,
        "mean_s": statistics.fmean(walls),
        "p10_s": walls[max(0, int(0.1 * len(walls)) - 1)],
        "p90_s": walls[min(len(walls) - 1, int(0.9 * len(walls)))],
        "min_s": walls[0],
        "max_s": walls[-1],
        "reps": reps,
    }


def latency_cell(model, ids, durations, threads: int, label: str, cfg_name: str,
                 results: Path) -> None:
    torch.set_num_threads(threads)
    hop = model.config.hop_length
    sr = model.config.sample_rate
    with torch.no_grad():
        probe = model(ids, durations=durations)
        t_mel = probe["n_mel_frames"]

        def run():
            model(ids, durations=durations)

    # adapt reps only for slow anchor cells (spec cells stay at 20)
    t0 = time.perf_counter()
    run()
    est = time.perf_counter() - t0
    reps = REPS if label == "predicted" else max(5, min(REPS, int(30.0 / max(est, 1e-3))))

    stats = time_forward(run, WARMUPS, reps)
    audio_s = t_mel * hop / sr
    append_row(results, {
        "type": "latency",
        "experiment": EXP_ID,
        "config": cfg_name,
        "label": label,
        "threads": threads,
        "torch_num_threads": torch.get_num_threads(),
        "n_phonemes": int(ids.shape[1]),
        "n_mel_frames": t_mel,
        "audio_seconds": round(audio_s, 4),
        "warmups": WARMUPS,
        **stats,
        "rtf_acoustic": round(stats["median_s"] / audio_s, 5),
        "label_evidence": "MEASURED",
    })


def run_child(config_name: str, results: Path) -> None:
    cfg_path = REPO / "configs" / f"{config_name}.json"
    model = load_model(cfg_path)  # seeded from config
    model.eval()

    # ---- params ------------------------------------------------------- #
    total = model.count_params()
    target = TARGETS[config_name]
    append_row(results, {
        "type": "params",
        "experiment": EXP_ID,
        "config": config_name,
        "total_params": total,
        "target_params": target,
        "delta_pct": round(100.0 * (total - target) / target, 2),
        "within_15pct": abs(total - target) <= 0.15 * target,
        "breakdown": model.param_breakdown(),
        "label_evidence": "MEASURED",
    })

    # ---- RSS (fresh process => native peak attributable) --------------- #
    torch.set_num_threads(1)
    ids = make_ids(200, seed=model.config.seed)
    with torch.no_grad():
        _, rss_perf = measure(lambda: model(ids))
    native = win_process_peaks()
    append_row(results, {
        "type": "rss",
        "experiment": EXP_ID,
        "config": config_name,
        "baseline_rss_bytes": rss_perf.baseline_rss_bytes,
        "sampler_peak_rss_bytes": rss_perf.peak_rss_bytes,
        "peak_over_baseline_bytes": rss_perf.peak_over_baseline_bytes,
        "native_peak_rss_bytes_process_lifetime": native[0] if native else None,
        "note": "native peak is Win32 PeakWorkingSetSize at this point in a "
                "fresh process (import torch + build + T=200 forward)",
        "label_evidence": "MEASURED",
    })

    # ---- latency sweep ------------------------------------------------- #
    for threads in THREAD_SWEEP:
        for n in LENGTHS:
            ids = make_ids(n, seed=model.config.seed + n)
            latency_cell(model, ids, None, threads, "predicted", config_name, results)
        # anchor: uniform 4 frames/phoneme at T=200 -> 800 mel frames
        ids = make_ids(200, seed=model.config.seed + 200)
        durs = torch.full((1, ids.shape[1]), ANCHOR_FPP, dtype=torch.long)
        latency_cell(model, ids, durs, threads, f"x{ANCHOR_FPP}_expansion",
                     config_name, results)


def run_parent(results: Path) -> None:
    if results.exists():
        results.unlink()
    append_row(results, {
        "type": "meta",
        "experiment": EXP_ID,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": capture(),
        "torch": torch.__version__,
        "protocol": {
            "latency": f"batch-1 forward, median-of-{REPS} after {WARMUPS} warmups, "
                       f"lengths {LENGTHS}, threads {THREAD_SWEEP}, "
                       "benchmarks.harness.measure",
            "anchor": f"T=200 phonemes at uniform {ANCHOR_FPP} mel frames/phoneme",
            "rss": "fresh child process per config; sampler peak + Win32 "
                   "PeakWorkingSetSize",
        },
        "label_evidence": "MEASURED",
    })
    for name in TARGETS:
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--config", name, "--results", str(results)],
            check=True, cwd=str(REPO),
        )
        print(f"[parent] {name} done in {time.perf_counter() - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", action="store_true")
    ap.add_argument("--config", choices=list(TARGETS))
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()
    results_path = Path(args.results)
    if args.parent:
        run_parent(results_path)
    else:
        assert args.config, "--config required without --parent"
        run_child(args.config, results_path)
