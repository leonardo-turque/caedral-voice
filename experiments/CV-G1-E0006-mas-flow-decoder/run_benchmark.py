"""CV-G1-E0006 — flow-decoder + MAS batch-1 CPU latency measurement driver.

Protocol (mission spec): median-of-10 after 3 warmups, intraop threads
{1, 4}, phoneme lengths {200, 800}, lite config upgraded with use_flow=True
(flow_hidden=256, flow_blocks=6). Two measured paths:

  training_loss : losses_for_training() on a random teacher mel whose frame
                  count equals the phoneme count -- the same geometry as the
                  E0002 baseline cells (untrained duration predictors emit
                  ~1 mel frame per phoneme there and here);
  inference     : full infer() incl. iSTFT waveform.

Plus one informational (non-gated) training cell at the E0002 x4-expansion
anchor geometry (T=200 phonemes -> 800 mel frames).

Budget gate (E0002, same box, forward-only): lite@800 ph 891.2 ms @1 thr /
592.4 ms @4 thr; lite@200 ph 194.2 / 123.8 ms. Gate = 1.3x those numbers,
applied to BOTH measured paths.

Usage: .venv/Scripts/python.exe experiments/CV-G1-E0006-mas-flow-decoder/run_benchmark.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from benchmarks.harness.measure import measure  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402
from caedral_voice.model import CaedralNAR, CaedralNARConfig  # noqa: E402

EXP_ID = "CV-G1-E0006"
LENGTHS = [200, 800]
THREAD_SWEEP = [1, 4]
WARMUPS = 3
REPS = 10
ANCHOR_PH, ANCHOR_FRAMES = 200, 800

# CV-G1-E0002 results.jsonl medians (same machine, torch 2.6.0+cpu), ms
E0002_LITE_FWD_MS = {
    ("predicted", 200, 1): 194.2,
    ("predicted", 200, 4): 123.8,
    ("predicted", 800, 1): 891.2,
    ("predicted", 800, 4): 592.4,
}
GATE_MULT = 1.3


def build_model() -> CaedralNAR:
    base = json.loads((REPO / "configs" / "lite.json").read_text(encoding="utf-8"))
    base.update({"use_flow": True, "flow_hidden": 256, "flow_blocks": 6})
    torch.manual_seed(base["seed"])
    return CaedralNAR(CaedralNARConfig(**base))


def make_ids(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 200, (1, n), generator=g)


def make_mel(frames: int, n_mels: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, frames, n_mels, generator=g)


def time_cell(fn, threads: int) -> dict:
    torch.set_num_threads(threads)
    for _ in range(WARMUPS):
        fn()
    walls = []
    for _ in range(REPS):
        _, perf = measure(fn, collect_gc=False)
        walls.append(perf.wall_seconds)
    walls.sort()
    return {
        "median_ms": round(1000 * statistics.median(walls), 2),
        "mean_ms": round(1000 * statistics.fmean(walls), 2),
        "min_ms": round(1000 * walls[0], 2),
        "max_ms": round(1000 * walls[-1], 2),
        "warmups": WARMUPS,
        "reps": REPS,
    }


def main() -> None:
    results = Path(__file__).parent / "results.jsonl"
    rows: list[dict] = []

    model = build_model()
    model.eval()
    total_params = model.count_params()
    rows.append({
        "type": "meta",
        "experiment": EXP_ID,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": capture(),
        "torch": torch.__version__,
        "config": "configs/lite.json + use_flow=True(flow_hidden=256, flow_blocks=6)",
        "total_params": total_params,
        "protocol": {
            "timing": f"median-of-{REPS} after {WARMUPS} warmups "
                      "(benchmarks.harness.measure)",
            "paths": ["training_loss (gated)", "inference forced_1fpp (gated)",
                      "inference predicted_durations (informational)"],
            "geometry": "gated cells: mel frames == phoneme count, the "
                        "E0002 baseline geometry; anchor cell 200 ph -> 800 "
                        "frames is informational only",
            "gate": f"{GATE_MULT}x E0002 lite forward medians",
            "e0002_reference_ms": {f"{k[0]}@{k[1]}ph@{k[2]}thr": v
                                   for k, v in E0002_LITE_FWD_MS.items()},
        },
        "label_evidence": "MEASURED",
    })

    def emit(row: dict) -> None:
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "path", "geometry", "threads", "n_phonemes", "n_mel_frames",
            "median_ms", "gate_ms", "ratio_vs_e0002", "gate_pass") if k in row}),
            flush=True)

    for threads in THREAD_SWEEP:
        for n in LENGTHS:
            ids = make_ids(n, seed=model.config.seed + n)
            mel = make_mel(n, model.config.n_mels, seed=model.config.seed + n)

            def run_train():
                out = model.losses_for_training(mel, ids, speaker_id=0)
                assert torch.isfinite(out["loss"])

            stats = time_cell(run_train, threads)
            ref = E0002_LITE_FWD_MS[("predicted", n, threads)]
            gate = GATE_MULT * ref
            emit({
                "type": "latency", "experiment": EXP_ID,
                "path": "training_loss", "threads": threads,
                "torch_num_threads": torch.get_num_threads(),
                "n_phonemes": n, "n_mel_frames": n,
                **stats,
                "e0002_ref_ms": ref, "gate_ms": round(gate, 1),
                "ratio_vs_e0002": round(stats["median_ms"] / ref, 3),
                "gate_pass": stats["median_ms"] <= gate,
                "label_evidence": "MEASURED",
            })

            def run_infer_forced():
                # gated cell: forced 1 frame/phoneme == E0002 baseline geometry
                with torch.no_grad():
                    model.infer(ids, durations=torch.ones(1, n, dtype=torch.long))

            stats = time_cell(run_infer_forced, threads)
            emit({
                "type": "latency", "experiment": EXP_ID,
                "path": "inference", "geometry": "forced_1fpp",
                "threads": threads,
                "torch_num_threads": torch.get_num_threads(),
                "n_phonemes": n, "n_mel_frames": n,
                **stats,
                "e0002_ref_ms": ref, "gate_ms": round(gate, 1),
                "ratio_vs_e0002": round(stats["median_ms"] / ref, 3),
                "gate_pass": stats["median_ms"] <= gate,
                "label_evidence": "MEASURED",
            })

            def run_infer_predicted():
                with torch.no_grad():
                    model.infer(ids)

            stats = time_cell(run_infer_predicted, threads)
            pred_frames = int(model.infer(ids)["durations"].sum())
            emit({
                "type": "latency", "experiment": EXP_ID,
                "path": "inference", "geometry": "predicted_durations",
                "threads": threads,
                "torch_num_threads": torch.get_num_threads(),
                "n_phonemes": n, "n_mel_frames": pred_frames,
                **stats,
                "e0002_ref_ms": ref, "gate_ms": None,
                "note": "informational: this init's untrained predictor emits "
                        "~1.3 frames/phoneme vs E0002's ~1.0, so the raw "
                        "comparison is geometry-confounded; gate is judged on "
                        "the forced_1fpp cell",
                "label_evidence": "MEASURED",
            })

        # informational x4-expansion anchor (matches E0002 anchor geometry)
        ids = make_ids(ANCHOR_PH, seed=model.config.seed + ANCHOR_PH)
        mel = make_mel(ANCHOR_FRAMES, model.config.n_mels,
                       seed=model.config.seed + ANCHOR_FRAMES)

        def run_anchor():
            out = model.losses_for_training(mel, ids, speaker_id=0)
            assert torch.isfinite(out["loss"])

        stats = time_cell(run_anchor, threads)
        emit({
            "type": "latency", "experiment": EXP_ID,
            "path": "training_loss_x4_anchor", "threads": threads,
            "n_phonemes": ANCHOR_PH, "n_mel_frames": ANCHOR_FRAMES,
            **stats,
            "e0002_ref_ms": None, "gate_ms": None,
            "note": "informational; E0002 anchor (forward-only) was "
                    "616.8 ms @1thr / 285.3 @4thr",
            "label_evidence": "MEASURED",
        })

    with results.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {results} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
