# PLACEHOLDER — model/

## What lands here

The final V1 model artifacts, exactly:

1. `caedral-v1-best.pt` — final torch checkpoint (model weights + vocab +
   speaker table) copied from the E0013 regressor run's best checkpoint.
2. ONNX graphs per `docs/architecture/ONNX-EXPORT-PLAN-G1.md` staged export:
   `frontend-t{50,200,800}.onnx` (stage A), `flow-inverse.onnx` (stage B,
   dynamic T_y), iSTFT head per stage C decision (torch-side for v0).
3. `model-card.md` — param count, architecture summary, training data
   provenance summary (real + teacher), measured quality row.

## Source paths in repo

- Checkpoint: `experiments/CV-G1-E0013-regressor/checkpoints/best.pt`
  (atomic checkpoints every 1000 steps; best.pt never overwritten in place)
- Model code: `caedral_voice/model/regressor.py` (+ `caedral_voice/model/`
  encoder/predictors); mel definition single-sourced in
  `caedral_voice/training/mel_dataset.py`
- Export wrappers: `runtime/export/` (to be created per RUNTIME-02 staged plan)

## Generating commands

Training (LAUNCH_READY line from `experiments/CV-G1-E0013-regressor/LAUNCH_READY`):

```bash
.venv/Scripts/python.exe -X utf8 experiments/CV-G1-E0013-regressor/train_regressor.py \
  --manifests librittsr-clean100,teacher-kokoro \
  --durations data/manifests/librittsr-clean100.durations.jsonl \
  --steps 15000 --threads 4 --exp-id CV-G1-E0013-regressor
```

ONNX export: per `docs/architecture/ONNX-EXPORT-PLAN-G1.md` stages A -> B -> C
(gate A parity <= 1e-4, gate B <= 3e-4, gate C <= 1e-3 max-abs before advancing).

## Status

PENDING — produced by **CV-G1-E0013** (training; prerequisites: CV-G1-E0011
durations file `data/manifests/librittsr-clean100.durations.jsonl`, teacher rows
from CV-G1-E0012) and **RUNTIME-02** (export). Delete this file when the real
artifacts land.
