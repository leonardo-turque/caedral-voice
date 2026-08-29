# RESUME.md — colab-r3 checkpoints back to local + E0013b continuation

## What lives where during the Colab run

| Location | Path | Contents |
|---|---|---|
| Colab disk | `/content/caedral-voice/experiments/colab-r3/` | `checkpoints/last.pt`, `best.pt`, `step_*.pt` (rotating 2), `metrics.jsonl`, `run_manifest.json` |
| Drive (synced ~every 20 s after each checkpoint write) | `MyDrive/caedral-r3/checkpoints/` | `last.pt` (latest), `prev-last.pt` (previous good), `best.pt`, `metrics.jsonl` only |
| Colab log | `/content/caedral-voice/colab-r3-train.log` | raw trainer stdout (not synced; copy manually if you want it) |

Drive keeps ONLY latest + previous-good + best — no step checkpoints. The ETA
reporter prints projected finish in America/Sao_Paulo every 10 min.
`KeyboardInterrupt` sends SIGINT to the trainer, whose `finally` block saves a
checkpoint, then flushes everything to Drive before exit.

## Bringing checkpoints back to the local machine (post-training)

1. Download from Drive `caedral-r3/checkpoints/` into
   `C:\Users\leona\Downloads\caedral-voice\experiments\colab-r3\checkpoints\`
   (create it; keep the same filenames). Grab `metrics.jsonl` into
   `experiments\colab-r3\` too.
2. Verify integrity: the trainer's checkpoint loader rejects torn files by
   construction, but sanity-check sizes and that
   `python -c "import torch; p=torch.load('experiments/colab-r3/checkpoints/last.pt', map_location='cpu', weights_only=False); print(p['trainer']['global_step'])"`
   prints the expected step.
3. Continue training locally (CPU) from that point if desired:
   `.venv\Scripts\python.exe -X utf8 experiments/CV-G1-E0013-regressor/train_regressor.py --manifests librittsr-clean100,teacher-kokoro --steps 20000 --exp-id colab-r3 --resume-from experiments/colab-r3/checkpoints/last.pt`
   (E0003 exact-resume: mid-epoch batch position, RNG streams and LR position
   all restore. First epoch on CPU rebuilds the mel cache — expect the usual
   ~25–35 min preprocessing warm-up; pass `--mel-cache-dir data/cache/mel-v1`
   to reuse the local cache.)

## E0013b audit continues locally

* The Colab arm is **`colab-r3`** — a separate experiment directory from the
  still-running local CPU arm `experiments/r3/`. They are independent
  trajectories (different device kernels ⇒ NOT bitwise-comparable); treat them
  as two arms of E0013b, compare via `metrics.jsonl` curves
  (mel_loss / dur_loss / val_loss vs step), not by weights.
* Audit inputs that matter locally: `run_manifest.json` (args, device, amp),
  `metrics.jsonl` (identical row schema to r1/r2/r3), `best.pt` +
  `phoneme_vocab.json` + `speaker_index.json` for inference/ONNX work
  (E0017 flow unchanged).
* AMP note: if the bootstrap AMP sanity gate passed, the run used fp16
  autocast + GradScaler; record that in the audit (flag is in
  `run_manifest.json -> versions.amp` and `args.amp`). If it failed, the run
  is pure FP32.
* The mel disk cache built on Colab (`data/cache/mel-v1` there) is NOT synced
  back — the local cache remains authoritative for local runs.

## Re-running / resuming on Colab

Rerun the same cell from COLAB_BOOTSTRAP_CELL.md. Hash-verified zips, the
downloaded corpus, preprocessing (sha256-audited) and checkpoints are all
detected and skipped; training resumes exactly from `last.pt`.
`CAEDRAL_FORCE_GATES=1 python /content/COLAB_BOOTSTRAP.py` forces the gate
suite to run again before relaunch.
