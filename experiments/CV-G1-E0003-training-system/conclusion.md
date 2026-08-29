# CV-G1-E0003 — Reproducible, crash-safe training system (Phase I)

**Date:** 2026-08-22 · **Scope:** infrastructure proof on SYNTHETIC tasks only; no real TTS training.
**Delivers:** `caedral_voice/training/` (`seed.py`, `trainer.py`), `configs/training/smoke.json`, `tests/test_training.py` (8 tests), this runner.
**Config:** `config.json` (effective cfg + code SHA-256s) · **Proof:** `results/resume-proof.json` · **Logs:** `logs/{1-scratch,2-resume,3-fresh}.log` · All numbers [MEASURED] on this machine.

---

## HYPOTHESIS

(H1) A Trainer whose fresh runs re-seed all RNGs at fit() entry produces bitwise-identical loss trajectories across independent same-seed runs. (H2) A run split as 30 steps / process-exit / separate-process resume-from-checkpoint reaches a state at step 60 that is BITWISE identical to an uninterrupted 60-step run — model weights, Adam moments, scheduler position, and torch RNG state. (H3) NaN/Inf batches can be survived by SKIP-and-log without weight damage or crashes. (H4) `best.pt` can be protected from worsening validation metrics (content + mtime untouched). (H5) All of the above holds with mid-epoch checkpoints and dataloader shuffling active.

## METHOD

Synthetic regression: y = xW^T + noise, n=128 train / 32 val, dim 16; MLP 16→64→16 tanh; Adam lr 1e-2; batch 16 (8 steps/epoch); seed 20260822; warmup 6 → cosine over 60 steps; grad-clip L2 1.0; val every 15 steps; rotating checkpoint every 15 steps (keep 2).

Three SEPARATE OS processes: (1) `scratch`: 30-step fit, process exits; (2) `resume`: new process builds fresh objects and continues ONLY from `checkpoints/last.pt` (restored global_step=30, epoch=3, batches_consumed_in_epoch=6 — i.e. mid-epoch); (3) `fresh`: uninterrupted 60-step reference in its own experiment dir. Orchestrator captures per-phase logs and verifies 10 boolean checks incl. per-tensor `torch.equal`, structural SHA-256 of optimizer state, and exact per-step loss/lr sequence equality across the 30/31 join. Unit suite covers the four mandated proofs deterministically (tmp dirs).

## RESULT

| Check | Value | Label |
|---|---|---|
| Same-seed reruns: loss trajectory | bitwise identical, 60/60 rows | [MEASURED] |
| Split vs fresh @ step 60: model weights | **max abs diff = 0.0**, every tensor `torch.equal` | [MEASURED] |
| Optimizer / scheduler / rng states | structurally identical | [MEASURED] |
| Loss trajectory across join | identical at every step (e.g. step 30 = 0.18065215…, 31 = 0.13410622…, 60 = 0.04657954…) | [MEASURED] |
| LR trajectory across join | identical (warmup→cosine, ends 0.0) | [MEASURED] |
| metrics.jsonl rows | 60 train rows, steps 1–60, no gaps/dups | [MEASURED] |
| Rotation | exactly {step_000000045.pt, step_000000060.pt}; no torn `.tmp-*` files | [MEASURED] |
| best tracking | best 0.20329@30 improved to 0.09079@60; best.pt rewritten only on improvement (unit-proven mtime+sha256 stable under worsening val) | [MEASURED] |
| NaN guard (unit test) | poisoned step skipped, weights byte-identical across it, event row logged, training completed 12/12 steps | [MEASURED] |
| Total wall time (3 processes) | **12.81 s** (fits: 0.11 s / 0.20 s / 0.32 s) | [MEASURED] |

10/10 verification checks passed → `results/resume-proof.json`. Full pytest suite: 104 passed (96 pre-existing + 8 new).

## INTERPRETATION

1. **H1–H5 CONFIRMED [MEASURED].** Exactness survives a real process boundary, mid-epoch interruption (resume point was 6 batches into epoch 4), and active shuffling.
2. **The subtle failure mode found and designed out [MEASURED via debug trace]:** DataLoader iterator creation consumes a `_base_seed` draw from the GLOBAL torch RNG even for `shuffle=False` loaders. Naive resume (restore post-step RNG + skip already-done actions) drifts one draw behind an uninterrupted run and diverges at the NEXT epoch boundary — our first implementation did exactly this (steps 31–32 matched, step 33 differed). Fixes now built in: (a) data order pinned to `(seed, epoch_index)` by cloning the train loader each epoch with a dedicated generator; (b) boundary-trigger REPLAY on resume re-executes the resume-step's val/checkpoint actions exactly as the uninterrupted run did, keeping even dropout streams aligned.
3. Atomicity is by construction (write tmp → fsync → `os.replace`), not by sampling: no torn-file window exists for readers; unit test asserts no temp residue after rotation.
4. Known honest artifact of append-only logging: the resumed phase's replayed val@step-30 appends a second val row for that step (val_steps shows "30" twice). Train-row comparison is unaffected; documented here rather than hidden.
5. Scope limits: CPU single-process/single-device; models must not spawn threads that consume torch RNG outside steps; exotic custom batch-samplers fall back to as-is use with a warning (order reproducibility then depends on the sampler).

## KEEP / REJECT / INVESTIGATE

**KEEP**
- `Trainer` contract: plain-dict cfg (JSON-loadable), atomic `last/best/step_%09d.pt` layout, rotation keep-2, JSONL metrics with (step, loss, lr, wall_s), NaN/Inf skip-and-log events, warmup+cosine `build_scheduler`.
- Resume protocol: post-step RNG snapshot in every checkpoint + derived-per-epoch data order + boundary-trigger replay. This trio is what makes two-process exactness hold; keep them inseparable.
- `set_determinism` seeding python/numpy/torch, pinning threads=1, returning a manifest-ready info dict.
- Anti-correlated-validation trick for testing best-checkpoint protection (reusable).

**REJECT**
- Naive "restore RNG and dedupe done actions" resume (draw-count drift, proven broken above).
- Relying on dataloader global-RNG shuffling across resumes (now overridden by explicit per-epoch generators).

**INVESTIGATE (next)**
- Multi-worker DataLoaders and persistent workers under this protocol.
- Crash-in-the-middle injection (SIGKILL during step N) rather than clean process exit; measure max lost work vs checkpoint cadence.
- Grad-accumulation and EMA-on-best semantics before first real acoustic model.
- Scale probe: same proof at ≥10k steps / larger model to confirm no drift accumulates.

*Reproduce:* `.venv/Scripts/python.exe -X utf8 experiments/CV-G1-E0003-training-system/run_experiment.py` (rewrites config.json/logs/results fresh; deterministic). Tests: `.venv/Scripts/python.exe -X utf8 -m pytest tests/test_training.py -v`.
