# PILOT TRAINING PLAN — Phase J (CV-G1-E0007)

Status: DESIGN (execution starts when LJSpeech slice lands; LibriTTS-R side
is ready). Owner: coordinator. Honesty rules apply to every claim below.

## What exists (all MEASURED unless noted)

| Asset | State |
|---|---|
| Data | librittsr-smoke-10spk: 1,441 utt / 1.66 h / 10 spk, manifest validated; phonemes generated (251 utt/s). LJSpeech 400-utt slice extracting (detached). |
| Model | CAEDRAL-NAR tiny 8.35M / lite 31.6M w/ MAS+flow decoder (E0006); batch-1 fwd latency measured |
| Trainer | bitwise-exact resume proven (E0003); NaN guard; best-checkpoint protection |
| Frontend | shippable ladder (E0001/E0004); phonemizer wired |

## Training-data pipeline for the pilot

mel targets: 80-mel, 24 kHz, hop 256 (matches istft_head + Vocos reference).
Per utterance: load processed wav from manifest → log-mel → phoneme ids from
.phonemes.jsonl → tensors. Speaker id per catalog mapping (10 speakers ⇒
speaker table indices 0–9); style index = neutral(0) for all G1-SMOKE rows
(no style labels exist in these datasets — documented limitation).

## Run configuration (SMOKE-scale pilot)

- Config: **tiny** (8.35M) first — full run feasible on this machine;
  lite only if tiny converges and time permits.
- Optimizer AdamW β=(0.9,0.98), wd 1e-6? (final in config), lr 1e-4 peak,
  warmup 400 steps, cosine to 1e-5.
- Batch: utterance-count 8 with length-bucketed batching by sorted duration.
- Steps: target 15k steps (~25–40 epochs equivalent on 1.66 h).
- Val: held-out 10% of utterances per speaker (stratified), flow_loss +
  duration_loss + mel diagnostic (no vocoder quality claims until K).
- Checkpoints every 1k steps, keep-last-2 + best-by-val-flow-loss.
- Threads: torch.set_num_threads(4) (measured sweet spot; avoids hybrid-core
  bimodality risk during long runs).

## Compute reality [ESTIMATED then verified in-run]

tiny forward @800 frames ≈ 0.13–0.23 s ⇒ fwd+bwd ≈ 0.4–0.7 s/utterance
⇒ epoch ≈ 8–17 min ⇒ 15k steps ≈ 6–14 h wall on this laptop, resumable.
This is a REAL multi-hour run executed via the detached-process pattern;
progress monitored; every restart must remain loss-trajectory-exact (E0003 guarantee).

## Honest expectation setting

With ≤2 h of data and CPU-only training, the pilot CANNOT produce competitive
speech. Success criteria for KEEP:
1. flow_loss ↓ >50% from init and still decreasing;
2. durations converge to plausible speaking rate (~12–18 phones/s);
3. inferred mels show phoneme-correlated structure (visual/CTC-style probe);
4. no NaN events after warmup; resume exactness maintained.
Failure against any criterion ⇒ document, diagnose, iterate (Phase K).

Vocoder for pilot listening samples: pretrained Vocos (MIT) on predicted mels —
labeled clearly as "pilot, not product" in artifacts.

## Next actions

1. LJSpeech lands → preprocess → merge manifests (both datasets, unified schema,
   speaker table built from union).
2. Implement dataset/mel pipeline + pilot_train.py wiring model↔trainer.
3. Launch detached training run; Monitor; checkpoint hourly.
