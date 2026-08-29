# FIRST-CAEDRAL-VOICE — priority track

Directive: 2026-08-22. Goal: earliest legitimate Caedral-pipeline audio.
Definition of done for v0: a .wav in artifacts/latest-preview/ produced by
OUR frontend → OUR model checkpoint → Vocos decode. PROTOTYPE — NOT
PRODUCTION QUALITY.

## Critical path

1. E0007 pipeline agent lands (running) → smoke-validated launch command.
2. LJSpeech slice extraction completes (running, detached, monitored).
3. Coordinator launches the REAL tiny-config run immediately (detached,
   monitored). First checkpoint @1k steps.
4. At first checkpoint: generate preview WAVs (3 fixed sentences + short
   paragraph × up to 2 speakers) → artifacts/latest-preview/ + provenance README.
5. Refresh previews at every meaningful val-flow-loss improvement; keep history
   under experiments/CV-G1-E0007-pilot-training/samples/.

## Honest quality ladder (expected)

| Checkpoint | Expected audio |
|---|---|
| ~1–2k steps | noise-like / vowel-ish babble — PROVES PIPELINE |
| ~5k steps | rhythm + phoneme structure emerging |
| ~15k steps | possibly word-like segments on known sentences |
Anything better is upside. Baseline-comparable quality is NOT expected from
≤2 h CPU training and will not be claimed.

## Dependency discipline

This track NEVER blocks research/benchmark/data tracks running in parallel;
it only consumes their outputs when ready.
