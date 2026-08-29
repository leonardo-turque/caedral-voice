# PLACEHOLDER — configs/

## What lands here

The frozen configs the shipped artifacts were produced with and must be run
with (copies, not references — this package is self-contained):

1. `model.json` — final model config (dims, n_speakers, mel params: 24 kHz,
   80-band ln-mel, hop 256, n_fft 1024) — from the E0013 run manifest args.
2. `inference.json` — runtime defaults: sigma **0.25**, chunker spec
   (LONGFORM-DESIGN v0: max ~420 chars/chunk), prosody clamps, thread policy.
3. `personas.json` — voice id -> persona mapping for the 15 catalog voices
   (founder grid vocabulary).
4. `training-manifest.json` — exact training provenance pointer: manifests
   used (`librittsr-clean100` + `teacher-kokoro`), durations file, steps,
   seed, lr schedule, threads.

## Source paths in repo

- Run args of record: `experiments/CV-G1-E0013-regressor/run_manifest.json`
- Model config family: `configs/tiny.json`, `configs/lite.json`,
  `configs/quality.json`; training: `configs/training/`
- Persona vocabulary: `configs/voices/README.md`
- Chunker + sigma decisions: `docs/architecture/LONGFORM-DESIGN.md`,
  E0009 conclusion (sigma 0.25)

## Generating commands

Copy at packaging time:

```bash
cp experiments/CV-G1-E0013-regressor/run_manifest.json artifacts/caedral-voice-v1/configs/training-manifest.json
```

(model.json / inference.json / personas.json are written once the final run
and runtime defaults are frozen.)

## Status

PENDING — assembled at packaging time from **CV-G1-E0013** outputs. Delete
this file when the real artifacts land.
