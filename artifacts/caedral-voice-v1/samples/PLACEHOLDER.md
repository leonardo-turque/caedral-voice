# PLACEHOLDER — samples/

## What lands here

Fixed, reproducible audio samples rendered by the FINAL V1 model:

1. `eval-corpus/` — every item of the frozen eval corpus
   (`benchmarks/corpus/eval-corpus-v1.0.json`) rendered per voice, or at
   minimum: the 10 GATE-NAT-01 sentences x 2 launch voices (20 clips) + one
   representative clip per remaining voice.
2. `longform/long_001.wav` — full long-form render (the E0014 anchor text).
3. `samples-manifest.json` — per clip: text id, voice, seed, sigma, threads,
   sha256, duration, peak level (headroom rule: peak-scaled to 0.98 max,
   `gain_applied` recorded — NATURALNESS-GATE protocol rule 3).

## Source paths in repo

- Corpus: `benchmarks/corpus/eval-corpus-v1.0.json` (frozen; never edited)
- Protocol: `docs/benchmarks/NATURALNESS-GATE.md` (sentences, voices-at-
  defaults, headroom, RAM discipline) and
  `docs/architecture/LONGFORM-DESIGN.md` v0 splitter for long_001 chunking
- Pattern to copy: `benchmarks/results/natgate/{kokoro,piper}/*.wav` +
  `gen-meta-*.json` layout from the baseline runs
- Inference defaults: sigma **0.25** (E0009 finding)

## Generating commands

```bash
# natgate clips (2 launch voices, library defaults, separate processes)
.venv/Scripts/python.exe benchmarks/run_natgate.py generate --engine caedral-v1
# long-form render via runtime CLI (chunker on), exact command recorded when RUNTIME lands
```

## Status

PENDING — produced by the **natgate** run against the final V1 model
(CV-G1-E0013 output) once the runtime exists. Delete this file when the real
artifacts land.
