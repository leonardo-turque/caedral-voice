# PLACEHOLDER — voices/

## What lands here

The 15 catalog voice definitions, exactly one entry per slate pick:

- `voices.json` — for each of the 15 voices: stable public voice id/codename,
  LibriTTS-R speaker id, speaker-table row index in the final checkpoint,
  persona label (founder grid: grave/medium/young/warm/professional/energetic/
  calm), measured F0 median/IQR + centroid + rate proxy from selection,
  listening-gate verdict.
- Per-voice reference wav (>= 5 s) used as the enrollment anchor for the
  speaker-similarity gate (ECAPA-TDNN per
  `docs/research/EVALUATION-BENCHMARKS.md` §3/G4).

## Source paths in repo

- Slate definition: `docs/data/VOICE-SLATE-V1.md` (VOICE-SLATE-15; picks:
  6209 narrator, 6415, 696, 2416, 6147, 1088, 3879, 2836, 78, 7447, 8419,
  7067, 8838, 7505, 7402; reserves F:40/1841, M:8088/3830)
- Persona vocabulary: `configs/voices/README.md`
- Speaker table source: final E0013 checkpoint speaker embedding table +
  `speaker_index.json` (pattern:
  `experiments/CV-G1-E0007-pilot-training/smoke-a/checkpoints/speaker_index.json`)
- Reference wavs: kept rows in `data/manifests/librittsr-clean100.jsonl`
  (hash-addressed via `sha256_audio`)

## Generating commands

Voice extraction from the final checkpoint:

```bash
.venv/Scripts/python.exe -c "import torch; ck=torch.load('experiments/CV-G1-E0013-regressor/checkpoints/best.pt', map_location='cpu', weights_only=False); print(ck['speaker_index'])" > voices/speaker-table-dump.txt
```

(then hand-freeze into `voices.json` with persona metadata from the slate doc;
reference wavs copied by manifest id with hashes recorded in provenance.)

## Status

PENDING — produced by **CV-G1-E0013** (final speaker table) + the human
listening gate (founder gate: >= 13 of 15 approved; two boundary voices
6415 @158 Hz / 7402 @151 Hz need presentation confirmation). Delete this file
when the real artifacts land.
