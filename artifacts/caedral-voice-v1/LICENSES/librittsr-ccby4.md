# LibriTTS-R — CC BY 4.0 (training corpus, attribution duty)

Summary of `docs/data/DATASET-SURVEY.md` (Tier A row [S3]) and
`docs/data/LICENSE-MATRIX.md`. Summary + sources only.

## What we use

- OpenSLR SLR141 `train-clean-100` subset, extracted and filtered by
  DATA-FULL-01: **44.11 h kept / 30,917 utterances / 247 speakers**
  (raw 33,232 wavs = 53.55 h; yield 82.4%). 24 kHz Miipher-restored audio,
  original + normalized transcripts.
- Also the text backbone for teacher generation (E0012 re-speaks the same
  normalized transcripts — no new text acquired).

## License terms (CC BY 4.0) — what we owe

- **Attribution**: credit the dataset ("LibriTTS-R", OpenSLR SLR141), link
  the license, indicate changes if redistributing modified versions.
- Commercial use: allowed; no share-alike; no field-of-use restriction.
- Underlying LibriVox lineage is public domain.

## Caveats on record

- Identity posture MED at scale (per-reader consent undocumented — survey
  Tier A caveat); counsel sign-off recommended before anchoring named
  identities (VOICE-SLATE-15 personas are derived, not reader-named).
- Counsel review of corpus licensing remains an open item in DATASET-SURVEY.

## Release-cut duties

- Ship CC BY 4.0 attribution in `NOTICE.md`.
- Pin manifest hashes: `data/manifests/librittsr-clean100.meta.json`.

## Sources

- https://www.openslr.org/141/ (retrieved 2026-08-22)
- `docs/data/DATASET-SURVEY.md` [S3], Tier A table + practice list
- `docs/data/LICENSE-MATRIX.md`; local manifests under `data/manifests/`
