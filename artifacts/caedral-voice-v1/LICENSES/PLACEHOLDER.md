# PLACEHOLDER — LICENSES/

## What lands here

Third-party license obligations for every SHIPPED component. Present now
(summaries + source URLs only; no license-text dumps — full verification lives
in the repo docs cited inside each file):

- `vocos-mit.md` — decoder code + weights (MIT) — PRESENT
- `kokoro-apache2-teacher-data.md` — teacher-model statement (Apache-2.0;
  training-data-only, never shipped as an asset) — PRESENT
- `librittsr-ccby4.md` — training-corpus attribution (CC BY 4.0) — PRESENT

Still to add at release cut:

- `NOTICE.md` — assembled attribution/notice block that must ship with the
  product (MIT notice retention for Vocos; CC BY 4.0 attribution for
  LibriTTS-R; Apache-2.0 attribution if any Kokoro-derived artifact ever ships)
- Re-verification stamp: pinned artifact sha256s re-checked at packaging time
  (license drift precedent: BigVGAN NC->MIT, EnCodec — DECODER-LICENSES.md)

## Source docs in repo (authoritative)

- `docs/data/DECODER-LICENSES.md`
  (LIC-DEC-01, verified from primary artifacts 2026-08-22)
- `docs/data/DATASET-SURVEY.md`, `docs/data/LICENSE-MATRIX.md`
- `docs/training/TEACHER-DATA-ENGINE.md` §8

## Status

PARTIAL — three summaries present; NOTICE.md + pin-time re-verification
pending. Replace this placeholder at packaging time.
