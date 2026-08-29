# Caedral Voice — Frozen Evaluation Corpus v1.0

Frozen 2026-08-22. **Freeze policy:** this corpus may never be edited to improve a
benchmark score. Changes require a new version (`v1.1`, `v2.0`...) with a logged
justification in the commit message, and all old benchmark rows must state which
corpus version produced them.

- `eval-corpus-v1.0.json` — machine-readable corpus (categories, item IDs, text).
- All text is ORIGINAL to this project (no third-party copyrighted material).
- Categories map 1:1 to charter §8 requirements (short/long/technical/numbers/
  dates/currency/acronyms/names/difficult phonemes/heteronyms/punctuation/
  questions/emotional/dialogue/formal/informal/long-form/edge cases).

Consumers MUST render items verbatim after JSON decoding — no normalization,
no re-punctuation, no silent substitutions. Edge-case category includes inputs
that must fail cleanly or be rejected by the frontend; those behaviors are part
of the spec, not bugs.
