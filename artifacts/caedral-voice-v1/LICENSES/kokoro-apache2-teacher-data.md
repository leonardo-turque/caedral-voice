# Kokoro-82M v1.0 — Apache-2.0 (TEACHER DATA ONLY — never a shipped component)

Summary of `docs/training/TEACHER-DATA-ENGINE.md` (TEACH-01) §1, §8 and
`docs/benchmarks/BASELINE-MEASURED.md`. Teacher audio is **training data**:
it never ships in `artifacts/caedral-voice-v1/`; only its statistical
influence on our from-scratch Caedral weights persists. This file documents
the posture so the shipped package can state it.

## What it covers

- Kokoro-82M v1.0 ONNX: code + weights Apache-2.0 [VERIFIED per program
  record; re-verify pinned artifacts at release cut]. Used here as teacher:
  re-speaks LibriTTS-R clean-100 normalized transcripts (E0012,
  `scripts/build_teacher_set.py --shard i/n`, 10-voice slate t00–t09,
  manifests `data/manifests/teacher-kokoro.*` with per-row sha256).

## Grant and posture

- Apache-2.0 grants use, modification, and commercial distribution with no
  field-of-use restriction; model outputs carry no NC-style contamination
  (contrast Emilia/F5-TTS precedent). Engineering risk posture: LOW-MED; GO
  as training-data-only augmentation (TEACHER-DATA-ENGINE Path A).
- Teacher voices are synthetic — no real person's publicity rights engaged.
- Input side is OUR frontend, target side is OUR canonical mel; Kokoro's
  internal phoneme/timing chain is never used.

## Release-cut duties (before any commercial release)

1. Re-verify pinned Kokoro artifacts by sha256 (license-drift precedent).
2. Close the counsel-review open item on outputs-of-ML-models as training
   data (engineering assessment on file is not legal advice).
3. Honor EU AI Act Art. 50 transparency duties flagged in DATASET-SURVEY.
4. Piper-distillation path is EXCLUDED (GPL-3.0 gray) — never adopt.

## Sources

- `docs/training/TEACHER-DATA-ENGINE.md` §8 (license & provenance table)
- `docs/benchmarks/BASELINE-MEASURED.md` (Kokoro baseline + license record)
- https://huggingface.co/hexgrad/Kokoro-82M (model family reference)
- Local teacher meta: `data/manifests/teacher-kokoro.meta.json`
