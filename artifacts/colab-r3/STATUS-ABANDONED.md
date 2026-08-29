# COLAB R3 MIGRATION PATH

**STATUS: ABANDONED BY FOUNDER — 2026-08-23**

The Google Colab T4 migration was abandoned before cutover. No data was
uploaded, no Colab runtime consumed the bootstrap, and the local CPU r3
worker was never interrupted.

Historical value retained (do not develop further):
- `COLAB_BOOTSTRAP.py` — 8-stage orchestrator with SHA256 verification,
  AMP sanity gating, benchmark-before-cutover, ETA reporter. The gate
  structure and persistence patterns are reusable references.
- `code-package.zip`, `teacher-kokoro.zip` — point-in-time snapshots of
  code + teacher corpus. SUPERSEDED: teacher corpus kept growing after
  this zip was cut; use the live `data/processed/teacher-kokoro/` +
  `data/manifests/teacher-kokoro*` as the authoritative set.
- `build_transfer_package.py`, `requirements-colab.txt`, `RESUME.md`,
  `r3-frozen-config.json` — r3-frozen-config.json remains the authoritative
  scientific freeze for the migration (see
  docs/releases/PRE-ARCH-VOICE-MIGRATION.md).

Preprocessing drift note (historical): the bootstrap planned to re-run
deterministic preprocessing on Linux from the OpenSLR tar. That parity was
never executed or validated; the Arch migration instead transfers the
already-preprocessed corpus + manifests directly (see
docs/releases/PRE-ARCH-VOICE-MIGRATION.md).
