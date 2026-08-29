# PLACEHOLDER — runtime/

## What lands here

The shippable inference runtime package:

1. `caedral_voice/inference/` package as shipped (API, chunker, stream
   assembly, session lifecycle, cancellation, CLI `cli.py`, stdlib HTTP v0
   `server.py`) — file layout per `docs/inference/RUNTIME-DESIGN.md` §8.
2. ONNX Runtime backend wiring (backend matrix §4.2) + warmup strategy (§4.4).
3. A wheel/sdist of the runtime, or the exact vendored-file list if shipping
   as a source tree.
4. `runtime-README.md` — public API signature, validation rules, prosody
   units/clamping decisions, determinism/seeding contract.

## Source paths in repo

- Design (normative): `docs/inference/RUNTIME-DESIGN.md` (RUNTIME-01)
- Export plan: `docs/architecture/ONNX-EXPORT-PLAN-G1.md` (RUNTIME-02)
- Implementation target: `caedral_voice/inference/` (does not exist yet —
  created by the RUNTIME implementation task)
- Frontend it wraps: E0001/E0004 ladder (`TextFrontend("lexicon+g2pen")`,
  GPL espeak never touched)

## Generating commands

Implementation per RUNTIME-01/RUNTIME-02 acceptance mapping
(`docs/inference/RUNTIME-DESIGN.md` §10); packaging command to be recorded
here when defined (expected shape: `python -m build` or a documented copy list).

Smoke check that must pass before this dir is DONE:
`python -m caedral_voice.inference.cli synthesize --text "..." --voice <v>` on
CPU with TTFA and RTF inside the values published in `benchmarks/`.

## Status

PENDING — produced by the **RUNTIME** implementation task (RUNTIME-01 design is
complete; RUNTIME-02 staged ONNX export plan committed). Delete this file when
the real artifacts land.
