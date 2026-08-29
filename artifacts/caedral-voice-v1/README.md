# caedral-voice-v1 — delivery package (SCAFFOLD)

Status: **SKELETON** (2026-08-22). Every directory currently holds only a
`PLACEHOLDER.md` naming exactly which artifact lands there, its source path in
the repo, and the command that generates it. Results drop into place as they
land; a directory is DONE when its placeholder is replaced by the real
artifacts listed inside it.

Package owner rule: everything under `artifacts/caedral-voice-v1/` is release
output only — no working files, no scratch. Provenance for every number lives in
`provenance/` and `docs/CAEDRAL-VOICE-V1-FINAL-REPORT.md`; gate definitions live
in `provenance/release-gates.json`.

## Tree

| Path | Contents when complete | Producer |
|---|---|---|
| `model/` | Final V1 weights (torch checkpoint + ONNX graphs) | CV-G1-E0013 (+ RUNTIME-02 export) |
| `runtime/` | Inference runtime package (code, CLI, HTTP v0, chunker) | RUNTIME-01/RUNTIME-02 (`caedral_voice/inference/`) |
| `voices/` | 15 catalog voice definitions + persona metadata | VOICE-SLATE-15 + E0013 final checkpoint |
| `samples/` | Fixed audio samples (eval-corpus renders per voice) | V1 model via runtime CLI |
| `benchmarks/` | RTF / RAM / TTFA @threads, long-form, comparative tables | thread sweep + GATE-NAT-01 + E0014 re-run |
| `configs/` | Frozen release configs (model, inference defaults, personas) | from E0013 run manifest |
| `provenance/` | Machine snapshot, git commit, dataset/teacher hashes, `release-gates.json` | assembled at packaging time |
| `LICENSES/` | Third-party license obligations for shipped components | LIC-DEC-01 / TEACH-01 / DATASET-SURVEY |

## Release gates

`provenance/release-gates.json` encodes ALL founder gates. The package ships
only when every gate flips from `PENDING` to PASS with measured values:

- naturalness median competitive with best measured baseline
  (anchors on THIS machine: Kokoro DNSMOS ovrl median **3.450**, Piper **3.382**;
  frozen numeric thresholds: median >= 3.15 AND min >= 2.85 — see
  `docs/benchmarks/NATURALNESS-GATE.md`)
- >= **13 of 15** voices approved by the listening gate
- long-form per-chunk minWER population variance <= **0.0025**
  (diagnostic prototype measured 0.0173 = FAIL — CV-G1-E0014)
- zero artifact classes outside the allow-list (`artifact_classes_allowed: []`)
  — no NC/unlicensed weights anywhere near the shipped product
- **CPU-only**: every claim reproducible on the i7-13620H CPU box; no GPU results

## Related documents

- Final report: `docs/CAEDRAL-VOICE-V1-FINAL-REPORT.md`
- Gate protocol: `docs/benchmarks/NATURALNESS-GATE.md`
- Runtime design: `docs/inference/RUNTIME-DESIGN.md`
- Export plan: `docs/architecture/ONNX-EXPORT-PLAN-G1.md`
- Voice slate: `docs/data/VOICE-SLATE-V1.md`
- Teacher-data posture: `docs/training/TEACHER-DATA-ENGINE.md` §8
