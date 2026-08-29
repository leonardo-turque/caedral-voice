# CHECKPOINT — PHASE A (Environment & Inventory)

Date: 2026-08-22 · Evidence labels per charter §9.

## 1. Research performed
Machine + toolchain inventory only (Phase A scope). No literature research yet.

## 2. Sources consulted
Direct local inspection via PowerShell CIM queries and `Get-Command` — MEASURED.
CPU ISA capabilities inferred from part number, pending CPUID verification — ESTIMATED.

## 3. Findings
- i7-13620H: 10C/16T Raptor Lake-H, base 2.4 GHz — MEASURED.
- 23.67 GB RAM installed, ~6.0 GB free at inspection — MEASURED.
- Intel UHD integrated GPU only; no CUDA — MEASURED. ⇒ all local training is CPU PyTorch.
- C: free 122.3 GB — MEASURED. Bounds dataset downloads; SMOKE/PILOT discipline enforced.
- Python 3.13.15 + pip 26.2.1 native Windows — MEASURED.
- No C/C++/Rust/cmake toolchain — MEASURED. ⇒ optimization must ship as wheels
  (ONNX Runtime / OpenVINO) or require later toolchain installation.
- Docker Desktop + WSL available for §18 torture lab — MEASURED.

## 4. Decisions made
- Repo skeleton adopted (charter §21 layout). Git initialized locally.
- Zero-shot voice cloning excluded from G1 scope (provenance risk).
- ONNX Runtime named primary CPU substrate candidate (revisit Phase F/L).

## 5. Files created/changed
README.md, .gitignore, docs/MASTER-PLAN.md, this file, directory skeleton (27 dirs).

## 6. Experiments executed
None yet (CV-G1-E#### series starts in Phase E/F). Environment inspection is not an experiment.

## 7. Exact measured results
See docs/MASTER-PLAN.md §4 table (all values MEASURED unless marked ESTIMATED).

## 8. Failed experiments
None.

## 9. Remaining risks
No discrete GPU (full training needs external compute); 122 GB disk ceiling;
free RAM ~6 GB during interactive use; GPL frontend dependency hazard unresolved.

## 10. Disk usage
Repo ≈ negligible (<1 MB). Free: 122.3 GB — MEASURED at checkpoint time.

## 11. Current best model metrics
NONE EXISTS YET. No Caedral model trained. Honest null result.

## 12. Next phase
PHASE B — state-of-the-art research. Eight parallel research tracks launched
(landscape/baselines, text frontend, acoustic architectures, vocoders/decoders,
datasets+licensing, CPU inference+quantization, evaluation, multi-speaker/prosody).
Coordinator reconciles into STATE-OF-THE-ART.md / BASELINE-MATRIX.md / SOURCES.md,
then Checkpoint B. No architecture commitment before that completes.
