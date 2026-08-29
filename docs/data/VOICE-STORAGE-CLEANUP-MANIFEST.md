# VOICE STORAGE CLEANUP MANIFEST
# Generated: 2026-08-23 · Status: INVENTORY ONLY (nothing deleted yet)

## Disk usage BEFORE cleanup

| Path | Size | Class | Action |
|---|---|---|---|
| models/kokoro/kokoro-v1.0.onnx | 310 MB | A REQUIRED | KEEP (production model) |
| models/kokoro/voices-v1.0.bin | 27 MB | A REQUIRED | KEEP (production voices) |
| data/cache/mel-v1/ | ~6.1 GB | D REGENERABLE | DELETE after confirming no from-scratch resume needed |
| data/processed/librittsr-clean100/ | ~6.5 GB | B UNIQUE | KEEP if Arch resume may include from-scratch; DELETE if Kokoro-only |
| data/processed/teacher-kokoro/ | ~3.6 GB | B UNIQUE | KEEP (17.87h irreplaceable) |
| data/raw/librittsr-full/ | ~7 GB raw wavs+txt | C REGENERABLE | DELETE (re-downloadable from OpenSLR) |
| data/downloads/*.tar.gz | ~10 GB | C REGENERABLE | DELETE (re-downloadable) |
| experiments/r2/checkpoints/ | 177 MB × 4 = 708 MB | B UNIQUE research | KEEP (suspended-path evidence) |
| experiments/r3/checkpoints/ | 177 MB × 4 = 708 MB | B UNIQUE research | KEEP (suspended-path evidence) |
| experiments/rescue-long/checkpoints/ | ~354 MB | B UNIQUE research | KEEP |
| experiments/rescue-prosody/checkpoints/ | ~177 MB | B UNIQUE research | KEEP |
| artifacts/pre-arch-top3/candidates/ | ~50 MB | C REGENERABLE | DELETE (superseded by v3-final) |
| artifacts/v1-listening-pack/ | ~30 MB | A REQUIRED | KEEP (founder listening pack) |
| artifacts/caedral-voice-v1/ | ~50 MB | A REQUIRED | KEEP (V1 deliverable) |
| .venv/ | ~2 GB | A REQUIRED | KEEP (or recreate on Arch) |

## Estimated space recoverable

| Item | Space |
|---|---|
| data/cache/mel-v1/ | ~6.1 GB |
| data/raw/librittsr-full/ | ~7 GB |
| data/downloads/*.tar.gz | ~10 GB |
| artifacts/pre-arch-top3/candidates/ | ~50 MB |
| **TOTAL** | **~23 GB** |

## Rules

1. DO NOT delete anything until founder confirms GO ARCH NOW
2. DO NOT delete suspended-path checkpoints (research evidence)
3. DO NOT delete teacher corpus (unique, 17.87h)
4. Backup groups must exist OUTSIDE C:\ before any deletion
