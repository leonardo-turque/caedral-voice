# CAEDRAL VOICE V1 — CPU PERFORMANCE REPORT

**Date:** 2026-08-23 · **Machine:** i7-13620H 10C/16T, 24GB RAM, CPU-only
**Model:** Kokoro v1.0 ONNX (Apache-2.0), 310 MB + Caedral frontend/runtime

## Thread sweep (warm, median of 5)

| Threads | RTF | Load time | Warm RSS |
|---|---|---|---|
| 1 | 1.66 | 5.0s | 794 MB |
| 2 | 0.92 | 3.6s | 795 MB |
| **4** | **0.72** | **3.5s** | **796 MB** |

## Speed control validation (caedral-m1 @4thr)

| Speed | Audio duration | Peak | Finite | Clipped |
|---|---|---|---|---|
| 0.5x | 3.03s | 0.592 | ✓ | ✗ |
| 1.0x | 1.47s | 0.572 | ✓ | ✗ |
| 1.5x | 0.87s | 0.511 | ✓ | ✗ |
| 2.0x | 0.75s | 0.362 | ✓ | ✗ |

All speeds produce finite audio without clipping. Pitch/timbre preserved by
Kokoro's native speed parameter (not WAV stretching).

## Technical vocabulary audit [MEASURED]

| Voice | WER on technical sentence |
|---|---|
| caedral-m1 | 0.444 PARTIAL_SPEECH |
| caedral-m2 | 0.444 PARTIAL_SPEECH |
| caedral-f1 | 0.444 PARTIAL_SPEECH |
| caedral-f2 | 0.444 PARTIAL_SPEECH |

TN expansion IS working at text level ("AVX-512" → "A V X five hundred twelve")
but the Kokoro acoustic model renders letter-spelled sequences less clearly than
its internal phonemizer handles them. Known limitation for V1.

## Quality summary across all sentences

| Sentence | All 4 voices |
|---|---|
| Control ("Caedral") | INTELLIGIBLE (WER 0.11) — correct pronunciation via our lexicon |
| Numbers ($1,247.83) | NATURAL (WER 0.00) — perfect TN |
| Propernoun | INTELLIGIBLE (WER 0.11–0.22) |
| Longform | NATURAL/INTELLIGIBLE (WER 0.05–0.13) |
| Technical (AVX) | PARTIAL_SPEECH (WER 0.44) — known limitation |
