# PRE-ARCH ENHANCED BASE — Kokoro + Caedral Frontend

**STATUS: WORKING INTEGRATION — Caedral frontend drives a proven acoustic
foundation with parity/better vs stock. NOT V1; quality iteration continues.**

## What this is

- **Foundation:** Kokoro v1.0 ONNX acoustic model (Apache-2.0), 82M params,
  proven CPU-friendly (RTF ~0.32–0.5 warm on i7-13620H [MEASURED]).
- **Caedral layer:** our text frontend (CMUdict lexicon + g2p_en fallback +
  char-level IPA→Kokoro-vocab mapping) drives the acoustic model directly via
  `is_phonemes=True`.

## Honest attribution (founder directive §2)

The underlying Kokoro weights are **third-party open weights (Apache-2.0)**.
They are NOT claimed as Caedral-trained. Caedral Voice V1 built this way is
documented as: *Caedral system derived from and improving upon the open
Kokoro foundation* — with our frontend, runtime, voices work, benchmarks, and
future trained deltas as the Caedral contribution.

## A/B evidence (vocab-mapped-v2, MEASURED 2026-08-23)

| sentence | baseline stock | enhanced mapped | winner |
|---|---|---|---|
| evalA (incl. "Caedral") | WER 0.222 | **WER 0.111** | enhanced |
| propernoun | 0.111 | 0.111 | tie |
| numbers | NATURAL 0.0 | NATURAL 0.0 | tie |
| abbrev (AVX2/AVX-512) | 0.222 | 0.333 | baseline |

Full rows: `benchmark.json` + `vocab-mapped-v2-results.json`.
Known weak spot: letter-by-letter acronyms ("AVX-512").

## Files

- `voice-01.wav` — af_heart, evalA via Caedral phonemes
- `voice-02.wav` — am_michael, numbers via Caedral TN+phonemes (NATURAL)
- `voice-03.wav` — af_heart, numbers variant
- `baseline/` — stock Kokoro renders for comparison
- `enhanced/` + `vocab-mapped*/` — full A/B sets

## Next levers (priority order)

1. acronym expansion rules in TN (fixes "AVX-512" regression)
2. longer-sentence prosody continuity
3. RTF re-measure at idle CPU (current numbers thermally inflated)
