# CV-G1-E0004 — neural G2P fallback (g2p_en) evaluation

Date: 2026-08-22. Owner: coordinator.

## HYPOTHESIS

A lightweight neural LTS fallback (g2p_en, Apache-2.0) placed after the
morphology rung of our OOV ladder covers the residual ~2% lexicon-miss rate
without GPL dependencies and without breaking determinism guarantees.

## METHOD

- Input: the exact top-30 OOV word list from CV-G1-E0001 (avx, simd,
  awnings, backoff, …).
- g2p_en via lazy import; NLTK cmudict + POS tagger data downloaded once to
  user cache. Validation: ≥1 valid ARPAbet phone per word; ARPAbet→IPA via
  the existing lexicon mapper.
- Wiring: `G2PPipeline(neural_fallback=True)` / `TextFrontend('lexicon+g2pen')`
  inserts the fallback between morphology and letter rules.

## RESULT [all MEASURED]

- Coverage on the E0001 OOV list: **30/30** at **5.9 ms/word** (batch-1).
- Frontend suite still green with fallback enabled path added (72/72).

## INTERPRETATION

The shippable frontend ladder is now: lexicon → morphology → g2p_en
(Apache-2.0) → letter rules. Combined miss rate ≈ 0% on corpus text;
espeak remains dev-only reference. Latency stays negligible vs synthesis
(0.19 ms/item base + ≤6 ms/word only on misses).

## KEEP / REJECT / INVESTIGATE

**KEEP** as the production default posture (`lexicon+g2pen`), pending one
open question for Phase J data prep: whether TRAINING transcripts should be
phonemized with this same ladder (consistency argument) or espeak_ref for
higher stress accuracy on dev-only preprocessing (quality argument).
Decision lands in Phase J config review — either way inference uses the
shippable ladder.
