# CV-G1-E0001 — Text frontend bake-off (Phase E)

**Date:** 2026-08-22 · **Corpus:** caedral-eval-corpus v1.0 (frozen 2026-08-22, 101 items, 16 categories)
**Candidate:** `caedral_voice.text.TextFrontend('lexicon')` — own rule-based TN → CMUdict 0.7b lookup → morphology → letter-rule fallback; ARPAbet→IPA via our own mapping table.
**Reference:** `TextFrontend('espeak_ref')` — phonemizer + espeakng_loader, **GPLv3, DEV-ONLY** (quarantined behind a lazy import; CI test asserts `caedral_voice.text` never imports it).
**Config:** `config.json` · **Raw results:** `results/*.jsonl` · **Runner:** `run_bakeoff.py` · All numbers below are [MEASURED] on this machine unless tagged otherwise.

---

## HYPOTHESIS

From docs/research/TEXT-FRONTEND.md: (H1) phoneme streams run ≈60–75% of character length for typical English prose (§6 estimate); if measured ≥90%, Candidate A's sequence-length benefit claim dies (E-FRONT-01 criterion). (H2) CMUdict lexicon coverage suffices at ≲2% OOV token rate; above that a neural G2P fallback is mandatory (E-FRONT-02 criterion). (H3) A lexicon-first frontend is cheap enough at profile-C serving (<10% of RTF budget). (H4) Heteronyms can be surfaced reliably from CMUdict variant structure without context machinery.

## METHOD

Single deterministic pass over the full frozen corpus, batch-1, one warmup item excluded from timing.

- **(a) Length ratio:** per-item `phoneme_tokens / chars` against ORIGINAL corpus text. Phoneme token = one IPA base symbol incl. stress marks ˈ ˌ; ː merges into its vowel; pause tokens | ‖ counted separately (both variants reported).
- **(b) OOV:** word tokens after TN, lowercased; miss = absent from CMUdict even after NFKD accent-fold retry. Second metric = tokens that end at the letter-rule fallback in the actual stream.
- **(c) Latency:** `perf_counter_ns` around `normalize_detailed()` and around full `process()`; g2p = total − normalize; median over all items.
- **(d) Heteronyms:** het_001–010 with known target word per item; recall = flagged-ambiguous occurrences / total target occurrences. Ambiguity := ≥2 distinct CMUdict pronunciations, minus a documented function-word/reduction exclusion list (the, as, into, …).
- **(e) Reference comparison:** 20 items sampled by even spacing over sorted IDs (deterministic; list in summary.json). Both systems fed OUR normalized text. Per positionally aligned word pair, compare the SET of vowels immediately following a primary stress mark ˈ. Strict rate counts ref-unmarked function words as mismatches; secondary rate restricts to pairs where espeak marks any primary stress (isolates content-word placement from citation-vs-contextual stress policy).
- **(f) TN accuracy:** 26 hand-authored gold substrings across num_001–010, checked verbatim post-normalization.

Graceful-skip contract exercised: edge_001/002 (empty/whitespace) skipped with reason; no item errored.

## RESULT

| Metric | Value | Label |
|---|---|---|
| (a) Corpus mean char→phoneme ratio (excl pauses) | **0.9443** (incl pauses 0.9628) | [MEASURED] |
| (a) Prose categories only (dialogue 0.767 … formal_prose 0.911) | 0.77–0.91 | [MEASURED] |
| (a) numbers_dates_currency category | **1.566** (TN expands length) | [MEASURED] |
| (a) acronyms 1.024 / technical 0.992 | ≥1.0 risk classes | [MEASURED] |
| (b) OOV raw miss rate | **2.01%** (50 / 2,493 tokens); 42 reach letter-rule fallback (84%) | [MEASURED] |
| (b) Top OOV classes | tech brands/acronyms (avx, simd, json, postgres, redis, hmac, ci, faq, covid), names (rajesh, kwame, hochreiter, schmidhuber), derived forms (awnings, hyperthreading, microservice, seashores) | [MEASURED] |
| (c) Median latency batch-1: normalize / g2p / total | **0.061 / 0.126 / 0.187 ms** per item; max 13.59 ms (long_001 ≈1,900 words) | [MEASURED] |
| (d) Heteronym flag recall | **22/22 = 100%**; precision 22/29 ≈ 76% (7 reduction-variant false flags: began, business, will, in, around, travelers×2) | [MEASURED] |
| (e) Stress agreement vs espeak_ref (395 aligned pairs) | strict **60.8%**; when espeak marks stress (295 pairs) **81.4%** | [MEASURED] |
| (f) TN gold accuracy num_001–010 | **26/26 = 100%**, zero mismatches | [MEASURED] |

## INTERPRETATION

1. **H1 FALSIFIED at corpus level [MEASURED].** The §6 estimate of 60–75% assumed latinate prose; this corpus is conversational + numeral-heavy. At 0.944 overall (and >1.0 where TN expands numerals or acronyms letter-spell), the sequence-length argument for Candidate A **dies on this corpus**: attention/embedding savings are marginal. The remaining Candidate-A arguments (licensing ownership, auditability, knowledge localization) stand; length is no longer one of them. Per E-FRONT-01's own pass-fail rule, report MEASURED ratio and drop the claim.
2. **H2 borderline [MEASURED].** 2.01% sits exactly at the E-FRONT-02 rejection threshold, and misses concentrate in proper nouns + tech vocabulary — precisely the audible-failure classes. Lexicon-only cannot ship; the OOV ladder needs a real fallback (DeepPhonemizer-class ONNX, E-FRONT-03) rather than letter rules (our crude rules produced e.g. "yuki"→"iʌkɪ" — loud but poor).
3. **H3 confirmed [MEASURED].** Sub-millisecond median total (0.187 ms/item) is orders of magnitude inside any RTF budget; frontend cost is a non-issue at every serving profile.
4. **H4 confirmed for recall, not precision [MEASURED].** CMUdict variant structure surfaces all 22 torture-set occurrences (incl. the 3× "lead" item). First-sense selection remains wrong ~50% of the time by construction (context disambiguation out of scope, documented in code); the 7 false flags are stress-reduction alternates (will/began/business), fixable by extending the reduction-exclusion list rather than new machinery.
5. **Stress agreement 81.4% content-word [MEASURED]** is respectable for a dictionary-vs-synthesizer comparison; residual mismatches are mostly espeak's contextual re-stressing of monosyllables and our schwa/ɚ transcription choices, not placement errors. Method caveat documented in config.json.
6. **GPL hygiene held [MEASURED by test]:** importing `caedral_voice.text` never imports phonemizer/espeakng_loader; regression-tested in tests/test_text_frontend.py.

## KEEP / REJECT / INVESTIGATE

**KEEP**
- Own-chain TN engine (rule-based, zero deps): 26/26 gold, deterministic, fast — production path confirmed for covered classes.
- CMUdict 0.7b as main lexicon (`data/lexicons/cmudict-0.7b`, BSD-style CMU license, header verified; provenance in `data/lexicons/README.md`) + own ARPAbet→IPA map.
- Heteronym surfacing contract (all options + ambiguous flag) — free recall, correct architecture for later context classifiers.
- Pause mapping , ; : → | and . ! ? → ‖ (feeds §5.1 design).

**REJECT**
- The "phonemes are 25–40% shorter" rationale for Candidate A on English prose like this corpus — replaced by measured 5.6% median saving (0.77–0.91 prose range).
- Letter-rule fallback as anything but a last-resort flag-bearer (quality visibly poor; keep only because it fails loudly).
- Lexicon-only shipping posture (no neural fallback) — at/over the 2% threshold with worst-case content classes.

**INVESTIGATE (next experiments)**
- E-FRONT-03: neural G2P bake-off (DeepPhonemizer ONNX vs CharsiuG2P vs g2p_en) on the 50-word OOV list captured in `results/oov_top30.json`.
- Extend reduction-variant exclusion (data-driven: mine CMUdict words whose variants differ only by stress/schwa) to lift heteronym precision above 90%.
- Re-measure ratio on a prose-skewed corpus before locking the acoustic alphabet (Appendix C): pause/stress token budget should assume ~1:1 with chars, not 0.65.
- espeak stress-policy deltas (81.4%→?) after adding contextual re-stressing heuristics for monosyllabic function/content words.

*Reproduce:* `.venv/Scripts/python.exe -X utf8 experiments/CV-G1-E0001-frontend-bakeoff/run_bakeoff.py` (writes config.json + results/* fresh; deterministic sampling).
