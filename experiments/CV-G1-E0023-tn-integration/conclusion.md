# CV-G1-E0023-tn-integration — Conclusion

Date: 2026-08-23 · Scope: verify CaedralVoiceKokoroEngine.phonemize() end-to-end
on the E0020 TN expansion; fix integration defects; add technical pronunciation
overrides · Model: models/kokoro/kokoro-v1.0.onnx (114-symbol vocab)

## Mission verdict

The canonical engine **is** on the latest TN: `phonemize()` routes through
`TextFrontend("lexicon+g2pen")` -> `normalize_detailed()`, so every E0020 rule
(CPU/AVX/GPT letter-spacing, AVX-512 compound, HTTP/2 slash-version, ISO date,
bare domains, currency) was already live in the engine path. Verification
found **two integration defects**, both fixed in this experiment.

## Per-string verification (post-fix; engine.phonemize(), skipped=[] on all five)

| # | Input | Normalized output | Verdict |
|---|---|---|---|
| 1 | The CPU supports AVX2 instructions but not AVX-512. | The C P U supports AVX two instructions but not AVX five hundred twelve. | PASS |
| 2 | RTX 5090 costs $29.99 at 3.5 GHz with 10 GB VRAM. | R T X fifty ninety costs twenty-nine dollars and ninety-nine cents at three point five gigahertz with ten gigabytes VRAM. | PASS (after fixes) |
| 3 | Python 3.13 and CUDA 12.8 support HTTP/2. | Python three point thirteen and CUDA twelve point eight support HTTP version two. | PASS (after fixes) |
| 4 | Visit https://caedral.com or email support@caedral.com | Visit caedral dot com or email support at caedral dot com | PASS (caedral override fires) |
| 5 | GPT-5 achieves 99.9% on 2026-08-23 benchmarks. | GPT five achieves ninety-nine point nine percent on August twenty-third twenty twenty-six benchmarks. | PASS |

Pre-fix evidence: `verify_results.json` (regenerated post-fix), defects
reproduced below.

## Defects found and fixed

**D1 — spaced units mangled before the unit rule** (`normalize.py`).
"3.5 GHz" normalized to "three point five G Hz" and "10 GB" to "ten G B":
the E0020 CamelCase rule R7 ("GHz" -> "G Hz") and acronym rule R5 ("GB" ->
"G B") fire inside `_expand_technical_tokens`, which runs BEFORE
`_convert_units`; only the tight forms ("3.5GHz", "10GB") survived. Fix: R5/R7
now skip cores in the `_UNITS` table so "NUM UNIT" pairs reach the unit rule,
which already accepts one space.

**D2 — acronym-split "A" read as the English article** (`g2p.py`).
TN splits VRAM/CUDA/RAM into single capitals; lone "A" hit
`_SINGLE_LETTER_WORD_EXCEPTIONS` and read as schwa "ə" — "vee arr *uh* em".
Fix: new `single_cap_run_members()` detects maximal runs of >= 2 single
capitals; members are forced to letter names (`resolve_word(...,
force_letter_name=True)`). A run of ONE keeps the word reading, so a
sentence-initial article ("A new era") is unaffected. Side benefit: time-of-day
"A M"/"P M" now reads "ay em" instead of "uh em".

## Pronunciation overrides — data/pronunciation/technical.json

Key insight: whole-word overrides can only fire if TN leaves the surface form
intact, so `_expand_technical_tokens` R5/R7 now also skip tokens whose
lowercase form has an override key (lazy-loaded from data/pronunciation/*.json;
missing dir degrades to an empty set). Entries (all vocab-safe, skipped=[]):

| Key | IPA | Replaces |
|---|---|---|
| cuda | kˈuːdə | letters C-U-D-A with schwa A |
| vram | vˈiːɹæm | letters V-R-A-M |
| nvme | ˌɛn vˈiː ˌɛm ˈiː | Camel split "NV Me" = "en vee me" |
| pcie | pˈiː sˈiː ˈaɪ ˈiː | Camel split "PC Ie", g2pen guessed |
| bios | bˈaɪɑs | letters "bee-eye-oh-ess" |
| ghz / mhz / khz | gigahertz/megahertz/kilohertz family | bare units with no preceding number (with a number the unit rule already expands them) |

## Test results

- tests/test_tn_expand.py — 50 passed (incl. new pins: spaced units, CUDA
  kept whole, letter-run A, sentence-initial article A, technical overrides)
- tests/test_kokoro_map.py — 8 passed
- tests/test_kokoro_engine.py — 4 passed / 2 FAILED, both NOT attributable to
  this experiment:
  - test_speed_validation: a concurrent engine change (committed d6d3b31)
    rewrote synth() to CLAMP speed into [0.25, 2.0] instead of raising; the
    test still pins the old raise. Deterministic failure of that agent's
    in-flight contract; synth()/phonemize() are out of E0023's lane per
    PROD-ENGINE-01.
  - test_two_synths_reuse_state_second_faster_or_equal: warm-vs-cold wall-time
    assert failed by 9 ms (4.085 vs 4.076 s) — the exact thermal-throttling
    flake documented in CV-G1-E0020/E0021 conclusions ("warm > cold in this
    run").
  Neither failing path touches normalize.py/g2p.py/pronunciation data
  ("Hello." contains no technical tokens); the other 106 tests across the four
  suites pass.
- tests/test_text_frontend.py (bonus guard) — 44 passed

## Notes for the record

- kokoro_engine.py was NOT modified by this experiment (read-only calls to
  phonemize()). A concurrent process committed the working tree as d6d3b31 at
  19:27 AST, sweeping these E0023 files into HEAD; no commit was made by
  E0023 itself.
- Artifacts: verify_engine_tn.py (rerunnable verification),
  verify_results.json (engine-level norm+phoneme+skipped per string).

## Files touched

- caedral_voice/text/normalize.py — D1 fix + override-key awareness
- caedral_voice/text/g2p.py — D2 fix (letter runs)
- data/pronunciation/technical.json — NEW (8 entries)
- tests/test_tn_expand.py — regression pins for D1/D2 + overrides
- experiments/CV-G1-E0023-tn-integration/{verify_engine_tn.py, verify_results.json, conclusion.md}
