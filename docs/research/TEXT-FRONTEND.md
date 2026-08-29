# TEXT FRONTEND SURVEY — Caedral G1 (English)

**Track:** Phase B, Researcher B (text frontend & phonemization)
**Date:** 2026-08-22
**Status:** COMPLETE for licensing-critical claims (checked against LICENSE/COPYING files or PyPI metadata on this date); see per-item verification tags.
**Scope:** Choose Caedral G1's English text frontend: input representation, G2P toolchain, pronunciation dictionaries, text normalization, prosody surface — with licensing rigor appropriate to a proprietary commercial product.
**Environment anchors (from MASTER-PLAN §4):** Windows 11 native, Python 3.13.15, no local C/C++/Rust compilers, ONNX Runtime wheels preferred, CPU-only serving profiles A–E.

Claim labels follow charter §9: `[MEASURED]` (ours), `[REPRODUCED FROM SOURCE]` (quote/paraphrase of a primary artifact read this session), `[REPORTED]` (third-party claim, e.g. search-result synthesis), `[ESTIMATED]`, `[HYPOTHESIS]`, `[UNVERIFIED]` (plausible but not source-checked — treat as suspect until verified).

---

## 1. Input representations: what working systems actually consume

### 1.1 Taxonomy

| Representation | Description | Who bears the G2P/TN burden |
|---|---|---|
| Raw characters | Orthography incl. punctuation, one symbol per char | The acoustic model |
| Bytes | UTF-8 bytes (ByT5-style) | The model; no OOV, no vocab design |
| Subwords (BPE/unigram) | LLM-style tokenizers (GPT-2 50k-class) | Mostly the model |
| Phonemes (IPA / ARPAbet) | Output of an external G2P+TN frontend | The frontend (deterministic at inference) |
| Hybrid grapheme+phoneme | Dual streams or mixed alphabet | Shared |

### 1.2 Evidence table — representation used by named systems

| System | Scale (params) | Text input | Frontend tool | Source |
|---|---|---|---|---|
| Tacotron 2 (2017) | ~30M | characters | none (learned) | common knowledge [UNVERIFIED here — cited for history only] |
| FastSpeech 2 | ~40M | phonemes (ARPAbet) | CMUdict + G2P fallback | common knowledge [UNVERIFIED here] |
| VITS (2021) | small | **IPA phonemes**: "We use International Phonetic Alphabet (IPA) sequences as input to the prior encoder… convert text sequences to IPA phoneme sequences using open-source software" | Phonemizer/espeak-ng | [REPRODUCED FROM SOURCE][35] |
| StyleTTS 2 | ~small-M | phonemes; demo/inference path installs `phonemizer` + `espeak-ng`; README: "the inference depends on a GPL-licensed package, so it is not included directly in this repository" | espeak-ng via phonemizer | [REPRODUCED FROM SOURCE][37] |
| Piper (VITS-family, ONNX, CPU) | x_low…high tiers (~15–80M est.) | **IPA phonemes**: "espeak-ng has a special mode where it…outputs phonemes…written using the International Phonetic Alphabet (IPA). Turning text into phonemes is hard, so Piper outsources it to espeak-ng" (maintainer M. Hansen) | espeak-ng via piper-phonemize | [REPRODUCED FROM SOURCE][6] |
| Piper successor `piper1-gpl` (2025) | same | same; project renamed "**gpl**" and relicensed GPLv3, "embeds espeak-ng for phonemization" | espeak-ng | [REPRODUCED FROM SOURCE][5] |
| Kokoro-82M (2024–25) | 82M | phonemes from **misaki** (dict + spaCy + num2words; optional espeak fallback) | misaki | [REPRODUCED FROM SOURCE][7][8] |
| KittenTTS (2025) | **15M / 40M / 80M**, ONNX CPU | phonemes: `EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)` via phonemizer + `espeakng_loader` | espeak-ng | [REPORTED][38] |
| E2 TTS (2024) | large (flow/diffusion) | characters padded with filler tokens to mel length (no phonemes, no duration model) | none | [REPORTED — quoted inside F5 paper][35] |
| F5-TTS (2024) | 336M DiT (reported) [ESTIMATED] | **characters**: "Following E2 TTS, we directly use alphabets and symbols for English"; "The character embedding vocabulary size is 2546, counting in the special filler token and all other language characters exist in the Emilia dataset" | jieba+pypinyin for zh only | [REPRODUCED FROM SOURCE][36] |
| Chatterbox (Resemble, 2025) | 0.5B-class LLM-stack | **graphemes/BPE**: EnTokenizer; Turbo variant = GPT-2 AutoTokenizer, `text_tokens_dict_size=50276`; vocab file literally named `grapheme_mtl_merged_expanded_v1.json` | none (raw text) | [REPORTED][39] |
| Orpheus (2025) | Llama-3B backbone | raw text through the native **Llama tokenizer** (BPE); emotive tags (`<laugh>`) as plain text | none | [REPORTED][40] |

### 1.3 What the pattern shows

- Every production-relevant **small (<100M)** CPU model in the table — VITS/Piper family, Kokoro-82M, KittenTTS down to 15M — consumes **external-G2P phonemes**. None of them learns TN+G2P internally; they cannot afford to. [REPORTED — pattern across sources 5–8, 35, 38]
- Systems that skip G2P (E2/F5 chars; Chatterbox/Orpheus LLM-tokenizers) are all **large-data, large-model** systems: Emilia-scale training corpora and/or pretrained LLM backbones provide the internal linguistic knowledge. Their released *checkpoints* are frequently non-commercial (F5-TTS weights CC-BY-NC due to Emilia [REPRODUCED FROM SOURCE][36]) or carry unclear provenance for our purposes. [REPORTED][39][40]
- Byte/subword inputs appear where the *model itself* is an LM over text (Bark-class, Chatterbox, Orpheus) — that paradigm conflicts with Caedral's CPU-small constraint (AR token throughput, hallucination modes; MASTER-PLAN §10-H3 risks). [HYPOTHESIS]
- For **G2P models themselves**, byte-level inputs win: CharsiuG2P showed "byte-level ByT5 significantly outperformed token-based mT5 for multilingual G2P" (Interspeech 2022). Relevant when choosing our *neural G2P fallback*, not our acoustic input. [REPORTED][19]

**Working conclusion for Caedral G1** [HYPOTHESIS, to be falsified in Phase E]: at Caedral's pilot-scale training budget and CPU-small parameter envelope, a deterministic phoneme frontend (external TN + G2P) is the dominant-strategy prior; char-level inputs should be tested only as a control arm and are expected to fail at pilot data scale.

---

## 2. G2P toolchains — licenses verified 2026-08-22

License column reflects the LICENSE/COPYING file or PyPI metadata fetched this session unless tagged otherwise.

### 2.1 espeak-ng — GPLv3. **NOT commercial-safe as a linked/bundled runtime component.**
- License: GNU GPL v3 — confirmed from `COPYING`. [REPRODUCED FROM SOURCE][1]
- Rule-based formant-era synthesizer repurposed as G2P; language-specific rule trees + exception dictionaries; extremely fast (C), tiny footprint, broadest language coverage of any OSS G2P. Speed/coverage reputation: universal adoption (Piper, KittenTTS, sherpa-onnx historically). [REPORTED][5][6][38]
- Accuracy reputation: serviceable but robotic-flavored pronunciations; stress and reduced-vowel mistakes on uncommon words/names; StyleTTS2 notes alternatives are "lower quality due to mismatch" while still shipping espeak. Names/proper nouns are its weakest area. [REPORTED][37][11]
- **Linking exception: refused.** Maintainers pointed complainants to gpl-violations.org; recommended safe paths for proprietary users are "unmodified, standalone eSpeak NG via its CLI" or calling "its unmodified API from a separate process/thread." [REPORTED][2]
- Real-world contamination scare: sherpa-onnx (Apache-2.0) is **removing espeak-ng and piper-phonemize entirely in v2.0** because "espeak-ng is licensed under GPL, it introduces license constraints that are incompatible with the Apache-2.0 license." [REPORTED][15]
- Python 3.13: irrelevant as library (C); CLI works fine; pip wrapper `espeakng_loader` ships prebuilt GPL binaries — fine for dev machines, a distribution landmine if loaded into our process. [REPORTED][38][12]
- Verdict: **dev-time oracle/reference only** (and even then, output data is fine to learn from; the GPL binds the program, not phoneme strings it emits — that reading is ours: [HYPOTHESIS, legal judgment not legal advice]).

### 2.2 phonemizer (bootphon) — GPLv3. Not shippable in-process.
- Confirmed GPLv3 from LICENSE. Python wrapper around espeak-ng festivals/espeak backends. Slow-ish startup, process-pool design. Python 3.13: works per community usage [REPORTED][12] [UNVERIFIED formally]. Irrelevant anyway: GPL excludes in-process shipping. [REPRODUCED FROM SOURCE][3]
- `phonemizer-fork` exists on PyPI (used by misaki's optional fallback) — fork of a GPL project; assume same license [UNVERIFIED]. Do not ship either in-process.

### 2.3 piper-phonemize (rhasspy) — MIT source, **GPL-in-effect binary** (links libespeak-ng).
- Repo license: MIT [REPRODUCED FROM SOURCE][4]. But its purpose is binding espeak-ng into a C++ lib; any binary built from it carries GPLv3 obligations (this is exactly why sherpa-onnx dropped it and why Piper's successor renamed itself `piper1-gpl` under GPLv3). [REPORTED][5][15]
- Contains the useful idea we can copy (license-permissively): punctuation→pause insertion and phoneme-id mapping tables. Ideas are not copyrightable; clean-room reimplementation from behavior is safe. [HYPOTHESIS — standard engineering practice, legal nuance unreviewed]

### 2.4 misaki (hexgrad) — Apache-2.0. Commercial-safe license; operational caveats. See §7 deep dive.
- License Apache-2.0 confirmed (repo sidebar + PyPI). Latest release **0.9.4 (2025-04-05)**; requires-python **>=3.8,<3.13** — **does not install on our Python 3.13.15** as-is. [REPRODUCED FROM SOURCE][8][9]
- English core: pronunciation dictionary + spaCy tokenization/POS + `num2words` for numbers; optional `EspeakFallback` (pulls GPL espeak-ng + phonemizer-fork — must be stripped for our runtime); planned seq2seq fallback and BERT-based homograph disambiguation on the TODO list. [REPRODUCED FROM SOURCE][8]
- Dependency chain flags: `num2words` is **LGPL-2.1** [REPRODUCED FROM SOURCE][32]; spaCy is MIT [REPORTED — well-known] [UNVERIFIED this session].

### 2.5 g2p_en (Kyubyong/g2p) — Apache-2.0. Safe but stale.
- Apache-2.0 confirmed. [REPRODUCED FROM SOURCE][16]
- CMUdict lookup + numpy-inference seq2seq for OOV (TensorFlow removed from deps in v2.0, Apr 2019; deps: numpy, nltk, inflect, distance). Last PyPI release **v2.1.0, 2019-12-31** — unmaintained, but the light dep set means Python 3.13 compatibility is plausible [REPORTED][17]; NLTK `averaged_perceptron_tagger` needed for POS-aware heteronym picks. Known weaknesses: US-only ARPAbet (not IPA — needs mapping), slow first load, dated lexicon. [REPORTED][17]
- A maintained fork appeared Oct 2025 (Mintplex-Labs/g2p-en) [REPORTED][17] [UNVERIFIED contents].
- Verdict: acceptable fallback engine; would need an ARPAbet→IPA layer and vendoring care.

### 2.6 DeepPhonemizer (as-ideas) — MIT. Safe; torch-based; quiet maintenance.
- MIT confirmed. Transformer G2P (forward/CTC or autoregressive), multilingual, TorchScript export, dict-accelerated batched inference — designed "to be used in text-to-speech production systems." Requires PyTorch at runtime unless TorchScript-exported to ONNX-compatible form [UNVERIFIED ONNX export path]. GitHub page shows no formal releases; models reference v0.0.10; activity level low [REPORTED][18]. English accuracy reputation: good (built by the FastSpeech2-PyTorch author). [REPORTED][18]
- Verdict: strong candidate for **our** neural OOV fallback; export to ONNX Runtime to drop the torch runtime dep. [HYPOTHESIS]

### 2.7 CharsiuG2P (lingjzhu) — MIT. Safe; ByT5 multilingual.
- MIT confirmed (repo listing). ByT5 G2P for ~100 languages, HF-hub models, greedy decoding recommended; needs word pre-tokenization and `<eng-us>: ` language prefixes; last push May 2023. [REPORTED][19]
- Verdict: viable fallback; heavier (ByT5 encoder-decoder) than DeepPhonemizer; word-level only (no sentence context, no heteronym logic).

### 2.8 transphone (xinjli) — MIT. Safe.
- MIT confirmed. Multilingual G2P/phone recognition toolkit oriented at low-resource languages; less TTS-production-focused; English quality not widely benchmarked. [REPORTED][20] — include as backup, not lead candidate.

### 2.9 gruut (rhasspy) — MIT, **ARCHIVED 2025-10-06 (read-only)**.
- MIT confirmed; repo archived. Lexicon-from-open-data + CRF G2P per language; no espeak dependency. Was StyleTTS2's suggested non-GPL alternative (at acknowledged quality cost). [REPRODUCED FROM SOURCE][21][37]
- Verdict: dead-end for new dependencies, but validates the "lexicon + statistical fallback without espeak" architecture.

### 2.10 Newer 2025–2026 options
- `espeakng_loader` + `phonemizer-fork`: pip-wheel packaging of GPL espeak-ng used by KittenTTS/misaki — convenient dev tool; **distribution landmine** if imported into our shipped process. [REPORTED][8][38]
- FluidAudio (Rust, Kokoro-compatible) documents that runtimes which bypass full misaki G2P get worse heteronym behavior (sherpa-onnx#2004) and warns of silent degradation to weaker G2P paths (FluidAudio#727) — evidence that **frontend fidelity is a real quality axis**, not a detail. [REPORTED][13][14]
- No Apache/MIT-licensed English G2P of espeak-grade coverage surfaced in searches this session; the niche "permissive-license, espeak-quality, pip-installable, py3.13" is **empty as of 2026-08-22** — an opportunity for Caedral to own it. [REPORTED across searches; absence-of-evidence caveat applies]

---

## 3. Pronunciation dictionaries, heteronyms, OOV

### 3.1 CMUdict and forks

- **CMUdict 0.7b** (cmusphinx/cmudict, maintained mirror): license is CMU's own permissive text — "Copyright (C) 1993–2014 by Carnegie Mellon University… Use of this dictionary for any research or commercial purpose is completely unrestricted," attribution requested on redistribution. [REPRODUCED FROM SOURCE][29] → **commercial-safe; keep the notice in our licenses dir.**
- Coverage ~134k English headwords incl. proper-noun entries and variant pronunciations (parenthesized variants); ARPAbet symbols with stress digits. [REPORTED][30][31]
- **IPA conversions:** `menelik3/cmudict-ipa` converts all of 0.7b to IPA (AH→ʌ/ə by stress, vowel lengthening rules) plus a Brown-corpus frequency list of 33,862 words [REPORTED][30]; repo's own license **[UNVERIFIED]** — but as a mechanical transformation of unrestricted CMUdict data with a published rule table, re-deriving our own conversion from the rules is trivially safe. `lingz/cmudict-ipa` is MIT-licensed with train/dev/test splits [REPORTED][31]. **Decision-relevant fact:** we can generate our own IPA lexicon from CMUdict + a ~30-line ARPAbet→IPA map; no third-party fork dependency needed. [HYPOTHESIS → cheap to verify in Phase E]
- Staleness: upstream frozen at 2014-era vocabulary (0.7b). Missing modern terms, brands, slang — this gap is exactly what OOV machinery must cover. [REPORTED][29]

### 3.2 Heteronyms / homographs ("read", "lead", "bass", "live", "wind", "tear")

- Dictionaries store multiple pronunciations; *selection* needs context. Approaches seen:
  - POS tagging then lookup (g2p_en uses NLTK perceptron tagger [REPORTED][17]); misaki uses spaCy tokenization/POS today, with planned BERT-embedding + logistic-regression classifiers for homograph disambiguation on its TODO (explicitly targeting "axes bass bow lead tear wind"). [REPRODUCED FROM SOURCE][8]
  - Full neural G2P over sentence context would resolve some cases implicitly, but word-level fallback models (CharsiuG2P, DeepPhonemizer) do not see context. [REPORTED][18][19]
  - Runtimes that simplify the frontend demonstrably lose context-dependent pronunciation quality (sherpa-onnx vs full misaki: "read" past/present handled only by the latter). [REPORTED][14]
- Heteronym errors are user-audible and reputation-damaging; they deserve a dedicated eval set regardless of chosen design. [HYPOTHESIS]

### 3.3 OOV ladder (recommended shape)

1. User/custom lexicon overrides (per-voice or global; misaki's "golds" pattern proves demand: `pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'`). [REPRODUCED FROM SOURCE][11]
2. Main lexicon lookup (CMUdict-derived IPA).
3. Morphological fallback (plural/-ed/-ing stripping + re-lookup) [ESTIMATED — standard technique].
4. Neural G2P (DeepPhonemizer-class, ONNX-exported, MIT) with confidence gating.
5. Last-resort letter-to-sound or letter-name spell-out, flagged in logs.
- Anti-pattern documented in the wild: silent degradation when the intended G2P path fails to load (FluidAudio#727: fluent audio with wrong words after hidden fallback to weaker G2P; warnings buried in logger). Our runtime MUST fail loudly / log-and-metric every fallback step-down. [REPORTED][13]

---

## 4. Text normalization (TN)

### 4.1 Landscape

| Tool | License | Approach | Platform/py3.13 reality |
|---|---|---|---|
| NeMo-text-processing (NVIDIA) | Apache-2.0 [REPRODUCED FROM SOURCE][22] | WFST grammars over pynini/OpenFst; semiotic classes (cardinal, ordinal, date, time, money, measure, address, electronic/URL/email, whitespace/verbatim class) | pip officially Linux x86_64; `nemo_text_processing==1.2.0` pins `pynini==2.1.6.post1`; Windows pip unsupported; py3.13 support still open (NeMo#14729, Sep 2025). pynini 2.1.7 itself now ships cp313 manylinux wheels. → usable via WSL2/Docker (dev-time), not native Windows prod. [REPORTED][23][24][25] |
| WeTextProcessing (wenet-e2e) | Apache-2.0 [REPRODUCED FROM SOURCE][26] | FST rule graphs (pynini/OpenFst), en/zh/ja, FST caching, n-best, C++ runtime option; reuses NeMo zh data | pip-installable; Windows-native untested here [UNVERIFIED]; same pynini substrate constraints |
| wetext-rs (SpenserCai) | Apache-2.0 [REPORTED][27] | Rust port of WeTextProcessing on rustfst | wheel/Rust story good for us later; compiler absent locally today (MASTER-PLAN §4) |
| text-processing-rs (FluidInference) | Apache-2.0 [REPORTED][28] | Rust port of NVIDIA NeMo TP (TN+ITN) on rustfst | same |
| inflect 7.5.0 | MIT [REPRODUCED FROM SOURCE][33] | pure-Python number→words, ordinals, articles a/an by pronunciation, plurals | py≥3.9; ideal CPU-first building block |
| num2words | **LGPL-2.1** [REPRODUCED FROM SOURCE][32] | number/currency/date → words, many languages | LGPL usable but a compliance obligation we can avoid — see below |

### 4.2 Learned TN
NeMo has moved toward neural/neural-hybrid TN (WFST + LM shallow fusion, Interspeech 2022 lineage) [REPORTED][22]; industry default remains deterministic WFST/rule engines because errors are auditable and fixable. For Caedral G1, learned TN adds an unaccountable failure surface on top of a small acoustic model — rejected as primary [HYPOTHESIS].

### 4.3 Known failure modes to test against (compile into the torture corpus)
Ambiguous bare numbers ("1776" year vs count), ranges ("3-5"), currencies and mixed units, phone numbers vs IDs vs dates, URLs/emails/handles, abbreviations needing expansion decisions ("Dr.", "St.", "vs.", "approx."), acronyms vs words ("READ", "LETS" all-caps bug class in Kokoro#71 [REPORTED][10]), ordinals in legal text, math expressions, emoji/symbols. Rule-based engines also break on nested/malformed constructs; WFSTs fail by *silently passing through* unmatched text — detection requires a coverage metric, not spot checks. [REPORTED across 10,22,23; synthesis HYPOTHESIS]

### 4.4 Caedral TN stance
Two-tier: (1) **dev-time silver reference** = NeMo TN en-US inside Docker/WSL2 (Apache-2.0, output data freely reusable) to bootstrap training-data normalization and score our engine; (2) **runtime engine = ours**, pure Python: tokenizer + regex semiotic spotters + inflect(MIT) + hand-written WFST-free expansions, with a pass-through-with-warning contract for anything unrecognized. LGPL num2words avoided entirely (inflect covers number→words; currency/unit templates are small custom code [ESTIMATED effort]). [HYPOTHESIS → E-experiments]

---

## 5. Prosody representation at the frontend

- **Punctuation → pauses:** the small-model families keep punctuation in the phoneme stream and map it to silence/pause symbols (KittenTTS: `preserve_punctuation=True` through espeak phonemization [REPORTED][38]; piper-phonemize implements pause insertion for punctuation [REPORTED][4]). Cheap, robust, and the single highest-value prosody cue at this scale. [HYPOTHESIS]
- **Stress:** carried by IPA stress marks (ˈ ˌ) from the lexicon/espeak/misaki output; KittenTTS explicitly requests `with_stress=True`. Stress marks materially help small models place rhythm. [REPORTED][38] [HYPOTHESIS for the effect size]
- **Emphasis / SSML:** commercial clouds expose SSML (`say-as`, `prosody`, emphasis); open small-model stacks expose essentially **none** (Piper/Kokoro/KittenTTS have no SSML surface; Kokoro's only inline control is phoneme-forcing `[word](/phonemes/)` [REPORTED][11]). Orpheus smuggles expressivity as plain-text emotive tags (`<laugh>`) — an LLM-era pattern requiring an LLM backbone. [REPORTED][40]
- **Style vectors:** StyleTTS2-lineage systems (incl. Kokoro) move global expressivity into a style/voice embedding, leaving the frontend to handle only local prosody (pauses, stress). This matches MASTER-PLAN H1/H4 (predictors carry rate/pitch/energy). [REPORTED][7][37]
- **Caedral G1 surface** [HYPOTHESIS → falsify in E]: phoneme stream with (a) explicit pause tokens derived from punctuation, (b) optional lightweight inline markup of our own design (emphasis + break + phoneme-override, superset-compatible with misaki's override syntax so we can interoperate), (c) everything global (rate/pitch tendency/energy) handled by acoustic-side predictors, not the frontend.

### 5.1 Proposed pause hierarchy (Candidate A detail) [HYPOTHESIS]
Map punctuation to a small ordered set of pause symbols so the duration predictor can learn graded silences:

| Input cue | Symbol | Intended behavior |
|---|---|---|
| `,` `;` `:` `—` | `|` | short clause break |
| `.` `?` `!` | `\|\|` | sentence break + pitch-reset cue |
| paragraph / explicit break tag | `\|\|\|` | long break |
| no punctuation across a long clause | (none) | model/predictor decides |

Rationale: Piper-family models trained with punctuation-derived pauses read naturally without any explicit prosody model; graded symbols cost nothing at train time and give the predictor a handle for rate control later. Risk: too many pause classes on small data → sparse symbols; start with 2 classes, split only if E-FRONT-07 pilot shows benefit. [HYPOTHESIS]

### 5.2 Emphasis markup sketch (G1-minimal) [HYPOTHESIS]
`<emphasis word="not"/>` or markdown-like `*not*` in our API, compiled to a per-word emphasis flag consumed by the energy/duration predictors (H1) rather than injected into the phoneme string. Keeps the acoustic input alphabet closed (no new symbols to learn) — emphasis becomes a predictor conditioning feature. This mirrors how StyleTTS2-lineage systems externalize expressivity into style vectors while keeping the phoneme alphabet fixed. [REPORTED][7][37]

---

## 6. Sequence-length implications for a small CPU model

- **Phonemes vs characters:** English orthography averages ≈4.7–5.0 chars/word including spaces; CMUdict entries average ≈3.0–3.3 phones/word. IPA adds stress marks and length diacritics (≈+0.3–0.5 symbols/word). Net: phoneme streams run ≈**60–75% of character length** for typical English prose. All figures [ESTIMATED] from dictionary statistics — **Phase E must measure on our frozen corpus (E1).**
- **Attention cost:** self-attention is O(L²); a 30–40% shorter sequence cuts attention FLOPs to ≈0.5–0.65×, and shortens KV caches / activation memory proportionally for AR or attention-bearing designs (MASTER-PLAN H1/H2/H3 all pay attention costs). At profile-C (1 vCPU, ≤1 GB) this is a first-order serving cost. [ESTIMATED — standard complexity argument]
- **Vocabulary/embedding effect:** phoneme alphabets are ~100–300 symbols (Piper-class phoneme_id tables; F5's char vocab was 2546 for multilingual coverage [REPRODUCED FROM SOURCE][36]; GPT-2-class 50k [REPORTED][39]). Small vocab = small embedding + softmax tables and more data per symbol — favorable exactly where Caedral is data-poor. [HYPOTHESIS]
- **The irreversibility tradeoff:** phonemization destroys spelling information — heteronym choice, acronym-vs-word, and name pronunciation are decided *before* the model and cannot be recovered. Char/byte inputs keep all information but push the learning burden into the model (needs scale we lack). This is the core representation tradeoff; hybrid grapheme+phoneme streams recover some spelling cues at +sequence length. [HYPOTHESIS]

### 6.1 Worked example [ESTIMATED]
Sentence: "The resolution was read aloud on March 3, 2026, to 1,500 attendees."
- Characters incl. spaces: ≈ 75 symbols.
- ARPAbet-class phonemization ≈ 16 words × ~3.2 phones + stress marks + 4 pause symbols ≈ 58–62 symbols → ratio ≈ 0.80–0.83 for this sentence (short words, numbers already expanded by TN add length back). For prose with longer latinate words the ratio typically drops toward 0.65–0.70.
- Attention FLOPs at L_char=75 vs L_ph=60: (60/75)² ≈ 0.64× — a real but not dramatic saving; the bigger wins are embedding-table size, more tokens of context per window in streaming designs, and fewer decoder steps per second of audio for frame-aligned architectures. **E-FRONT-01 replaces these guesses with corpus measurements.**

### 6.2 What does NOT get shorter
Phonemes shorten *text-conditioning* sequences, not audio: mel/frame lengths are fixed by audio duration regardless of frontend. Duration predictors (H1) must still map phoneme count → frame count; AR codec-LMs (H3) pay per audio-token regardless of input representation. The frontend choice therefore mostly moves encoder-side compute and knowledge-localization, not total decode cost — worth stating plainly so Phase F doesn't over-credit the frontend for RTF. [HYPOTHESIS]

---

## 7. Misaki deep dive — what it does differently, and is it our foundation?

**What misaki[en] actually does** [REPRODUCED FROM SOURCE][8]: lexicon-first G2P (dictionary of phoneme strings) + spaCy for tokenization/POS/context + `num2words` for numbers; `trf=True` option swaps in a transformer spaCy pipeline (better, slower); `british=True` accent switch; OOV routed to an optional `EspeakFallback` (requires GPL espeak-ng + phonemizer-fork); TODO items include seq2seq fallback trained on its dictionaries and BERT-embedding homograph classifiers. Distinctive features: inline per-word phoneme forcing `[quijote](/kixote/)` and programmatic `golds` lexicon overrides. [REPRODUCED FROM SOURCE][8][11]

**Quality regard:** Kokoro-82M's reputation (small model, outsized quality [REPORTED][7]) is partly credit to the frontend, but the issue tracker shows a consistent failure taxonomy attributable to misaki, not the acoustic model:
- all-caps tokenization bug → letter-spelling ("GO" → "G O") via spaCy sm model (kokoro#71) [REPORTED][10]
- wrong pronunciations needing manual golds/inline overrides (kokoro#156) [REPORTED][11]
- ordinary words ("ambiance", "verified") producing **no phonemes at all** without fallback; espeak-fallback broken on Python ≥3.12 with espeakng-loader (misaki#58) [REPORTED][12]
- silent degradation to a weaker G2P path when the lexicon path fails, with warnings hidden in the logger (FluidAudio#727) [REPORTED][13]
- third-party runtimes replacing full misaki with simplified lexicon lookups measurably lose heteronym handling (sherpa-onnx#2004) [REPORTED][14]

**Viability as Caedral foundation:**

| Factor | Assessment |
|---|---|
| License | Apache-2.0 — clean. [REPRODUCED FROM SOURCE][8][9] |
| Python 3.13 | **Blocked: requires-python `>=3.8,<3.13`** (0.9.4, 2025-04-05). We are on 3.13.15. Options: pin a 3.12 venv for the frontend process, or fork and forward-port (likely small: the pin may be conservative) [UNVERIFIED — actual 3.13 breakage unknown]. [REPRODUCED FROM SOURCE][9] |
| GPL hygiene | Only if `EspeakFallback` stripped; core is clean. |
| LGPL hygiene | `num2words` (LGPL-2.1) in core number path — replaceable with inflect(MIT) or our own. |
| Dep weight | spaCy model required (sm at minimum; trf optional). CPU latency cost must be measured (E6). |
| Maintenance cadence | 92 commits on main at read time; no formal GitHub releases; PyPI releases active through Apr 2025. Single-vendor (hexgrad) project tied to Kokoro's roadmap — bus factor real. [REPORTED][8][9] |

**Verdict** [HYPOTHESIS]: misaki is the best *reference architecture* in the permissive-license space (lexicon + context + overrides), but as a direct dependency it drags py<3.13, spaCy, and LGPL deps. Recommended posture: **fork-and-own** the architecture (attribution under Apache-2.0), re-implementing the thin parts we need with our own deps — not vendoring it wholesale. Kokoro's own docs pattern (dict + POS + overrides + loud fallback ladder) is the blueprint; the code we write is small.

---

## 8. Licensing decision table (commercial-shipping view)

Legend — **Ship?**: in-process inside Caedral's proprietary runtime. **Dev-only?**: usable on dev machines / CI / Docker for data generation. License facts verified this session unless tagged [UNVERIFIED].

| Tool | License | Commercial-safe (ship in-process)? | Runtime-dependency risk | Notes |
|---|---|---|---|---|
| espeak-ng | GPLv3 [1] | **NO** | n/a | CLI/separate-process only per maintainer guidance [2]; phoneme *output* reusable as data [HYPOTHESIS-legal] |
| phonemizer / phonemizer-fork | GPLv3 [3] | **NO** | n/a | wrapper only |
| piper-phonemize | MIT source [4] | **NO as binary** (links libespeak-ng → GPL effect) | high | sherpa-onnx removed it for exactly this [15] |
| piper1-gpl | GPLv3 [5] | NO | n/a | reference code only |
| misaki | Apache-2.0 [8][9] | YES (core) | medium | py<3.13 pin; strip espeak fallback; replace num2words(LGPL) |
| num2words | LGPL-2.1 [32] | CAUTION (weak copyleft obligations) | low | avoid by using inflect/own code |
| inflect | MIT [33] | YES | very low | |
| spaCy (+ sm model) | MIT [REPORTED] [UNVERIFIED this session] | YES | medium (model size, latency) | measure in E6 |
| g2p_en | Apache-2.0 [16] | YES | medium (stale since 2019; NLTK data dep) [17] | fallback option |
| DeepPhonemizer | MIT [18] | YES | low-medium (torch→ONNX export needed) [UNVERIFIED export path] | lead neural-G2P candidate |
| CharsiuG2P | MIT [19] | YES | medium (ByT5 size; HF-hub fetch at load) | alternate neural G2P |
| transphone | MIT [20] | YES | medium (English quality unproven for TTS) | backup |
| gruut | MIT, ARCHIVED 2025-10 [21] | YES but dead | high (no maintenance) | architecture reference only |
| CMUdict | CMU permissive, "completely unrestricted" incl. commercial [29] | YES | very low | keep attribution notice |
| cmudict-ipa (menelik3) | [UNVERIFIED] | derive our own IPA from CMUdict instead | low | rules are published [30] |
| NeMo-text-processing | Apache-2.0 [22][23] | YES (pure-Python + pynini) | high operationally: Linux-oriented pip, pynini pin, py3.13 open issue [23][24][25] | dev-time Docker silver reference |
| pynini | Apache-2.0 [REPORTED PyPI] [UNVERIFIED file] | YES | cp313 wheels exist (2.1.7) [24] | only if we go WFST |
| WeTextProcessing | Apache-2.0 [26] | YES | medium (pynini substrate; Windows-native untested) [UNVERIFIED] | alternative silver reference |
| wetext-rs / text-processing-rs | Apache-2.0 [27][28] | YES | low (rustfst wheels) [REPORTED] | future CPU TN acceleration |
| Kokoro weights/code | Apache-2.0 [7] | YES | — | baseline model for Phase C benchmarks |
| StyleTTS2 code/models | MIT + disclosure condition on pretrained voices [37] | code YES; their voice checkpoints need the disclosure condition honored | — | benchmark only |
| F5-TTS | code MIT; **weights CC-BY-NC** [36] | code YES; weights NO for commercial | — | representation evidence; not a shipping base |

**GPL landmines actually found (summary):**
1. The entire convenient "pip-install espeak" ecosystem (`espeakng_loader`, `phonemizer-fork`, `piper-phonemize` wheels) silently binds GPLv3 into a process that imports it. KittenTTS and misaki's optional path both do this — copying their frontend code naively imports the landmine. [8][12][38]
2. Piper's own successor renamed to `piper1-gpl` and went GPLv3 because of embedded espeak-ng — the upstream project chose copyleft over untangling it. [5]
3. sherpa-onnx is spending a major-version break (v2.0) purging espeak-ng/piper-phonemize from an Apache-2.0 codebase. [15]
4. espeak-ng maintainers explicitly refuse linking exceptions. [2]
5. Quiet LGPL inside an Apache package: misaki's number path depends on num2words (LGPL-2.1). [8][9][32]

---

## 9. Candidate frontend designs for Caedral G1

### Candidate A — "Own-chain phoneme frontend" (lexicon-first, GPL-free, misaki-shaped)
Pipeline: Unicode cleanup → semiotic spotter (regex classes) → TN (own rules + inflect) → tokenizer+POS (spaCy sm or lighter) → heteronym resolution (POS + custom lexicon overrides) → lexicon lookup (CMUdict→our IPA build) → OOV ladder → neural G2P (DeepPhonemizer ONNX) with confidence gate → letter-spell-out last resort → pause/emphasis symbol insertion → phoneme-id sequence.
- Pros: zero GPL/LGPL at runtime; full ownership (charter §3); every stage auditable and metric-able; aligns with risk-register mitigation ("evaluate Apache/MIT alternatives before commitment"); portable across H1–H4 acoustic candidates.
- Cons: we build/maintain ~all of it; heteronym quality starts below misaki-trf until classifiers added; names/OOV quality bounded by our G2P choice.
- Effort [ESTIMATED]: 3–5 person-weeks to working v1 (lexicon build 2–3 d; TN engine core 1.5–2 wk; G2P ONNX integration 3–5 d; pause/emphasis + API 3–4 d), plus E-experiments.

### Candidate B — "Fork misaki, sanitize, forward-port"
Take misaki[en] architecture + selected code under Apache-2.0 attribution; port to py3.13; swap num2words→inflect/own; delete EspeakFallback, substitute DeepPhonemizer-ONNX; keep golds/inline-override syntax; optionally keep spaCy.
- Pros: battle-tested behavior surface (Kokoro fleet exposes its failure modes already documented); fastest route to good heteronym/override ergonomics; stays close to an ecosystem we'll benchmark against.
- Cons: inherits spaCy weight (or needs that swapped too); upstream divergence maintenance; must audit every dependency line-by-line anyway; py<3.13 forward-port effort unknown [UNVERIFIED].
- Effort [ESTIMATED]: 2–4 person-weeks if 3.13 port is easy, more if spaCy also must go.

### Candidate C — "Char-level, no-G2P runtime" (E2/F5-style)
Feed normalized text characters (+ filler padding scheme if flow-matching acoustic chosen); all pronunciation knowledge learned in training.
- Pros: simplest possible runtime; no lexicon/G2P deps ever; nothing to license; no OOV problem; keeps spelling info (heteronyms theoretically learnable from context).
- Cons: requires training scale we do not have (Emilia-class corpora trained F5/E2; our pilot data will be orders of magnitude smaller) — expected pronunciation failures on names/rare words; longer sequences (~1.4–1.6× attention cost vs phonemes [ESTIMATED]); TN still needed upstream (chars don't fix "£3.50"); weaker prosody control surface.
- Effort [ESTIMATED]: <1 person-week of frontend code, but shifts enormous cost to data+training (Phase D/I) — likely infeasible at pilot scale. Included mainly as the falsification control arm for A/B.

**Recommendation** [HYPOTHESIS pending Phase E]: A as primary, B as accelerator (mine misaki's lexicon/POS logic and eval sets while writing A), C only as a small-scale ablation to quantify how much phonemes buy at our data scale.

---

## 10. Phase E experiments (falsification plan)

IDs follow charter §8 (`CV-G1-E####`). Each states what result would kill which candidate.

| ID | Experiment | Falsifies / decides | Pass-fail criterion |
|---|---|---|---|
| E-FRONT-01 | Sequence-length census: run frozen corpus through char pipeline and a draft phoneme pipeline; measure symbol counts, pause distribution | My ESTIMATED 60–75% ratio (§6); Candidate C viability | Report MEASURED ratio; if phonemes ≥90% of char length, A's length benefit claim dies |
| E-FRONT-02 | Lexicon coverage audit: % of corpus word tokens found in CMUdict-derived lexicon; collect full OOV list w/ frequencies | Size of the OOV problem; neural-G2P requirement | If OOV token rate >~2% on production-like text, lexicon-only (no neural fallback) rejected |
| E-FRONT-03 | Neural G2P bake-off: DeepPhonemizer vs CharsiuG2P vs g2p_en vs (dev-only) espeak-ng, on 1000-word human-checked set (in-dictionary OOV-style + proper nouns), PER + latency + ONNX-export feasibility | Which OOV engine enters Candidate A | Pick winner by PER @ ≤X ms/word budget; document ties honestly |
| E-FRONT-04 | Heteronym torture set (read/lead/bass/live/wind/tear/bow/axes…, ≥200 sentences): POS-based selection vs always-first-sense vs misaki outputs | Whether context machinery earns its complexity | If POS method <80% correct, escalate to embedding classifier or accept documented limitation |
| E-FRONT-05 | TN torture corpus (numbers/dates/currency/URLs/abbrev/all-caps/ranges): own rule engine accuracy, scored against NeMo-TN-in-Docker silver output + manual gold | §4.4 two-tier stance; Candidate A's TN core | Coverage metric (pass-through rate ≈0 on known classes); error taxonomy published either way |
| E-FRONT-06 | Frontend latency/RAM microbench on target profiles B/C/D: spaCy-sm vs none vs lightweight tagger; ONNX G2P batch-1 latency; total frontend ms + MB | Candidate A/B serving cost; profile-D feasibility | Frontend must fit ≤10% of RTF budget @ profile C [ESTIMATED budget — confirm with Phase C numbers] |
| E-FRONT-07 | Smoke-scale acoustic ablation: tiny model trained twice (same pilot data, fixed steps) on (a) phoneme input (b) char input; ASR-round-trip WER + listening spot-check | §1.3 central hypothesis (phonemes dominate at small scale) | If (b) ≈ (a) at pilot scale, representation constraint relaxes and Candidate C reopens |
| E-FRONT-08 | py3.13 compatibility matrix: attempt misaki 0.9.4 install/import under 3.13; catalog actual breakage; time-boxed forward-port spike | Candidate B cost estimate [UNVERIFIED item] | Port ≤2 days ⇒ B viable shortcut; else commit to A |

Dependency note: E-FRONT-01/02 unblock 03–05; 06 needs Phase C RTF budgets; 07 needs Phase I smoke trainer.

---

## Appendix A — Python 3.13 / platform quick matrix (frontend candidates only)

| Package | py3.13 native Windows | Basis |
|---|---|---|
| misaki 0.9.4 | **NO (requires-python <3.13)** [REPRODUCED FROM SOURCE][9] | PyPI metadata; actual breakage untested → E-FRONT-08 |
| inflect 7.5.0 | YES (py≥3.9, pure Python) [33] | PyPI |
| g2p_en 2.1.0 | LIKELY YES (numpy/nltk only) [REPORTED][17] | dep set; untested → E-FRONT-03 env check |
| DeepPhonemizer | YES if torch ≥2.6-class wheels; ONNX export removes torch at runtime [UNVERIFIED] | torch publishes cp313 wheels [REPORTED][17-adjacent TF/torch wheel data]; our export path to verify in E-FRONT-03 |
| CharsiuG2P | Same torch caveat; ByT5 runs fine on CPU for word-level latency? — measure [ESTIMATED] | model size ~50–300M params by variant [REPORTED][19] |
| phonemizer/espeak-ng CLI | YES as external process (GPL boundary respected) [2] | maintainer guidance |
| NeMo-text-processing | NO native (Linux-oriented pip; pynini pin) → Docker/WSL2 dev-only [REPORTED][23][24][25] | PyPI + issue tracker |
| CMUdict/cmudict-ipa data | n/a (data files) [29][30] | — |

## Appendix B — TN semiotic class checklist for the torture corpus (E-FRONT-05)

From NeMo/WeTextProcessing class inventories and documented failure discussions [REPORTED][22][23][26], plus Kokoro#71 all-caps class [10]:

1. Cardinals incl. thousands separators ("1,500", "1500", "3.5 million")
2. Ordinals ("3rd", "twenty-first") and fractions ("1/2", "½")
3. Ranges & hyphenated numerics ("3-5", "2020–2024", phone numbers "555-0142")
4. Dates ("March 3, 2026", "03/03/26", ISO "2026-03-03") — ambiguity vs ranges
5. Times ("3:30 pm", "0900 hours")
6. Currency ("£3.50", "$5M", "€1,2", "US$10")
7. Measures & units ("5 kg", "180 lb", "km/h", °C/°F reading rules)
8. Electronic/URL/email/handle ("caedral.ai", "user@host", "#tag")
9. Abbreviations context-dependent ("Dr.", "St.", "vs.", "approx.", "No. 5")
10. Acronyms vs words: all-caps handling ("LETS", "READ", "NASA", "FAQ") — letter-spell vs word-read decision needs a whitelist+heuristic layer [REPORTED bug class][10]
11. Whitespace/verbatim fallback: anything unmatched must pass through with a logged warning (NeMo's VERBATIM class pattern [22])
12. Math/symbols ("=", "<", "%", "&" → "and"?), emoji policy (drop/log)
13. Roman numerals ("Chapter IV") — often missed by rule engines [HYPOTHESIS]
14. Sentence-boundary edge cases: quotes, ellipses, em-dash interruptions (feeds §5.1 pause mapping)

Each class gets ≥20 corpus sentences with hand-gold verbalizations; coverage = fraction of tokens the engine transforms *at all*; accuracy = correct verbalization vs gold.

## Appendix C — Caedral phoneme alphabet proposal (Candidate A) [HYPOTHESIS]

- Base: IPA broad transcription of en-US, derived deterministically from ARPAbet via published mappings (menelik3 rules as reference: AH→ʌ/ə stress-conditioned, vowel length marks AA→ɑː, ER→ɝː, IY→iː, UW→uː) [REPORTED][30]. British accent support deferred unless catalog requires it (misaki `british=True` shows it is a lexicon-level switch [8]).
- Symbols: ~40 phones + 2–4 stress/syllable markers (ˈ ˌ ː) + 2–3 pause classes + word boundary + 2 reserved control symbols ⇒ alphabet ≤ 64 ids.
- Rationale for staying espeak-compatible-ish: lets us (a) reuse espeak CLI output as *dev-time* silver data and eval reference without shipping GPL code [2], (b) compare against Piper/KittenTTS/Kokoro baselines that share IPA-ish spaces [6][38][8], (c) adopt misaki lexicon data where licenses permit (Apache-2.0). Exact symbol identity with any of them is NOT required — models are trained from scratch on our alphabet.
- Locking rule: alphabet frozen via config hash before pilot training (charter reproducibility); changes after pilot = new experiment ID.

## Appendix D — Candidate risk registers

**A (own-chain):** heteronym classifier underperforms (mitigate: E-FRONT-04 early); TN long-tail burns time (mitigate: torture corpus first, ship pass-through-with-warning contract); G2P ONNX quality gap vs espeak (E-FRONT-03 decides); scope creep toward WFST formalism (resist until metrics demand it).
**B (fork misaki):** upstream divergence (pin fork point, document); hidden GPL path re-entering via EspeakFallback (delete module + CI import-linter rule banning `espeak`, `phonemizer`, `espeakng_loader` imports); spaCy latency on profile C/D (E-FRONT-06); py<3.13 port rabbit hole (time-boxed spike E-FRONT-08).
**C (char-level):** pilot-scale pronunciation failure (expected primary outcome — that IS the experiment's value); names/proper nouns worst-hit; TN still mandatory so "no frontend" is a misnomer; if it unexpectedly matches phonemes at smoke scale, treat as signal to re-plan data strategy, not proof at production scale (scale-dependence cuts both ways).

---

## Sources

All fetched/verified 2026-08-22 unless noted. Licenses quoted from LICENSE/COPYING files or repo/PyPI metadata.

1. https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/COPYING — espeak-ng COPYING — GNU General Public License v3, 29 June 2007. [REPRODUCED FROM SOURCE]
2. https://github.com/espeak-ng/espeak-ng/issues/908 — espeak-ng#908: linking exception declined; standalone-CLI/separate-process guidance. [REPORTED]
3. https://github.com/bootphon/phonemizer — phonemizer LICENSE — GPLv3. [REPRODUCED FROM SOURCE]
4. https://github.com/rhasspy/piper-phonemize — piper-phonemize — MIT license (LICENSE.md). [REPRODUCED FROM SOURCE]
5. https://github.com/OHF-Voice/piper1-gpl — Piper successor — GPL-3.0; "embeds espeak-ng for phonemization". [REPRODUCED FROM SOURCE]
6. https://github.com/rhasspy/piper (+ discussions/390) — Piper repo (MIT, archived; moved to OHF-Voice/piper1-gpl); maintainer quote on outsourcing phonemization to espeak-ng IPA mode. [REPRODUCED FROM SOURCE]
7. https://github.com/hexgrad/kokoro — Kokoro — Apache-2.0 code+weights; 82M params; StyleTTS2-lineage; misaki frontend. [REPRODUCED FROM SOURCE]
8. https://github.com/hexgrad/misaki — misaki README — Apache-2.0; English = dictionary + spaCy + num2words; optional EspeakFallback; homograph/seq2seq TODOs; override syntax. [REPRODUCED FROM SOURCE]
9. https://pypi.org/project/misaki/ — misaki 0.9.4 (2025-04-05), Apache Software License, requires-python >=3.8,<3.13. [REPRODUCED FROM SOURCE]
10. https://github.com/hexgrad/kokoro/issues/71 — all-caps spelling bug via spaCy sm model. [REPORTED]
11. https://github.com/hexgrad/kokoro/issues/156 — wrong pronunciations; `[word](/phonemes/)` inline forcing; golds lexicon overrides. [REPORTED]
12. https://github.com/hexgrad/misaki/issues/58 — missing-dictionary words produce no phonemes; espeak fallback broken on Python ≥3.12 via espeakng-loader. [REPORTED]
13. https://github.com/FluidInference/FluidAudio/issues/727 — silent degradation to weaker G2P path; wrong words in fluent audio. [REPORTED]
14. https://github.com/k2-fsa/sherpa-onnx/issues/2004 — simplified G2P in sherpa-onnx loses heteronym handling vs full misaki. [REPORTED]
15. https://github.com/k2-fsa/sherpa-onnx/issues/3731 — sherpa-onnx removing espeak-ng+piper-phonemize in v2.0 due to GPLv3/Apache-2.0 incompatibility. [REPORTED]
16. https://github.com/Kyubyong/g2p — g2p_en — Apache-2.0 (LICENSE.txt). [REPRODUCED FROM SOURCE]
17. https://pypi.org/project/g2p-en/ — g2p-en v2.1.0 (2019-12-31); NumPy inference (TF removed v2.0); deps numpy/nltk/inflect/distance; Mintplex-Labs fork Oct 2025. [REPORTED]
18. https://github.com/as-ideas/DeepPhonemizer — DeepPhonemizer — MIT; PyTorch transformer G2P; TorchScript export; no formal GitHub releases (models ref v0.0.10). [REPRODUCED FROM SOURCE]
19. https://github.com/lingjzhu/CharsiuG2P — CharsiuG2P — MIT; ByT5 multilingual G2P; Interspeech 2022 paper arXiv:2204.03067; byte-level beats mT5 subwords for G2P; last push May 2023. [REPORTED]
20. https://github.com/xinjli/transphone — transphone — MIT. [REPRODUCED FROM SOURCE]
21. https://github.com/rhasspy/gruut — gruut — MIT; archived read-only 2025-10-06; lexicon+CRF G2P without espeak. [REPRODUCED FROM SOURCE]
22. https://github.com/NVIDIA/NeMo-text-processing — NeMo TP — Apache-2.0; WFST TN/ITN on pynini/OpenFst. [REPRODUCED FROM SOURCE]
23. https://pypi.org/project/nemo-text-processing/ — v1.2.0 pins pynini==2.1.6.post1; Linux x86_64 pip officially; conda path for others. [REPORTED]
24. https://pypi.org/project/pynini/ — pynini 2.1.7 (2025-09-04) supports Python 3.9–3.13, manylinux wheels; source build needs OpenFst+C++17. [REPORTED]
25. https://github.com/NVIDIA-NeMo/NeMo/issues/14729 — Python 3.13/3.14 support open issue (Sep 2025). [REPORTED]
26. https://github.com/wenet-e2e/WeTextProcessing — WeTextProcessing — Apache-2.0; FST TN/ITN en/zh/ja; caching, n-best, C++ runtime; pynini/OpenFst substrate. [REPRODUCED FROM SOURCE]
27. https://github.com/SpenserCai/wetext-rs — wetext-rs — Apache-2.0; Rust port of WeTextProcessing on rustfst. [REPORTED]
28. https://github.com/fluidinference/text-processing-rs — text-processing-rs — Apache-2.0; Rust port of NeMo TP. [REPORTED]
29. https://github.com/cmusphinx/cmudict — CMUdict — "(C) 1993–2014 Carnegie Mellon University"; use "for any research or commercial purpose is completely unrestricted"; attribution requested. [REPRODUCED FROM SOURCE]
30. https://github.com/menelik3/cmudict-ipa — full IPA conversion of CMUdict 0.7b + Brown frequency list (33,862 words); conversion rules documented; license not checked. [REPORTED; license UNVERIFIED]
31. https://github.com/lingz/cmudict-ipa — MIT; CMUdict as IPA with splits. [REPORTED]
32. https://raw.githubusercontent.com/savoirfairelinux/num2words/master/COPYING — num2words — GNU Lesser General Public License v2.1. [REPRODUCED FROM SOURCE]
33. https://pypi.org/project/inflect/ — inflect 7.5.0 (2024-12-28) — MIT; number→words, ordinals, a/an. [REPRODUCED FROM SOURCE]
34. *(reserved)*
35. VITS — "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" — arXiv:2106.06103 (ar5iv HTML): "We use International Phonetic Alphabet (IPA) sequences as input to the prior encoder… convert text sequences to IPA phoneme sequences using open-source software." [REPRODUCED FROM SOURCE]
36. F5-TTS — "A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" — arXiv:2410.06885 (HTML v2): character-level input quotes; vocab 2546; filler-token padding; plus repo https://github.com/SWivid/F5-TTS (code MIT; pretrained CC-BY-NC due to Emilia). [REPRODUCED FROM SOURCE]
37. https://github.com/yl4579/StyleTTS2 — StyleTTS2 — MIT code+models with synthesized-audio disclosure condition; README notes inference "depends on a GPL-licensed package" (phonemizer) excluded from repo; gruut alternative noted lower-quality. [REPRODUCED FROM SOURCE]
38. https://github.com/KittenML/KittenTTS (+ Mintlify docs "how-it-works") — KittenTTS — 15M/40M/80M ONNX CPU models; espeak-ng via phonemizer EspeakBackend(en-us, preserve_punctuation=True, with_stress=True) with espeakng_loader. [REPORTED]
39. https://github.com/resemble-ai/chatterbox — Chatterbox — grapheme tokenizers; Turbo uses GPT-2 AutoTokenizer (50,276); vocab file `grapheme_mtl_merged_expanded_v1.json`; punc_norm preprocessing. Repo license not checked this session [UNVERIFIED]. [REPORTED]
40. https://github.com/canopyai/Orpheus-TTS — Orpheus — Llama-3B backbone, native Llama tokenizer, emotive tags as text. Repo license not checked this session [UNVERIFIED]. [REPORTED]

*End of survey.*
