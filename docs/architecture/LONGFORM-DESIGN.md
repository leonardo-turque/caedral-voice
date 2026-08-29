# LONG-FORM GENERATION DESIGN — G1

Task: LONGFORM-01 · Status: DESIGN v0 (pre-pilot) · Date: 2026-08-22
Scope: minutes-scale synthesis (>~1 k chars) on CAEDRAL-NAR (phoneme encoder +
duration/F0/energy predictors + flow decoder + iSTFT head; Tiny 8.35M /
Lite 31.6M / Quality 80.7M). Every number carries an evidence label per charter §3.
Drift findings cited from `docs/research/MULTI-SPEAKER-PROSODY-STYLE.md` §7
(hereafter PROSODY §7); no re-research performed.

---

## 1. Chunking policy

### 1.1 Sentence-boundary splitting

Naive whole-input calls are banned: caller-side chunking is mandatory for this
model class (KittenTTS crashes >~1 k chars without it; Kokoro's native streaming
does not chunk usefully — `docs/benchmarks/BASELINE-MEASURED.md`, lesson 1).

Splitter (pure function of text; deterministic):

1. **Paragraph split** (`\n\n`) — always a chunk boundary, longest pause class.
2. **Sentence split** at `[.!?]+` followed by whitespace, guarded against
   abbreviations (initial list: Mr., Dr., e.g., i.e., etc., digits+`.`, single
   capitals). Sentence text is never merged across a paragraph break.
3. **Greedy packing**: merge consecutive sentences into one chunk while
   `chars ≤ MAX_CHARS = 300` (soft target 150–300). Rationale: literature tunes
   neural TTS at 2–20 s utterance scale [PROSODY §7.1]; 300 chars ≈ 280 phones
   (char→phone ratio 0.77–0.91 prose, MEASURED E0001) ≈ 15–20 s audio at
   ~14 chars/s — inside the tuned band while keeping per-chunk predictor
   context short enough for consistent behavior.
4. **Overflow fallback**: a single sentence > 420 chars splits at clause
   boundaries (`,` `;` `:` = our `|` pause tokens); a clause still > 420 splits
   at word boundaries. No chunk ever exceeds 420 chars.

**Measured on long_001** (this policy, simulated locally): 5687 chars,
969 words, 46 sentences, 10 paragraphs → **28 chunks**, char min/mean/max =
103/202/291. Pre-registered expectation for the eval run: chunk_count ∈ [24, 32]
(policy v0; re-simulate if MAX_CHARS changes).

### 1.2 Prosody-carryover rules at boundaries

Two mechanisms, both post-predictor / pre-decoder transforms (no weight changes;
consistent with PROSODY §8.2 "every knob is a track transform"):

- **Pause insertion**: each boundary appends a silence of duration
  `pause_class(punct, style)` taken from a *document-global* table
  (`comma≈150 ms` class, `period≈450 ms`, `paragraph≈700 ms`; exact values ship
  in voice-pack `prosody.yaml` — PROSODY §7.3 item 4: pauses must derive from
  document-global rules, not per-chunk randomness). Pause audio is appended to
  the **tail** of the earlier chunk so concatenation stays trivial and the next
  chunk begins clean (matters for streaming, §4).
- **F0/energy continuation targets**: after chunk *i* is predicted, capture its
  terminal voiced F0 (median over final ~50 ms of voiced frames) and RMS energy.
  Chunk *i+1*'s initial F0 track is blended toward that carried F0 and its
  initial energy is RMS-aligned (gain clamped ±2 dB) via a linear cross-fade
  over the first ~50 ms (PROSODY §7.3 items 5–6; MagpieTTS-LF halved boundary
  energy discontinuity with history carry-over [PROSODY §7.2]). At an
  unvoiced join, only the energy alignment applies. Carry state is strictly
  forward-flowing and tiny (a few scalars) — see §4.

## 2. Drift prevention, ranked by implementation cost (our arch)

Failure modes being addressed (all [REPORTED] in PROSODY §7.1): energy
discontinuities and warbles at boundaries, inconsistent speaking rates, lost
intonation continuity, speaker/style drift on paragraph-scale synthesis.

| # | Mechanism | Cost | Notes |
|---|---|---|---|
| 1 | **Fixed conditioning per document** (+ fixed decoding seed per document) | **Free** | Speaker⊕style vectors come from the pack lookup anyway (PROSODY §8.1); hoist outside the chunk loop and reuse one RNG state ⇒ identity/timbre drift is impossible by construction, chunk-to-chunk variation collapses (PROSODY §7.3 items 1–2; full determinism precedented by Kokoro's zeroed SineGen phases [REPORTED]). |
| 2 | **Per-document rate normalization** | Low | Realized sec/phone per chunk vs document median → apply the existing global `rate` transform (already implemented in forward API) to hold words/sec flat; directly targets the rate-drift mode [PROSODY §7.1]. One scalar feedback loop, no model change. |
| 3 | **Cross-chunk duration smoothing** | Low–Medium | Smooth per-chunk mean phone duration with an EWMA across the document before length regulation; removes sawtooth rate jitter between adjacent chunks that per-document normalization alone misses. Predictor-side only. |
| 4 | Document-level F0/energy anchors (normalize each chunk to doc-median register/range) | Medium | Extra dry predictor pass (or first-N-chunk stats) + track normalization; prevents inter-chunk register drift [PROSODY §7.3 item 3, HYPOTHESIS → E-J2]. |
| 5 | Predictor-side cross-sentence context (prev-sentence phoneme tail feeds predictors only) | Medium | Cheap MagpieTTS-LF analogue; carries phrase-final pitch movement across boundaries [PROSODY §7.3 item 5, HYPOTHESIS → E-J2]. Supersedes part of §1.2 blending if it works. |
| 6 | Full MagpieTTS-LF encoder-state carry-over + soft monotonic attention prior | High | AR-oriented machinery; out of scope for NAR G1. Recorded as G2 direction [PROSODY §7.2]. |

G1 ships tiers 1–3 unconditionally (they are near-free and address the dominant
drift modes); tiers 4–5 ride on experiment E-J2; tier 6 rejected for G1.

## 3. Evaluation protocol — corpus item `long_001`

Item: `benchmarks/corpus/eval-corpus-v1.0.json`, category `longform` —
5687 chars, 969 words, 46 sentences, 10 paragraphs (MEASURED above).

Setup: trained tiny-config checkpoint through our
frontend→chunker→model→Vocos path; warm run, perf_counter wall time, peak RSS,
per-chunk audio written for scoring. Scoring stack = Phase-C dual-ASR normalized
min-WER (faster-whisper small.en int8 + base.en int8; documented deviation from
Parakeet) per `docs/research/EVALUATION-BENCHMARKS.md` §7.

**Expected wall-time at tiny config** (audio ≈ 390–410 s [ESTIMATED at
2.5 words/s, cross-checked ~13 phones/s]):

| Component | Basis | Estimate |
|---|---|---|
| Frontend (whole item) | MEASURED 13.59 ms (E0001) | ~0.01 s |
| Acoustic fwd, 28 chunks ≈ 5.1–5.4 k phones | MEASURED 134 ms@4thr / 229 ms@1thr per 800-phone pass (CURRENT-STATE-REPORT); linear scaling [ESTIMATED] | 0.9–1.5 s |
| Vocoder | MEASURED decode RTF 0.033–0.048 (`benchmarks/results/decoders/vocos-*`) × ~400 s | 13–20 s |
| Per-chunk glue/concat/IO | 28 × ~25 ms [HYPOTHESIS] | ~0.7 s |
| **Total** | | **≈ 15–22 s ⇒ effective RTF ≈ 0.04–0.06** |

Finding to verify: at tiny config the **vocoder dominates long-form wall time**
(~85%), not the acoustic model — optimization priority for Phase L follows the
measurement, not parameter counts.

Gates (thresholds are pre-registered [HYPOTHESIS], tune once after first run):

- **chunk_count**: 24–32 (§1.1 simulation: 28).
- **wall-time sanity**: total ≤ 60 s at tiny config (≥ 3× margin over estimate);
  record actual minimum per charter §9.5 if missed.
- **WER-stability**: transcribe each chunk independently against that chunk's
  reference text. Per-chunk score `minWER_i = min(WER_ASR1, WER_ASR2)` using the
  dual-ASR stack above. Metric: **population variance
  `Var = (1/N) Σ (minWER_i − mean(minWER_i))²` over all chunks ≤ 0.0025**
  (σ ≤ 5 WER points), AND no chunk with `minWER_i > 0.15` (excluding documented
  ASR-normalization artifact classes, cf. `acr_005` hyphenation finding,
  BASELINE-MEASURED). Rationale: a low document-average WER can hide one or two
  catastrophically degraded chunks; variance catches localized failure, the
  signature of drift/boundary artifacts.
- **Repetition-fraction gate**: for word 4-grams g,
  `rep_fraction(x) = Σ_g max(0, n_x(g) − max(1, n_ref(g))) · 4 / len(words(x))`.
  Gate: `rep_fraction ≤ 0.02` for every chunk AND ≤ 0.01 document-wide, AND
  compression ratio `len(hyp)/len(ref) ≤ 1.5` (existing spec convention,
  EVALUATION-BENCHMARKS §7.3), AND no trailing loop (final tokens repeated ≥ 3
  consecutive times). Any violation = fail regardless of WER.
- Secondary (diagnostic, non-gating): words/sec CV over sliding 30 s windows
  (EVALUATION-BENCHMARKS T3) — quantifies residual rate drift after §2 tier 2/3.

## 4. Interaction with streaming TTFA design

The runtime/streaming design lives in `docs/inference/RUNTIME-DESIGN.md`
(**owned by another agent — referenced here, not edited**). The interface this
doc commits to:

1. **Iterator contract**: the long-form pipeline exposes
   `synthesize_long(text, pack, style, seed) -> Iterator[ChunkResult]`,
   `ChunkResult = {chunk_id, text, phonemes, pause_prefix_ms, carry_state_out,
   mel|wav}`. The runtime may begin synthesizing chunk 0 while later chunks are
   still in frontend/packing, so TTFA ≈ frontend (~14 ms) + first-chunk
   acoustic+vocoder (< ~1 s at tiny: ~190-phone chunk ≈ 55 ms acoustic@1thr +
   ~0.7 s vocoder [ESTIMATED from MEASURED rates]) instead of the ~15–22 s
   full-document time.
2. **Forward-only carry state**: all prosody carryover (§1.2) and smoothing
   (§2) consume only the *previous* chunk's tail scalars + last-K text tokens.
   Nothing in chunk i+1 waits on chunk i+2 ⇒ pipelining never stalls on
   look-ahead. Doc-global anchors (tier 4), when enabled, exempt chunk 0 rather
   than delaying it.
3. **Two-level splitting**: prosody chunks (sentence-granular, this doc,
   quality-driven) are decoupled from the runtime's transport/vocoder
   sub-stream buffering (frame-level, RUNTIME-DESIGN's domain). Transport
   backpressure may throttle pacing but must never change prosody boundaries or
   pause placement; the chunker runs at most one chunk ahead of the consumer.
4. **Constant memory & failure isolation**: only one chunk's tensors + scalar
   carry state are resident (fits profile C/D RAM targets); a per-chunk error
   degrades that chunk and logs, never fails the document — institutionalizing
   the KittenTTS crash lesson (BASELINE-MEASURED lesson 1).

## 5. Open items

- E-J2 must falsify tiers 4–5 before they become default-on (PROSODY §7.3).
- Pause-table values per style land with voice packs (`configs/voices/`);
  defaults here are placeholders until `prosody.yaml` schema freezes.
- Abbreviation guard list grows from corpus misses only (logged commits).
- MAX_CHARS=300 / HARD=420 re-tuned once after first pilot listening pass;
  any change re-runs the §3 protocol and updates the pre-registered bands.
