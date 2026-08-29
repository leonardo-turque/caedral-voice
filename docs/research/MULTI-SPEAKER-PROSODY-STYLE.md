# MULTI-SPEAKER, STYLE & PROSODY CONTROL — Research Dossier

**CAEDRAL VOICE G1 · Researcher H · Track: speaker identity representation + expressive/prosodic control**
**Date:** 2026-08-22 · **Status:** COMPLETE (research pass; all claims labeled per honesty charter §3)

Evidence labels used here: `[REPORTED]` = stated by the cited source; `[ESTIMATED]` =
inference from evidence across sources, not directly stated anywhere; `[HYPOTHESIS]` =
our working assumption, must be falsified in Phases H/J/K; `[UNVERIFIED]` = plausible
but not yet confirmed against a primary source (Phase D must re-check).

Scope reminder (charter): zero-shot cloning of arbitrary people is OUT of G1 scope;
catalog voices come from licensed data; CPU-first inference; modest training budget.
This reframes the whole question: we do NOT need speaker *generalization* machinery —
we need *stability, distinctness, determinism, and control* over a small fixed set of
owned voices.

---

## TL;DR (decisions this dossier supports)

1. **Speaker identity = learnable embedding table** (128–256-d), injected by **AdaIN/FiLM**
   feature-wise modulation at every stage (duration encoder, F0/energy predictors,
   decoder) — the StyleTTS2/Kokoro pattern [REPORTED][1][3][4]. No runtime speaker
   encoder for G1; tables are deterministic, tiny, serializable.
2. **Style = second embedding table** (10 named styles × degree scalar), trained with
   **style-labeled corpora mixed into the main data**, NOT textual style prompts
   (prompt conditioning needs caption pipelines + large models to work well
   [REPORTED][21][22]) and NOT reference clips (defeats catalog determinism).
3. **Prosody knobs come free** from FastSpeech2-class variance adapters: rate scaling,
   F0 shift/scale, energy gain, pause insertion are inference-time transforms on
   predicted duration/F0/energy tracks [REPORTED][17][19].
4. **Emphasis**: start with the zero-data trick (boost duration+F0 on marked words;
   +40% listener identification, +7.3% naturalness [REPORTED][49]), upgrade to
   markup-tag conditioning if Phase H bake-off favors it [REPORTED][45][46][47].
5. **Few-speaker risk is real but bounded**: ~85–100 speakers is the literature floor
   for *generalizing* multi-speaker TTS [REPORTED][41][43]; a closed catalog does not
   need generalization — it needs per-voice data volume (~1.5–4 h/voice) and acoustic
   contrast between voices [HYPOTHESIS → E-J1].
6. Long-form consistency: reuse identical conditioning vectors per chunk + fixed seeds +
   document-level prosody anchors; optionally history-state stitching à la
   MagpieTTS-LF [REPORTED][55].

---

## 1. Speaker identity representations

### 1.1 Taxonomy

Three mechanisms recur across modern systems. Note that (c) is an *injection*
mechanism while (a)/(b) are *sources* of the identity vector; production designs pair a
source with an injection path.

**(a) Learned embedding table (per-speaker ID vector)**

- A `nn.Embedding(n_speakers, d)` lookup, trainable end-to-end. Deep Voice 2
  established trainable speaker embeddings for multi-speaker neural TTS [REPORTED][13].
- **Matcha-TTS** uses exactly this: `torch.nn.Embedding(n_spks, spk_emb_dim)` with
  default `spk_emb_dim=128`, added to the encoder output and concatenated into the
  CFM decoder input channels (`in_channels + spk_emb_dim` when `n_spks > 1`) [REPORTED][7][8].
- Pros: zero runtime encoder cost; fully deterministic; trivially serializable into a
  voice pack (one tensor + metadata); identity cannot "wander" at inference because it
  is not computed from audio.
- Cons: cannot represent a new voice without retraining or post-hoc insertion into a
  space never trained to accept it; prone to overfitting when n_speakers is small
  (§5); embeddings are only comparable *within* one checkpoint (versioning hazard).

**(b) Speaker/prosody encoders over reference audio (d-vector/x-vector class)**

- The classic d-vector: L2-normalized output of an LSTM trained with GE2E loss over
  N-speakers×M-utterances batches; became the standard conditioner for multi-speaker
  Tacotron-era TTS and the CorentinJ-style cloning pipeline [REPORTED][14][15].
- YourTTS conditions on external speaker embeddings to get zero-shot capability
  [REPORTED][16]. StyleTTS2's multi-speaker mode computes `c = E(x_ref)` from reference
  audio with dual acoustic+prosodic style encoders and injects it via AdaLN
  [REPORTED][1].
- Pros: generalizes to unseen speakers; enables clustering/averaging; supports
  "voice similarity" tooling.
- Cons: needs a large and diverse speaker corpus to train the encoder itself (the
  failure mode behind weak cloning on small-data models); adds a network pass at
  inference; outputs are *not* constrained to the manifold the synthesizer was trained
  on. Concrete evidence of the last point: in Kokoro, extracting style vectors from raw
  wavs with a StyleTTS2-style encoder produces poor voices due to embedding-space
  mismatch; the community workaround is blending 30% custom / 70% built-in to pull the
  vector back toward known-good space [REPORTED][61].

**(c) Feature-wise conditional modulation: FiLM / AdaIN**

- FiLM: per-channel affine transform `γ_c·F_c + β_c` generated from a conditioning
  vector; cheap (two scalars per channel), general, robust [REPORTED][63].
- AdaIN: the normalization-based variant — align channel-wise mean/variance of
  activations to statistics predicted from the style input, `(1+γ)·norm(x)+β`
  [REPORTED][64][4].
- This is the dominant injection path in the systems closest to CAEDRAL's target:
  - **StyleTTS2**: AdaIN-based style injection throughout predictors and decoder; the
    diffusion-sampled style vector `s = [s_p, s_a]` (prosodic ⊕ acoustic halves)
    modulates the whole synthesis stack; in the multi-speaker setting the speaker
    embedding enters via adaptive layer norm [REPORTED][1].
  - **Kokoro-82M** inherits it wholesale: style vector concatenated onto every LSTM
    timestep + AdaLayerNorm in the duration encoder; separate F0 and energy predictor
    chains built from AdaIN residual blocks; ISTFTNet decoder built from
    AdaIN-Snake residual blocks where `h = fc(s); γ,β = chunk(h,2);
    out = (1+γ)·norm(x)+β` [REPORTED][4][5][6].
- Pros: strongest empirical association with small-model expressivity (an 82M model
  topping human-vote arenas was trained on <100 h using exactly this conditioning
  surface [REPORTED][3]); near-zero parameter overhead; smooth interpolation behavior
  (§6).
- Cons: none meaningful for G1; it is simply how the vector gets in.

### 1.2 What each flagship system actually does (conditioning audit)

| System | Identity source | Injection | Notes |
|---|---|---|---|
| StyleTTS2 (multi-spk) | ref-audio style encoders E_a/E_p → s=[s_p,s_a]; diffusion samples p(s\|t) at inference | AdaLN everywhere | 3-step EDM solver ≈ 9-layer transformer forward for style sampling [REPORTED][1] |
| VITS2 | learned speaker vector | added to text encoder block 3 + flows | speaker-conditioned text encoder raised similarity MOS 3.79→3.99 [REPORTED][10] |
| Kokoro-82M | precomputed 256-d style vector per voice pack, indexed by phoneme length bucket (`pack[len-1]`) | concat+AdaLayerNorm (duration), AdainResBlk (F0/N), AdaIN-Snake (decoder) | fully deterministic synthesis; SineGen phases/noise zeroed [REPORTED][3][4][5][6] |
| Matcha-TTS | `nn.Embedding(n_spks,128)` | add to encoder; concat into CFM decoder input | simplest possible table design [REPORTED][7][8] |
| F5-TTS | none — identity carried by in-context prompt audio+transcript | ConvNeXt-refined text tokens concatenated with noisy latents; adaLN-zero for flow step | zero-shot cloning paradigm; irrelevant without reference clips [REPORTED][9] |

Two structural lessons:

1. **F5/Voicebox-class models get identity "for free" only by consuming reference
   audio.** Since CAEDRAL excludes cloning, our architecture family (H1/H4 in MASTER-PLAN
   §10) must carry identity explicitly — table or encoder. This is a genuine advantage of
   the FastSpeech2/StyleTTS2 lineage for us, not a limitation.
2. **Kokoro is existence proof of the entire G1 concept**: 82M params, <100 h of
   permissive audio, Apache 2.0, arena-competitive quality, catalog of distinct voices
   delivered as standalone style-vector packs, blended voices created by averaging
   packs (default voice `af.pt` = verified 50/50 mean of Bella and Sarah vectors)
   [REPORTED][3][60]. Its known gaps are equally instructive: no cloning ability
   (small data), external g2p dependency (espeak-ng licensing risk — see track B),
   and narration-heavy training data (weak conversational register) [REPORTED][3].

### 1.3 Table vs encoder for a fixed catalog — verdict

For a closed catalog, embedding tables dominate on every axis that matters:
determinism (same input → bit-identical output given same seed), cost (a lookup vs a
network pass), pack portability (a .npy/.pt file + metadata), and auditability (an
identity is a hashable artifact). The encoder's one advantage — new speakers without
retraining — buys nothing in G1 because new catalog voices will be recorded under
license anyway, i.e. we will retrain with more data regardless [ESTIMATED]. The
future-extensibility concern is addressed in §6 (pack format keeps room for an optional
encoder path in G2 without changing the G1 API).

A middle path worth keeping in the back pocket: train the table, then *freeze* it and
define new voices as convex combinations of existing rows (Kokoro blending works in
practice, including SLERP procedural voices [REPORTED][60][62]). Cheap "derived" voices
for testing catalog breadth without new recordings [HYPOTHESIS: derived blends are
perceptually weaker than recorded voices — test in E-J1].

---

## 2. Disentangling content / speaker / prosody

### 2.1 The classical toolkit

- **Reference encoder + GST (Global Style Tokens)**: a conv-stack+GRU reference encoder
  compresses prosody of variable-length audio into a query; multi-head attention over a
  bank of randomly initialized shared tokens produces a style embedding that conditions
  the text encoder every timestep. Trained with no style labels at all. Three useful
  findings survive a decade of follow-up work: (i) token-bank conditioning generalizes
  far better than direct reference-embedding conditioning (which fails on short
  references by squeezing synthesis into the reference's time interval); (ii) tokens
  trained on noisy unlabeled data learn to factorize noise and speaker identity;
  (iii) individual tokens become semi-interpretable (length, speed, pitch styles) and
  can be conditioned on directly at inference — style control without any reference
  clip [REPORTED][12].
- **VITS variational bottleneck**: the posterior encoder + stochastic duration
  predictor give the model latent freedom for one-to-many mapping (prosody diversity),
  but provide no handle to *control* which sample you get beyond temperature — and the
  latent entangles speaker and prosody unless extra structure is added [REPORTED][11].
- **VITS2** shows the speaker vector belongs *early*: conditioning the text encoder
  lets pronunciation/intonation become speaker-specific during encoding, improving
  similarity MOS by +0.2 [REPORTED][10].

### 2.2 Known failure modes (and their fixes)

1. **Speaker leakage into prosody.** In reference-encoder multi-speaker systems, the
   "prosody" code secretly carries speaker timbre; cross-speaker style transfer then
   fails or drifts the target identity. Fix demonstrated at Interspeech 2021: replace
   the deterministic reference encoder with a learned latent distribution (variational
   bottleneck) *and* condition on flow-normalized speaker embeddings so the prosody
   path stops being the shortest route for speaker info [REPORTED][33]. Microsoft's
   prosody-bottleneck work identifies the two root causes as (i) single-reference
   embeddings lacking fine-grained prosody and (ii) content/prosody/timbre entanglement,
   and beats cycle-consistency and GMVAE baselines with an explicit bottleneck
   [REPORTED][32].
2. **Content leakage into style codes.** Prosody/style embeddings that encode phonetic
   content degrade intelligibility when transferred. The "information sieve"
   downsample–upsample filter + instance normalization reduces it, beating GST Tacotron
   baselines on WER [REPORTED][35].
3. **Timbre/pitch/rhythm entanglement is fundamental, not accidental.** AutoPST's
   information-theoretic analysis explains why timbre disentanglement succeeds wherever
   a label exists (speaker ID lets a bottleneck "subtract" timbre) while pitch/rhythm
   stay entangled absent analogous labels [REPORTED][34]. Practical corollary: our
   *style labels are exactly such labels* — supervising style with named categories is
   what makes disentanglement tractable at small scale [ESTIMATED].
4. **Timbre collapse / identity blending with few speakers.** With few speakers the
   embedding table and the decoder co-adapt until identities share most of their
   realization; Lai et al. observed Tacotron overfitting to seen VCTK speakers'
   characteristics/dialects and hypothesized the speaker count itself was too small
   [REPORTED][41]. Mitigations with evidence: VTLP speaker augmentation to inflate the
   population [REPORTED][41]; low-dimensional speaker embeddings fine-tuned alone
   instead of all parameters (far less prone to adaptation overfitting)
   [REPORTED][44]; pretrained (frozen) speaker encoders instead of learned tables for
   unseen-speaker coverage [REPORTED][42].
5. **Language/pronunciation leaking through style** in cross-lingual settings — handled
   with HuBERT bottlenecks + mutual-information minimization (GenerTTS) [REPORTED][36];
   English-only G1 mostly dodges this.

### 2.3 What disentanglement CAEDRAL actually needs [ESTIMATED]

G1 does not need research-grade full factorization. It needs three weaker properties,
each independently measurable:

- **P1 (style transferability)**: applying style S to voice V changes expressive
  character without shifting perceived identity. Measurable with a speaker classifier
  confusion matrix + ABX (E-J3).
- **P2 (speaker stability under knobs)**: rate/pitch/energy knob sweeps leave the
  speaker classifier confident (FastPitch reports uniform ±50 Hz F0 shift preserves
  perceived identity [REPORTED][19]; larger shifts will not — clamp ranges accordingly).
- **P3 (no content damage from style conditioning)**: WER on style-conditioned output
  ≈ neutral-conditioned WER.

Full unsupervised disentanglement (GST-style token discovery, variational prosody
bottlenecks) is a Phase K+ investigation, not a G1 dependency — supervised style labels
substitute for most of its function at our scale [HYPOTHESIS → E-H1/E-J3].

---

## 3. Style/emotion control WITHOUT a reference clip

Four families exist. G1 wants family (B) as primary; (C) as a stretch; (A) as
fallback; (D) as the numeric substrate under all of them.

### 3.A Global style tokens (unsupervised)

GST tokens can be addressed directly at inference — style control with zero reference
audio and zero labels [REPORTED][12]. Attractive for us in theory (no annotation cost),
but token semantics are emergent: you get *some* controllable dimensions (speed,
pitch register), not a guaranteed "warm" or "professional" token. Token semantics also
shift between training runs, which is hostile to a versioned catalog [ESTIMATED].
Verdict: keep GST as a Phase K exploration for discovering extra styles beyond the
named ten; do not build the G1 API on it.

### 3.B Style-conditioned training on labeled corpora

Train with a style embedding (a second small table, or a projection of a label one-hot)
supervised by per-utterance style labels. This is the EmotiVoice pattern inverted:
EmotiVoice (JETS-based, CPU-relevant) feeds a BERT-pooled *text prompt* embedding as
`inputs_style_embedding` alongside speaker and content embeddings [REPORTED][30] — but
the same conditioning slot accepts a label-derived vector just as happily, and a label
table is far cheaper and more deterministic than a BERT pass [ESTIMATED].

What the labeled corpora actually contain (licensing column is decisive for Phase D):

| Corpus | Speakers | Styles | Size | Lexical variety | License status |
|---|---|---|---|---|---|
| ESD [26] | 10 EN + 10 ZH (balanced M/F) | neutral, happy, angry, sad, surprise | ~29 h, 350 parallel utts/speaker/emotion (~35k clips) | high (parallel across emotions) | research release — **verify commercial terms in Phase D** |
| RAVDESS [27] | 24 actors (12M/12F) | 8 emotions × 2 intensities, speech+sung | small; **only 2 lexical sentences** | minimal | CC BY-NC-SA [UNVERIFIED] |
| CREMA-D [28] | 91 actors | 6 emotions × 4 intensity levels | 7,442 clips, 12 sentences, 2,443-rater annotations | minimal | open license [UNVERIFIED] |
| Expresso [29] | 4 NA-English actors | 7 parallel read styles + 25 improvised styles (calm, sad, whisper, laughing, sarcastic, sleepy, …) | 40–47 h, studio 48 kHz | high; whisper ~2.8 h, sad ~3.0 h, laughing ~3.3 h | **CC BY-NC 4.0 — non-commercial, excluded unless negotiated** |

Design facts that matter:

- ESD was built *because* RAVDESS/CREMA-D-class corpora have too little lexical
  variability for TTS training [REPORTED][26]; its parallel utterances across emotions
  are what make cross-speaker style transfer learnable [REPORTED][26][ESTIMATED].
- Expresso is the only corpus whose style taxonomy overlaps CAEDRAL's ten styles
  substantially (calm, sad/serious, whisper/soft, laughing/excited, conversational) —
  and it is NC-licensed. This is the single biggest style-data gap for Phase D
  [ESTIMATED].
- Actor corpora skew theatrical; "professional" and "conversational" registers are
  underrepresented everywhere [REPORTED][3] (Kokoro's own noted weakness: mostly
  long-form narration, not conversation).

Practical recipe supported by evidence: mix style-labeled data into multi-speaker
training with the style vector entering the same AdaIN slot as the speaker vector
(concatenated, as StyleTTS2 concatenates prosodic+acoustic style [REPORTED][1]); balance
styles × speakers so no (style, speaker) cell is empty, otherwise the model learns
speaker-correlated style priors — the leakage failure of §2.2 [HYPOTHESIS → E-H1].
Precomputing averaged per-(voice,style) vectors instead of diffusion-sampling them at
inference is exactly what low-resource StyleTTS2 work and Kokoro both do for stability
[REPORTED][31][3].

### 3.C Textual style prompts

- **Parler-TTS** (Lyth & King, Stability/Edinburgh): natural-language descriptions
  replace reference clips; annotations generated by a pipeline (dataspeech) that
  measures speaking rate, SI-SDR, reverberation, speech monotony (+pitch), bins them,
  and has an LLM word them naturally [REPORTED][21][23]. Trained at 45k hours → Mini
  880M / Large 2.3B AR models [REPORTED][22]. Works, but note the price of admission:
  a captioning pipeline, LLM paraphrasing, and 45k hours; controllable granularity is
  coarse; and as an AR decoder the description tokens consume context. This is a
  large-model recipe, not a small-CPU one [ESTIMATED].
- **Orpheus** (Llama-3B speech LM): 8 inline emotive tags (`<laugh>`, `<sigh>`,
  `<gasp>`…) plus free-text emotion descriptors; generation quality depends on
  LLM-style sampling params — `repetition_penalty ≥ 1.1` is *required* for stable
  output, temperature trades expressiveness vs speed [REPORTED][24]. Evidence that
  prompt-style control in AR speech LMs is real but entangled with sampler state —
  nondeterministic by default, hostile to catalog determinism [ESTIMATED].
- **Chatterbox** (Resemble, 0.5B): exposes `exaggeration` (an `emotion_adv` scalar
  0–1 folded into conditionals) and `cfg_weight` (pacing/text-adherence); official
  guidance: raise exaggeration AND lower cfg_weight for dramatic speech because
  exaggeration speeds speech up; capitalization shifts emphasis [REPORTED][25].
  Honest signal here: even the flagship "emotion control" is one scalar + a CFG knob,
  not a rich semantic space.

Verdict for G1 [ESTIMATED]: textual prompts are the right *interface* only if the model
is an AR LM (family H3); for H1/H4 NAR models they add a text encoder and a captioning
data dependency for little gain over label tables. We can still expose prompt-like
*names* in the API ("style='warm'") — the string maps to a table row; users get the UX
of prompts at none of the cost.

### 3.D Continuous dimensional control (arousal–valence–dominance)

- EmoSphere-TTS: spherical emotion vector (angle = style, radius = intensity) with
  continuous A/V/D pseudo-labels from a wav2vec 2.0 SER model [REPORTED][38].
- UDDETTS: unifies discrete labels with A-D-V tokens and demonstrates *linear* control
  along all three dimensions [REPORTED][39].
- Laugh Now Cry Later (flow-matching TTS): per-phoneme time-varying A/V conditioning
  works; notably **adding the dominance dimension degraded audio quality** [REPORTED][40].
- Emotion arithmetic via weight-space interpolation of emotion vectors (Interspeech
  2024) supports continuous blending of labeled emotions [REPORTED][37].

Implication: if we want continuous style intensity, arousal+valence (2-D) is the
evidence-backed core; dominance is optional and possibly harmful [REPORTED][40]. A
`style_degree` scalar (cf. Azure's `styledegree` 0.01–2.0 with fallback-to-neutral on
invalid values [REPORTED][51]) is the production-proven 1-D compression of this idea
[ESTIMATED]. SER pseudo-labeling of our own recordings with an A/V regressor is a cheap
way to get continuous supervision alongside the 10 labels [ESTIMATED].

---

## 4. Explicit prosody control — the deterministic knobs

### 4.1 What FastSpeech2-class variance adapters give "for free"

The variance adaptor sits between the phoneme encoder and decoder: a duration predictor
(log-domain phoneme lengths from MFA alignments), a pitch predictor (frame-level F0),
and an energy predictor (per-frame energy); each a small 2-layer 1-D conv stack. Values
are extracted from waveforms during training and *predicted* at inference — which makes
them interceptable: any inference-time transform of the predicted tracks is a control
knob requiring no retraining [REPORTED][17][18].

| Knob | Mechanism | Evidence |
|---|---|---|
| rate | scale durations by α before the length regulator (α<1 fast, α>1 slow) | original FastSpeech length regulator [REPORTED][17] |
| pitch shift | add constant to predicted F0 contour | FastPitch: uniform ±50 Hz shift preserves perceived speaker identity, "voluntary vocal chord modulation" [REPORTED][19][20] |
| pitch range | multiply contour deviations around utterance median | standard practice; ESPnet exposes `pitch_transform` [REPORTED][18] |
| energy | add gain (dB) to predicted energy track | FastSpeech2 energy conditioning [REPORTED][17] |
| pauses | insert silence spans at punctuation/tagged positions (SSML `<break>` is the production interface; fully supported by Polly across engines) [REPORTED][50] |
| emphasis | see §4.3 | |

ESPnet's implementation confirms the API shape: inference params for speed `alpha`,
plus optional teacher-forced duration/pitch/energy — i.e., the knobs are standard
practice, not exotic [REPORTED][18].

### 4.2 What breaks when knobs are pushed [REPORTED unless noted]

1. **Uniform duration scaling is the wrong model of human rate change.** Real speech
   compresses/stretches phonemes *differently* by class. Measured asymmetries:
   slowing is much worse than speeding — at 0.5× rate, conventional duration control
   produced 57.3% CER vs 10.6% for phoneme-aware Gaussian duration modeling; duration
   *expansion* errors are ~6× larger than contraction errors [REPORTED][52]. SCTTS
   (Interspeech 2020) independently reports unnaturalness is more noticeable for slow
   speech, and that learned non-uniform lengthening (which phonemes/words/pauses to
   stretch) beats uniform scaling [REPORTED][53]. Attention-based rate models reach
   MOS 4.35 vs 3.98 for FastSpeech baseline speed control [REPORTED][54].
   → G1 rule: clamp rate to roughly 0.75–1.5 for the default path [ESTIMATED]; treat
   extreme rates as a Phase K feature requiring learned duration re-prediction
   conditioned on target rate, not α scaling [HYPOTHESIS → E-H4].
2. **Rounding quantization** in the length regulator introduces small artifacts
   (durations × α then rounded [REPORTED][17]) — negligible alone, visible combined
   with extreme α [ESTIMATED].
3. **Pitch shifts are safe only within identity-preserving range.** ±50 Hz uniform
   shift preserves identity [REPORTED][19]; large shifts change perceived gender/age
   and desynchronize F0 from formants (the decoder was never trained on such
   (F0, timbre) pairs) [ESTIMATED]. → clamp pitch_shift to ±3 semitones default
   [ESTIMATED → E-H4 measures the real cliff].
4. **Energy gain** interacts with vocoder/decoder saturation; large positive gains
   clip or distort [ESTIMATED — verify empirically in E-H4]. Keep gain in ±6 dB.
5. **Knob interactions are non-additive**: Chatterbox's own docs — higher exaggeration
   speeds speech up, so you must lower cfg_weight to compensate [REPORTED][25].
   Our equivalent: raising pitch_range raises perceived energy; rate changes shift
   optimal pause lengths. The API should expose clamps and document interactions
   rather than pretend orthogonality [ESTIMATED].

### 4.3 Emphasis — four evidence-ranked options

1. **Zero-data duration boost (SSW 2023)**: increase predicted phoneme duration of the
   emphasized word (no retraining, no recordings). +7.3% naturalness over spectrogram
   modification and +40% correct-emphasis identification, validated in 4 languages;
   benchmarked against FLAG-EMPH, an upper bound using explicit emphasis recordings
   [REPORTED][49]. → **G1 default.**
2. **Markup tags in training text** (Interspeech 2024): prosodic markup fed alongside
   text; a RoBERTa-based predictor learns to place/choose labels, and threshold tuning
   controls expressiveness [REPORTED][45]. Related: unsupervised tag discovery via
   decision-tree + GMM clustering of prosody embeddings beat FastSpeech2 in
   naturalness while enabling word-level control [REPORTED][46]. → Phase K upgrade if
   E-H3 shows the duration boost is insufficient.
3. **Auto-annotated emphasis corpora** (SSW 2025): annotate training data with emphasis
   scores, train the acoustic model to consume them [REPORTED][47]. Pairs well with
   Helsinki's CWT-based prominence/boundary labels over LibriTTS (~160k sentences,
   ~2.8M words, 3-level labels + continuous scores) [REPORTED][59] — a ready-made
   supervision source for Phase D.
4. **Latent emphasis features** (Interspeech 2022): hierarchical parallel TTS with a
   latent space directly corresponding to emphasis; at inference, raise the latent
   value on target words [REPORTED][48]. Elegant, but requires the special
   architecture — skip for G1 [ESTIMATED].

---

## 5. Multi-speaker training on FEW speakers — catalog strategy

### 5.1 What the evidence says about speaker counts

- **~85 speakers can cover much of a population** — for *voice fitting with a priming
  encoder*, Nachmani et al. found a small population (~85 VCTK speakers) sufficient to
  capture much of general-population variation [REPORTED][43].
- **~100 speakers is NOT enough for generalizing multi-speaker TTS**: Lai et al.'s
  Tacotron on VCTK overfit to seen speakers; unseen voices/dialects were poorly
  reproduced; they hypothesized the speaker count itself was too small and added VTLP
  speaker augmentation + non-TTS-quality data with factorized embeddings [REPORTED][41].
- **Tables do not generalize, encoders do** — a systematic comparison of pretrained vs
  learnable speaker representations: lookup tables are trainable but cannot represent
  unseen speakers; GST-style conditioning can capture identity without a fixed table
  [REPORTED][42].
- **Adaptation overfitting is parameter-count-dependent**: fine-tuning only a
  low-dimensional speaker embedding vastly outperforms fine-tuning all parameters when
  adding speakers (meta-learning WaveNet TTS) [REPORTED][44].

Reconciliation [ESTIMATED]: the "85 vs 100+" tension dissolves once you separate the
two goals. *Generalization to unseen speakers* needs ≥~100 speakers + augmentation.
*A fixed catalog of K distinct voices* needs, per voice: enough hours for stable
decoder conditioning and enough *acoustic contrast* between voices. The catalog goal is
strictly easier and is what G1 pursues.

### 5.2 Hours per voice — evidence points

- Kokoro: <100 h total across its whole voice set, <20 epochs, ~$400 of A100 time —
  produced arena-topping distinct voices [REPORTED][3].
- StyleTTS2 fine-tuning recipe: ~1 h (≈1k clips) suffices to adapt a LibriTTS
  checkpoint to a single new voice in ~4 h on 4×A100 [REPORTED][2].
- ESD: ~1.5 h/speaker (350 utts × 5 emotions) produced learnable speaker×emotion
  structure [REPORTED][26].
- → Planning number for G1: **2–4 h per catalog voice for full quality, ≥1 h for a
  pilot-acceptable voice**, with style coverage inside each voice's script (each of the
  10 styles represented by ≥15–20 min of material) [ESTIMATED]. Pilot-scale reality
  check: CPU training means G1 pilots will use fewer hours and must be judged as
  dynamics/ablation evidence, not final quality (charter §7 SMOKE/PILOT discipline).

### 5.3 Identity blending and other few-speaker risks

- **Blending**: with few acoustically-similar voices, embeddings co-adapt and outputs
  converge toward a "mean voice" [HYPOTHESIS, motivated by §2.2-4 evidence]. Mitigations:
  (i) cast voices for maximal contrast (gender, F0 register, accent, age, phonation)
  [ESTIMATED]; (ii) add a speaker-classification auxiliary loss on the embedding so the
  table is forced to be linearly separable [HYPOTHESIS → E-H1]; (iii) VTLP-augment each
  voice to widen its realized acoustic range without new identities [REPORTED][41].
- **Style×speaker imbalance** creates speaker-correlated style priors (only the
  cheerful voice ever gets cheerful data) — the catalog then cannot do cross-style
  [HYPOTHESIS; prevention = balanced recording scripts, §3.B].
- **Adding voices later**: an embedding table cannot represent a new voice without
  touching the model [REPORTED][42]. Options ranked: (a) full retrain with replay of
  old voices' data (safest for identity preservation) [HYPOTHESIS]; (b) freeze
  everything, train only the new embedding row + light LoRA on predictors [HYPOTHESIS,
  motivated by the low-dim-embedding result [44]]; (c) convex blend of existing rows
  (no training, weakest identity) [REPORTED][60][62]. Whether (b) degrades old voices
  is a Phase K falsification target (E-K1).

---

## 6. Voice-catalog engineering

### 6.1 How production systems define voices

- **Azure**: voices are named endpoints; expressive behavior is a per-sentence SSML
  overlay `mstts:express-as` with `style` (voice-specific: cheerful, empathetic, calm,
  newscast…), `styledegree` (0.01–2.0, default 1), and `role` (age/gender role-play).
  Invalid style values **fall back to neutral** — a safety pattern worth copying
  [REPORTED][51]. Styles are per-sentence, and only a subset of neural voices supports
  them (discoverable via list-voices API) [REPORTED][51].
- **Amazon Polly**: SSML support is engine-dependent — neural/long-form/generative
  voices support `rate`/`volume` but **not `pitch`**, and **not `<emphasis>`** at all;
  generative voices restrict `<prosody>` to full sentences [REPORTED][50]. Cautionary
  tale: production systems ship only the knobs that survive quality testing; exposing
  a knob that produces artifacts is worse than not exposing it [ESTIMATED].

### 6.2 Proposed CAEDRAL voice-pack format (G1)

A voice = **directory of immutable, hashed artifacts** (Kokoro's standalone pack files
are the precedent [REPORTED][3][60]):

```
voices/<voice_id>/
  voice.yaml          # id, name, gender, locale, accent, description, version,
                      #   base_model (git hash), dataset_manifest (hash), license_ref,
                      #   created, checksum-of-everything-below
  speaker.npy         # float32 [d_speaker] — the embedding-table row (d=128 or 256)
  styles.npy          # float32 [n_styles, d_style] — per-style style vectors
                      #   (rows: the 10 G1 styles; neutral = row 0)
  style_defaults.yaml # per-style: degree default, allowed degree range
  prosody.yaml        # clamps: rate [0.75,1.5], pitch_shift_semitones [-3,3],
                      #   pitch_scale [0.8,1.25], energy_gain_db [-6,6],
                      #   pause defaults (comma/period/paragraph ms)
  golden/             # fixed text list + reference WAVs + hashes for A/B regression
```

Design rules, each with a reason:

1. **Immutable + content-hashed** — the best-known pack is never overwritten in place
   (mirrors charter §8 checkpoint rule); A/B safety = hash comparison + golden-sample
   regression before any pack ships [ESTIMATED].
2. **Versioned against a base-model hash** — speaker/style vectors are meaningless
   across checkpoints (trained co-adapted with the decoder); a pack is only valid for
   the model hash it names [REPORTED — the Kokoro issue-#314 space-mismatch failure is
   the same lesson][61].
3. **Prosody clamps live in the pack, not the code** — different voices tolerate
   different ranges (a whisper voice must not take +6 dB); per-voice clamps encode
   measured safety margins from E-H4 sweeps [ESTIMATED].
4. **Derived voices are first-class**: `blend: {of: [a, b], weights: [0.3, 0.7]}` —
   mean/SLERP blends of style vectors demonstrably produce coherent new voices
   (Kokoro's default voice is itself a verified 50/50 mean [REPORTED][3][60][62]).
   Derived packs record their parents for provenance.
5. **Room for G2**: optional `speaker_encoder.onnx` slot so future reference-audio
   conditioning can be added without breaking the G1 API (table path remains the
   default) [ESTIMATED].

### 6.3 A/B safety workflow

1. Golden set: ~50 fixed texts (short/medium/long, all 10 styles, edge cases: numbers,
   quotes, emphasis marks, max-rate text) per voice.
2. Any pack change → regenerate goldens → automated checks (duration sanity, clipping
   rate, WER vs text, speaker-classifier self-match ≥ threshold) + human pairwise
   MOS-lite vs previous pack version [ESTIMATED].
3. Ship = git tag of pack hash; rollback = previous tag. No in-place mutation
   [ESTIMATED].

---

## 7. Long-form consistency (minutes-scale)

### 7.1 Why chunked synthesis drifts

Neural TTS quality is tuned at 2–20 s utterance scale; naive sentence-chunk-and-concat
long-form output shows prosodic drift, speaker inconsistency, and boundary artifacts —
energy discontinuities, "warbles" at boundaries, inconsistent speaking rates, and lost
intonation shifts across sentences [REPORTED][55]. Paragraph-level synthesis without
context also loses style/timbre/rate consistency and expressiveness [REPORTED][56].
Listeners report fatigue on long synthetic speech, which motivated cross-sentence
prosody modeling (HiGNN-TTS: word/sentence/cross-sentence graph hierarchy with context
aggregation from previous/next sentences; improves both ~5 s and ~40 s naturalness)
[REPORTED][57].

### 7.2 What long-form-capable systems do

Three paradigms in the 2024–2026 literature [REPORTED][55]:

1. **Sequence compression** — e.g., VibeVoice's 7.5 Hz token rate fits up to ~90 min in
   one pass, sacrificing temporal resolution. Not available to a small CPU model.
2. **Streaming/block-wise methods** — bounded-memory attention over blocks (CosyVoice 2
   uses block-wise attention with hard information cutoffs); baked into training.
3. **Inference-time stateful stitching** — **MagpieTTS-LF** is the cleanest template:
   no retraining at all. It maintains (i) history text tokens (last K text tokens of
   the previous chunk, for discourse-level prosodic planning), (ii) history encoder
   states carried across boundaries, and (iii) attention tracking that seeds a *soft*
   monotonic attention prior for the next chunk (softmax with λ·log-prior added to
   logits, keeping ε-mass on distant tokens instead of hard masking). Result on
   3–4-minute passages: boundary energy discontinuity halved vs competitors, no
   measurable speaker-similarity drift over minutes, most stable UTMOSv2 [REPORTED][55].

### 7.3 Recipe for an NAR FastSpeech2-class CAEDRAL model [ESTIMATED unless noted]

Our architecture family makes drift easier to fight than AR systems', because most of
what varies between chunks *should* be constant:

1. **Conditioning reuse**: identical speaker + style vectors for every chunk of a
   document (they come from the pack anyway — no per-chunk encoder noise).
   Deterministic by construction [ESTIMATED].
2. **Fixed decoding seed per document**: predictor dropout/sampling states identical
   across chunks so run-to-run and chunk-to-chunk variation collapses [ESTIMATED;
   Kokoro's zeroed SineGen phases show full determinism is achievable in this family
   [REPORTED][6]].
3. **Document-level prosody anchors**: compute F0 median/range once per document
   (from the first N chunks or a dry pass of all predictors) and normalize each chunk's
   predicted F0/energy to those anchors — prevents inter-chunk register drift
   [HYPOTHESIS → E-J2].
4. **Consistent pause policy**: pause durations derived from document-global rules
   (punctuation class × style default), not per-chunk randomness [ESTIMATED].
5. **Cross-sentence continuity**: cheap version of MagpieTTS-LF's idea — feed the
   previous sentence as side-context to the *predictors only* (not the decoder), so
   phrase-final pitch movement carries across boundaries without any architectural
   change to the acoustic path [HYPOTHESIS → E-J2; if it fails, ship plain chunking +
   anchors, which already addresses identity/energy drift].
6. Boundary smoothing: cross-fade predictor outputs (F0/energy) over ~50 ms across
   chunk joins before decoding [ESTIMATED].

---

## 8. RECOMMENDED G1 SPEAKER/STYLE/PROSODY ARCHITECTURE

### 8.1 Representation

| Component | Choice | Basis |
|---|---|---|
| Speaker identity | learnable embedding table, **d=128** (Matcha default [7]; sweep {64,128,256} in E-H1) | deterministic, zero runtime cost, packable [§1.3] |
| Style | second embedding table, **d=64–128**, rows = 10 G1 styles + neutral; `style_degree` scalar interpolates row↔neutral (Azure styledegree precedent [51]) | supervised labels beat emergent tokens/prompt encoders at our scale [§3.B][ESTIMATED] |
| Injection | FiLM/AdaIN everywhere: AdaLayerNorm in duration encoder, AdainResBlk in F0/N predictors, AdaIN blocks in decoder — concatenated conditioning vector `s = [speaker ⊕ style]` | the Kokoro/StyleTTS2 surface that makes an 82M model expressive [REPORTED][1][4] |
| Runtime conditioning source | precomputed per-(voice, style, phoneme-length-bucket) vectors shipped in packs (Kokoro pattern [3]) | removes any inference-time encoder; length-bucketing lets prosody adapt to utterance length [REPORTED][5] |
| Prosody predictors | FastSpeech2-class duration/F0/energy heads conditioned on s | knobs for free [REPORTED][17] |

Explicitly NOT in G1: reference-audio speaker encoder (no cloning; space-mismatch
hazard [61]), diffusion-sampled style vectors (CPU cost + nondeterminism),
textual-prompt style encoder (data+model cost, AR-oriented [21][22][24]).

### 8.2 Conditioning path (forward pass)

```
phonemes ──► text encoder ──────────────► duration predictor(s=[spk⊕sty], prev-sent ctx)
                                              │ durations × rate_knob
                                              ▼
                                        length regulator
                                              │
              s=[spk⊕sty] ──► F0 predictor ──► + pitch_shift / pitch_scale / doc-anchor norm
              s=[spk⊕sty] ──► N predictor  ──► + energy_gain
                                              ▼
              emphasis: boost duration+F0 of marked words (pre-regulator)
              pauses: insert per pause-policy at punctuation/tags
                                              ▼
              decoder (AdaIN-conditioned on s) ──► mel/waveform ──► vocoder
```

Every knob is a transform on predicted tracks — no knob touches model weights.

### 8.3 Control API surface (v0 proposal)

```yaml
synthesize(text, voice="mira", style="warm", style_degree=1.0,
           rate=1.0,            # clamped by voices/mira/prosody.yaml
           pitch_shift_st=0.0,  # ±3 st default clamp
           pitch_scale=1.0,
           energy_gain_db=0.0,  # ±6 dB
           pauses={"comma_ms":null,"period_ms":null,"custom":[[after_word_i,ms]]},
           emphasis=[word_indices],
           seed=<int>)          # required for reproducibility; None = fixed default
```

Rules: unknown style string → fall back to neutral + log warning (Azure fallback
pattern [REPORTED][51]); every output records voice-pack hash + all knob values +
seed (reproducibility charter); out-of-clamp values are clamped and reported, never
silently accepted.

### 8.4 Single riskiest assumption

**That a small supervised style table trained on modest licensed data will produce
perceptibly distinct, reliably transferable styles across ALL catalog voices.**
Everything else has strong precedent (tables work [7][3]; AdaIN works [4]; knobs work
[17]); but no public system ships exactly "10 named styles × few owned speakers ×
small model" with published evidence — ESD-class corpora are small, Expresso is
NC-locked, and prompt-control systems bought their control with 45k hours [21].
If styles fail to transfer across voices (P1 failure, §2.3), the G1 style story degrades
to per-(voice,style) pairs actually recorded — which changes Phase D recording scripts
from "styles × fewer hours" to "every voice × every style". This is exactly what E-J3 /
E-H1 must falsify FIRST, at pilot scale, before Phase F freezes the architecture.
[HYPOTHESIS]

---

## 9. DATASET IMPLICATIONS FOR PHASE D

Priority order for style/speaker-relevant data:

1. **Custom studio recordings of catalog voices (highest value, must-have)** —
   6–10 licensed voices × 2–4 h, script engineered so each of the 10 styles gets
   ≥15–20 min per voice, with parallel-style passages where feasible (ESD's parallel
   design is what makes style transfer learnable [REPORTED][26]). This is the only way
   to own the exact style taxonomy commercially. Budget driver for G1.
2. **LibriTTS-R + LibriTTS-P annotations** — multi-speaker breadth for stable training;
   P ships utterance-level pitch/speed/loudness tags (5-level quantiles) + raw
   F0/rate/LUFS statistics + human-written speaker prompts; CC-friendly lineage from
   LibriTTS-R [REPORTED][58] — verify exact license in Phase D.
3. **Helsinki prominence annotations over LibriTTS** — word-level prominence/boundary
   labels (~2.8M words) = free emphasis supervision [REPORTED][59].
4. **ESD English subset** — the only sizeable parallel-emotion corpus; research
   release: commercial terms MUST be verified before any training use [UNVERIFIED →
   Phase D license gate].
5. **Expresso** — best style taxonomy match (calm/whisper/laughing/conversational…)
   but CC BY-NC 4.0: excluded unless Meta licensing is negotiated [REPORTED][29].
6. **RAVDESS / CREMA-D** — too lexically thin to train on (2 and 12 sentences
   respectively [REPORTED][27][28]); useful only as evaluation material for emotion
   recognition metrics, if their licenses permit redistribution of derived artifacts.
7. **dataspeech-style auto-annotation of everything we ingest** — rate/SI-SDR/
   reverberation/monotony/pitch bins → text descriptions [REPORTED][23]; gives us
   Parler-grade metadata cheaply, plus SER-derived arousal/valence pseudo-labels as
   continuous supervision [ESTIMATED].

Also feed Phase D: espeak-ng GPL dependency risk interacts with g2p choice here (Kokoro
relies on external g2p [REPORTED][3]) — track B owns that call.

---

## 10. FALSIFICATION EXPERIMENTS (Phases H/J/K)

Phase H (speaker/style/prosody system):
- **E-H1 Identity/style representation bake-off**: d_speaker ∈ {64,128,256} × with/without
  speaker-classifier auxiliary loss × style-table vs style-free baseline. Metrics:
  speaker-classifier accuracy between catalog voices, style-classification accuracy,
  linear-probe leakage (predict F0 range/timbre features from style vector alone).
  Falsifies: "128-d table suffices", "style table stays speaker-clean".
- **E-H2 Determinism audit**: same seed + same pack → bit-identical waveform across
  runs/processes; different seeds → bounded variation. Falsifies: "NAR pipeline is
  deterministic given fixed seed".
- **E-H3 Emphasis bake-off**: duration+F0 boost vs markup-tag conditioning vs Helsinki-
  label-supervised emphasis. Metric: forced-choice listener identification of target
  word + WER regression. Falsifies: "zero-data boost is enough for G1".
- **E-H4 Knob cliff mapping**: grid sweeps of rate×pitch×energy per voice; measure
  naturalness proxy (UTMOS/MOS-lite) + WER; derive per-voice clamps for prosody.yaml.
  Falsifies: assumed safe ranges (esp. slow-rate α>1.33 [52][53]).

Phase J (first pilot model):
- **E-J1 Catalog distinctness**: ABX listening + speaker-classifier confusion matrix
  over ≥6 pilot voices incl. one derived blend pack. Falsifies: "blends are
  perceptually distinct voices" and "6 recorded voices stay separable".
- **E-J2 Long-form drift**: generate 3-min documents; per-chunk UTMOS, per-chunk F0
  median/range, boundary discontinuity metric, with vs without doc-anchors + predictor
  context. Falsifies §7.3 items 3/5.
- **E-J3 Cross-voice style transfer**: apply each style to every voice; listener
  ratings of style strength AND identity preservation (P1/P3 checks). THE experiment
  for §8.4's riskiest assumption.

Phase K (quality iteration):
- **E-K1 Voice addition without collapse**: add an 11th voice via (a) full retrain with
  replay, (b) frozen-backbone new-row fine-tune; regression-test original 10 against
  goldens. Falsifies: "few-shot voice addition is safe".
- **E-K2 Style-degree linearity**: degree 0→2 sweep rated for monotonicity; falsifies
  "style_degree behaves as continuous intensity".
- **E-K3 GST extension**: unsupervised token bank trained alongside labeled styles;
  do discovered tokens map to usable extra styles? Falsifies §3.A's Phase-K potential.

---

## Sources

All sources verified via WebSearch/WebFetch against primary or near-primary pages,
2026-08-22, unless marked otherwise.

1. StyleTTS 2: Towards Human-Level TTS through Style Diffusion and Adversarial Training
   with Large Speech Language Models — Li et al., NeurIPS 2023 (arXiv:2306.07691), Jun
   2023 — https://arxiv.org/abs/2306.07691 ;
   https://proceedings.neurips.cc/paper_files/paper/2023/file/3eaad2a0b62b5ed7a2e66c2188bb1449-Paper-Conference.pdf
2. StyleTTS2 official repo + fine-tuning guide — Y. Li (yl4579/StyleTTS2), 2023 —
   https://github.com/yl4579/StyleTTS2
3. Kokoro-82M model card — hexgrad, Hugging Face, v0.19 Dec 25 2024, Apache 2.0 —
   https://huggingface.co/hexgrad/Kokoro-82M/blob/main/README.md
4. Kokoro source: modules.py / istftnet.py — hexgrad/kokoro GitHub & hexgrad/Kokoro-82M
   HF, Dec 2024 — https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py ;
   https://huggingface.co/hexgrad/Kokoro-82M/blob/main/istftnet.py
5. "Kokoro: how it works" explainer (style vector per length bucket; pipeline) —
   Rudrite Research, ~2025 — https://research.rudrite.com/kokoro
6. inferencelayer Rust docs, Kokoro chapter (duration encoder AdaLayerNorm, SineGen
   determinism) — docs.rs, 2025 —
   https://docs.rs/inferencelayer/latest/inferencelayer/kokoro/index.html
7. Matcha-TTS: A fast TTS architecture with conditional flow matching — Mehta et al.,
   ICASSP 2024 (arXiv:2309.03199), Sep 2023 — https://arxiv.org/abs/2309.03199
8. Matcha-TTS official code — shivammehta25/Matcha-TTS —
   https://github.com/shivammehta25/Matcha-TTS
9. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching —
   Chen et al., ACL 2025 (arXiv:2410.06885), Oct 2024 — https://arxiv.org/abs/2410.06885 ;
   https://github.com/SWivid/F5-TTS
10. VITS2: Improving Quality and Efficiency of Single-Stage TTS with Adversarial
    Learning and Architecture Design — Jung et al., Interspeech 2023 submission
    (arXiv:2307.16430), Jul 2023 — https://arxiv.org/abs/2307.16430
11. VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End
    TTS — Kim et al., ICML 2021 (arXiv:2106.06103), Jun 2021 [ID from memory —
    re-verify in Phase D] — https://arxiv.org/abs/2106.06103
12. Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
    Speech Synthesis — Wang et al., ICML 2018 (arXiv:1803.09017) —
    https://arxiv.org/abs/1803.09017 ;
    https://proceedings.mlr.press/v80/wang18h.html
13. Deep Voice 2: Multi-Speaker Neural Text-to-Speech — Gibiansky et al., NIPS 2017
    [arXiv ID not verified this session] —
    https://papers.nips.cc/paper_files/paper/2017/hash/7fcf752c8d63bc0da5b1727511ebe4d4-Abstract.html
14. Generalized End-to-End Loss for Speaker Verification (GE2E d-vectors) — Wan, Wang,
    Papir, López Moreno, Google, ICASSP 2018 (arXiv:1710.10467) —
    https://arxiv.org/abs/1710.10467 ; https://google.github.io/speaker-id/publications/GE2E/
15. Voice Cloning: a Multi-Speaker TTS System (CorentinJ-style pipeline) — Jia et al.
    (arXiv:2102.05630), Feb 2021 — https://arxiv.org/abs/2102.05630
16. YourTTS: Towards Zero-Shot Multi-Speaker TTS and Cross-Domain Voice Cloning —
    Casanova et al., ICML 2022 (arXiv:2112.02418) — https://arxiv.org/abs/2112.02418
17. FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech — Ren et al., ICLR
    2021 (arXiv:2006.04558) — https://arxiv.org/abs/2006.04558
18. Microsoft Research article on FastSpeech 2 variance adaptor (pitch/energy/duration
    predictors) — Microsoft Research, 2020/2021 —
    https://www.microsoft.com/en-us/research/articles/fastspeech-2-fast-and-high-quality-end-to-end-text-to-speech/
    ; ESPnet2 FastSpeech2 module docs (speed alpha, teacher forcing) —
    https://espnet.github.io/espnet/guide/espnet2/tts/FastSpeech2.html
19. FastPitch: Conditional Text-to-Speech with F0 Conditioning — Łańcucki, ICASSP 2021
    (arXiv:2006.06873) — https://arxiv.org/abs/2006.06873
20. FastPitch project page (interactive pitch manipulation demos) — NVIDIA, 2020 —
    https://fastpitch.github.io/
21. Natural language guidance of high-fidelity text-to-speech with synthetic
    annotations — Lyth & King, Stability AI / U. Edinburgh (arXiv:2402.01912), Feb 2024
    — https://arxiv.org/abs/2402.01912 ; samples:
    https://text-description-to-speech.com/
22. Parler-TTS repo (Mini 880M / Large 2.3B, Aug 2024; reproduction of [21]) —
    Hugging Face — https://github.com/huggingface/parler-tts
23. Data-Speech annotation toolkit (rate/SI-SDR/reverberation/monotony bins → natural-
    language descriptions) — Hugging Face — https://github.com/huggingface/dataspeech
24. Orpheus TTS (Llama-3B speech LM; emotive tags; sampling-param requirements) —
    Canopy Labs, Mar 2025 — https://github.com/canopyai/Orpheus-TTS ;
    https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
25. Chatterbox TTS (exaggeration=emotion_adv, cfg_weight guidance; tts.py source) —
    Resemble AI, 2025 — https://github.com/resemble-ai/chatterbox ;
    https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts.py
26. The Emotional Speech Dataset (ESD) — Zhou et al., Interspeech 2021
    (arXiv:2105.14762); corpus page — https://arxiv.org/abs/2105.14762 ;
    https://github.com/HLTSingapore/Emotional-Speech-Data
27. RAVDESS — Livingstone & Russo, 2018 (Zenodo; 24 actors, 8 emotions × 2 intensities)
    [Zenodo ID not verified this session] — https://zenodo.org/record/1188976
28. CREMA-D — Cao et al., IEEE Trans. Affective Computing 2014 (91 actors, 6 emotions;
    crowd-rated) [details via ESD paper comparison; primary site unverified this
    session] — https://ieeexplore.ieee.org/document/6849440
29. Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis —
    Nguyen et al., Interspeech 2023; CC BY-NC 4.0 — https://speechbot.github.io/expresso/
30. EmotiVoice (JETS-based; BERT style-prompt encoder as pooled style embedding) —
    NetEase Youdao, 2023 — https://github.com/netease-youdao/EmotiVoice/
31. Thesis: adding emotional control to StyleTTS2 via HuBERT emotion embeddings +
    precomputed averaged style vectors — Ukrainian Catholic University, ~2024–25 —
    https://er.ucu.edu.ua/bitstreams/871b65b6-f7aa-47d6-a590-38ecfb1cab4a/download
32. Cross-speaker Style Transfer with Prosody Bottleneck in Neural Speech Synthesis —
    Microsoft Research publication page —
    https://www.microsoft.com/en-us/research/publication/cross-speaker-style-transfer-with-prosody-bottleneck-in-neural-speech-synthesis/
33. Improving Multi-Speaker TTS Prosody Variance with a Residual Encoder and
    Normalizing Flows — Vallès-Pérez et al., Interspeech 2021 —
    https://www.isca-archive.org/interspeech_2021/vallesperez21_interspeech.pdf
34. AutoPST: Unsupervised Speech Decomposition via Triple Information Bottleneck —
    Qian et al., ICML 2020 — http://proceedings.mlr.press/v119/qian20a/qian20a.pdf
35. Information Sieve: Content Leakage Reduction in End-to-End Prosody Transfer —
    (arXiv:2108.01831), Aug 2021 — https://arxiv.org/abs/2108.01831
36. GenerTTS: Pronunciation Disentanglement for Timbre and Style Generalization in
    Cross-Lingual TTS — (arXiv:2306.15304), Jun 2023 — https://arxiv.org/abs/2306.15304
37. Emotion Arithmetic: Emotional Speech Synthesis via Weight Space Interpolation —
    Kalyan et al.(?), Interspeech 2024 —
    https://www.isca-archive.org/interspeech_2024/kalyan24_interspeech.pdf
38. EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector —
    (arXiv:2406.07803), Jun 2024 — https://arxiv.org/abs/2406.07803
39. UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional TTS —
    (arXiv:2505.10599), May 2025 — https://arxiv.org/abs/2505.10599
40. Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching
    Zero-Shot TTS — (arXiv:2407.12229), Jul 2024 — https://arxiv.org/abs/2407.12229
41. Can Speaker Augmentation Improve Multi-Speaker E2E TTS? — Lai, Dheram, Watanabe
    (MIT SLS), Interspeech 2020 —
    https://sls.csail.mit.edu/publications/2020/JeffLai2_Interspeech_2020.pdf
42. Investigating Pretrained and Learnable Speaker Representations for Multi-Speaker
    TTS — (arXiv:2103.04088), Mar 2021 — https://arxiv.org/abs/2103.04088
43. Voice Fitting: fitting new speakers from a short untranscribed sample (~85-speaker
    VCTK result) — Nachmani et al., ICML 2018 —
    https://proceedings.mlr.press/v80/nachmani18a/nachmani18a.pdf
44. Meta-Learning Text-to-Speech Task from Unlabeled Data with Many Speakers (SEA-EMB
    vs SEA-ALL overfitting result) — Jia et al. (Google), 2018 (arXiv:1809.10460) —
    https://arxiv.org/abs/1809.10460
45. Word-level Text Markup for Prosody Control in Speech Synthesis — Korotkova et al.,
    Interspeech 2024 —
    https://www.isca-archive.org/interspeech_2024/korotkova24_interspeech.pdf
46. Unsupervised Word-Level Prosody Tagging for Controllable Speech Synthesis —
    (arXiv:2202.07200), Feb 2022 — https://arxiv.org/abs/2202.07200
47. Explicit Emphasis Control in Text-to-Speech Synthesis — Bauer et al., SSW 2025 —
    https://www.isca-archive.org/ssw_2025/bauer25_ssw.pdf
48. Emphasis Control in Parallel Neural Text-to-Speech Using Latent Features —
    Seshadri et al., Interspeech 2022 —
    https://www.isca-archive.org/interspeech_2022/seshadri22_interspeech.pdf
49. Controllable Emphasis with Zero Data for Text-to-Speech — Joly et al., SSW 2023 —
    https://www.isca-archive.org/ssw_2023/joly23_ssw.pdf
50. Amazon Polly SSML docs (engine-dependent prosody/emphasis support) — AWS —
    https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html ;
    https://docs.aws.amazon.com/polly/latest/dg/prosody-tag.html
51. Azure Speech SSML: mstts express-as (style/styledegree/role; neutral fallback) —
    Microsoft Learn —
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
52. Rate-modification artifacts study (0.5× CER 57.3% vs 10.6%; expansion errors 6×
    contraction) — IEEE Xplore doc 11062880, 2025 —
    https://ieeexplore.ieee.org/document/11062880
53. SCTTS: sentence-level speed conditioning (slow-speed asymmetry) — Bae et al.,
    Interspeech 2020 —
    https://www.isca-archive.org/interspeech_2020/bae20_interspeech.pdf
54. SRA-TTS: speaking-rate attention (MOS 4.35 vs 3.98 baseline speed control) —
    (arXiv:2310.08846), Oct 2023 — https://arxiv.org/abs/2310.08846
55. MagpieTTS-LF: Inference-Time Long-Form Speech Generation Without Training on
    Long-Form data — NVIDIA, arXiv:2606.18485, Jun 2026 —
    https://arxiv.org/abs/2606.18485 ; NeMo docs:
    https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/magpietts-longform.html
56. Context-Aware Memory TTS (paragraph-level consistency) — arXiv:2508.14713,
    Interspeech 2025 — https://arxiv.org/abs/2508.14713
57. HiGNN-TTS: hierarchical graph cross-sentence prosody modeling —
    (arXiv:2309.13907), Sep 2023 — https://arxiv.org/abs/2309.13907
58. LibriTTS-P: corpus with speaking-style and speaker-identity prompts — Kawamura et
    al. (LINE), Interspeech 2024 (arXiv:2406.07969) — https://arxiv.org/abs/2406.07969 ;
    https://github.com/line/LibriTTS-P
59. Helsinki-NLP prosody: CWT prominence/boundary annotations over LibriTTS (~160k
    sentences) — https://github.com/Helsinki-NLP/prosody
60. DeepWiki: Kokoro custom voice creation (weighted averaging blends; native comma-
    blending mean+cache) — DeepWiki/hexgrad, 2025 —
    https://deepwiki.com/zboyles/Kokoro-82M/5.2-custom-voice-creation ;
    https://deepwiki.com/hexgrad/kokoro/4-languages-and-voices
61. Kokoro issue #314: raw-wav style extraction fails (embedding-space mismatch);
    30%/70% blend workaround — hexgrad/kokoro GitHub, 2025 —
    https://github.com/hexgrad/kokoro/issues/314
62. ZeroVoice-KokoroTTS: procedural SLERP voice arcs with deterministic seeding —
    MushroomFleet GitHub — https://github.com/MushroomFleet/ZeroVoice-KokoroTTS
63. FiLM: Visual Reasoning with a General Conditioning Layer — Perez et al., AAAI 2018
    (arXiv:1709.07871) — https://arxiv.org/abs/1709.07871
64. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization — Huang
    & Belongie, ICCV 2017 (arXiv:1703.06868) — https://arxiv.org/abs/1703.06868

*End of dossier. Items flagged [UNVERIFIED] (sources 11, 13, 27, 28 IDs/details) are
citation-detail gaps only; their substantive claims came from verified secondary
discussions and need no re-research beyond a Phase D license check.*
