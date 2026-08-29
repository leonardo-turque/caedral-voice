# STYLE BOOTSTRAP G1 — ten named styles without style-labeled corpora

**Task:** VOICE-STYLE-01 · **Status:** PLAN (nothing here is MEASURED until executed)
**Owner:** coordinator · **Depends on:** nothing (runs on already-frozen SMOKE/PILOT manifests)
**Evidence labels** per MASTER-PLAN §3/§9: `[MEASURED]` `[REPORTED]` `[ESTIMATED]` `[HYPOTHESIS]`.
All research claims cite `docs/research/MULTI-SPEAKER-PROSODY-STYLE.md` (below: "dossier §x / [n]");
no new literature research was done for this document.

---

## 0. Problem

G1 promises 10 styles (MASTER-PLAN §2): `neutral, conversational, professional,
energetic, calm, serious, warm, excited, soft, dramatic`. The dossier's primary recipe —
supervised style table trained on style-labeled corpora (dossier TL;DR-2, §3.B) — is
blocked at pilot scale: SMOKE/PILOT data (`librittsr-smoke-10spk`, 1.66 h / 10 spk;
LJSpeech slice) carries NO style labels, and the licensing-grade labeled corpora are
unusable (Expresso NC-locked, ESD research-release, lexical-thin actor sets — dossier
§3.B table). The model already has the conditioning slot ready: `style_table =
nn.Embedding(n_styles+1, 64)`, neutral = last row, FiLM-injected via
`s = [speaker ⊕ style]` (`caedral_voice/model/variance.py::ConditioningProvider`;
dossier §8.1). Today `pilot_train.py` hardcodes `style_id=0` for every row — which is a
*learned named-style row*, not the neutral sentinel (documented quirk, lines 28–29).
That must be fixed before any style training (§6).

Plan in one sentence: **manufacture style labels ourselves** with a dataspeech-class
prosody auto-annotator (dossier §9 item 7 [23]), cluster them unsupervised, map
clusters onto the 10 names with explicit threshold tables, and hold an honest,
pre-committed degradation ladder (knob presets → neutral) if the clusters do not
separate — because emergent-token semantics are exactly what we cannot ship in a
versioned catalog (dossier §3.A [12]).

---

## 1. Decision summary

| Question | Decision | Basis |
|---|---|---|
| Label source | Self-computed prosody metrics (pitch stats, energy variance, rate, pause ratio) per utterance | dataspeech recipe [REPORTED][23]; LibriTTS-P ships the same metric class as 5-level quantile tags [REPORTED][58]; dossier §9-7 |
| Labeling method | Unsupervised GMM clusters over per-speaker-normalized metrics → mapped to style names by signed target-profile matching | GST shows unsupervised prosody factors emerge (speed, pitch register) but semantics are not guaranteeable [REPORTED][12]; AutoPST: absent labels, dimensions stay entangled — our named mapping IS the missing label [REPORTED][34][ESTIMATED] |
| Degree control | `style_degree` interpolates/extrapolates style-row↔neutral-row; anchored to cluster-strength quantiles | Azure styledegree 0.01–2.0 precedent [REPORTED][51]; dossier §8.1 |
| Failure path | Ladder L0→L3, floor = deterministic prosody-knob presets (zero training data needed) | knobs are inference-time transforms on predicted tracks [REPORTED][17][19]; single-scalar honesty precedent Chatterbox [REPORTED][25] |
| Verification | Gate CV-G1-E0008 (data-only, CPU, anytime) → perceptual falsification CV-G1-E-K4 (Phase K) with pre-registered acceptance numbers | charter §8; extends dossier E-J3/E-K2 |

---

## 2. Auto-annotation pipeline spec (CV-G1-E0008 stage A)

Inputs: processed WAVs + manifests + `.phonemes.jsonl` already generated for the smoke
set (phonemization throughput 251 utt/s `[MEASURED]`, PILOT-PLAN). Everything below
runs offline on this laptop's CPU.

### 2.1 Per-utterance feature vector `x_u ∈ R^8`

| # | Feature | Definition | Tool |
|---|---|---|---|
| 1 | `f0_med_st` | median voiced-frame F0, `12·log2(f0/100)` | librosa pyin, 60–400 Hz |
| 2 | `f0_range_st` | P90 − P10 of voiced F0 (st) | same |
| 3 | `f0_std_st` | std of voiced F0 (st) | same |
| 4 | `e_var_db` | std of voiced-frame RMS in dB — the "speech monotony" inverse | frame RMS, 25 ms/10 ms |
| 5 | `e_dyn_db` | P90 − P10 of frame RMS (dB) | same |
| 6 | `e_med_dbfs` | median frame RMS (dBFS) | same |
| 7 | `rate_phps` | n_phonemes / voiced_duration_s (phonemes already exist) | manifest join |
| 8 | `pause_ratio` | silence frames / total frames; silence = RMS < (utt median − 35 dB) sustained ≥ 120 ms; also emit `pauses_per_min` (diagnostic only) | energy gate |

This is the dataspeech measurement class — rate, pitch statistics, monotony/energy
variance — minus SI-SDR/reverb (quality gates, not style signals; our data is already
clean-split) `[ESTIMATED from REPORTED][23]`. Pause insertion is one of the five free
variance-adapter knobs (dossier §4.1 [17][50]), so pause behavior is both a feature and
later a control axis.

### 2.2 Normalization and leakage controls (non-negotiable)

1. **Per-speaker z-score** every feature, then winsorize |z| > 3. Raw registers would
   make clusters == speakers; reference-prosody codes secretly carrying speaker timbre
   is failure mode #1 of dossier §2.2 [REPORTED][32][33].
2. **Kill-switch:** compute `MI(cluster; speaker) / H(cluster)` on the contingency
   table. If ≥ 0.8, clusters ARE speaker identities → declare auto-annotation failed,
   descend the ladder (§5). Do not relabel speakers as "styles."
3. Emit everything into `data/style_metrics.yaml`: pooled quantiles (q20…q80 per
   feature = the threshold tables of §4), cluster centroids, per-utterance
   `{utt_id, x_u, cluster_id, posterior γ_u, projected_strength t_u}` JSONL, manifest
   hash. Immutable-artifact discipline per charter §7.

Compute cost: pyin ≈ 0.2–0.5× realtime per core `[ESTIMATED]`; 1.66 h smoke ≈ tens of
minutes; even a ~16 GB PILOT corpus stays an offline batch job.

### 2.3 Clustering

Diagonal-covariance GMM (posterior γ_u needed later for degree anchoring and
confidence gating), k swept 4–16, select by BIC elbow + silhouette on an 80/20 split;
stability = adjusted Rand index across 5 seeds.

**Separation gate (pre-registered, `[HYPOTHESIS]` thresholds):**
- silhouette ≥ 0.20 on held-out split;
- coverage: fraction of utterances with γ_u ≥ 0.5 inside mapped clusters ≥ 60%;
- ARI(seed pairs) ≥ 0.6.

Expectation honesty: prosody metrics plausibly support 4–6 archetypes (rate × pitch
register × pause × energy-variance axes), NOT automatically 10 — GST experience says
you get speed/pitch-register-like factors, not guaranteed semantic slots
[REPORTED][12][ESTIMATED]. Styles that receive no cluster simply do not get
pseudo-labels; they fall to ladder rung L2 (presets) individually.

---

## 3. Threshold tables (two kinds)

### 3.1 Metric binning table (dataspeech-style)

Emit pooled per-speaker-normalized quintiles as the canonical bins:

| Feature | bin −2 | bin −1 | bin 0 | bin +1 | bin +2 |
|---|---|---|---|---|---|
| any `x ∈ R^8` | z < q20 | q20–q40 | q40–q60 | q60–q80 | z > q80 |

Same 5-level quantile-tag design as dataspeech's binned descriptions [REPORTED][23]
and LibriTTS-P's pitch/speed/loudness tags [REPORTED][58]. Numbers are produced by
measurement, never hand-set. Optional external validation: if Phase D confirms
utterance-ID overlap between our LibriTTS-R slice and LibriTTS-P's subset, join their
published tags and report agreement of our bins vs theirs (free sanity check, no new
research) `[HYPOTHESIS]`.

### 3.2 Cluster→style assignment table (signed target profiles)

Style names get *a-priori signed expectations* over the 8 normalized features
(order: f0_med, f0_rng, f0_std, e_var, e_dyn, e_med, rate, pause_ratio).
These priors are design hypotheses seeded from common usage of the style names —
`[HYPOTHESIS]`, not measurements; they only bias the assignment, the centroids decide.

| Style | f0_med | f0_rng | f0_std | e_var | e_dyn | e_med | rate | pause |
|---|---|---|---|---|---|---|---|---|
| neutral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| conversational | 0 | +1 | +1 | +1 | 0 | 0 | +1 | 0 |
| professional | 0 | −1 | −1 | −1 | 0 | 0 | 0 | +1 |
| energetic | +1 | +1 | +1 | +2 | +1 | +1 | +1 | −1 |
| calm | 0 | −1 | −1 | −1 | −1 | 0 | −1 | +1 |
| serious | −1 | −1 | −1 | −1 | 0 | 0 | −1 | +1 |
| warm | +1 | 0 | 0 | 0 | 0 | 0 | −1 | +1 |
| excited | +2 | +2 | +2 | +2 | +1 | +1 | +2 | −1 |
| soft | 0 | −1 | 0 | −1 | −1 | −1 | −1 | +1 |
| dramatic | 0 | +2 | +2 | +2 | +1 | +1 | −1 | +2 |

Assignment rule: direction `d_s = c_cluster − μ_global`; score =
cosine(d_s, profile_row). Assign cluster→style iff cosine ≥ **0.6**, cluster mass ≥ 3%
of utterances, and the match is mutual argmax (one cluster ⇔ at most one style).
Unmatched clusters contribute NO style label (their utterances train as neutral);
unmatched styles go preset-only. Acceptance-band definition of neutral for labeling:
all |z| < 0.5.

---

## 4. Training integration (pseudo-label supervised style table)

1. Fix the sentinel bug first: pilot currently writes `style_id=0` (a learned
   named-row) instead of the neutral last row. Before any style run: labels file
   `data/style_labels.jsonl` drives `style_id`; rows without an accepted label use
   `None` → neutral sentinel.
2. Loss weighting: style-supervised rows weighted by GMM posterior γ_u (equivalently:
   L0 includes γ_u ≥ 0.5 only). Prevents noisy-label dilution.
3. Balance check per (speaker × style) cell before the run — empty cells create
   speaker-correlated style priors, the exact leakage failure the dossier warns about
   (§3.B recipe, [HYPOTHESIS → E-H1]). On 10-speaker smoke data some cells will be
   thin; report the table honestly and downweight starving cells rather than pretend
   balance.
4. Pack side: `voices/<id>/styles.npy` rows stay in MASTER-PLAN §2 order (neutral =
   separate sentinel row per ConditioningProvider); `style_defaults.yaml` gains
   per-style degree defaults; `prosody.yaml` gains the `style_presets` block (§5-L2).

---

## 5. Fallback ladder (pre-committed, honest degradation)

Each rung keeps the API identical (`synthesize(text, voice, style=..., style_degree=...)`);
only the mechanism behind the string changes. Unknown/unavailable style → neutral +
warning, mirroring Azure's invalid-style fallback [REPORTED][51].

| Rung | Mechanism | Entry condition | Cost |
|---|---|---|---|
| **L0 full** | style table trained on all mapped pseudo-labels (γ ≥ 0.5) | separation gate passed (§2.3) | already-scheduled training |
| **L1 confident-core** | train table on top-50%-posterior core of each mapped cluster only; rest neutral | gate marginal (silhouette 0.15–0.20 or coverage 40–60%) | same |
| **L2 presets** | NO style training. Style string → deterministic prosody-knob preset applied to a neutral-trained model (table below). Works TODAY, zero data dependency — knobs are inference-time transforms on predicted duration/F0/energy tracks [REPORTED][17][18]; emphasis via zero-data duration+F0 boost [REPORTED][49] | gate failed, or L0/L1 fail E-K4 perception | ~0 (config only) |
| **L3 neutral-only** | ship neutral catalog; style API returns warning+neutral; publish measured minimum (charter §9.5) | even presets fail quality gates for a style (E-H4 cliff mapping) | none |

L2 preset table (initial values `[ESTIMATED]`, all inside the §4.2 safety clamps:
rate 0.75–1.5 [REPORTED slow-rate cliff][52][53]; pitch ±3 st, identity-safe zone
[REPORTED][19]; pitch_scale 0.8–1.25; energy ±6 dB; per-voice re-derived by E-H4):

| Style | rate α | pitch_shift (st) | pitch_scale | energy (dB) | pause × | emphasis policy |
|---|---|---|---|---|---|---|
| neutral | 1.00 | 0.0 | 1.00 | 0 | 1.00 | — |
| conversational | 1.05 | +0.5 | 1.10 | +1 | 0.90 | light |
| professional | 1.00 | 0.0 | 0.90 | 0 | 1.15 | — |
| energetic | 1.15 | +1.0 | 1.15 | +2 | 0.80 | medium |
| calm | 0.90 | 0.0 | 0.85 | −1 | 1.30 | — |
| serious | 0.90 | −1.0 | 0.80 | −1 | 1.25 | — |
| warm | 0.95 | +0.5 | 0.95 | +1 | 1.10 | light |
| excited | 1.25 | +2.0 | 1.25 | +3 | 0.70 | strong |
| soft | 0.95 | −0.5 | 0.90 | −4 | 1.15 | — |
| dramatic | 0.85 | −1.0 | 1.30 | +2 | 1.60 | strong |

Honest framing shipped with L2: these are *prosody signatures*, not learned expressive
styles; the product copy and pack metadata must say so. This mirrors the field's own
honesty — even Chatterbox's flagship "emotion control" is one scalar plus a CFG knob
[REPORTED][25], and Polly ships only knobs that survive quality testing (dossier §6.1
[50]).

---

## 6. `style_degree` ↔ cluster geometry

Semantics (extends dossier §8.1 "degree scalar interpolates row↔neutral"; Azure range
0.01–2.0, default 1.0, fallback-to-neutral on invalid [REPORTED][51]):

1. **Runtime formula.** With neutral sentinel row `v_0` and style row `v_s`:
   `v(s,d) = v_0 + w(d) · (v_s − v_0)`, `w(d) = clamp(d, 0.01, 2)`.
   d ≤ 1 interpolates toward the style row; d > 1 continues along the same direction
   (extrapolation) capped at 2× displacement and by the embedding-norm cap derived
   from E-H4 knob cliffs (extrapolation past the data support is where artifacts live).
2. **Cluster-distance anchoring (what makes d mean one thing).** In normalized metric
   space let `û_s` be the unit vector μ_global→centroid `c_s` and `t_u = x_u · û_s`
   each member's projected strength. Anchor the trained row so that **d = 1.0
   corresponds to the within-cluster 75th percentile of t_u** — i.e., "default" means a
   typical strong realization, robust to outliers. d = 0.5 ≈ half-way to typical-strong;
   d = 2.0 ≈ twice the anchor displacement (validated, not assumed, by the monotonicity
   arm of E-K4). If t_75 ≤ 0 the cluster does not clear the neutral band → style
   declared non-separable → L2 preset for that style.
3. **Confidence coupling at train time:** row updates weighted by γ_u, so weakly
   clustered utterances pull the row less — degree then measures movement through
   *occupied* territory of the prosody distribution, not noise.
4. Determinism: `v(s,d)` is a closed-form function of pack tensors + d; no sampling,
   no encoder (Kokoro-pattern precomputed conditioning, dossier §8.1 [3]).
5. Perceptual validation: E-K2's degree-linearity sweep (dossier §10) is folded into
   E-K4 arm D below.

---

## 7. Falsification experiments

### 7.1 CV-G1-E0008 — "style-cluster separability" (gate; data-only, CPU, run anytime)

HYPOTHESIS: dataspeech-class metrics over SMOKE/PILOT data form clusters that map onto
≥ 4 of the 10 target styles under §3.2 thresholds, without collapsing onto speakers.
METHOD: §2 pipeline verbatim; pre-registered gates: silhouette ≥ 0.20, coverage(γ≥0.5)
≥ 60%, ARI ≥ 0.6, MI(cluster;speaker)/H(cluster) < 0.8, ≥ 4 styles matched with
cosine ≥ 0.6. RESULT → KEEP/REJECT decides entry rung of the ladder. Registered in
EXPERIMENT-INDEX when executed.

### 7.2 CV-G1-E-K4 — "style bootstrap falsification" (Phase K; extends dossier §10)

HYPOTHESIS H: pseudo-label-supervised style conditioning produces perceptibly distinct
styles **beyond what knob presets alone achieve** (this is the sharp question — if
presets suffice, L0/L1 machinery is dead weight and G1 ships L2).

- **Arms.** A = neutral-trained model + L2 presets; B = best of L0/L1; C = neutral
  baseline (no knobs); D = B swept at d ∈ {0.5, 1.0, 1.5} (absorbs E-K2).
- **Stimuli.** Golden set per voice (dossier §6.3: ~50 texts, short/medium/long,
  edge cases) × 10 styles × ≥ 3 catalog voices; fixed seeds; pack hashes logged.
- **Acceptance numbers (pre-registered):**
  - Primary — 10-way forced-choice style identification (chance 10%):
    **B ≥ 30% correct AND B − A ≥ +10 pts.**
  - Identity preservation P1: ABX same-voice judgment **≥ 80%** vs neutral
    (or speaker-classifier margin drop ≤ 10 pts).
  - Intelligibility P3: minWER(B) − minWER(C) **≤ +2 pts absolute** (dual-ASR harness
    from Phase C); UTMOS drop ≤ 0.3.
  - Degree linearity (arm D): Spearman ρ(rated intensity, d) **≥ 0.8** pooled.
- **Decision table.**
  - E0008 gate failed → evaluate A only; A identification ≥ 25% ⇒ adopt L2 as the G1
    style story (decision-log entry; recording scripts unchanged).
  - A pass, B fails primary ⇒ one L1 confident-core retrain allowed; still failing ⇒ L2.
  - B passes all ⇒ L0/L1 ships; presets remain as prosody.yaml convenience block.
  - Any style individually failing while others pass ⇒ that style drops to L2/L3;
    per-style results published (charter §9.4/9.5).
- **Measurement honesty:** if no human panel is available at K, run the proxy stack
  (UTMOS + dual-ASR minWER + a prosody-feature classifier trained on our own labels)
  and mark results PROXY — circularity documented; no product claim without a human
  forced-choice panel. This experiment is the pilot-scale falsifier for the dossier's
  single riskiest assumption (§8.4) adapted to the no-label reality; the full
  cross-voice transfer question remains E-J3 once real style-bearing recordings exist.

---

## 8. Explicitly out of scope for this bootstrap

GST token banks as the API basis (semantics drift between runs — dossier §3.A [12];
kept as E-K3 exploration for extra styles beyond the ten); textual prompt encoders
(large-model recipe [21][22]); reference-audio style extraction (space-mismatch hazard
[61]); SER arousal/valence enrichment of labels (dossier §3.D [38][39]) — optional
post-E0008 add-on only if L0 passes and it earns its CPU cost `[HYPOTHESIS]`.

## 9. Immediate actions (no dependencies)

1. Implement §2 extractor + `data/style_metrics.yaml` emitter; run on
   librittsr-smoke-10spk (minutes-scale) → register CV-G1-E0008.
2. Fix the `style_id=0` sentinel quirk in the pilot trainer before any style run.
3. Land L2 preset table in `configs/prosody_presets.yaml` (works against today's
   forward-API knobs — CURRENT-STATE-REPORT: "Prosody knobs IMPLEMENTED").
4. Queue E-K4 arms in EXPERIMENT-INDEX as PLANNED.
