# V1 Voice Slate — 15 catalog voices from LibriTTS-R clean100

Task: **VOICE-SLATE-15** · Date: 2026-08-22 · Status: MEASURED (all numbers below are
computed, not estimated). Scripts + raw outputs live in `experiments/VOICE-SLATE-15/`
(`extract_features.py` → `features.json`, `select.py` → `picks.json`, `extract.log`).

## Method

1. **Eligibility** — speakers from `experiments/DATA-FULL-01/speaker_stats.json` with
   ≥18 min kept audio: **44 of 247** qualify.
2. **Sampling** — ≤12 utts/speaker chosen deterministically (evenly spaced indices,
   endpoints included, sorted file list); features measured on the first 8 sampled utts,
   capped at 10 s each (native 24 kHz).
3. **Features (per speaker)** — `librosa.pyin` (fmin 65 / fmax 400 Hz, frame 1024,
   hop 320 ≈ 13.3 ms): median F0 + IQR over **voiced frames only**, pooled across utts.
   Spectral centroid (n_fft 1024): pooled-frame median → timbre brightness proxy.
   Speaking-rate proxy: onset density (energy-envelope events / analyzed second).
   Energy stability: std of RMS dB over frames within 40 dB of clip peak (lower = steadier).
   Runtime measured: **4.3 min** wall for all 44 speakers (352 clips). No model loading.
4. **Gender split** — no local SPEAKERS metadata exists; presentation inferred from
   measured median F0 with a 155 Hz threshold → 25 feminine / 18 masculine among eligible.
5. **Selection** — narrator slot = longest clean hours (**6209**, 0.469 h). Then 7 + 7 via
   farthest-point sampling on ln(F0) inside each gender (seeded at range minimum),
   maximizing spread; ties broken by z(hours) + z(−RMS-std) + small bonus for centroid
   distance from nearest pick. The task's 15% neighbor rule is applied as a large
   penalty (−10 in-gender, −1 cross-gender/narrator) rather than a hard veto: uniform
   15% chaining over 7 voices needs a 2.3× F0 span per gender, but the measured spans
   are only 1.44× (F) / 1.60× (M) — a hard rule is arithmetically infeasible. Measured
   adjacent-pick gaps: F 2.9–9.1%, M 6.0–12.9%.
6. **Terciles (measured)** — feminine [184.9, 197.0] Hz → picks land 3 low / 2 mid /
   2 high; masculine [109.7, 122.5] Hz → 3 / 1 / 3.

## The slate

| Rank | libritts_speaker | Hours | F0 med/IQR (Hz) | Centroid (Hz) | Rate proxy (onsets/s) | Energy σ (dB) | Proposed persona | Risk notes |
|---|---|---|---|---|---|---|---|---|
| 01 | 6209 ★narrator | 0.469 | 117 / 30 | 1013 | 4.5 | 9.9 | Narration anchor — steady medium-male read | Longest clean hours by mandate; F0 sits inside masc cluster (≈8% from 8419) — reserve for narration slots |
| 02 | 6415 | 0.371 | 158 / 67 | 921 | 5.3 | 9.5 | Feminine warm/mature (low register) | At gender boundary (158 Hz); confirm presentation at Phase K listen gate |
| 03 | 696 | 0.346 | 170 / 64 | 1386 | 5.1 | 10.0 | Warm-bright conversational | Bright timbre on low-mid F0 — unusual pairing worth a listen check |
| 04 | 2416 | 0.365 | 181 / 34 | 1457 | 4.6 | 10.2 | Professional/neutral anchor (mid) | Narrowest F0 IQR of slate — monotone risk, but ideal for stable "professional" |
| 05 | 6147 | 0.389 | 190 / 51 | 1321 | 6.7 | 9.2 | Energetic conversational (mid-high) | Fastest onset rate on slate; only 5.3% F0 gap to rank 06 |
| 06 | 1088 ⚡fast-path | 0.448 | 196 / 52 | 1146 | 5.2 | 8.5 | Young professional (mid-high) | Already in 10-speaker pilot → embeddings exist; kept despite tight gap to 6147 (trade-off documented above); steadiest energy of F picks |
| 07 | 3879 | 0.385 | 209 / 63 | 1128 | 5.0 | 10.9 | Bright expressive (high) | High pitch variance (IQR 63) + high energy σ — style-control stress test |
| 08 | 2836 | 0.331 | 228 / 49 | 1340 | 4.4 | 9.2 | Young/airy top register | Highest F0 (228) and fewest hours among F picks |
| 09 | 78 | 0.385 | 95 / 20 | 971 | 4.4 | 11.1 | Grave/deep bass | Deepest voice (95 Hz); highest energy variability on slate |
| 10 | 7447 | 0.403 | 101 / 27 | 780 | 4.8 | 10.2 | Calm documentary | Darkest timbre (centroid 780 Hz) — watch intelligibility |
| 11 | 8419 | 0.334 | 108 / 24 | 1023 | 4.8 | 10.2 | Serious medium-low | Only 8% F0 from narrator 6209 — distinctness vs narrator slot |
| 12 | 7067 | 0.370 | 122 / 72 | 1263 | 5.1 | 9.4 | Dynamic storyteller (mid) | Widest F0 IQR on slate (72 Hz) — expressive but less controllable |
| 13 | 8838 | 0.307 | 134 / 51 | 828 | 5.5 | 10.1 | Warm baritone | Fewest hours among M picks |
| 14 | 7505 | 0.358 | 143 / 51 | 955 | 4.7 | 10.0 | Medium professional | Mid-pack on every metric — safest pick |
| 15 | 7402 | 0.326 | 151 / 52 | 922 | 5.2 | 9.8 | Younger tenor-range | 151 Hz at gender boundary; confirm presentation |

⚡ **Fast-path voices** (also in the current 10-speaker pilot
`experiments/CV-G1-E0007-pilot-training/smoke-a/checkpoints/speaker_index.json`,
so speaker embeddings are already trained): **1088** (rank 06). Pilot members 103,
1034, 1040, 1069, 1081, 1098, 118, 1183 are ineligible (<18 min); pilot member
1116 is eligible (F, 195 Hz, 0.366 h) but unpicked — first substitute if a feminine
swap is ever needed.

**Reserves** (highest-hours unpicked, measured): F — 40 (197 Hz, 0.459 h), 1841
(185 Hz, 0.381 h); M — 8088 (98 Hz, 0.450 h), 3830 (121 Hz, 0.422 h).

## Constraints & next gates

- Auto-metrics cannot close identity quality (EVALUATION-BENCHMARKS finding):
  every pick needs the Phase K/N human listening gate before freeze.
- Persona labels map onto the founder grid (`configs/voices/README.md`,
  MASTER-PLAN style vocabulary: grave/medium/young/warm/professional/energetic/calm)
  using measured bands: persona follows F0 tercile + centroid + rate proxies.
- Gender presentation is F0-inferred (no ground-truth labels locally); the two
  boundary voices (6415 @158, 7402 @151) must be perceptually confirmed.
- Codenames per `DATA-PILOT-ACQUISITION.md` §5 remain provisional until provenance
  + listening gates pass.
