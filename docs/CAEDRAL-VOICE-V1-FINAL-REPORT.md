# CAEDRAL VOICE V1 — FINAL REPORT

Status: **SKELETON** (2026-08-22). Sections follow the founder's required list.
Every unfilled metric carries `[PENDING — <producer>]` wired to its producing
experiment: `CV-G1-E0011` (CTC aligner), `CV-G1-E0012` (teacher corpus),
`CV-G1-E0013` (V1 regressor model), `CV-G1-E0014` (long-form), **natgate**
(GATE-NAT-01 naturalness/intelligibility run), **threads** (RTF/RAM/TTFA sweep).
The report is DONE only when every PENDING marker is replaced by a MEASURED value
with an artifact path, and every gate in
`artifacts/caedral-voice-v1/provenance/release-gates.json` is PASS.

Evidence labels per charter §9: [MEASURED] (artifact on this machine),
[REPORTED BY THIRD PARTY], [ESTIMATED] (arithmetic shown), [HYPOTHESIS].
Machine of record: i7-13620H (10C/16T), Windows 11, CPU-only.

---

## 1. Architecture (final)

Final V1 system: our text frontend (TN + lexicon + g2pen ladder; GPL espeak
never in the path) -> CAEDRAL-NAR regressor backbone with supervised CTC
durations -> L1 log-mel decoder -> Vocos-class iSTFT head. Canonical mel:
24 kHz, 80-band ln-mel, hop 256, n_fft 1024 (single-sourced in
`caedral_voice/training/mel_dataset.py`). Inference sigma default 0.25
(E0009 finding).

- Final config (dims, n_speakers, buckets): [PENDING — CV-G1-E0013 final run_manifest]
- ONNX export shape (stages A/B/C per `docs/architecture/ONNX-EXPORT-PLAN-G1.md`,
  opset 17, static T_x buckets {50,200,800}, dynamic T_y): [PENDING — RUNTIME-02 export gates A/B/C]
- Design lineage: E0002 (NAR sizing), E0006 (flow variant), E0005/E0009,
  ARCH-NOTES-G1.

## 2. Changes since the first bad prototype (incl. the MAS -> CTC -> regressor pivot)

History, already on record:

1. **First bad prototype** — pilot tiny trained on the 10-speaker smoke slice
   (`experiments/pilot-tiny-15k`, then E0007) produced unintelligible babble.
   Founder verdict: intelligible prototypes are diagnostic only; DNSMOS ~2.0 is
   not product.
2. **E0009 diagnosis** — vocoder/mel chain proven CLEAN (GT audio survives our
   mel -> Vocos path intelligibly); overfit gate steps 1–3 PASS (flow_loss
   38.65 -> −33.94) but gate 4 FAIL (best WER 0.5 vs <= 0.35 requirement).
   Probes EXONERATED mel-norm, expand_tokens masking, prior collapse
   (text-sensitivity ratio 1.05), and duration prediction (r=0.998); suspects
   narrowed to degenerate MAS silence-dumping and i.i.d. per-frame flow
   sampling losing inter-frame detail.
3. **THE PIVOT (MAS/flow -> CTC durations -> mel regressor)**, reasons on record:
   - MAS alignment inside the flow objective was the remaining suspect for
     silence-dumping; replacing learned-alignment risk with supervised CTC
     durations (own IP, CV-G1-E0011) removes that failure mode entirely.
   - Regression to mels with L1 loss converges far faster than flow matching
     on CPU budget (E0006 showed the flow line was cheap per step but the
     E0009 result showed it could not close the quality gate).
   - Founder-authorized Kokoro teacher mels (E0012, Apache-2.0,
     training-data-only) give the regressor clean targets at scale.
   - A-arm control retained: clean100-only flow retrain queued behind B-arm
     resources as the honest from-real-data comparison.
4. Remaining deltas to V1: [PENDING — CV-G1-E0013 final quality iteration list]

## 3. Datasets & licenses

| Dataset | Role | License | Scale used | Attribution duty |
|---|---|---|---|---|
| LibriTTS-R train-clean-100 | real speech backbone + text source | CC BY 4.0 | 44.11 h / 30,917 utts / 247 spk [MEASURED, DATA-FULL-01] | NOTICE.md credit |
| Kokoro-82M v1.0 re-speaks (teacher) | training targets ONLY, never shipped | Apache-2.0 | h-kept [PENDING — CV-G1-E0012 final meta] | statement in LICENSES/ |
| Vocos (code + weights) | shipped decoder dependency | MIT | 13.5M reference decoder | MIT notice retention |

Full dossier: `artifacts/caedral-voice-v1/LICENSES/`; manifests hashed under
`data/manifests/`. Prohibited-by-policy rows (HiFi-GAN original checkpoints,
Piper distillation, NC corpora): see TEACHER-DATA-ENGINE §8 and
DECODER-LICENSES.md §6. Counsel-review open items before commercial cut:
outputs-as-training-data doctrine; EU AI Act Art. 50 transparency.

## 4. Training compute

All CPU (charter §15 honesty: never fake training scale).

| Stage | Producer | Measured cost |
|---|---|---|
| CTC aligner (E0011) | CV-G1-E0011 | config: 15 epochs, 6 threads, 1.61M-param aligner over 29,146 train utts [MEASURED manifest]; total wall [PENDING — CV-G1-E0011 conclusion] |
| Teacher corpus generation | CV-G1-E0012 | shards + hours generated + RTF observed [PENDING — CV-G1-E0012 meta/conclusion] |
| V1 regressor run | CV-G1-E0013 | ~2.1 s/step @4 threads, 15k steps, budget 8–10 h [MEASURED smoke]; actual wall + peak RSS [PENDING — CV-G1-E0013 final] |

## 5. Parameters

| Component | Params |
|---|---|
| Final V1 model | [PENDING — CV-G1-E0013 final checkpoint] (smoke build measured 15.91M) |
| CTC duration aligner (inference: none — durations supervised/precomputed) | 1.61M train-time only [MEASURED, CV-G1-E0011 run_manifest] |
| Vocos mel decoder | 13.5M [MEASURED, DECODER-LICENSES §1] |
| Reference family (context): tiny / lite / quality NAR prototypes | 8.35M / 31.6M / 80.7M [MEASURED, E0002] |

## 6. The 15 voices

Slate frozen by VOICE-SLATE-15 (measured F0/centroid/rate selection):
6209 narrator, 6415, 696, 2416, 6147, 1088, 3879, 2836, 78, 7447, 8419, 7067,
8838, 7505, 7402; reserves F:40/1841, M:8088/3830.

- Per-voice listening-gate verdicts (gate: >= 13 of 15 approved):
  [PENDING — human listening gate over CV-G1-E0013 renders]
- Boundary-presentation confirmations (6415 @158 Hz, 7402 @151 Hz):
  [PENDING — listening gate]
- Final persona mapping: [PENDING — CV-G1-E0013 freeze into configs/personas.json]

## 7. Samples

Fixed renders ship in `artifacts/caedral-voice-v1/samples/` (eval-corpus items
per voice + long_001.wav + sha256 manifest).

- Sample set: [PENDING — natgate generate --engine caedral-v1]

## 8. Naturalness (GATE-NAT-01, protocol frozen)

Anchors [MEASURED on THIS machine, frozen before any Caedral V1 audio]:
Kokoro DNSMOS ovrl median **3.450** (min 3.197), Piper median **3.382**.
Gate: median >= 3.15 AND min >= 2.85.

| System | DNSMOS med | DNSMOS min | n |
|---|---|---|---|
| caedral-v1 | [PENDING — natgate] | [PENDING — natgate] | 20 |
| kokoro anchor | 3.450 | 3.197 | 20 |
| piper anchor | 3.382 | 3.200 | 10 |

Gate verdict: [PENDING — natgate]. Human-listening confirmation remains
required regardless (NATURAL labels are candidates only).

## 9. WER / CER (intelligibility)

Dual-ASR normalized min-WER (faster-whisper small.en + base.en int8) per eval
spec v0; anchors: Kokoro WER med 0.00 / max 0.33 (known acr_005 hyphen
artifact), Piper med 0.00.

- caedral-v1 WER median/max: [PENDING — natgate scoring stage]
- CER: [PENDING — natgate scoring stage (CER added to scorer output)]
- Long-form worst-chunk WER: see §12.

## 10. Speaker similarity

Per-voice ECAPA-TDNN (Apache-2.0) cosine gen-vs-reference with GT-anchor delta;
drift flag if delta < −0.08 (EVALUATION-BENCHMARKS G4); Resemblyzer cross-check.

- Per-voice sim table: [PENDING — natgate harness extension, ECAPA run]
- Distinctness across the 15 slate voices: [PENDING — same run]

## 11. Prosody

Honest v0 posture: pitch/energy predictors are implemented but frozen OFF in
the E0013 regressor (documented rationale in code); prosody comes from data +
durations, not explicit knobs.

- F0/energy contour fidelity vs references: [PENDING — CV-G1-E0013 final eval]
- Rate plausibility screen (chars/s band): [PENDING — natgate informational]
- Explicit prosody knobs (speed/pitch/energy) status: [PENDING — CV-G1-E0013 decision log]

## 12. Long-form

Protocol: LONGFORM-DESIGN v0 splitter (~<=420 chars/chunk), per-chunk dual-ASR
minWER stability. Baseline anchor on the diagnostic prototype [MEASURED,
CV-G1-E0014]: variance **0.0173 = FAIL** vs gate 0.0025; eff RTF 0.220 @4thr;
localized failures chunks 16/9/24.

| Metric | Diagnostic prototype | caedral-v1 (gate <= 0.0025 variance) |
|---|---|---|
| population variance | 0.0173 [MEASURED] | [PENDING — CV-G1-E0014 re-run on V1] |
| worst-chunk minWER | 0.667 [MEASURED] | [PENDING — CV-G1-E0014 re-run] |
| doc-wide minWER | 0.305 med [MEASURED] | [PENDING — CV-G1-E0014 re-run] |
| eff RTF | 0.220 [MEASURED] | [PENDING — CV-G1-E0014 re-run] |
| trailing-loop / clipping checks | PASS [MEASURED] | [PENDING — CV-G1-E0014 re-run] |

## 13. RTF / RAM / TTFA tables @threads

Fresh-process runs, OMP_NUM_THREADS {1,2,4,default}, utterances conv_001 /
para_001 / long_001. Known hazard from baselines: weak scaling past ~4 threads
and hybrid P/E-core bimodality [MEASURED, BASELINE-MEASURED thread sweep].

| Threads | RTF conv_001 | RTF para_001 | RTF long_001 | Peak RSS MB | TTFA s |
|---|---|---|---|---|---|
| 1 | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] |
| 2 | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] |
| 4 | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] |
| default (16) | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] | [PENDING — threads] |

Baseline context [MEASURED, Kokoro ORT]: 1thr 0.60/0.46, 4thr 0.55/0.47,
default 0.35/0.32; unoptimized eager prototype RTF 0.23@1thr / 0.12@4thr
(E0002 lite-class). TTFA design estimate 0.2–0.6 s [ESTIMATED, RUNTIME-01].

## 14. Size on disk

| Artifact | Size |
|---|---|
| V1 weights (.pt) | [PENDING — CV-G1-E0013 packaging] |
| ONNX graphs (A+B) | [PENDING — RUNTIME-02 export] |
| Full runtime package | [PENDING — runtime packaging] |

Context: Kokoro 337.4 MB, Piper 60.3 MB [MEASURED]; profile caps B <= 2 GB RAM,
C <= 1 GB RAM (MASTER-PLAN §5).

## 15. Comparative benchmark (same machine, same corpus)

Fill row-in-place of `benchmarks/results/natgate/comparison-template.json`
(schema: dnsmos_median/min, wer_median, rtf, ram_mb, size_mb):

| Model | DNSMOS med | DNSMOS min | WER med | RTF | RAM MB | Size MB |
|---|---|---|---|---|---|---|
| caedral-v1 | [PENDING — natgate + threads] | [PENDING — natgate] | [PENDING — natgate] | [PENDING — threads] | [PENDING — threads] | [PENDING — packaging] |
| kokoro-82M-v1.0-onnx | 3.45 | 3.197 | 0.0 | 0.5052 | 707 | 337.4 |
| piper lessac-medium | 3.382 | 3.200 | 0.0 | 0.3027 | 157.4 | 60.3 |
| kittentts-nano | [PENDING — natgate rerun] | — | — | — | — | — |

## 16. Ablations

Planned/targeted ablations (Phase K discipline; add rows as run):

- Teacher-data mix ratio (real-only vs mixed): [PENDING — CV-G1-E0013 ablation arm / A-arm control]
- Sampling sigma: [PENDING — CV-G1-E0013 eval sweep]
- Duration source (CTC vs uniform vs predictor): [PENDING — CV-G1-E0011 x E0013 joint]
- Threads/backend (torch hybrid vs full-ONNX): [PENDING — threads + RUNTIME-02 stage-C decision]
- Others discovered during K-phase: [PENDING — K-phase experiments, IDs not yet assigned]

## 17. Limitations

Known and stated up front (extend, never shrink, at release):

- DNSMOS is a proxy with a ~3.3–3.6 clean ceiling on this stack; only
  same-machine relative comparisons are valid (GATE-NAT-01 caveat).
- Human listening evidence limited until the approval gate runs
  ([PENDING — listening gate]); NATURAL labels are candidates.
- Gender presentation is F0-inferred, no ground-truth labels locally.
- Prosody knobs frozen off in v0 (§11).
- Training scale is CPU-bound; no GPU claims anywhere (charter §15).
- LibriTTS-R identity consent undocumented (MED posture); counsel items open
  (§3).
- Single-machine benchmarking: cross-hardware generalization unproven.

## 18. Negative results (on record, cited honestly)

- E0007/E0009: first trained models unintelligible; overfit gate 4 FAIL
  (WER 0.5+) despite flow_loss −33.94 — root cause analysis in E0009 postmortem.
- E0014: long-form stability FAIL on the diagnostic prototype (variance 0.0173
  vs 0.0025 gate) — the reason the V1 gate exists.
- E0002: tiny-profile RSS hypothesis missed (235 MB vs <= 200 MB; torch floor
  ~215 MB); 4 threads hurt short utterances (up to +52%).
- Thread scaling weak beyond ~4 on this hybrid CPU [MEASURED].
- DATA-FULL-01: clean-100 alone cannot supply >= 60 min/catalog-voice.
- LIC-DEC-01 [F1]/[F2]: original HiFi-GAN checkpoints have NO license
  (PROHIBITED); claimed iSTFTNet2 home repo does not contain it.
- TEACHER-DATA-ENGINE Path B: Piper distillation EXCLUDED (GPL gray).
- Deferred baselines: Chatterbox-Nano (no nano class in released pkg), F5-TTS
  (NC weights) — documented, honest gaps in the comparative table.

## 19. Serving cost estimate

Derived after §13 lands. Formula (to be filled with measured numbers):
cost_per_audio_hour_cpu_s = RTF × 3600 × (vCPU price per thread-second);
concurrency factor per profile from torture-lab matrix (TORTURE-LAB-PLAN).

| Profile (MASTER-PLAN §5) | Cap | Est. audio-hours/day @RTF | Est. $/month | Status |
|---|---|---|---|---|
| A max-quality 4–8 vCPU / <=4GB | — | [PENDING — threads] | [PENDING — threads] | PENDING |
| B economical 2 vCPU / <=2GB | — | [PENDING — threads] | [PENDING — threads] | PENDING |
| C ultra-low 1 vCPU / <=1GB | — | [PENDING — threads] | [PENDING — threads] | PENDING |

## 20. Minimum & recommended hardware

- Benchmark machine [MEASURED]: i7-13620H (6P+4E, 16T), Windows 11, CPU-only,
  torch 2.6.0+cpu / onnxruntime 1.29.0.
- Measured minimum found (published instead of assumed, charter rule 5):
  [PENDING — threads sweep + profile measurements]
- Recommended: [PENDING — threads sweep + profile measurements]

---

### Sign-off checklist (all must be checked before ship)

- [ ] All PENDING markers above resolved with MEASURED values + artifact paths
- [ ] `artifacts/caedral-voice-v1/provenance/release-gates.json`: every gate PASS
- [ ] Independent review protocol executed (`docs/benchmarks/INDEPENDENT-REVIEW-PROTOCOL.md`)
- [ ] LICENSES/ NOTICE.md assembled + pin-time re-verification stamped
- [ ] Counsel open items closed or explicitly waived by founder
