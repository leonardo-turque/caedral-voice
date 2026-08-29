# Prosody Evaluation Protocol — Caedral V1 Candidates (PROSODY-EVAL)

Task: **PROSODY-EVAL-01** · Created: 2026-08-22 · Status: ACTIVE (detectors calibrated on real audio same day)

> **RECALIBRATION NOTICE (2026-08-23, E0016):** teacher anchor renders for
> qex_002..006 (Kokoro ×2 + Piper ×2) FAIL the proposed P1 direction table
> (8/14 compliance), the Tier-C band on qex_003/qex_006, and the P3
> exclamation-pause clause (3/4 voices). A gate that anchors cannot pass is
> invalid: **Tier C stays LOG-ONLY for qex_002..006** and gates only qex_001
> until recalibrated against these measured anchors.
> Source: experiments/CV-G1-E0016-qex-anchors/results.json
Owner file: this document only. It does **not** modify the frozen NATURALNESS-GATE, the
INDEPENDENT-REVIEW-PROTOCOL, or the frozen corpus (`benchmarks/corpus/eval-corpus-v1.0.json`).

Honesty frame (from `docs/research/EVALUATION-BENCHMARKS.md` §1/§4): **no automated
metric resolves prosody.** MOS-style predictors are measurably prosody-blind; F0 RMSE-class
metrics need owned reference takes we do not have. Everything below is therefore a
*detector* layer that catches gross failures and anchors comparisons — the human tier
(§5) remains the only instrument that can pass or fail expressive quality. Automated
prosody signals here are **necessary-context and regression alarms, never sufficient**.

Six prosody axes under evaluation: **rhythm, stress, pauses, questions, exclamations,
intonation.** Mapping to probes and to the human checklist is in §6.

Environment for all [MEASURED] numbers: this repo @ git `06aaa43`, Windows 11,
i7-13620H, venv numpy 2.5.2 / librosa 0.11.0 / soundfile. Audio: 22.05 kHz mono.

---

## 1. Clip sets

| Set | Contents | Source |
|---|---|---|
| EVAL-QEX | all 6 `qex_*` items × 3 voices (one female, one male, one reserve — same voice slots every run, recorded in provenance) | our system renders |
| EVAL-PUN | all 5 `pun_*` items × same 3 voices | our system renders |
| EVAL-EMO | all 8 `emo_*` items × same 3 voices, each rendered in its `suggested_style` **and** in `neutral` style (same text) | our system renders |
| ANCHOR-QEX | same qex items × teacher anchors: Kokoro `af_heart`, `am_michael`; Piper `lessac` (the NATURALNESS-GATE anchor systems) | `benchmarks/results/natgate/` — **existing clips cover `qex_001` only; rendering qex_002–006 + conv_001-style items for the anchors is a prerequisite task for the natgate owner before Tier C can gate the full qex set** |
| CAL-LIBRI | 6 polar questions + 6 wh-questions + 6 statements from LibriTTS-R clean100 (§7) — calibration only, not a gate input | `data/manifests/librittsr-clean100.jsonl` |

## 2. Tier O — objective probes (runnable, CPU-only)

Shared front-end for every probe: load WAV → mono float → trim leading/trailing
silence with `librosa.effects.split(top_db=40)` → `librosa.pyin(fmin=60, fmax=400,
frame_length=1024, hop_length=256)` (≈11.6 ms frames). Measured cost: **2–4 s per
clip** on this machine (no subsampling needed; full-budget run ≈ 57 clips × 3 s ≈
3 min, well inside the <20 min task budget).

### P1 — Question-final F0 rise (axis: questions)

Calibration on 18 real LibriTTS clips (§7) **rejected the originally specified
detector** (median F0 of final 500 ms ≥ 1.08 × utterance median): it fired on
**1/6** real polar questions [MEASURED] — the terminal rise occupies only the
last 1–2 accented syllables, so a 500 ms *median* dilutes it — and it also fails
all three teacher anchors on `qex_001` (ratios 0.84 / 0.93 / 0.93) and `conv_001`
(0.86 / 0.81 / 0.87). A gate our teachers cannot pass is not a gate.

**Adopted detector (rise flag):** `B = max(F0 in final 700 ms) / median(F0 over
utterance)`; rise present iff `B ≥ 1.15` [MEASURED 2026-08-22, CAL-LIBRI:
polar questions 6/6 hits; wh-questions 3/6 (the 3 fires are echoic/emotive
readings); statements 3/6 false-fires — see limits below].

**Joint diagnostic (logged, not gated):** terminal slope `C` = linear fit of F0 in
semitones (rel. utterance median) over final 800 ms voiced frames, st/s. Joint rule
`B ≥ 1.15 AND C ≥ +2.0 st/s` → 5/6 polar, 2/6 statement FP, 2/6 wh [MEASURED].
C is strongly voice-dependent (teacher anchors on `qex_001`: +2.3 / −6.3 / −7.2 st/s
— spread 9.5 st/s), so **C must never gate**; a C gate would fail teachers.

**Expectations per item** (English intonation phonology, verified on CAL-LIBRI where
all 6 wh-items fell or stayed flat on C except echoic ones):
rise REQUIRED on `qex_001`, `qex_003` (tag question); rise FORBIDDEN (flat/fall,
`B < 1.15` expected) on `qex_002`, `qex_006` (wh-questions); not applicable on
`qex_004`, `qex_005` (exclamations → P3).

### P2 — Exclamation energy boost (axis: exclamations)

For `qex_004`, `qex_005`, and `emo_002` (excited) vs the neutral-style render of the
identical text: `ΔRMS = 10·log10(mean square over voiced frames)_styled −
(...)_neutral`. Pass: `ΔRMS ≥ +2.0 dB` on ≥ 2 of 3 voices [PROVISIONAL — not
calibratable today because it needs our own styled renders; the +2 dB figure is the
task-specified design value, to be re-derived from anchor spread once teacher
multi-style clips exist]. Log the distribution; a *negative* Δ on an excited-style
item is a hard fail regardless of threshold.

### P3 — Pause accounting at punctuation boundaries (axes: pauses, rhythm)

Ground truth comes from **our frontend token stream** (`caedral_voice/text/g2p.py`:
`,;:` → short `|`, `.!?` → long `‖`, dashes/quotes folded silently, trailing pause
stripped). For each clip: expected internal boundaries `E = count(|)+count(‖)` from
the stream, `L = count(‖)`.

Measurement: 10 ms frames, RMS dB; silence = frame < max(90th-percentile-frame-dB − 35,
−50 dBFS); count internal runs (leading/trailing excluded) ≥ 150 ms (`P150`) and
≥ 350 ms (`P350`); record all run durations.

**Pass rule (upper-bound only — deliberately asymmetric):** `P150 ≤ E + 2` AND no
internal silence > 700 ms AND (`L ≥ 1` ⇒ at least one pause ≥ 200 ms exists).
[MEASURED anchor behavior: Piper 10/10 clips within budget; Kokoro 6/10 — it
breathes mid-sentence on unpunctuated long technical/formal items, overages of
+1…+3, max internal silence 560 ms; no teacher ever exceeded +3 or 700 ms.
Requiring every comma to be realized as a ≥150 ms gap fails even teachers
(e.g. conv_001: all three anchors realize only 1 of 2 comma pauses as ≥150 ms).]
Missing pauses (P150 = 0 where E ≥ 2) is logged as rhythm-flatness evidence for
the human tier, not an auto-fail.

### P4 — Speech-rate variance across styles (axis: rhythm)

`rate = (frontend word count) / trimmed duration` per EVAL-EMO clip. Metric:
coefficient of variation of rate across the 8 emo styles **within one voice**.
Pass: `CV ≥ 0.05` for ≥ 2 of 3 voices [PROVISIONAL — anchor reference range from
existing single-style natgate clips is 1.6–2.1 words/s (tech_001) vs 3.3–3.9
words/s (emo_001), i.e. content alone spans ~2×, so the gate is deliberately
loose and within-voice; re-derive once multi-style anchors exist]. Log-only in V1
if the styled renders are not ready.

### Runtime

Full Tier O pass ≈ 57 our-clips + 18 anchor clips ≈ 4–5 min CPU. No GPU, no network.

## 3. Tier C — comparative vs teacher anchors

Same probes (P1 flag value `B`, P3 counts, P4 rate) run on ANCHOR-QEX clips.
Because absolute values are voice- and item-dependent, comparison is **per item,
against the per-item anchor median**, using voice-normalized relative metrics only
(B is already relative; never compare raw Hz across genders — anchor F0 medians
measured 104–205 Hz).

**Tolerance X:** measured anchor spread on the two question items rendered by all
three anchors: `B` spread 0.134 (qex_001) and 0.123 (conv_001) [MEASURED].
**X = 0.20** (max observed spread rounded up with margin). Pass: `|B_ours −
median(B_anchor,item)| ≤ 0.20` on each rise-required item, and P3 upper-bound rule
satisfied by both our clips and (sanity) the anchors themselves.

Prerequisite before this tier gates qex_002–006: extend natgate renders (owner:
natgate task, `benchmarks/run_natgate.py`); until then Tier C gates `qex_001` only
and logs the rest.

## 4. Decision rule for V1

```
prosody PASS  ⇔  (P1: rise present on qex_001+qex_003, absent on qex_002+qex_006,
                   ≥ 5 of 6 voice-item checks per rule direction)
              AND (P3 upper-bound rule on all EVAL-QEX + EVAL-PUN clips, ≥ 90% clips)
              AND (Tier C band |ΔB| ≤ 0.20 on qex_001 vs anchor median)
              AND (P2, P4: no hard-fail condition; provisional thresholds logged)
```
Prosody PASS is **input to** the human tier and the REVIEW-PROTOCOL verdict, never a
substitute: an automated pass with a human FAIL on A4/A6 is a FAIL. An automated
prosody FAIL forces a targeted re-listen; it does not by itself reject a candidate
(detector specificity measured at only 4/6 on literary statements — see §7 limits).

## 5. Tier H — human checklist mapping (axes ↔ REVIEW-PROTOCOL)

The protocol is frozen; axes are judged **inside its existing items**, with this
mapping supplied to reviewers in the assignment brief:

| Axis | Protocol item | What the reviewer listens for |
|---|---|---|
| Rhythm | A6 (+ A3d choppiness) | word-grouping cadence fits syntax; list/parenthetical timing; no staccato splicing |
| Stress | A5 | wrongly stressed words listed per A5 rule (quote + escalate on meaning change) |
| Pauses | A4 | mid-word gaps, dead air, >1 s intra-sentence breaks (A4 defaults) |
| Questions | A6 | rise on yes/no & tag questions, fall on wh-questions (qex_001/002/003/006 are in the packet's novel-sentence pool) |
| Exclamations | A6 | energy/emphasis commensurate with `!`; no shouted monotone |
| Intonation | A6 | contour contradicts punctuation = FAIL per A6; flat-monotone through expressive content = FAIL per A6 |

Long-form (`long_001`, mandatory voice-01 paragraph) is additionally checked for
rate/prosody drift under A6 evidence rules. Reviewers may run the §2 probes
themselves (read-only posture permits analysis scripts) but every verdict still
needs a timestamped quote per protocol §4.

## 6. Axis → probe summary

| Axis | Objective | Comparative | Human |
|---|---|---|---|
| questions | P1 rise flag B (± joint C log) | Tier C band ±0.20 | A6 |
| exclamations | P2 ΔRMS ≥ +2 dB vs neutral same-text | — (no styled anchors yet) | A6 |
| pauses | P3 P150/P350 vs frontend stream budget | anchor sanity | A4 |
| rhythm | P4 rate CV + P3 missing-pause log | anchor rate range | A6, long-form drift |
| stress | — (no reliable detector; honest gap) | — | A5 only |
| intonation | P1 direction checks (rise/fall per item type) | Tier C | A6 |

## 7. Calibration record [MEASURED 2026-08-22]

Method: LibriTTS-R clean100 manifest (30,917 rows) carries terminal punctuation in
transcripts; deterministic first-match-by-id selection, duration 2–10 s, single
sentence (no internal `.?!`). **Polar-Q** (auxiliary-initial): 1034_121119_000064_000002,
1034_121119_000064_000003, 1088_129236_000028_000005, 1116_137572_000001_000005,
1116_137572_000030_000005, 118_47824_000035_000004. **Wh-Q (controls):**
1088_129236_000008_000002, 1116_137572_000030_000007, 1183_133255_000044_000000,
1363_139304_000009_000001, 1502_122619_000045_000001, 1594_135914_000005_000003.
**Statements:** 1034_121119_000010_000004, 1034_121119_000010_000006,
1034_121119_000028_000001, 1034_121119_000052_000000, 1034_121119_000064_000006,
1034_121119_000065_000003.

Detector scores on these 18 clips:

| Rule | Polar hit | Wh fire | Statement FP |
|---|---|---|---|
| A: median(final 500 ms) ≥ 1.08 × median | 1/6 | 2/5* | 0/6 |
| B: max(final 700 ms) ≥ 1.15 × median | **6/6** | 3/6 | 3/6 |
| C: slope ≥ +2 st/s | 5/6 | 2/6 | 2/6 |
| B ∧ C | 5/6 | 2/6 | 2/6 |

\* one wh-clip had <3 voiced frames in the final window (None).

Anchor measurements (30 natgate clips): B on qex_001 = 1.169 / 1.122 / 1.256
(kokoro af_heart / am_michael, piper lessac); on conv_001 = 1.129 / 1.006 / 1.017.
P3 budgets: see §2 P3. pyin runtime 1.5–4.1 s/clip.

## 8. Known limits (do not hide these in reports)

1. **n = 18 calibration clips, literary LibriTTS domain** — hit rates carry wide
   uncertainty (±1 clip ≈ ±17%); qex items are conversational. Re-calibrate if a
   gate decision ever comes down to a single borderline clip.
2. Statement FP rate 2–3/6: long multi-phrase sentences put phrase-final rises in
   the terminal window. P1 is valid only on single-sentence clips (all EVAL-QEX
   items qualify; pun_002-style embedded quotes do not — P1 skipped there).
3. Wh-questions canonically fall; treating "question ⇒ rise" universally would
   penalize correct TTS. Direction table in §2 P1 is normative.
4. pyin octave errors and creaky voice can distort B/C; any clip with voiced
   fraction < 0.30 is reported but excluded from gating.
5. Energy gate (−35 dB rel. speech percentile) is sensitive to noise floor; anchors
   and candidates must be measured with the identical code path.
6. P2/P4 thresholds are provisional by construction (no styled teacher anchors
   exist yet); both are marked, and neither can alone reject a candidate.
