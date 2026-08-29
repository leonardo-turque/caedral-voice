# CV-G1-E0016 — ANCHOR-QEX: missing teacher anchors qex_002..qex_006 (render + score + gate validation)

**Task:** ANCHOR-QEX · **Date:** 2026-08-22 · **Status:** COMPLETE
**Prerequisite for:** `docs/benchmarks/PROSODY-EVAL.md` Tier C gating of the full
qex set (§1 flagged that anchor clips covered `qex_001` only).

## What was done

Rendered the five missing `questions_exclamations` items with the
NATURALNESS-GATE teacher systems, following `benchmarks/run_natgate.py` exactly:

- **Kokoro-82M v1.0** (Apache-2.0, in-process kokoro-onnx): voices `af_heart`,
  `am_michael`, defaults `speed=1.0 lang=en-us`, same headroom rule (no clip
  needed gain; peaks 0.44–0.80).
- **Piper en_US-lessac-medium + en_US-ryan-medium** (GPL-3.0): **isolated
  piper.exe subprocess per synthesis**, black-box child only, cold+warm per
  clip, warm clip saved (identical to `run_natgate.py`).
- RAM policy honored: engines ran as separate sequential processes (Kokoro
  process exited before any Piper child spawned); scoring ran as its own
  process (dual-ASR) and probes as their own (librosa/pyin). Total wall
  ~9 min, well inside the 25 min budget.

20 new clips → `audio/<teacher>-<voice>-<item>.wav`; scored with
`scripts/classify_preview.py` signals (dual-ASR minWER, DNSMOS ovrl via
speechmos, rate) + PROSODY-EVAL §2 detectors (P1 rise flag B, joint C logged,
P3 pause accounting) implemented inline from the doc's formulas (no in-repo
implementation existed). E/L expected-pause budgets came from the **real
frontend** (`caedral_voice.text.g2p`), not a re-implementation.

**Code-path validation:** re-measuring the 3 legacy `qex_001` natgate anchors
with this probe implementation reproduces PROSODY-EVAL §7 to 3 decimals
(B = 1.169 / 1.122 / 1.256) — identical measurement path confirmed (§8.5).

## Per-anchor results (20 new + 3 legacy qex_001, [MEASURED])

minWER = dual-ASR min; DNS = DNSMOS ovrl; B = rise flag (threshold 1.15);
Dir = P1 direction-table verdict per item expectation (req = rise required,
forb = rise forbidden, – = n/a exclamation); P3 = upper-bound pause rule.

| Clip (teacher-voice-item) | dur s | minWER | DNS | B | Dir | E/L | P150 | maxSil ms | P3 |
|---|---|---|---|---|---|---|---|---|---|
| kokoro-af_heart-qex_002 | 1.90 | 0.00 | 3.34 | 1.139 | forb ok | 0/0 | 0 | 60 | PASS |
| kokoro-af_heart-qex_003 | 1.84 | 0.00 | 3.42 | 1.122 | **req FAIL** | 1/0 | 0 | 130 | PASS |
| kokoro-af_heart-qex_004 | 1.24 | 0.00 | 3.34 | 1.084 | – | 0/0 | 0 | 40 | PASS |
| kokoro-af_heart-qex_005 | 2.39 | 0.00 | 3.42 | 1.238 | – | 4/1 | 0 | 70 | **FAIL** (no ≥200 ms after ‖) |
| kokoro-af_heart-qex_006 | 3.14 | 0.00 | 3.41 | 1.047 | forb ok | 0/0 | 1 | 210 | PASS |
| kokoro-am_michael-qex_002 | 2.20 | 0.00 | 3.38 | None (vfrac 0.39) | – | 0/0 | 0 | 50 | PASS |
| kokoro-am_michael-qex_003 | 2.24 | 0.00 | 3.15 | 1.135 | **req FAIL** | 1/0 | 0 | 100 | PASS |
| kokoro-am_michael-qex_004 | 1.54 | 0.00 | 3.29 | 1.282 | – | 0/0 | 0 | 30 | PASS |
| kokoro-am_michael-qex_005 | 2.77 | 0.00 | 3.39 | 1.122 | – | 4/1 | 0 | 130 | **FAIL** (no ≥200 ms after ‖) |
| kokoro-am_michael-qex_006 | 3.76 | 0.00 | 3.46 | 0.977 | forb ok | 0/0 | 1 | 180 | PASS |
| piper-lessac-qex_002 | 2.15 | 0.00 | 3.26 | 1.078 | forb ok | 0/0 | 0 | 90 | PASS |
| piper-lessac-qex_003 | 2.33 | 0.00 | 3.11 | 1.473 | req ok | 1/0 | 1 | 249 | PASS |
| piper-lessac-qex_004 | 1.26 | 0.00 | 2.58 | 1.182 | – | 0/0 | 0 | 40 | PASS |
| piper-lessac-qex_005 | 3.78 | 0.00 | 3.13 | 1.312 | – | 4/1 | 3 | 429 | PASS |
| piper-lessac-qex_006 | 3.20 | 0.00 | 3.19 | **1.228** | **forb FAIL** | 0/0 | 0 | 50 | PASS |
| piper-ryan-qex_002 | 1.76 | 0.00 | 3.14 | **1.224** | **forb FAIL** | 0/0 | 0 | 50 | PASS |
| piper-ryan-qex_003 | 1.91 | 0.00 | 2.99 | 1.615 | req ok | 1/0 | 1 | 259 | PASS |
| piper-ryan-qex_004 | 1.15 | 0.00 | 2.49 | 1.145 | – | 0/0 | 0 | 20 | PASS |
| piper-ryan-qex_005 | 2.87 | 0.22 | 2.91 | 1.282 | – | 4/1 | 2 | 199.5 | **FAIL** (199.5 < 200 ms) |
| piper-ryan-qex_006 | 3.03 | 0.00 | 3.28 | **1.203** | **forb FAIL** | 0/0 | 0 | 70 | PASS |
| *(legacy)* natgate-kokoro-af_heart-qex_001 | 2.80 | 0.00 | 3.28 | 1.169 | req ok | 0/0 | 0 | 40 | PASS |
| *(legacy)* natgate-kokoro-am_michael-qex_001 | 3.46 | 0.00 | 3.20 | 1.122 | **req FAIL** | 0/0 | 0 | 40 | PASS |
| *(legacy)* natgate-piper-lessac-qex_001 | 3.65 | 0.00 | 3.40 | 1.256 | req ok | 0/0 | 0 | 50 | PASS |

Per-voice DNSMOS ovrl medians (new clips): af_heart **3.41**, am_michael **3.38**,
lessac **3.13**, ryan **2.99**. minWER 0.00 everywhere except
piper-ryan-qex_005 (0.22). Rate plausible 23/23. Joint C-slope spread on
single items spans −26.0 … +23.4 st/s — reconfirms §2: **C must never gate**.

## GATE VALIDATION — verdicts

Mission step 4: do the teachers pass the gates PROSODY-EVAL proposes?

### FLAGGED — P1 direction table fails on teachers (recalibrate before gating)

Teachers comply with their own normative direction table on only **8 of 14**
measurable voice-item checks:

- **qex_003 (tag question, rise REQUIRED): 2/4** — both Kokoro voices flat
  (B = 1.122 / 1.135, just under 1.15) while both Piper voices rise strongly
  (1.473 / 1.615). The absolute threshold B ≥ 1.15 splits the teacher families.
- **qex_006 (wh-question, rise FORBIDDEN): 2/4 FAIL** — both Piper voices fire
  the rise flag (1.228 / 1.203) on an item the table says must fall.
- **qex_002 (wh, FORBIDDEN): ryan fires** (1.224); 2/3 measurable comply.
- **qex_001 (REQUIRED): 2/3** — am_michael 1.122 (already known from §7).

The V1 decision rule (§4) demands "≥5 of 6 voice-item checks" from *our*
system per rule direction. Teachers themselves do not reach that bar on the
full qex set (8/14 ≈ 57%). **An absolute-B direction gate at 1.15 is not
usable as-is for qex_002/003/006** — consistent with §8.1–8.3 limits
(n = 18 literary calibration clips, conversational domain shift, wh-questions
canonically falling while Piper inserts phrase-final rises). Recalibration
options to evaluate: per-item-type thresholds, family-relative (Tier C style)
direction checks, or direction-gating only where all teachers agree.

### FLAGGED — Tier C band X = 0.20 is violated by an anchor itself

- Formal rule `|B − median(B_anchor,item)| ≤ 0.20`: **piper-ryan on qex_003
  deviates 0.311** from the anchor median 1.304 — a teacher outside its own
  proposed band. All other items pass (max dev ≤ 0.148).
- Spread derivation basis (§3 derived X from max observed anchor spread):
  qex_003 spread **0.493** and qex_006 spread **0.250** exceed X = 0.20.
- Legacy qex_001 re-check: spread 0.134, max dev 0.088 — matches §3 exactly.
- Root pattern: Piper voices produce much larger terminal F0 movement than
  Kokoro on questions; a single global X cannot span both families on
  rise-realized items. **X needs per-item or per-family recalibration before
  Tier C gates qex_003/qex_006.**

### FLAGGED — P3 "(L ≥ 1 ⇒ pause ≥ 200 ms)" clause fails 3/4 teachers on qex_005

`qex_005` ("Get out! No — wait, come back, I was joking!") is the only new
item with a long mark (L = 1). Realized max internal silence after the `‖`:
lessac 429 ms (PASS), ryan 199.5 ms (**fails by 0.5 ms**), am_michael 130 ms,
af_heart 70 ms. The §2 anchor evidence behind this clause came from 30 natgate
clips that did **not** include qex_005 (only qex_001 was rendered there), so
this failure mode was never measured. Arguably a rushed exclamatory
continuation is *correct* expressive prosody — the clause needs an
expressive-context exemption or a lower floor before gating EVAL-QEX. All
other P3 sub-rules pass teachers everywhere (P150 ≤ E+2 and ≤700 ms hold on
23/23 clips; the two Kokoro mid-sentence breaths on qex_006 are within
budget).

### PASSES

- **Intelligibility/quality sanity:** all 23 clips minWER ≤ 0.22 (21/23 at
  0.00), DNSMOS ovrl 2.49–3.46, no NOISE/BABBLE labels — anchors are healthy
  references for the new items.
- **P3 core upper bound** (P150 ≤ E+2, no >700 ms internal gap): 23/23.
- **Probe implementation** reproduces §7 calibration exactly.

## Consequence for the program (honesty rules)

PROSODY-EVAL §4's automated decision rule **cannot gate qex_002..qex_006 as
written**: two of its three quantitative legs (P1 absolute direction, Tier C
X = 0.20) are failed by the teacher anchors themselves on the newly rendered
items, and the third (P3) has one clause that 3/4 teachers fail on qex_005.
Until recalibrated, Tier C should keep logging qex_002–006 and gate only
qex_001 (where anchors pass band and P3; direction already known 2/3). The
clips and numbers here are the recalibration dataset. This document flags;
it does **not** modify the frozen protocol (owner: PROSODY-EVAL-01).

## Artifacts

- `audio/` — 20 clips (`<teacher>-<voice>-<item>.wav`)
- `generate.py`, `gen-meta-kokoro.json`, `gen-meta-piper.json` — generation + provenance
- `references.json`, `scores.json` — classify_preview signals (separate process)
- `probe_prosody.py`, `prosody.json` — P1/P3 detectors (23 clips incl. legacy)
- `aggregate.py`, `results.json` — merged table + gate-validation verdicts
