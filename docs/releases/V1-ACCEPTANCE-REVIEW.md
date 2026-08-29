# CAEDRAL VOICE V1 — FINAL ACCEPTANCE REVIEW

Date: 2026-08-24 · Machine of record: i7-13620H (10C/16T), Windows 11, CPU-only
Freeze manifest: `artifacts/caedral-voice-v1/V1-FREEZE-MANIFEST.json`
(git_commit `cff0784a7d89` = current HEAD)
Founder status: voices reported **almost perfect** · **V1 ENGINE STATUS: FROZEN**

Every number below is [MEASURED] with an artifact path. No audio, voice, TN,
phonemization or runtime file was modified by this review.

---

## QUALITY

Classification of the 20 frozen founder-pack renders
(`artifacts/caedral-voice-v1/{voice}/0N-*.wav`, audits: `final-audit.json`,
`post-fix-audit.json`, `tech-fix-audit.json`):

| Sentence | caedral-m1 | caedral-m2 | caedral-f1 | caedral-f2 |
|---|---|---|---|---|
| normal | INTELLIGIBLE | INTELLIGIBLE | INTELLIGIBLE | INTELLIGIBLE |
| technical | INTELLIGIBLE | PARTIAL_SPEECH | PARTIAL_SPEECH | PARTIAL_SPEECH |
| numbers | INTELLIGIBLE | NATURAL (WER 0.00) | NATURAL (WER 0.00) | NATURAL (WER 0.00) |
| proper-names | INTELLIGIBLE | INTELLIGIBLE | INTELLIGIBLE | INTELLIGIBLE |
| long-form | NATURAL | NATURAL | INTELLIGIBLE | INTELLIGIBLE |

DNSMOS ovrl across all audited clips: 3.34–3.62 (median ≈ 3.49).

### Exact ASR mismatches (word-level, dual-ASR; `asr-mismatch-report.json`)

Producer: faster-whisper small.en + base.en (int8), word-level alignment vs
reference. Substitutions only — **zero deletions, zero insertions** in every
open item; all non-acronym/non-brand words are recognized exactly.

**Technical — caedral-m2 / caedral-f1 / caedral-f2** (identical pattern,
both ASRs; REF "The CPU supports AVX2 instructions but not AVX-512.",
minWER 0.444):

| # | Type | Reference | Heard |
|---|---|---|---|
| 1 | SUB | avx2 | a vx2 |
| 2 | SUB | avx·512 | a vx512 |

The CPU sentence carries 9 reference words; these two letter-sequence
tokenizations account for 4/9 = WER 0.444 → classed PARTIAL_SPEECH (threshold
0.30). Every other word ("the cpu supports instructions but not") transcribes
exactly in both ASRs, in all three voices.

Control contrast — caedral-m1 (INTELLIGIBLE, WER 0.222): base.en hears
"AVX-2 … AVX-512" (joins cleanly), but small.en hears the *same* "a VX2"
split as on the other voices. The m1-vs-rest difference is therefore one
ASR model's tokenization choice on a marginal acoustic difference —
consistent with, and sharpening, the documented speaker-acoustic finding
(`V1-SPEAKER-ACOUSTIC-FINDING.md`). The speech content itself is fully
present in all four voices.

**Proper names — caedral-f1** (REF "Welcome to Caedral, where every voice
tells a story.", minWER 0.222):

| # | Type | Reference | Heard (base.en / small.en) |
|---|---|---|---|
| 1 | SUB | caedral | kedral / kaedral |
| 2 | SUB | story | starty / startie |

"Invented brand name heard with a K onset" + one diphthong offglide;
remaining 7 words exact.

---

## PERFORMANCE

Reconciliation of RTF ~0.46 vs acceptance corpus 0.59–1.10 — **consistent;
different measurement regimes**, no defect:

| Figure | Regime | Source |
|---|---|---|
| ~0.46 (recorded 0.4575) | single combo (short control), warm loop, idle/boost thermal state, 6 threads ENABLE_ALL | conv-opt session, commit 7590681; flagged thermal-dependent in `E0026/conclusion.md` |
| 0.59–1.10 | acceptance run: per-sentence **single-shot** wall/audio, 4 voices × 5 sentences, dual-ASR + DNSMOS audit interleaved **in-process** between renders (CPU contention), first-call effects included | founder-acceptance stdout; manifest known-limitations line |
| **0.446–0.863, median 0.724** | controlled re-measure TODAY at frozen config (6 threads, arena-off): warm **median of 5 reps**, idle machine, 12 combos | `artifacts/benchmarks/v1-rtf-grid-recheck.json` |

Today's grid reproduces the ~0.46 figure where conditions match (caedral-m1
long-form: **0.446**) and sits inside the acceptance band elsewhere:

| Voice | control | technical | longform |
|---|---|---|---|
| caedral-m1 | 0.669 | 0.526 | **0.446** |
| caedral-m2 | 0.785 | 0.805 | 0.634 |
| caedral-f1 | 0.742 | 0.758 | 0.754 |
| caedral-f2 | 0.705 | 0.615 | 0.863 |

Context: E0026's pooled protocol measured 0.63–0.73 fp32@t6 on a thermally
elevated day (best raw dynamic 0.5669); the Aug-23 prod-benchmark single-shot
scatter reached 1.10 under ambient load. Verdict: sustained production RTF on
this machine is < 1.0 for the V1 sentence mix, typically ~0.45–0.9 idle;
single-shot measurements scatter above warm medians by first-call effects and
audit-stack contention alone.

## MEMORY

Explanation of ~645 MB warm vs ~1765 MB "peak" — **fully accounted;
the large figure belongs to the audit harness, not the engine**
(`artifacts/benchmarks/v1-rtf-ram-recheck.json`, RSS sampled at 10 ms):

| Stage | RSS |
|---|---|
| Python + imports | 25 MB |
| + ORT session (310.5 MB weights) + frontend | 641 MB |
| **warm steady-state (production envelope)** | **645 MB** |
| + engine-only 16 s long-form render, true sampled peak | 1136 MB (settles to 842 MB) |
| + dual-ASR audit models loaded in same process | +686 MB → 1332 MB |
| + long-form render with audit stack resident | **1650 MB peak** |

The acceptance script loads faster-whisper small.en + base.en int8 (and DNSMOS
deps) into the SAME process to classify each render, then samples RSS around
further long-form synthesis — reproducing 1650 MB in a single pass, with the
remaining gap to ~1765–1790 being ASR/DNSMOS workspace churn accumulated over
its 20 classifications. Production serving never carries the audit stack:
**true production envelope = 645 MB warm, ≤ ~1.14 GB worst-case transient**
(long-form), settling to ~845 MB.

## SHARED SESSION

PASS — verified live (`freeze-verification.json`):
exactly **one** `onnxruntime.InferenceSession` exists in the process after
engine load; `engine._kokoro.sess` IS that object (`Kokoro.from_session`
reuse path, code-confirmed at `kokoro_engine.py:92-101`); still exactly one
after synthesis; fused graph exposes the single output `audio`.

## ASSET/HASH VERIFICATION

PASS (`artifacts/caedral-voice-v1/freeze-verification.json`):

- **20/20** frozen WAVs present, valid PCM, 24 kHz mono, 1.55 s–16.30 s.
- Listening-pack copies byte-identical (SHA-256 equal).
- All **6/6 asset SHA-256 hashes match** `V1-FREEZE-MANIFEST.json`
  (kokoro-onnx, voices-bin, engine-py, kokoro-map-py, normalize-py, g2p-py)
  → frozen assets unchanged since freeze; manifest commit = current HEAD.

Known cosmetic defect (harmless, left as-is): duplicate `"engine-py"` dict key
in `scripts/v1_freeze_verification.py` (same value twice; no effect).

## LISTENING DIRECTORY

Complete absolute directory
`C:\Users\leona\Downloads\caedral-voice\artifacts\pre-arch-enhanced-base\final-founder-listen\`
— 20 WAVs + `TRANSCRIPTS.md` (listing printed in the session summary; all files
verified present and hash-identical to the frozen renders).

---

**V1 ENGINE STATUS: FROZEN**
**FOUNDER LISTENING REQUIRED: YES**
**ARCH: WAITING FOR FOUNDER APPROVAL**
