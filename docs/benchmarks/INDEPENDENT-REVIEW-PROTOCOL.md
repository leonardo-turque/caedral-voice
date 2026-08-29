# Independent Review Protocol — Caedral V1 Candidates (REVIEW-PROTOCOL)

Task: **REVIEW-PROTOCOL** · Created: 2026-08-22 · Status: ACTIVE
Scope-defining rule (founder): **implementers never self-approve.** No V1 voice
or V1 system passes because its builder believes it works. Every approval below
is issued by reviewers who did not build the thing and whose mandate is to
REJECT it.

Relationship to the frozen gate: the numeric gate in
`docs/benchmarks/NATURALNESS-GATE.md` (DNSMOS ovrl median ≥ 3.15, min ≥ 2.85,
anchor K = 3.45) is **frozen and untouched** by this document. This protocol is
the human-proxy listening layer on top of it. Both layers must pass; neither
substitutes for the other. Where numbers appear here they are quoted from the
gate for calibration only.

Reviewers are fresh Claude subagent instances acting as human-proxy listeners;
they may use any local analysis tooling (playback, spectrograms, ASR, F0
extraction) but every claim must be grounded in a timestamped, quotable
observation.

---

## 1. Reviewer independence rules

1. **No self-review.** A reviewer must not have built, trained, fine-tuned,
   debugged, diagnosed, or produced samples for the candidate under review.
   Authorship is checked against `experiments/EXPERIMENT-INDEX.md` and the
   candidate's experiment folder attribution. An implementer who touches the
   candidate is disqualified from reviewing it — permanently for that candidate.
2. **Fresh instances, zero shared context.** Each reviewer is a newly spawned
   subagent with **no conversation history, memory, notes, or session state
   shared with the implementing agent or with each other**. The reviewer
   receives only: this document, the packet directory (§2), and a neutral
   assignment brief. No implementation-session transcripts, no conclusions.md,
   no interpretive notes from the builders travel with the packet.
3. **Adversarial mandate, stated in the brief:** *"Your job is to find reasons
   to REJECT this candidate. You are not helping the team ship; you are trying
   to keep a defective voice set out of the product. Approval requires positive
   evidence on every checklist item; doubt resolves to FAIL."*
4. **Read-only posture.** Reviewers may run analysis scripts and write only
   into `benchmarks/results/review/<candidate>/reports/`. They may not modify
   the candidate, retrain, re-render audio, edit packets, or touch source code.
5. **No collusion.** Reviewers R1/R2 never see each other's reports before both
   verdicts are filed. Reports live in separate files and are opened together
   only at aggregation time (§7).
6. **Anti-shopping cap.** A candidate gets at most **two** full review cycles.
   A third attempt requires a new candidate ID from a different checkpoint.
   Resubmission reports must state what changed between rounds.

## 2. Review packet contents

Built by the coordinator (not the reviewer) at
`benchmarks/results/review/<candidate-id>/packet/`:

| Item | Spec |
|---|---|
| 15 × 3 voice WAVs | `voices/voice-NN-{evalA,evalB,novel}.wav` for NN=01..15. evalA/evalB are the fixed preview sentences (`Hello. This is the first generation of Caedral Voice.` / `Artificial intelligence should sound natural, expressive, and human.`). **novel** = one frozen-corpus sentence per voice, assigned deterministically *before rendering*: voice NN takes the ((NN−1) mod 10)-th sentence of the NATURALNESS-GATE §Protocol list (conv_001, num_001, acr_005, het_002, pho_001, qex_001, frm_001, nm_001, tech_001, emo_001). The mapping is recorded in provenance.json so the implementer cannot cherry-pick flattering sentences. |
| Long-form paragraph | `longform/voice-01-para_001.wav` — `para_001` from `benchmarks/corpus/eval-corpus-v1.0.json`, rendered by the narrator-slot voice-01 (mandatory; additional voices optional, listed in manifest). |
| provenance.json | Git commit; checkpoint path + sha256; model config; voice→training-speaker map (e.g. voice-06 ↔ LibriTTS-R 1088); decoding params (sigma etc.); headroom `gain_applied` per clip (gate §Protocol.3 rule); sha256 of every WAV; novel-sentence map; render commands + UTC timestamps; packet-manifest hash. |
| persona-labels.json | Per voice, straight from `configs/voices/<voice_id>.json` / the VOICE-SLATE-V1 table: `gender_presentation` + target character (grave/deep, calm documentary, warm-bright conversational, …). No measurements, no risk notes — the reviewer judges against the label alone. |
| metrics.json — **SEALED** | Per-clip `classify_preview.py` output (class, dual-ASR min-WER, DNSMOS sig/bak/ovrl, rate, repetition flags) and pooled stats. Sealed = the reviewer must complete ALL of §4 Section A before opening it. Reading it early voids the review; the report must carry the attestation line. |

Packet manifest: `packet-manifest.json` lists every file with sha256; a packet
modified after first review round opens is a protocol violation logged in
VERDICT.md.

## 3. Listening order

1. Read this protocol + persona-labels.json. Do NOT open metrics.json.
2. Section A per voice (all 15), then long-form, then Section B.
3. Only then open metrics.json and write the corroboration note.
4. File the report. Verdicts are final once filed except via §7.

## 4. Checklist

Every item: **PASS / FAIL + evidence quote** — a short verbatim line from the
reviewer's own notes with a clip id and timestamp, e.g.
`voice-07-novel.wav 0:12–0:14: "the 'ch' in 'chains' rings metallic on each
repetition"`. An item without a quotable observation is scored FAIL (burden of
proof sits on approval).

### Section A — per voice (repeat for each of the 15)

| # | Check | Class | FAIL when |
|---|---|---|---|
| A1 | **Intelligibility without transcript** — write out what you hear BEFORE consulting any reference text; then compare | CRITICAL | any content word is unrecoverable, or your transcription diverges from the reference beyond trivial homophones |
| A2 | **Naturalness vs anchor** — independent 1–5 ordinal (scale in §5). Packet supplies the anchor description + the machine-measured Kokoro DNSMOS reference (median 3.45, span 3.197–3.55) as context for what "medium modern TTS" measures like on this machine; the ordinal is yours, not a DNSMOS copy | scores, not P/F | n/a — recorded per clip (evalA, evalB, novel) |
| A3 | **Artifact detection** — (a) clipping/hard-limiting crunch, (b) buzz/hum/tonal drone, (c) metallic/spectral smearing/vocoder ring, (d) choppiness/staccato word-splicing | CRITICAL | any sub-artifact audible anywhere in the voice's three clips |
| A4 | **Weird pauses** — mid-word gaps; unexplained silences; dead air | CRITICAL | mid-word gap; leading silence >2 s; trailing dead air >1 s; any intra-sentence gap >1 s that breaks phrasing (deviate from these defaults only downward, never upward) |
| A5 | **Pronunciation errors** — list every mispronounced or wrongly stressed word | ADVISORY* | any word listed (quote it). *Escalates to CRITICAL if a mispronunciation replaces one English word with another, changing meaning |
| A6 | **Rhythm/intonation plausibility** — stress and melody fit the sentence type (question rise, declarative fall, list cadence) | ADVISORY | delivery is flat-monotone through expressive content, or contour contradicts punctuation |
| A7 | **Collapse / repetition** — stutter loops, repeated syllables/words/phrases, echo of earlier material, degenerate babble | CRITICAL | any occurrence, however brief |
| A8 | **Gender/timbre vs persona label** — judge blind against persona-labels.json | CRITICAL for gender; ADVISORY for timbre | perceived presentation contradicts `gender_presentation`; target-character descriptor clearly wrong (partial timbre nuance mismatch stays advisory) |

### Section B — slate level (once per packet)

| # | Check | Spec |
|---|---|---|
| B1 | **Speaker distinctness, pairwise spot-checks, ≥8 pairs** | Mandatory pairs from VOICE-SLATE-V1 risk rows: 05–06 (5.3% F0 gap), 01–11 (~8%). Plus ≥6 more seeded deterministically (seed recorded), ≥2 same-gender and ≥2 cross-gender. Per pair: `DISTINCT / CONFUSABLE`. ≥2 CONFUSABLE pairs sharing a voice = distinctness flag on that voice (feeds B3) |
| B2 | **Systemic artifact scan** | Tabulate every A3/A4/A7 finding across all 15 voices + long-form by artifact subtype. A subtype is SYSTEMIC when: present on ≥3 distinct voices, OR present on any long-form clip, OR attributable to shared pipeline stages (resampler, vocoder, gain staging) regardless of count |
| B3 | **Distinctness/systemic roll-up** | Distinctness flags and systemic subtypes listed with counts and example quotes |

## 5. Naturalness ordinal (independent, per clip)

Half-points allowed. Calibrate against the anchor description, not against
metrics.json:

| Score | Meaning |
|---|---|
| 1 | Unintelligible or severely broken audio |
| 2 | Clearly robotic/synthetic; unpleasant to hear |
| 3 | Acceptable synthetic; noticeably artificial; below modern medium TTS |
| 3.5 | Competitive with medium modern TTS (Kokoro-class as described/measured on this machine) |
| 4 | Natural; minor quirks a listener forgives |
| 5 | Indistinguishable from a competent human recording |

## 6. Scoring and verdicts

Voice level (per voice NN):

```
nat_median(NN) = median(A2 ordinals of evalA, evalB, novel)  # long-form excluded
voice APPROVED  ⇔  #CRITICAL fails (A1,A3,A4,A7,gender-A8) == 0
                   AND nat_median(NN) >= 3.5
```

System level:

```
V1 APPROVED  ⇔  approved voices >= 13 / 15
                AND B2 finds NO systemic artifact class
                AND the frozen NATURALNESS-GATE numeric check passes
                    unchanged (median >= 3.15 AND min >= 2.85)
```

Any single voice REJECTED does not auto-reject V1 (one weak voice is tolerated
by the 13/15 margin, mirroring the gate's min-floor philosophy); a SYSTEMIC
finding always rejects V1 regardless of voice count. Review approval is
necessary, never sufficient: packaging (Phase N) additionally requires cleared
provenance rows per `configs/voices/README.md`.

## 7. Disagreement handling

1. Exactly **two independent reviewers** (R1, R2) per candidate per cycle,
   both satisfying §1, both reviewing the identical packet manifest hash.
2. Aggregate per-voice verdicts: `agree-approve`, `agree-reject`, `SPLIT`.
   SPLIT on any voice, or disagreement on whether a B2 class is systemic,
   triggers a **third tie-breaker** (T1) — another fresh, independent instance.
3. T1 performs its own full review of the contested material FIRST (same
   checklist, same packet), then — only after filing its own scores — reads R1
   and R2 reports and issues the deciding verdict with a one-paragraph
   reconciliation note.
4. Final verdict = majority (T1 always breaks 1–1). All three reports are
   retained; none is edited after filing.

## 8. Rejection report template

Filed at `benchmarks/results/review/<candidate-id>/reports/REJECTION-<round>.md`
whenever any voice or the system fails:

```markdown
# REJECTION REPORT — <candidate-id>
Round: <1|2> · Date: <UTC> · Packet sha256: <manifest-hash>
Checkpoint: <path + sha256> · Git commit: <hash>

## Reviewers
R1: <instance id> · Attestation: "I did not build/train this candidate and
hold no shared context with its implementers." Metrics opened after Section A: YES
R2: <same>

## Voice verdict table
| voice | nat_median | critical fails (items) | verdict |
|---|---|---|---|
| voice-01 | 3.0 | A3c metallic | REJECTED |
| ...      |     |                       | 13/15 approved |

## Failed checks with evidence quotes
- [voice-07 / A3c / CRITICAL] `voice-07-novel.wav 0:12–0:14: "<quote>"`
- [slate / B2 / SYSTEMIC] metallic timbre on voices 03, 07, 12 + longform

## Pairwise spot-checks (B1)
Pairs run: <list incl. mandatory 05–06, 01–11, seed> · CONFUSABLE: <pairs>

## Metrics corroboration (post-hoc, after Section A)
<where sealed metrics agree/disagree with listening findings>

## Required fixes before resubmission
1. <concrete, actionable item tied to a failed check>

## Resubmission conditions
<delta expected; note this consumes round 2 of 2>
```

Approval uses the same skeleton with verdict APPROVED and empty failure
sections — an approval with silent sections is invalid; every checklist item
still carries its PASS quote.

## 9. Explicitly out of scope

This protocol does not amend the frozen gate's thresholds, corpus, scorer, or
re-run instructions; it adds no DNSMOS requirements of its own; it does not
replace the Phase P scientific audit (which audits process, not just audio);
it does not clear legal/provenance release gates (Phase N).
