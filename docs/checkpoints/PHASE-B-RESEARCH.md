# CHECKPOINT — PHASE B (State-of-the-Art Research)

Date: 2026-08-22 · All claims carry evidence labels in their source documents.

## 1. Research performed
Eight parallel tracks (A–H), executed by research subagents under coordinator
reconciliation. Severe API instability during execution: 8 agents failed at
least once; 2 were replaced wholesale after unrecoverable context loss.
Recovery protocol (incremental disk flushes + resume-from-disk) preserved all
work — zero net data loss.

## 2. Sources consulted
~500 URL citations across 10 documents (MEASURED counts in SOURCES.md);
primary sources preferred: arXiv, official repos, HF cards, vendor docs,
LICENSE files. 49 claims remain explicitly flagged UNVERIFIED (backlog in
docs/research/SOURCES.md).

## 3. Findings (headline)
- License fragmentation is the defining 2025–26 trend: permissive code +
  encumbered weights is the default trap (Emilia/F5 CC-BY-NC, EnCodec NC
  weights, NSF-HiFiGAN NC-SA, Piper MIT→GPL flip, Fish research-only).
- CPU-relevant frontier converged on NAR mel models with small iSTFT-head
  vocoders (Kokoro-class); on-device segment emerged (Chatterbox-Nano,
  NeuTTS Air, KittenTTS). Nobody publishes CPU batch-1 latency → we must measure.
- AR codec-LM class disqualified for G1 primarily by DATA hunger
  (60k–100k+ h) and hallucination failure modes, not FLOPs.
- INT8: encoder-side win (~1.5–2× w/ VNNI), vocoder-side measured trap;
  FP16 = storage-only on AVX2.
- Frontend: espeak-ng GPL ecosystem contamination risk confirmed; misaki
  Apache but py<3.13-pinned + LGPL/GPL optional paths; CMUdict unrestricted.
- Data: SMOKE ~1.4 GB / PILOT ~16 GB plans ready; Expresso/Emilia excluded
  from commercial path; 5 items queued for human/legal review.
- Evaluation: credible all-CPU auto-stack defined (dual-ASR min-WER, UTMOS/
  DNSMOS/SQUIM tiers, ECAPA identity, hallucination gates); predicted-MOS
  saturation documented → human tests remain irreplaceable for prosody.

## 4. Decisions made
- NO final architecture decision yet (charter: none before this checkpoint —
  now permitted). Preliminary direction recorded as HYPOTHESIS ranking:
  CAEDRAL-NAR > CAEDRAL-CFM >> AR-codec-LM, to be falsified in Phases C/F.
- Phase C baseline shortlist adopted (priority order): Kokoro-82M, Piper
  (isolated process, GPL), Chatterbox-Nano, KittenTTS-mini, F5-TTS
  (reference-only, NC weights).
- Own-chain phoneme frontend (CMUdict-derived + inflect TN +
  DeepPhonemizer-ONNX OOV) is leading Phase E candidate vs char-control.

## 5. Files created/changed
docs/research/: STATE-OF-THE-ART, BASELINE-MATRIX, TEXT-FRONTEND,
ACOUSTIC-ARCHITECTURES, VOCODERS-DECODERS, CPU-INFERENCE-OPTIMIZATION,
EVALUATION-BENCHMARKS, MULTI-SPEAKER-PROSODY-STYLE, SOURCES.md (new index).
docs/data/: DATASET-SURVEY, LICENSE-MATRIX. MASTER-PLAN.md updated.

## 6. Experiments executed
None (research phase). Local measurements begin Phase C.

## 7. Exact measured results
Only environment-level measurements so far (Phase A). All third-party
performance figures are labeled REPORTED BY THIRD PARTY inside documents and
are NOT accepted as our measurements.

## 8. Failed experiments
N/A. Process failures: 12 agent crashes (API timeouts/malformed responses),
2 agent replacements — documented here per §26 honesty rules.

## 9. Remaining risks
49 UNVERIFIED flags pending re-verification before decision-grade use;
5 dataset/legal questions need human review; no human listening test has
occurred yet (predicted-MOS only until Phase J samples exist); A2's
Kokoro-standing claim needs primary confirmation.

## 10. Disk usage
Repo+venv ≈ 75 MB (MEASURED via git/du estimate). C: free ≈122 GB unchanged;
no datasets downloaded yet.

## 11. Current best model metrics
NONE EXISTS YET. No Caedral model trained (honest null).

## 12. Next phase
PHASE C — benchmark harness completion + local baseline benchmarking:
install shortlisted baselines (license-respecting isolation for Piper),
measure RTF/TTFA/RAM cold+warm on frozen corpus subset, produce the first
MEASURED numbers of the program against which Caedral models will compete.
