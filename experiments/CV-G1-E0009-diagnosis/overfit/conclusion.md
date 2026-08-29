# CV-G1-E0009 part 2 — MANDATORY OVERFIT GATE (DIAG-02): **FAIL**

Date: 2026-08-22. Driver: `../overfit_driver.py` (+ `../overfit_probes.py`,
`../overfit_probes2.py`). Artifacts: `results.json`, `audio/*.wav`,
`postmortem_probes*.json`, `checkpoints/last.pt`, `metrics.jsonl`.

**Question:** can the pilot pipeline memorize 16 utterances (2 speakers x 8,
21.8 s, LibriTTS-R smoke slice) in 2000 tiny-config steps? If not, no step
count fixes the E0007 babble.

**Setup:** pilot wiring reused verbatim (`pilot_train.py` imported, not
edited); full-corpus vocab (43); AdamW lr 2e-4 (betas 0.9/0.98, wd 1e-6),
50-step warmup then constant; batches max 3 utts / 600 frames (6/epoch, 333
epochs, every utterance ~1000 exposures); seed 1234, deterministic, 4 threads;
ran alongside the 15k pilot (PID 50376). Wall: 1266 s train + ~4 min eval.
0 NaN events. Vocab coverage: **0 skipped symbols** on all 16 utterances.

## Gates

| # | Gate | Result | Pass |
|---|---|---|---|
| 1 | flow_loss(2000) < 15% of flow_loss(200) | **38.65 → −33.94** (ratio −0.88; window means 34.63 → −47.82) | PASS |
| 2 | inferred phones/s ∈ [8, 25] | mean **16.16**, per-utt [12.26, 25.34] | PASS |
| 3 | decode 4 memorized utts → wav | 20 wavs in `audio/` (3 noise scales + teacher-forced-durations per clip) | done |
| 4 | best WER < 0.35 on ≥ 3/4 clips (whisper small.en int8) | **0.5 / 1.0 / 1.0 / 1.0 → only 1/4** | **FAIL** |

**VERDICT: FAIL** — the loss memorizes; the synthesis does not.

## What the failure is NOT (probed, evidence in results.json)

- **Not the vocoder chain**: GT log-mel through our exact 80→100 interp +
  Vocos path transcribes at WER 0.0 / 0.2 on the two clean clips and reads
  clips 1–2 nearly phonetically ("Make the perch.", "Perch a la la mande.").
- **Not mel-norm mismatch**: GT vs inferred mel stats same family
  (mean −1.82/−1.71, min ≈ −6.5/−5.7).
- **Not expand_tokens padding/mask**: B=1 probes fully consistent
  (sum(d)=mask=expanded_T=73=reported frames; wav len == T·hop).
- **Not prior collapse**: prior heads are text-sensitive (|m_p true−shuffled|
  / within-seq spread = 1.05); σ healthy (exp(logs_p) mean 0.80); data-latent
  residual |z−m|/σ = 0.80 ≈ E|N(0,1)| — the prior is Gaussian-calibrated to
  the aggregate posterior.
- **Not durations globally**: predicted-vs-MAS duration corr 0.998;
  dur_loss 0.001.

## Top-3 suspects (ranked)

1. **Degenerate MAS alignments on short utterances + silent duration clamp.**
   On clip 1 MAS assigns ONE phoneme 65 of 87 frames (silence dumping; the ‖
   silence token is reserved but unused), and `infer()` clamps durations to
   [1, max_duration=50] so synthesized length silently shrinks 87→72 frames.
   Content reproduction collapses exactly where alignment degenerates:
   teacher-forced ns=0.1 mel corr with GT = 0.04 on clip 1 vs 0.72–0.74 on the
   cleanly-aligned clips 3–4 (L1 ≈ 2.0 vs ≈ 1.0).
2. **i.i.d. per-frame prior sampling loses inter-frame structure.** Synthesis
   draws z frame-independently; decoded mels are under-detailed (inferred mel
   std 1.20 vs GT 2.08; high-energy frame fraction 0.18 vs 0.51 on clip 1) —
   consonant transients vanish while rhythm/prosody survive, which is exactly
   the observed whisper behavior (fluent English, wrong words: "It's perfect
   sloth" for "His dishonor reflects upon us").
3. **The objective never measures sample fidelity.** flow_loss reaches −48
   per frame via logdet + marginal calibration without forcing reconstruction
   of intelligible content from the sampled path; deeply negative NLL
   coexists with only ~half of mel variance reproduced (corr ≈ 0.73 even
   teacher-forced at ns=0.1). "Memorized likelihood" ≠ memorized speech.

## Gate-4 calibration caveat

Raw-GT-audio whisper ceilings were themselves 1.0/1.33 on sub-second clips
1–2 ("BAKED PERCH", "PERCH A L'ALLEMANDE") but 0.0/0.2 through the chain, so
future ASR gates should reference chain-decoded GT mels, not raw audio.
Best model result on a fair clip: "Well, I own it." → "I am it." (WER 0.5).

## Recommended next probes (Phase K)

Add explicit silence handling (activate ‖ / trim silences before MAS);
penalize per-phoneme duration outliers (max_duration headroom); try
noise_scale→0 with variance floor; evaluate a posterior-mean decoder path
(z from GT mel statistics rather than prior sampling) as an upper-bound probe.
