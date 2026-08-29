# CAEDRAL VOICE FROM-SCRATCH PATH — SUSPENDED BY FOUNDER

**STATUS: SUSPENDED INDEFINITELY · NO AUTOMATIC RESUME · NO WATCHDOG · NO BACKGROUND TRAINING**

Suspended 2026-08-23 by founder override. Only an explicit future founder
command may reopen this path.

## Why suspended

The from-scratch acoustic pipeline (MAS/CTC durations + L1 mel regression,
15.9M params, CPU-trained on LibriTTS-R clean100 + Kokoro-teacher corpus)
reached INTELLIGIBLE speech (15/15 slate voices WER ≤0.25) but **failed the
founder's perceptual bar**: robotic cadence, flat prosody, metallic texture.
DNSMOS median ~2.7 vs product target ≥3.15 (Kokoro anchor 3.45).

## Key findings preserved for any future resumption

1. **Prosody was structurally absent**: pitch=off energy=off means the decoder
   received no F0/loudness conditioning — robotic output was architectural,
   not a training-time defect (`regressor.py:268-271`).
2. **CTC aligner works**: E0011 produced 100% aligned durations across
   44,417 utterances (p50 14.7 phones/s, 99.4% in natural band).
3. **Vocos chain is clean**: E0009 proved teacher-mel→Vocos renders are
   ASR-identical to native audio.
4. **Onset defect systematic**: "Artificial" degraded in 7/7 renders across
   checkpoints/speakers — pipeline-level onset weakness, unresolved.
5. r2-best@15k val mel-L1 0.8647; loss converged but perceptual quality did
   not follow — L1-regression over-smoothing is the suspected mechanism.
6. Word-boundary spacing in mapped phonemes HURT metrics (v4 killed);
   v2 char-level mapping remains the only validated Kokoro integration.

## Last useful checkpoints

| Checkpoint | Step | Val | Path |
|---|---|---|---|
| r3 best | 8000 | mel-L1 0.8487* | experiments/r3/checkpoints/best.pt (sha256 prefix 325ba9b1be3f3807) |
| r3 last | ~2000+ of rescue-long continuation | — | experiments/rescue-long/checkpoints/last.pt |
| r2 best | 15000 | mel-L1 0.8647 | experiments/r2/checkpoints/best.pt |

*r3's own val at its step; r3 continued to ~step 11000+ before stop.

## Exact resume requirements (if ever authorized)

1. Restore repo @ current commit + `data/manifests/*` + `data/cache/mel-v1/`
   (mel cache, regenerable in ~30 min if lost) + `experiments/*/checkpoints/`.
2. Resume command: `train_regressor.py --manifests librittsr-clean100,teacher-kokoro
   --durations data/manifests/all-durations.jsonl --mel-cache-dir data/cache/mel-v1
   --steps <N> --threads 6 --exp-id <id>` (auto-resumes from last.pt).
3. Trainer guarantees bitwise-exact resume (E0003).
4. Teacher corpus (17.87 h final) is intact under
   `data/processed/teacher-kokoro/` + manifests.

## Replacement strategy

Founder pivoted to: **Kokoro foundation (Apache-2.0) + Caedral frontend/
runtime/optimization layers**. See docs/releases/PRE-ARCH-VOICE-MIGRATION.md
and the V2 production path (scripts/test_kokoro_vocab_path_v2.py).
