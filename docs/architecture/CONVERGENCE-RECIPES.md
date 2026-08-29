# CONVERGENCE-RECIPES — ARCH-CONV-01 (2026-08-22)

What published recipes say about steps-to-intelligibility and low-resource
accelerators for MAS+flow NAR TTS (Glow-TTS lineage), applied to our run:
tiny 8.7M, 1.66h multi-speaker, babble at step 1000–1600, loss 178→39.5,
~1 step/s CPU (⇒ 1k steps ≈ 17 CPU-min, 10k ≈ 2.8 h, 30k ≈ 8.3 h, 70k ≈ 19 h,
240k ≈ 67 h). Sources: see `docs/research/WEB-SOURCE-INDEX.md`, track
`conv/recipes` (cached under `research/cache/`).

## 1. Evidence table

| source | arch | data | steps to intelligible / final | trick | label |
|---|---|---|---|---|---|
| Glow-TTS paper (arXiv 2005.11129) | Glow-TTS | LJSpeech 24h | 240k iters to final quality; Adam+Noam (implies ~4k-step warmup); prior σ=1 during training, sampling temperature 0.333 at inference | Noam warmup; train-time σ=1, low-σ inference | REPORTED |
| jaywalnut310/glow-tts issue #16 (author) | Glow-TTS | LJSpeech | 240k steps paper; "with 4× batch size it would make sense to quit training at less than 24k steps"; MAS runs on CPU cores and bottlenecks when batch grows without proportional CPU | steps scale ~inversely with batch; MAS CPU cost | REPORTED |
| coqui TTS discussion #1703 | Glow-TTS | RUSLAN ~30h (single-spkr RU) | "at 10k steps I already begin to understand speech", plateau after 40k | — | REPORTED (anecdote) |
| RAD-TTS (openreview 0NQwnnwAORi, NVIDIA) | flow TTS + generative duration | LJSpeech, MCV | alignment "converges rapidly towards a usable state in a few thousand iterations"; staged schedule Asoft (0–6k) → Viterbi Ahard (6–18k) → +binarization loss (18k+); beta-binomial diagonal prior improves MCD over first 20k steps | alignment staging; diagonal alignment prior; inference z_dur σ=0.7, z_mel σ∈{0.5,0.667} | REPORTED |
| Matcha-TTS (arXiv 2309.03199) | OT-CFM NAR (flow-matching) | LJSpeech | 500k updates, batch 32, lr 1e-4, 2 GPUs (to eval quality) | — | REPORTED |
| VITS paper (arXiv 2106.06103) | VITS | LJSpeech / VCTK | up to 800k steps, batch 64×4 GPUs | — | REPORTED |
| coqui VCTK VITS release (discussion #1734) | VITS | VCTK 44h / 109 spk | released at 1M steps, batch ≈24 | — | REPORTED |
| coqui discussion #2615 | VITS | custom single-spkr | 70k steps "beautiful"; 2–2.5 A100-h per 10k steps | — | REPORTED (anecdote) |
| coqui discussion #3377 | VITS | small custom data | eval loss bottoms then rises after 10–20k steps (overfit signature) → checkpoint-select, don't assume monotone | early checkpoint selection | REPORTED (anecdote) |
| rhasspy/piper discussion #706 | VITS (Piper) | 10h single-spkr JA | still noise at 724k steps with effective batch 512 (64×8 DDP) — big-batch from-scratch caution (confounded: config/data bugs possible) | batch-size failure mode | REPORTED (confounded) |
| SC-GlowTTS (arXiv 2104.05557) | multi-spkr Glow-TTS + external speaker encoder | VCTK 44h/109 spk: 150k steps; LJSpeech pretrain 290k → finetune on 11 speakers 70k (best ckpt 66k) | single-spkr pretrain → multi-spkr finetune; converges with only 11 speakers; AP-loss speaker embeddings injected in all 12 flow blocks + duration predictor | REPORTED |
| AdaVITS (arXiv 2206.00208) | tiny VITS 8.97M, iSTFT decoder | LibriTTS 242h/1151 spk pretrain 800k steps batch 32 (one 2080Ti) → speaker adaptation in 2000 steps | tiny models still need large pretrain; adaptation is cheap; PPG auxiliary loss used to stabilize simplified model | REPORTED |
| StyleTTS 2 training schedule (via emergentmind summary) | StyleTTS 2 | LJSpeech/VCTK/LibriTTS 24–245h | acoustic modules pre-trained first (100/50/30 epochs), then joint end-to-end ≤60 epochs | staged curriculum pretrain→joint | REPORTED (secondary summary) |
| VITS-fast-fine-tuning (README) | VITS | pretrained trilingual → user voices | finetune 20 min–2 h on Colab yields usable clones | finetune-from-pretrained | REPORTED |
| Kokoro-82M (HF model card) | StyleTTS 2 arch + iSTFTNet, 82M | v0.19 <100 h; v1.0 few hundred hrs | ~500 A100-80GB-h per release (~$400) — production-quality small TTS from ≤100h is demonstrated 2024–2025 | ≤100h feasibility datapoint | REPORTED |
| Lai et al., ICASSP 2020 (MIT/JHU) | E2E multi-spkr TTS | VCTK etc. | neural speaker embeddings (LDE/AP-class) beat x-vectors on similarity AND naturalness for unseen speakers — evidence is about quality/generalization, not convergence speed | conditioning choice | REPORTED |
| coqui finetuning docs | vendor | — | finetuning pretrained ckpt = faster learning + better small-data results | finetune-from-pretrained | REPORTED (vendor doc) |
| icefall VITS-VCTK recipe | VITS | VCTK | 1000 epochs, 4 GPUs (recipe default) | — | REPORTED (recipe) |
| EMA weights | — | — | NOT found in probed canonical code: coqui vits_config.py, coqui shared_configs.py, Grad-TTS train.py, Matcha train.py/utils — no EMA shipped | EMA not standard in this class | ESTIMATED (negative check) |
| Spec/pitch augmentation for TTS acoustic models | — | — | no controlled TTS-side evidence found this session; standard only in low-resource ASR (SpecAugment, speed-perturb studies) | augmentation | ESTIMATED |
| Duration-prior σ *scheduling* during training | — | — | not found as a published trick; published practice is σ=1 fixed during training, σ lowered only at inference (Glow-TTS, RAD-TTS) | sigma scheduling | ESTIMATED (absence of evidence) |

## 2. Synthesis against our four questions

1. **Step counts.** For Glow-TTS-class models, alignment becomes usable within a
   few thousand steps (RAD-TTS), first intelligible speech is reported at ~8–20k
   steps on 10–45h datasets (coqui #1703), and *final* quality runs 240k–1M
   steps (Glow-TTS paper, VITS, Matcha, coqui VCTK). Our babble at 1000–1600 is
   **on schedule**, not a failure signal: MAS is binding (loss 178→39.5) but the
   flow decoder lags. At 1 step/s, first-intelligibility territory (~10–20k
   steps) costs 3–6 CPU-h; a meaningful verdict needs ~30k steps (~8.3 CPU-h).
2. **Accelerators.** Strongest-evidenced: (a) pretrain→finetune staging
   (SC-GlowTTS 290k→70k; AdaVITS 800k→2k adaptation; VITS-fast-fine-tuning
   minutes-to-hours; StyleTTS 2 staged curriculum); (b) checkpoint selection —
   eval loss can bottom at 10–20k then *rise* (#3377); (c) low inference σ
   (0.333–0.667) recovers intelligibility from partially-converged flows
   (Glow-TTS T=0.333, RAD-TTS σ tables). EMA and spec/pitch augmentation are
   plausible but unverified in this model class this session.
3. **Multi-speaker conditioning.** No source reports a *convergence-speed*
   comparison of speaker-table vs x-vector conditioning. Quality-side evidence
   favors learned neural embeddings for unseen speakers (Lai 2020; SC-GlowTTS),
   but those add an external encoder dependency. SC-GlowTTS converged with 11
   speakers; nothing suggests 247 thin speakers is *required* — per-speaker
   hour concentration is the untested lever.
4. **≤100h small-TTS reports.** Kokoro-82M (StyleTTS 2 + iSTFTNet): <100h
   (v0.19, Dec 2024) → production quality, ~500 A100-h. AdaVITS: 8.97M model,
   800k steps batch 32 on a single consumer GPU. Both confirm the regime is
   CPU-feasible-ish only with GPU-class throughput or aggressive staging; from
   scratch on 1.66h alone is below every published from-scratch datapoint we
   found (smallest: 10h Piper, which failed under big-batch DDP).

## 3. Top-5 recipe changes for the next run (ranked by expected impact/effort)

Each is falsifiable; metrics use our fixed eval: decode 20 held-out sentences
per checkpoint, score WER with a local ASR (wav2vec2-base already in
`research/cache`), plus loss.

**R1. Re-baseline + checkpoint-gated evaluation (no config change).**
Continue the current recipe to ≥30k steps; snapshot every 2.5k steps (coqui
community practice); auto-decode + WER each snapshot; select best checkpoint,
never assume monotone improvement (#3377). Cost ≈ 8.3 CPU-h.
*Falsifier:* if WER@30k > 70% AND the 12.5k/15k/…/30k WER trend is flat
(no ≥10% relative drop across 3 consecutive snapshots), declare
scratch-on-1.66h infeasible and escalate to R2.

**R2. Two-stage curriculum: single-speaker pretrain → multi-speaker finetune.**
Pretrain on LJSpeech (24h, CC-BY, already acquired per DATA-PILOT-01) 50–100k
steps (14–28 CPU-h), then finetune on the 1.66h multi-speaker set 5–10k steps.
Directly mirrors SC-GlowTTS (290k→70k) and AdaVITS (800k→2k). When clean-100
lands, same staging applies (few-speaker → all-speaker).
*Falsifier:* steps-to-WER<50% must improve ≥2× vs the R1 scratch baseline; if
not, curriculum is rejected for this arch/size.

**R3. Warmup and batch/LR hygiene.** Ensure a Noam-style warmup of ~4k steps
(Glow-TTS paper schedule) — our babble window (1000–1600) sits inside a
default warmup, so early babble may partly be LR-schedule, not capacity. Keep
effective batch 16–32: MAS cost scales with batch (jaywalnut310 CPU note),
large-batch from-scratch runs have a documented noise-failure (piper #706),
and per-step sample efficiency dominates on tiny data.
*Falsifier:* A/B warmup 500 vs 4000 steps at fixed LR peak; the 4k arm must
show lower loss@5k variance and better WER@15k, else revert.

**R4. Zero-cost inference σ sweep on existing checkpoints.** Re-decode
checkpoints 1000/1600/2500 with prior sampling σ ∈ {1.0, 0.667, 0.5, 0.333}
(Glow-TTS trains at σ=1 and ships T=0.333; RAD-TTS uses σ_mel 0.5–0.667).
No training cost; may already convert babble→near-speech.
*Falsifier:* if no σ improves WER ≥10% relative on ckpt-2500 vs σ=1.0, the
bottleneck is alignment/decoder capacity, not prior variance — deprioritize
inference tuning.

**R5. Speaker-table conditioning + hour-concentration experiment.** Keep the
speaker lookup table for now (fewest moving parts; x-vector/learned-encoder
conditioning is evidenced for unseen-speaker *quality*, not convergence
speed). When clean-100 lands, run paired: all-247-speakers vs top-64-by-minutes
subset (SC-GlowTTS converged with 11; thin per-speaker coverage may slow MAS).
*Falsifier:* subset arm must beat all-speaker arm on WER@20k at equal step
count; if it does, adopt subset pretraining with all-speaker finetuning.

Honorable mentions (not top-5, low evidence): EMA of weights for eval stability
(negative-checked absent from canonical repos; cheap to add, ~10 lines);
SpecAugment-style masking / speed-perturbation (standard in low-resource ASR,
unproven here); duration-prior σ *scheduling* (no published precedent —
published practice is fixed σ=1 in training, low σ at inference only).
