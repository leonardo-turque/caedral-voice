# TEACHER-DATA-ENGINE — synthetic teacher audio for Caedral training (TEACH-01)

Task TEACH-01 · **DESIGN ONLY** this wave. The §6 falsification gate (a 2 h
pilot, incl. the small `--val-from` trainer change of §7) is the first and only
implementation step DIAG-02 needs to rule on **mass generation**; everything
beyond the gate waits for the DIAG-02 verdict. Date: 2026-08-22.
Authorization on record: founder approved teacher models "temporariamente
durante desenvolvimento" while the final model walks toward Caedral ownership.
Honesty labels per charter: every number is [MEASURED] (with artifact path) or
[ESTIMATED] (with arithmetic shown).

Positioning: teacher audio is **training data, never a shipped component**.
The product remains OUR frontend → OUR CAEDRAL-NAR weights → Vocos-class
decoder. Teacher audio never leaves the training pipeline as an asset; only
its statistical influence on our from-scratch weights persists.

---

## 1. Teacher inventory [MEASURED from local artifacts]

`models/kokoro/voices-v1.0.bin` (the exact voice bank `kokoro-onnx` loads,
verified in `benchmarks/run_baseline_kokoro.py`) contains **54 voices**:
`af×11, am×9, bf×4, bm×4` (English) plus ef/em/ff/hf/hm/if/im/jf/jm/pf/pm/zf/zm
(non-English cells of v1.0-z). **No `bz_*` prefix exists** — the British cells
are `bf_*` (female) and `bm_*` (male). English pool = **28 voices**.

Kokoro-82M v1.0 ONNX on this box [MEASURED, `docs/benchmarks/BASELINE-MEASURED.md`]:
warm RTF 0.32–0.49 (long-form steady ≈0.37; default-16-thread conv 0.3512 /
para 0.3229); peak RSS ~1.25 GB; quality min-WER 0.00–0.33, DNSMOS ovrl
3.22–3.51 (af_heart/am_michael × 5 corpus items). License: Apache-2.0 code +
weights [VERIFIED per program record; see §8].

### Teacher slate (frozen; 10 voices, 5F/5M, 7 US / 3 UK)

All names verified present in `voices-v1.0.bin` [MEASURED]. Selection rule was
cell coverage across the four accent×gender cells; no per-voice quality
ranking exists in our records, so the slate is coverage-based, not quality-based.

| slot | voice | cell |
|---|---|---|
| t00 | af_heart | F / US |
| t01 | af_bella | F / US |
| t02 | af_nicole | F / US |
| t03 | bf_alice | F / UK |
| t04 | bf_emma | F / UK |
| t05 | am_adam | M / US |
| t06 | am_fenrir | M / US |
| t07 | am_michael | M / US |
| t08 | am_puck | M / US |
| t09 | bm_george | M / UK |

Zero-padded prefixes `t00..t09` sort lexicographically, so the existing
`id.split("_")[0]` → speaker-index logic (mel_dataset.py, pilot_train.py)
assigns slot = NN with zero code change in teacher-only runs; in mixed runs
numeric LibriTTS-R prefixes sort first (ASCII digits < letters), so teachers
land after all real speakers, deterministically.

## 2. Scale math (target +40–80 h teacher audio)

### Wall-clock from measured RTF

Single sequential ONNX session, default threads (the measured-fastest mode).
`wall = audio_seconds × RTF`:

| Target audio | @ RTF 0.35 (central) | @ 0.32 | @ 0.49 | @ long-form 0.37 |
|---|---|---|---|---|
| +40 h | 14.0 h | 12.8 h | 19.5 h | 14.8 h |
| +60 h | 21.0 h | 19.2 h | 29.4 h | 22.2 h |
| +80 h | 28.0 h | 22.7 h | 39.2 h | 29.6 h |
| 2 h pilot | 0.70 h (42 min) | 38 min | 59 min | 44 min |

Phonemization is negligible (measured 251 utt/s, `docs/training/PILOT-PLAN.md`:
33k utts ≈ 2.2 min). Wav writing and preprocessing are I/O-trivial next to
synthesis. Optional 2×8-thread two-session split gives ~1.55× combined
throughput [ESTIMATED from the BASELINE-MEASURED thread sweep] but re-introduces
the documented hybrid-core bimodality risk; sequential default is the default
plan. Schedule generation so it does NOT contend with a live training run
(both are CPU-hungry); the natural slot is after `pilot-tiny-15k` completes.

### Disk at 24 kHz PCM16 mono

48,000 B/s = 172.8 MB/h [arithmetic]. 40 h = 6.9 GB · 60 h = 10.4 GB ·
80 h = 13.8 GB; manifests ≈ tens of MB. Free disk today [MEASURED]: **82 GB**
→ worst case (13.8 GB + working copies) fits with wide margin alongside the
existing corpora.

### Utterance math

LibriTTS-R train-clean-100 extraction has **33,232 utterances** on disk (wav +
sibling normalized-transcript .txt per utt)
[MEASURED `find data/raw/librittsr-full -name '*.wav' | wc -l` = 33,232].
Mean utterance duration: the 10-speaker slice measured 1.6616 h / 1,441 utt =
4.15 s; the full set is expected in the 4.2–4.6 s band [ESTIMATED — slice may
under-represent longer utts; DATA-FULL-01 preprocessing will measure it].

Expected teacher hours = 33,232 × d̄ × K / 3600, where K = voices per text:

| K | d̄ = 4.2 s | d̄ = 4.6 s |
|---|---|---|
| 1 (each text, one voice) | 38.8 h | 42.5 h |
| 2 (each text, two voices) | 77.5 h | 85.0 h |

K is the single dial for the 40–80 h target: **Stage A (K=1) ≈ +39–43 h,
Stage B (K=2) ≈ +78–85 h**. Stage B only after the §6 gate passes.

## 3. Text source strategy

**Backbone: re-spoken LibriTTS-R transcripts.** The 33,232 train-clean-100
normalized transcripts already on disk (public-domain LibriVox lineage,
CC BY 4.0 dataset terms per `docs/data/DATASET-SURVEY.md` Tier A) are
re-spoken by teachers. No new text is acquired ⇒ no new text licensing
questions; text rights are already cleared by the real-data track.

**Eval-corpus firewall (hard rule).** `benchmarks/corpus/eval-corpus-v1.0.json`
items are FORBIDDEN as training text. Any teacher item whose normalized text
matches an eval-corpus item (normalized comparison) is rejected. Rationale:
training on benchmark text invalidates every quality number we publish.
Corpus items instead serve as periodic held-out probes during long runs
(FIRST-CAEDRAL-VOICE preview discipline).

**Dedupe rules.**
1. *Within teacher set:* exact-duplicate rejection on sha256 of normalized
   text **per voice** — key = (voice, text-hash). Note: `preprocess.py`'s
   global norm-hash dedupe must NOT be reused as-is for Stage B (K=2), or the
   second voice's copy of a text would be wrongly rejected.
2. *Real ↔ teacher:* text overlap is **intentional by design** (same text,
   different speakers) and MUST NOT be deduped away. Cross-manifest dedupe
   is disabled; provenance fields keep the sets auditable.
3. *Content addressing:* every teacher row carries `sha256_audio` +
   `sha256_transcript` (survey manifest design, `docs/data/DATASET-SURVEY.md`
   "Deterministic manifest design"), so any future dedupe/audit is hash-based,
   never name-based.

## 4. Pair construction — ours on both sides

For every kept (text, voice) pair:

- **Input side = OUR frontend.** Text → `TextFrontend("lexicon+g2pen")`
  (the E0001/E0004 shippable ladder; GPL espeak never touched) →
  `phoneme_string`, via `data/scripts/phonemize_manifest.py` (measured
  251 utt/s). The teacher's own phoneme chain is never used as model input —
  Kokoro consumes text internally and only its audio is harvested.
- **Target side = OUR mel.** Teacher wav (24 kHz native) → standard
  preprocess chain → at train time `mel_dataset.py` computes the canonical
  80-band ln-mel (24 kHz, hop 256, power mel, natural log, floor 1e-5) — the
  exact representation the E0009 audit proved decodes cleanly through the
  Vocos preview adapter. We store PCM16 wav, not mels, so the feature
  definition stays single-sourced in `mel_dataset.py` and can evolve without
  regenerating audio.
- **speaker_id = teacher index.** Ids `tNN_<source_id>` make the id-prefix
  mechanism assign one embedding-table slot per teacher (§1). No trainer
  change.
- **Durations still come from MAS at train time.** The manifest's
  `duration_s` is used ONLY for batch-sampler frame estimates
  (`PilotDataset.est_frames`); alignment/durations for the loss are produced
  by MAS inside `losses_for_training` exactly as for real data. We never
  trust Kokoro's internal timing.
- **Preprocessing parity.** Teacher wavs run the same
  `preprocess.py` chain as real data: DC removal, 70–80 Hz high-pass, two-pass
  loudnorm to −23 LUFS (EBU R128, survey practice #3), silence trim, duration
  filter 0.3–15 s, clip rejection — with `--target-sr 24000` (native, no
  resample). MAS-infeasible rows (est_frames < n_phones) are dropped at
  dataset build with the reason recorded (existing preflight).

### Generation QA screens (sanity tier; stack from BASELINE-MEASURED)

- Dual-ASR min-WER screen (faster-whisper small.en + base.en int8): reject
  rows with WER > 0.15; acceptance target ≥95% of kept rows ≤ 0.05 (clean
  synthetic speech should sit far below the real-data 20–25% rejection bar of
  survey practice #8).
- DNSMOS v8 ovrl sanity band ≥ 3.0 (clean ceiling observed 3.3–3.5 here);
  rows below flag a pipeline fault, not a style judgment.
- Both screens run post-preprocess, pre-manifest-freeze; rejects logged to
  `<name>.rejected.jsonl` with reason `qa_wer` / `qa_dnsmos`.

## 5. Provenance & provenance meta (per manifest)

`<name>.meta.json` gains a `teacher` block: kokoro-onnx version, model +
voices-bin sha256, git commit, per-voice utt counts, RTF observed during the
run, K factor, text source (`librittsr-full` + archive sha256), QA thresholds.
Per row: `source_id` (LibriTTS-R utt id), voice name, speed (1.0 default;
speed jitter is an optional diversity lever, off by default), `sha256_*`.
Freeze discipline per DATASET-SURVEY: frozen manifests + committed hashes,
read-only downstream.

## 6. Falsification gate — experiment CV-G1-E0010-teacher-pilot

**Question:** is teacher audio worth generating at all? Decides DIAG-02.

**Setup (all [MEASURED] baselines from the live run `experiments/pilot-tiny-15k/`,
run_id `pilot-20260822-160333`, manifest `librittsr-smoke-10spk`, 1,441 utt /
1.6616 h / 10 spk, tiny 8.35M config, seed 1234, val_frac 0.1 → 1,293 train /
148 val):**

Generate the **2 h pilot teacher set**: 1,800 texts × 10 teachers
(round-robin, 180 texts each; 1,800 × 4.15 s ≈ 2.07 h). Texts = deterministic
stride sample of the 33,232 transcripts AFTER the §3 firewall + dedupe.
Generation wall ≈ 42 min @ RTF 0.35 [ESTIMATED from MEASURED RTF].

Then train the **identical tiny config** (same seed, lr 1e-4, warmup 400,
batching, 15k steps, val_every 1000) on teacher-only data, validated on the
**exact real val split** (the same 148 real LibriTTS-R rows — requires the
small `--val-from` override described in §7; without it the stratified split
would leak teacher rows into val and the comparison is void). Teacher voice k
occupies speaker slot k (t00..t09), so both runs use slots 0–9 of the same
32-slot table — val loss is then scored against comparably-trained speaker
embeddings.

**Reference trajectory [MEASURED, real-only run, val flow_loss_mean]:**

| step | 1000 | 2000 | 3000 |
|---|---|---|---|
| real 1.66 h | 43.236 | 36.854 | 32.994 |

**Acceptance (gate PASS requires ALL):**
1. Teacher-pilot val flow_loss_mean **strictly below the real run at every
   matched val step** {1000, 2000, 3000};
2. **≤ 31.34 at step 3000** (= 0.95 × 32.994 — margin, not noise);
3. Hygiene: zero NaN events post-warmup; resume-exact if restarted; dur_loss
   consistent with the 12–18 phones/s plausibility band (PILOT-PLAN);
4. QA screens from §5 passed on the pilot set itself.

**Extension clause:** the real run continues toward 15k steps (~0.94 s/step
[MEASURED] ≈ 3.9 h wall). If the mass-generation decision waits for its
completion, the gate extends to every additional matched val step, with the
same rule: strictly below at all matched steps and ≤0.95 × real at the final
matched step. Teacher run trains to the same step count as the real run at
gate time.

**Honesty notes (required in the E0010 conclusion):**
- Step-matched, not hour-matched: the teacher set is 2.07 h vs 1.49 h real
  train hours (+39%). If DIAG-02 wants hour-parity, run the optional variant
  subsampling teacher to 1.66 h before concluding.
- The gate measures whether teacher data teaches the phones→mel mapping at
  least as well as real data *on real-speech validation*; it does not measure
  accent diversity or downstream perceptual quality (no perceptual claims —
  Phase K discipline).
- Secondary diagnostic (not gated): mixed run
  `--manifests librittsr-smoke-10spk,kokoro-teach-pilot` vs the real-only
  trajectory, to see whether mixing helps or shifts the distribution before
  mass generation commits to a ratio.

**Kill criteria:** any acceptance failure ⇒ DIAG-02 verdict NO-GO for mass
generation; the engine stops as a documented dead end and the real-data scale-up
(DATA-FULL-01, ~100 h) is the augmentation path instead. No partial credit.

## 7. Integration with DATA-FULL-01 (manifest schema, two-name mixing)

The trainer already mixes manifests by name:
`--manifests librittsr-full,kokoro-teach-v1` loads
`data/manifests/<name>.jsonl` + `<name>.phonemes.jsonl` for each name and
concatenates rows (pilot_train.py). Teacher engine emits exactly the DATA-FULL-01
row schema (`id, audio_path, transcript, duration_s, sample_rate, lufs_before,
lufs_after, clipped_frac, sha256_audio, sha256_transcript` + `source_id`,
`teacher_voice`) plus the `.phonemes.jsonl` pair
(`id, transcript_norm, phoneme_string, n_phones, g2pen_used`) and `.meta.json`
with the §5 teacher block.

Required code changes (small, design-specified here, implemented post-DIAG-02):
1. **`--val-from <name>`** in pilot_train.py: when set, val indices come only
   from that manifest (same per-speaker stride logic); train = all other rows;
   vocab built from train-side phoneme strings only, val encoded with it
   (OOV → skip + log). Default absent = current behavior, zero regression.
2. Teacher id scheme `tNN_<source_id>` (§1) — **no code change**; verified
   against `id.split("_")[0]` in mel_dataset.py and `build_speaker_index`.
3. Speaker-table capacity: tiny config `n_speakers=32`. Gate + mixed-pilot
   (10 real + 10 teacher = 20 ≤ 32) fits. At DATA-FULL-01 scale (~247
   LibriTTS-R speakers + teachers) the table MUST be resized in config — a
   fresh-run decision to make when DATA-FULL-01 lands, noted here so it is
   not discovered late.

Checkpoint/vocab freeze: vocab and speaker index are rebuilt per run from the
merged manifests; teacher runs are always fresh runs (no resume into or out of
different manifest sets), consistent with E0003 resume guarantees.

## 8. License & provenance — what is Caedral vs external

| Component | Status | Evidence |
|---|---|---|
| Text frontend (TN + lexicon + g2p ladder) | **Caedral** | our code, E0001/E0004 |
| CAEDRAL-NAR architecture + training system | **Caedral** | E0002/E0003/E0006 |
| 80-band ln-mel definition + preprocessing pipeline | **Caedral** | mel_dataset.py, preprocess.py |
| Trained weights (from scratch) | **Caedral** | this program's output |
| Teacher model (Kokoro-82M v1.0, code + weights) | External, Apache-2.0 [VERIFIED] | BASELINE-MEASURED.md; program license record |
| Teacher audio | External-derived **training data ONLY** — never shipped | this doc |
| Vocos decoder dependency | External, MIT code + weights, SHIP-SAFE | DECODER-LICENSES.md §1 |
| Real corpora (LibriTTS-R CC BY 4.0, LJSpeech PD, CV CC0) | External, attribution duties | DATASET-SURVEY.md |

### Commercial-release risk assessment (engineering assessment, not legal advice; counsel review remains an open item per DATASET-SURVEY)

**Path A — Kokoro-distilled (recommended, gated).** Apache-2.0 grants use,
modification, and commercial distribution of the work with no field-of-use
restriction; the licensor affirmatively licensed broadly, and model *outputs*
carry no NC-style contamination (contrast the Emilia/F5-TTS NC downgrade
precedent, DATASET-SURVEY [S15][S17]). Teacher voices are synthetic — no real
person's publicity rights are engaged (cleaner identity posture than
LibriVox-reader anchors, whose consent is the survey's open item #4). Teacher
audio is never redistributed. Residual risks: the outputs-of-ML-models
doctrine is generally unsettled (mitigated by the permissive licensor and
training-data-only use); license drift — pin v1.0 artifacts by sha256 and
re-verify at release cut (DECODER-LICENSES drift precedent). Posture:
**LOW-MED risk. GO as training-data-only augmentation behind the §6 gate.**

**Path B — Piper-distilled (excluded).** Piper binary is GPL-3.0 with a
mandatory subprocess-isolation policy already in force
(`benchmarks/run_baseline_piper.py`); its voice datasets carry permissive
terms, but using outputs of a GPL program as training data is legally gray,
isolation does not cure the training-use question, and measured quality sits
at or below Kokoro's (DNSMOS 3.07–3.46 vs 3.22–3.51) while RTF is not the
bottleneck for offline generation. Posture: **MED-HIGH gray. EXCLUDE from any
release path.** Revisit only with explicit counsel sign-off.

**Path C — real-only (safe fallback, always available).** DATA-FULL-01 scale-up
(~100 h LibriTTS-R CC BY 4.0 + LJSpeech PD) is the cleanest license posture
(attribution duties only); costs are wall-clock training time on CPU and the
survey's MED identity-anchor provenance caveat. This path remains the backbone
**regardless of the gate** — teacher data is additive, never a replacement.

**Recommendation:** Path A as gated augmentation, Path B excluded, Path C
continues. Ownership story stays intact: the shipped artifact is Caedral
weights trained from scratch; teacher audio never ships; founder's
temporary-development-use authorization is on record, and before any
commercial release: (a) re-verify pinned Kokoro artifacts, (b) close the
counsel-review open item on outputs-as-training-data, (c) honor EU AI Act
Art. 50 transparency duties flagged in DATASET-SURVEY.

## 9. What DIAG-02 must decide

1. Run the §6 pilot now (generation 42 min + training ≈ real-run-equivalent
   wall) or wait for the real 15k run to finish for a longer matched gate.
2. Hour-matched variant: yes/no (§6 honesty note).
3. Stage B (K=2) pre-authorization upon gate PASS, within the 82 GB disk and
   ~28 h generation budgets.
4. Counsel-review scheduling for Path A's open item before any release cut.
