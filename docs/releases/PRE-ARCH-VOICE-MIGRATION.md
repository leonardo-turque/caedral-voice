# PRE-ARCH VOICE MIGRATION — full state snapshot for the Windows → Arch Linux wipe

Written: 2026-08-23 · Audience: the future coordinator (OpenCode on Arch) · Companion
first-read doc: `docs/OPENCODE-HANDOFF.md` (read that FIRST; this file is the deep reference).

Labeling per charter: every number carries `MEASURED` (with date), `REPORTED`, or
`[VERIFY AT MIGRATION TIME]`. A moving-target fact is marked as such — re-measure it
after the training run stops and BEFORE the backup is cut.

---

## 1. STATUS

| Item | State |
|---|---|
| Program | Caedral Voice V1 — B-arm (CTC-aligned supervised durations → mel regressor) |
| Machine being wiped | i7-13620H (10C/16T, AVX2/FMA3, **no AVX-512**), 23.7 GB RAM, Windows 11, disk layout under `C:\Users\leona\Downloads\caedral-voice` [MEASURED 2026-08-22] |
| Destination machine | Arch Linux; repo root restored to `~/caedral-voice` |
| Current experiment | **r3** (`experiments/r3/`, run_id `regressor-20260823-082518`) — CV-G1-E0013 regressor long run, 20,000-step target |
| Stage at write time | TRAINING, live. Worker PID 21728 confirmed alive at write-time recheck 2026-08-23 (started 08:25 local), step ~9,100/20,000, avg ~1.55 s/step incl. val overhead [MEASURED 2026-08-23] → remaining ~10,900 steps ≈ 4–5 h [ESTIMATED] |
| Best checkpoint at write time | `experiments/r3/checkpoints/best.pt` — global_step **9000**, val_loss **0.84872** [MEASURED 2026-08-23 15:18 UTC]. This ADVANCES while training runs — see §2 |
| Previous-good fallback | `experiments/r2/checkpoints/best.pt` — val_loss 0.86473 @ 15,000 steps, run COMPLETE (fit_end 2026-08-23T10:57:12Z), therefore FROZEN/stable [MEASURED 2026-08-23] |
| Do we have a Caedral-generated product voice? | NO — prototypes only; DNSMOS ~2.7 vs frozen gate 3.15 (§7). Do not pretend otherwise (charter §26) |

## 2. CURRENT EXPERIMENT / STAGE / CHECKPOINT / RESUME POINT

- **Run**: r3, launched detached 2026-08-23 ~08:25 local, PID file convention
  `experiments/CV-G1-E0013-regressor/*.pid`; live telemetry in `experiments/r3/metrics.jsonl`
  (append-only JSONL; the ONLY reliable log — see known issue K2 about buffered stdout).
- **Checkpoint cadence**: atomic save every 1000 steps (`step_%07d.pt`, `last.pt`,
  `best.pt`) + sidecars `phoneme_vocab.json` and `speaker_index.json` in the same
  directory. Sidecars MUST travel with every checkpoint copy.
- **Checkpoint contents [MEASURED]**: each `.pt` is a dict with keys
  `{cfg, exp_id, format_version, model, optimizer, rng, run_id, saved_at_utc,
  scheduler, trainer, val_metrics}`; `trainer.global_step` holds the step counter;
  optimizer/scheduler/RNG states are inside — bitwise-exact resume is PROVEN infra
  (CV-G1-E0003).
- **Resume point**: the trainer auto-resumes from `experiments/<exp-id>/checkpoints/last.pt`
  when present (pass `--fresh` to ignore). To resume r3 after restore: put the newest
  good `.pt` at `experiments/r3/checkpoints/last.pt` (+ sidecars) and relaunch with the
  r3 argument set (§11). No other state is needed.
- **Moving-target warning**: the coordinator's morning measurement froze best.pt at
  global_step 8000, sha256 prefix `325ba9b1be3f3807` [MEASURED 2026-08-23 morning].
  At write-time recheck it had already advanced to step 9000 /
  sha256 prefix `C1C8478B0E649A32` / 185,324,952 bytes [MEASURED 2026-08-23].
  **The true final step/hash must be captured AFTER stopping the worker and BEFORE
  cutting the backup** — record it in the migration log. Size of every r3 `.pt`:
  185,324,952 bytes (185.3 MB) [MEASURED].

## 3. ARCHITECTURE + PARAMETERS

Model name `regressor-medium` (CAEDRAL-NAR B-arm: phoneme encoder → duration predictor
(supervised, CTC-aligner targets) → length regulator → mel decoder with conv front;
pitch/energy predictors present but FROZEN-OFF in v0 — documented rationale in code).
All values below from `experiments/r3/run_manifest.json` [MEASURED]:

- **params_total 15,942,707** · params_trainable 15,171,185
- Breakdown: decoder 10,355,328 · encoder 3,549,696 · conv_front 737,664 ·
  duration_predictor 385,761 · pitch_predictor 385,761 (off) · energy_predictor 385,761 (off) ·
  phoneme_embed 76,800 · conditioning 33,600 · mel_proj 30,800 · pitch_proj 768 · energy_proj 768
- d_model 384, n_heads 6 (n_kv_heads 6), ff_mult 4.0, dropout 0.0, RoPE on, non-causal
- n_layers_enc 2 · n_layers_dec 5 · decoder_conv_layers 1 (kernel 5)
- variance_hidden 160, variance_layers 2, kernel 3
- Audio: 24 kHz, 80-band ln-mel, n_fft 1024, hop 256
- Conditioning: speaker embedding table n_speakers 257 × dim 128 (FiLM); style table 10 × dim 64
- Effective phoneme vocab 43 entries (config pads table to 200)
- ONNX export EXISTS: `experiments/CV-G1-E0017-onnx-export/regressor-t{50,200,800}.onnx`,
  opset 17, ~60.7 MB fp32 each, parity max-abs-diff ≤2.7e-06 vs eager [MEASURED 2026-08-23]

## 4. OPTIMIZER / SCHEDULER / BATCH / SEED (exact r3 values)

| Field | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.98), weight_decay 1e-6 |
| Schedule | warmup 1000 steps, cosine decay to min_lr_ratio 0.1 |
| Batch | max_utts 16, max_frames 2000, bucket_size 32 (duration-bucketed) |
| Seed | 1234 (also seeds dataloader/RNG state saved in checkpoints) |
| grad_clip | 1.0 |
| val_frac | 0.05 → 42,068 train / 2,349 val utterances |
| Losses | spec_loss_weight 1.0 (L1 log-mel), dur_loss_weight 1.0 (MSE on log-durations) |
| Threads | 6 (torch CPU); box has 16 logical cores |
| Steps | 20,000 target; ckpt_every 1000; val_every 1000; log_every 100 |
| Versions | Python 3.13.15 · torch 2.6.0+cpu · Windows-11-10.0.26200 [from run_manifest] |

On Arch the versions WILL differ — pin equivalents in `bootstrap-dev.sh` (ARCH-PORT-01);
resume fidelity relies on saved RNG/optimizer/scheduler state, not on bit-identical libs,
but re-validate with the §12 smoke test before trusting a resumed long run.

## 5. DATA CONFIG

Training manifests (names resolvable by the loader under `data/manifests/`):

| Corpus | Manifest | Scale | Notes |
|---|---|---|---|
| librittsr-clean100 | `librittsr-clean100{,.phonemes,.durations,.rejected}.jsonl` + `.meta.json` | **44.11 h / 30,917 utts / 247 speakers** [MEASURED 2026-08-22, DATA-FULL-01] | Real speech; yield 82.4% of 53.55 h raw |
| teacher-kokoro | `teacher-kokoro{,.phonemes,.durations}.jsonl` + shard/meta files | Snapshot used by r3 ≈ **19.6 h** (63.68 h total_audio_h − 44.11 h) [MEASURED from run_manifest]; **FINAL merged corpus = 17.87 h / 15,498 wavs** post-snapshot [REPORTED — VERIFY AT MIGRATION TIME] | Kokoro-82M re-speaks clean100 texts (Apache-2.0, founder-authorized). Snapshot-vs-final drift is expected dedup/rejection cleanup, not corruption |

**Final teacher corpus ground truth [MEASURED 2026-08-23 write-time recheck]:**
`data/processed/teacher-kokoro/` holds **15,498 WAVs = 3.65 GB**, and
`data/manifests/teacher-kokoro.jsonl` has **exactly 15,498 rows** — 1:1 match.
`teacher-kokoro.phonemes.jsonl` currently 13,500 rows (phonemization/duration coverage
of the aligned subset). Raw sources: `data/raw/librittsr-full/` (8.62 GB extracted),
LJSpeech slice archives — all RE-DOWNLOADABLE (§13).

Speaker table: 257 entries (auto-expanded floor 40 → real corpora), serialized as
`speaker_index.json` NEXT TO THE CHECKPOINTS — required for restore.

## 6. ALIGNMENT / DURATIONS / CACHE STATE

- **Alignment provenance**: clean100 durations come from our OWN CTC aligner
  (CV-G1-E0011, ALIGN-01) → `data/manifests/librittsr-clean100.durations.jsonl`
  (30,917 rows). Teacher durations shipped with the teacher pipeline
  (CV-G1-E0012) → `teacher-kokoro.durations.jsonl` (**13,500 rows, 100% aligned**
  [MEASURED 2026-08-23]).
- **Merged training durations**: `data/manifests/all-durations.jsonl` —
  **44,417 rows** [MEASURED 2026-08-23; 30,917 + 13,500 = 44,417 exact]. r3 consumed
  all 44,417 (skipped_rows 275 were malformed/foreign ids).
- Format contract: rows `{id, durations:[int] (one per vocab token), total_frames}`;
  expansion contract `n_duration_adjusted=0` on every logged step (verified in metrics).
- **Mel disk cache**: `data/cache/mel-v1/` — sha256(audio-bytes+recipe)-keyed,
  interruption-safe (CV-G1-E0018). 30,917-entry warm baseline was 4.6 GiB; at
  write-time recheck **81,048 files / 5.86 GiB** (grows as r3 touches teacher mels)
  [MEASURED 2026-08-23]. **REGENERABLE** — never back this up as unique data; on Arch
  either restore it for an instant warm start or rebuild with
  `experiments/CV-G1-E0018-melcache/warm_cache.py`.

## 7. RELEASE GATES (what "done" means)

1. **Naturalness gate GATE-NAT-01 — FROZEN** (`docs/benchmarks/NATURALNESS-GATE.md`):
   median DNSMOS ovrl **≥ 3.15** AND min **≥ 2.85** over the frozen 10-sentence
   protocol, anchor = Kokoro median **3.450** measured on THIS machine 2026-08-22.
   Current Caedral state ≈ **DNSMOS ~2.7** [REPORTED by coordinator — VERIFY AT
   MIGRATION TIME] → **FAILING; E0013b quality iteration pending**. Re-run protocol is
   scripted end-to-end in the gate doc (works unchanged on Arch once deps land).
2. V1 bar (founder directive 2026-08-22): **≥15 distinct product-quality voices**
   (slate chosen: `docs/data/VOICE-SLATE-V1.md`), naturalness competitive with medium
   modern TTS (= the numeric gate above), full benchmark table
   (`benchmarks/results/natgate/comparison-template.json`), independent reviewers,
   packaged `artifacts/caedral-voice-v1/` + final report.
3. Longform stability gate: dual-ASR minWER population variance **≤ 0.0025** —
   current diagnostic prototype FAILS at 0.0173 (CV-G1-E0014) — must be re-gated on
   the trained regressor.
4. Speaker identity gate (calibrated in CV-G1-E0015): synth-vs-anchor cosine
   PASS ≥ 0.85 / review 0.80–0.85 / FAIL < 0.80; slate distinctness already PASSED
   (LOO 92.2%, min inter-anchor 0.566).

## 8. KNOWN ISSUES

- **K1 — Quality gap**: DNSMOS ~2.7 vs gate 3.15. The r3 regressor is intelligible but
  NOT product-quality; E0013b quality iteration is the active frontier. Do not ship
  previews as product (charter §26).
- **K2 — Buffered stdout hides logs until flush**: detached runs redirect stdout, and
  Python block-buffers it, so `.out.log` looks empty/stale for long stretches. Truth
  lives in `metrics.jsonl` (appends per step/val event) and `.err.log`. On Arch launch
  long runs with `python -u` (or `PYTHONUNBUFFERED=1`) to kill this class of confusion.
- **K3 — Session restarts kill detached children**: killing the coordinator session
  (Claude Code restart, terminal close) orphaned/killed "detached" children started
  with plain `Start-Process`. Pattern that survived on Windows: fully detached
  `cmd.exe /c start /b` chain + `.pid` file + a coordinator watchdog monitoring
  `metrics.jsonl` mtime/liveness. On Arch use `setsid nohup … &` (or a systemd user
  unit) + the same `.pid` + watchdog pattern; re-arm watchdogs on session start (§9 of
  OPENCODE-HANDOFF).
- **K4 — r3 trains on a pre-final teacher snapshot** (≈19.6 h vs final 17.87 h): the
  final corpus landed mid-run. Not a bug; expect the NEXT run (or an E0013b rerun) to
  consume the final merged corpus via `all-durations.jsonl`.
- **K5 — Moving checkpoint hashes**: any hash quoted for r3 while training runs is
  already stale. Freeze hashes only after the worker is stopped (HANDOFF §verify).
- **K6 — teacher phonemes coverage**: `teacher-kokoro.phonemes.jsonl` had 13,500 of
  15,498 rows at write time [MEASURED 2026-08-23] — verify whether the phonemizer job
  completed before migrating; if interrupted, finish it on Arch rather than redoing
  audio generation.

## 9. REMAINING V1 DAG (in priority order)

1. **Finish/stop r3** at best val (target 20k steps; early-stop allowed if val plateaus).
2. **Restore + resume on Arch** (OPENCODE-HANDOFF checklist), smoke-test first (§12 there).
3. **E0013b quality iteration** — close DNSMOS 2.7 → ≥3.15 (options queued in research
   docs: longer training, teacher-data weighting, decoder/vocoder upgrades). Gate: §7.1.
4. **NATGATE run for two V1 launch voices** — frozen protocol, fill
   `caedral-v1` row in the comparison template.
5. **Longform re-gate** on the trained model (variance ≤ 0.0025, worst-chunk ≤ 0.15).
6. **Voice slate listening gates** (Phase K/N, human) for the 15 picked speakers +
   persona/codename freeze (`docs/data/VOICE-SLATE-V1.md`).
7. **ONNX/vocoder chain packaging** — E0017 graphs exist; wire Vocos-class vocoder +
   frontend into a packaged CLI/server (RUNTIME track).
8. **A-arm control**: clean100 flow retrain (honest from-real-data comparison for the
   final report) — queued behind B-arm resources.
9. **Packaging + independent review + final report** → `artifacts/caedral-voice-v1/`,
   Phase P fresh-agent audit. DO NOT STOP until gates pass (founder directive).

## 10. EXACT SOURCE PATHS (Windows side, repo root `C:\Users\leona\Downloads\caedral-voice`)

Code/config/docs (git-tracked): `caedral_voice/` (model incl. `regressor.py`,
training incl. `regressor_dataset.py`, `mel_cache.py`), `configs/`,
`tests/` (131 passed at E0018 + 11 regressor + cache suite), `data/scripts/`,
`benchmarks/`, `docs/`, `experiments/EXPERIMENT-INDEX.md`.

Unique artifacts (NOT in git — must be backed up):

- Checkpoints: `experiments/r3/checkpoints/*` (best/last/step_*.pt + `phoneme_vocab.json`
  + `speaker_index.json`), `experiments/r2/checkpoints/*`, `experiments/r1/checkpoints/*`,
  `experiments/pilot-tiny-15k/*`, `experiments/CV-G1-E0017-onnx-export/regressor-t*.onnx`
- Manifests/durations: `data/manifests/**` (incl. `all-durations.jsonl`,
  `librittsr-clean100*`, `teacher-kokoro*`)
- Teacher audio: `data/processed/teacher-kokoro/**` (15,498 wavs, 3.65 GB)
  and sibling processed corpora under `data/processed/`
- Experiment history: `experiments/*/run_manifest.json`, `*/metrics.jsonl`, `*/logs/`,
  `*/progress.json`, `*/conclusion.md`, `.pid` files
- Eval assets: `benchmarks/corpus/eval-corpus-v1.0.json`, `benchmarks/results/natgate/**`,
  `artifacts/audio/**` (previews + anchors)

Regenerable (do NOT spend backup budget): `data/cache/mel-v1/**` (5.86 GiB),
`data/raw/**` downloads (8.62 GB librittsr-full + slices; deterministic re-download +
`data/scripts/extract_librittsr_full.py`), `.venv/`.

## 11. BACKUP GROUP PATHS

Five groups per founder spec, implemented by
`scripts/windows/build-backup-groups.ps1` [script authored on the Windows side; not
present in repo at this doc's write time — the script is the AUTHORITATIVE mapping;
VERIFY AT MIGRATION TIME]. Intended membership:

| Group | Contents (this doc's proposal) | Approx size |
|---|---|---|
| **VOICE-SOURCE** | Full git repo working tree + `.git` (code, configs, docs, tests, benchmarks corpus definitions) | few hundred MB |
| **VOICE-STATE** | Experiment state: all `run_manifest.json`, `metrics.jsonl`, `logs/`, `progress.json`, `.pid`, `conclusion.md`, EXPERIMENT-INDEX, natgate results JSON | small (<1 GB) |
| **VOICE-CHECKPOINTS** | Every `.pt` under `experiments/{r1,r2,r3,pilot-tiny-15k,...}/checkpoints/` + `phoneme_vocab.json` + `speaker_index.json` sidecars + E0017 `.onnx` graphs | ~2.9 GB (0.52 + 0.69 + 0.69 + 0.33 + 0.18) [MEASURED 2026-08-23] |
| **VOICE-UNIQUE-DATA** | `data/manifests/**` + `data/processed/**` (teacher-kokoro wavs 3.65 GB + clean100 processed) + voice-slate outputs; EXCLUDES mel cache and raw downloads | ~10.4 GB [MEASURED 2026-08-23] |
| **VOICE-PREVIEWS** | `artifacts/audio/**`, `artifacts/latest-preview/**` + provenance READMEs, natgate anchor clips | small |

## 12. RESTORE PATHS ON ARCH

- Repo root: `~/caedral-voice` (mirror the Windows tree 1:1; relative paths in
  manifests/run_manifests like `data/manifests/all-durations.jsonl` then work as-is).
  Absolute paths embedded in old run_manifests point at `C:\...` — cosmetic only; do
  not "fix" them retroactively.
- Python env: create `.venv` inside the repo root (launcher scripts assume
  `.venv/Scripts/python.exe` on Windows → becomes `.venv/bin/python` on Arch).
- Environment setup / package pins / thread-affinity notes:
  **[SEE bootstrap-dev.sh when ARCH-PORT-01 lands]** — until that script exists in the
  repo, the coordinator must assemble the env from `run_manifest.json` versions
  (Python 3.13, torch 2.6.0+cpu) and `requirements`/pyproject at repo root, then run
  the HANDOFF smoke test before any real work.
- Post-restore order: (1) verify artifacts (HANDOFF checklist + hashes),
  (2) smoke test 30 steps, (3) place newest good checkpoint, (4) resume r3,
  (5) re-arm watchdogs.

## 13. WHAT MUST NOT BE REGENERATED vs MAY BE REGENERATED

**MUST NOT BE REGENERATED (hours-to-days of compute or founder-scoped uniqueness;
if lost, the program loses weeks):**

- **Teacher corpus** — `data/processed/teacher-kokoro/` (15,498 Kokoro-rendered wavs)
  + its manifests/shards/rejected files. Founder-authorized teacher data; regeneration
  re-pays ~12 h of dual-shard generation and may not reproduce bit-exactly.
- **Durations & alignments** — `data/manifests/all-durations.jsonl`,
  `librittsr-clean100.durations.jsonl`, `teacher-kokoro.durations.jsonl`, and the
  CV-G1-E0011 CTC aligner checkpoints/artifacts behind them.
- **ALL checkpoints** — every `.pt` (r1/r2/r3/pilot/E0007/E0011 lineage) WITH their
  `phoneme_vocab.json` + `speaker_index.json` sidecars; E0017 ONNX graphs.
- **Manifests** — all phonemized manifests, meta/stats/rejected files (provenance).
- **Experiment history** — metrics.jsonl, run_manifests, conclusion.md files,
  EXPERIMENT-INDEX.md, natgate frozen anchor measurements (gate depends on them).

**MAY BE REGENERATED (deterministic pipelines exist; cheap relative to the above):**

- **Mel disk cache** `data/cache/mel-v1/` — sha-keyed, roundtrip-identity proven
  (max-abs-diff 0.0), warm script `experiments/CV-G1-E0018-melcache/warm_cache.py`
  rebuilds ~31k entries in ~8 min.
- **LibriTTS-R raw download + extraction** — deterministic pipeline
  (`data/scripts/extract_librittsr_full.py`); archives re-downloadable.
- **LJSpeech slice / smoke sets** — scripted downloads.
- **Previews** — `scripts/generate_v1_previews.py` / `generate_caedral_preview.py`.
- **Python venv** — rebuilt by bootstrap on Arch.

Rule of thumb: if a file encodes COMPUTE ALREADY PAID or MEASUREMENT PROVENANCE,
it is unique. If it is a pure function of a unique file, it may be regenerated.
