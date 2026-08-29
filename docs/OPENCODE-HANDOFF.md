# OPENCODE HANDOFF — read this FIRST (Caedral Voice, post Windows-wipe)

Written 2026-08-23 for the incoming coordinator (OpenCode on Arch Linux). The Windows
machine this program ran on was wiped; you are resuming from restored backups into
`~/caedral-voice`. Deep reference: `docs/releases/PRE-ARCH-VOICE-MIGRATION.md`.

---

## THE FOUR PROHIBITIONS (verbatim, founder-ordered)

> **DO NOT START FROM ZERO**
>
> **DO NOT REDO RESEARCH**
>
> **DO NOT REBUILD ALIGNMENT IF VALID ARTIFACTS WERE RESTORED**
>
> **DO NOT REGENERATE TEACHER DATA IF THE FINAL CORPUS WAS RESTORED**

Phase A–I are COMPLETE. ~500 cited research sources, a frozen evaluation gate, proven
training infrastructure (bitwise-exact resume), a final teacher corpus, valid CTC-aligner
durations, and trained checkpoints already exist. Your job is RESUME + quality iteration,
not rediscovery. If a restored artifact verifies (checklist below), it is canonical —
never rebuild it "to be safe."

## 1. READ-FIRST (in this order)

1. `docs/OPENCODE-HANDOFF.md` — this file.
2. `docs/CURRENT-STATE-REPORT.md` — program phases, capability inventory, honest "NO
   product voice yet" statement.
3. `docs/EXECUTION-QUEUE.md` — standing rules + the founder V1 directive and active
   arms (B-arm primary).
4. `experiments/EXPERIMENT-INDEX.md` — every experiment E0001–E0018 with measured
   conclusions; do not rerun what is already COMPLETE.
5. `docs/benchmarks/NATURALNESS-GATE.md` — the FROZEN numeric gate: DNSMOS median
   ≥ 3.15 AND min ≥ 2.85 vs Kokoro anchor median 3.450 measured on the original machine.

Then, as needed: `docs/data/VOICE-SLATE-V1.md` (the 15 chosen voices),
`docs/releases/PRE-ARCH-VOICE-MIGRATION.md` (full state snapshot).

## 2. STEP 0 — VERIFY RESTORED ARTIFACTS before anything else

Run these from `~/caedral-voice`. Every check must PASS before training work. Hashes
marked FROZEN are stable; r3's were moving at doc-write time — the stop-time values
must come from the migration log / your own measurement.

| # | Check | Expected |
|---|---|---|
| V1 | r2 fallback checkpoint | `experiments/r2/checkpoints/best.pt`: exactly **185,324,952 bytes**; sha256 prefix **`C55AC1E909074272`** (FROZEN — run completed 2026-08-23T10:57:12Z); loads with global_step **15000**, val_loss **0.8647432296943292** |
| V2 | r3 best checkpoint | `experiments/r3/checkpoints/best.pt`: **185,324,952 bytes** per file; morning-fact sha prefix was `325ba9b1be3f3807` @ step 8000 [MEASURED 2026-08-23 AM]; last observed while still training: `C1C8478B0E649A32` @ step 9000, val_loss 0.84872. **[VERIFY AT MIGRATION TIME]** — accept any step ≥ 9000 whose hash is recorded in the migration log |
| V3 | checkpoint anatomy | torch-load prints keys `{cfg, model, optimizer, rng, scheduler, trainer, val_metrics, ...}` and `trainer.global_step` = expected step |
| V4 | sidecars present | `phoneme_vocab.json` + `speaker_index.json` sit NEXT TO every checkpoint dir you intend to use |
| V5 | teacher corpus | `data/processed/teacher-kokoro/`: **15,498 wavs / 3.65 GB**; `data/manifests/teacher-kokoro.jsonl`: **15,498 rows** (exact match) |
| V6 | durations | `data/manifests/all-durations.jsonl`: **44,417 rows** (= clean100 30,917 + teacher 13,500); `teacher-kokoro.durations.jsonl`: **13,500 rows, 100% aligned** |
| V7 | manifests | `librittsr-clean100.jsonl` 30,917 rows; `.phonemes/.durations/.rejected` siblings present; `teacher-kokoro.phonemes.jsonl` ≥ 13,500 rows (was mid-fill at write time) [VERIFY AT MIGRATION TIME] |
| V8 | eval assets | `benchmarks/corpus/eval-corpus-v1.0.json`, `benchmarks/results/natgate/gate-summary.json` (anchor 3.450), comparison template present |

Load-check snippet (V3):

```bash
python -X utf8 -c "import torch; ck=torch.load('experiments/r3/checkpoints/best.pt', map_location='cpu', weights_only=False); print(sorted(ck.keys())); print('step:', ck['trainer']['global_step']); print(ck['val_metrics'])"
```

If ANY unique artifact fails verification: STOP, report, consult backups — do not
regenerate (see prohibitions).

## 3. CPU ENVIRONMENT DETECTION (first commands on Arch)

```bash
lscpu | grep -oE 'avx2|fma3|avx512[a-z_]*' | sort -u   # original box: avx2+fma3, NO avx-512
nproc                                                   # original: 16 logical / 10 physical cores
free -h                                                 # original: 23.7 GB RAM
.venv/bin/python -c "import torch; print(torch.__version__, torch.get_num_threads())"
```

Expectations: torch CPU build (original 2.6.0+cpu, Python 3.13). Thread guidance:
training used **6 threads**; inference/eval sweet spot 4 threads (4 threads HURT short
utterances in E0002 — keep batch-1 latency paths single/low-thread). Env assembly pins:
**[SEE bootstrap-dev.sh when ARCH-PORT-01 lands]** — until it lands, derive deps from
repo root packaging files + the versions above, then prove the env with §4.

## 4. SMOKE TEST (mandatory before trusting the restore)

30-step regressor run under a THROWAWAY experiment id (never `r3` — auto-resume would
touch real checkpoints):

```bash
cd ~/caedral-voice && .venv/bin/python -X utf8 \
  experiments/CV-G1-E0013-regressor/train_regressor.py \
  --manifests librittsr-clean100 --durations data/manifests/all-durations.jsonl \
  --steps 30 --threads 4 --exp-id smoke-post-migration \
  --mel-cache-dir data/cache/mel-v1
```

PASS criteria (from the original smoke, CV-G1-E0013): runs to step 30, **0 NaN events**,
train mel-L1 trending down (~1.68 → lower on cold start), duration MSE falling,
`n_duration_adjusted=0` on logged steps, metrics appended under
`experiments/smoke-post-migration/metrics.jsonl`. First steps pay an audio-decode tax if
the mel cache was not restored (warm cache makes this fast; either way it must COMPLETE,
not crash). Delete `experiments/smoke-post-migration/` afterwards.

## 5. RESTORE LATEST-GOOD CHECKPOINT

Preference order:

1. **r3 `best.pt`** if it verifies (V2/V3) with step ≥ 9000 and val_loss < 0.865 —
   this is the live frontier.
2. Else **r2 `best.pt`** (previous-good fallback: val mel-L1 **0.86473** @ 15k steps,
   hash `C55AC1E909074272`) — frozen, always trustworthy.

Steps:

```bash
mkdir -p experiments/r3/checkpoints
cp <backup>/r3/checkpoints/best.pt        experiments/r3/checkpoints/last.pt
cp <backup>/r3/checkpoints/best.pt        experiments/r3/checkpoints/best.pt   # optional seed
cp <backup>/r3/checkpoints/phoneme_vocab.json experiments/r3/checkpoints/
cp <backup>/r3/checkpoints/speaker_index.json experiments/r3/checkpoints/
```

The trainer auto-resumes from `experiments/<exp-id>/checkpoints/last.pt`; optimizer +
scheduler + RNG states ride inside the file, so no other state files are needed.

## 6. RESUME TRAINING (r3 continuation)

Match r3's exact argument set (`experiments/r3/run_manifest.json`):

```bash
setsid nohup .venv/bin/python -u -X utf8 \
  experiments/CV-G1-E0013-regressor/train_regressor.py \
  --manifests librittsr-clean100,teacher-kokoro \
  --durations data/manifests/all-durations.jsonl \
  --steps 20000 --threads 6 --exp-id r3 \
  --mel-cache-dir data/cache/mel-v1 \
  >> experiments/r3/resume.out.log 2>> experiments/r3/resume.err.log &
echo $! > experiments/r3/r3.pid
```

Notes: `-u` fixes the Windows-era stdout-buffering trap; `setsid nohup … &` detaches so
a coordinator restart cannot kill it (Windows equivalent that failed is documented in
MIGRATION K3); remaining budget ≈ 20k − current step at ~1.55 s/step [ESTIMATED].

## 7. WATCHDOG RESTORATION

The previous coordinator ran monitor loops (pid-file liveness + `metrics.jsonl`
freshness). They DIE with the session by design — re-arm them at the start of EVERY
coordinator session, e.g. watch that `experiments/r3/metrics.jsonl` grows every few
minutes and alert on process exit or stale mtime; log-only watchdogs are useless during
long detached runs. Keep the `.pid`-file convention (`experiments/*/…pid`). On completion
or kill: freeze hashes (sha256sum of all `.pt`), append results to EXPERIMENT-INDEX.md,
commit artifacts per standing rules.

## 8. OPEN PLACEHOLDERS — coordinator MUST fill at migration time

- Exact stop-time r3 step + val_loss + sha256 for every r3 `.pt` [VERIFY AT MIGRATION TIME]
- Final teacher-corpus hour figure (17.87 h REPORTED; recount from manifests) and whether
  `teacher-kokoro.phonemes.jsonl` reached all 15,498 rows
- Current DNSMOS of the newest Caedral checkpoint (~2.7 REPORTED — re-measure under the
  frozen protocol before planning E0013b)
- Environment pins + launch wrapper: **[SEE bootstrap-dev.sh when ARCH-PORT-01 lands]**
- Authoritative backup-group mapping: `scripts/windows/build-backup-groups.ps1` output
  manifest (script authored on the Windows side; verify which groups landed on the backup
  medium and where they mount on Arch)
