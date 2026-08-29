# RESUME-R3-LINUX — exact Arch resume sequence for the E0013 regressor

Continues run **`r3`** (launched 2026-08-23 on Windows 11, torch 2.6.0+cpu,
python 3.13.15, exp-id `r3`, currently stepping toward 20,000) on a fresh
Arch Linux install. Copy-paste in order; nothing else is assumed.

r3 ground truth (from `experiments/r3/run_manifest.json`, written at launch):
manifests `librittsr-clean100,teacher-kokoro`, durations
`data/manifests/all-durations.jsonl` (44,417 rows), mel disk cache ON
(`data/cache/mel-v1`), steps 20000, threads 6, seed 1234, every other knob at
its script default, 63.68 h / 44,417 utts / 257 speakers / vocab 43,
model regressor-medium 15,942,707 params.

## 0. What to transfer from this Windows box

| Path | Why |
|---|---|
| repo tree (git clone/copy) | code + configs + docs + this file |
| `data/manifests/` | manifests, `*.phonemes.jsonl`, **`all-durations.jsonl`** (15.2 MB) |
| `data/processed/librittsr-clean100/` | clean100 wavs (22.05 kHz) |
| `data/processed/teacher-kokoro/` | teacher wavs (24 kHz) |
| `experiments/r3/checkpoints/` | **the resume point** — copy `last.pt`, `best.pt`, newest `step_*.pt`, plus `phoneme_vocab.json` + `speaker_index.json`. Take files newest-at-copy-time (last.pt advances every 1000 steps) |
| `data/cache/mel-v1/` (~6.1 GB) | optional but recommended: skips the ~25-35 min mel re-warm. Keys are sha256(audio bytes + config fingerprint) and sidecars/npy are platform-neutral binaries — verified portable |

Manifest audio paths are repo-relative forward-slash strings and every
directory is lowercase — valid on a case-sensitive filesystem unchanged
(audit F9: no case traps). Do NOT copy `.venv` (rebuild it) or `__pycache__`.

## 1. Bootstrap (one command)

```bash
cd /path/to/caedral-voice        # any location; scripts are cwd-independent
./scripts/arch/bootstrap-dev.sh
```

Installs git, base-devel, python, python-pip, ffmpeg, libsndfile via pacman;
creates `.venv`; installs pinned CPU wheels (`torch==2.6.0+cpu`,
`torchaudio==2.6.0+cpu`, numpy>=2.5,<3, pytest>=9,<10) via
`requirements-arch.txt`; runs the pytest gates (`tests/test_training.py`,
`tests/test_mel_cache.py`, `tests/test_regressor_pipeline.py` — must print
`30 passed`) and a 30-step synthetic-duration micro-smoke when the dataset is
present.

No nltk/g2p_en data download is needed: training consumes frozen phoneme
strings from the manifests (g2p_en is inference/phonemization-only — import
sweep verified).

## 2. Verify the checkpoint landed

```bash
ls experiments/r3/checkpoints/           # last.pt present
.venv/bin/python - <<'PY'
import torch
m = torch.load("experiments/r3/checkpoints/last.pt", map_location="cpu",
               weights_only=False)
print("step", m["trainer"]["global_step"], "epoch", m["trainer"]["epoch"],
      "format", m["format_version"])
PY
```

Expect `format 1` and the global_step of the newest Windows-side checkpoint
(r3 passed step 9000 on 2026-08-23).

Optional cache sanity check:

```bash
.venv/bin/python -c \
  "from caedral_voice.training.mel_cache import MelDiskCache; print(MelDiskCache().scan_stale())"
# expect {'entries': ~44k, 'stale': 0, ...}
```

## 3. The resume command

```bash
nohup .venv/bin/python -X utf8 \
    experiments/CV-G1-E0013-regressor/train_regressor.py \
    --manifests librittsr-clean100,teacher-kokoro \
    --durations data/manifests/all-durations.jsonl \
    --mel-cache-dir data/cache/mel-v1 \
    --steps 20000 --threads 6 --device cpu --exp-id r3 \
    >> experiments/r3/r3-linux.out.log 2>&1 &
```

Argument provenance (must not drift — mirrors `run_manifest.json` args):
`--manifests librittsr-clean100,teacher-kokoro`;
`--durations data/manifests/all-durations.jsonl`;
`--mel-cache-dir data/cache/mel-v1`; `--steps 20000`; seed/lr/warmup/
min-lr-ratio/max-frames/max-utts/bucket/ckpt-every/val-every/log-every/
grad-clip/weight-decay left at script defaults == manifest values;
**`--threads 6`** (thread count participates in the determinism context);
**`--device cpu`** pins the CPU path so an unexpected CUDA-capable Arch box
cannot silently switch device mid-run. No `--fresh`: auto-resumes
`experiments/r3/checkpoints/last.pt`.

## 4. Post-launch verification

Within ~10 min of startup check the log for:

```
[resume] experiments/r3/checkpoints/last.pt -> step <N> epoch <E> ...
[data] librittsr-clean100: ... utt ...
[data] usable utterances=44417 audio=63.680 h
[data] speakers=257 vocab_size=43
[model] regressor-medium params=15,942,707 (trainable 15,171,185) pitch=off energy=off
[melcache] ... hits=... computed=...
```

then confirm `experiments/r3/metrics.jsonl` grows and no
`nan_loss_skipped`/`nonfinite_grad_skipped` events appear in the first window.
Reference trajectory at step 9800 (Windows side): loss ~1.75,
loss_window_avg ~1.61, lr ~6.02e-05.

## Cross-platform determinism caveat (read once)

The E0003 "exact resume" contract is bitwise within one OS + torch build.
Windows->Linux restores optimizer/scheduler/RNG/data-order state exactly, but
CPU floating-point kernel implementations can differ between BLAS builds, so
the loss trajectory may show small numerical deltas after the resume point
while remaining statistically equivalent. If bit-faithful continuation ever
matters, it must happen on the original platform. Keep `--threads 6` either
way.

## Optional: warm/extend the mel cache

If you skipped transferring `data/cache/mel-v1/`, pre-warm before resuming so
the trainer never pays decode cost inline:

```bash
nohup .venv/bin/python -X utf8 \
    experiments/CV-G1-E0018-melcache/warm_cache.py \
    --manifest librittsr-clean100 >> experiments/r3/warm.out.log 2>&1 &
```

The warmer already has a POSIX branch (`os.nice(10)` on non-Windows).
Cache entries are keyed by audio bytes + config fingerprint only — no OS or
path information participates, so a copied cache serves identically on Arch.
