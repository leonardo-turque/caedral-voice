#!/usr/bin/env bash
# scripts/arch/bootstrap-dev.sh — Arch Linux bring-up for the Caedral Voice
# training core (CV-G1-E0019-archport).
#
# Goal: fresh Arch install -> this script -> gate tests green -> r3 resume.
# Run from the repository root:
#   ./scripts/arch/bootstrap-dev.sh [--with-data-deps]
#
# --with-data-deps additionally installs the optional data-preprocessing
# packages (soundfile/scipy/pyloudnorm/soxr/requests) commented out in
# requirements-arch.txt. The training core does not need them.

set -euo pipefail

WITH_DATA_DEPS=0
[ "${1:-}" = "--with-data-deps" ] && WITH_DATA_DEPS=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

echo "==> repo root: $REPO"

# ---------------------------------------------------------------------------
# 1. System packages (pacman)
# ---------------------------------------------------------------------------
# Justification per package (audit F6/F7 and import sweep):
#   git        - repo transfer/updates; several repo scripts shell out to
#                `git rev-parse` (benchmarks, experiments launchers)
#   base-devel - gcc/make toolchain; covers any rare sdist source build
#                (deliberately NOT installing cmake/ninja: every pinned dep
#                ships a binary wheel for cp313 — nothing here compiles)
#   python     - CPython interpreter hosting the venv (Arch rolling ~= 3.13.x,
#                matching the 3.13.15 baseline venv)
#   python-pip - rescue/system-level pip; `python -m venv` ensurepip normally
#                suffices inside the venv
#   ffmpeg     - audio insurance: torchaudio's bundled sox backend decodes WAV
#                without it, but ffmpeg keeps non-WAV corpora/baselines usable
#                (weak requirement — listed honestly as such)
#   libsndfile - runtime lib for soundfile (data/scripts pipeline); only
#                exercised with --with-data-deps / re-preprocessing
PKGS="git base-devel python python-pip ffmpeg libsndfile"
if [ "$WITH_DATA_DEPS" = "1" ]; then
    # libsndfile becomes a hard requirement in this mode
    :
fi
echo "==> installing system packages: $PKGS"
sudo pacman -Syu --needed $PKGS

# ---------------------------------------------------------------------------
# 2. Virtualenv + pinned Python deps
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
    echo "==> creating .venv"
    python -m venv .venv
fi
PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip >/dev/null
echo "==> pip install -r requirements-arch.txt (torch/torchaudio CPU wheels)"
"$PY" -m pip install -r requirements-arch.txt
if [ "$WITH_DATA_DEPS" = "1" ]; then
    echo "==> optional data-prep deps"
    "$PY" -m pip install "soundfile>=0.14,<1" "scipy>=1.18,<2" \
        "pyloudnorm>=0.2,<0.3" "soxr>=1.1,<2" "requests>=2,<3"
fi

# nltk/g2p_en data is intentionally NOT downloaded here: g2p_en is an
# INFERENCE-time and phonemization-time dependency only. Training reads frozen
# phoneme strings from data/manifests/*.phonemes.jsonl (verified by import
# sweep). See requirements-arch.txt if you ever need to re-phonemize.

# ---------------------------------------------------------------------------
# 3. Smoke test — gate tests must pass before any resume attempt
# ---------------------------------------------------------------------------
echo "==> smoke test: pytest gates"
"$PY" -X utf8 -m pytest tests/test_training.py tests/test_mel_cache.py \
    tests/test_regressor_pipeline.py -q

# Optional end-to-end micro-smoke (only meaningful with the dataset on disk;
# 30 steps x <=16 utts of real mel preprocessing, writes under arch-smoke/):
if [ -f data/manifests/librittsr-clean100.jsonl ] && \
   [ -d data/processed/librittsr-clean100 ]; then
    echo "==> dataset present — running 30-step synthetic-duration micro-smoke"
    "$PY" -X utf8 experiments/CV-G1-E0013-regressor/train_regressor.py \
        --manifests librittsr-clean100 --synthetic-durations \
        --steps 30 --threads 4 --exp-id CV-G1-E0013-regressor/arch-smoke \
        --fresh
else
    echo "==> dataset not present yet — skipping micro-smoke (pytest gates passed)"
fi

cat <<'EOF'

Bootstrap complete. Next step (after copying checkpoints into
experiments/r3/checkpoints/ and data/cache/mel-v1/) — see
experiments/CV-G1-E0013-regressor/RESUME-R3-LINUX.md:

  nohup .venv/bin/python -X utf8 \
      experiments/CV-G1-E0013-regressor/train_regressor.py \
      --manifests librittsr-clean100,teacher-kokoro \
      --durations data/manifests/all-durations.jsonl \
      --mel-cache-dir data/cache/mel-v1 \
      --steps 20000 --threads 6 --device cpu --exp-id r3 \
      >> experiments/r3/r3-linux.out.log 2>&1 &
EOF
