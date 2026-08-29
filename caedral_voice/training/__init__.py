"""caedral_voice.training — reproducible, crash-safe training infrastructure.

Phase I (experiment CV-G1-E0003): the system is proven on synthetic tasks;
no real TTS training runs through this package yet.

Public surface
--------------
- :func:`set_determinism` -- seed python/numpy/torch, pin threading, return info dict.
- :func:`get_rng_state` / :func:`set_rng_state` -- full RNG snapshot/restore.
- :class:`Trainer` -- fit/resume/validate with atomic checkpoints, NaN guard,
  warmup+cosine schedule support, JSONL metrics log.
- :func:`build_scheduler` -- linear-warmup-then-cosine-decay LR schedule.
- :func:`load_config` -- cfg dict from JSON (merged over DEFAULT_CFG).
- :func:`read_jsonl` -- read back metrics/events logs in tests and proofs.
"""

from __future__ import annotations

from .seed import get_rng_state, set_determinism, set_rng_state
from .trainer import (
    DEFAULT_CFG,
    FORMAT_VERSION,
    Trainer,
    build_scheduler,
    load_config,
    read_jsonl,
    sha256_file,
)

__all__ = [
    "DEFAULT_CFG",
    "FORMAT_VERSION",
    "Trainer",
    "build_scheduler",
    "get_rng_state",
    "load_config",
    "read_jsonl",
    "set_determinism",
    "set_rng_state",
    "sha256_file",
]
