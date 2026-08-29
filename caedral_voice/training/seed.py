"""Determinism helpers: seeding, RNG snapshot/restore.

``PYTHONHASHSEED`` can only be fixed at interpreter launch; :func:`set_determinism`
reports what it is set to rather than pretending it can change it at runtime.
The training loop does not depend on set/dict iteration order of Python strings
for numerics, so hash randomization does not affect trajectory equality in
practice on CPython for the ops used here.
"""

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch


def set_determinism(seed: int,
                    deterministic_algorithms: bool = True,
                    num_threads: int | None = 1) -> dict:
    """Seed python ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices).

    Also pins intra-op threads (default 1) so CPU reduction order is stable and
    tiny-model runs are not throttled by thread oversubscription, and optionally
    enables ``torch.use_deterministic_algorithms(True)``.

    Returns a seed info dict (seed, flags, previous flags, versions) suitable
    for dropping into a run manifest.
    """
    seed = int(seed)
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_threads = torch.get_num_threads()

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)

    cuda = {"available": torch.cuda.is_available(), "devices_seeded": 0}
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda["devices_seeded"] = torch.cuda.device_count()
    if deterministic_algorithms:
        # Required by some CUDA kernels for determinism; harmless on CPU-only.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    elif prev_det:
        torch.use_deterministic_algorithms(False)
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))

    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "prev_deterministic_algorithms": prev_det,
        "num_threads": torch.get_num_threads(),
        "prev_num_threads": prev_threads,
        "pythonhashseed_env": os.environ.get(
            "PYTHONHASHSEED", "<unset; fixable only at interpreter launch>"),
        "cuda": cuda,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def get_rng_state() -> dict:
    """Snapshot every global RNG used by the training loop into a picklable dict."""
    state = {
        "torch": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state


def set_rng_state(state: dict) -> None:
    """Restore a dict produced by :func:`get_rng_state` exactly."""
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
