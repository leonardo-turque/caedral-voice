"""CAEDRAL inference runtime (RUNTIME-01 v0 core, task CLI-01).

Public surface (v0):
    CaedralEngine.from_run(run_dir, checkpoint_path=None, config_path=None)
        .voices                       -> [{id, speaker_name, index}, ...]
        .synth(text, voice_index, ...) -> np.float32 waveform @ 24 kHz
        .last_stats                    -> SynthStats of the last synth()

Design doc: docs/inference/RUNTIME-DESIGN.md. This package implements the
single-process CPU synthesis core (engine + CLI); the HTTP/streaming layers
of the design doc arrive in later phases.

The legacy NAR ``CaedralEngine`` imports torch at module scope; it resolves
lazily via PEP 562 so the production Kokoro runtime can ship without the
training stack.
"""
from __future__ import annotations

__all__ = ["CaedralEngine", "SynthStats"]


def __getattr__(name: str):
    if name in ("CaedralEngine", "SynthStats"):
        from . import engine as _engine

        return getattr(_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
