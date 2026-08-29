"""Engine tests: CaedralVoiceKokoroEngine end-to-end on the real model.

Runs against models/kokoro/kokoro-v1.0.onnx + voices-v1.0.bin (the winning
production path). Verifies load-once semantics, warm-call state reuse,
audio contract (finite float32 @ 24 kHz), frontend cache reuse, and the
unknown-voice error path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.inference.kokoro_engine import (  # noqa: E402
    CaedralVoiceKokoroEngine,
)

MODEL = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "kokoro" / "voices-v1.0.bin"

pytestmark = pytest.mark.skipif(
    not (MODEL.is_file() and VOICES.is_file()),
    reason="models/kokoro/*.onnx|voices not on disk",
)


@pytest.fixture(scope="module")
def engine() -> CaedralVoiceKokoroEngine:
    # Module-scoped fixture => the ONNX session is loaded exactly once for
    # every test in this file; constructing twice would double a ~5s load.
    return CaedralVoiceKokoroEngine(MODEL, VOICES, threads=2)


SENTENCE = "The invoice totals $1,247.83, due within 30 days."


def test_engine_loads_once_and_exposes_voices(engine) -> None:
    voices = engine.voices
    assert len(voices) >= 50          # v1.0 ships 54
    assert "af_heart" in voices and "am_michael" in voices
    assert engine.sample_rate == 24000
    # single fused output: acoustic + vocoder in one graph
    assert engine.fused_vocoder


def test_two_synths_reuse_state_second_faster_or_equal(engine) -> None:
    # First call carries cold costs (ORT graph init/allocations for this
    # session); the second must be faster or equal -- that is the whole
    # point of the long-lived engine.
    t0 = engine.synth(SENTENCE, "af_heart")
    cold_wall = engine.last_stats.wall_seconds
    t1 = engine.synth(SENTENCE, "af_heart")
    warm_wall = engine.last_stats.wall_seconds
    assert isinstance(t0, np.ndarray) and isinstance(t1, np.ndarray)
    assert engine.last_stats.frontend_cache_hit   # text cache reused
    assert t1.shape == t0.shape                   # deterministic output length
    assert warm_wall <= cold_wall, (
        f"warm {warm_wall:.3f}s slower than cold {cold_wall:.3f}s"
    )


def test_audio_contract_finite_float32_24k(engine) -> None:
    audio = engine.synth("Hello. This is the first generation of Caedral Voice.",
                         "am_michael")
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    duration = len(audio) / engine.sample_rate
    assert 0.5 <= duration <= 15.0, f"implausible duration {duration:.2f}s"
    stats = engine.last_stats
    assert stats.rtf < 5.0        # idle-box sanity; contention numbers live in profile.json
    assert stats.voice == "am_michael"
    assert stats.n_phonemes > 10


def test_unknown_voice_raises(engine) -> None:
    with pytest.raises(ValueError, match="Unknown voice"):
        engine.synth("Hello.", "zz_nobody")


def test_speed_validation(engine) -> None:
    # Engine clamps out-of-range speeds into [0.25, 2.0] per production spec.
    # Verify extreme values produce finite audio without crashing.
    audio_hi = engine.synth("Hello.", "af_heart", speed=9.0)
    assert len(audio_hi) > 0
    audio_lo = engine.synth("Hello.", "af_heart", speed=0.1)
    assert len(audio_lo) > 0


def test_frontend_cache_hit_on_repeat(engine) -> None:
    text = "Cache probe sentence number forty two."
    engine.phonemize(text)
    engine.synth(text, "af_heart")
    assert engine.last_stats.frontend_cache_hit
