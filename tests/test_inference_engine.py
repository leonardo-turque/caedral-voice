"""CLI-01 smoke tests: CaedralEngine end-to-end + CLI surface.

Runs against the real smoke-a pilot checkpoint on disk (step 60 of
CV-G1-E0007-pilot-training): engine load, voices catalog, one full
frontend -> flow infer -> Vocos decode synthesis, stats reporting, and
`python -m caedral_voice.inference.cli --help`. Quality is irrelevant at
this stage; the deliverable is the integrated path executing today.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.inference import CaedralEngine  # noqa: E402

SMOKE_RUN = REPO / "experiments" / "CV-G1-E0007-pilot-training" / "smoke-a"
SMOKE_CKPT = SMOKE_RUN / "checkpoints" / "step_000000060.pt"

pytestmark = pytest.mark.skipif(
    not SMOKE_CKPT.is_file(), reason="smoke-a step_000000060.pt not on disk"
)


@pytest.fixture(scope="module")
def engine() -> CaedralEngine:
    return CaedralEngine.from_run(SMOKE_RUN, SMOKE_CKPT, threads=4)


def test_engine_loads_and_exposes_voices(engine: CaedralEngine) -> None:
    voices = engine.voices
    assert len(voices) >= 1
    for row in voices:
        assert set(row) == {"id", "speaker_name", "index"}
    assert voices[0]["index"] == 0
    assert engine.sample_rate == 24000
    assert engine.checkpoint_path == SMOKE_CKPT


def test_synth_sentence_produces_valid_audio(engine: CaedralEngine) -> None:
    # Long enough that the ~1 frame/phoneme duration predictor at step 60
    # still clears the 0.5 s floor (measured ~1.1 s for this sentence).
    audio = engine.synth(
        "The engine loads the pilot checkpoint, resolves the voice "
        "catalog, and synthesizes audio without any manual glue.", 0,
    )
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert engine.sample_rate == 24000
    duration_s = len(audio) / engine.sample_rate
    assert 0.5 <= duration_s <= 10.0, f"implausible duration {duration_s:.2f}s"


def test_synth_reports_rtf(engine: CaedralEngine) -> None:
    engine.synth("Hello there.", 0)
    stats = engine.last_stats
    assert stats is not None
    assert stats.rtf > 0.0 and math.isfinite(stats.rtf)
    assert stats.n_phonemes > 0
    assert stats.voice_index == 0


def test_unknown_voice_raises(engine: CaedralEngine) -> None:
    with pytest.raises(ValueError, match="unknown voice"):
        engine.synth("Hi.", 999)


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "caedral_voice.inference.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--voice", "--text", "--out", "--run-dir", "--checkpoint",
                 "--speed", "--pitch", "--energy", "--threads"):
        assert flag in proc.stdout, f"missing {flag} in --help"
