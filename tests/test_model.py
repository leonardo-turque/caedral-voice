"""CAEDRAL-NAR prototype tests (Phase F, CV-G1-E0002).

Contracts under test (untrained stub — no quality claims):
- parameter-count targets {8M, 30M, 80M} within ±15% for shipped configs;
- forward shapes (mel, waveform, tracks) incl. batching;
- seed determinism (same seed -> bitwise-identical outputs; repeated
  forward on one instance -> identical);
- speaker-conditioning and style-conditioning sensitivity;
- prosody-knob semantics (speed / pitch_shift / energy_gain transform the
  predicted tracks, per MULTI-SPEAKER-PROSODY-STYLE.md section 8.2).

Run: .venv/Scripts/python.exe -m pytest tests/test_model.py -q
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from caedral_voice.model import CaedralNAR, CaedralNARConfig, load_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TARGETS = {"tiny": 8_000_000, "lite": 30_000_000, "quality": 80_000_000}


def _ids(n: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 200, (1, n), generator=g)


# --------------------------------------------------------------------- #
# parameter targets
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["tiny", "lite", "quality"])
def test_param_targets_within_15pct(name):
    model = load_model(CONFIGS / f"{name}.json")
    total = model.count_params()
    target = TARGETS[name]
    assert abs(total - target) <= 0.15 * target, (
        f"{name}: {total:,} vs target {target:,}"
    )
    assert model.count_params() == sum(p.numel() for p in model.parameters())
    breakdown = model.param_breakdown()
    assert sum(breakdown.values()) == total


# --------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------- #
def test_forward_shapes_batch1():
    model = load_model(CONFIGS / "tiny.json").eval()
    out = model(_ids(40))
    t_mel = out["n_mel_frames"]
    hop = model.config.hop_length
    assert out["mel"].shape == (1, t_mel, 80)
    assert out["waveform"].shape == (1, t_mel * hop)
    assert out["durations"].shape == (1, 40)
    assert out["pitch_hz"].shape == (1, t_mel)
    assert out["energy_db"].shape == (1, t_mel)
    assert t_mel == int(out["durations"].sum())
    assert torch.isfinite(out["waveform"]).all()
    assert torch.isfinite(out["mel"]).all()


def test_forward_shapes_batch2():
    model = load_model(CONFIGS / "tiny.json").eval()
    ids = torch.stack([_ids(30, seed=1)[0], _ids(30, seed=2)[0]])
    out = model(ids, speaker_id=torch.tensor([0, 3]))
    b, t_mel, n_mels = out["mel"].shape
    assert (b, n_mels) == (2, 80)
    assert out["waveform"].shape == (2, t_mel * model.config.hop_length)
    assert out["durations"].shape == (2, 30)


def test_duration_override_shapes():
    model = load_model(CONFIGS / "tiny.json").eval()
    d = torch.full((1, 10), 4, dtype=torch.long)
    out = model(_ids(10), durations=d)
    assert out["n_mel_frames"] == 40
    assert out["waveform"].shape == (1, 40 * model.config.hop_length)


# --------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------- #
def test_seed_determinism():
    ids = _ids(25)
    m1 = load_model(CONFIGS / "tiny.json", seed=123)
    m2 = load_model(CONFIGS / "tiny.json", seed=123)
    m3 = load_model(CONFIGS / "tiny.json", seed=124)
    with torch.no_grad():
        o1 = m1(ids)["waveform"]
        o2 = m2(ids)["waveform"]
        o1_again = m1(ids)["waveform"]
        o3 = m3(ids)["waveform"]
    assert torch.equal(o1, o2), "same seed must reproduce bitwise"
    assert torch.equal(o1, o1_again), "repeat forward must be identical"
    assert not torch.equal(o1, o3), "different seed must change init"


def test_rope_and_gqa_variants_forward():
    cfg = CaedralNARConfig(name="probe", d_model=128, n_heads=4, n_kv_heads=1,
                           n_layers_enc=1, n_layers_dec=1, variance_hidden=64,
                           head_dim=128, head_blocks=1)
    model = CaedralNAR(cfg).eval()
    cfg_no_rope = CaedralNARConfig(**{**cfg.__dict__, "rope": False})
    model_norope = CaedralNAR(cfg_no_rope).eval()
    # same durations for both so mel shapes are directly comparable
    durs = torch.full((1, 12), 3, dtype=torch.long)
    out_rope = model(_ids(12), durations=durs)
    out_norope = model_norope(_ids(12), durations=durs)
    assert out_rope["waveform"].shape == out_norope["waveform"].shape
    assert not torch.equal(out_rope["mel"], out_norope["mel"])


# --------------------------------------------------------------------- #
# conditioning sensitivity
# --------------------------------------------------------------------- #
def test_speaker_conditioning_sensitivity():
    model = load_model(CONFIGS / "tiny.json").eval()
    ids = _ids(30)
    # fixed duration override: isolates conditioning effects from the
    # legitimate per-speaker duration differences (which change T_mel)
    durs = torch.full((1, 30), 2, dtype=torch.long)
    with torch.no_grad():
        a = model(ids, speaker_id=0, durations=durs)
        b = model(ids, speaker_id=7, durations=durs)
    diff = (a["mel"] - b["mel"]).norm() / a["mel"].norm()
    assert diff > 1e-3, f"speaker switch barely changes mel (rel diff {diff:.2e})"
    assert not torch.allclose(a["pitch_hz"], b["pitch_hz"])
    assert not torch.allclose(a["energy_db"], b["energy_db"])


def test_style_conditioning_sensitivity():
    model = load_model(CONFIGS / "tiny.json").eval()
    ids = _ids(30)
    durs = torch.full((1, 30), 2, dtype=torch.long)
    with torch.no_grad():
        neutral = model(ids, style_id=None, durations=durs)["mel"]  # None -> neutral row
        s3 = model(ids, style_id=3, durations=durs)["mel"]
        s8 = model(ids, style_id=8, durations=durs)["mel"]
    d_neutral_vs_3 = (neutral - s3).norm() / neutral.norm()
    d_3_vs_8 = (s3 - s8).norm() / s3.norm()
    assert d_neutral_vs_3 > 1e-3, f"style vs neutral barely differs ({d_neutral_vs_3:.2e})"
    assert d_3_vs_8 > 1e-3, f"two styles barely differ ({d_3_vs_8:.2e})"


# --------------------------------------------------------------------- #
# prosody knobs transform tracks (research doc section 8.2)
# --------------------------------------------------------------------- #
def test_speed_scales_durations():
    model = load_model(CONFIGS / "tiny.json").eval()
    ids = _ids(50)
    raw = torch.exp(model(ids)["log_duration"])  # pre-clamp predicted durations
    fast = model(ids, speed=2.0)
    slow = model(ids, speed=0.5)

    def expected(spd: float) -> int:
        d = torch.round(raw / spd).clamp(1, model.config.max_duration)
        return int(d.sum())

    assert fast["n_mel_frames"] < slow["n_mel_frames"]
    assert fast["n_mel_frames"] == expected(2.0)
    assert slow["n_mel_frames"] == expected(0.5)


def test_pitch_shift_and_energy_gain_tracks():
    model = load_model(CONFIGS / "tiny.json").eval()
    ids = _ids(20)
    base = model(ids)
    up = model(ids, pitch_shift=3.0)      # +3 semitones
    loud = model(ids, energy_gain=6.0)    # +6 dB
    assert torch.allclose(
        up["pitch_hz"], base["pitch_hz"] * 2 ** (3.0 / 12.0), rtol=1e-5
    )
    assert torch.allclose(loud["energy_db"], base["energy_db"] + 6.0, atol=1e-4)
    assert not torch.equal(up["mel"], base["mel"])
    assert not torch.equal(loud["mel"], base["mel"])
