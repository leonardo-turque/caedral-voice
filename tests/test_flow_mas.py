"""Flow decoder + MAS tests (Phase F+, CV-G1-E0006).

Contracts under test:
- invertibility: inverse(forward(x)) == x within 1e-3 (with and without
  per-frame cond / global vec conditioning);
- log-det consistency: flow's accumulated log-det equals the autograd
  Jacobian's slogdet on a tiny case (float64);
- MAS: synthetic cost matrices with known optimal durations; fast row-scan
  DP returns the EXACT optimum of the textbook nested-loop DP on random
  matrices incl. ragged batch lengths; monotonicity + full frame coverage;
- determinism: same seed -> bitwise-equal training losses;
- integration: one Adam training run on random tensors reduces flow_loss,
  keeps every loss finite, and pushes gradients into flow/prior/duration
  parameters; flow-mode infer shapes + noise_scale semantics.

Run: .venv/Scripts/python.exe -m pytest tests/test_flow_mas.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.model import (  # noqa: E402
    CaedralNAR,
    CaedralNARConfig,
    FlowDecoder,
    Invertible1x1Conv,
    maximum_path,
    maximum_path_reference,
)


def _tiny_flow(**kw) -> FlowDecoder:
    torch.manual_seed(kw.pop("seed", 0))
    cfg = dict(
        n_mels=80, hidden_channels=32, n_blocks=3, cond_dim=16, vec_dim=8
    )
    cfg.update(kw)
    return FlowDecoder(**cfg)


# --------------------------------------------------------------------- #
# invertibility
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("use_cond", [False, True])
def test_invertibility_roundtrip_within_1e3(use_cond):
    f = _tiny_flow().double()
    mel = torch.randn(2, 80, 37, dtype=torch.float64)
    cond = torch.randn(2, 37, 16, dtype=torch.float64) if use_cond else None
    vec = torch.randn(2, 8, dtype=torch.float64)
    z, logdet = f(mel, cond=cond, vec=vec)
    rec = f.inverse(z, cond=cond, vec=vec)
    err = (rec - mel).abs().max().item()
    assert err < 1e-3, f"inverse(forward(mel)) max abs err {err:.2e}"
    assert z.shape == mel.shape
    assert logdet.shape == (2, 37)  # per-frame log-det


def test_invertible_1x1_conv_roundtrip_and_det():
    torch.manual_seed(7)
    conv = Invertible1x1Conv(80).double()
    x = torch.randn(4, 80, 11, dtype=torch.float64)
    with torch.no_grad():  # move off the identity-ish init permutation
        conv.p_low.normal_(0, 0.3)
        conv.p_up.normal_(0, 0.3)
        conv.p_log_diag.normal_(0, 0.5)
    y = conv(x)
    rec = conv(y, reverse=True)
    assert (rec - x).abs().max().item() < 1e-9
    w, _ = conv.weight()
    det_w = torch.linalg.det(w)
    expected = torch.exp(conv.log_abs_det())
    assert torch.allclose(det_w.abs(), expected, rtol=1e-10)


# --------------------------------------------------------------------- #
# log-det consistency vs autograd jacobian (analytic cross-check)
# --------------------------------------------------------------------- #
def test_logdet_matches_autograd_jacobian():
    torch.manual_seed(3)
    f = FlowDecoder(
        n_mels=4, hidden_channels=8, n_blocks=2, cond_dim=0, vec_dim=5,
    ).double()
    # perturb parameters so the flow is not at its identity-ish init
    with torch.no_grad():
        for p in f.parameters():
            p.normal_(0.0, 0.25)
        for m in f.modules():  # keep bounded log-scales well inside bounds
            if hasattr(m, "out_proj"):
                m.out_proj.weight.normal_(0.0, 0.05)
                m.out_proj.bias.normal_(0.0, 0.05)
    vec = torch.randn(1, 5, dtype=torch.float64)
    x = torch.randn(1, 4, 6, dtype=torch.float64)

    def fn(flat):
        z, _ = f(flat.view(1, 4, 6), cond=None, vec=vec)
        return z.reshape(-1)

    z, logdet = f(x, cond=None, vec=vec)
    J = torch.autograd.functional.jacobian(fn, x.reshape(-1))
    sign, logabsdet = torch.linalg.slogdet(J)
    assert sign.item() > 0, "flow orientation must be preserved"
    diff = abs(logdet.sum().item() - logabsdet.item())
    assert diff < 1e-3, (
        f"flow log-det {logdet.sum().item():.8f} vs jacobian "
        f"slogdet {logabsdet.item():.8f} (diff {diff:.2e})"
    )


# --------------------------------------------------------------------- #
# MAS: known solutions + exactness vs textbook DP + path properties
# --------------------------------------------------------------------- #
def test_mas_known_uniform_solution():
    # reward cell (i, j) iff phoneme i owns the middle of frame j's span:
    # optimal alignment is exactly two frames per phoneme
    C = torch.full((4, 8), -1.0)
    for j in range(8):
        C[min(3, j // 2), j] = 1.0
    durs, score = maximum_path(C)
    assert durs[0].tolist() == [2, 2, 2, 2]
    assert score[0].item() == pytest.approx(8.0)


def test_mas_known_skewed_solution_and_monotonicity():
    C = torch.full((2, 8), -1.0)
    for j in range(8):
        C[0 if j < 6 else 1, j] = 1.0
    durs, _ = maximum_path(C)
    assert durs[0].tolist() == [6, 2]  # plausible: 6 frames then 2
    # properties: every duration >= 1, sum covers all frames, indices rise
    assert (durs[0] >= 1).all()
    assert durs[0].sum().item() == 8


def test_mas_fast_matches_reference_exactly_on_random_matrices():
    torch.manual_seed(0)
    for trial in range(60):
        tx = int(torch.randint(1, 14, (1,)).item())
        ty = tx + int(torch.randint(0, 15, (1,)).item())
        C = torch.randn(3, tx, ty)
        xa = torch.zeros(3, dtype=torch.long)
        ya = torch.zeros(3, dtype=torch.long)
        for b in range(3):
            n = int(torch.randint(1, tx + 1, (1,)).item())
            xa[b] = n
            ya[b] = n + int(torch.randint(0, ty - n + 1, (1,)).item())
        d_fast, s_fast = maximum_path(C, xa, ya)
        d_ref, s_ref = maximum_path_reference(C, xa, ya)
        assert torch.equal(d_fast, d_ref), f"trial {trial}: duration mismatch"
        assert torch.allclose(s_fast, s_ref, atol=1e-6), f"trial {trial}"
        for b in range(3):
            d_b = d_fast[b, : xa[b]]
            assert d_b.sum().item() == ya[b].item(), "must cover all frames"
            assert (d_b >= 1).all().item(), "every phoneme needs >= 1 frame"


def test_mas_rejects_fewer_frames_than_phonemes():
    C = torch.randn(1, 5, 4)
    with pytest.raises(ValueError):
        maximum_path(C)


# --------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------- #
def test_training_loss_determinism_same_seed():
    ids = torch.randint(1, 200, (1, 20), generator=torch.Generator().manual_seed(9))
    mel = torch.randn(1, 26, 80, generator=torch.Generator().manual_seed(10))

    def build_run(seed: int) -> dict:
        torch.manual_seed(seed)
        cfg = CaedralNARConfig(
            name="probe", d_model=128, n_heads=4, n_layers_enc=1,
            variance_hidden=64, head_dim=128, head_blocks=1,
            use_flow=True, flow_hidden=64, flow_blocks=2, seed=seed,
        )
        model = CaedralNAR(cfg)
        return model.losses_for_training(mel.clone(), ids.clone())

    r1 = build_run(777)
    r2 = build_run(777)
    r3 = build_run(778)
    for key in ("flow_loss", "duration_loss", "loss"):
        assert torch.equal(r1[key], r2[key]), f"{key} must reproduce bitwise"
        assert not torch.equal(r1[key], r3[key]), f"{key} must depend on seed"


# --------------------------------------------------------------------- #
# integration: one training step chain reduces flow_loss
# --------------------------------------------------------------------- #
def test_integration_training_reduces_flow_loss_and_flows_grads():
    torch.manual_seed(42)
    cfg = CaedralNARConfig(
        name="probe", d_model=128, n_heads=4, n_layers_enc=2,
        variance_hidden=64, head_dim=128, head_blocks=1,
        use_flow=True, flow_hidden=64, flow_blocks=3, seed=42,
    )
    model = CaedralNAR(cfg)
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1, 200, (1, 12), generator=g)
    mel = torch.randn(1, 36, 80, generator=g)  # fixed target "utterance"
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first = last = None
    for step in range(50):
        opt.zero_grad(set_to_none=True)
        out = model.losses_for_training(mel, ids, speaker_id=0)
        for k in ("flow_loss", "duration_loss", "pitch_loss", "energy_loss"):
            assert torch.isfinite(out[k]), f"{k} not finite at step {step}"
        out["loss"].backward()
        grads = {
            n for n, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        if step == 0:
            first = out["flow_loss"].item()
            assert any(n.startswith("flow.") for n in grads), \
                "no gradient reached the flow decoder"
            assert any(n.startswith("flow_prior_proj") for n in grads), \
                "no gradient reached the prior heads"
            assert any(n.startswith("duration_predictor") for n in grads), \
                "no gradient reached the duration predictor"
            assert any(n.startswith("encoder") for n in grads), \
                "no gradient reached the encoder"
        last = out["flow_loss"].item()
        opt.step()

    assert last < 0.9 * first, (
        f"flow_loss did not reduce meaningfully: {first:.3f} -> {last:.3f}"
    )
    # MAS durations must remain a valid partition of the mel frames
    d = out["mas_durations"][0]
    assert (d >= 1).all() and int(d.sum()) == mel.shape[1]


def test_flow_infer_shapes_and_noise_scale_semantics():
    torch.manual_seed(5)
    cfg = CaedralNARConfig(
        name="probe", d_model=128, n_heads=4, n_layers_enc=1,
        variance_hidden=64, head_dim=128, head_blocks=1,
        use_flow=True, flow_hidden=64, flow_blocks=2, seed=123,
    )
    model = CaedralNAR(cfg).eval()
    ids = torch.randint(1, 200, (1, 10), generator=torch.Generator().manual_seed(6))

    with torch.no_grad():
        det = model.infer(ids, noise_scale=0.0)
        hot1 = model.infer(ids, noise_scale=0.667)
        torch.manual_seed(99)
        hot2 = model.infer(ids, noise_scale=0.667)
        default_ns = model.infer(ids)["noise_scale"]

    t_mel = int(det["durations"].sum())
    assert det["mel"].shape == (1, t_mel, 80)
    assert det["waveform"].shape == (1, t_mel * model.config.hop_length)
    assert torch.isfinite(det["waveform"]).all() and torch.isfinite(det["mel"]).all()
    # noise_scale=0 collapses to the deterministic prior mean
    assert torch.equal(det["mel"], model.infer(ids, noise_scale=0.0)["mel"])
    # nonzero temperature + different RNG stream changes the sample
    assert not torch.equal(hot1["mel"], hot2["mel"])
    assert not torch.allclose(hot1["mel"], det["mel"], atol=1e-4)
    assert math.isclose(default_ns, model.config.flow_noise_scale)
    # forward() delegates to infer() for flow models
    with torch.no_grad():
        fwd = model(ids, noise_scale=0.0)
    assert torch.equal(fwd["mel"], det["mel"])


def test_regression_path_unaffected_by_flow_code():
    """E0002 contract spot-checks still hold with use_flow=False."""
    torch.manual_seed(3)
    cfg = CaedralNARConfig(
        name="probe", d_model=128, n_heads=4, n_layers_enc=1, n_layers_dec=1,
        variance_hidden=64, head_dim=128, head_blocks=1,
    )  # use_flow defaults False
    model = CaedralNAR(cfg).eval()
    ids = torch.randint(1, 200, (1, 15), generator=torch.Generator().manual_seed(4))
    out = model(ids)
    t_mel = out["n_mel_frames"]
    assert out["mel"].shape == (1, t_mel, 80)
    assert out["waveform"].shape == (1, t_mel * model.config.hop_length)
    with pytest.raises(RuntimeError):
        model.losses_for_training(out["mel"], ids)
    with pytest.raises(RuntimeError):
        model.infer(ids)
