"""Variance adapter + conditioning tables for CAEDRAL-NAR (Phase F).

Per docs/research/MULTI-SPEAKER-PROSODY-STYLE.md section 8 ("RECOMMENDED G1
SPEAKER/STYLE/PROSODY ARCHITECTURE"):

- speaker identity: learnable embedding table, d=128, deterministic, zero
  runtime cost (section 8.1);
- style: second embedding table with rows = 10 G1 styles + 1 neutral row;
- conditioning vector ``s = [speaker | style]`` feeds FiLM injection in the
  decoder blocks and the variance predictors (section 8.1 "Injection");
- FastSpeech2-class duration / F0 / energy predictors conditioned on s give
  the explicit prosody knobs "for free" (section 8.1, FastSpeech2 [17]);
- every control knob is a transform on PREDICTED TRACKS — no knob touches
  model weights (section 8.2): speed divides durations, pitch_shift scales
  F0 in log domain, energy_gain adds dB.

All predictors are conv-bank stacks (kernel 3) + LayerNorm + FiLM + linear
output, CPU-deterministic at dropout=0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FiLMGenerator

FiLM = FiLMGenerator  # re-export: single FiLM surface across the package

NEUTRAL_STYLE_ID = None  # sentinel: resolved to the table's last row


class ConditioningProvider(nn.Module):
    """Speaker (d=128) and style (+neutral) tables -> s = [speaker | style]."""

    def __init__(
        self,
        n_speakers: int,
        n_styles: int,
        speaker_dim: int = 128,
        style_dim: int = 64,
    ):
        super().__init__()
        self.n_styles = n_styles
        self.neutral_style_id = n_styles  # neutral is the LAST table row
        self.speaker_table = nn.Embedding(n_speakers, speaker_dim)
        self.style_table = nn.Embedding(n_styles + 1, style_dim)
        self.out_dim = speaker_dim + style_dim

    def forward(
        self,
        speaker_id: int | torch.Tensor,
        style_id: int | torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns s of shape (B, out_dim). Batch inferred from ids if given."""
        if not torch.is_tensor(speaker_id):
            speaker_id = torch.tensor([speaker_id], dtype=torch.long)
        if style_id is None:
            style_id = self.neutral_style_id
        if not torch.is_tensor(style_id):
            # broadcast one style id across the whole speaker batch
            style_id = torch.full(
                (speaker_id.shape[0],), int(style_id), dtype=torch.long
            )
        elif style_id.shape[0] != speaker_id.shape[0]:
            if style_id.shape[0] == 1:
                style_id = style_id.expand(speaker_id.shape[0])
            else:
                raise ValueError(
                    f"style_id batch {style_id.shape[0]} != "
                    f"speaker_id batch {speaker_id.shape[0]}"
                )
        spk = self.speaker_table(speaker_id)
        sty = self.style_table(style_id)
        return torch.cat([spk, sty], dim=-1)


class VariancePredictor(nn.Module):
    """Conv-bank predictor conditioned on s via per-layer FiLM.

    FastSpeech2-style: Conv1d(k=3) -> LN -> FiLM -> activation, xN ->
    Linear(hidden, 1). Shared by duration / pitch / energy heads; the
    interpretation of the scalar output lives in the caller.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int,
        n_layers: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        pads = kernel_size // 2
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [d_in] + [hidden] * n_layers
        for d_prev, d_next in zip(dims[:-1], dims[1:]):
            self.convs.append(
                nn.Conv1d(d_prev, d_next, kernel_size, padding=pads)
            )
            self.norms.append(nn.LayerNorm(d_next))
        self.films = nn.ModuleList(
            FiLM(cond_dim, hidden, sites=1) for _ in range(n_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in), cond: (B, cond_dim) -> (B, T)."""
        h = x
        for i, (conv, ln, film) in enumerate(
            zip(self.convs, self.norms, self.films)
        ):
            h = conv(h.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
            h = ln(h)
            gamma, beta = film(cond, site=0)
            h = h * (1.0 + gamma) + beta
            h = F.gelu(h)
            h = self.dropout(h)
        return self.out_proj(h).squeeze(-1)


class DurationPredictor(VariancePredictor):
    """Outputs log-duration per phoneme; exp() + speed division downstream."""


class PitchPredictor(VariancePredictor):
    """Outputs log-F0 (Hz) per expanded frame; shift applies in log domain."""


class EnergyPredictor(VariancePredictor):
    """Outputs energy (dB) per expanded frame; gain adds dB directly."""


def expand_tokens(
    x: torch.Tensor, durations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Length regulator: repeat each phoneme vector by its predicted duration.

    x: (B, T, D); durations: (B, T) integer.
    Returns (expanded (B, T_mel, D) padded to the batch max, mask (B, T_mel)).
    """
    batch = x.shape[0]
    outs = []
    masks = []
    for b in range(batch):
        reps = durations[b].tolist()
        outs.append(torch.repeat_interleave(x[b], torch.as_tensor(reps), dim=0))
        masks.append(
            torch.ones(int(durations[b].sum()), dtype=torch.bool, device=x.device)
        )
    t_mel = max(o.shape[0] for o in outs)
    d = x.shape[-1]
    padded = torch.zeros(batch, t_mel, d, dtype=x.dtype, device=x.device)
    mask = torch.zeros(batch, t_mel, dtype=torch.bool, device=x.device)
    for b, (o, m) in enumerate(zip(outs, masks)):
        padded[b, : o.shape[0]] = o
        mask[b, : m.shape[0]] = m
    return padded, mask
