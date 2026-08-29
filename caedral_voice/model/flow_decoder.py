"""Conditional normalizing flow over mel frames (Phase F+, CV-G1-E0006).

Glow-TTS-lineage decoder upgrade decided in CV-G1-E0005: replaces the
deterministic ``decoder -> mel_proj`` regression stack of CV-G1-E0002 with an
invertible affine-coupling stack so that MAS training (mas.py) has a proper
likelihood to maximize and inference samples z ~ N(0, noise_scale) through
the inverse.

Structure per flow block (data->latent order):
    invertible 1x1 Conv (LU-decomposed weight, fixed random permutation at
    init) -> affine coupling on channel halves:
        (logs, b) = WaveNet-ish gated-residual conditioner(other half,
                    per-frame cond, FiLM(vec));
        z2 = (x2 - b) * exp(logs), |log-scale| bounded.

Conditioning surfaces:
    cond : (B, T, C_cond) per-frame vector broadcast over the conditioner
           input (e.g. phoneme-expanded encoder features);
    vec  : (B, C_vec) global vector applied by per-layer FiLM inside every
           conditioner (speaker | style).

The pilot wiring (caedral_nar.CaedralNAR.use_flow) feeds only ``vec``
(Glow-TTS ``g``); text enters through the flow's base distribution via prior
heads on the encoder, keeping MAS single-pass (see caedral_nar.py docstring).
The per-frame ``cond`` surface exists for future experiments and is fully
exercised by tests/test_flow_mas.py.

All ops are CPU-deterministic at dropout=0 in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LOG_SCALE_BOUND = 2.0  # tanh-bounded |log-scale| keeps the affine step invertible


class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 conv with LU-decomposed weight (Glow-TTS section 3.2).

    W = P @ (I + strict_lower(L_raw)) @ (strict_upper(U_raw)
          + diag(sign * exp(log_diag))),  P a FIXED random permutation.
    det(L)=1 and |det(diag(U))| = prod(exp(log_diag)), so log|det W| =
    sum(log_diag) exactly and W is invertible for every parameter value.
    Init: raw factors zero => W = P (pure channel shuffle), while the
    following coupling starts near-identity (zero-init output projection),
    so the whole stack starts close to a bijection between mel and N(0, I).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.p_low = nn.Parameter(torch.zeros(channels, channels))
        self.p_up = nn.Parameter(torch.zeros(channels, channels))
        self.p_log_diag = nn.Parameter(torch.zeros(channels))
        perm = torch.randperm(channels)
        sign = torch.tensor(
            [1.0 if i % 2 == 0 else -1.0 for i in range(channels)]
        )
        self.register_buffer("perm", perm)
        self.register_buffer("sign", sign)
        eye = torch.eye(channels)
        lower_mask = torch.tril(torch.ones(channels, channels), -1)
        upper_mask = torch.triu(torch.ones(channels, channels), 1)
        self.register_buffer("eye", eye)
        self.register_buffer("lower_mask", lower_mask)
        self.register_buffer("upper_mask", upper_mask)

    def weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (W, W_inv), both (C, C)."""
        L = self.eye + self.p_low * self.lower_mask
        U = (
            self.p_up * self.upper_mask
            + torch.diag(self.sign * torch.exp(self.p_log_diag))
        )
        w = self.eye[self.perm] @ L @ U  # P @ L @ U
        # W_inv = U^-1 @ L^-1 @ P^-1; triangular solves against I. C <= 128
        # here, so this is negligible next to the conditioner convs.
        linv = torch.linalg.solve_triangular(
            L, self.eye, upper=False, unitriangular=True
        )
        uinv = torch.linalg.solve_triangular(U, self.eye, upper=True)
        pinv = self.eye[:, self.perm]  # P^-1 = P^T
        w_inv = uinv @ linv @ pinv
        return w, w_inv

    def log_abs_det(self) -> torch.Tensor:
        return self.p_log_diag.sum()

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        # x: (B, C, T). Row-vector form: y_r = x_r @ W^T (fwd) / @ W^-T (inv).
        w, w_inv = self.weight()
        m = w_inv.t() if reverse else w.t()
        return (x.transpose(1, 2) @ m).transpose(1, 2)


class GatedResidualConditioner(nn.Module):
    """WaveNet-ish conditioner: sigmoid-gated residual layers + FiLM(vec).

    Input is the untouched half of the mel channels (optionally concatenated
    with a per-frame cond vector); output is per-frame (logs, bias) for the
    transformed half. Every layer is tanh(a)*sigmoid(b) gated (WaveNet
    gating) and receives the global conditioning vector by per-layer FiLM
    (research doc section 8.1 injection surface), replacing the additive g
    projection of the reference implementations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        vec_dim: int = 0,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        pad = kernel_size // 2
        self.in_layers = nn.ModuleList()
        self.films = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.residuals = nn.ModuleList()
        ch = in_channels
        for _ in range(n_layers):
            self.in_layers.append(
                nn.Conv1d(ch, 2 * hidden_channels, kernel_size, padding=pad)
            )
            if vec_dim > 0:
                self.films.append(nn.Linear(vec_dim, 2 * hidden_channels))
            self.skips.append(nn.Conv1d(hidden_channels, hidden_channels, 1))
            if ch != hidden_channels:
                self.residuals.append(nn.Conv1d(ch, hidden_channels, 1))
            else:
                self.residuals.append(nn.Identity())
            ch = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_channels, 2 * out_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_half: torch.Tensor,               # (B, in_ch, T)
        cond: torch.Tensor | None = None,   # (B, T, C_cond)
        vec: torch.Tensor | None = None,    # (B, C_vec)
    ):
        if cond is not None:
            x_half = torch.cat([x_half, cond.transpose(1, 2)], dim=1)
        h = x_half
        for i, (conv, skip, res) in enumerate(
            zip(self.in_layers, self.skips, self.residuals)
        ):
            r = res(h)
            a, b_gate = conv(h).chunk(2, dim=1)
            h = torch.tanh(a) * torch.sigmoid(b_gate)  # sigmoid-gated
            if len(self.films) > 0:
                gamma, beta = self.films[i](vec).unsqueeze(-1).chunk(2, dim=1)
                h = h * (1.0 + gamma) + beta
            h = self.dropout(skip(h))
            h = h + r
        o = self.out_proj(h.transpose(1, 2))  # (B, T, 2*out)
        logs, bias = o.chunk(2, dim=-1)
        logs = LOG_SCALE_BOUND * torch.tanh(logs / LOG_SCALE_BOUND)
        return logs.transpose(1, 2), bias.transpose(1, 2)


class AffineCoupling(nn.Module):
    """One affine-coupling layer over channel halves (Glow-TTS eqs. 7-8).

    data->latent (forward): z2 = (x2 - bias) * exp(logs);
    latent->data (reverse): x2 = z2 * exp(-logs) + bias.
    Per-frame log-det contribution: sum_c logs[c].
    """

    def __init__(
        self,
        half: int,
        hidden_channels: int,
        cond_dim: int = 0,
        vec_dim: int = 0,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.half = half
        self.cond_dim = cond_dim
        self.conditioner = GatedResidualConditioner(
            in_channels=half + cond_dim,
            hidden_channels=hidden_channels,
            out_channels=half,
            vec_dim=vec_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,                  # (B, n_mels, T)
        cond: torch.Tensor | None = None,
        vec: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        x1, x2 = torch.split(x, self.half, dim=1)
        if cond is None and self.cond_dim > 0:
            # conditioning surface kept live with zeros (VITS g convention)
            cond = x1.new_zeros(x1.shape[0], x1.shape[2], self.cond_dim)
        logs, bias = self.conditioner(x1, cond=cond, vec=vec)
        if reverse:
            x2 = x2 * torch.exp(-logs) + bias
        else:
            x2 = (x2 - bias) * torch.exp(logs)
        # per-frame log-det of this coupling: sum over transformed channels
        return torch.cat([x1, x2], dim=1), logs.sum(dim=1)


class FlowDecoder(nn.Module):
    """Stack of [invertible 1x1 conv + affine coupling] blocks over mel.

    forward(mel, cond, vec)              -> (z, logdet_per_frame (B, T))
    inverse(z, cond, vec, noise_scale)   -> mel

    ``noise_scale`` multiplies z before the inverse transform, i.e. it sets
    the std of the base distribution the caller samples from; callers that
    pre-scale their sample pass noise_scale=1.0.
    """

    def __init__(
        self,
        n_mels: int,
        hidden_channels: int = 256,
        n_blocks: int = 6,
        cond_dim: int = 0,
        vec_dim: int = 0,
        n_layers_per_block: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_mels % 2 != 0:
            raise ValueError("n_mels must be even for channel-halved coupling")
        self.n_mels = n_mels
        self.half = n_mels // 2
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "invconv": Invertible1x1Conv(n_mels),
                    "coupling": AffineCoupling(
                        half=self.half,
                        hidden_channels=hidden_channels,
                        cond_dim=cond_dim,
                        vec_dim=vec_dim,
                        n_layers=n_layers_per_block,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    ),
                }
            )
            for _ in range(n_blocks)
        )

    def forward(
        self,
        mel: torch.Tensor,                  # (B, n_mels, T)
        cond: torch.Tensor | None = None,   # (B, T, C_cond)
        vec: torch.Tensor | None = None,    # (B, C_vec)
    ):
        """data -> latent. Returns (z, logdet) with logdet shaped (B, T)."""
        logdet = None
        for block in self.blocks:
            mel = block["invconv"](mel)
            ld_conv = block["invconv"].log_abs_det()
            mel, ld = block["coupling"](mel, cond=cond, vec=vec, reverse=False)
            total = ld + ld_conv
            logdet = total if logdet is None else logdet + total
        return mel, logdet

    def inverse(
        self,
        z: torch.Tensor,                    # (B, n_mels, T)
        cond: torch.Tensor | None = None,
        vec: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """latent -> data. z is scaled by ``noise_scale`` first."""
        z = z * float(noise_scale)
        for block in reversed(self.blocks):
            z, _ = block["coupling"](z, cond=cond, vec=vec, reverse=True)
            z = block["invconv"](z, reverse=True)
        return z
