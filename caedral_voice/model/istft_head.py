"""Mel -> waveform iSTFT head for CAEDRAL-NAR (Phase F).

Vocos/iSTFTNet2-class decoder skeleton (ARCH-NOTES-G1 "Decoder direction":
small MIT-licensed iSTFT-head vocoder retrained in-house). The head maps a
mel spectrogram to linear-domain magnitude + phase and synthesizes the
waveform with a single ``torch.istft`` — no transposed-conv upsampling stack,
so inference cost stays bandwidth-bound rather than conv-bound.

TRAINABLE-STUB STATUS: this is an untrained prototype; correctness contract =
exact output shapes + deterministic forward. Audio quality is out of scope
until Phase G trains it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock1d(nn.Module):
    """Channels-first ConvNeXt block (Vocos backbone): DWConv(k=7) -> LN ->
    pointwise expand -> GELU -> pointwise project, with residual."""

    def __init__(self, dim: int, expand: int = 4, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        hidden = dim * expand
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        res = x
        x = self.dwconv(x).transpose(1, 2)  # (B, T, C) for LN/Linear
        x = self.norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        return res + x.transpose(1, 2)


class IstftHead(nn.Module):
    """mel frames -> waveform via magnitude/phase prediction + torch.istft.

    Input (B, n_mels, T) -> waveform (B, T * hop_length).
    """

    def __init__(
        self,
        n_mels: int = 80,
        dim: int = 256,
        n_blocks: int = 2,
        expand: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.in_proj = nn.Conv1d(n_mels, dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock1d(dim, expand=expand) for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(dim)
        n_freqs = n_fft // 2 + 1
        self.mag_head = nn.Linear(dim, n_freqs)
        # phase from two components: atan2(im, re), Vocos-style
        self.phase_head = nn.Linear(dim, 2 * n_freqs)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) -> waveform (B, T * hop_length)."""
        t_frames = mel.shape[-1]
        x = self.in_proj(mel)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x.transpose(1, 2))  # (B, T, C)

        magnitude = torch.exp(self.mag_head(x)).transpose(1, 2)  # (B, F, T)
        re_im = self.phase_head(x)
        phase = torch.atan2(re_im[..., : re_im.shape[-1] // 2],
                            re_im[..., re_im.shape[-1] // 2 :])
        phase = phase.transpose(1, 2)

        spec = torch.complex(
            magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        )  # (B, F, T) complex64

        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(spec.real.dtype),
            center=True,
            length=t_frames * self.hop_length,
        )
        return wav
