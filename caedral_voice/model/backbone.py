"""Pre-LN transformer backbone for CAEDRAL-NAR (Phase F, CV-G1-E0002).

Design per docs/research/ACOUSTIC-ARCHITECTURES.md section 12 (Candidate 1,
"CAEDRAL-NAR", Kokoro/StyleTTS2 lineage) and docs/architecture/ARCH-NOTES-G1.md:

- pre-LN residual blocks: stable at small scale, standard for this class of
  TTS backbones;
- optional RoPE (rotary position embeddings, Su et al. 2021): the NAR decoder
  runs over expanded frame sequences; RoPE is relative and parameter-free;
- optional GQA (Ainslie et al. 2023): ``n_kv_heads < n_heads`` shrinks the KV
  projections. Weight-stream bandwidth (~40-42 GB/s GEMV) is the binding
  constraint on this silicon (ARCH-NOTES-G1), so every saved weight matters.

All ops are CPU-deterministic in eval mode with dropout=0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    """Hyperparameters for one transformer stack."""

    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 0  # 0 -> MHA (n_kv_heads = n_heads); < n_heads -> GQA
    ff_mult: float = 4.0
    dropout: float = 0.0
    rope: bool = True
    causal: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.n_kv_heads <= 0:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.rope and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def _rope_cos_sin(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
):
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half convention. x: (B, H, T, Dh); cos/sin: (T, Dh/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional (or causal) self-attention with optional RoPE and GQA."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        d, dh, kv = cfg.d_model, cfg.head_dim, cfg.kv_dim
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, kv, bias=True)
        self.v_proj = nn.Linear(d, kv, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        if cfg.rope:
            cos, sin = _rope_cos_sin(
                T, cfg.head_dim, cfg.rope_theta, x.device, x.dtype
            )
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if cfg.n_kv_heads != cfg.n_heads:  # GQA: broadcast KV groups to heads
            group = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0 if not self.training else cfg.dropout,
            is_causal=cfg.causal,
        )
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        hidden = int(cfg.d_model * cfg.ff_mult)
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)); x + FFN(LN(x)).

    If ``film`` (and a conditioning vector s) are supplied at forward time,
    FiLM modulation is applied to each sublayer input right after its LN —
    the AdaLayerNorm-style conditioning surface recommended by
    MULTI-SPEAKER-PROSODY-STYLE.md section 8.1.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        film=None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        if film is not None:
            gamma, beta = film(cond, site=0)
            h = h * (1.0 + gamma) + beta
        x = x + self.attn(h)

        h = self.ln2(x)
        if film is not None:
            gamma, beta = film(cond, site=1)
            h = h * (1.0 + gamma) + beta
        x = x + self.ffn(h)
        return x


class Transformer(nn.Module):
    """Stack of pre-LN blocks + final LN. Optional per-block FiLM conditioning.

    ``cond_dim`` > 0 enables FiLM conditioning: one generator per block with
    two application sites (pre-attention, pre-FFN).
    """

    def __init__(
        self,
        cfg: BackboneConfig,
        n_layers: int,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_layers))
        self.films = (
            nn.ModuleList(FiLMGenerator(cond_dim, cfg.d_model) for _ in range(n_layers))
            if cond_dim
            else None
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            if self.films is not None:
                x = block(x, self.films[i], cond)
            else:
                x = block(x)
        return self.ln_f(x)


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation: cond vector -> per-site (gamma, beta).

    Kept here (not variance.py) so the transformer owns its conditioning
    surface; variance.py re-exports it for the predictors.
    """

    def __init__(self, cond_dim: int, feature_dim: int, sites: int = 2):
        super().__init__()
        self.sites = sites
        self.feature_dim = feature_dim
        self.proj = nn.Linear(cond_dim, sites * 2 * feature_dim)

    def forward(self, cond: torch.Tensor, site: int):
        out = self.proj(cond)  # (B, sites*2*F)
        chunk = 2 * self.feature_dim
        out = out[:, site * chunk : (site + 1) * chunk]
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)  # broadcast over T


# variance.py imports this; keep the public name identical in both modules.
FiLM = FiLMGenerator
