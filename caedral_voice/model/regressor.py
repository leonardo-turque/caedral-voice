"""Supervised-duration mel regressor (experiment CV-G1-E0013, B-arm).

The DIAG-02 verdict (experiments/CV-G1-E0009-diagnosis/overfit/conclusion.md)
showed the MAS/flow objective smears: i.i.d. per-frame latent sampling loses
inter-frame detail and degenerate alignments dump silence onto single phonemes.
The B-arm pivot (docs/EXECUTION-QUEUE.md V1 directive) replaces both with
supervision:

    phonemes -> text encoder -> duration predictor  (SUPERVISED on CTC-aligner
                durations, ALIGN-01; log-domain MSE -- no MAS, no prior NLL)
             -> length regulator (ground-truth durations)
             -> conv-transformer mel decoder -> log-mel
                (masked L1 + optional spectral-convergence loss)

Same encoder/variance skeleton as CaedralNAR (backbone.py / variance.py), but
the flow decoder + prior heads are replaced by a plain regression decoder that
predicts the log-mel directly. The iSTFT head is dropped: the output is a mel,
vocoded downstream by the audited Vocos chain (E0009 part 1 proved chain
fidelity).

v0 prosody-supervision decision (documented per mission):
* **duration predictor: TRAINED** -- supervised on ALIGN-01 per-phoneme frame
  counts (``durations`` argument of :meth:`losses_for_training`); during the
  pre-ALIGN smoke window the dataset synthesizes uniform splits instead.
* **pitch predictor: FROZEN OFF.** Ground-truth F0 needs an offline tracker;
  torchaudio's ``detect_pitch_frequency`` is unvalidated on this corpus at the
  CPU budget and a bad track is worse than no track. The module exists but its
  parameters get ``requires_grad=False`` and its projection term is excluded
  from the decoder input. Enabling it later requires a validated GT-F0 pass.
* **energy predictor: IMPLEMENTED, DEFAULT OFF** (--train-energy). GT tracks
  are derived deterministically from the target mel itself (frame RMS over
  magnitude bins -> dB), so enabling it is one flag; kept off in v0 to hold
  the ablation surface minimal.

Speaker table: ``n_speakers`` floor of **40** per the E0013 sizing directive
(real LibriTTS-R voices + teacher slots t00..tNN); callers auto-expand to the
corpus count when it exceeds the floor (clean100 alone has 247 speakers).

Parameter target ~15 M (between tiny 8.35 M and lite 31.6 M, E0002).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneConfig, Transformer
from .variance import (
    ConditioningProvider,
    DurationPredictor,
    EnergyPredictor,
    PitchPredictor,
    expand_tokens,
)

ENERGY_DB_FLOOR = -100.0


@dataclass
class CaedralRegressorConfig:
    """One CaedralRegressor instantiation. Serialized 1:1 to JSON."""

    name: str = "regressor-medium"
    # signal front
    vocab_size: int = 200
    n_mels: int = 80
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    # backbone (sized for ~15M: d384 enc2 + dec5)
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 6  # 0 -> MHA; < n_heads -> GQA
    ff_mult: float = 4.0
    dropout: float = 0.0
    rope: bool = True
    causal: bool = False
    n_layers_enc: int = 2
    n_layers_dec: int = 5
    # conv front of the decoder ("conv-transformer" decoder)
    decoder_conv_layers: int = 1
    decoder_conv_kernel: int = 5
    # variance adapter
    variance_hidden: int = 160
    variance_layers: int = 2
    variance_kernel: int = 3
    # conditioning (speaker table floor 40 per E0013 directive)
    n_speakers: int = 40
    speaker_dim: int = 128
    n_styles: int = 10  # +1 neutral row appended by ConditioningProvider
    style_dim: int = 64
    # duration post-processing at inference only (training uses GT durations)
    max_duration: int = 50
    # losses
    spec_loss_weight: float = 1.0   # 0 disables spectral-convergence aux loss
    dur_loss_weight: float = 1.0    # weight of the supervised duration MSE
    # v0 prosody supervision switches (see module docstring)
    train_pitch: bool = False
    train_energy: bool = False
    seed: int = 1234

    @classmethod
    def from_json(cls, path: str | Path) -> "CaedralRegressorConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )


def fit_durations_to_frames(
    durations: torch.Tensor,
    target_frames: torch.Tensor,
    phoneme_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Adjust integer durations so each row sums exactly to its mel length.

    The aligner emits durations on the same 24 kHz / hop-256 grid as our mels,
    so ``sum(durations) == T_mel`` should already hold; this guards the residual
    rounding cases deterministically instead of crashing a long run. Following
    the aligner convention (ctc_aligner.expand_ds_durations_to_mel), the LAST
    phoneme absorbs any positive remainder; deficits are trimmed from trailing
    phonemes down to a floor of 1 frame each. Only the first
    ``phoneme_lengths[i]`` (or all) entries are touched -- padded slots stay 0
    so ``expand_tokens`` emits no frames for them.

    durations: (B, T_x) int; target_frames: (B,) valid mel frames per row.
    Returns (fitted durations, total frames moved).
    """
    fitted = durations.clone()
    moved = 0
    b, tx = durations.shape
    for i in range(b):
        n = int(phoneme_lengths[i]) if phoneme_lengths is not None else tx
        t = int(target_frames[i])
        diff = t - int(fitted[i, :n].sum())
        if diff == 0:
            continue
        moved += abs(diff)
        j = n - 1
        if diff > 0:
            fitted[i, j] += diff          # last phoneme absorbs the remainder
        else:
            while diff < 0 and j >= 0:
                take = min(int(fitted[i, j]) - 1, -diff)
                if take > 0:
                    fitted[i, j] -= take
                    diff += take
                j -= 1
        fitted[i, :n] = fitted[i, :n].clamp_min(1)
    return fitted, moved


class CaedralRegressor(nn.Module):
    """Non-autoregressive phonemes+durations -> log-mel regression model."""

    def __init__(self, config: CaedralRegressorConfig | str | Path | dict):
        super().__init__()
        if isinstance(config, (str, Path)):
            config = CaedralRegressorConfig.from_json(config)
        elif isinstance(config, dict):
            config = CaedralRegressorConfig(**config)
        self.config = config
        c = config

        self.phoneme_embed = nn.Embedding(c.vocab_size, c.d_model)

        enc_cfg = BackboneConfig(
            d_model=c.d_model, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
            ff_mult=c.ff_mult, dropout=c.dropout, rope=c.rope, causal=c.causal,
        )
        self.encoder = Transformer(enc_cfg, c.n_layers_enc, cond_dim=0)

        cond_dim = c.speaker_dim + c.style_dim
        self.conditioning = ConditioningProvider(
            c.n_speakers, c.n_styles, c.speaker_dim, c.style_dim
        )

        vp = dict(
            cond_dim=cond_dim, n_layers=c.variance_layers,
            kernel_size=c.variance_kernel, dropout=c.dropout,
        )
        self.duration_predictor = DurationPredictor(c.d_model, c.variance_hidden, **vp)
        self.pitch_predictor = PitchPredictor(c.d_model, c.variance_hidden, **vp)
        self.energy_predictor = EnergyPredictor(c.d_model, c.variance_hidden, **vp)

        # decoder input projections for the predicted prosody tracks (same
        # surfaces as the NAR non-flow path); pitch/energy terms enter the
        # decoder input ONLY when their supervision is enabled
        self.pitch_proj = nn.Linear(1, c.d_model)
        self.energy_proj = nn.Linear(1, c.d_model)

        dec_cfg = BackboneConfig(
            d_model=c.d_model, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
            ff_mult=c.ff_mult, dropout=c.dropout, rope=c.rope, causal=c.causal,
        )
        self.conv_front = nn.ModuleList(
            nn.Conv1d(c.d_model, c.d_model, c.decoder_conv_kernel,
                      padding=c.decoder_conv_kernel // 2)
            for _ in range(max(0, int(c.decoder_conv_layers)))
        )
        self.decoder = Transformer(dec_cfg, c.n_layers_dec, cond_dim=cond_dim)
        self.mel_proj = nn.Linear(c.d_model, c.n_mels)

        self._apply_freeze()

    # ------------------------------------------------------------------ #
    # freezing (v0: pitch off always unless configured; energy likewise)
    # ------------------------------------------------------------------ #
    def _apply_freeze(self) -> None:
        c = self.config
        for pred in (
            () if c.train_pitch else (self.pitch_predictor,)
        ) + (() if c.train_energy else (self.energy_predictor,)):
            for p in pred.parameters():
                p.requires_grad_(False)

    @property
    def use_pitch(self) -> bool:
        return bool(self.config.train_pitch)

    @property
    def use_energy(self) -> bool:
        return bool(self.config.train_energy)

    # ------------------------------------------------------------------ #
    # parameter accounting
    # ------------------------------------------------------------------ #
    def count_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        buckets: dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            buckets[top] = buckets.get(top, 0) + p.numel()
        return dict(sorted(buckets.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ #
    # shared pieces
    # ------------------------------------------------------------------ #
    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.phoneme_embed(ids))       # (B, T_x, D)

    def _decode(
        self,
        h_exp: torch.Tensor,
        s: torch.Tensor,
        log_f0: torch.Tensor | None = None,
        e_db: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expanded frames -> mel logits (B, T_y, n_mels)."""
        c = self.config
        dec_in = h_exp
        if self.use_pitch and log_f0 is not None:
            dec_in = dec_in + self.pitch_proj(log_f0.unsqueeze(-1))
        if self.use_energy and e_db is not None:
            dec_in = dec_in + self.energy_proj((e_db / 20.0).unsqueeze(-1))
        y = dec_in.transpose(1, 2)                          # (B, D, T_y)
        for conv in self.conv_front:
            y = F.gelu(conv(y))
        y = y.transpose(1, 2)
        y = self.decoder(y, s)                              # (B, T_y, D)
        return self.mel_proj(y)

    @staticmethod
    def _mel_energy_db(logmel: torch.Tensor) -> torch.Tensor:
        """Frame energy (dB) from a log-magnitude mel: RMS across bins."""
        lin = torch.exp(logmel)
        rms = lin.pow(2).mean(dim=-1).sqrt()
        return 20.0 * torch.log10(rms.clamp_min(1e-5))

    # ------------------------------------------------------------------ #
    # training step (no optimizer logic here)
    # ------------------------------------------------------------------ #
    def losses_for_training(
        self,
        mel: torch.Tensor,                       # (B, T_y, n_mels) log-mel
        phoneme_ids: torch.Tensor,               # (B, T_x)
        durations: torch.Tensor,                 # (B, T_x) GT frames per phone
        speaker_id: int | torch.Tensor = 0,
        style_id: int | torch.Tensor | None = None,
        mel_lengths: torch.Tensor | None = None,     # (B,) valid mel frames
        phoneme_lengths: torch.Tensor | None = None, # (B,) valid phones
    ) -> dict:
        """One supervised training step's losses.

        Returns dict with masked means over valid extents:
          mel_loss   -- L1 between predicted and target log-mel
                        (per element over valid frames x bins);
          spec_loss  -- spectral convergence ||P-T||_F/||T||_F computed on
                        linear (magnitude) mels over valid frames (aux
                        anti-smear regularizer; weight via config);
          dur_loss   -- MSE between predicted and log GT durations;
          pitch_loss / energy_loss -- 0.0 placeholders in v0 (see module
                        docstring; energy becomes L1 when train_energy=True);
          loss       -- weighted total actually optimized;
          n_duration_adjusted -- frames moved by fit_durations_to_frames.
        """
        c = self.config
        if not torch.is_tensor(mel):
            mel = torch.as_tensor(mel, dtype=torch.get_default_dtype())
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        ids = phoneme_ids if torch.is_tensor(phoneme_ids) else torch.tensor(
            phoneme_ids, dtype=torch.long
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if not torch.is_tensor(durations):
            durations = torch.tensor(durations, dtype=torch.long)
        if durations.dim() == 1:
            durations = durations.unsqueeze(0)

        b, t_y, _mels = mel.shape
        t_x = ids.shape[1]
        dev, dtp = mel.device, mel.dtype

        xl = (torch.full((b,), t_x, dtype=torch.long, device=dev)
              if phoneme_lengths is None else phoneme_lengths.to(dev))
        yl = (torch.full((b,), t_y, dtype=torch.long, device=dev)
              if mel_lengths is None else mel_lengths.to(dev))
        y_ar = torch.arange(t_y, device=dev)
        x_ar = torch.arange(t_x, device=dev)
        y_mask_f = (y_ar[None, :] < yl[:, None]).to(dtp)     # (B, T_y)
        x_mask_f = (x_ar[None, :] < xl[:, None]).to(dtp)     # (B, T_x)

        h = self._encode(ids)
        s = self.conditioning(speaker_id, style_id)          # (B, C_cond)

        # ---- supervised duration loss (log domain) -----------------------
        log_d_hat = self.duration_predictor(h, s)            # (B, T_x)
        log_d_tgt = torch.log(durations.to(dev).to(dtp).clamp_min(1.0))
        dur_loss = (((log_d_hat - log_d_tgt) ** 2) * x_mask_f).sum() \
            / xl.to(dtp).sum().clamp_min(1.0)

        # ---- teacher-forced expansion on GT durations --------------------
        fitted, n_adj = fit_durations_to_frames(
            durations.to(dev), yl, phoneme_lengths=xl)
        h_exp, mask = expand_tokens(h, fitted)               # (B, T_y', D)
        t_e = h_exp.shape[1]
        if t_e != t_y:
            # expand_tokens pads to batch max sum(fitted); realign masks
            pad = t_e - t_y
            if pad > 0:
                mel = F.pad(mel, (0, 0, 0, pad))
                y_mask_f = F.pad(y_mask_f, (0, pad))
                yl = yl.clone()  # lengths unchanged; padding is masked out
            else:
                raise RuntimeError(
                    f"expanded len {t_e} < target mel frames {t_y}; "
                    "fit_durations_to_frames contract broken")

        e_db = None
        energy_loss = torch.zeros((), dtype=dtp, device=dev)
        if self.use_energy:
            with torch.no_grad():                            # targets are const
                e_tgt = self._mel_energy_db(mel)
            e_db = self.energy_predictor(h_exp, s)
            energy_loss = ((e_db - e_tgt) * y_mask_f).abs().sum() \
                / y_mask_f.sum().clamp_min(1.0)
            e_db = e_db.detach()                             # input side only

        log_f0 = self.pitch_predictor(h_exp, s) if self.use_pitch else None

        mel_hat = self._decode(h_exp, s, log_f0=log_f0, e_db=e_db)

        # ---- masked L1 on log-mel ----------------------------------------
        l1 = (mel_hat - mel).abs() * y_mask_f.unsqueeze(-1)
        mel_loss = l1.sum() / (y_mask_f.sum() * c.n_mels).clamp_min(1.0)

        # ---- optional spectral-convergence aux (anti-smear) --------------
        zero = torch.zeros((), dtype=dtp, device=dev)
        if c.spec_loss_weight > 0:
            lin_p = torch.exp(mel_hat.clamp_min(math.log(1e-5)))
            lin_t = torch.exp(mel.clamp_min(math.log(1e-5)))
            m3 = y_mask_f.unsqueeze(-1)
            num = ((lin_p - lin_t) * m3).pow(2).sum().sqrt()
            den = (lin_t * m3).pow(2).sum().sqrt().clamp_min(1e-8)
            spec_loss = num / den
        else:
            spec_loss = zero

        loss = (
            mel_loss
            + float(c.spec_loss_weight) * spec_loss
            + float(c.dur_loss_weight) * dur_loss
            + energy_loss
        )

        return {
            "loss": loss,
            "mel_loss": mel_loss,
            "spec_loss": spec_loss,
            "dur_loss": dur_loss,
            "pitch_loss": zero.clone(),
            "energy_loss": energy_loss,
            "log_duration": log_d_hat,
            "n_duration_adjusted": int(n_adj),
            "n_mel_frames": t_y,
            "n_phonemes": t_x,
        }

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_id: int = 0,
        style_id: int | None = None,  # None -> neutral style row
        speed: float = 1.0,
        durations: torch.Tensor | list[int] | None = None,  # override (B, T_x)
    ) -> dict:
        """Synthesize a mel: predict (or accept) durations, expand, decode."""
        c = self.config
        ids = phoneme_ids if torch.is_tensor(phoneme_ids) else torch.tensor(
            phoneme_ids, dtype=torch.long
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        h = self._encode(ids)
        s = self.conditioning(speaker_id, style_id)

        log_d = None
        if durations is None:
            log_d = self.duration_predictor(h, s)
            d = torch.round(torch.exp(log_d) / float(speed)).long()
            d = d.clamp(1, c.max_duration)
        else:
            if not torch.is_tensor(durations):
                durations = torch.tensor(durations, dtype=torch.long)
            d = durations if durations.dim() == 2 else durations.unsqueeze(0)
            if d.shape[1] < ids.shape[1]:
                # unpadded (valid-extent) durations: right-pad with 0s so pad
                # phonemes emit no frames (same convention as collate_regressor)
                d = F.pad(d, (0, ids.shape[1] - d.shape[1]))
            elif d.shape[1] > ids.shape[1]:
                raise ValueError(
                    f"durations width {d.shape[1]} > phoneme width "
                    f"{ids.shape[1]}")
            # clamp valid entries to [1, max_duration]; keep pad zeros at 0
            d = torch.where(d > 0, d.clamp_min(1), d).clamp_max(c.max_duration)
            d = d.to(ids.device)
        h_exp, mask = expand_tokens(h, d)

        e_db = None
        if self.use_energy:
            e_db = self.energy_predictor(h_exp, s)
        log_f0 = self.pitch_predictor(h_exp, s) if self.use_pitch else None

        mel = self._decode(h_exp, s, log_f0=log_f0, e_db=e_db)
        mel = mel * mask.unsqueeze(-1).to(mel.dtype)
        return {
            "mel": mel,
            "durations": d,
            "log_duration": log_d,   # None when durations were overridden
            "n_mel_frames": int(mask.sum(dim=1).max()),
        }
