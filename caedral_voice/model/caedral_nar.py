"""CAEDRAL-NAR prototype wiring (Phase F, experiment CV-G1-E0002).

Full NAR pipeline per docs/research/ACOUSTIC-ARCHITECTURES.md section 12
(Candidate 1, Kokoro/StyleTTS2-lineage decoder-only NAR) and
docs/research/MULTI-SPEAKER-PROSODY-STYLE.md section 8.2 (conditioning path):

    phonemes -> text encoder -> duration predictor (s=[spk|sty])
             -> length regulator (durations / speed)
             -> pitch predictor (+pitch_shift) / energy predictor (+energy_gain)
             -> FiLM-conditioned decoder -> mel -> iSTFT head -> waveform

Every knob is a transform on predicted tracks (speed divides durations,
pitch_shift rescales F0 in the log domain, energy_gain adds dB) — no knob
touches model weights (research doc section 8.2).

PROTOTYPE STATUS: untrained stub (no training in Phase F). Contracts that
ARE enforced: parameter-count targets, exact output shapes, deterministic
forward, conditioning sensitivity, CPU latency/RSS measurability.

CV-G1-E0006 addition — ``use_flow=True`` (Glow-TTS-lineage MAS training):
the decoder transformer + mel_proj are replaced by a conditional normalizing
flow (flow_decoder.FlowDecoder) and per-phoneme prior heads
(flow_prior_proj -> mel-space mean/log-scale). Conditioning follows the
Glow-TTS factorization: the flow itself is conditioned only on the global
speaker|style vector (FiLM inside every conditioner = Glow-TTS ``g``), while
text enters through the prior statistics compared against flow latents. This
keeps MAS a single forward pass plus an O(T_x*T_y) score matrix; feeding
per-frame text conditioning into the flow would force T_x flow passes per
MAS step and blow the E0002 latency budget gate. ``losses_for_training``
returns {flow_loss=-loglik under the MAS alignment, duration_loss=
log-normal prior NLL of MAS durations, pitch_loss/energy_loss placeholders};
``infer`` samples z ~ N(m_expanded, noise_scale * sigma_expanded) through the
inverse flow.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneConfig, Transformer
from .flow_decoder import FlowDecoder
from .istft_head import IstftHead
from .mas import maximum_path
from .variance import (
    ConditioningProvider,
    DurationPredictor,
    EnergyPredictor,
    PitchPredictor,
    expand_tokens,
)

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class CaedralNARConfig:
    """One CAEDRAL-NAR instantiation. Serialized 1:1 to configs/*.json."""

    name: str = "prototype"
    # signal front
    vocab_size: int = 200
    n_mels: int = 80
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    # backbone
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 0  # 0 -> MHA; < n_heads -> GQA
    ff_mult: float = 4.0
    dropout: float = 0.0
    rope: bool = True
    causal: bool = False
    n_layers_enc: int = 2
    n_layers_dec: int = 4
    # variance adapter
    variance_hidden: int = 128
    variance_layers: int = 2
    variance_kernel: int = 3
    # conditioning (research doc section 8.1)
    n_speakers: int = 32
    speaker_dim: int = 128
    n_styles: int = 10  # +1 neutral row appended by ConditioningProvider
    style_dim: int = 64
    # iSTFT head
    head_dim: int = 256
    head_blocks: int = 2
    head_expand: int = 4
    # duration post-processing
    max_duration: int = 50
    # Glow-TTS-lineage flow decoder + MAS training (CV-G1-E0006). When True
    # the deterministic decoder->mel regression stack is REPLACED by a
    # conditional normalizing flow; text enters through prior heads on the
    # encoder and MAS supplies durations during training (no external
    # aligner; CV-G1-E0005 decision).
    use_flow: bool = False
    flow_hidden: int = 256        # conditioner hidden width per coupling block
    flow_blocks: int = 6          # number of [inv-1x1-conv + coupling] blocks
    flow_kernel: int = 3          # conditioner conv kernel
    flow_cond_dim: int = 0        # optional per-frame cond width into the flow
    flow_noise_scale: float = 0.667  # inference latent temperature (VITS conv.)
    # init seed used by benchmark/eval harnesses (not applied here)
    seed: int = 1234

    @classmethod
    def from_json(cls, path: str | Path) -> "CaedralNARConfig":
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


class CaedralNAR(nn.Module):
    """Non-autoregressive phonemes->mel->waveform model (CPU prototype)."""

    def __init__(self, config: CaedralNARConfig | str | Path | dict):
        super().__init__()
        if isinstance(config, (str, Path)):
            config = CaedralNARConfig.from_json(config)
        elif isinstance(config, dict):
            config = CaedralNARConfig(**config)
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

        dec_cfg = BackboneConfig(
            d_model=c.d_model, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
            ff_mult=c.ff_mult, dropout=c.dropout, rope=c.rope, causal=c.causal,
        )
        if c.use_flow:
            # Flow mode: the transformer decoder + mel_proj are replaced by
            # the normalizing flow; prior heads map encoder states to
            # mel-space (mean, log-scale) per phoneme.
            self.decoder = None
            self.pitch_proj = None
            self.energy_proj = None
            self.mel_proj = None
            self.flow_prior_proj = nn.Linear(c.d_model, 2 * c.n_mels)
            self.flow = FlowDecoder(
                n_mels=c.n_mels,
                hidden_channels=c.flow_hidden,
                n_blocks=c.flow_blocks,
                cond_dim=c.flow_cond_dim,
                vec_dim=cond_dim,
                kernel_size=c.flow_kernel,
                dropout=c.dropout,
            )
        else:
            self.decoder = Transformer(dec_cfg, c.n_layers_dec, cond_dim=cond_dim)
            self.pitch_proj = nn.Linear(1, c.d_model)
            self.energy_proj = nn.Linear(1, c.d_model)
            self.mel_proj = nn.Linear(c.d_model, c.n_mels)

        self.head = IstftHead(
            n_mels=c.n_mels, dim=c.head_dim, n_blocks=c.head_blocks,
            expand=c.head_expand, n_fft=c.n_fft, hop_length=c.hop_length,
        )

    # ------------------------------------------------------------------ #
    # parameter accounting
    # ------------------------------------------------------------------ #
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Params bucketed by top-level component (insertion order kept)."""
        buckets: dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            buckets[top] = buckets.get(top, 0) + p.numel()
        return dict(sorted(buckets.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_id: int = 0,
        style_id: int | None = None,  # None -> neutral style
        speed: float = 1.0,
        pitch_shift: float = 0.0,  # semitones
        energy_gain: float = 0.0,  # dB
        durations: torch.Tensor | None = None,  # optional teacher/override (B, T)
        noise_scale: float | None = None,  # flow mode only (see infer)
    ) -> dict:
        """Batch-1-friendly NAR pass. Returns dict with waveform/mel/tracks.

        waveform: (B, T_mel * hop_length) float32
        mel:      (B, T_mel, n_mels)
        """
        if self.config.use_flow:
            # Flow models have no deterministic regression path; MAS owns the
            # training alignment. Teacher-forced ``durations`` are honored.
            return self.infer(
                phoneme_ids, speaker_id=speaker_id, style_id=style_id,
                speed=speed, pitch_shift=pitch_shift, energy_gain=energy_gain,
                noise_scale=noise_scale, durations=durations,
            )
        c = self.config
        ids = phoneme_ids if torch.is_tensor(phoneme_ids) else torch.tensor(
            phoneme_ids, dtype=torch.long
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        x = self.phoneme_embed(ids)                       # (B, T, D)
        h = self.encoder(x)                               # (B, T, D)
        s = self.conditioning(speaker_id, style_id)       # (B, speaker+style)

        # --- duration -> length regulator (speed divides durations) -------
        log_d = None
        if durations is None:
            log_d = self.duration_predictor(h, s)
            d = torch.round(torch.exp(log_d) / float(speed))
            durations = d.clamp(1, c.max_duration).long()
        else:
            durations = durations if torch.is_tensor(durations) else torch.tensor(
                durations, dtype=torch.long
            )
            if durations.dim() == 1:
                durations = durations.unsqueeze(0)
        h_exp, _mask = expand_tokens(h, durations)        # (B, T_mel, D)
        t_mel = h_exp.shape[1]

        # --- pitch / energy on expanded frames; knobs transform tracks ----
        log_f0 = self.pitch_predictor(h_exp, s)           # (B, T_mel)
        f0_hz = torch.exp(log_f0) * (2.0 ** (float(pitch_shift) / 12.0))
        e_db = self.energy_predictor(h_exp, s) + float(energy_gain)

        dec_in = h_exp
        dec_in = dec_in + self.pitch_proj(
            (torch.log(f0_hz.clamp_min(1e-5))).unsqueeze(-1)
        )
        dec_in = dec_in + self.energy_proj((e_db / 20.0).unsqueeze(-1))

        y = self.decoder(dec_in, s)                       # (B, T_mel, D)
        mel = self.mel_proj(y)                            # (B, T_mel, n_mels)
        waveform = self.head(mel.transpose(1, 2))         # (B, T_mel*hop)

        return {
            "waveform": waveform,
            "mel": mel,
            "durations": durations,
            "log_duration": log_d,  # None when durations were overridden
            "pitch_hz": f0_hz,
            "energy_db": e_db,
            "n_mel_frames": t_mel,
        }

    # ------------------------------------------------------------------ #
    # flow-mode paths (CV-G1-E0006; require use_flow=True)
    # ------------------------------------------------------------------ #
    def infer(
        self,
        phoneme_ids: torch.Tensor,
        speaker_id: int = 0,
        style_id: int | None = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,  # semitones
        energy_gain: float = 0.0,  # dB
        noise_scale: float | None = None,  # None -> config.flow_noise_scale
        durations: torch.Tensor | None = None,  # optional override (B, T_x)
    ) -> dict:
        """Flow-model synthesis: sample z ~ N(m_exp, noise_scale*sigma_exp)
        per expanded frame and run the inverse flow (Glow-TTS section 3.3).

        ``noise_scale`` is the latent temperature: it multiplies the sampled
        deviation from the prior mean (noise_scale=0 -> deterministic prior
        mean; the config default follows the VITS convention 1/sqrt(2)).
        ``durations`` overrides the duration predictor (teacher-forced
        synthesis / latency measurement at controlled expansion).
        """
        if not self.config.use_flow:
            raise RuntimeError("infer() requires CaedralNARConfig(use_flow=True)")
        c = self.config
        ns = float(c.flow_noise_scale if noise_scale is None else noise_scale)
        ids = phoneme_ids if torch.is_tensor(phoneme_ids) else torch.tensor(
            phoneme_ids, dtype=torch.long
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        x = self.phoneme_embed(ids)
        h = self.encoder(x)                                # (B, T_x, D)
        s = self.conditioning(speaker_id, style_id)        # (B, C_cond)

        log_d = None
        if durations is None:
            log_d = self.duration_predictor(h, s)          # (B, T_x)
            d = torch.round(torch.exp(log_d) / float(speed))
            d = d.clamp(1, c.max_duration).long()
        else:
            if not torch.is_tensor(durations):
                durations = torch.tensor(durations, dtype=torch.long)
            d = durations if durations.dim() == 2 else durations.unsqueeze(0)
            d = d.clamp(1, c.max_duration).long().to(ids.device)
        h_exp, mask = expand_tokens(h, d)                  # (B, T_y, D)

        m_p, logs_p = self.flow_prior_proj(h).chunk(2, dim=-1)  # (B,T_x,n_mels)
        m_e, _mask2 = expand_tokens(m_p, d)                # (B, T_y, n_mels)
        logs_e, _ = expand_tokens(logs_p, d)

        eps = torch.randn_like(m_e)
        z = m_e + eps * torch.exp(logs_e) * ns             # (B, T_y, n_mels)
        mel = self.flow.inverse(
            z.transpose(1, 2), cond=None, vec=s, noise_scale=1.0
        )                                                  # (B, n_mels, T_y)
        mel = mel.transpose(1, 2) * mask.unsqueeze(-1).to(mel.dtype)

        # prosody tracks on expanded frames; knobs transform tracks as in
        # the E0002 path (research doc section 8.2)
        log_f0 = self.pitch_predictor(h_exp, s)
        f0_hz = torch.exp(log_f0) * (2.0 ** (float(pitch_shift) / 12.0))
        e_db = self.energy_predictor(h_exp, s) + float(energy_gain)

        waveform = self.head(mel.transpose(1, 2))          # (B, T_y*hop)
        return {
            "waveform": waveform,
            "mel": mel,
            "durations": d,
            "log_duration": log_d,
            "pitch_hz": f0_hz,
            "energy_db": e_db,
            "n_mel_frames": int(mask.sum(dim=1).max()),
            "noise_scale": ns,
        }

    def losses_for_training(
        self,
        mel: torch.Tensor,                       # (B, T_y, n_mels) teacher mel
        phoneme_ids: torch.Tensor,               # (B, T_x)
        speaker_id: int | torch.Tensor = 0,
        style_id: int | torch.Tensor | None = None,
        mel_lengths: torch.Tensor | None = None,     # (B,) valid mel frames
        phoneme_lengths: torch.Tensor | None = None, # (B,) valid phonemes
    ) -> dict:
        """One MAS training step's losses (no optimizer logic here).

        Returns dict with:
          flow_loss      - negative log-likelihood of ``mel`` under the MAS
                           alignment (flow change-of-variables + Gaussian
                           prior), normalized per valid mel frame;
          duration_loss  - log-normal prior NLL (sigma=1) of MAS durations
                           under the duration predictor, per valid phoneme;
          pitch_loss / energy_loss - placeholders (0.0) for Phase G+;
          loss           - flow_loss + duration_loss;
          mas_durations  - hard alignment run lengths (B, T_x);
          alignment_score- summed log-likelihood along the found path.
        """
        if not self.config.use_flow:
            raise RuntimeError(
                "losses_for_training() requires CaedralNARConfig(use_flow=True)"
            )
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
        b, t_y, _mels = mel.shape
        t_x = ids.shape[1]
        dev, dtp = mel.device, mel.dtype

        xl = (
            torch.full((b,), t_x, dtype=torch.long, device=dev)
            if phoneme_lengths is None
            else phoneme_lengths.to(dev)
        )
        yl = (
            torch.full((b,), t_y, dtype=torch.long, device=dev)
            if mel_lengths is None
            else mel_lengths.to(dev)
        )
        y_ar = torch.arange(t_y, device=dev)
        x_ar = torch.arange(t_x, device=dev)
        y_mask = y_ar[None, :] < yl[:, None]         # (B, T_y)
        x_mask = x_ar[None, :] < xl[:, None]         # (B, T_x)

        x = self.phoneme_embed(ids)
        h = self.encoder(x)                          # (B, T_x, D)
        s = self.conditioning(speaker_id, style_id)

        mu_dur = self.duration_predictor(h, s)       # (B, T_x) log-domain
        m_p, logs_p = self.flow_prior_proj(h).chunk(2, dim=-1)

        # one flow pass data->latent (conditioned on speaker|style via FiLM)
        z, logdet_f = self.flow.forward(
            mel.transpose(1, 2), cond=None, vec=s
        )                                            # z: (B, n_mels, T_y)
        z = z.transpose(1, 2)                        # (B, T_y, n_mels)

        # ---- MAS score matrix: ln N(z_j; m_p_i, sigma_p_i) over channels -- #
        # chunked over frames to bound peak memory at long utterances
        frame_loglik = mel.new_empty(b, t_x, t_y)
        chunk = max(1, min(t_y, int(6_000_000 // max(t_x * _mels, 1))))
        for j0 in range(0, t_y, chunk):
            j1 = min(j0 + chunk, t_y)
            diff = z[:, None, j0:j1, :] - m_p[:, :, None, :]
            inv_std = torch.exp(-logs_p[:, :, None, :])
            ll = -0.5 * (
                (diff * inv_std) ** 2 + 2.0 * logs_p[:, :, None, :] + _LOG_2PI
            ).sum(dim=-1)                            # (B, T_x, j1-j0)
            frame_loglik[:, :, j0:j1] = ll

        # hard alignment (no gradients through the search)
        with torch.no_grad():
            durations, align_score = maximum_path(
                frame_loglik.detach(), x_lengths=xl, y_lengths=yl
            )
        durations = durations.to(dev)                # (B, T_x), each >= 1

        # ---- flow loss under the chosen alignment (grads restored) ------- #
        # frame->phoneme map from durations; every valid frame covered once
        maps = []
        for bi in range(b):
            maps.append(
                torch.repeat_interleave(x_ar, durations[bi], dim=0)
            )                                        # (y_len_b,)
        frame_to_ph = torch.stack(
            [
                F.pad(m, (0, t_y - m.numel()), value=0)
                for m in maps
            ]
        )                                            # (B, T_y), pad masked out
        m_f = m_p.gather(
            1, frame_to_ph[..., None].expand(-1, -1, m_p.shape[-1])
        )
        logs_f = logs_p.gather(
            1, frame_to_ph[..., None].expand(-1, -1, logs_p.shape[-1])
        )
        n_frames = yl.sum()
        n_phones = xl.sum()
        ll_frame = -0.5 * (
            ((z - m_f) / torch.exp(logs_f)) ** 2
            + 2.0 * logs_f
            + _LOG_2PI
        ).sum(dim=-1)                                # (B, T_y)
        loglik_total = (
            (ll_frame * y_mask.to(dtp)).sum()
            + (logdet_f * y_mask.to(dtp)).sum()      # change-of-variables term
        )
        flow_loss = -loglik_total / n_frames.to(dtp)

        # ---- duration loss: log-normal prior NLL (sigma = 1) ------------- #
        d_mas = durations.to(dtp)
        dur_nll = 0.5 * (torch.log(d_mas.clamp_min(1e-6)) - mu_dur) ** 2
        duration_loss = (dur_nll * x_mask.to(dtp)).sum() / n_phones.to(dtp)

        zero = torch.zeros((), dtype=dtp, device=dev)
        return {
            "flow_loss": flow_loss,
            "duration_loss": duration_loss,
            "pitch_loss": zero,
            "energy_loss": zero.clone(),
            "loss": flow_loss + duration_loss,
            "mas_durations": durations,
            "alignment_score": align_score.to(dev),
            "n_mel_frames": t_y,
            "n_phonemes": t_x,
        }


def load_model(config_path: str | Path, seed: int | None = None) -> CaedralNAR:
    """Build a seeded prototype from a configs/*.json file."""
    cfg = CaedralNARConfig.from_json(config_path)
    if seed is None:
        seed = cfg.seed
    torch.manual_seed(seed)
    return CaedralNAR(cfg)
