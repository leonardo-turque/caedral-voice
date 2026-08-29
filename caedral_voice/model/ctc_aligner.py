"""Supervised CTC text->mel aligner (experiment CV-G1-E0011).

DIAG-02 condemned MAS alignments (silence-dumping) and flow sampling. The
founder pivot: learn durations SUPERVISED with our own tiny CTC aligner,
then train a FastSpeech2-style mel regressor on them. This module is that
aligner::

    log-mel [B, T, 80] -> conv subsample (time /2, freq 80->20)
                       -> linear + sinusoidal positions
                       -> 3x pre-LN transformer encoder (d 192, ff 768)
                       -> linear -> log-softmax over vocab + blank

~1.6 M parameters (budget <=5 M). Two measured design constraints:

* Encoder: a 2x BiLSTM at equivalent quality costs ~10x more WALL CLOCK on
  CPU (full-size batch fwd+bwd 12.6 s vs 1.3 s -- sequential dependency
  kills thread utilization), so the encoder is a conv-transformer.
* Time subsample is /2, NOT /4: standard CTC needs >=1 blank frame between
  every pair of ADJACENT DIFFERENT symbols, so the shortest valid path is
  ~L + distinct-adjacent-pairs (~2L for typical phoneme strings). At 93.75
  mel fps that demands >= 32.9 frames/s (median need; p99 47.8) -- /4
  (23.4 fps) leaves only ~6 % of librittsr-clean100 alignable, /2 (46.9
  fps) leaves ~99.3 %. Measured on the corpus, CV-G1-E0011.

The blank class is APPENDED after the phoneme vocab (class index ==
``vocab_size``), so phoneme ids stay exactly the
:class:`~caedral_voice.training.mel_dataset.PhonemeVocab` ids and the pause
markers ``|`` (word) / ``‖`` (utterance) remain ordinary classes -- the
network learns to spend frames on them during silent spans, which is
precisely what downstream duration extraction needs (E0009 showed MAS
instead dumps silence onto content phonemes).

Forced alignment (:func:`viterbi_force_align`) runs the textbook CTC
dynamic program over the blank-interleaved label sequence and backtracks
the maximum-score path. Per-token durations are the spans between
successive symbol-state first emissions, with blank frames attached to the
PRECEDING token (leading blanks fold into token 0, trailing into the last).
Durations are strictly monotone and sum to the frame count by construction;
:meth:`expand_ds_durations_to_mel` maps them back to full-rate mel frames
(sum preserved, minimum ``TIME_SUBSAMPLE`` frames per phone except possibly
the last).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

#: time subsampling factor of the conv frontend (single stride-2 stage)
TIME_SUBSAMPLE = 2


def min_ctc_frames(ids) -> int:
    """Shortest valid CTC path length for an id sequence.

    One frame per symbol plus one blank between every pair of ADJACENT
    DIFFERENT symbols (identical neighbours may collapse via the skip
    transition). Callers must guarantee the encoder emits at least this
    many frames -- otherwise NO valid alignment exists and both training
    (``zero_infinity``) and forced alignment reject the row.
    """
    ids = list(ids)
    return len(ids) + sum(a != b for a, b in zip(ids, ids[1:]))


def _conv_out_len(length: int) -> int:
    """Output length of one k=3, s=2, p=1 stage: ceil(length / 2)."""
    return (length - 1) // 2 + 1


@dataclass
class CTCAlignerConfig:
    """Hyperparameters; 1:1 serializable via :func:`dataclasses.asdict`."""

    n_mels: int = 80
    #: number of PHONEME symbols (PhonemeVocab size); blank is appended, so
    #: the output layer has ``vocab_size + 1`` classes.
    vocab_size: int = 43
    conv_channels: int = 32          # stage 1; stage 2 doubles it
    d_model: int = 192
    nhead: int = 4
    dim_feedforward: int = 768
    enc_layers: int = 3
    dropout: float = 0.1
    max_positions: int = 2048        # sinusoidal table size (frames /4)

    @property
    def num_classes(self) -> int:
        return self.vocab_size + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CTCAlignerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def _sinusoidal_table(max_positions: int, d_model: int) -> torch.Tensor:
    """Standard transformer sinusoidal position encoding [1, P, D]."""
    pos = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_positions, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe.unsqueeze(0)


class CTCAligner(nn.Module):
    """Tiny conv-transformer CTC acoustic model over log-mels."""

    def __init__(self, config: CTCAlignerConfig | None = None):
        super().__init__()
        cfg = self.config = config or CTCAlignerConfig()
        self.blank_id = cfg.vocab_size          # blank appended after vocab

        self.conv = nn.Sequential(
            nn.Conv2d(1, cfg.conv_channels, kernel_size=3,
                      stride=(2, 2), padding=1),          # time /2, freq /2
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.conv_channels, cfg.conv_channels * 2,
                      kernel_size=3, stride=(1, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )                                                  # freq 40 -> 20
        freq_out = _conv_out_len(_conv_out_len(cfg.n_mels))
        proj_in = cfg.conv_channels * 2 * freq_out
        self.norm = nn.LayerNorm(proj_in)
        self.proj = nn.Linear(proj_in, cfg.d_model)
        self.register_buffer(
            "pos", _sinusoidal_table(cfg.max_positions, cfg.d_model),
            persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.enc_layers, enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def out_length(self, mel_length: int) -> int:
        """/2 time subsample law for one utterance (conv stage 1)."""
        return _conv_out_len(int(mel_length))

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """mel [B, T, n_mels] -> (log_probs [B, T', num_classes],
        out_lengths [B]) with T' = ceil(T/4) per item."""
        if mel_lengths is None:
            mel_lengths = torch.full((mel.shape[0],), mel.shape[1],
                                     dtype=torch.long)
        out_lens = torch.tensor(
            [self.out_length(int(l)) for l in mel_lengths.tolist()],
            dtype=torch.long)
        x = mel.unsqueeze(1)                       # [B, T, F] -> [B, 1, T, F]
        x = self.conv(x)                           # [B, C, T', F']
        x = x.permute(0, 2, 1, 3).flatten(2)       # [B, T', C*F']
        max_t = x.shape[1]
        x = self.proj(self.norm(x)) + self.pos[:, :max_t]   # [B, T', D]
        pad_mask = torch.arange(max_t, device=x.device).unsqueeze(0) \
            >= out_lens.to(x.device).unsqueeze(1)           # True = ignore
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        logits = self.head(x)                      # [B, T', num_classes]
        return logits.log_softmax(dim=-1), out_lens


# ---------------------------------------------------------------------- #
# forced (Viterbi) alignment
# ---------------------------------------------------------------------- #
_NEG = -1e30


def viterbi_force_align(
    log_probs: "np.ndarray | torch.Tensor",
    ids,
    blank_id: int,
) -> dict:
    """Maximum-score CTC forced alignment of one utterance.

    Args:
        log_probs: [T, num_classes] log-softmax output for ONE utterance
            (numpy or torch; converted to float64 numpy).
        ids: target phoneme-id sequence (length L, no blanks).
        blank_id: class index of blank (= ``vocab_size`` in our scheme).

    Returns dict with:
        ``durations_ds``: per-token durations in SUBSAMPLED frames
            (sum == T, each >= 1);
        ``starts_ds``: first subsampled frame of each token;
        ``states``: [T] non-decreasing CTC state index per frame;
        ``score``: path log-prob.
    Raises ValueError when T < L (no valid alignment exists).
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()
    lp = np.asarray(log_probs, dtype=np.float64)
    T, _C = lp.shape
    y = np.asarray(list(ids), dtype=np.int64)
    L = int(y.size)
    if T < L:
        raise ValueError(f"cannot align: frames {T} < tokens {L}")
    S = 2 * L + 1
    z = np.full(S, int(blank_id), dtype=np.int64)
    z[1::2] = y

    dp = np.full(S, _NEG, dtype=np.float64)
    dp[0] = lp[0, z[0]]
    dp[1] = lp[0, z[1]]
    choice = np.zeros((T, S), dtype=np.int8)       # 0 stay, 1 from s-1, 2 skip

    for t in range(1, T):
        prev = dp
        best = prev.copy()                         # default transition: stay
        ch = np.zeros(S, dtype=np.int8)
        # enter from s-1
        cand1 = prev[:-1]
        move1 = cand1 > best[1:]
        best[1:] = np.where(move1, cand1, best[1:])
        ch[1:] = np.where(move1, 1, ch[1:])
        # skip s-2 -> s (only identical NON-blank neighbours may collapse)
        ok2 = np.zeros(S, dtype=bool)
        ok2[2:] = (z[2:] != blank_id) & (z[2:] == z[:-2])
        cand2 = prev[:-2]
        move2 = ok2[2:] & (cand2 > best[2:])
        best[2:] = np.where(move2, cand2, best[2:])
        ch[2:] = np.where(move2, 2, ch[2:])
        dp = best + lp[t, z]                       # emit this frame's label
        choice[t] = ch

    # ---- backtrack -------------------------------------------------------
    if max(dp[S - 1], dp[S - 2]) <= _NEG / 2:
        raise ValueError(
            f"no valid CTC alignment: {T} frames cannot cover {L} tokens "
            "(need ~L + number of distinct-consecutive-symbol blanks)")
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        if t > 0:
            c = int(choice[t, s])
            if c == 1:
                s -= 1
            elif c == 2:
                s -= 2
    assert np.all(np.diff(states) >= 0), "backtracked path must be monotone"
    durs_ds, starts_ds = states_to_durations(states, L)
    return {
        "durations_ds": [int(d) for d in durs_ds],
        "starts_ds": [int(s0) for s0 in starts_ds],
        "states": states,
        "score": float(max(dp[S - 1], dp[S - 2])),
    }


def states_to_durations(states: np.ndarray, n_tokens: int
                        ) -> tuple[list[int], list[int]]:
    """CTC state path -> (durations_ds, starts_ds) per original token.

    Blank frames attach to the PRECEDING token; leading blanks fold into
    token 0, trailing blanks into the last token. Sum == len(states).
    """
    starts = []
    for i in range(n_tokens):
        pos = np.searchsorted(states, 2 * i + 1, side="left")
        starts.append(int(pos))
    # token i owns [edge_i, edge_i+1): from its first symbol frame to the
    # NEXT token's first symbol frame; blanks in between attach forward to
    # the earlier token. Leading blanks (before starts[0]) fold into
    # token 0 via edge_0 == 0.
    edges = [0] + starts[1:] + [len(states)]
    durs = [edges[k + 1] - edges[k] for k in range(n_tokens)]
    return durs, starts


def expand_ds_durations_to_mel(durations_ds, total_mel_frames: int,
                               subsample: int = TIME_SUBSAMPLE) -> list[int]:
    """Subsampled-frame durations -> full-rate mel-frame durations.

    Token boundaries live on the subsample grid (start_i * subsample); the
    LAST token absorbs the sub-grid remainder so the durations sum exactly
    to ``total_mel_frames``. Every duration is >= ``subsample`` except
    possibly the last (>= 1).
    """
    starts_ds = np.cumsum([0] + list(durations_ds[:-1]))
    starts_mel = [int(s) * subsample for s in starts_ds]
    edges = starts_mel + [int(total_mel_frames)]
    durs = [edges[k + 1] - edges[k] for k in range(len(durations_ds))]
    if min(durs) <= 0:
        raise ValueError("non-positive duration after expansion")
    return durs
