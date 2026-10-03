"""CaedralEngine: one-process CPU synthesis engine around a trained run.

Loads everything a checkpointed pilot run needs (phoneme vocab, speaker
index, model weights, config) with zero manual glue, then turns text into
24 kHz float32 audio via the frozen G1 path:

    TextFrontend('lexicon+g2pen') -> CaedralNAR(use_flow=True).infer()
    -> 80-band ln-mel -> linear interp to Vocos's 100-band grid
    -> pretrained Vocos decode (MIT, charactr/vocos-mel-24khz)

The mel adapter is the documented approximation from the preview pipeline
(scripts/generate_caedral_preview.py); the model's own iSTFT head remains an
early-training stub for these checkpoints.

Thread model (design doc section 4.1): constructed once, immutable after
warmup, shared read-only; all synthesis is serialized by an internal lock so
a single process can call synth() from multiple threads safely. Batch-1
tensors are request-local and inference runs under torch.inference_mode().
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig
from caedral_voice.text import TextFrontend

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "tiny.json"
VOCOS_MODEL_ID = "charactr/vocos-mel-24khz"
FRONTEND_BACKEND = "lexicon+g2pen"
# 'caedral' is out-of-lexicon on purpose: it forces the one-time g2p_en
# (NLTK LSTM) construction during warmup instead of on a first request
# (~12 s cold, measured on this box).
WARMUP_TEXT = "Hello caedral."


@dataclass
class SynthStats:
    """Diagnostics of the most recent synth() call."""

    wall_seconds: float
    audio_seconds: float
    rtf: float                      # wall / audio; <1.0 means faster than real time
    n_phonemes: int
    skipped_symbols: list[str] = field(default_factory=list)
    voice_index: int = -1
    sample_rate: int = 24000


class CaedralEngine:
    """Load-once synthesis engine bound to one training run directory."""

    def __init__(
        self,
        *,
        model: CaedralNAR,
        frontend: TextFrontend,
        vocoder,
        symbols: list[str],
        specials: dict,
        speaker_index: dict[str, int],
        run_dir: Path,
        checkpoint_path: Path,
    ) -> None:
        self.model = model
        self.frontend = frontend
        self.vocoder = vocoder
        self.symbols = list(symbols)
        self.specials = dict(specials)
        self.speaker_index = {
            str(k): int(v) for k, v in speaker_index.items()
        }
        self.run_dir = Path(run_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.sample_rate: int = int(model.config.sample_rate)
        self._sym2id = {s: i for i, s in enumerate(self.symbols)}
        # speaker_index.json values ARE the model's integer speaker ids;
        # a "voice index" is one of those ids.
        self._voice_ids = set(self.speaker_index.values())
        self._lock = threading.Lock()
        self.last_stats: SynthStats | None = None

    # ------------------------------------------------------------ loading --

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        *,
        threads: int | None = None,
        warmup: bool = True,
    ) -> "CaedralEngine":
        """Build an engine from a run dir + checkpoint with no manual glue.

        checkpoint_path None -> checkpoints/last.pt > best.pt > newest
        step_*.pt inside the run. config_path None -> repo configs/tiny.json.
        The checkpoint's embedded cfg overrides file-config keys where both
        define them (mirrors scripts/generate_caedral_preview.py); use_flow
        is forced True (the G1 inference path).
        """
        if threads is not None:
            torch.set_num_threads(int(threads))
        run_dir = Path(run_dir)
        ckpt_path = _resolve_checkpoint(run_dir, checkpoint_path)
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

        ck = _load_checkpoint(ckpt_path)
        config = _merge_config(cfg_path, ck.get("cfg") or {})
        model = CaedralNAR(config)
        state, n_dropped = _strip_model_prefix(ck["model"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "state_dict mismatch in %s: %d missing, %d unexpected keys",
                ckpt_path.name, len(missing), len(unexpected),
            )
        if n_dropped:
            logger.warning("dropped %d non-'model.' keys from checkpoint", n_dropped)
        model.eval()

        vocab = json.loads(
            (run_dir / "checkpoints" / "phoneme_vocab.json").read_text(encoding="utf-8")
        )
        speaker_index = json.loads(
            (run_dir / "checkpoints" / "speaker_index.json").read_text(encoding="utf-8")
        )

        frontend = TextFrontend(FRONTEND_BACKEND)
        vocoder = _load_vocoder()

        engine = cls(
            model=model,
            frontend=frontend,
            vocoder=vocoder,
            symbols=vocab["symbols"],
            specials=vocab.get("specials", {}),
            speaker_index=speaker_index,
            run_dir=run_dir,
            checkpoint_path=ckpt_path,
        )
        step = (ck.get("trainer") or {}).get("global_step")
        logger.info(
            "loaded %s (%s) step=%s use_flow=%s threads=%s",
            ckpt_path, run_dir.name, step, config.use_flow,
            threads if threads is not None else "default",
        )
        if warmup:
            first = min(engine.speaker_index.values())
            t0 = time.perf_counter()
            engine.synth(WARMUP_TEXT, first)
            logger.info(
                "warmup synthesis done in %.2fs incl. frontend/g2p_en "
                "preheat (excluded from steady-state timing)",
                time.perf_counter() - t0,
            )
        return engine

    # ------------------------------------------------------------- voices --

    @property
    def voices(self) -> list[dict]:
        """Catalog rows [{id, speaker_name, index}] ordered by index."""
        return [
            {"id": name, "speaker_name": name, "index": idx}
            for name, idx in sorted(self.speaker_index.items(), key=lambda kv: kv[1])
        ]

    def resolve_voice(self, voice: str | int) -> int:
        """'103' or 0-style identifiers -> internal voice index."""
        if isinstance(voice, bool):
            raise ValueError(f"invalid voice identifier: {voice!r}")
        if isinstance(voice, int):
            idx = voice
        else:
            text = str(voice).strip()
            if text in self.speaker_index:          # exact name wins
                return self.speaker_index[text]
            if text.lstrip("-").isdigit():          # then numeric index
                idx = int(text)
            else:
                raise ValueError(_unknown_voice_msg(text, self.voices))
        if idx not in self._voice_ids:
            raise ValueError(_unknown_voice_msg(idx, self.voices))
        return idx

    # ---------------------------------------------------------- synthesis --

    def synth(
        self,
        text: str,
        voice_index: int,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy_gain: float = 0.0,
        noise_scale: float | None = None,
        # E0009: config default 0.667 smears consonants (iid prior sampling);
        # 0.25 measured WER 0.44->0.11 on live checkpoints. Applied as the
        # engine-level default until the acoustic model is retrained.
    ) -> np.ndarray:
        """text -> np.float32 waveform at ``sample_rate`` (24 kHz).

        Knobs follow the model contract: ``speed`` divides durations,
        ``pitch_shift`` is semitones on F0, ``energy_gain`` adds dB to the
        energy track; ``noise_scale`` is the flow latent temperature
        (None -> config.flow_noise_scale).
        """
        idx = self.resolve_voice(voice_index)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        with self._lock:
            t0 = time.perf_counter()
            result = self.frontend.process(text)
            ids, skipped = self._tokens_to_ids(result.phoneme_tokens)
            if skipped:
                # ascii() keeps IPA-safe logging on cp1252 consoles too.
                logger.warning(
                    "skipped %d frontend symbol(s) not in phoneme vocab: %s",
                    len(skipped), ascii(sorted(set(skipped))),
                )
            if not ids:
                raise ValueError(
                    "text produced no in-vocab phonemes "
                    "(only punctuation/emoji?)"
                )

            spk_id = int(idx)
            with torch.inference_mode():
                out = self.model.infer(
                    torch.tensor([ids], dtype=torch.long),
                    speaker_id=spk_id,
                    style_id=0,
                    speed=float(speed),
                    pitch_shift=float(pitch_shift),
                    energy_gain=float(energy_gain),
                    noise_scale=0.25 if noise_scale is None else noise_scale,
                )
                mel = out["mel"].squeeze(0).to(torch.float32).cpu().numpy()
                n_valid = int(out["n_mel_frames"])
                mel = mel[:n_valid]
                mel100_t = _mel80_to_vocos100(mel)
                wav = (
                    self.vocoder.decode(mel100_t).squeeze(0).to(torch.float32)
                    .cpu().numpy()
                )

            audio = np.asarray(wav, dtype=np.float32)
            wall = time.perf_counter() - t0
            audio_s = len(audio) / self.sample_rate
            self.last_stats = SynthStats(
                wall_seconds=wall,
                audio_seconds=audio_s,
                rtf=wall / max(audio_s, 1e-6),
                n_phonemes=len(ids),
                skipped_symbols=sorted(set(skipped)),
                voice_index=idx,
                sample_rate=self.sample_rate,
            )
            return audio

    def _tokens_to_ids(self, tokens) -> tuple[list[int], list[str]]:
        ids: list[int] = []
        skipped: list[str] = []
        for tok in tokens:
            if tok in self._sym2id:
                ids.append(self._sym2id[tok])
            elif tok in self.specials:
                ids.append(int(self.specials[tok]))
            else:
                skipped.append(tok)
        return ids, skipped


# ------------------------------------------------------------------ helpers --


def _resolve_checkpoint(run_dir: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    ck_dir = run_dir / "checkpoints"
    for name in ("last.pt", "best.pt"):
        p = ck_dir / name
        if p.is_file():
            return p
    steps = sorted(ck_dir.glob("step_*.pt"))
    if steps:
        return steps[-1]
    raise FileNotFoundError(f"no checkpoint found under {ck_dir}")


def _unsafe_checkpoint_fallback_allowed() -> bool:
    """Checkpoints that embed non-tensor pickle objects (e.g. RNG state)
    need weights_only=False, which executes pickle code. Off by default so
    a third-party checkpoint file cannot run arbitrary unpickling; set
    CV_ALLOW_UNSAFE_CHECKPOINT=1 for trusted local artifacts."""
    return os.environ.get("CV_ALLOW_UNSAFE_CHECKPOINT", "").strip() == "1"


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it's a pickle gate
        transient = "Weights only load failed" in str(exc) or "Unpickler" in type(exc).__name__
        if not transient:
            raise
        if not _unsafe_checkpoint_fallback_allowed():
            raise ValueError(
                f"{path.name} embeds non-tensor pickle objects; refusing to "
                "unpickle them. Re-export the checkpoint tensors-only or set "
                "CV_ALLOW_UNSAFE_CHECKPOINT=1 for trusted local artifacts."
            ) from exc
        logger.warning(
            "%s embeds non-tensor pickle objects (e.g. RNG state); loading "
            "with weights_only=False -- local trusted artifacts only", path.name,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _merge_config(config_path: Path, ck_cfg: dict) -> CaedralNARConfig:
    known = set(CaedralNARConfig.__dataclass_fields__)
    merged = {
        k: v for k, v in json.loads(
            Path(config_path).read_text(encoding="utf-8")).items()
        if k in known
    }
    merged.update({k: v for k, v in ck_cfg.items() if k in known})
    merged["use_flow"] = True
    return CaedralNARConfig(**merged)


def _strip_model_prefix(raw: dict) -> tuple[dict, int]:
    """Strip the wrapper 'model.' prefix; count keys that carry neither."""
    state: dict = {}
    dropped = 0
    for k, v in raw.items():
        if k.startswith("model."):
            state[k[len("model."):]] = v
        elif k in _known_state_keys_hint():  # already bare (defensive)
            state[k] = v
        else:
            dropped += 1
    return state, dropped


def _known_state_keys_hint() -> set[str]:
    # Top-level module names of CaedralNAR; enough to recognize bare keys.
    return {
        "phoneme_embed", "encoder", "conditioning", "duration_predictor",
        "pitch_predictor", "energy_predictor", "flow_prior_proj", "flow",
        "head", "decoder", "pitch_proj", "energy_proj", "mel_proj",
    }


def _load_vocoder():
    try:
        from vocos import Vocos
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the 'vocos' package is required for decoding "
            f"({VOCOS_MODEL_ID}); install it into the active environment"
        ) from exc
    return Vocos.from_pretrained(VOCOS_MODEL_ID)


def _mel80_to_vocos100(mel80: np.ndarray) -> torch.Tensor:
    """(T, 80) ln-mel -> (1, 100, T) tensor on Vocos's band grid.

    Linear interpolation between our 80 mel bands onto 100 bands — the same
    documented approximation as the preview pipeline (not a perceptual or
    ERB-domain conversion).
    """
    xs100 = np.linspace(0.0, 79.0, 100)
    grid80 = np.arange(80)
    mel100 = np.stack(
        [np.interp(xs100, grid80, frame) for frame in mel80], axis=0
    ).T.astype(np.float32)                                   # (100, T)
    return torch.from_numpy(mel100).unsqueeze(0)             # (1, 100, T)


def _unknown_voice_msg(voice, voices: list[dict]) -> str:
    catalog = ", ".join(
        f"{v['speaker_name']}={v['index']}" for v in voices
    )
    return f"unknown voice {voice!r}; available: {catalog}"
