"""CaedralVoiceKokoroEngine: canonical long-lived engine on the winning
production path (founder §18): kokoro_onnx ONNX graph + Caedral text
frontend + the char-level vocab mapping promoted in caedral_voice.text.kokoro_map.

Load-once semantics mirror CaedralEngine (design doc section 4.1):
constructed once, immutable after warmup, shared read-only; synthesis is
serialized by an internal lock so one process can call synth() from many
threads. The kokoro_onnx graph fuses acoustic model and vocoder into a
single ``session.run`` (the v1.0 export has exactly one output, ``audio``),
so there is no separate vocoder decode phase to manage.

Frontend results are cached keyed by SHA-256 of the input text: repeated
requests for the same string skip TN + G2P + vocab mapping entirely.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SAMPLE_RATE

from caedral_voice.text import TextFrontend
from caedral_voice.text.kokoro_map import (
    caedral_tokens_to_kokoro,
    kokoro_vocab_set,
)

logger = logging.getLogger(__name__)

FRONTEND_BACKEND = "lexicon+g2pen"
# 'caedral' is deliberately out-of-lexicon so warmup() also pays the one-time
# g2p_en (NLTK LSTM) construction instead of the first real request.
WARMUP_TEXT = "Hello caedral."
_FRONTEND_CACHE_MAX = 1024

# Production voice slate (locked): public ids -> underlying kokoro_v1.0 voice.
VOICE_PROFILES: dict[str, dict[str, str]] = {
    "caedral-m1": {"kokoro_voice": "am_michael", "gender": "male"},
    "caedral-m2": {"kokoro_voice": "am_echo", "gender": "male"},
    "caedral-f1": {"kokoro_voice": "af_sky", "gender": "female"},
    "caedral-f2": {"kokoro_voice": "af_nova", "gender": "female"},
}
SPEED_MIN, SPEED_MAX = 0.25, 2.0
# kokoro_onnx._prepare hard-rejects speed < 0.5 even though the graph's speed
# input is an unconstrained float scalar; slower "slow-motion" rates are
# served by calling the wrapper's internal batch renderer directly.
WRAPPER_SPEED_MIN = 0.5


@dataclass
class KokoroSynthStats:
    """Diagnostics of the most recent synth() call."""

    wall_seconds: float
    audio_seconds: float
    rtf: float                      # wall / audio; <1.0 means faster than real time
    n_phonemes: int
    voice: str = ""
    speed: float = 1.0
    voice_profile: str = ""         # caedral-* profile id when one matched
    speed_clamped: bool = False     # requested speed outside [SPEED_MIN, MAX]
    frontend_cache_hit: bool = False
    skipped_symbols: list[str] = field(default_factory=list)
    sample_rate: int = SAMPLE_RATE


class CaedralVoiceKokoroEngine:
    """Long-lived kokoro_onnx synthesis engine with the Caedral frontend."""

    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        opts = rt.SessionOptions()
        if threads is not None:
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
        # E0027: cpu_arena_off saves ~141 MB (−23%) with no RTF regression
        opts.enable_cpu_mem_arena = False
        t0 = time.perf_counter()
        session = rt.InferenceSession(
            str(self.model_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.session_create_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Kokoro.from_session reuses our thread-pinned session instead of
        # building its own default one (create_session offers no options hook).
        self._kokoro = Kokoro.from_session(session, str(self.voices_path))
        self.wrapper_setup_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.frontend = TextFrontend(FRONTEND_BACKEND)
        self.frontend_ctor_seconds = time.perf_counter() - t0

        self._vocab = kokoro_vocab_set(self._kokoro.tokenizer.vocab)
        self._frontend_cache: dict[str, tuple[str, list[str]]] = {}
        self._lock = threading.Lock()
        self._stats = KokoroSynthStats(wall_seconds=0.0, audio_seconds=0.0,
                                       rtf=0.0, n_phonemes=0)
        logger.info(
            "CaedralVoiceKokoroEngine ready: %d voices, %d vocab symbols, "
            "threads=%s, load %.2fs",
            len(self._kokoro.voices), len(self._vocab),
            threads, self.total_load_seconds,
        )

    @property
    def total_load_seconds(self) -> float:
        return (
            self.session_create_seconds
            + self.wrapper_setup_seconds
            + self.frontend_ctor_seconds
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def voices(self) -> list[str]:
        return sorted(self._kokoro.voices.keys())

    @property
    def last_stats(self) -> KokoroSynthStats:
        return self._stats

    @property
    def fused_vocoder(self) -> bool:
        """True when the export has no separate vocoder/duration outputs."""
        return [o.name for o in self._kokoro.sess.get_outputs()] == ["audio"]

    def phonemize(self, text: str, use_cache: bool = True) -> tuple[str, list[str]]:
        """Caedral frontend -> Kokoro-vocab phoneme string (+ skipped tokens).

        Word-boundary spaces are inserted between words so the Kokoro model
        can detect word edges (stock Kokoro phonemizer also produces spaced
        output; without spaces the model merges words into unintelligible
        streams). Results are memoized by SHA-256 of the raw text.
        """
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if use_cache:
            hit = self._frontend_cache.get(key)
            if hit is not None:
                return hit
        result = self.frontend.process(text)

        # Map each word independently and join with spaces (word boundaries).
        # Stock Kokoro's phonemizer produces space-separated words; our mapping
        # must do the same for the acoustic model to detect word edges.
        word_phonemes: list[str] = []
        skipped_all: list[str] = []
        for w in result.words:
            mapped, skipped = caedral_tokens_to_kokoro(
                list(w.selected), self._vocab
            )
            word_phonemes.append(mapped)
            skipped_all.extend(skipped)
        phonemes = " ".join(word_phonemes)
        if text.rstrip().endswith("."):
            phonemes += "."

        if len(self._frontend_cache) >= _FRONTEND_CACHE_MAX:
            self._frontend_cache.clear()
        self._frontend_cache[key] = (phonemes, skipped_all)
        return phonemes, skipped_all

    def synth(
        self,
        text: str,
        voice_id: str = "caedral-m1",
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize ``text`` to np.float32 audio at 24 kHz.

        ``voice_id`` must be a key of VOICE_PROFILES (e.g. "caedral-m1").
        Raw kokoro voice names are still accepted as a back-compat
        passthrough for internal tooling; production callers should use the
        caedral-* ids only. ``speed`` is clamped into [0.25, 2.0].
        """
        profile = VOICE_PROFILES.get(voice_id)
        if profile is not None:
            voice_name = profile["kokoro_voice"]
        elif voice_id in self._kokoro.voices:
            # Back-compat passthrough (internal scripts use raw kokoro names).
            voice_name = voice_id
        else:
            raise ValueError(
                f"Unknown voice_id {voice_id!r}; profiles: "
                f"{sorted(VOICE_PROFILES)}"
            )

        requested_speed = float(speed)
        clamped = not SPEED_MIN <= requested_speed <= SPEED_MAX
        speed = min(max(requested_speed, SPEED_MIN), SPEED_MAX)

        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            cache_hit = key in self._frontend_cache
            t0 = time.perf_counter()
            phonemes, skipped = self.phonemize(text)
            if speed >= WRAPPER_SPEED_MIN:
                audio, _ = self._kokoro.create(
                    phonemes, voice=voice_name, speed=speed,
                    lang="en-us", is_phonemes=True,
                )
            else:
                audio = self._synth_slow_motion(
                    phonemes, voice_name, speed)
            wall = time.perf_counter() - t0
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        n_tokens = len(self._kokoro.tokenizer.tokenize(phonemes))
        self._stats = KokoroSynthStats(
            wall_seconds=wall,
            audio_seconds=len(audio) / SAMPLE_RATE,
            rtf=wall / max(len(audio) / SAMPLE_RATE, 1e-9),
            n_phonemes=n_tokens,
            voice=voice_name,
            speed=speed,
            voice_profile=voice_id if profile is not None else "",
            speed_clamped=clamped,
            frontend_cache_hit=cache_hit,
            skipped_symbols=list(skipped),
        )
        return audio

    def _synth_slow_motion(
        self, phonemes: str, voice_name: str, speed: float
    ) -> np.ndarray:
        """Render speed < WRAPPER_SPEED_MIN by bypassing kokoro_onnx's guard.

        ``Kokoro._prepare`` rejects speed < 0.5, but the graph's ``speed``
        input is a plain float duration scalar, so we replicate its batching
        (split + inter-batch pauses, trim on) and call ``_create_batch`` per
        batch with the requested sub-floor rate.
        """
        from kokoro_onnx.chunker import pause_after

        style = self._kokoro.get_voice_style(voice_name)
        batches = self._kokoro._split_phonemes(phonemes)
        if not batches:
            raise ValueError(f"Nothing to synthesize: {phonemes!r}")
        chunks = []
        for i, batch in enumerate(batches):
            audio_b, _ = self._kokoro._create_batch(
                batch, style, speed=speed, trim=True,
                pause=0.0 if i == len(batches) - 1
                else pause_after(batch, 0.25, 0.1),
            )
            chunks.append(np.asarray(audio_b))
        return np.concatenate(chunks)

    def infer_from_phonemes(
        self, phoneme_str: str, voice_name: str, speed: float = 1.0
    ) -> np.ndarray:
        """Raw path: run the fused graph on a pre-mapped phoneme string."""
        audio, _ = self._kokoro.create(
            phoneme_str, voice=voice_name, speed=speed,
            lang="en-us", is_phonemes=True,
        )
        return np.ascontiguousarray(audio, dtype=np.float32)

    def warmup(self) -> float:
        """Pay all one-time costs (NLTK g2p construction, ORT graph init).

        Returns the warmup wall seconds.
        """
        t0 = time.perf_counter()
        self.phonemize(WARMUP_TEXT, use_cache=False)
        self.synth(WARMUP_TEXT, "caedral-m1")
        return time.perf_counter() - t0
