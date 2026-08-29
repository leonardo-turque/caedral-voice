"""Caedral text frontend (Phase E): TN + G2P, lexicon-first, GPL-free at runtime.

Public surface:
    TextFrontend(backend='lexicon')     -> deterministic CMUdict + rules chain
    TextFrontend(backend='espeak_ref')  -> DEV-ONLY GPLv3 espeak-ng reference

The 'lexicon' backend is the only shippable path; 'espeak_ref' exists purely
for dev-time comparison and is quarantined in backends_espeak (imported
lazily, never at package import time).
"""
from __future__ import annotations

import json
from pathlib import Path

from .cmudict_lexicon import CmuDict, arpabet_to_ipa, get_lexicon
from .g2p import FrontendResult, G2PPipeline, WordPron, g2p, tokenize
from .normalize import normalize, normalize_detailed

# E0020 R11: exact-match (case-insensitive) pronunciation overrides. Every
# *.json in this directory maps a surface word to an explicit IPA string that
# bypasses lexicon + G2P entirely.
_PRONUNCIATION_DIR = Path(__file__).resolve().parents[2] / "data" / "pronunciation"


def load_pronunciation_overrides(directory: Path | str | None = None) -> dict[str, str]:
    """Load {word: ipa} overrides from every *.json in `directory`.

    Missing directory or unreadable files degrade gracefully to {} so the
    frontend never fails to construct because of optional data.
    """
    d = Path(directory) if directory is not None else _PRONUNCIATION_DIR
    out: dict[str, str] = {}
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.update({str(k).lower(): str(v) for k, v in data.items()})
    return out


class TextFrontend:
    """Caedral text frontend facade.

    backend='lexicon': normalize -> tokenize -> CMUdict lookup ->
                       morphological fallback -> letter-rule fallback.
                       Heteronyms surface ALL pronunciations + ambiguous flag.
    backend='espeak_ref': normalize -> dev-only espeak-ng phonemization
                       (GPLv3; dev/reference use only, never shippable).

    pronunciation_dir=None uses repo data/pronunciation/*.json (E0020);
    pass an explicit directory (or a path that does not exist) to override.
    """

    _VALID_BACKENDS = ("lexicon", "lexicon+g2pen", "espeak_ref")

    def __init__(self, backend: str = "lexicon", lexicon=None,
                 pronunciation_dir: str | Path | None = None):
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {self._VALID_BACKENDS}"
            )
        self.backend = backend
        neural = backend == "lexicon+g2pen"
        self.pronunciation_overrides = load_pronunciation_overrides(pronunciation_dir)
        self._pipeline = G2PPipeline(
            lexicon, neural_fallback=neural,
            pronunciation_overrides=self.pronunciation_overrides,
        )
        self._espeak = None

    def process(self, text: str) -> FrontendResult:
        if self.backend in ("lexicon", "lexicon+g2pen"):
            return self._pipeline.process(text)
        # Dev-only reference path.
        from .backends_espeak import get_espeak_ref_backend  # lazy, quarantined

        if self._espeak is None:
            self._espeak = get_espeak_ref_backend()
        norm = normalize_detailed(text)
        ipa = self._espeak.phonemize([norm.text])[0]
        result = FrontendResult(
            text_original=text,
            text_normalized=norm.text,
            notes=norm.notes + ["backend:espeak_ref (GPLv3, dev-only)"],
        )
        result.phoneme_string = ipa
        result.phoneme_tokens = ipa.split()
        return result

    def phonemize(self, text: str) -> str:
        return self.process(text).phoneme_string


__all__ = [
    "TextFrontend",
    "FrontendResult",
    "WordPron",
    "G2PPipeline",
    "CmuDict",
    "get_lexicon",
    "arpabet_to_ipa",
    "g2p",
    "tokenize",
    "normalize",
    "normalize_detailed",
    "load_pronunciation_overrides",
]
