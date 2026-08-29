"""CMUdict lexicon loader + ARPAbet -> simple IPA conversion.

Data: cmudict-0.7b (134,373 entries), stored at <repo>/data/lexicons/cmudict-0.7b.
License (verified from the file header this session, 2026-08-22): Copyright
(C) 1993-2015 Carnegie Mellon University; BSD-style permissive — use "for any
research or commercial purpose is completely unrestricted" per the CMU Sphinx
distribution; redistribution requires the copyright notice, which we keep in
data/lexicons/cmudict-LICENSE and must ship with any redistribution.

Variant convention in 0.7b: the unmarked entry is one pronunciation and
WORD(1), WORD(2)... are alternates. lookup() returns ALL pronunciations so
heteronym handling upstream can surface ambiguity instead of hiding it.

The IPA mapping is our own standard ARPAbet table following the rules
published by menelik3/cmudict-ipa (AH->^/schwa stress-conditioned, length
marks AA->A:, ER->long schwa-R, IY->i:, UW->u:, AO->O:) — re-derived here;
no third-party fork is depended on.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CMUDICT_PATH = _REPO / "data" / "lexicons" / "cmudict-0.7b"

# ---------------------------------------------------------------------------
# ARPAbet -> IPA
# ---------------------------------------------------------------------------

ARPA_TO_IPA: dict[str, str] = {
    # Vowels (with length marks where the published convention adds them)
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝː", "EY": "eɪ", "IH": "ɪ", "IY": "iː", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "uː",
    # Consonants
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "r", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Stress-conditioned overrides (menelik3 rule: AH -> schwa when unstressed,
# ER -> r-colored schwa when unstressed).
_UNSTRESSED_OVERRIDE = {"AH": "ə", "ER": "ɚ"}

_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW",
    "OY", "UH", "UW",
}

_STRESS_MARKS = {"1": "ˈ", "2": "ˌ"}

_ENTRY_RE = re.compile(r"^([^\s(]+)(?:\((\d)\))?\s+(.+)$")


def arpabet_to_ipa(pron: list[str] | tuple[str, ...]) -> str:
    """Convert one ARPAbet pronunciation to an IPA string.

    Stress marks (ˈ primary / ˌ secondary) are emitted immediately before
    the stressed vowel's IPA symbol.
    """
    out: list[str] = []
    for symbol in pron:
        base = symbol.rstrip("012")
        stress = symbol[len(base):] if len(symbol) > len(base) else ""
        if base not in ARPA_TO_IPA:
            raise ValueError(f"unknown ARPAbet symbol: {symbol!r}")
        ipa = ARPA_TO_IPA[base]
        if base in _VOWELS:
            if stress == "0" and base in _UNSTRESSED_OVERRIDE:
                ipa = _UNSTRESSED_OVERRIDE[base]
            mark = _STRESS_MARKS.get(stress, "")
            out.append(mark + ipa)
        else:
            out.append(ipa)
    return "".join(out)


class CmuDict:
    """In-memory CMUdict with all variant pronunciations retained."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_CMUDICT_PATH
        self._entries: dict[str, list[list[str]]] = {}
        self._load()

    def _load(self) -> None:
        entries: dict[str, list[tuple[int, list[str]]]] = {}
        with open(self.path, encoding="latin-1") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(";;;"):
                    continue
                m = _ENTRY_RE.match(line)
                if not m:
                    continue
                headword, variant, phones = m.groups()
                key = headword.lower()
                pron = phones.split()
                idx = int(variant) if variant else 0
                entries.setdefault(key, []).append((idx, pron))
        self._entries = {
            k: [p for _, p in sorted(vs, key=lambda t: t[0])]
            for k, vs in entries.items()
        }

    # -- lookups -----------------------------------------------------------

    def lookup(self, word: str) -> list[list[str]] | None:
        """All ARPAbet pronunciations for `word` (case-insensitive), or None."""
        return self._entries.get(word.lower())

    def contains(self, word: str) -> bool:
        return word.lower() in self._entries

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def __len__(self) -> int:
        return len(self._entries)

    def distinct_pronunciations(self, word: str) -> list[list[str]]:
        """Pronunciations deduplicated at the ARPAbet level."""
        prons = self.lookup(word)
        if not prons:
            return []
        seen: set[tuple[str, ...]] = set()
        out: list[list[str]] = []
        for p in prons:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    # -- IPA convenience ----------------------------------------------------

    def ipa_options(self, word: str) -> list[str]:
        """IPA string per distinct pronunciation."""
        return [arpabet_to_ipa(p) for p in self.distinct_pronunciations(word)]

    def is_ambiguous(self, word: str) -> bool:
        """True when CMUdict lists >=2 distinct pronunciations (heteronym signal)."""
        return len(self.distinct_pronunciations(word)) > 1


@lru_cache(maxsize=None)
def get_lexicon(path: str | None = None) -> CmuDict:
    """Process-wide singleton loader."""
    return CmuDict(path)
