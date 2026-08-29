"""Char-level mapping from Caedral IPA to the Kokoro ONNX phoneme vocabulary.

Promoted verbatim (logic-identical) from the winning production path
`scripts/test_kokoro_vocab_path_v2.py` (experiment CV-G1-E0016: the mapped
v2 route beat raw-text baseline on WER/DNSMOS and is the founder-approved
kokoro_onnx + Caedral-frontend production path).

Caedral's TextFrontend emits IPA tokens that are a superset of what the
Kokoro v1.0 ONNX export carries in its embedded vocab. This module filters
each token down to characters the model actually knows, applying a small
alias table for known mismatches:

    ɝ  -> ɜr   r-colored stressed vowel approximated as ɜ + r
    |  -> " "  Caedral short-pause marker becomes a word gap
    ‖  -> ". " long pause becomes a sentence break
    ˈ ˌ      stress marks are dropped but remembered, then re-attached
               to the next aliased vowel (Kokoro HAS ˈ/ˌ in-vocab, so they
               normally pass through untouched; the re-attach only matters
               when a stress mark directly precedes an out-of-vocab char).

Anything still unmapped is collected in the returned ``skipped`` list so
callers can log/telemeter it; nothing silently disappears from the string
without being reported.
"""
from __future__ import annotations

from collections.abc import Iterable

# Short pause inside a sentence -> word gap; long pause -> sentence break.
SHORT_PAUSE_TOKEN = "|"
LONG_PAUSE_TOKEN = "‖"
SHORT_PAUSE_CHAR = " "
LONG_PAUSE_CHAR = ". "

# Known symbol-level mismatches between the Caedral frontend and Kokoro vocab.
KOKORO_ALIAS: dict[str, str] = {
    "ɝ": "ɜr",       # r-colored stressed vowel -> ɜ + r approximation
    "ˈ": "",         # primary stress handled via prev_stress below
    LONG_PAUSE_TOKEN: LONG_PAUSE_CHAR,
    SHORT_PAUSE_TOKEN: SHORT_PAUSE_CHAR,
}

_STRESS_MARKS = ("ˈ", "ˌ")


def caedral_tokens_to_kokoro(
    phoneme_tokens: Iterable[str], vocab: set[str] | frozenset[str]
) -> tuple[str, list[str]]:
    """Map Caedral IPA tokens onto the Kokoro character vocabulary.

    Returns ``(phoneme_string, skipped)`` where ``skipped`` is the sorted,
    de-duplicated list of Caedral tokens that could not be fully mapped.
    """
    out_chars: list[str] = []
    skipped: list[str] = []
    prev_stress = ""
    for tok in phoneme_tokens:
        if tok == SHORT_PAUSE_TOKEN:
            out_chars.append(SHORT_PAUSE_CHAR)
            continue
        if tok == LONG_PAUSE_TOKEN:
            out_chars.append(LONG_PAUSE_CHAR)
            continue
        if tok in vocab:
            out_chars.append(tok)
            continue
        # token may be multi-char like 'iː' — split into chars
        ok = True
        for ch in tok:
            if ch in vocab:
                out_chars.append(ch)
                prev_stress = ""
            elif ch in _STRESS_MARKS:
                prev_stress = ch
            elif ch in KOKORO_ALIAS:
                mapped = KOKORO_ALIAS[ch]
                if prev_stress:
                    mapped = prev_stress + mapped
                    prev_stress = ""
                for c2 in mapped:
                    if c2 in vocab or c2 == " ":
                        out_chars.append(c2)
                    else:
                        ok = False
            else:
                ok = False
        if not ok:
            skipped.append(tok)
    return "".join(out_chars), sorted(set(skipped))


def kokoro_vocab_set(tokenizer_vocab: dict) -> frozenset[str]:
    """The synthesizable character set of a kokoro_onnx Tokenizer."""
    return frozenset(tokenizer_vocab.keys())
