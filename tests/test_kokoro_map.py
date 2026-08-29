"""Unit tests for the Caedral->Kokoro char-level vocab mapping.

Pure logic tests on a synthetic vocab; no model files needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.text.kokoro_map import (  # noqa: E402
    LONG_PAUSE_TOKEN,
    SHORT_PAUSE_TOKEN,
    caedral_tokens_to_kokoro,
)

# Tiny synthetic vocab standing in for the Kokoro embedded one.
VOCAB = frozenset("həlˈoʊ ðɪsfɜrst.ɛŋaɪːʃ")


def test_in_vocab_tokens_pass_through() -> None:
    out, skipped = caedral_tokens_to_kokoro(["h", "ə", "l", "o", "ʊ"], VOCAB)
    assert out == "həloʊ"
    assert skipped == []


def test_short_pause_becomes_word_gap() -> None:
    out, _ = caedral_tokens_to_kokoro(["h", "ə", SHORT_PAUSE_TOKEN, "l"], VOCAB)
    assert out == "hə l"


def test_long_pause_becomes_sentence_break() -> None:
    out, _ = caedral_tokens_to_kokoro(["h", LONG_PAUSE_TOKEN, "l"], VOCAB)
    assert out == "h. l"


def test_multichar_token_split_into_chars() -> None:
    # 'aɪ' is not itself a vocab entry but both chars are
    out, skipped = caedral_tokens_to_kokoro(["aɪ"], VOCAB)
    assert out == "aɪ"
    assert skipped == []


def test_r_colored_vowel_alias_with_stress_reattach() -> None:
    # ɝ is aliased to ɜr; a preceding stress mark re-attaches to the alias.
    out, skipped = caedral_tokens_to_kokoro(["f", "ˈɝ", "st"], VOCAB)
    assert out == "fˈɜrst"
    assert skipped == []


def test_unknown_char_reports_token_as_skipped() -> None:
    out, skipped = caedral_tokens_to_kokoro(["ʃ", "æ"], VOCAB)  # 'æ' unknown
    assert out == "ʃ"
    assert skipped == ["æ"]


def test_skipped_is_sorted_dedup() -> None:
    _, skipped = caedral_tokens_to_kokoro(["æ", "χ", "æ"], VOCAB)
    assert skipped == sorted(set(skipped)) == ["æ", "χ"]


def test_mapping_is_deterministic() -> None:
    tokens = ["h", "ˈɝ", SHORT_PAUSE_TOKEN, "iː", "æ"]
    assert caedral_tokens_to_kokoro(tokens, VOCAB) == \
        caedral_tokens_to_kokoro(tokens, VOCAB)
