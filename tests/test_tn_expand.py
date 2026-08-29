"""CV-G1-E0020-tn-expand: technical-vocabulary TN + G2P upgrades.

Covers: acronym letter-spacing (CPU -> C P U), alnum tech compounds
(AVX2 / AVX-512 / GPT-5), paired model numbers (RTX 5090), product-context
version decimals (Python 3.13), slash versions (HTTP/2), ISO dates,
bare domains, storage/frequency units, CamelCase splitting, letter-name
pronunciation with per-letter word groups in the G2P stream, and the
data/pronunciation/*.json override hook. Every expansion is deterministic:
the same input always yields byte-identical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.text import (  # noqa: E402
    TextFrontend,
    load_pronunciation_overrides,
    normalize,
)


# ---------------------------------------------------------------------------
# Normalized-text expectations (TN layer)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # alnum compounds: letter prefix stays grouped at TN level; the G2P
        # layer pronounces it as individually spaced letters.
        ("AVX2", "AVX two"),
        ("AVX-512", "AVX five hundred twelve"),
        ("GPT-5", "GPT five"),
        ("COVID-19 changed everything.", "COVID nineteen changed everything."),
        # acronyms letter-spaced at TN level
        ("CPU", "C P U"),
        ("GPU", "G P U"),
        ("The CPU supports AVX2 instructions but not AVX-512.",
         "The C P U supports AVX two instructions but not AVX five hundred twelve."),
        # paired model-number reading after an all-caps token
        ("RTX 5090", "R T X fifty ninety"),
        ("GTX 1080", "G T X ten eighty"),
        # product-context version decimal (uppercase prev token)
        ("Python 3.13", "Python three point thirteen"),
        # bare decimal stays digit-wise on the fraction
        ("3.13", "three point one three"),
        ("99.9%", "ninety-nine point nine percent"),
        ("$29.99", "twenty-nine dollars and ninety-nine cents"),
        # slash versions
        ("HTTP/2", "HTTP version two"),
        ("HTTP/1.1", "HTTP version one point one"),
        # ISO date (not phone-style digit soup)
        ("2026-08-23", "August twenty-third twenty twenty-six"),
        # out-of-range ISO components fall back to the digit-string class
        ("9999-99-99", "nine nine nine nine nine nine nine nine"),
        # bare domains (URLs/emails keep their own rules)
        ("caedral.com", "caedral dot com"),
        ("support@caedral.com", "support at caedral dot com"),
        # storage / frequency units
        ("10GB", "ten gigabytes"),
        ("3.5GHz", "three point five gigahertz"),
        # E0023: SPACED unit forms reach the unit rule too — R5/R7 must not
        # eat "GHz"/"GB" before _convert_units sees the number+unit pair.
        ("3.5 GHz", "three point five gigahertz"),
        ("10 GB VRAM", "ten gigabytes VRAM"),
        # CamelCase split
        ("Use CaedralVoice today.", "Use Caedral Voice today."),
        # E0023: caps tokens carrying a pronunciation override stay whole so
        # the G2P layer can match the override key on the surface form.
        ("CUDA 12.8 support", "CUDA twelve point eight support"),
    ],
)
def test_tn_expansion_expected_text(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", [
    "AVX2", "AVX-512", "CPU", "RTX 5090", "Python 3.13", "$29.99",
    "99.9%", "2026-08-23", "caedral.com", "HTTP/2",
    "The CPU supports AVX2 instructions but not AVX-512.",
])
def test_determinism_same_input_same_output(raw):
    assert normalize(raw) == normalize(raw)


def test_existing_classes_not_regressed():
    # spot guards: E0020 must not eat the classes the base suite pins
    assert normalize("I have 3 apples.") == "I have three apples."
    assert normalize("Version 2.11.3 shipped") == \
        "Version two point eleven point three shipped"
    assert normalize("Call 555-0142 now.") == "Call five five five oh one four two now."
    assert normalize("between 9:30 AM and 5:00 PM") == \
        "between nine thirty A M and five o'clock P M"
    assert " minus " not in normalize("AVX2 and AVX-512 differ.")
    assert normalize("In 1947, prices rose.") == "In nineteen forty-seven, prices rose."


# ---------------------------------------------------------------------------
# G2P layer: per-letter word groups + pronunciation overrides
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frontend() -> TextFrontend:
    return TextFrontend("lexicon")


def test_acronym_letters_are_separate_word_groups(frontend):
    # Standalone caps are letter-spaced by TN, so each letter reaches G2P
    # as its own word (source='letter_name') -> separate word groups.
    r = frontend.process("The CPU works.")
    letters = [w for w in r.words if w.source == "letter_name"]
    assert [w.token for w in letters] == ["C", "P", "U"]
    assert letters[0].selected == "sˈiː"
    assert letters[2].selected == "jˈuː"


def test_grouped_prefix_keeps_token_and_letters(frontend):
    # TN keeps "AVX" grouped (contract with the base suite's token test);
    # G2P still pronounces it letter by letter with word gaps.
    r = frontend.process("AVX2 and int8.")
    toks = [w.token.lower() for w in r.words]
    assert "avx" in toks and "two" in toks and "int" in toks and "eight" in toks
    avx = [w for w in r.words if w.token == "AVX"]
    assert avx and avx[0].source == "acronym_letters"
    assert avx[0].selected == "eɪ vˈiː ˈɛks"


def test_acronym_phoneme_stream_carries_word_gaps(frontend):
    # grouped prefixes (from compounds/long caps) carry space-separated
    # letter groups inside one WordPron -> explicit word gaps downstream
    r = frontend.process("GPT-5 and HTTP/2.")
    acronyms = [w for w in r.words if w.source == "acronym_letters"]
    assert [w.token for w in acronyms] == ["GPT", "HTTP"]
    for w in acronyms:
        parts = w.selected.split(" ")
        assert all(parts), f"empty gap inside {w.token!r}: {w.selected!r}"


def test_single_capital_is_letter_name_but_a_and_i_keep_words(frontend):
    r = frontend.process("Drive C is ready.")
    drive_c = [w for w in r.words if w.token == "C"]
    assert drive_c and drive_c[0].source == "letter_name"
    assert drive_c[0].selected == "sˈiː"
    # article/pronoun singles stay lexical words
    r2 = frontend.process("I can help.")
    i_tok = [w for w in r2.words if w.token == "I"]
    assert i_tok and i_tok[0].source != "letter_name"


def test_pronunciation_override_bypasses_lexicon(frontend):
    r = frontend.process("Hello caedral.")
    hits = [w for w in r.words if w.token.lower() == "caedral"]
    assert hits and hits[0].source == "override"
    assert hits[0].selected == "kəˈdɹɑːl"


def test_pronunciation_override_case_insensitive(frontend):
    r = frontend.process("Hello Caedral Voice.")
    hits = [w for w in r.words if w.token.lower() == "caedral"]
    assert hits and hits[0].source == "override"


def test_caps_brand_token_uses_override_not_letters(frontend):
    # an override on an all-caps token wins over spelling it out
    r = frontend.process("CAEDRAL ships.")
    hits = [w for w in r.words if w.token.lower() == "caedral"]
    assert hits and hits[0].source == "override"


def test_default_pronunciation_file_ships_caedral():
    overrides = load_pronunciation_overrides()
    assert overrides.get("caedral") == "kəˈdɹɑːl"


# ---------------------------------------------------------------------------
# E0023: letter-run reading + technical override file
# ---------------------------------------------------------------------------

def test_letter_run_reads_a_as_letter_name(frontend):
    # 'RAM' -> TN 'R A M'; the lone A must be the letter name, not the article
    r = frontend.process("RAM and ROM chips.")
    a_tok = [w for w in r.words if w.token == "A"]
    assert a_tok and a_tok[0].source == "letter_name"
    assert a_tok[0].selected == "eɪ"


def test_sentence_initial_article_a_still_a_word(frontend):
    # run of ONE capital keeps the ordinary-word reading
    r = frontend.process("A new era begins.")
    a_tok = [w for w in r.words if w.token == "A"]
    assert a_tok and a_tok[0].source != "letter_name"
    assert a_tok[0].selected == "ə"


def test_technical_overrides_fire_on_caps_tokens(frontend):
    r = frontend.process("The GPU has 24 GB VRAM and CUDA 12.8.")
    hits = {w.token.upper(): w for w in r.words}
    assert hits["VRAM"].source == "override"
    assert hits["VRAM"].selected == "vˈiːɹæm"
    assert hits["CUDA"].source == "override"
    assert hits["CUDA"].selected == "kˈuːdə"
    # non-overridden acronyms still letter-space end to end
    gpu = [w for w in r.words if w.token == "G"]
    assert gpu and gpu[0].source == "letter_name"


def test_technical_override_file_loads():
    overrides = load_pronunciation_overrides()
    assert overrides.get("cuda") == "kˈuːdə"
    assert overrides.get("nvme") == "ˌɛn vˈiː ˌɛm ˈiː"
    assert overrides.get("ghz") == "dʒˈɪɡəhˌɜːts"


def test_frontend_pipeline_determinism(frontend):
    text = "The CPU supports AVX2 instructions but not AVX-512."
    a = frontend.process(text)
    b = frontend.process(text)
    assert a.phoneme_string == b.phoneme_string
    assert [w.selected for w in a.words] == [w.selected for w in b.words]
