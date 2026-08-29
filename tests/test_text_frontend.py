"""Unit tests for the Phase E text frontend (caedral_voice/text).

Covers: rule-based TN (numbers/currency/dates/times/percent/phones/abbrev/
URL/email), CMUdict lexicon loading + ARPAbet->IPA, the G2P pipeline
(lexicon hit, morphology, letter-rule fallback, heteronym flagging, pauses),
and the GPL quarantine contract (importing caedral_voice.text must NOT pull
in espeak/phonemizer; TextFrontend('espeak_ref') is dev-only and lazily
imported).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.text import (  # noqa: E402
    CmuDict,
    TextFrontend,
    arpabet_to_ipa,
    normalize,
    normalize_detailed,
    tokenize,
)


# ---------------------------------------------------------------------------
# TN: numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("I have 3 apples.", "I have three apples."),
        ("48,213 people", "forty-eight thousand two hundred thirteen people"),
        ("up to 16,384 simultaneous connections",
         "up to sixteen thousand three hundred eighty-four simultaneous connections"),
        ("12.5 percent of cases", "twelve point five percent of cases"),
        ("It rose 2.1% today", "It rose two point one percent today"),
        ("within 30 days", "within thirty days"),
    ],
)
def test_numbers(raw, expected):
    assert normalize(raw) == expected


def test_ordinals():
    assert normalize("the 21st guest") == "the twenty-first guest"
    assert normalize("February 29th") == "February twenty-ninth"


def test_ordinal_words():
    from caedral_voice.text.normalize import ordinal_to_words
    assert ordinal_to_words(1) == "first"
    assert ordinal_to_words(2) == "second"
    assert ordinal_to_words(3) == "third"
    assert ordinal_to_words(11) == "eleventh"
    assert ordinal_to_words(12) == "twelfth"
    assert ordinal_to_words(20) == "twentieth"
    assert ordinal_to_words(29) == "twenty-ninth"
    assert ordinal_to_words(47) == "forty-seventh"


@pytest.mark.parametrize(
    "year,expected",
    [
        (1947, "nineteen forty-seven"),
        (1996, "nineteen ninety-six"),
        (2026, "twenty twenty-six"),
        (2005, "two thousand five"),
        (2000, "two thousand"),
        (1900, "nineteen hundred"),
    ],
)
def test_year_words(year, expected):
    from caedral_voice.text.normalize import year_to_words
    assert year_to_words(year) == expected


def test_bare_year_in_context():
    assert normalize("In 1947, prices rose.") == "In nineteen forty-seven, prices rose."
    # A bare 4-digit number without year context stays a cardinal.
    assert normalize("The code is 1947.") == "The code is one thousand nine hundred forty-seven."


def test_versions():
    assert normalize("Version 2.11.3 shipped") == "Version two point eleven point three shipped"


def test_mixed_alphanumeric_identifiers_split():
    # letter-digit identifiers must not glue into pseudo-words
    out = normalize("The model runs int8 math.")
    assert "int eight" in out
    out2 = normalize("AVX2 and AVX-512 differ.")
    assert " minus " not in out2  # tight hyphen is NOT a subtraction operator
    assert "five hundred twelve" in out2
    fe = TextFrontend("lexicon")
    toks = [w.token.lower() for w in fe.process("AVX2 and int8.").words]
    assert "avxtwo" not in toks and "int8" not in toks
    assert "avx" in toks and "two" in toks and "int" in toks and "eight" in toks


def test_spaced_math_operators_verbalized():
    assert normalize("2 + 2 = 4") == "two plus two equals four"
    assert normalize("x - y matters") == "x minus y matters"


# ---------------------------------------------------------------------------
# TN: currency / dates / times / phones
# ---------------------------------------------------------------------------

def test_currency_usd():
    out = normalize("The invoice totals $1,247.83, due soon.")
    assert "one thousand two hundred forty-seven dollars and eighty-three cents" in out


def test_currency_whole_and_euros():
    assert normalize("It costs €2,400 a month.") == "It costs two thousand four hundred euros a month."
    assert normalize("around €150") == "around one hundred fifty euros"
    assert normalize("£1 exactly") == "one pound exactly"
    assert normalize("£2") == "two pounds"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("She was born on February 29th, 1996.",
         "She was born on February twenty-ninth, nineteen ninety-six."),
        ("on March 3, 2026 we", "on March third, twenty twenty-six we"),
        ("by Q3 2026 inflation had cooled",
         "by the third quarter of twenty twenty-six inflation had cooled"),
        ("Q1 2027 results", "the first quarter of twenty twenty-seven results"),
    ],
)
def test_dates_and_quarters(raw, expected):
    assert normalize(raw) == expected


def test_times():
    assert normalize("between 9:30 AM and 5:00 PM") == (
        "between nine thirty A M and five o'clock P M"
    )
    assert normalize("at 9:05 am sharp") == "at nine oh five A M sharp"


def test_phone_sequences_digitwise():
    out = normalize("Call 555-0142 now.")
    assert out == "Call five five five oh one four two now."
    out2 = normalize("Dial +44 20 7946 0958 if you're abroad.")
    assert out2.startswith("Dial plus four four two oh seven nine four six")


# ---------------------------------------------------------------------------
# TN: abbreviations, URL/email, misc robustness
# ---------------------------------------------------------------------------

def test_abbreviations():
    assert normalize("Dr. Smith met Mr. Jones.") == "Doctor Smith met Mister Jones."
    assert normalize("cats vs. dogs") == "cats versus dogs"
    assert normalize("approx. 5 miles") == "approximately five miles"


def test_st_ambiguous_contextual():
    assert normalize("St. Peter arrived.").startswith("Saint Peter")
    assert normalize("on Main St. downtown").startswith("on Main Street")


def test_url_email_graceful():
    notes = normalize_detailed(
        "Visit https://example.com/some/path?q=1 or email someone@example.org."
    ).notes
    out = normalize(
        "Visit https://example.com/some/path?q=1 or email someone@example.org."
    )
    assert "example dot com" in out
    assert "someone at example dot org" in out
    assert any(n.startswith("url_path_or_query_dropped") for n in notes)


def test_emoji_and_control_chars_dropped_with_note():
    res = normalize_detailed("\U0001F600 voices \U0001F3B5 here")
    assert res.text.strip() == "voices here"
    assert any("dropped" in n for n in res.notes)


def test_empty_and_whitespace_safe():
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize(None) == ""


def test_determinism():
    sample = "In 1947, Dr. Who paid $1,247.83 for 350g of flour by 9:30 AM."
    assert normalize(sample) == normalize(sample)


# ---------------------------------------------------------------------------
# Lexicon + ARPAbet -> IPA
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lexicon() -> CmuDict:
    return CmuDict()


def test_lexicon_loads_and_sizes(lexicon):
    assert len(lexicon) > 120000
    assert lexicon.contains("hello")
    assert not lexicon.contains("zzzqqqxyzzy")


def test_lookup_returns_all_variants(lexicon):
    prons = lexicon.lookup("read")
    assert prons is not None and len(prons) >= 2
    assert ["R", "EH1", "D"] in prons
    assert ["R", "IY1", "D"] in prons


def test_arpabet_to_ipa_mapping():
    assert arpabet_to_ipa(["HH", "AH0", "L", "OW1"]) == "həlˈoʊ"
    assert arpabet_to_ipa(["R", "EH1", "D"]) == "rˈɛd"
    assert arpabet_to_ipa(["R", "IY1", "D"]) == "rˈiːd"
    assert arpabet_to_ipa(["T", "IH1", "R"]) == "tˈɪr".replace("ir", "ɪr")  # t + ɪ + r
    # unstressed AH -> schwa, stressed ER -> r-colored long vowel
    assert arpabet_to_ipa(["ER0"]) == "ɚ"
    assert arpabet_to_ipa(["ER1"]) == "ˈɝː"
    assert arpabet_to_ipa(["AA1"]) == "ˈɑː"


def test_heteronym_detection_via_lexicon(lexicon):
    for word in ("read", "lead", "live", "wind", "tear", "bow", "close",
                 "object", "produce", "dove", "wound", "desert"):
        assert lexicon.is_ambiguous(word), f"{word} should be ambiguous"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frontend() -> TextFrontend:
    return TextFrontend("lexicon")


def test_pipeline_end_to_end(frontend):
    r = frontend.process("The wind was too strong to wind the clock.")
    assert r.text_normalized.startswith("The wind was too strong")
    assert r.phoneme_count > 10
    assert len(r.words) == 9
    winds = [w for w in r.words if w.token.lower() == "wind"]
    assert len(winds) == 2 and all(w.ambiguous for w in winds)


def test_heteronyms_flagged_with_all_options(frontend):
    r = frontend.process("Please read it.")
    reads = [w for w in r.words if w.token.lower() == "read"]
    assert reads and reads[0].ambiguous
    assert set(reads[0].options) == {"rˈɛd", "rˈiːd"}
    # selected = first sense; context disambiguation explicitly out of scope
    assert reads[0].selected == "rˈɛd"


def test_oov_morphology_then_letter_fallback(frontend):
    r = frontend.process("The flurbsion flurbed quietly.")
    sources = {w.token: w.source for w in r.words}
    # -ed forms fall back to morphology or rules, never crash
    assert all(s in ("morphology", "rule_fallback", "lexicon")
               for s in sources.values())
    flagged = [w for w in r.words if w.source == "rule_fallback"]
    assert flagged, "unknown stems should be loudly flagged rule_fallback"


def test_pause_tokens_from_punctuation(frontend):
    r = frontend.process("Wait, stop! Okay?")
    assert "|" in r.phoneme_tokens
    assert "‖" in r.phoneme_tokens
    assert r.pause_count >= 2
    assert not r.phoneme_string.endswith("|")


def test_tokenize_basic():
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        TextFrontend("bogus_backend")


# ---------------------------------------------------------------------------
# GPL quarantine contract
# ---------------------------------------------------------------------------

def test_package_import_does_not_pull_gpl_modules():
    """Importing caedral_voice.text must never import espeak tooling.

    This is the CI guard for the TEXT-FRONTEND.md licensing decision: the
    GPLv3 reference backend stays behind a lazy import in backends_espeak.
    """
    for mod in list(sys.modules):
        if mod.split(".")[0] in ("phonemizer", "espeakng_loader", "backends_espeak"):
            del sys.modules[mod]
    import caedral_voice.text as ct

    assert "phonemizer" not in sys.modules
    assert "espeakng_loader" not in sys.modules
    assert hasattr(ct, "TextFrontend")


def test_dev_only_module_documents_gpl_status():
    import caedral_voice.text.backends_espeak as be

    assert "GPL" in be.__doc__
