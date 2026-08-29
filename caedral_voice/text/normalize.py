"""Caedral rule-based text normalization (TN).

Pure-Python, deterministic, zero third-party dependencies (no inflect, no
num2words/LGPL). Covers the semiotic classes Caedral G1 needs: cardinals,
ordinals, decimals, thousands separators, currency ($/EUR/GBP), dates
("February 29th, 1996" / bare years / "Q3 2026"), clock times (9:30 AM),
percentages, phone-like sequences, common abbreviations, versions, small
units, URL/email graceful handling.

Contract for anything unrecognized: PASS THROUGH unchanged and record a note
(NeMo VERBATIM-class behavior). Every transformation is a pure function of
the input string; calling normalize() twice is a no-op on its own output.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Number -> words (own implementation; avoids num2words LGPL entirely)
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = ["", " thousand", " million", " billion", " trillion"]


def _under_thousand(n: int) -> str:
    parts: list[str] = []
    if n >= 100:
        parts.append(_ONES[n // 100] + " hundred")
        n %= 100
        if n == 0:
            return " ".join(parts)
    if n >= 20:
        tens = _TENS[n // 10]
        if n % 10:
            tens += "-" + _ONES[n % 10]
        parts.append(tens)
    elif n > 0:
        parts.append(_ONES[n])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Integer to American English words, no 'and' (e.g. 1247 -> one thousand two hundred forty-seven)."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    scale_idx = 0
    chunks: list[str] = []
    while n > 0:
        cur = n % 1000
        if cur:
            head = _under_thousand(cur) + _SCALES[scale_idx]
            chunks.insert(0, head)
        n //= 1000
        scale_idx += 1
    return " ".join(chunks)


_IRREGULAR_ORDINALS = {1: "first", 2: "second", 3: "third", 5: "fifth",
                       8: "eighth", 9: "ninth", 12: "twelfth"}
_TEN_TO_TIETH = {"twenty": "twentieth", "thirty": "thirtieth",
                 "forty": "fortieth", "fifty": "fiftieth", "sixty": "sixtieth",
                 "seventy": "seventieth", "eighty": "eightieth",
                 "ninety": "ninetieth"}


def ordinal_to_words(n: int) -> str:
    """Integer to ordinal words (29 -> twenty-ninth). Ordinalizes only the
    final word of the cardinal form."""
    card = number_to_words(n)
    words = card.split(" ")
    final = words[-1].split("-")
    tail = final[-1]
    if n % 100 in (11, 12, 13):
        repl = {11: "eleventh", 12: "twelfth", 13: "thirteenth"}[n % 100]
        final[-1] = repl
    elif tail in _TEN_TO_TIETH:
        final[-1] = _TEN_TO_TIETH[tail]
    elif tail in _ONES[1:]:
        idx = _ONES.index(tail)
        final[-1] = _IRREGULAR_ORDINALS.get(idx, tail + "th")
    else:
        # hundred / thousand finals
        final[-1] = tail + "th"
    words[-1] = "-".join(final)
    return " ".join(words)


def year_to_words(y: int) -> str:
    """Year reading: 1947 -> nineteen forty-seven, 2026 -> twenty twenty-six,
    2005 -> two thousand five, 1900 -> nineteen hundred."""
    if y % 100 == 0 and y % 1000 != 0:
        return f"{number_to_words(y // 100)} hundred"
    if 1100 <= y <= 1999:
        century = number_to_words(y // 100)
        rest = y % 100
        if rest == 0:
            return f"{century} hundred"
        if rest < 10:
            return f"{century} oh {_ONES[rest]}"
        return f"{century} {_under_thousand(rest)}"
    if 2000 <= y <= 2009:
        return f"two thousand {_ONES[y % 10]}" if y != 2000 else "two thousand"
    if 2010 <= y <= 2099:
        rest = y % 100
        if rest < 10:
            return f"twenty oh {_ONES[rest]}"
        return f"twenty {_under_thousand(rest)}"
    return number_to_words(y)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}
_CURRENCY_RE = re.compile(r"(?P<sym>[$€£])(?P<mult>[KMB])?(?P<num>\d[\d,]*(?:\.\d+)?)")
_MULT_WORD = {"K": "thousand", "M": "million", "B": "billion"}

_UNITS = {
    "kg": "kilograms", "g": "grams", "mg": "milligrams",
    "ml": "milliliters", "l": "liters",
    "km": "kilometers", "cm": "centimeters", "mm": "millimeters", "m": "meters",
    "km/h": "kilometers per hour", "mph": "miles per hour",
    "lb": "pounds", "lbs": "pounds", "oz": "ounces",
    # E0020: storage / frequency units (uppercase, so they never collide with
    # the case-sensitive lowercase single-letter units above).
    "GB": "gigabytes", "MB": "megabytes", "KB": "kilobytes",
    "TB": "terabytes", "GHz": "gigahertz", "MHz": "megahertz",
}
_UNIT_RE = re.compile(
    r"\b(?P<num>\d[\d,]*(?:\.\d+)?)\s?"
    r"(?P<unit>km/h|mph|GHz|MHz|kg|mg|ml|km|cm|mm|TB|GB|MB|KB|lbs|lb|oz|l|g|m)\b"
)

_ABBREVIATIONS = {
    "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss",
    "Prof.": "Professor", "Rev.": "Reverend", "Hon.": "Honorable",
    "Gen.": "General", "Sen.": "Senator", "Rep.": "Representative",
    "Capt.": "Captain", "Sgt.": "Sergeant", "Lt.": "Lieutenant",
    "Col.": "Colonel", "Maj.": "Major", "Mt.": "Mount", "Ft.": "Fort",
    "Jr.": "Junior", "vs.": "versus", "approx.": "approximately",
    "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is",
    "No.": "number", "Dept.": "Department", "Univ.": "University",
    "Inc.": "Incorporated", "Ltd.": "Limited", "Co.": "Company",
    "Corp.": "Corporation", "Ave.": "Avenue", "Blvd.": "Boulevard",
    "Rd.": "Road", "Jan.": "January", "Feb.": "February", "Mar.": "March",
    "Apr.": "April", "Jun.": "June", "Jul.": "July", "Aug.": "August",
    "Sep.": "September", "Sept.": "September", "Oct.": "October",
    "Nov.": "November", "Dec.": "December",
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_YEAR_CONTEXT_WORDS = {
    "in", "since", "by", "around", "during", "circa", "until", "before",
    "after", "between", "of", "from",
}

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# CV-G1-E0020-tn-expand: bare-domain handling. Runs AFTER full URLs/emails
# are consumed, so only naked hosts like "caedral.com" reach this. The TLD
# allowlist keeps prose abbreviations ("e.g.", "vs.") and filenames
# ("normalize.py") from being misread as domains.
_TLDS = (
    "com|org|net|edu|gov|io|ai|dev|app|co|uk|de|fr|jp|us|info|xyz|me|tv"
)
_BARE_DOMAIN_RE = re.compile(
    rf"\b(?:[A-Za-z0-9-]+\.)+(?:{_TLDS})\b"
)

_TIME_RE = re.compile(
    r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<meridiem>[AaPp]\.?[Mm]\.?)?\b"
)

_QUARTER_RE = re.compile(
    r"\bQ(?P<qnum>[1-4])\s+(?P<year>\d{4})\b"
)

_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,\s*(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)

_PHONE_SPAN_RE = re.compile(r"(?<![\w./])\+?\d(?:[\s-]?\d){6,14}(?![\w.])")
_PHONE_HYPHENATED_RE = re.compile(r"(?<![\w.])(?:\d{1,4}-){1,4}\d{1,4}(?![\w.])")

_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)+\b")
_PERCENT_RE = re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)\s?%")
_ORDINAL_SUFFIX_RE = re.compile(r"\b(?P<num>\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_PUNCT_SHORT_PAUSE = set(",;:")
_PUNCT_LONG_PAUSE = set(".!?")


@dataclass
class NormalizationResult:
    text: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_number(num_str: str) -> float | int:
    cleaned = num_str.replace(",", "")
    if "." in cleaned:
        return float(cleaned)
    return int(cleaned)


def _cardinal_words(num_str: str) -> str:
    value = _parse_number(num_str)
    if isinstance(value, int):
        return number_to_words(value)
    whole, frac = num_str.split(".")
    whole_words = number_to_words(int(whole)) if whole else "zero"
    frac_words = " ".join(_ONES[int(d)] for d in frac)
    return f"{whole_words} point {frac_words}"


def _money_words(sym: str, num_str: str) -> str:
    unit_sing, unit_plur, cent_sing, cent_plur = _CURRENCIES[sym]
    value = float(num_str.replace(",", ""))
    whole = int(value)
    cents = round((value - whole) * 100)
    parts: list[str] = []
    if whole or cents == 0:
        parts.append(f"{number_to_words(whole)} {unit_sing if whole == 1 else unit_plur}")
    if cents:
        cword = cent_sing if cents == 1 else cent_plur
        if parts:
            parts.append(f"and {number_to_words(cents)} {cword}")
        else:
            parts.append(f"{number_to_words(cents)} {cword}")
    return " ".join(parts)


def _time_words(hour: str, minute: str, meridiem: str | None) -> str:
    h = int(hour)
    m = minute
    hwords = number_to_words(h)
    if m == "00":
        out = f"{hwords} o'clock"
    elif m.startswith("0"):
        out = f"{hwords} oh {_ONES[int(m[1])]}"
    else:
        out = f"{hwords} {_under_thousand(int(m))}"
    if meridiem:
        letter1 = "A" if meridiem.strip(".").upper().startswith("A") else "P"
        out += f" {letter1} M"
    return out


def _phone_digits_words(span: str) -> str:
    words: list[str] = []
    for ch in span:
        if ch == "+":
            words.append("plus")
        elif ch.isdigit():
            words.append("oh" if ch == "0" else _ONES[int(ch)])
        elif ch in "- ":
            continue
    return " ".join(words)


def _is_phone_like(span: str) -> bool:
    digits = sum(c.isdigit() for c in span)
    groups = [g for g in re.split(r"[- ]+", span.strip()) if g]
    if not 7 <= digits <= 15:
        return False
    if len(groups) < 2:
        return False
    # A decimal like 1234567.89 would be caught by the generic number path
    # first; anything reaching here with a dot is not phone material.
    return "." not in span


def _ascii_fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _strip_emoji_and_controls(text: str, notes: list[str]) -> str:
    out: list[str] = []
    dropped = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in "\n\t":
            dropped += 1
            continue
        if ord(ch) >= 0x1F000 or (0x2600 <= ord(ch) <= 0x27BF):
            dropped += 1
            continue
        out.append(ch)
    if dropped:
        notes.append(f"dropped_{dropped}_emoji_or_control_chars")
    return "".join(out)


def _fold_punctuation(text: str) -> str:
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": ",", "…": "...", "«": '"',
        "»": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _expand_urls(text: str, notes: list[str]) -> str:
    def repl(match: re.Match) -> str:
        raw = match.group(0)
        host = re.sub(r"^https?://|^www\.", "", raw, flags=re.IGNORECASE)
        host = host.split("/")[0].split("?")[0].split("#")[0]
        labels = [lbl for lbl in host.split(".") if lbl]
        notes.append(f"url_path_or_query_dropped:{raw[:48]}")
        return " dot ".join(labels)

    return _URL_RE.sub(repl, text)


def _expand_emails(text: str, notes: list[str]) -> str:
    def repl(match: re.Match) -> str:
        local, domain = match.group(0).split("@", 1)
        labels = " dot ".join(lbl for lbl in domain.split(".") if lbl)
        return f"{local} at {labels}"

    return _EMAIL_RE.sub(repl, text)


def _expand_bare_domains(text: str, notes: list[str]) -> str:
    """caedral.com -> caedral dot com (naked hosts; URLs/emails already gone).

    TLD allowlist prevents prose like "e.g." or "vs." and filenames like
    "normalize.py" from being misread as domains.
    """
    def repl(match: re.Match) -> str:
        host = match.group(0)
        labels = [lbl for lbl in host.lower().split(".") if lbl]
        notes.append(f"bare_domain_expanded:{host}")
        return " dot ".join(labels)

    return _BARE_DOMAIN_RE.sub(repl, text)


def _expand_abbreviations(text: str) -> str:
    tokens = text.split(" ")
    out: list[str] = []
    for i, tok in enumerate(tokens):
        stripped = tok.rstrip(",;:!?)\"'")
        suffix = tok[len(stripped):]
        prefix = ""
        core = stripped
        if core.startswith("("):
            prefix, core = "(", core[1:]
        expansion = None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if core == "St.":
            expansion = "Saint" if nxt[:1].isupper() else "Street"
        elif core in _ABBREVIATIONS:
            expansion = _ABBREVIATIONS[core]
        out.append(prefix + (expansion + suffix if expansion else tok))
    return " ".join(out)


def _convert_times(text: str) -> str:
    def repl(m: re.Match) -> str:
        mer = m.group("meridiem")
        return _time_words(m.group("hour"), m.group("minute"), mer)

    return _TIME_RE.sub(repl, text)


def _convert_quarters(text: str) -> str:
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth"}

    def repl(m: re.Match) -> str:
        q = ordinals[int(m.group("qnum"))]
        yw = year_to_words(int(m.group("year")))
        return f"the {q} quarter of {yw}"

    return _QUARTER_RE.sub(repl, text)


def _convert_dates(text: str) -> str:
    def repl(m: re.Match) -> str:
        month = m.group("month").capitalize()
        day = ordinal_to_words(int(m.group("day")))
        year = m.group("year")
        out = f"{month} {day}"
        if year:
            out += f", {year_to_words(int(year))}"
        return out

    return _MONTH_DAY_YEAR_RE.sub(repl, text)


def _convert_currency(text: str) -> str:
    def repl(m: re.Match) -> str:
        sym = m.group("sym")
        num = m.group("num")
        mult = m.group("mult")
        if mult:
            unit_sing, unit_plur = _CURRENCIES[sym][0], _CURRENCIES[sym][1]
            value = float(num.replace(",", ""))
            w = f"{_cardinal_words(num)} {_MULT_WORD[mult]}"
            return f"{w} {unit_sing if value == 1 else unit_plur}"
        return _money_words(sym, num)

    return _CURRENCY_RE.sub(repl, text)


def _convert_percentages(text: str) -> str:
    return _PERCENT_RE.sub(lambda m: f"{_cardinal_words(m.group('num'))} percent", text)


def _convert_ordinals(text: str) -> str:
    return _ORDINAL_SUFFIX_RE.sub(
        lambda m: ordinal_to_words(int(m.group("num"))), text
    )


def _convert_versions(text: str) -> str:
    def repl(m: re.Match) -> str:
        comps = m.group(0).split(".")
        return " point ".join(number_to_words(int(c)) for c in comps)

    return _VERSION_RE.sub(repl, text)


def _convert_units(text: str) -> str:
    def repl(m: re.Match) -> str:
        unit = m.group("unit")
        word = _UNITS[unit]
        return f"{_cardinal_words(m.group('num'))} {word}"

    return _UNIT_RE.sub(repl, text)


def _convert_years(text: str) -> str:
    out = []
    pos = 0
    for m in re.finditer(r"\b(1[1-9]\d{2}|20[0-9]\d)\b", text):
        start, end = m.span()
        before = text[:start].rstrip()
        prev_word = re.findall(r"[A-Za-z]+", before.split(" ")[-1] if before else "")
        prev = prev_word[-1].lower() if prev_word else ""
        after_ok = True  # context word alone decides; keep deterministic
        if prev in _YEAR_CONTEXT_WORDS and after_ok:
            out.append(text[pos:start])
            out.append(year_to_words(int(m.group(1))))
            pos = end
    out.append(text[pos:])
    return "".join(out)


def _convert_cardinal_numbers(text: str) -> str:
    def repl(m: re.Match) -> str:
        return _cardinal_words(m.group(0))

    return _NUMBER_TOKEN_RE.sub(repl, text)


def _convert_bare_math_symbols(text: str) -> str:
    # Only SPACED operators are verbalized; tight hyphens belong to compound
    # identifiers (AVX-512) and phone groups, which are handled elsewhere.
    text = re.sub(r"\s*=\s*", " equals ", text)
    text = re.sub(r"\s+\+\s+", " plus ", text)
    text = re.sub(r"\s+-\s+", " minus ", text)
    text = re.sub(r"\^", "", text)
    return text


def _split_alphanumeric(text: str) -> str:
    """Separate letter-digit boundaries so identifiers read cleanly:
    AVX2 -> AVX 2, int8 -> int 8 (then normal cardinal rules apply).
    Runs after ordinal/date/time/currency classes are already expanded."""
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return text


# ---------------------------------------------------------------------------
# Technical vocabulary expansion (CV-G1-E0020-tn-expand)
#
# Deterministic, unit-testable token-level rewrites for the technical strings
# that the base TN pipeline reads as run-on garbage: acronyms (CPU), alnum
# compounds (AVX2, GPT-5), slash versions (HTTP/2), model numbers (RTX 5090),
# ISO dates, product-context version decimals and CamelCase identifiers.
#
# Design contract with the existing suite (test_text_frontend.py):
#   * letter-digit compounds keep their letter prefix GROUPED ("AVX two",
#     not "A V X two") so tokens 'avx'/'int' survive to the G2P layer;
#   * tight hyphens never become "minus";
#   * "AVX-512" keeps the contiguous phrase "five hundred twelve".
# The G2P layer then pronounces any surviving all-caps token as individually
# word-gapped letter names (see g2p.py).
# ---------------------------------------------------------------------------

_MONTH_NAMES = {v: k.capitalize() for k, v in _MONTHS.items()}

_ISO_DATE_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")

_SLASH_VERSION_RE = re.compile(r"^([A-Za-z]{2,})/(\d{1,4}(?:\.\d{1,3})*)$")

_HYPHEN_TECH_RE = re.compile(r"^([A-Za-z]{2,})-(\d{1,12})$")

_DECIMAL_ONLY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d+$")

_CAPS_ACRONYM_RE = re.compile(r"^[A-Z]{2,5}$")

_FOUR_DIGIT_RE = re.compile(r"^\d{4}$")

_CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)

# Edge punctuation stripped only to find a token's CORE; preserved verbatim
# around the replacement. Currency symbols deliberately excluded.
_EDGE_PUNCT = ",.;:!?)\"'“”"

# Meridiem markers belong to the clock-time class (_convert_times matches
# "9:30 AM" as one span); spacing them here would break that match.
_MERIDIEM_TOKENS = frozenset({"AM", "PM"})


# ---------------------------------------------------------------------------
# Pronunciation-override awareness (CV-G1-E0023-tn-integration)
#
# data/pronunciation/*.json map surface words to explicit IPA strings that
# the G2P layer applies verbatim (caedral_voice.text). A token whose
# lowercase form carries an override must reach G2P INTACT: letter-spacing
# it (R5) or Camel-splitting it (R7) would destroy the very surface form the
# override is keyed on ("CUDA" -> "C U D A" can never hit {"cuda": ...}).
# Keys are loaded once, lazily; a missing/unreadable directory degrades to
# an empty set so normalization never depends on optional data.
# ---------------------------------------------------------------------------

_OVERRIDE_KEYS: frozenset[str] | None = None


def _pronunciation_override_keys() -> frozenset[str]:
    global _OVERRIDE_KEYS
    if _OVERRIDE_KEYS is None:
        d = Path(__file__).resolve().parents[2] / "data" / "pronunciation"
        keys: set[str] = set()
        if d.is_dir():
            for path in sorted(d.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict):
                    keys.update(str(k).lower() for k in data)
        _OVERRIDE_KEYS = frozenset(keys)
    return _OVERRIDE_KEYS


def _split_edges(tok: str) -> tuple[str, str, str]:
    """'("CPU",' -> ('("', 'CPU', ',')"""
    start = 0
    end = len(tok)
    while start < end and tok[start] in _EDGE_PUNCT + "(":
        start += 1
    while end > start and tok[end - 1] in _EDGE_PUNCT:
        end -= 1
    return tok[:start], tok[start:end], tok[end:]


def _convert_iso_dates(text: str) -> str:
    """2026-08-23 -> 'August twenty-third twenty twenty-six'.

    Must run BEFORE phone-span detection, which otherwise reads ISO dates as
    digit-group phone numbers. Out-of-range components pass through unchanged.
    """
    def repl(m: re.Match) -> str:
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{_MONTH_NAMES[mo]} {ordinal_to_words(d)} {year_to_words(y)}"

    return _ISO_DATE_RE.sub(repl, text)


def _pair_number_words(digits: str) -> str:
    """Paired-tens model-number reading: '5090' -> fifty ninety,
    '1080' -> ten eighty, '5005' -> fifty oh five, '5000' -> fifty hundred."""
    hi = int(digits[:2])
    lo = int(digits[2:])
    parts = [_under_thousand(hi)]
    if lo == 0:
        parts.append("hundred")
    elif lo < 10:
        parts.append(f"oh {_ONES[lo]}")
    else:
        parts.append(_under_thousand(lo))
    return " ".join(parts)


def _expand_technical_tokens(text: str) -> str:
    """One deterministic left-to-right pass over whitespace tokens."""
    raw = text.split(" ")
    out: list[str] = []
    prev_core = ""
    i = 0
    while i < len(raw):
        tok = raw[i]
        nxt_core = ""
        if i + 1 < len(raw):
            nxt_core = _split_edges(raw[i + 1])[1]
        prefix, core, suffix = _split_edges(tok)
        replaced: str | None = None

        # R3 slash-version: HTTP/2 -> HTTP version two
        m = _SLASH_VERSION_RE.match(core)
        if m:
            comps = " point ".join(
                number_to_words(int(c)) for c in m.group(2).split(".")
            )
            replaced = f"{m.group(1)} version {comps}"

        # R2 hyphen tech compound: AVX-512 -> AVX five hundred twelve,
        # GPT-5 -> GPT five, COVID-19 -> COVID nineteen. Letter prefix stays
        # grouped (G2P letter-names it); tight hyphen is NOT a minus sign.
        if replaced is None:
            m = _HYPHEN_TECH_RE.match(core)
            if m:
                replaced = f"{m.group(1)} {number_to_words(int(m.group(2)))}"

        # R4 product-context version decimal: Python 3.13 ->
        # three point thirteen (major + cardinal minor). Bare decimals fall
        # through to the generic digit-wise rule later.
        if replaced is None and _DECIMAL_ONLY_RE.match(core):
            if prev_core[-1:].isalpha() and prev_core[:1].isupper():
                whole, frac = core.split(".", 1)
                replaced = (
                    f"{number_to_words(int(whole.replace(',', '')))} point "
                    f"{number_to_words(int(frac))}"
                )

        # R5+R6 all-caps acronym letter-spacing (+ paired model number):
        # CPU -> C P U; RTX 5090 -> R T X fifty ninety. Skipped for known
        # units ("GHz"/"GB" must survive to the unit rule, which also matches
        # the SPACED "3.5 GHz"/"10 GB" forms) and for tokens carrying a
        # pronunciation override (the whole surface form is the override key).
        if (replaced is None and core not in _MERIDIEM_TOKENS
                and core not in _UNITS
                and core.lower() not in _pronunciation_override_keys()
                and _CAPS_ACRONYM_RE.match(core)):
            spelled = " ".join(core)
            if _FOUR_DIGIT_RE.match(nxt_core):
                i += 1  # consume the model number into this rewrite
                spelled += " " + _pair_number_words(nxt_core)
            replaced = spelled

        # R7 CamelCase split: CaedralVoice -> Caedral Voice. Tokens with
        # digits are excluded — units like "3.5GHz"/"10MB" belong to the
        # unit rule, not word-splitting. Same two exemptions as R5.
        if (replaced is None and len(core) >= 2 and core.isalpha()
                and core not in _UNITS
                and core.lower() not in _pronunciation_override_keys()):
            split = _CAMEL_BOUNDARY_RE.sub(" ", core)
            if split != core:
                replaced = split

        if replaced is None:
            out.append(tok)
        else:
            out.append(prefix + replaced + suffix)
        prev_core = core
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_detailed(text: str) -> NormalizationResult:
    """Normalize `text`; returns normalized string plus audit notes."""
    notes: list[str] = []
    if text is None:
        return NormalizationResult("", ["null_input"])
    work = unicodedata.normalize("NFC", text)
    work = _fold_punctuation(work)
    work = _strip_emoji_and_controls(work, notes)
    work = _expand_emails(work, notes)
    work = _expand_urls(work, notes)
    work = _expand_bare_domains(work, notes)          # E0020 R8
    work = _convert_iso_dates(work)                   # E0020 R1
    work = _expand_technical_tokens(work)             # E0020 R2-R7
    work = _expand_abbreviations(work)
    work = _convert_versions(work)
    work = _convert_times(work)
    work = _convert_quarters(work)
    work = _convert_dates(work)
    work = _convert_currency(work)
    work = _convert_percentages(work)
    work = _convert_ordinals(work)
    work = _split_alphanumeric(work)
    work = _convert_years(work)
    work = _convert_units(work)
    work = _convert_phone_spans(work, notes)
    work = _convert_bare_math_symbols(work)
    work = _convert_cardinal_numbers(work)
    work = re.sub(r"[ \t]{2,}", " ", work).strip()
    return NormalizationResult(work, notes)


def _convert_phone_spans(text: str, notes: list[str]) -> str:
    spans: list[tuple[int, int]] = []

    def collect_hyphenated(m: re.Match) -> str:
        span = m.group(0)
        if _is_phone_like(span):
            spans.append(m.span())
            return "\x00" * len(span)
        return span

    staged = _PHONE_HYPHENATED_RE.sub(collect_hyphenated, text)

    def collect_general(m: re.Match) -> str:
        span = m.group(0)
        if _is_phone_like(span):
            spans.append(m.span())
            return "\x00" * len(span)
        return span

    staged = _PHONE_SPAN_RE.sub(collect_general, staged)

    result: list[str] = []
    last = 0
    for start, end in sorted(spans):
        result.append(text[last:start])
        result.append(_phone_digits_words(text[start:end]))
        last = end
    result.append(text[last:])
    return "".join(result)


def normalize(text: str) -> str:
    """Deterministic rule-based normalization. Unrecognized constructs pass through."""
    return normalize_detailed(text).text
