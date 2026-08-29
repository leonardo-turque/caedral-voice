"""Caedral G1 G2P pipeline: normalize -> tokenize -> per-token pronunciation.

Per token, in order (the OOV ladder from docs/research/TEXT-FRONTEND.md §3.3):
  1. lexicon hit            -> source='lexicon'
  2. morphological retry    -> source='morphology'   (-s/-ed/-ing/'s stripped, re-hit)
  3. letter-rule fallback   -> source='rule_fallback' (crude deterministic LTS)
  4. external backend flag  -> backend='espeak_ref' routes the WHOLE text to the
     dev-only GPL espeak-ng reference (never a Caedral dependency path).

Heteronyms: any token with >=2 distinct CMUdict pronunciations is returned
with ALL options plus ambiguous=True and a first-sense `selected` value.
Context disambiguation is explicitly OUT OF SCOPE for G1 (documented
limitation; see TEXT-FRONTEND.md §3.2 and experiment CV-G1-E0001-d).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cmudict_lexicon import arpabet_to_ipa, get_lexicon
from .normalize import _ascii_fold, normalize_detailed

# ---------------------------------------------------------------------------
# Technical vocabulary support (CV-G1-E0020-tn-expand)
# ---------------------------------------------------------------------------

# Letter names as IPA. Every glyph here is in the Kokoro vocab-mapped path,
# so each letter becomes its own speakable word group.
_LETTER_IPA = {
    "A": "eɪ", "B": "bˈiː", "C": "sˈiː", "D": "dˈiː", "E": "iː",
    "F": "ˈɛf", "G": "dʒˈiː", "H": "eɪtʃ", "I": "aɪ", "J": "dʒˈeɪ",
    "K": "kˈeɪ", "L": "ˈɛl", "M": "ˈɛm", "N": "ˈɛn", "O": "ˈoʊ",
    "P": "pˈiː", "Q": "kˈjuː", "R": "ˈɑːr", "S": "ˈɛs", "T": "tˈiː",
    "U": "jˈuː", "V": "vˈiː", "W": "ˈdʌbəljuː", "X": "ˈɛks",
    "Y": "wˈaɪ", "Z": "zˈiː",
}

# All-caps pure-alpha tokens in this length range are pronounced as
# individually spaced letter names (CPU -> C P U). TN letter-spaces tokens
# up to length 5; the G2P backstop covers slightly longer leftovers (HTTPS).
_ACRONYM_MIN_LEN = 2
_ACRONYM_MAX_LEN = 6

# Single uppercase letters that are ordinary English words keep their word
# pronunciation ("A" article, "I" pronoun); every other lone capital is a
# letter name (drive letters, list labels, "M" in the TN's "A M"/"P M").
# Exception to the exception: a capital inside a RUN of two or more single
# capitals ("V R A M", "A M") is always a letter name — an acronym letter,
# not the English word (E0023).
_SINGLE_LETTER_WORD_EXCEPTIONS = frozenset({"A", "I"})

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][A-Za-z]+)?|\d+", re.UNICODE)
_TOKEN_SPLIT_RE = re.compile(
    r"([^\W\d_]+(?:['’][A-Za-z]+)?|\d+|[^\s])", re.UNICODE
)

_PUNCT_SHORT_PAUSE = set(",;:")
_PUNCT_LONG_PAUSE = set(".!?")
_PAUSE_SHORT = "|"
_PAUSE_LONG = "‖"

# Function words whose CMUdict alternates are stress/reduction variants
# (THE -> ðə/ðiː, AS -> æz/əz, INTO stress shifts), not lexical heteronyms.
# Marking them ambiguous would flood the heteronym channel with noise;
# reduction choice belongs to prosody, not G2P selection. Documented design
# decision for CV-G1-E0001-d.
_FUNCTION_WORD_REDUCTION = {
    "the", "a", "an", "and", "or", "but", "that", "as", "to", "into", "at",
    "of", "for", "on", "from", "by", "is", "was", "are", "has", "have", "had",
    "be", "been", "am", "do", "does",
}


def tokenize(normalized_text: str) -> list[str]:
    """Split normalized text into word/number/punctuation tokens."""
    return [t for t in _TOKEN_SPLIT_RE.split(normalized_text) if t and not t.isspace()]


def _is_single_cap(tok: str) -> bool:
    return len(tok) == 1 and tok.isalpha() and tok.isupper()


def single_cap_run_members(tokens: list[str]) -> set[int]:
    """Indices of single-capital tokens inside a maximal run of two or more
    consecutive single capitals ('V R A M' -> {0, 1, 2, 3}).

    Inside such a run every letter — 'A' and 'I' included — is a letter
    name; a sentence-initial article ("A new era") sits in a run of one and
    keeps its word reading. Punctuation, digits and words break runs.
    """
    members: set[int] = set()
    i = 0
    while i < len(tokens):
        if _is_single_cap(tokens[i]):
            j = i + 1
            while j < len(tokens) and _is_single_cap(tokens[j]):
                j += 1
            if j - i >= 2:
                members.update(range(i, j))
            i = j
        else:
            i += 1
    return members


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WordPron:
    token: str
    options: list[str]          # all candidate IPA strings for this token
    selected: str               # IPA chosen for the output stream
    source: str                 # 'lexicon' | 'morphology' | 'rule_fallback' |
                                # 'g2pen' | 'digit_spell' | 'punct' |
                                # 'override' | 'letter_name' | 'acronym_letters'
    ambiguous: bool = False


@dataclass
class FrontendResult:
    text_original: str
    text_normalized: str
    words: list[WordPron] = field(default_factory=list)
    phoneme_string: str = ""        # space-separated IPA tokens incl. pause marks
    phoneme_tokens: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def phoneme_count(self) -> int:
        """Number of phone/stress tokens (pauses excluded)."""
        return sum(1 for t in self.phoneme_tokens if t not in (_PAUSE_SHORT, _PAUSE_LONG))

    @property
    def pause_count(self) -> int:
        return sum(1 for t in self.phoneme_tokens if t in (_PAUSE_SHORT, _PAUSE_LONG))


# ---------------------------------------------------------------------------
# Morphological fallback
# ---------------------------------------------------------------------------

_VOICELESS_END = set("fkpstʃθʃ")

_MORPH_SUFFIXES = ("'s", "s", "es", "ed", "ing")


def _split_suffix(word: str) -> tuple[str, str] | None:
    low = word.lower()
    if low.endswith("'s") or low.endswith("’s"):
        return word[:-2], "'s"
    for suf in ("es", "ed", "ing", "s"):
        if len(low) > len(suf) + 2 and low.endswith(suf):
            base = word[: -len(suf)]
            # doubled consonant: stopped -> stop, running -> run
            if suf == "ed" and len(base) > 2 and base[-1] == base[-2] and base[-1] not in "slz":
                base = base[:-1]
            if suf == "ing" and len(base) > 2 and base[-1] == base[-2] and base[-1] not in "slz":
                base = base[:-1]
            return base, suf
    return None


def _inflect_ipa(base_ipa: str, suffix: str) -> str:
    """Crude but standard suffix phonology on an IPA base."""
    last = base_ipa[-1] if base_ipa else ""
    if suffix in ("s", "'s"):
        if last in "szʃʒtʒ":
            return base_ipa + "ɪz"
        if last in "pkftθ":
            return base_ipa + "s"
        return base_ipa + "z"
    if suffix == "es":
        return base_ipa + "ɪz"
    if suffix == "ed":
        if last in "td":
            return base_ipa + "ɪd"
        if last in "pkfsʃtθ":
            return base_ipa + "t"
        return base_ipa + "d"
    if suffix == "ing":
        return base_ipa + "ɪŋ"
    return base_ipa


# ---------------------------------------------------------------------------
# Letter-to-sound rules (last resort; deliberately simple, always flagged)
# ---------------------------------------------------------------------------

_LTS_RULES: list[tuple[str, str]] = [
    ("tion", "ʃən"), ("sion", "ʒən"), ("ough", "ʌf"), ("augh", "æf"),
    ("ck", "k"), ("sh", "ʃ"), ("ch", "tʃ"), ("th", "θ"), ("ph", "f"),
    ("wh", "w"), ("qu", "kw"), ("ng", "ŋ"), ("nk", "ŋk"), ("gh", "ɡ"),
    ("ee", "iː"), ("ea", "iː"), ("oo", "uː"), ("ou", "aʊ"), ("ow", "oʊ"),
    ("ai", "eɪ"), ("ay", "eɪ"), ("oi", "ɔɪ"), ("oy", "ɔɪ"), ("ar", "ɑːr"),
    ("er", "ɚ"), ("ir", "ɝː"), ("or", "ɔːr"), ("ur", "ɝː"),
]

_SINGLE_MAP = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ",
    "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɑ", "p": "p", "q": "kw", "r": "r", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "i", "z": "z",
}


def letter_rule_ipa(word: str) -> str:
    """Deterministic crude letter-to-sound. Quality is knowingly poor; every
    use is flagged rule_fallback by the pipeline (loud step-down contract)."""
    w = word.lower().strip("'’")
    # drop likely-silent final e
    if len(w) > 3 and w.endswith("e") and not w.endswith(("le", "ee")):
        w = w[:-1]
    out: list[str] = []
    i = 0
    while i < len(w):
        for pat, ipa in _LTS_RULES:
            if w.startswith(pat, i):
                out.append(ipa)
                i += len(pat)
                break
        else:
            ch = w[i]
            if ch.isalpha():
                nxt = w[i + 1] if i + 1 < len(w) else ""
                if ch == "c" and nxt in "ei":
                    out.append("s")
                elif ch == "g" and nxt in "ei":
                    out.append("dʒ")
                else:
                    out.append(_SINGLE_MAP.get(ch, ""))
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class G2PPipeline:
    """Lexicon-first G2P over our own CMUdict-derived IPA.

    neural_fallback=True inserts g2p_en (Apache-2.0; NLTK-based LSTM LTS)
    between morphology and letter rules. It covered 30/30 of CV-G1-E0001's
    OOV list at ~5.9 ms/word [MEASURED, CV-G1-E0004]; lazy-imported so the
    base pipeline stays dependency-light.

    pronunciation_overrides maps lowercase tokens to explicit IPA strings;
    a hit bypasses lexicon, morphology and letter rules entirely (E0020 R11).
    """

    def __init__(self, lexicon=None, neural_fallback: bool = False,
                 pronunciation_overrides: dict[str, str] | None = None):
        self.lexicon = lexicon if lexicon is not None else get_lexicon()
        self.neural_fallback = neural_fallback
        self.pronunciation_overrides = dict(pronunciation_overrides or {})
        self._g2pen = None

    def _neural_ipa(self, token: str) -> str | None:
        if self._g2pen is None:
            from g2p_en import G2p  # lazy; Apache-2.0

            self._g2pen = G2p()
        phones = [p for p in self._g2pen(token.lower()) if p and p.strip()]
        good = [p for p in phones if re.match(r"^[A-Z]+[0-9]?$", p)]
        if not good:
            return None
        return arpabet_to_ipa(good)  # takes the phone list directly

    def resolve_word(self, token: str,
                     force_letter_name: bool = False) -> WordPron:
        """Resolve one token to its pronunciation.

        force_letter_name: the caller detected the token sits inside a run of
        single capitals (see single_cap_run_members); 'A'/'I' lose their
        ordinary-word reading there (E0023).
        """
        override = self.pronunciation_overrides.get(token.lower())
        if override:
            return WordPron(token, [override], override, "override")
        if (len(token) == 1 and token.isalpha()
                and token.isupper()
                and (force_letter_name
                     or token not in _SINGLE_LETTER_WORD_EXCEPTIONS)):
            ipa = _LETTER_IPA[token.upper()]
            return WordPron(token, [ipa], ipa, "letter_name")
        options = self.lexicon.ipa_options(token)
        if not options:
            # accent-fold retry: café -> cafe (deterministic NFKD strip)
            folded = _ascii_fold(token).lower()
            if folded != token.lower():
                options = self.lexicon.ipa_options(folded)
        if options:
            ambiguous = len(options) > 1 and token.lower() not in _FUNCTION_WORD_REDUCTION
            return WordPron(token, options, options[0], "lexicon", ambiguous=ambiguous)
        split = _split_suffix(token)
        if split:
            base, suffix = split
            base_opts = self.lexicon.ipa_options(base)
            if base_opts:
                inflected = [_inflect_ipa(b, suffix) for b in base_opts]
                ambiguous = len(inflected) > 1 and base.lower() not in _FUNCTION_WORD_REDUCTION
                return WordPron(token, inflected, inflected[0], "morphology",
                                ambiguous=ambiguous)
        if self.neural_fallback:
            neural_ipa = self._neural_ipa(token)
            if neural_ipa:
                return WordPron(token, [neural_ipa], neural_ipa, "g2pen",
                                ambiguous=False)
        fb = letter_rule_ipa(token)
        return WordPron(token, [fb], fb, "rule_fallback", ambiguous=False)

    def process(self, text: str) -> FrontendResult:
        norm_result = normalize_detailed(text)
        result = FrontendResult(text_original=text,
                                text_normalized=norm_result.text,
                                notes=norm_result.notes)
        tokens = tokenize(norm_result.text)
        run_members = single_cap_run_members(tokens)
        pieces: list[str] = []
        prev_pause = None
        for ti, tok in enumerate(tokens):
            if tok[0].isdigit():
                # numbers were expanded by TN; digit runs here are IDs — spell out
                digits = [("oh" if d == "0" else "zero one two three four five six seven eight nine".split()[int(d)])
                          for d in tok]
                ipa = " ".join(digits)
                result.words.append(WordPron(tok, [ipa], ipa, "digit_spell"))
                pieces.append(ipa)
                prev_pause = None
                continue
            if tok[0].isalpha() or tok in ("'", "’"):
                # E0020: all-caps acronyms are pronounced as individually
                # named letters. The letter-name IPA groups are joined with
                # SPACES inside one WordPron: the token survives as a single
                # word ('CPU'), while the embedded spaces reach the mapped
                # Kokoro stream as explicit word gaps so the letters no
                # longer merge into a run-on. An override on the whole
                # token wins over spelling it out.
                if (tok.isupper() and tok.isalpha()
                        and _ACRONYM_MIN_LEN <= len(tok) <= _ACRONYM_MAX_LEN
                        and tok.lower() not in self.pronunciation_overrides):
                    parts = [
                        self.pronunciation_overrides.get(ch.lower())
                        or _LETTER_IPA[ch]
                        for ch in tok
                    ]
                    ipa = " ".join(parts)
                    wp = WordPron(tok, [ipa], ipa, "acronym_letters")
                    result.words.append(wp)
                    pieces.append(ipa)
                    prev_pause = None
                    continue
                wp = self.resolve_word(
                    tok, force_letter_name=ti in run_members)
                result.words.append(wp)
                pieces.append(wp.selected)
                prev_pause = None
                continue
            # punctuation
            if tok in _PUNCT_SHORT_PAUSE:
                pause = _PAUSE_SHORT
            elif tok in _PUNCT_LONG_PAUSE:
                pause = _PAUSE_LONG
            else:
                continue  # quotes, parentheses, dashes folded by TN: drop silently
            if pause != prev_pause:
                pieces.append(pause)
                prev_pause = pause
        result.phoneme_tokens = _split_ipa_tokens(pieces)
        result.phoneme_string = " ".join(result.phoneme_tokens)
        while result.phoneme_tokens and result.phoneme_tokens[-1] in (_PAUSE_SHORT, _PAUSE_LONG):
            result.phoneme_tokens.pop()
        result.phoneme_string = " ".join(result.phoneme_tokens)
        return result


def _split_ipa_tokens(pieces: list[str]) -> list[str]:
    """Split per-word IPA strings into symbol tokens.

    A token is one character (each phone base letter, each stress mark);
    the length mark ː merges into the preceding vowel's token so a long
    vowel counts as one phone.
    """
    tokens: list[str] = []
    for piece in pieces:
        for ch in piece:
            if ch == "ː" and tokens:
                tokens[-1] += ch
            else:
                tokens.append(ch)
    return tokens


def g2p(text: str, lexicon=None) -> FrontendResult:
    """Convenience one-shot function."""
    return G2PPipeline(lexicon).process(text)
