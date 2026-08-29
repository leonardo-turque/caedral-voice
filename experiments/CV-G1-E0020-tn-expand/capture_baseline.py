"""CV-G1-E0020-tn-expand: capture PRE-upgrade (old) frontend behavior.

Run BEFORE any code edits. Stores old normalized texts and old mapped
Kokoro phoneme streams for the validation sentence, so the E2E comparison
can replay the true old behavior after the upgrade lands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.text import TextFrontend, normalize  # noqa: E402

SENTENCE = "The CPU supports AVX2 instructions but not AVX-512."

CASES = [
    "AVX2", "AVX-512", "CPU", "GPU", "GPT-5", "RTX 5090",
    "Python 3.13", "$29.99", "99.9%", "10GB", "3.5GHz", "3.13",
    "2026-08-23", "caedral.com", "support@caedral.com", "HTTP/2",
]

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "abbrev": SENTENCE,
}


def map_stream(res, vocab):
    """v4-style vocab mapping: word budgets from len(selected), word-gap
    spaces between words, '|' -> ' ', '||' -> '. '."""
    pending = ""

    def map_token(tok):
        nonlocal pending
        seg = []
        ok = True
        for ch in tok:
            if ch in vocab:
                seg.append(pending + ch)
                pending = ""
            elif ch in ("ˈ", "ˌ"):
                pending += ch
            elif ch == "ɝ":
                seg.append(pending + "ɜ")
                if "r" in vocab:
                    seg.append("r")
                pending = ""
            else:
                ok = False
        return (seg if ok else []), pending

    out = []
    wi = 0
    consumed = 0
    for tok in res.phoneme_tokens:
        if tok == "|":
            out.append(" ")
            continue
        if tok == "‖":
            out.append(". ")
            continue
        seg, pending = map_token(tok)
        if not seg:
            continue
        out.extend(seg)
        consumed += len(seg)
        if wi < len(res.words) and consumed >= len(res.words[wi].selected):
            out.append(" ")
            wi += 1
            consumed = 0
    return "".join(out)


def main() -> None:
    from kokoro_onnx import Kokoro

    k = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
               str(REPO / "models/kokoro/voices-v1.0.bin"))
    vocab = set(k.tokenizer.vocab.keys())
    tf = TextFrontend("lexicon+g2pen")

    cap = {"cases": {}, "sentences": {}}
    for c in CASES:
        cap["cases"][c] = normalize(c)

    for sname, text in SENTENCES.items():
        res = tf.process(text)
        cap["sentences"][sname] = {
            "text": text,
            "normalized": res.text_normalized,
            "mapped_stream": map_stream(res, vocab),
            "phoneme_string": res.phoneme_string,
            "words": [w.selected for w in res.words],
        }

    outdir = REPO / "experiments" / "CV-G1-E0020-tn-expand"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "baseline_capture.json").write_text(
        json.dumps(cap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cap["cases"], indent=1))
    print("abbrev normalized:", cap["sentences"]["abbrev"]["normalized"])
    print("abbrev stream:", repr(cap["sentences"]["abbrev"]["mapped_stream"]))


if __name__ == "__main__":
    main()
