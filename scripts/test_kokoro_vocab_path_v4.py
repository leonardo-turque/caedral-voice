"""V4: explicit word-boundary spaces via per-word token counting + punctuation
pauses preserved."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kokoro_onnx import Kokoro
from caedral_voice.text import TextFrontend
import classify_preview as cp

k = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
           str(REPO / "models/kokoro/voices-v1.0.bin"))
VOCAB = set(k.tokenizer.vocab.keys())
tf = TextFrontend("lexicon+g2pen")

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "abbrev": "The CPU supports AVX2 instructions but not AVX-512.",
}


def map_token(tok: str, pending: str) -> tuple[list[str], str]:
    seg: list[str] = []
    ok = True
    for ch in tok:
        if ch in VOCAB:
            seg.append(pending + ch)
            pending = ""
        elif ch in ("ˈ", "ˌ"):
            pending += ch
        elif ch == "ɝ":
            seg.append(pending + "ɜ")
            if "r" in VOCAB:
                seg.append("r")
            pending = ""
        else:
            ok = False
    return (seg if ok else []), pending


def main() -> None:
    results = []
    for voice in ("af_heart", "am_michael"):
        for sname, text in SENTENCES.items():
            res = tf.process(text)
            # token budget per word = number of chars in its selected IPA
            word_budgets = [len(w.selected) for w in res.words]
            out_stream: list[str] = []
            skipped: list[str] = []
            wi = 0            # word index
            consumed = 0      # tokens consumed of current word
            pending = ""
            for tok in res.phoneme_tokens:
                if tok in ("|", "‖"):
                    out_stream.append(" " if tok == "|" else ". ")
                    continue
                seg, pending = map_token(tok, pending)
                if not seg:
                    skipped.append(tok)
                    continue
                out_stream.extend(seg)
                consumed += len(seg)
                if wi < len(word_budgets) and consumed >= word_budgets[wi]:
                    out_stream.append(" ")   # word boundary
                    wi += 1
                    consumed = 0
            phoneme_str = "".join(out_stream)

            audio_b, sr_b = k.create(text, voice=voice, speed=1.0, lang="en-us")
            outdir = REPO / "artifacts" / "pre-arch-enhanced-base" / "vocab-mapped-v4"
            outdir.mkdir(parents=True, exist_ok=True)
            import soundfile as sf2
            fb = outdir / f"baseline-{voice}-{sname}.wav"
            sf2.write(fb, audio_b, sr_b)

            try:
                audio_e, sr_e = k.create(phoneme_str, voice=voice, speed=1.0,
                                         lang="en-us", is_phonemes=True)
            except Exception as exc:
                print(f"{voice}/{sname}: FAILED {exc}")
                continue
            fe = outdir / f"mappedv4-{voice}-{sname}.wav"
            sf2.write(fe, audio_e, sr_e)

            mb = cp.classify_file(str(fb), text)
            me = cp.classify_file(str(fe), text)
            row = {"voice": voice, "sentence": sname,
                   "baseline": {"class": mb.get("class"), "wer": mb.get("min_wer"),
                                "dnsmos": mb.get("dnsmos_ovrl")},
                   "mapped_v4": {"class": me.get("class"), "wer": me.get("min_wer"),
                                 "dnsmos": me.get("dnsmos_ovrl")}}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    json.dump(results, io.open(outdir.parent / "vocab-mapped-v4-results.json", "w",
                               encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
