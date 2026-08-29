"""V2: char-level filtering + alias table (ɝ->ɜ, | -> space, etc.)."""
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

ALIAS = {
    "ɝ": "ɜː" if False else "ɜr",   # r-colored stressed vowel -> ɜ + r approximation
    "ˈ": "",      # primary stress handled below (Kokoro HAS ˈ at 156)
    "‖": ". ",    # long pause -> sentence break
    "|": " ",
}

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "abbrev": "The CPU supports AVX2 instructions but not AVX-512.",
}


def caedral_to_kokoro(text: str) -> tuple[str, list[str]]:
    res = tf.process(text)
    out_chars = []
    skipped = []
    prev_stress = ""
    for tok in res.phoneme_tokens:
        if tok == "|":
            out_chars.append(" ")          # short pause -> word gap
            continue
        if tok == "‖":
            out_chars.append(". ")
            continue
        if tok in VOCAB:
            out_chars.append(tok)
            continue
        # token may be multi-char like 'iː' — split into chars
        ok = True
        for ch in tok:
            if ch in VOCAB:
                out_chars.append(ch)
                prev_stress = ""
            elif ch == "ˈ":
                prev_stress = "ˈ"
            elif ch == "ˌ":
                prev_stress = "ˌ"
            elif ch in ALIAS:
                mapped = ALIAS[ch]
                if prev_stress:
                    mapped = prev_stress + mapped
                    prev_stress = ""
                for c2 in mapped:
                    if c2 in VOCAB or c2 == " ":
                        out_chars.append(c2)
                    else:
                        ok = False
            else:
                ok = False
        if not ok:
            skipped.append(tok)
    return "".join(out_chars), sorted(set(skipped))


results = []
for voice in ("af_heart", "am_michael"):
    for sname, text in SENTENCES.items():
        phoneme_str, skipped = caedral_to_kokoro(text)

        audio_b, sr_b = k.create(text, voice=voice, speed=1.0, lang="en-us")
        from pathlib import Path as _P
        outdir = REPO / "artifacts" / "pre-arch-enhanced-base" / "vocab-mapped-v2"
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
        fe = outdir / f"mapped-{voice}-{sname}.wav"
        sf2.write(fe, audio_e, sr_e)

        mb = cp.classify_file(str(fb), text)
        me = cp.classify_file(str(fe), text)
        row = {"voice": voice, "sentence": sname,
               "skipped": sorted(set(skipped)),
               "baseline": {"class": mb.get("class"), "wer": mb.get("min_wer"),
                            "dnsmos": mb.get("dnsmos_ovrl")},
               "mapped_v2": {"class": me.get("class"), "wer": me.get("min_wer"),
                             "dnsmos": me.get("dnsmos_ovrl")}}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

json.dump(results, io.open(outdir.parent / "vocab-mapped-v2-results.json", "w",
                           encoding="utf-8"), indent=2, ensure_ascii=False)
