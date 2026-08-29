"""Test: Caedral frontend -> filter to Kokoro vocab -> continuous string -> is_phonemes."""
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

results = []
for voice in ("af_heart", "am_michael"):
    for sname, text in SENTENCES.items():
        res = tf.process(text)
        toks = []
        skipped = []
        for tok in res.phoneme_tokens:
            if tok in VOCAB:
                toks.append(tok)
            elif tok in ("|",):
                toks.append(" ")   # pause -> space (Kokoro handles phrase gaps)
            else:
                skipped.append(tok)
        phoneme_str = "".join(toks)

        # baseline stock render for reference
        audio_b, sr_b = k.create(text, voice=voice, speed=1.0, lang="en-us")
        from pathlib import Path as _P
        outdir = REPO / "artifacts" / "pre-arch-enhanced-base" / "vocab-mapped"
        outdir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf2
        fb = outdir / f"baseline-{voice}-{sname}.wav"
        sf2.write(fb, audio_b, sr_b)

        try:
            audio_e, sr_e = k.create(phoneme_str, voice=voice, speed=1.0,
                                     lang="en-us", is_phonemes=True)
        except Exception as exc:
            print(f"{voice}/{sname}: PHONEME PATH FAILED {exc}")
            continue
        fe = outdir / f"mapped-{voice}-{sname}.wav"
        sf2.write(fe, audio_e, sr_e)

        mb = cp.classify_file(str(fb), text)
        me = cp.classify_file(str(fe), text)
        row = {"voice": voice, "sentence": sname,
               "skipped_symbols": sorted(set(skipped)),
               "baseline": {"class": mb.get("class"), "wer": mb.get("min_wer"),
                            "dnsmos": mb.get("dnsmos_ovrl")},
               "mapped_enhanced": {"class": me.get("class"), "wer": me.get("min_wer"),
                                   "dnsmos": me.get("dnsmos_ovrl")}}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

json.dump(results, io.open(outdir.parent / "vocab-mapped-results.json", "w",
                           encoding="utf-8"), indent=2, ensure_ascii=False)
