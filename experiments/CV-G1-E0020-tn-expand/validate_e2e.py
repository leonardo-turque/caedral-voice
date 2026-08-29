"""CV-G1-E0020-tn-expand E2E validation.

Renders "The CPU supports AVX2 instructions but not AVX-512." through
kokoro_onnx (is_phonemes=True) three ways:
  OLD   : pre-upgrade mapped stream replayed from baseline_capture.json
  NEW-A : upgraded frontend (AVX-512 -> "... five hundred twelve")
  NEW-B : variant with "five one two" digit-wise reading for 512

Each wav is classified by scripts/classify_preview.classify_file against two
references: the literal sentence and its spoken form. Results -> results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kokoro_onnx import Kokoro  # noqa: E402

from caedral_voice.text import TextFrontend, normalize  # noqa: E402
import classify_preview as cp  # noqa: E402

SENTENCE = "The CPU supports AVX2 instructions but not AVX-512."
SPOKEN_REF = ("The C P U supports A V X two instructions but not "
              "A V X five hundred twelve.")
OUTDIR = REPO / "experiments" / "CV-G1-E0020-tn-expand"
VOICES = ("af_heart", "am_michael")


def map_stream(res, vocab):
    """v4 vocab mapping: word budgets from len(selected); '|' -> ' ',
    '||' -> '. '; per-word boundary spaces."""
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
    skipped = []
    for tok in res.phoneme_tokens:
        if tok == "|":
            out.append(" ")
            continue
        if tok == "‖":
            out.append(". ")
            continue
        seg, pending = map_token(tok)
        if not seg:
            skipped.append(tok)
            continue
        out.extend(seg)
        consumed += len(seg)
        if wi < len(res.words) and consumed >= len(res.words[wi].selected):
            out.append(" ")
            wi += 1
            consumed = 0
    return "".join(out), skipped


def main() -> None:
    import soundfile as sf

    cap = json.loads(
        (OUTDIR / "baseline_capture.json").read_text(encoding="utf-8"))
    old_stream = cap["sentences"]["abbrev"]["mapped_stream"]

    k = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
               str(REPO / "models/kokoro/voices-v1.0.bin"))
    vocab = set(k.tokenizer.vocab.keys())

    tf = TextFrontend("lexicon+g2pen")

    # NEW-A: shipped behavior. NEW-B: digit-wise 512 reading for WER compare.
    res_a = tf.process(SENTENCE)
    stream_a, skipped_a = map_stream(res_a, vocab)
    norm_b = normalize(SENTENCE).replace("five hundred twelve", "five one two")
    res_b = tf.process(norm_b)
    stream_b, _ = map_stream(res_b, vocab)

    print("OLD stream :", ascii(old_stream))
    print("NEW-A norm :", ascii(res_a.text_normalized))
    print("NEW-A strm :", ascii(stream_a))
    print("NEW-B norm :", ascii(norm_b))
    print("skipped NEW-A:", ascii(sorted(set(skipped_a))))

    audiodir = OUTDIR / "audio"
    audiodir.mkdir(parents=True, exist_ok=True)
    models = cp.load_models()

    variants = {
        "old": old_stream,
        "new_a": stream_a,
        "new_b": stream_b,
    }
    results = []
    for vname, stream in variants.items():
        for voice in VOICES:
            wav_path = audiodir / f"e0020-{vname}-{voice}.wav"
            audio, sr = k.create(stream, voice=voice, speed=1.0,
                                 lang="en-us", is_phonemes=True)
            sf.write(wav_path, audio, sr)

            m_lit = cp.classify_file(wav_path, SENTENCE, models=models)
            m_spk = cp.classify_file(wav_path, SPOKEN_REF, models=models)
            row = {
                "variant": vname,
                "voice": voice,
                "file": str(wav_path.relative_to(REPO)),
                "seconds": round(len(audio) / sr, 2),
                "vs_literal_ref": {
                    "class": m_lit["class"],
                    "min_wer": m_lit["min_wer"],
                    "dnsmos_ovrl": m_lit["dnsmos_ovrl"],
                    "hypothesis": m_lit["hypothesis"],
                },
                "vs_spoken_ref": {
                    "class": m_spk["class"],
                    "min_wer": m_spk["min_wer"],
                    "dnsmos_ovrl": m_spk["dnsmos_ovrl"],
                },
            }
            results.append(row)
            print(json.dumps({**row,
                              "vs_literal_ref": {k: row["vs_literal_ref"][k]
                                                 for k in ("class", "min_wer", "dnsmos_ovrl")},
                              "vs_spoken_ref": {k: row["vs_spoken_ref"][k]
                                                for k in ("class", "min_wer", "dnsmos_ovrl")},
                              }, ensure_ascii=True))

    (OUTDIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUTDIR / "results.json")


if __name__ == "__main__":
    main()
