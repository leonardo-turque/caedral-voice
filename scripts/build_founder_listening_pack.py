"""Founder listening pack: 4 V1 voices x frozen set via CAEDRAL frontend."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import soundfile as sf2
import torch
from kokoro_onnx import Kokoro

from caedral_voice.text import TextFrontend
import classify_preview as cp

SELECTED = {
    "MALE-01": "am_echo",
    "MALE-02": "am_onyx",
    "FEMALE-01": "af_sky",
    "FEMALE-02": "af_nova",
}
SET = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "longform": ("The morning market opened early, as it always did on Saturdays. "
                 "Vendors arranged their stalls under striped awnings while the "
                 "streetlights were still on, and by seven o'clock the narrow road "
                 "smelled of bread, wet stone, and strong coffee."),
}

k = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
           str(REPO / "models/kokoro/voices-v1.0.bin"))
VOCAB = set(k.tokenizer.vocab.keys())
tf = TextFrontend("lexicon+g2pen")


def caedral_to_kokoro(text: str) -> tuple[str, list[str]]:
    res = tf.process(text)
    out: list[str] = []
    skipped: list[str] = []
    pending = ""
    for tok in res.phoneme_tokens:
        if tok == "|":
            out.append(" ")
            continue
        if tok == "‖":
            out.append(". ")
            continue
        if tok in VOCAB:
            out.append(tok)
            continue
        ok = True
        for ch in tok:
            if ch in VOCAB:
                out.append(pending + ch)
                pending = ""
            elif ch in ("ˈ", "ˌ"):
                pending += ch
            elif ch == "ɝ":
                out.append(pending + "ɜ")
                pending = ""
                if "r" in VOCAB:
                    out.append("r")
            else:
                ok = False
        if not ok:
            skipped.append(tok)
    return "".join(out), sorted(set(skipped))


pack_dir = REPO / "artifacts" / "v1-listening-pack"
pack_dir.mkdir(parents=True, exist_ok=True)
results = []

for label, voice in SELECTED.items():
    vdir = pack_dir / label
    vdir.mkdir(exist_ok=True)
    sid_rows = []
    for sname, text in SET.items():
        phoneme_str, skipped = caedral_to_kokoro(text)
        t0 = time.time()
        audio, sr = k.create(phoneme_str, voice=voice, speed=1.0,
                             lang="en-us", is_phonemes=True)
        dt = time.time() - t0
        f = vdir / f"{sname}.wav"
        sf2.write(f, audio, sr)
        r = cp.classify_file(str(f), text)
        row = {"caedral_label": label, "kokoro_voice": voice,
               "sentence": sname, "file": str(f.relative_to(pack_dir)),
               "class": r.get("class"), "wer": r.get("min_wer"),
               "dnsmos": r.get("dnsmos_ovrl"),
               "audio_seconds": round(len(audio) / sr, 2),
               "synth_wall_s": round(dt, 2),
               "skipped_symbols": skipped}
        results.append(row)
        sid_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

json.dump({"selected": SELECTED, "renders": results},
          io.open(pack_dir / "listening-pack.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nlistening pack -> {pack_dir}")
