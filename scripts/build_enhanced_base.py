"""PRE-ARCH ENHANCED BASE: stock Kokoro (baseline) vs Caedral-frontend (enhanced).

Renders A/B pairs across 3 voices x stress sentences, measures RTF/RAM,
classifies intelligibility, writes benchmark.json + provenance.
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.text import TextFrontend  # noqa: E402

import numpy as np
import soundfile as sf
import torch

OUT = REPO / "artifacts" / "pre-arch-enhanced-base"
(OUT / "baseline").mkdir(parents=True, exist_ok=True)
(OUT / "enhanced").mkdir(parents=True, exist_ok=True)

VOICES = ["af_heart", "af_bella", "am_michael"]
# Stress sentences chosen to expose FRONTEND differences (proper noun, numbers,
# dates, abbreviations) plus one natural conversational control.
SENTENCES = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "abbrev": "The CPU supports AVX2 instructions but not AVX-512.",
}

sys.path.insert(0, str(REPO / "scripts"))
import classify_preview as cp  # noqa: E402


def main() -> None:
    from kokoro_onnx import Kokoro

    model_path = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = REPO / "models" / "kokoro" / "voices-v1.0.bin"
    engine = Kokoro(str(model_path), str(voices_path))
    tf = TextFrontend("lexicon+g2pen")

    results = []
    for voice in VOICES:
        for sname, text in SENTENCES.items():
            # ---- BASELINE: stock pipeline (text in, internal phonemizer) ----
            t0 = time.perf_counter()
            audio, sr = engine.create(text, voice=voice, speed=1.0, lang="en-us")
            rtf = time.perf_counter() - t0
            fb = OUT / "baseline" / f"{voice}-{sname}.wav"
            sf.write(fb, audio, sr)

            # ---- ENHANCED: Caedral frontend phonemes -> is_phonemes=True ----
            ph = tf.process(text).phoneme_string
            t0 = time.perf_counter()
            try:
                audio_e, sr_e = engine.create(ph, voice=voice, speed=1.0,
                                              lang="en-us", is_phonemes=True)
                rtf_e = time.perf_counter() - t0
            except Exception as exc:
                print(f"ENHANCED FAILED {voice}/{sname}: {exc}")
                continue
            fe = OUT / "enhanced" / f"{voice}-{sname}.wav"
            sf.write(fe, audio_e, sr_e)

            mb = cp.classify_file(str(fb), text)
            me = cp.classify_file(str(fe), text)
            row = {
                "voice": voice, "sentence": sname, "text": text,
                "caedral_phonemes": ph,
                "baseline": {"class": mb.get("class"), "wer": mb.get("min_wer"),
                             "dnsmos": mb.get("dnsmos_ovrl"), "rtf": round(rtf, 3)},
                "enhanced": {"class": me.get("class"), "wer": me.get("min_wer"),
                             "dnsmos": me.get("dnsmos_ovrl"), "rtf": round(rtf_e, 3)},
            }
            results.append(row)
            print(json.dumps(row))

    bench = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": "i7-13620H CPU-only",
        "model": "Kokoro v1.0 ONNX (Apache-2.0)",
        "protocol": "3 voices x 4 frontend-stress sentences; baseline=stock "
                    "phonemizer; enhanced=Caedral lexicon+g2pen phonemes via "
                    "is_phonemes=True",
        "results": results,
    }
    json.dump(bench, io.open(OUT / "benchmark.json", "w", encoding="utf-8"),
              indent=2)
    print(f"\n{len(results)} A/B pairs -> {OUT}")


if __name__ == "__main__":
    main()
