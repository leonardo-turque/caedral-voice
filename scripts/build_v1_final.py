"""PRE-ARCH ENHANCED BASE v3: Caedral TN + stock Kokoro phonemizer.

Fastest path to quality: our TN normalizes/expands technical text, then
stock Kokoro renders it (proven NATURAL/INTELLIGIBLE). No phoneme mapping
needed for V1 — eliminates the IPA mismatch risk entirely.

Caedral-specific words (like "Caedral") are pre-expanded in TN so stock
Kokoro can pronounce them correctly.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kokoro_onnx import Kokoro  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import classify_preview as cp  # noqa: E402

# Caedral-specific pronunciation overrides applied as TEXT PRE-PROCESSING.
# These replace unknown proper nouns with phonetic spellings that stock
# Kokoro's phonemizer can handle correctly.
PRONUNCIATION_OVERRIDES = {
    "Caedral": "Kedral",
    "caedral.com": "kedral dot com",
    "support@caedral.com": "support at kedral dot com",
}

VOICES = {
    "MALE-01": "am_michael",
    "MALE-02": "am_echo",
    "FEMALE-01": "af_sky",
    "FEMALE-02": "af_nova",
}
SENTENCES = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
}


def apply_overrides(text: str) -> str:
    for original, replacement in PRONUNCIATION_OVERRIDES.items():
        text = text.replace(original, replacement)
    return text


def main() -> None:
    k = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
               str(REPO / "models/kokoro/voices-v1.0.bin"))
    out_dir = REPO / "artifacts" / "pre-arch-enhanced-base" / "v3-final"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, voice in VOICES.items():
        vdir = out_dir / label
        vdir.mkdir(exist_ok=True)
        for sname, raw_text in SENTENCES.items():
            # Step 1: Caedral TN overrides for unknown proper nouns
            expanded = apply_overrides(raw_text)
            # Step 2: stock Kokoro renders the expanded text
            t0 = time.perf_counter()
            audio, sr = k.create(expanded, voice=voice, speed=1.0, lang="en-us")
            wall = time.perf_counter() - t0
            f = vdir / f"{sname}.wav"
            sf.write(str(f), audio, sr)
            audio_s = len(audio) / sr
            rtf = wall / audio_s if audio_s > 0 else 99

            row = {"voice_label": label, "kokoro_voice": voice,
                   "sentence": sname, "raw_text": raw_text,
                   "expanded_text": expanded,
                   "class": None, "wer": None, "dnsmos": None,
                   "audio_seconds": round(audio_s, 2),
                   "rtf": round(rtf, 3)}
            # Classify
            try:
                r = cp.classify_file(str(f), raw_text)
                row["class"] = r.get("class")
                row["wer"] = r.get("min_wer")
                row["dnsmos"] = r.get("dnsmos_ovrl")
            except Exception:
                pass
            results.append(row)
            print(f"{label}/{sname}: {row['class']} WER={row['wer']} "
                  f"RTF={row['rtf']}")

    json.dump(results, io.open(out_dir / "results.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n{len(results)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
