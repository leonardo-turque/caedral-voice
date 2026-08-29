"""Score ALL V1 voice candidate clips: whisper WER + DNSMOS, no agent deps."""
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

REFS = {
    "A": "Hello. This is the first generation of Caedral Voice.",
    "B": "The CPU supports AVX2 instructions but not AVX-512.",
    "C": "The invoice totals $1,247.83, due within 30 days.",
    "D": "Artificial intelligence should sound natural, expressive, and human.",
}

clips_dir = REPO / "artifacts" / "v1-voices" / "clips"
out_path = REPO / "artifacts" / "v1-voices" / "candidates-scored.json"

# Import classification tools directly (bypassing agent code)
import numpy as np
import soundfile as sf
import librosa

print("loading models...")
from faster_whisper import WhisperModel
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

# DNSMOS via speechmos
from speechmos.dnsmos import DNSMOS
pkg_dir = Path(REPO) / ".venv" / "Lib" / "site-packages" / "speechmos" / "dnsmos_models"
dnsmos_model = None
try:
    from speechmos.dnsmos import DNSMOS as _D
    dnsmos_model = _D(str(pkg_dir / "sig_bak_ovr.onnx"), str(pkg_dir / "model_v8.onnx"))
except Exception:
    # try alternate init
    pass


def score_clip(path: Path, ref_text: str) -> dict:
    audio, sr = sf.read(str(path), dtype="float32")
    duration = len(audio) / sr

    # WER via whisper
    segments, _ = whisper.transcribe(str(path), beam_size=1)
    hyp = " ".join(s.text for s in segments).strip()
    ref_words = set(ref_text.lower().split())
    hyp_words = hyp.lower().split()
    if ref_words:
        # simple word-level WER
        import jiwer
        wer = jiwer.wer(ref_text.lower(), hyp.lower())
    else:
        wer = 1.0

    # DNSMOS
    dnsmos_val = None
    if dnsmos_model is not None:
        try:
            # resample to 16k if needed
            if sr != 16000:
                import scipy.signal as sig
                audio_16k = sig.resample_poly(audio, 16000, sr)
            else:
                audio_16k = audio
            result = dnsmos_model(np.ascontiguousarray(audio_16k, dtype=np.float32), 16000, False)
            dnsmos_val = round(float(result["ovrl_mos"]), 3)
        except Exception:
            pass

    return {"file": path.name, "wer": round(float(wer), 4),
            "dnsmos": dnsmos_val, "duration_s": round(duration, 2),
            "hypothesis": hyp[:80]}


# Process all clips
rows = []
files = sorted(clips_dir.glob("*.wav"))
print(f"scoring {len(files)} clips...")
for i, f in enumerate(files):
    stem = f.stem  # e.g., af_bella_A
    parts = stem.split("_")
    letter = parts[-1]
    voice = "_".join(parts[:-1])
    ref = REFS.get(letter, "")
    r = score_clip(f, ref)
    row = {"voice": voice, "gender": "M" if voice.startswith("am_") else "F",
           "sentence": letter, **r}
    rows.append(row)
    print(f"  [{i+1}/{len(files)}] {stem}: WER={row['wer']} DNSMOS={row['dnsmos']}")

# Aggregate per voice
by_voice = {}
for row in rows:
    by_voice.setdefault(row["voice"], []).append(row)

agg = []
for voice, vrs in sorted(by_voice.items()):
    wers = [v["wer"] for v in vrs if v["wer"] is not None]
    dmos = [v["dnsmos"] for v in vrs if v["dnsmos"] is not None]
    agg.append({
        "voice": voice,
        "gender": vrs[0]["gender"],
        "median_wer": round(statistics.median(wers), 4) if wers else None,
        "max_wer": round(max(wers), 4) if wers else None,
        "median_dnsmos": round(statistics.median(dmos), 3) if dmos else None,
        "min_dnsmos": round(min(dmos), 3) if dmos else None,
        "sentences": vrs,
    })

out = {"scored_at": "2026-08-23", "total_clips": len(rows), "candidates": agg}
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)

print("\n=== RANKED BY MEDIAN DNSMOS ===")
for a in sorted(agg, key=lambda x: -(x["median_dnsmos"] or 0)):
    print(f"{a['voice']:14s} {a['gender']} | DNSMOS={a['median_dnsmos']}/{a['min_dnsmos']} "
          f"| WER={a['median_wer']}")
