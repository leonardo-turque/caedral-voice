"""Score all 48 V1 candidate clips -> candidates.json with per-voice aggregates."""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import classify_preview as cp  # noqa: E402

REFS = {
    "A": "Hello. This is the first generation of Caedral Voice.",
    "B": "The CPU supports AVX2 instructions but not AVX-512.",
    "C": "The invoice totals $1,247.83, due within 30 days.",
    "D": "Artificial intelligence should sound natural, expressive, and human.",
}

clips_dir = REPO / "artifacts" / "v1-voices" / "clips"
print(f"DEBUG clips_dir={clips_dir} exists={clips_dir.exists()} "
      f"wavs={len(list(clips_dir.glob('*.wav')))}")
rows = []
for f in sorted(clips_dir.glob("*.wav")):
    stem = f.stem
    if "_" not in stem:
        print(f"SKIP: {stem!r}")
        continue
    voice, letter = stem.rsplit("_", 1)
    gender = "M" if voice.startswith(("am_", "im_")) else "F"
    try:
        r = cp.classify_file(str(f), REFS.get(letter, ""))
    except ValueError:
        # peak >1 or resampler ringing — headroom-scale like GATE-NAT fix
        import numpy as np
        import soundfile as sfile
        data, sr = sfile.read(str(f), dtype="float32")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.abs(data).max()) if data.size else 0.0
        scale = 0.90 / peak if peak > 0.9 else 1.0
        data = data * scale
        clamped_path = str(f) + ".scaled.wav"
        sfile.write(clamped_path, data, sr)
        r = cp.classify_file(clamped_path, REFS.get(letter, ""))
        r["headroom_scaled"] = True
        r["peak"] = round(peak, 3)
    rows.append({"voice": voice, "gender": gender, "sentence": letter,
                 "class": r.get("class"), "wer": r.get("min_wer"),
                 "dnsmos": r.get("dnsmos_ovrl")})
    print(rows[-1])

# aggregate per voice
by_voice: dict[str, list] = {}
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
        "all_intelligible_or_better": all(
            v["class"] in ("INTELLIGIBLE", "NATURAL", "PARTIAL_SPEECH") for v in vrs),
        "sentences": vrs,
    })

out = {"candidates": agg}
json.dump(out, io.open(REPO / "artifacts" / "v1-voices" / "candidates.json", "w",
                       encoding="utf-8"), indent=2)
print("\n=== AGGREGATES (sorted by median DNSMOS desc) ===")
for a in sorted(agg, key=lambda x: -(x["median_dnsmos"] or 0)):
    print(f"{a['voice']:14s} {a['gender']} | DNSMOS med {a['median_dnsmos']} "
          f"min {a['min_dnsmos']} | WER med {a['median_wer']}")
