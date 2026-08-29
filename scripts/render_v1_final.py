"""CAEDRAL VOICE V1 — render the complete founder deliverable.

4 locked voices x 5-sentence validation set + speed samples + benchmark.
Uses the canonical CaedralVoiceKokoroEngine with our frontend + vocab mapping.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import soundfile as sf2
import torch

OUT = REPO / "artifacts" / "caedral-voice-v1"
VOICES = {
    "caedral-m1": {"kokoro": "am_michael", "gender": "male",
                   "persona": "natural balanced male"},
    "caedral-m2": {"kokoro": "am_echo", "gender": "male",
                   "persona": "distinct second male identity"},
    "caedral-f1": {"kokoro": "af_sky", "gender": "female",
                   "persona": "natural balanced female"},
    "caedral-f2": {"kokoro": "af_nova", "gender": "female",
                   "persona": "distinct second female identity"},
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
SPEED_SAMPLES = [0.5, 1.0, 1.5, 2.0]


def main() -> None:
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    print("[V1] loading engine...")
    t0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=4,
    )
    load_s = time.perf_counter() - t0
    print(f"[V1] loaded in {load_s:.2f}s")

    # warmup
    engine.synth("Warm.", VOICES["caedral-m1"]["kokoro"], speed=1.0)

    bench = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_load_seconds": round(load_s, 3),
        "model_size_mb": 310.5,
        "voices": {},
        "speed_samples": [],
    }

    for label, info in VOICES.items():
        vdir = OUT / label
        vdir.mkdir(exist_ok=True)
        voice_rows = []
        for sname, text in SET.items():
            t0 = time.perf_counter()
            audio = engine.synth(text, voice_name=info["kokoro"], speed=1.0)
            wall = time.perf_counter() - t0
            audio_s = len(audio) / 24000
            rtf = wall / audio_s if audio_s > 0 else 99
            f = vdir / f"{sname}.wav"
            sf2.write(str(f), audio, 24000)
            voice_rows.append({"sentence": sname, "wall_s": round(wall, 3),
                               "audio_s": round(audio_s, 2), "rtf": round(rtf, 3)})
            print(f"  {label}/{sname}: RTF {rtf:.3f}")

        med_rtf = sorted(r["rtf"] for r in voice_rows)[len(voice_rows)//2]
        bench["voices"][label] = {
            "kokoro_voice": info["kokoro"], "persona": info["persona"],
            "renders": voice_rows, "median_rtf": round(med_rtf, 3),
        }

    # speed samples on caedral-m1/control
    spdir = OUT / "speed"
    spdir.mkdir(exist_ok=True)
    text_c = SET["control"]
    sid_m1 = None
    for s in SPEED_SAMPLES:
        audio = engine.synth(text_c, voice_name=VOICES["caedral-m1"]["kokoro"], speed=s)
        f = spdir / f"speed-{s}x.wav"
        sf2.write(str(f), audio, 24000)
        bench["speed_samples"].append({"speed": s, "file": str(f.relative_to(OUT)),
                                       "audio_s": round(len(audio)/24000, 2)})
        print(f"  speed {s}x: {len(audio)/24000:.2f}s")

    # baseline copies (stock renders for comparison)
    base_dir = OUT / "comparison"
    base_dir.mkdir(exist_ok=True)
    for src in (REPO / "artifacts/pre-arch-enhanced-base/vocab-mapped-v2").glob("*.wav"):
        shutil.copy2(src, base_dir / src.name)
    for src in (REPO / "artifacts/pre-arch-enhanced-base/baseline").glob("*.wav"):
        shutil.copy2(src, base_dir / f"stock-{src.name}")

    bench_path = OUT / "benchmark.json"
    json.dump(bench, io.open(bench_path, "w", encoding="utf-8"), indent=2)

    # voice profiles
    for label, info in VOICES.items():
        prof = {
            "caedral_voice_id": label,
            "base_kokoro_voice": info["kokoro"],
            "base_provenance": "Kokoro v1.0 ONNX (Apache-2.0), https://github.com/thewh1teagle/kokoro-onnx",
            "caedral_contribution": "text frontend, TN expansion, IPA->Kokoro vocab mapping, pronunciation overrides",
            "persona": info["persona"],
            "version": "v1.0",
        }
        atomic_write_json(OUT / label / "profile.json", prof)

    print(f"\n[V1] DONE -> {OUT}")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    import os
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
