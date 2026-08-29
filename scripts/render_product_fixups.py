"""Attempt-2 fixups for product-experience assets (duration-target only).

Re-renders ONLY clips whose measured duration missed their brief:
video narration (46.35 s -> target 27-33 s) via tightened wording, and
sample-caedral-m1 (12.18 s -> <= 10 s). Wording changes are authorized for
pacing and are documented per clip in the manifest. Engine untouched.
"""
from __future__ import annotations

import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "out" / "product-experience"

# tightened scripts (chars tuned to measured ~62 ms/char incl. pauses)
FIXUPS = [
    ("video-01", "caedral-f1",
     "Meet Caedral. One API for every model you build with.",
     "PACE-TIGHTENED: dropped added second sentence; back near the original "
     "starting script."),
    ("video-02", "caedral-f1",
     "Sign up in minutes, verify your account, and Caedral greets you with "
     "a voice of its own.",
     "PACE-TIGHTENED: merged into one sentence; no claim change."),
    ("video-03", "caedral-m1",
     "Inside, discover models, compare live prices, and listen before you "
     "choose. This is Caedral Voice.",
     "PACE-TIGHTENED: trimmed 'side by side'/'even'; kept listening feature "
     "and engine identity."),
    ("video-04", "caedral-m1",
     "Create your key, call any model from your own code, and watch usage "
     "appear in real time.",
     "PACE-TIGHTENED: shortened ledger phrasing to 'appear in real time'."),
    ("video-05", "caedral-f1",
     "Choose a plan, and your included credits are ready the moment your "
     "subscription activates.",
     "PACE-TIGHTENED: removed closing clause 'No waiting, no setup.'"),
    ("video-06", "caedral-f1",
     "Caedral. Build with a voice of its own.",
     "PACE-TIGHTENED: restored the original starting script exactly."),
    ("sample-caedral-m1", "caedral-m1",
     "From first line of code to finished product, one steady voice can "
     "guide the whole journey: natural, reliable, always clear.",
     "DURATION FIX: sample shortened from 12.18 s to fit the 6-10 s brief."),
]

# import helpers from the main renderer
import render_product_experience as rpe  # noqa: E402


def main() -> None:
    from caedral_voice.inference.kokoro_engine import (
        VOICE_PROFILES, CaedralVoiceKokoroEngine)

    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=6,
    )

    manifest_path = OUT / "PRODUCT-EXPERIENCE-AUDIO-MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {e["filename"].rsplit(".", 1)[0]: e for e in manifest["clips"]}

    for name, voice_id, text, note in FIXUPS:
        profile = VOICE_PROFILES[voice_id]
        phonemes, _ = engine.phonemize(text)
        t0 = time.perf_counter()
        audio = engine.infer_from_phonemes(
            phonemes, profile["kokoro_voice"], speed=1.0)
        wall = time.perf_counter() - t0
        trimmed = rpe.trim_silence(audio.astype(np.float32), 24000,
                                   name.startswith("video"))
        dur = len(trimmed) / 24000
        data = rpe.pcm16(trimmed)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(data)
        path = OUT / f"{name}.wav"
        path.write_bytes(buf.getvalue())

        e = by_name[name]
        e.update({
            "script_final": text,
            "duration_seconds": round(dur, 2),
            "generation_wall_s": round(wall, 3),
            "rtf": round(wall / max(len(audio) / 24000, 1e-9), 4),
            "render_attempts": 2,
            "wording_changed": True,
            "wording_note": note,
            "sha256": rpe.sha256_bytes(path.read_bytes()),
        })
        print(f"{name}: attempt 2 dur={dur:.2f}s wall={wall:.2f}s", flush=True)

    vid = sum(e["duration_seconds"] for e in manifest["clips"][-6:])
    manifest["totals"]["video_narration_s"] = round(vid, 2)
    manifest["totals"]["all_clips_s"] = round(
        sum(e["duration_seconds"] for e in manifest["clips"]), 2)
    manifest_path.write_text(json.dumps(manifest, indent=2),
                             encoding="utf-8")
    print(f"video narration total now {vid:.2f} s")
    print(f"updated {manifest_path}")


if __name__ == "__main__":
    main()
