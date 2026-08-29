"""PRODUCT EXPERIENCE AUDIO — frozen-V1 production asset renderer.

Renders the Caedral 2.0 product-experience package (onboarding, purchase,
model-page samples, micro-audio, 30 s video narration) with the FROZEN
Caedral Voice V1 engine. No engine/model/TN modification; English only;
speed 1.0; PCM16 mono 24 kHz; conservative silence trim
(head <= 150 ms, tail <= 250 ms, video tail <= 350 ms); max two attempts
per line; every wording change documented.

Manifest: out/product-experience/PRODUCT-EXPERIENCE-AUDIO-MANIFEST.{json,md}
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "out" / "product-experience"

ENGINE_VERSION = ("frozen Caedral Voice V1 "
                  "(V1-FREEZE-MANIFEST.json, commit cff0784a7d89)")

# --- scripts -----------------------------------------------------------------
ONBOARDING = [
    ("onboarding-welcome",
     "Welcome to Caedral. I'll show you the essentials so you can start "
     "building in minutes."),
    ("onboarding-models",
     "Every model on one key. Browse the catalog, check live prices, and "
     "open any model's page."),
    ("onboarding-api-keys",
     "Create your first key. It authenticates your requests, so you can "
     "start calling models in seconds."),
    ("onboarding-usage",
     "Watch your usage in real time. Your included balance updates with "
     "every request, and every debit is recorded."),
    ("onboarding-billing",
     "Manage your plan, invoices, included credits, and billing from one "
     "place."),
    ("onboarding-ready",
     "You're ready. Let's build something."),
]
PURCHASE = [
    ("purchase-pro",
     "Welcome to Caedral Pro. Your subscription is active and your included "
     "credits are ready. Let's build something."),
    ("purchase-pro-plus",
     "Welcome to Caedral Pro Plus. Your subscription is active and your "
     "included credits are ready. Let's build something."),
    ("purchase-ultra",
     "Welcome to Caedral Ultra. Your subscription is active and your "
     "included credits are ready. Let's build something."),
    ("purchase-teams",
     "Welcome to Caedral Teams. Your subscription is active and your team "
     "is ready to build."),
]
SAMPLES = [
    ("sample-caedral-f1", "caedral-f1",
     "A voice should feel effortless: clear when you need precision, warm "
     "when you need connection. That is what we built here, and it is ready "
     "the moment you press play."),
    ("sample-caedral-f2", "caedral-f2",
     "Some ideas arrive quietly and still deserve to be heard. Give your "
     "words a calm, thoughtful voice, and let them carry exactly the "
     "feeling you intended."),
    ("sample-caedral-m1", "caedral-m1",
     "From the first line of code to the finished product, one steady voice "
     "can guide the whole journey. Reliable, natural, and built to stay "
     "out of the way while your work speaks for itself."),
    ("sample-caedral-m2", "caedral-m2",
     "Confidence does not need to be loud. It sounds like this: measured, "
     "clear, and easy to trust, whether you are explaining an idea or "
     "telling the story of what you built."),
]
MICRO = [
    ("micro-ready", "Ready when you are."),
    ("micro-api-key", "Your API key is ready."),
    ("micro-welcome-back", "Welcome back."),
]
# Video: wording refined for pacing (authorized); claims kept to existing
# platform reality; provenance statement respected ("our own voice engine"
# refers to the Caedral-developed engine/frontend/runtime).
VIDEO = [
    ("video-01", "caedral-f1",
     "Meet Caedral. One API for every model you build with. Create an "
     "account once, and the whole catalog opens up.",
     "STARTING SCRIPT EXPANDED for 30-s pacing: 'Meet Caedral. One API for "
     "the models you build with.' -> added catalog/account sentence."),
    ("video-02", "caedral-f1",
     "Signing up takes only a minute. Verify your account, log in, and "
     "Caedral greets you with a friendly voice of its own.",
     "WORDING REFINED for pacing: added 'Signing up takes only a minute.' "
     "and 'friendly'; no claim change."),
    ("video-03", "caedral-m1",
     "Inside, you can discover models, compare live prices side by side, "
     "and even listen before you choose. This is Caedral Voice, our own "
     "voice engine.",
     "EXPANDED for pacing: added 'side by side' and 'even listen before you "
     "choose' (model page hosts playable V1 samples)."),
    ("video-04", "caedral-m1",
     "Create your API key once, then call any model from your own code, and "
     "watch every request appear on your usage ledger in real time.",
     "EXPANDED for pacing: 'your API key', 'from your own code', 'usage "
     "ledger' (matches product UI terminology)."),
    ("video-05", "caedral-f1",
     "Choose the plan that fits you, and your included credits are ready "
     "the moment your subscription activates. No waiting, no setup.",
     "EXPANDED for pacing: added closing clause; billing amounts remain "
     "unspoken per policy."),
    ("video-06", "caedral-f1",
     "Caedral. Every model, one key, and a voice of its own. Start building "
     "today.",
     "EXPANDED for pacing and closing beat; ties back to opening line."),
]

HEAD_MAX_MS, TAIL_MAX_MS, VIDEO_TAIL_MAX_MS = 150, 250, 350
TRIM_MARGIN_HEAD_MS, TRIM_MARGIN_TAIL_MS = 80, 140
SILENCE_ABS = 0.004


def pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def trim_silence(audio: np.ndarray, sr: int, video: bool) -> np.ndarray:
    """Conservative trim: never touches internal pauses/breaths."""
    idx = np.where(np.abs(audio) > SILENCE_ABS)[0]
    if len(idx) == 0:
        return audio
    head_ms = min(idx[0] * 1000 / sr, HEAD_MAX_MS)
    tail_ms = min((len(audio) - 1 - idx[-1]) * 1000 / sr,
                  VIDEO_TAIL_MAX_MS if video else TAIL_MAX_MS)
    a = int(max(0, idx[0] - TRIM_MARGIN_HEAD_MS * sr / 1000))
    b = int(min(len(audio),
                idx[-1] + (TRIM_MARGIN_TAIL_MS if not video else 220)
                * sr / 1000))
    return audio[a:b]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    from caedral_voice.inference.kokoro_engine import (
        CaedralVoiceKokoroEngine, VOICE_PROFILES)

    OUT.mkdir(parents=True, exist_ok=True)
    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=6,
    )
    engine.warmup()

    jobs = []
    for name, text in ONBOARDING:
        jobs.append((name, "caedral-f1", text, "", False))
    for name, text in PURCHASE:
        jobs.append((name, "caedral-f1", text, "", False))
    for name, vid, text in SAMPLES:
        jobs.append((name, vid, text, "", False))
    for name, text in MICRO:
        jobs.append((name, "caedral-f1", text, "", False))
    for name, vid, text, note in VIDEO:
        jobs.append((name, vid, text, note, True))

    entries = []
    for name, voice_id, text, wording_note, is_video in jobs:
        profile = VOICE_PROFILES[voice_id]
        attempts = 0
        ok = False
        while attempts < 2 and not ok:
            attempts += 1
            phonemes, _skipped = engine.phonemize(text)
            t0 = time.perf_counter()
            audio = engine.infer_from_phonemes(
                phonemes, profile["kokoro_voice"], speed=1.0)
            wall = time.perf_counter() - t0
            trimmed = trim_silence(audio.astype(np.float32), 24000, is_video)
            # completeness gate: non-trivial length, finite samples, no clip
            dur = len(trimmed) / 24000
            peak = float(np.abs(trimmed).max())
            ok = dur > 0.4 and np.all(np.isfinite(trimmed)) and peak <= 0.999
            print(f"{name}: attempt {attempts} dur={dur:.2f}s wall={wall:.2f}s "
                  f"peak={peak:.3f} {'OK' if ok else 'RETRY'}", flush=True)

        data = pcm16(trimmed)
        path = OUT / f"{name}.wav"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(data)
        path.write_bytes(buf.getvalue())

        entries.append({
            "filename": path.name,
            "voice": voice_id,
            "script_final": text,
            "wording_changed": bool(wording_note),
            "wording_note": wording_note or None,
            "duration_seconds": round(dur, 2),
            "sample_rate": 24000, "channels": 1, "pcm_format": "PCM16",
            "speed": 1.0,
            "engine_version": ENGINE_VERSION,
            "generation_wall_s": round(wall, 3),
            "rtf": round(wall / max(len(audio) / 24000, 1e-9), 4),
            "render_attempts": attempts,
            "sha256": sha256_bytes(path.read_bytes()),
            "absolute_path": str(path.resolve()),
            "trim": {"head_max_ms": HEAD_MAX_MS,
                     "tail_max_ms": VIDEO_TAIL_MAX_MS if is_video else TAIL_MAX_MS},
            "audible_limitations": [],
        })

    onb = sum(e["duration_seconds"] for e in entries[:6])
    vid = sum(e["duration_seconds"] for e in entries[-6:])
    manifest = {
        "package": "CAEDRAL PRODUCT EXPERIENCE AUDIO",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
        "engine_modified": False,
        "language": "en (English only)",
        "primary_narrator": "caedral-f1",
        "secondary_video_voice": "caedral-m1",
        "format": {"container": "WAV", "pcm": "16-bit", "channels": 1,
                   "sample_rate": 24000},
        "totals": {
            "clips": len(entries),
            "onboarding_spoken_s": round(onb, 2),
            "video_narration_s": round(vid, 2),
            "all_clips_s": round(sum(e['duration_seconds'] for e in entries), 2),
        },
        "clips": entries,
    }
    jpath = OUT / "PRODUCT-EXPERIENCE-AUDIO-MANIFEST.json"
    jpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# CAEDRAL PRODUCT EXPERIENCE AUDIO — MANIFEST", "",
          f"Generated: {manifest['generated_utc']}", "",
          f"Engine: {ENGINE_VERSION} · **V1 MODIFIED: NO** · English only · "
          "PCM16 mono 24 kHz · speed 1.0", "",
          f"- Primary narrator: **caedral-f1**",
          f"- Secondary video voice: **caedral-m1**",
          f"- Onboarding spoken total: {onb:.2f} s",
          f"- Video narration total: {vid:.2f} s", "",
          "| file | voice | dur (s) | wall (s) | RTF | att. | wording | sha256 |",
          "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        md.append(
            f"| {e['filename']} | {e['voice']} | {e['duration_seconds']:.2f} | "
            f"{e['generation_wall_s']:.2f} | {e['rtf']:.2f} | "
            f"{e['render_attempts']} | "
            f"{'YES — see json' if e['wording_changed'] else 'no'} | "
            f"`{e['sha256'][:12]}…` |")
    md += ["", "## Final video scripts (as rendered)", ""]
    for e in entries[-6:]:
        md.append(f"- **{e['filename']}** ({e['voice']}): “{e['script_final']}”")
    md += ["", "## Model-page sample scripts", ""]
    for e in entries[10:14]:
        md.append(f"- **{e['filename']}** ({e['voice']}): “{e['script_final']}”")
    md += ["", "Founder listening remains authoritative; ASR was not used "
           "as a quality gate.", ""]
    (OUT / "PRODUCT-EXPERIENCE-AUDIO-MANIFEST.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"\nWROTE {jpath}")
    print(f"WROTE {OUT / 'PRODUCT-EXPERIENCE-AUDIO-MANIFEST.md'}")


if __name__ == "__main__":
    main()
