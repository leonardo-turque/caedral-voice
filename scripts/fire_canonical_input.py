"""Select + freeze the canonical ~30 s fire-test input (caedral-f1, speed 1.0).

Text-selection ONLY: uses the frozen engine unmodified to MEASURE which
candidate lands inside 29.0-31.0 s. The winning text is written to
artifacts/fire-test/canonical-input.txt (+ SHA-256 sidecar record inside
canonical-input.meta.json).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CANDIDATES = {
    "cand-a": (
        "The old lighthouse keeper climbed the spiral stairs each evening, "
        "counting the worn steps out of habit rather than need. At the top, he "
        "polished the great lamp, checked the wick, and watched the first stars "
        "appear over the water. Fishing boats returned to the harbor in a slow "
        "procession, their lights trembling on the swells."),
    "cand-b": (
        "The old lighthouse keeper climbed the spiral stairs each evening, "
        "counting the worn steps out of habit rather than need. At the top, he "
        "polished the great lamp, checked the wick, and watched the first stars "
        "appear over the water. Fishing boats returned to the harbor in a slow "
        "procession, their lights trembling on the swells. He remembered when "
        "the fleet had been larger, when children had run along the pier to "
        "meet their fathers, when the whole town had smelled of salt and tar "
        "and fresh bread."),
    "cand-c": (
        "The old lighthouse keeper climbed the spiral stairs each evening, "
        "counting the worn steps out of habit rather than need. At the top, he "
        "polished the great lamp, checked the wick, and watched the first stars "
        "appear over the water. Fishing boats returned to the harbor in a slow "
        "procession, their lights trembling on the swells. He remembered when "
        "the fleet had been larger, when children had run along the pier to "
        "meet their fathers, when the whole town had smelled of salt and tar "
        "and fresh bread. Now the streets grew quiet soon after sunset, but the "
        "light still turned, steady and patient, sweeping its long white arm "
        "across the bay, keeping faith with every vessel that dared the dark."),
    "cand-d": (
        "The old lighthouse keeper climbed the spiral stairs each evening, "
        "counting the worn steps out of habit rather than need. At the top, he "
        "polished the great lamp, checked the wick, and watched the first stars "
        "appear over the water. Fishing boats returned to the harbor in a slow "
        "procession, their lights trembling on the swells. He remembered when "
        "the fleet had been larger, when children had run along the pier to "
        "meet their fathers, when the whole town had smelled of salt and tar "
        "and fresh bread. The tide rose quietly beneath the old wooden pier."),
}


def main() -> None:
    from caedral_voice.inference.kokoro_engine import (
        CaedralVoiceKokoroEngine, VOICE_PROFILES)

    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=6,
    )

    def render_30s(text: str):
        """Frozen-engine public API path for >510-phoneme inputs.

        engine.synth() raises in its post-render stats call for such inputs
        (frozen limitation); phonemize() + infer_from_phonemes() exercise the
        identical frontend + fused graph with the wrapper's internal batching
        and are the production-service entry points. No engine modification.
        """
        phonemes, _ = engine.phonemize(text)
        return engine.infer_from_phonemes(
            phonemes, VOICE_PROFILES["caedral-f1"]["kokoro_voice"], speed=1.0)

    results = {}
    for name, text in CANDIDATES.items():
        audio = render_30s(text)
        dur = len(audio) / 24000
        results[name] = (dur, len(text))
        print(f"{name}: {len(text)} chars -> {dur:.2f} s  "
              f"{'IN RANGE' if 29.0 <= dur <= 31.0 else ''}")

    winner = min(results.items(),
                 key=lambda kv: abs(kv[1][0] - 30.0))
    name, (dur, nchars) = winner
    print(f"\nWINNER: {name} at {dur:.2f} s ({nchars} chars)")
    if not 29.0 <= dur <= 31.0:
        print("NOT IN RANGE - extend candidates before freezing")
        return
    out_dir = REPO / "artifacts" / "fire-test"
    text = CANDIDATES[name]
    (out_dir / "canonical-input.txt").write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta = {"selected": name, "duration_s_measured": round(dur, 3),
            "chars": nchars, "voice": "caedral-f1", "speed": 1.0,
            "sample_rate": 24000, "sha256": sha,
            "note": "Frozen canonical fire-test input; identical text for both profiles."}
    import json
    json.dump(meta, open(out_dir / "canonical-input.meta.json", "w"),
              indent=2)
    print(f"wrote canonical-input.txt sha256={sha}")


if __name__ == "__main__":
    main()
