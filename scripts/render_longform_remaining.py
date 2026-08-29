"""Render longform for caedral-m2/f1/f2 + classify all missing V1 renders."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
import classify_preview as cp  # noqa: E402
import soundfile as sf  # noqa: E402

engine = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e = engine
e.warmup()

LONGFORM = ("The morning market opened early, as it always did on Saturdays. "
            "Vendors arranged their stalls under striped awnings while the "
            "streetlights were still on, and by seven o'clock the narrow road "
            "smelled of bread, wet stone, and strong coffee.")
SENTENCES = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
}
voices_to_render = {
    "caedral-m2": {"kokoro": "am_echo"},
    "caedral-f1": {"kokoro": "af_sky"},
    "caedral-f2": {"kokoro": "af_nova"},
}

for vid, info in voices_to_render.items():
    vdir = REPO / "artifacts" / "caedral-voice-v1" / vid
    vdir.mkdir(exist_ok=True)

    # Render longform
    audio = e.synth(LONGFORM, voice_id=vid, speed=1.0)
    f = vdir / "longform.wav"
    sf.write(str(f), audio, 24000)
    audio_s = len(audio) / 24000
    print(f"{vid}/longform: {audio_s:.2f}s")

    # Classify existing renders that haven't been scored
    for sname in ("control", "technical", "numbers", "propernoun"):
        wf = vdir / f"{sname}.wav"
        if not wf.exists():
            continue
        refs = {**SENTENCES, "longform": LONGFORM}
        ref = refs.get(sname, "")
        if sname == "longform":
            r = cp.classify_file(str(f), LONGFORM)
        elif sname == "technical":
            r = cp.classify_file(str(wf),
                                 "The CPU supports AVX2 instructions but not AVX-512.")
        else:
            r = cp.classify_file(str(wf), ref)
        print(f"  {vid}/{sname}: {r.get('class')} WER={r.get('min_wer')}")

print("\nDONE")
