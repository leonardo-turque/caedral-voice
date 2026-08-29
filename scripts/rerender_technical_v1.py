"""Re-render technical.wav for all 4 V1 voices with E0023 fixes active."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
import classify_preview as cp  # noqa: E402
import soundfile as sf  # noqa: E402

text = "The CPU supports AVX2 instructions but not AVX-512."
e = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e.warmup()

results = {}
for vid in ("caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"):
    ph, sk = e.phonemize(text, use_cache=False)
    audio = e.synth(text, voice_id=vid, speed=1.0)
    out = REPO / "artifacts" / "caedral-voice-v1" / vid / "technical.wav"
    sf.write(str(out), audio, 24000)
    r = cp.classify_file(str(out), text)
    results[vid] = {"class": r.get("class"), "wer": r.get("min_wer"),
                    "dnsmos": r.get("dnsmos_ovrl")}
    print(f"{vid}: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")

json.dump(results,
          open(REPO / "artifacts/caedral-voice-v1/tech-fix-audit.json", "w"),
          indent=2)
