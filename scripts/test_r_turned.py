import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kokoro_onnx import Kokoro
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
import classify_preview as cp  # noqa: E402
import soundfile as sf  # noqa: E402

text = "The CPU supports AVX2 instructions but not AVX-512."
e = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e.warmup()

ph, sk = e.phonemize(text, use_cache=False)
ph_ɹ = ph.replace("r", "ɹ")
print("phonemes:", repr(ph_ɹ[:80]))

a = e._kokoro.create(ph_ɹ, voice="am_echo", speed=1.0,
                     lang="en-us", is_phonemes=True)
out = REPO / "artifacts" / "rescue-comparison" / "am_echo-tech-ɹ.wav"
sf.write(str(out), a, 24000)
r = cp.classify_file(str(out), text)
result = {"class": r.get("class"), "wer": r.get("min_wer"),
          "dnsmos": r.get("dnsmos_ovrl")}
print(json.dumps(result))
