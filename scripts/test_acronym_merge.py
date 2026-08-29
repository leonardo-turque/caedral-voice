import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
import classify_preview as cp

text = "The CPU supports AVX2 instructions but not AVX-512."
e = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e.warmup()
ph, sk = e.phonemize(text, use_cache=False)
print("FIXED phonemes:", repr(ph))

a = e.synth(text, voice_name="am_echo", speed=1.0)
out = REPO / "artifacts" / "rescue-comparison" / "step-2000"
out.mkdir(parents=True, exist_ok=True)
import soundfile as sf
sf.write(str(out / "rescue-m2-tech-fixed.wav"), a, 24000)
r = cp.classify_file(str(out / "rescue-m2-tech-fixed.wav"), text)
print(f"m2 technical: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")
