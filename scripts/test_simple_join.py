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
print("phonemes:", repr(ph))

for voice in ("am_echo", "am_michael", "af_sky", "af_nova"):
    a = e.synth(text, voice_name=voice, speed=1.0)
    out = REPO / "artifacts" / "rescue-comparison" / "simple-join"
    out.mkdir(parents=True, exist_ok=True)
    fpath = str(out / f"{voice}-tech.wav")
    import soundfile as sf
    sf.write(fpath, a, 24000)
    r = cp.classify_file(fpath, text)
    print(f"{voice}: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")
