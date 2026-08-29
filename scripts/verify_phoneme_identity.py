"""Verify: same text -> identical phoneme string across all 4 voices?"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine

e = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e.warmup()

text = "The CPU supports AVX2 instructions but not AVX-512."
phonemes = {}
for vid in ("caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"):
    ph, sk = e.phonemize(text, use_cache=False)
    phonemes[vid] = ph
    print(f"{vid}: {ph[:60]}...")

all_same = len(set(phonemes.values())) == 1
print(f"\nAll phoneme strings identical: {all_same}")
