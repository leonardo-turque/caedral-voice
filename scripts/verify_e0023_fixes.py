"""Verify E0023 TN fixes reach the production path for technical text."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
from caedral_voice.text.kokoro_map import caedral_tokens_to_kokoro

e = CaedralVoiceKokoroEngine(
    model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"), threads=4)
e.warmup()

tests = [
    ("AVX-512 sentence", "The CPU supports AVX2 instructions but not AVX-512."),
    ("RTX 5090", "RTX 5090 costs $29.99 at 3.5 GHz with 10 GB VRAM."),
    ("Python/CUDA/HTTP", "Python 3.13 and CUDA 12.8 support HTTP/2."),
    ("URL/email", "Visit https://caedral.com or email support@caedral.com"),
]

for label, text in tests:
    ph, sk = e.phonemize(text, use_cache=False)
    print(f"{label}:")
    print(f"  normalized: {e.frontend.process(text).text_normalized[:80]}...")
    print(f"  mapped: {ph[:100]}...")
    print(f"  skipped: {sk}")
    print()
