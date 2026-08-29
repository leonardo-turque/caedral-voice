"""CV-G1-E0023-tn-integration: verify CaedralVoiceKokoroEngine.phonemize()
against the five TN acceptance strings.

Runs the REAL production frontend path (TextFrontend("lexicon+g2pen") inside
CaedralVoiceKokoroEngine) so the vocab used by caedral_tokens_to_kokoro is the
model's own embedded vocab, and prints normalized text + mapped phoneme stream
+ skipped symbols per string. Results also land in verify_results.json.

Read-only w.r.t. kokoro_engine.py: we call phonemize(), we do not modify it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine  # noqa: E402

MODEL = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "kokoro" / "voices-v1.0.bin"

TEST_STRINGS = [
    "The CPU supports AVX2 instructions but not AVX-512.",
    "RTX 5090 costs $29.99 at 3.5 GHz with 10 GB VRAM.",
    "Python 3.13 and CUDA 12.8 support HTTP/2.",
    "Visit https://caedral.com or email support@caedral.com",
    "GPT-5 achieves 99.9% on 2026-08-23 benchmarks.",
]

OUT_JSON = Path(__file__).resolve().parent / "verify_results.json"


def main() -> None:
    t_load0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(MODEL, VOICES, threads=2)
    print(f"engine load {time.perf_counter() - t_load0:.2f}s "
          f"(vocab {len(engine._vocab)} symbols)")

    rows = []
    for s in TEST_STRINGS:
        norm = engine.frontend.process(s).text_normalized
        t0 = time.perf_counter()
        phonemes, skipped = engine.phonemize(s, use_cache=False)
        dt_ms = (time.perf_counter() - t0) * 1000
        row = {
            "input": s,
            "normalized": norm,
            "phonemes": phonemes,
            "skipped": sorted(set(skipped)),
            "phonemize_ms": round(dt_ms, 2),
        }
        rows.append(row)
        print("=" * 78)
        print(f"IN   : {s}")
        print(f"NORM : {norm}")
        print(f"PHON : {phonemes}")
        print(f"SKIP : {skipped}")

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print("=" * 78)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
