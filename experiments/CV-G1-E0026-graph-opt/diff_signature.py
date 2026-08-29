"""Chaos-signature probe: is the static-vs-dynamic output diff a semantic
break or amplification of execution-order ulp noise?

Renders the frozen sentence through kokoro-conservative (dynamic) and
kokoro-static-L{L} on the SAME padded window, then reports diff statistics
per 1-second block across the timeline. Amplification signature: diff grows
with t (divergence accumulates through autoregressive-ish conv stack);
semantic break: uniform large diff from sample 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from equivalence import make_session, run_case          # noqa: E402

VOICES = REPO / "models/kokoro/voices-v1.0.bin"
SENTENCE = "Hello. This is the first generation of Caedral Voice."

# real tokens via frontend
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
eng = CaedralVoiceKokoroEngine(
    str(REPO / "models/kokoro/kokoro-v1.0.onnx"), str(VOICES), threads=4)
phonemes, _ = eng.phonemize(SENTENCE)
toks = eng._kokoro.tokenizer.tokenize(phonemes)
del eng

L = int(sys.argv[1]) if len(sys.argv) > 1 else 100
target = L - 2
toks_pad = (list(toks) + [0] * target)[:target]

dyn = make_session(str(REPO / "artifacts/optimized/kokoro-conservative.onnx"))
stat = make_session(str(REPO / f"artifacts/optimized/kokoro-static-L{L}.onnx"))

a, _ = run_case(dyn, toks_pad)
bb, _ = run_case(stat, toks_pad)
n = min(len(a), len(bb))
d = np.abs(a[:n].astype(np.float64) - bb[:n].astype(np.float64))

block = 24000   # 1 s
rows = []
for s in range(0, n, block):
    seg = d[s:s + block]
    rows.append({"t_s": round(s / 24000, 1),
                 "max": round(float(seg.max()), 6),
                 "mean": float(f"{seg.mean():.3e}")})

cross_1e_2 = next((r["t_s"] for r in rows if r["max"] > 1e-2), None)
out = {
    "bucket": L, "samples": n,
    "first_block_max": rows[0]["max"],
    "last_block_max": rows[-1]["max"],
    "t_first_max_gt_1e-2": cross_1e_2,
    "global_mean": float(f"{d.mean():.3e}"),
    "blocks": rows[:12],
}
(HERE / f"diff_signature_L{L}.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")
print(json.dumps(out, indent=2))
