"""Build the CONSERVATIVE variant: no onnxsim rewrite, just manual
initializer dedup + dead pruning from optimize.py. Baseline semantics
bit-for-bit (values untouched, only graph topology cleaned trivially)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx
from optimize import dedup_initializers, prune_dead

REPO = HERE.parents[1]
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
OUT = REPO / "artifacts/optimized/kokoro-conservative.onnx"

m = onnx.load(str(MODEL))
n_dedup = dedup_initializers(m.graph)
pn, pi = prune_dead(m.graph)
OUT.parent.mkdir(parents=True, exist_ok=True)
onnx.save(m, str(OUT))
print(json.dumps({
    "out": str(OUT.relative_to(REPO)),
    "nodes": len(m.graph.node),
    "initializers": len(m.graph.initializer),
    "duplicates_removed": n_dedup,
    "dead_nodes_removed": pn,
    "unused_inits_removed": pi,
    "mb": round(OUT.stat().st_size / 2**20, 2),
}))
