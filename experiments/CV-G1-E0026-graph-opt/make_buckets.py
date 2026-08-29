"""CV-G1-E0026-graph-opt step 4: shape-specialized bucket variants.

Creates static-sequence-length copies of the CONSERVATIVE (bitwise-exact)
model for buckets SHORT=100 / MEDIUM=300 / LONG=600 tokens:
    1. fix input 'tokens' dim_param 'sequence_length' -> L
    2. propagate shapes via onnx.shape_inference (data_prop)
    3. NO onnxsim here: its op canonicalization provably shifts numerics on
       this graph (see equiv_opt.log), so buckets stay pure shape edits of
       an exact-semantics base -> ORT still folds/replans at load time.
Each variant is gated by an actual ORT load + smoke render.
Saves artifacts/optimized/kokoro-static-L{100,300,600}.onnx.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "artifacts/optimized/kokoro-conservative.onnx"
OUT_DIR = REPO / "artifacts/optimized"
# LONG is capped at 512: encoder.bert position_embeddings.weight is
# [512, 128], so no variant of this model can process >512 window positions.
BUCKETS = [100, 300, 512]


def loads_and_runs(path: str, L: int) -> tuple[bool, str]:
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        tok = np.zeros((1, L), dtype=np.int64)
        tok[0, 5] = 12
        out = sess.run(["audio"], {"tokens": tok,
                                   "style": np.zeros((1, 256), np.float32),
                                   "speed": np.array([1.0], np.float32)})
        return True, f"len={out[0].size}"
    except Exception as exc:                    # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


stats = {}
for L in BUCKETS:
    t0 = time.perf_counter()
    m = onnx.load(str(SRC))
    inp = next(i for i in m.graph.input if i.name == "tokens")
    dim = inp.type.tensor_type.shape.dim[1]
    dim.ClearField("dim_param")
    dim.dim_value = L
    fixed_vi = 0
    for vi in m.graph.value_info:
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and d.dim_param == "sequence_length":
                d.ClearField("dim_param")
                d.dim_value = L
                fixed_vi += 1
    try:
        m2 = onnx.shape_inference.infer_shapes(m, strict_mode=False,
                                               data_prop=True)
        inferred = True
    except Exception as exc:                    # noqa: BLE001
        m2, inferred = m, f"failed: {type(exc).__name__}"
    out = OUT_DIR / f"kokoro-static-L{L}.onnx"
    onnx.save(m2, str(out))
    ok, msg = loads_and_runs(str(out), L)
    stats[str(L)] = {
        "bucket_tokens": L, "nodes": len(m2.graph.node),
        "initializers": len(m2.graph.initializer),
        "value_info_fixed": fixed_vi,
        "shape_inference": bool(inferred) if isinstance(inferred, bool)
        else inferred,
        "loads": ok, "load_msg": msg,
        "mb": round(out.stat().st_size / 2**20, 2),
        "wall_s": round(time.perf_counter() - t0, 1),
        "file": str(out.relative_to(REPO)),
    }
    print(L, json.dumps(stats[str(L)]), flush=True)

(HERE / "bucket_stats.json").write_text(json.dumps(stats, indent=2),
                                        encoding="utf-8")
