"""Repair attempts for the onnxsim output that fails ORT load
(Gemm_11 ShapeInferenceError 768 vs 128).

Variants tried, each gated by an actual ORT session load:
    V1: strip all graph.value_info from kokoro-opt.onnx (stale annotations)
    V2: re-simplify from scratch with risky fusion passes disabled
        (fuse_consecutive_transposes, eliminate_consecutive_transpose,
         eliminate_nop_transpose, merge_duplicated...)
First variant that loads wins; writes artifacts/optimized/kokoro-opt.onnx.
"""
from __future__ import annotations

import time
from pathlib import Path

import onnx
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments/CV-G1-E0026-graph-opt"
OPT = REPO / "artifacts/optimized/kokoro-opt.onnx"


def loads(path, threads=2) -> tuple[bool, str]:
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        # smoke run: 8 tokens
        import numpy as np
        tok = np.array([[0, 5, 12, 34, 56, 78, 90, 0]], dtype=np.int64)
        style = np.zeros((1, 256), dtype=np.float32)
        out = sess.run(["audio"], {"tokens": tok, "style": style,
                                   "speed": np.array([1.0], np.float32)})
        return True, f"ok {time.perf_counter()-t0:.1f}s len={out[0].size}"
    except Exception as exc:                    # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def strip_value_info(m):
    del m.graph.value_info[:]
    return m


def resim_skip(skipped):
    import onnxsim
    m = onnx.load(str(REPO / "models/kokoro/kokoro-v1.0.onnx"))
    msim, ok = onnxsim.simplify(m, check_n=0, skipped_optimizers=skipped)
    return msim


if __name__ == "__main__":
    print("V0 current opt:", loads(OPT), flush=True)

    m = onnx.load(str(OPT))
    strip_value_info(m)
    v1 = OPT.with_name("kokoro-opt-v1-stripvi.onnx")
    onnx.save(m, str(v1))
    ok1, msg1 = loads(v1)
    print("V1 strip value_info:", ok1, msg1, flush=True)

    ok2 = msg2 = None
    if not ok1:
        skips = ["fuse_consecutive_transposes",
                 "eliminate_consecutive_transpose",
                 "eliminate_nop_transpose"]
        try:
            msim = resim_skip(skips)
            v2 = OPT.with_name("kokoro-opt-v2-nofuse.onnx")
            onnx.save(msim, str(v2))
            ok2, msg2 = loads(v2)
            print("V2 no-transpose-fusion:", ok2, msg2, flush=True)
        except Exception as exc:                # noqa: BLE001
            ok2, msg2 = False, f"{type(exc).__name__}: {exc}"
            print("V2 failed:", msg2, flush=True)

    if ok1:
        v1.replace(OPT)
        print("PROMOTED V1 -> kokoro-opt.onnx", flush=True)
    elif ok2:
        v2.replace(OPT)
        print("PROMOTED V2 -> kokoro-opt.onnx", flush=True)
    else:
        print("NO VARIANT LOADS - onnxsim path is dead for this graph",
              flush=True)
