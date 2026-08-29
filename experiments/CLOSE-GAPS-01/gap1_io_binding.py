"""CLOSE-GAPS-01 GAP 1: IO binding vs regular session.run() for kokoro-v1.0.onnx.

Previous attempts hit a shape mismatch on style: voices bin gives (510, 1, 256)
per voice; the graph wants (1, 256). Correct extraction:
    vd = k.voices["af_sky"]          # (510, 1, 256)
    style = np.array(vd[0], dtype=np.float32).reshape(1, 256)

Method: ONE InferenceSession (ORT_ENABLE_ALL @ intra_op=4), same feeds, two
paths timed in interleaved rounds to cancel thermal drift:
    A) sess.run(None, feeds)                      [regular]
    B) sess.run_with_bindings(io) + copy to CPU   [IO binding; production needs
                                                   the numpy array back]
Gate: <5% improvement -> KILL; >=5% -> KEEP + quality check (outputs must match).
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"

REPS = 10          # interleaved A/B rounds (each round = one A wall + one B wall)
THREADS = 4
TOKEN_LEN = 100    # typical sentence token count


def main() -> None:
    import onnxruntime as ort
    import kokoro_onnx

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(MODEL), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0

    ins = {i.name: i for i in sess.get_inputs()}
    out_name = sess.get_outputs()[0].name
    assert set(ins) == {"tokens", "style", "speed"}, list(ins)

    k = kokoro_onnx.Kokoro.from_session(sess, str(VOICES))
    vd = k.voices["af_sky"]
    assert tuple(vd.shape) == (510, 1, 256), vd.shape

    tokens = np.array([[0, *tokens_from_rng(), 0]], dtype=np.int64)
    style = np.array(vd[0], dtype=np.float32).reshape(1, 256)
    speed = np.array([1.0], dtype=np.float32)
    feeds = {"tokens": tokens, "style": style, "speed": speed}

    # ---- IO binding setup ----
    io = sess.io_binding()
    io.bind_ortvalue_input("tokens",
                           ort.OrtValue.ortvalue_from_numpy(tokens, "cpu", 0))
    io.bind_ortvalue_input("style",
                           ort.OrtValue.ortvalue_from_numpy(style, "cpu", 0))
    io.bind_ortvalue_input("speed",
                           ort.OrtValue.ortvalue_from_numpy(speed, "cpu", 0))
    io.bind_output(out_name, "cpu", 0)

    # warmup both paths
    ref = sess.run(None, feeds)[0]
    for _ in range(3):
        sess.run(None, feeds)
    for _ in range(3):
        sess.run_with_iobinding(io)
        io.copy_outputs_to_cpu()

    # interleaved A/B timing
    walls_run, walls_bind, walls_bind_copy = [], [], []
    for _ in range(REPS):
        t = time.perf_counter()
        sess.run(None, feeds)
        walls_run.append(time.perf_counter() - t)

        t = time.perf_counter()
        sess.run_with_iobinding(io)
        w = time.perf_counter() - t
        arr = io.copy_outputs_to_cpu()[0]
        walls_bind.append(w)
        walls_bind_copy.append(time.perf_counter() - t)

    bound_out = np.asarray(arr).ravel()
    run_out = np.asarray(sess.run(None, feeds)[0]).ravel()

    m_run = statistics.median(walls_run)
    m_bind = statistics.median(walls_bind)
    m_bindc = statistics.median(walls_bind_copy)
    gain_runonly = 100 * (m_run - m_bind) / m_run
    gain_prodfair = 100 * (m_run - m_bindc) / m_run

    maxdiff = float(np.abs(bound_out[:run_out.size] -
                           run_out[:bound_out.size]).max())

    keep = gain_prodfair >= 5.0
    verdict = ("KEEP" if keep else
               f"KILL (<5% improvement: prod-fair {gain_prodfair:.2f}%")
    if keep and maxdiff > 1e-4:
        verdict = f"KILL (quality: output maxdiff {maxdiff})"
    report = {
        "model": MODEL.name, "load_s": round(load_s, 2),
        "threads": THREADS, "token_len": TOKEN_LEN,
        "input_shapes": {"tokens": list(tokens.shape),
                         "style": list(style.shape),
                         "speed": list(speed.shape)},
        "reps_interleaved": REPS,
        "median_regular_run_s": round(m_run, 4),
        "median_io_bound_s": round(m_bind, 4),
        "median_io_bound_plus_copy_s": round(m_bindc, 4),
        "improvement_pct_bound_only": round(gain_runonly, 2),
        "improvement_pct_prod_fair": round(gain_prodfair, 2),
        "quality_maxdiff_bound_vs_run": maxdiff,
        "verdict": verdict,
    }
    print("RESULT_JSON:" + json.dumps(report))


def tokens_from_rng() -> list[int]:
    rng = np.random.default_rng(42)
    return rng.integers(1, 100, size=TOKEN_LEN).tolist()


if __name__ == "__main__":
    main()
