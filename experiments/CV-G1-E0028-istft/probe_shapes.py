"""E0028 step 1: probe runtime shapes of the iSTFT subgraph.

Builds an instrumented COPY of the graph (original model untouched) with
every interesting pre/post-iSTFT tensor exposed as a graph output, runs one
inference, and dumps shapes + stats to probe_shapes.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
INSTRUMENTED = EXP / "instrumented.onnx"

PROBE_TENSORS = [
    "/decoder/decoder/generator/Exp_output_0",          # magnitude (pre-exp? post-exp)
    "/decoder/decoder/generator/Sin_output_0",          # phase input to cos/sin
    "/decoder/decoder/generator/istft/stft/Mul_output_0",
    "/decoder/decoder/generator/istft/stft/Mul_1_output_0",
    "/decoder/decoder/generator/istft/stft/Concat_output_0",   # packed real|imag spec  <-- CUT POINT
    "/decoder/decoder/generator/istft/stft/ConvTranspose_output_0",
    "/decoder/decoder/generator/istft/stft/Squeeze_1_output_0",
    "/decoder/decoder/generator/istft/stft/Slice_1_output_0",
    "/decoder/decoder/generator/istft/stft/Div_output_0",
    "/decoder/decoder/generator/istft/stft/Reshape_1_output_0",
    "/decoder/decoder/generator/istft/stft/ScatterND_output_0",
    "/decoder/decoder/generator/istft/stft/Mul_4_output_0",
    "/decoder/decoder/generator/istft/stft/Slice_3_output_0",
    "/decoder/decoder/generator/istft/stft/Slice_4_output_0",
    "audio",
]


def build_instrumented():
    m = onnx.load(str(MODEL))
    g = m.graph
    existing = {o.name for o in g.output}
    vt = onnx.TensorProto.FLOAT
    for name in PROBE_TENSORS:
        if name in existing:
            continue
        out = g.output.add()
        out.name = name
        out.type.tensor_type.elem_type = vt
    onnx.save(m, str(INSTRUMENTED))


def main():
    if not INSTRUMENTED.exists():
        build_instrumented()
        print(f"built {INSTRUMENTED}")

    import kokoro_onnx

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = rt.InferenceSession(str(INSTRUMENTED), sess_options=opts,
                               providers=["CPUExecutionProvider"])
    k = kokoro_onnx.Kokoro.from_session(
        sess, str(REPO / "models/kokoro/voices-v1.0.bin"))

    text = "Hello. This is the first generation of Caedral Voice."
    phonemes = k.tokenizer.phonemize(text)
    tokens = k.tokenizer.tokenize(phonemes)
    style = k.get_voice_style("af_heart")
    style_row = np.asarray(style[min(len(tokens), len(style)) - 1],
                           dtype=np.float32).reshape(1, -1)
    feed = {
        "tokens": np.array([[0, *tokens, 0]], dtype=np.int64),
        "style": style_row,
        "speed": np.array([1.0], dtype=np.float32),
    }
    t0 = time.perf_counter()
    outs = sess.run(PROBE_TENSORS, feed)
    wall = time.perf_counter() - t0

    report = {"wall_s_with_probes": round(wall, 3), "tensors": {}}
    for name, arr in zip(PROBE_TENSORS, outs):
        a = np.asarray(arr)
        entry = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None,
            "absmax": float(np.abs(a).max()) if a.size else None,
        }
        report["tensors"][name] = entry
        print(f"{name.split('/')[-1]:35s} {str(list(a.shape)):20s} "
              f"absmax={entry['absmax']}")

    # audio vs Slice_4 relationship: is final Gather just a squeeze?
    audio = np.asarray(outs[PROBE_TENSORS.index("audio")]).ravel()
    s4 = np.asarray(outs[PROBE_TENSORS.index(
        "/decoder/decoder/generator/istft/stft/Slice_4_output_0")])
    report["audio_len"] = int(audio.size)
    report["slice4_shape"] = list(s4.shape)
    report["slice4_ravel_absmax"] = float(np.abs(s4).max())
    d = np.abs(s4.ravel()[:audio.size] - audio).max()
    report["audio_vs_slice4_maxdiff"] = float(d)
    print(f"audio len={audio.size} slice4={s4.shape} maxdiff={d}")

    (EXP / "probe_shapes.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("wrote probe_shapes.json")


if __name__ == "__main__":
    main()
