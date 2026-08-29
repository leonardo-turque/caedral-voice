"""ORT operator-level profiling: where does CPU time ACTUALLY go?"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[1]
os_cwd = str(REPO)

# Load with profiling enabled
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.enable_profiling = True
opts.profile_file_prefix = str(REPO / "artifacts" / "benchmarks" / "ort-profile")

session = rt.InferenceSession(
    str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)

from kokoro_onnx import Kokoro
k = Kokoro.from_session(session, "models/kokoro/voices-v1.0.bin")

text = "Hello. This is the first generation of Caedral Voice."
phonemes = k.tokenizer.phonemize(text)
ids = k.tokenizer.tokenize(phonemes)
voice_data = k.voices["af_sky"]
style = np.array(voice_data[0], dtype=np.float32).reshape(1, -1).astype(np.float32)
speed = np.array([1.0], dtype=np.float32)

feed = {
    session.get_inputs()[0].name: np.array([[0] + ids + [0]], dtype=np.int64),
    session.get_inputs()[1].name: style,
    session.get_inputs()[2].name: speed,
}

# Warmup
for _ in range(5):
    session.run(None, feed)

# Profiled runs
for _ in range(10):
    session.run(None, feed)

profile_file = session.end_profiling()
print(f"profile written: {profile_file}")

# Parse the Chrome trace format profiling output
import json as _json
events = []
with open(profile_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip().rstrip(",")
        if line.startswith("{") and '"cat"' in line:
            try:
                events.append(_json.loads(line))
            except Exception:
                pass

# Aggregate by operator name
op_times = {}
total_time = 0
for ev in events:
    name = ev.get("name", "")
    dur = ev.get("dur", 0)  # microseconds
    cat = ev.get("cat", "")
    if dur > 0 and cat in ("Kernel", "Node"):
        op_times[name] = op_times.get(name, 0) + dur
        total_time += dur

if total_time > 0:
    ranked = sorted(op_times.items(), key=lambda x: -x[1])
    print(f"\nTotal profiled kernel time: {total_time/1000:.1f} ms")
    print(f"\nTop 20 operators by cumulative time:")
    print(f"{'Operator':<50s} {'Time(ms)':>10s} {'%':>7s} {'Count':>6s}")
    print("-" * 80)
    count_by_op = {}
    for ev in events:
        n = ev.get("name", "")
        if n in op_times:
            count_by_op[n] = count_by_op.get(n, 0) + 1
    for name, us in ranked[:20]:
        ms = us / 1000
        pct = 100 * us / total_time if total_time else 0
        cnt = count_by_op.get(name, 0)
        print(f"{name:<50s} {ms:>10.1f} {pct:>6.1f}% {cnt:>6d}")

    # Categorize
    categories = {
        "MatMul/Gemm": ["MatMul", "Gemm", "MatMulInteger"],
        "Attention": ["Attention", "MultiHeadAttention", "SparseAttention"],
        "Conv": ["Conv", "ConvTranspose"],
        "LayerNorm": ["LayerNormalization", "InstanceNormalization"],
        "Activation": ["Relu", "Gelu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Softplus", "Mish"],
        "Elementwise": ["Add", "Mul", "Sub", "Div", "Exp", "Sqrt", "Pow"],
        "Shape/Reshape": ["Reshape", "Transpose", "Concat", "Slice", "Gather", "Unsqueeze", "Squeeze", "Split"],
        "LSTM/RNN": ["LSTM", "GRU", "RNN"],
        "STFT/iSTFT": ["STFT", "ISTFT", "DFT"],
        "Normalization": ["BatchNormalization", "GroupNormalization"],
        "Other": [],
    }
    assigned = set()
    print(f"\n=== TIME BY CATEGORY ===")
    for cat_name, ops in categories.items():
        cat_time = sum(us for name, us in op_times.items() if name in ops)
        if cat_time > 0:
            print(f"  {cat_name}: {cat_time/1000:.1f} ms ({100*cat_time/total_time:.1f}%)")
            assigned.update(ops)
    other_time = sum(us for n, us in op_times.items() if n not in assigned)
    if other_time > 0:
        print(f"  Other: {other_time/1000:.1f} ms ({100*other_time/total_time:.1f}%)")
else:
    print("No kernel events captured")
