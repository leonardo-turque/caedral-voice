"""Test ORT session configurations targeting Conv-heavy generator performance."""
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import psutil

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))

text = "Hello. This is the first generation of Caedral Voice."

def bench_session(opts_name: str, opts: rt.SessionOptions, n_reps: int = 10):
    from kokoro_onnx import Kokoro
    session = rt.InferenceSession(
        str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        sess_options=opts, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(session, "models/kokoro/voices-v1.0.bin")

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

    # Warmup 5
    for _ in range(5):
        session.run(None, feed)

    # Measure 10
    times = []
    for _ in range(n_reps):
        t = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t)

    audio_s = 3.4
    rtfs = sorted(t / audio_s for t in times)
    rss = psutil.Process().memory_info().rss // 1e6

    result = {
        "config": opts_name,
        "rtf_median": round(rtfs[5], 4),
        "rtf_min": round(rtfs[0], 4),
        "rss_mb": rss,
    }
    print(json.dumps(result))
    return result


results = []

# Baseline: current production config
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 1
results.append(bench_session("baseline_enable_all_4thr", opts))

# Test 1: Disable memory pattern (may reduce allocation overhead for dynamic shapes)
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.enable_mem_pattern = False
results.append(bench_session("no_mem_pattern", opts))

# Test 2: Enable arena explicitly + extend strategy
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.enable_cpu_mem_arena = True
opts.add_session_config_entry("session.memory.arena.extend_strategy", "kSameAsRequested")
results.append(bench_session("arena_requested", opts))

# Test 3: Sequential mode with more aggressive optimization search
opts = rt.SessionOptions()
opts.optimized_model_filepath = str(REPO / "models" / "kokoro" / "optimized-gen.onnx")
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
results.append(bench_session("sequential_optimized", opts))

# Test 4: Disable execution mode optimization entirely (pure sequential)
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
opts.intra_op_num_threads = 4
opts.log_severity_level = 3
results.append(bench_session("basic_opt_4thr", opts))

# Test 5: ORT_DISABLE_ALL (rawest possible — shows optimizer contribution)
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.intra_op_num_threads = 4
results.append(bench_session("disable_all_4thr", opts))

# Test 6: 6 threads with full opt
opts = rt.SessionOptions()
opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 6
results.append(bench_session("enable_all_6thr", opts))


print("\n=== RANKED BY RTF MEDIAN ===")
for r in sorted(results, key=lambda x: x["rtf_median"]):
    print(f"{r['rtf_median']:.4f} | {r['config']} | RSS {r['rss_mb']}MB")
