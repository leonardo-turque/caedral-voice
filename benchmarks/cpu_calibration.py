"""CPU calibration microbenchmarks for architecture cost modeling.

Measures what an 80M-param-class model can actually expect from this silicon:
- GEMV (memory-bandwidth-bound): decoder/token-streaming regime
- GEMM small-batch (compute-bound-ish): NAR parallel passes regime
- Elementwise/memory-scan: mel/activation transforms

All results MEASURED on this machine; feed docs/research/ACOUSTIC-ARCHITECTURES.md
cost equations instead of paper assumptions.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.harness.system_info import capture  # noqa: E402

RESULTS = REPO / "benchmarks" / "results"


def bench(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    rng = np.random.default_rng(42)
    out = {"machine": capture(),
           "created": datetime.now(timezone.utc).isoformat(),
           "note": "numpy default BLAS threading; per-thread sweep deferred"}

    # --- GEMV: memory-bound (weights streamed once per call) ---
    for n in (1024, 4096):
        W = rng.standard_normal((n, n), dtype=np.float32) * 0.02
        x = rng.standard_normal(n, dtype=np.float32)
        t = bench(lambda: W @ x)
        bytes_moved = W.nbytes + x.nbytes + n * 4
        out[f"gemv_{n}x{n}_ms"] = round(t * 1e3, 3)
        out[f"gemv_{n}x{n}_GBps"] = round(bytes_moved / t / 1e9, 2)

    # --- GEMM: NAR-style wide parallel pass ---
    for m, k, n in ((256, 512, 512), (512, 512, 2048), (64, 512, 4096)):
        A = rng.standard_normal((m, k), dtype=np.float32) * 0.02
        B = rng.standard_normal((k, n), dtype=np.float32) * 0.02
        t = bench(lambda: A @ B)
        flops = 2 * m * k * n
        out[f"gemm_{m}x{k}x{n}_ms"] = round(t * 1e3, 3)
        out[f"gemm_{m}x{k}x{n}_GFLOPS"] = round(flops / t / 1e9, 2)

    # --- elementwise scan ---
    v = rng.standard_normal(8_000_000, dtype=np.float32)
    t = bench(lambda: np.tanh(v))
    out["tanh_8M_ms"] = round(t * 1e3, 3)
    out["tanh_8M_GBps"] = round(v.nbytes * 2 / t / 1e9, 2)

    # --- derived: what does an 80M NAR forward cost at these rates? ---
    if "gemv_4096x4096_GBps" in out:
        gbps = out["gemv_4096x4096_GBps"]
        out["derived_80M_fp32_weight_stream_ms"] = round(80e6 * 4 / gbps / 1e6, 2)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"cpu-calibration-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
