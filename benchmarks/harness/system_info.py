"""Capture the exact hardware/software baseline for benchmark headers.

Every benchmark result file must embed the dict returned by `capture()`,
so no measurement can ever be quoted without its machine context.
"""
from __future__ import annotations

import io
import json
import platform
import sys


def capture_cpu() -> dict:
    info: dict = {"python_fallback": True}
    try:
        import cpuinfo

        raw = cpuinfo.get_cpu_info()
        info = {
            "model": raw.get("brand_raw"),
            "arch": raw.get("arch"),
            "bits": raw.get("bits"),
            "count_logical": raw.get("count"),
            "hz_advertised": raw.get("hz_actual_friendly"),
            "flags": sorted(raw.get("flags", [])),
        }
        # Derive the instruction-set summary we actually care about.
        # py-cpuinfo reports FMA3 as "fma"; AVX-VNNI may be absent from older
        # py-cpuinfo versions even on supporting CPUs -> reported as None (unknown).
        flags = set(info["flags"])
        info["isa"] = {
            "SSE4_2": ("sse4_2" in flags or "sse4.2" in flags),
            "AVX": "avx" in flags,
            "AVX2": "avx2" in flags,
            "AVX512F": any(f.startswith("avx512") for f in flags),
            "FMA3": ("fma" in flags or "fma3" in flags),
            "BMI2": "bmi2" in flags,
            # None = unknown (py-cpuinfo version lacks the leaf), never assume absence.
            "AVX_VNNI": True if "avx_vnni" in flags else None,
        }
        info.pop("python_fallback", None)
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = repr(exc)
    return info


def capture_memory() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "total_bytes": vm.total,
        "available_bytes": vm.available,
        "swap_total_bytes": sw.total,
    }


def capture_python() -> dict:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def capture_os() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def capture() -> dict:
    """Full baseline snapshot — cheap enough to call at every benchmark start."""
    return {
        "os": capture_os(),
        "cpu": capture_cpu(),
        "memory": capture_memory(),
        "python": capture_python(),
    }


if __name__ == "__main__":
    snap = capture()
    text = json.dumps(snap, indent=2, default=str)
    if len(sys.argv) > 1:
        with io.open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {sys.argv[1]}")
    print(text)
