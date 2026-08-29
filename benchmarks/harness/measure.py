"""Timing + memory measurement primitives.

Design constraints:
- batch-1 CPU synthesis is the target workload; measurements must be
  reproducible from a cold process and a warm process alike;
- RSS sampling threads starve on the GIL during large single allocations,
  so on Windows we ALSO read the kernel's exact PeakWorkingSetSize counter;
- results are plain dataclasses so they serialize to JSON losslessly.

Semantics of the two peak fields:
- `peak_rss_bytes`          : per-run maximum seen by the 5 ms sampler thread
                              (cross-platform, approximate under heavy GIL use);
- `native_peak_rss_bytes`   : Win32 PeakWorkingSetSize — EXACT but process-
                              LIFETIME. Valid per-run ONLY when the measured
                              code runs in a fresh process (our benchmark
                              runner spawns one process per benchmark).

Evidence-labeling discipline lives in the CALLER: this module only measures.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, asdict

import psutil

_SAMPLE_INTERVAL_S = 0.005


@dataclass
class PerfResult:
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    baseline_rss_bytes: int
    post_rss_bytes: int
    samples: int
    native_peak_rss_bytes: int | None = None
    native_post_rss_bytes: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def peak_over_baseline_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.baseline_rss_bytes)


def _sampler(proc: psutil.Process, stop: threading.Event, peak: list) -> None:
    while not stop.is_set():
        try:
            rss = proc.memory_info().rss
        except Exception:  # process gone
            break
        if rss > peak[0]:
            peak[0] = rss
        stop.wait(_SAMPLE_INTERVAL_S)


def win_process_peaks():
    """Return (peak_working_set, current_working_set) in bytes via Win32, else None."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.windll.kernel32
        # GetCurrentProcess returns the pseudo-handle (HANDLE)-1; without an
        # explicit restype ctypes truncates it to c_int and the zero-extension
        # of the 32-bit register makes the kernel reject the handle on x64.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = k32.GetCurrentProcess()
        psapi_fn = getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wintypes.DWORD]
        ok = False
        try:  # Windows 7+ exposes this via kernel32
            ok = bool(k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb))
        except AttributeError:
            pass
        if not ok and psapi_fn is not None:
            psapi_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wintypes.DWORD]
            ok = bool(psapi_fn(handle, ctypes.byref(pmc), pmc.cb))
        if ok:
            return int(pmc.PeakWorkingSetSize), int(pmc.WorkingSetSize)
    except Exception:
        pass
    return None


def measure(fn, *args, collect_gc: bool = True, **kwargs):
    """Run fn(*args) once under wall/CPU/RSS instrumentation.

    Returns (fn_result, PerfResult).
    """
    proc = psutil.Process()
    if collect_gc:
        gc.collect()
    baseline_rss = proc.memory_info().rss
    ct0 = proc.cpu_times()
    t0 = time.perf_counter()

    stop = threading.Event()
    peak = [baseline_rss]
    sampler = threading.Thread(target=_sampler, args=(proc, stop, peak), daemon=True)
    sampler.start()
    error = None
    result = None
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # surface after metrics are finalized
        error = exc
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        wall = time.perf_counter() - t0
        ct1 = proc.cpu_times()
        cpu = (ct1.user - ct0.user) + (ct1.system - ct0.system)
        post_rss = proc.memory_info().rss

    native = win_process_peaks()
    perf = PerfResult(
        wall_seconds=wall,
        cpu_seconds=cpu,
        peak_rss_bytes=peak[0],
        baseline_rss_bytes=baseline_rss,
        post_rss_bytes=post_rss,
        samples=max(1, int(wall / _SAMPLE_INTERVAL_S)),
        native_peak_rss_bytes=native[0] if native else None,
        native_post_rss_bytes=native[1] if native else None,
    )
    if error is not None:
        raise error
    return result, perf


def rtf(wall_seconds: float, audio_seconds: float) -> float:
    """Real-time factor: <1.0 means faster than real time."""
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be > 0")
    return wall_seconds / audio_seconds
