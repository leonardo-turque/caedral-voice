"""Sanity tests for the measurement harness — loose bounds, no flaky timing asserts."""
import platform
import time

import psutil
import pytest

from benchmarks.harness.measure import measure, rtf, win_process_peaks


def test_measure_sleep_wall_time():
    result, perf = measure(time.sleep, 0.2)
    assert 0.15 <= perf.wall_seconds <= 1.5
    assert perf.samples >= 1
    assert result is None


def test_measure_detects_allocation():
    # Keep the buffer referenced so post-run RSS is deterministic
    # (a transient in-function peak can hide behind the GIL — see measure.py docs).
    result, perf = measure(lambda: bytearray(64 * 1024 * 1024))
    assert len(result) == 64 * 1024 * 1024
    live_rss = psutil.Process().memory_info().rss
    assert live_rss > perf.baseline_rss_bytes + 32 * 1024 * 1024
    native = win_process_peaks()
    if platform.system() == "Windows" and native is not None:
        assert perf.native_peak_rss_bytes >= perf.baseline_rss_bytes + 32 * 1024 * 1024


def test_win_peaks_available_on_windows():
    native = win_process_peaks()
    if platform.system() == "Windows":
        assert native is not None
        peak, current = native
        assert peak >= current > 0
    else:
        assert native is None or native[0] >= 0


def test_rtf_math():
    assert rtf(wall_seconds=0.5, audio_seconds=2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        rtf(1.0, 0.0)


def test_measure_reraises_errors():
    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        measure(boom)
