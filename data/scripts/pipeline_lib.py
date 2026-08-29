"""Reusable, testable building blocks for the Phase D preprocessing pipeline.

Every function here is deterministic and side-effect free (except the *_wav
writers). The CLI entry points (`preprocess.py`, `validate_manifest.py`,
`download_ljspeech.py`) are thin wrappers around this library so unit tests
can exercise the DSP/filtering logic directly on synthetic arrays.

Pipeline order follows docs/data/DATASET-SURVEY.md ("Preprocessing best
practices"): ingest/verify -> resample -> loudness normalize -> DC/high-pass
-> silence trim -> clipping/noise rejection -> duration filters -> dedup
-> freeze manifests.

Constraints honored:
- Deterministic: no RNG, no timestamps, sorted iteration at call sites.
- Raw stays immutable: processing functions never write to data/raw.
- Deps limited to stdlib + numpy/scipy/soundfile/pyloudnorm (soxr optional,
  scipy fallback).
"""

from __future__ import annotations

import hashlib
import math
import unicodedata

import numpy as np
import soundfile as sf

try:  # survey practice #2: soxr-quality resampler preferred
    import soxr as _soxr
except ImportError:  # pragma: no cover - exercised only without soxr
    _soxr = None

from scipy import signal as _signal

import pyloudnorm as _pyln

# ---------------------------------------------------------------------------
# Constants (single source of truth for defaults used across the toolkit)
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"

TARGET_SR_DEFAULT = 22050
LUFS_TARGET_DEFAULT = -23.0          # EBU R128 integrated target (survey #3)
TRUE_PEAK_DBFS_LIMIT = -1.0          # TP <= -1 dBTP approximated by sample peak
CLIP_ABS_THRESHOLD = 0.99            # |x| >= 0.99 FS counts as a clipped sample
CLIP_REJECT_FRAC_DEFAULT = 0.001     # survey #6: drop >~0.1% clipped samples
MIN_DURATION_S_DEFAULT = 0.3         # mission default (survey allows 0.5)
MAX_DURATION_S_DEFAULT = 15.0        # survey #9
HIGHPASS_HZ_DEFAULT = 75.0           # survey #4: 70-80 Hz rumble removal
TRIM_TOP_DB_DEFAULT = 40.0           # energy headroom below peak frame
TRIM_FRAME_MS = 25.0
TRIM_HOP_MS = 10.0
TRIM_PAD_MS = 50.0                   # survey #5: 10-50 ms padding kept
PEAK_CEIL = 10.0 ** (TRUE_PEAK_DBFS_LIMIT / 20.0)

# Reason codes emitted into <name>.rejected.jsonl
REASON_CORRUPT = "CORRUPT_AUDIO"
REASON_MISSING_TRANSCRIPT = "MISSING_TRANSCRIPT"
REASON_TOO_SHORT = "TOO_SHORT"
REASON_TOO_LONG = "TOO_LONG"
REASON_CLIPPED = "CLIPPED"
REASON_TOO_QUIET = "TOO_QUIET"
REASON_ALL_SILENCE = "ALL_SILENCE"
REASON_DUPLICATE = "DUPLICATE_TRANSCRIPT"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file (works for multi-GB archives)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Transcript utilities
# ---------------------------------------------------------------------------

def normalize_transcript(text: str) -> str:
    """Deterministic normalization used for dedup hashes and reports.

    NFKC fold -> casefold -> keep alphanumerics/space/apostrophe ->
    collapse whitespace. Punctuation variants therefore collide, which is
    what we want for duplicate detection.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept = []
    for ch in folded:
        if ch.isalnum() or ch in (" ", "'"):
            kept.append(ch)
        else:
            kept.append(" ")
    return " ".join("".join(kept).split())


def transcript_nonascii_stats(text: str) -> tuple[int, bool]:
    """Return (#non-ascii chars in text, whether text contains any)."""
    bad = [ch for ch in text if ord(ch) > 127]
    return len(bad), bool(bad)


# ---------------------------------------------------------------------------
# Audio DSP
# ---------------------------------------------------------------------------

def load_wav_mono(path) -> tuple[np.ndarray, int]:
    """Load any soundfile-readable file as float32 mono + sample rate.

    Raises (soundfile errors, ValueError) on corrupt input; callers turn that
    into a CORRUPT_AUDIO rejection. Raw file is opened read-only only.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        return x.mean(axis=1).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x, dtype=np.float64)).astype(np.float32)


def highpass_filter(x: np.ndarray, sr: int, cutoff_hz: float = HIGHPASS_HZ_DEFAULT,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass (survey #4: ~70-80 Hz).

    Falls back to the input unchanged for signals too short to filter.
    """
    nyq = sr / 2.0
    wn = cutoff_hz / nyq
    if not (0.0 < wn < 1.0) or x.size <= 32:
        return x
    sos = _signal.butter(order, wn, btype="highpass", output="sos")
    return _signal.sosfiltfilt(sos, x).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """soxr HQ resample with a deterministic scipy.poly fallback."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    if _soxr is not None:
        y = _soxr.resample(np.asarray(x, dtype=np.float32), sr_in, sr_out,
                           quality="HQ")
        return np.asarray(y, dtype=np.float32)
    gcd = math.gcd(int(sr_in), int(sr_out))
    up, down = sr_out // gcd, sr_in // gcd
    y = _signal.resample_poly(np.asarray(x, dtype=np.float64), up, down)
    return y.astype(np.float32)


def clipped_fraction(x: np.ndarray,
                     threshold: float = CLIP_ABS_THRESHOLD) -> float:
    """Fraction of samples at or beyond `threshold` FS (mission: >0.99 flag)."""
    if x.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(x) >= threshold) / x.size)


def measure_lufs(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness.

    Returns -inf for digital silence AND for signals shorter than the meter's
    gating block (pyloudnorm would raise); callers treat non-finite as
    "unmeterable" -> TOO_QUIET rejection downstream.
    """
    meter = _pyln.Meter(int(sr))
    min_len = int(math.ceil(meter.block_size * sr))
    if x.size < min_len:
        return float("-inf")
    value = float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    return value


def loudness_normalize(x: np.ndarray, sr: int,
                       target_lufs: float = LUFS_TARGET_DEFAULT,
                       max_passes: int = 2) -> tuple[np.ndarray, float, float, bool]:
    """Two-pass loudnorm to `target_lufs` integrated, sample-peak-limited.

    Returns (y, lufs_before, lufs_after, peak_limited). Static-gain limiting:
    if the normalized peak would exceed TRUE_PEAK_DBFS_LIMIT the whole signal
    is scaled back down (no dynamic processing -> deterministic).
    """
    xa = np.asarray(x, dtype=np.float64)
    lufs_before = measure_lufs(xa, sr)
    if not math.isfinite(lufs_before):
        return (np.asarray(x, dtype=np.float32), lufs_before, float("nan"), False)
    y = xa * (10.0 ** ((target_lufs - lufs_before) / 20.0))
    # Second pass corrects meter/gain rounding (survey #3: two-pass loudnorm).
    for _ in range(max(0, max_passes - 1)):
        lufs_mid = measure_lufs(y, sr)
        if not math.isfinite(lufs_mid):
            break
        y = y * (10.0 ** ((target_lufs - lufs_mid) / 20.0))
    peak_limited = False
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > PEAK_CEIL:
        y = y * (PEAK_CEIL / peak)
        peak_limited = True
    lufs_after = measure_lufs(y, sr) if peak_limited else target_lufs
    return y.astype(np.float32), lufs_before, lufs_after, peak_limited


def frame_rms(x: np.ndarray, sr: int,
              frame_ms: float = TRIM_FRAME_MS,
              hop_ms: float = TRIM_HOP_MS) -> np.ndarray:
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n_frames = 1 + max(0, (x.size - frame) // hop) if x.size >= frame else 0
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1))


def trim_silence(x: np.ndarray, sr: int,
                 top_db: float = TRIM_TOP_DB_DEFAULT,
                 pad_ms: float = TRIM_PAD_MS,
                 frame_ms: float = TRIM_FRAME_MS,
                 hop_ms: float = TRIM_HOP_MS) -> np.ndarray:
    """Energy-based leading/trailing silence trim, deterministic.

    Frames whose RMS is more than `top_db` below the loudest frame count as
    silence. Keeps `pad_ms` of padding on each voiced edge. Returns an empty
    array when no frame rises above the floor (all-silence input).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    rms = frame_rms(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if rms.size == 0:
        return x[:0]
    peak = float(rms.max())
    if peak <= 0.0:
        return x[:0]
    floor = peak * (10.0 ** (-abs(top_db) / 20.0))
    active = np.nonzero(rms > floor)[0]
    if active.size == 0:
        return x[:0]
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    start = int(active[0] * hop)
    end_frame = int(active[-1])
    end = min(x.size, end_frame * hop + int(round(sr * frame_ms / 1000.0)))
    pad = int(round(sr * pad_ms / 1000.0))
    start = max(0, start - pad)
    end = min(x.size, end + pad)
    if end <= start:
        return x[:0]
    return x[start:end]


def duration_seconds(n_samples: int, sr: int) -> float:
    return float(n_samples) / float(sr)


def duration_ok(duration_s: float, min_s: float = MIN_DURATION_S_DEFAULT,
                max_s: float = MAX_DURATION_S_DEFAULT) -> tuple[bool, str]:
    if duration_s < min_s:
        return False, REASON_TOO_SHORT
    if duration_s > max_s:
        return False, REASON_TOO_LONG
    return True, ""
