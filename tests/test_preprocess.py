"""Unit tests for the Phase D preprocessing toolkit (synthetic wavs only).

Generates sine/silence/noise WAVs with numpy, drives pipeline_lib functions
directly plus end-to-end preprocess.main()/validate_manifest.main() runs on a
tiny synthetic raw dir. No network, no real datasets.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pipeline_lib as pl  # noqa: E402
import preprocess  # noqa: E402
import validate_manifest  # noqa: E402

SR = 22050


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def sine_sweep_free(freq_hz: float = 220.0, seconds: float = 1.0,
                    sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def with_silence_padding(x: np.ndarray, sr: int, pad_s: float = 0.5,
                         amplitude: float = 0.0) -> np.ndarray:
    pad = np.zeros(int(pad_s * sr), dtype=np.float32) + amplitude * 1e-4
    return np.concatenate([pad, x.astype(np.float32), pad])


def clipped_signal(seconds: float = 1.0, sr: int = SR,
                   frac_clipped: float = 0.05) -> np.ndarray:
    """Sine whose loudest `frac_clipped` fraction of samples is pinned at
    +/-1.0 FS. Tonal (no DC), so it survives high-pass and trimming."""
    x = sine_sweep_free(seconds=seconds, sr=sr, amplitude=0.9)
    k = int(frac_clipped * x.size)
    idx = np.argsort(np.abs(x))[::-1][:k]
    x[idx] = np.sign(x[idx]).astype(np.float32)
    return x.astype(np.float32)


@pytest.fixture()
def params(tmp_path):
    return {
        "target_sr": SR,
        "lufs_target": -23.0,
        "min_dur": 0.3,
        "max_dur": 15.0,
        "max_clipped_frac": 0.001,
        "clip_abs_threshold": pl.CLIP_ABS_THRESHOLD,
        "trim_top_db": pl.TRIM_TOP_DB_DEFAULT,
        "trim_pad_ms": pl.TRIM_PAD_MS,
        "highpass_hz": pl.HIGHPASS_HZ_DEFAULT,
    }


@pytest.fixture(autouse=True)
def _repo_root_is_tmp(tmp_path, monkeypatch):
    """Point repo-relative path resolution at the test sandbox."""
    monkeypatch.setattr(preprocess, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_manifest, "REPO_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# Hashing / transcripts
# ---------------------------------------------------------------------------

def test_sha256_text_is_stable():
    assert pl.sha256_text("hello") == pl.sha256_text("hello")
    assert pl.sha256_text("hello") != pl.sha256_text("Hello ")
    assert len(pl.sha256_text("")) == 64


def test_normalize_transcript_collides_punctuation_and_case():
    a = pl.normalize_transcript("Hello, WORLD!")
    b = pl.normalize_transcript("hello world")
    assert a == b
    assert pl.normalize_transcript("  it's   fine\t ") == "it's fine"


def test_nonascii_stats():
    n, has = pl.transcript_nonascii_stats("café ok")
    assert n == 1 and has is True
    n, has = pl.transcript_nonascii_stats("plain ascii")
    assert n == 0 and has is False


# ---------------------------------------------------------------------------
# DSP functions
# ---------------------------------------------------------------------------

def test_trim_silence_removes_leading_trailing_padding():
    x = with_silence_padding(sine_sweep_free(seconds=1.0), SR, pad_s=0.6)
    y = pl.trim_silence(x, SR)
    assert 0.9 < pl.duration_seconds(y.size, SR) < 1.25
    assert y.size < x.size


def test_trim_silence_all_silence_returns_empty():
    x = np.zeros(SR // 2, dtype=np.float32)
    assert pl.trim_silence(x, SR).size == 0


def test_trim_keeps_pad_margin():
    x = with_silence_padding(sine_sweep_free(seconds=0.8), SR, pad_s=0.4)
    y = pl.trim_silence(x, SR, pad_ms=50)
    voiced = int(0.8 * SR)
    assert y.size >= voiced  # never shorter than the voiced region
    assert y.size <= voiced + 2 * (int(0.05 * SR) + 3 * int(0.01 * SR))


def test_clipped_fraction_detects_hot_samples():
    clean = sine_sweep_free(amplitude=0.3)
    assert pl.clipped_fraction(clean) == pytest.approx(0.0, abs=1e-6)
    hot = clipped_signal(frac_clipped=0.10)
    frac = pl.clipped_fraction(hot)
    assert 0.09 <= frac <= 0.11
    assert pl.clipped_fraction(np.zeros(10, dtype=np.float32)) == 0.0


def test_loudness_normalize_reaches_target(params):
    quiet = sine_sweep_free(seconds=2.0, amplitude=0.005)
    y, before, after, limited = pl.loudness_normalize(
        quiet, SR, target_lufs=params["lufs_target"])
    assert before < -45.0          # starts far too quiet
    assert abs(after - (-23.0)) <= 1.5
    assert limited is False
    assert abs(float(np.max(np.abs(y)))) <= 1.0


def test_loudness_normalize_limits_peaks():
    # Quiet carrier (-23 LUFS needs a big boost) plus one hot transient:
    # the boost would push the transient past the ceiling -> static limiter.
    x = sine_sweep_free(seconds=2.0, amplitude=0.05)
    x[1234] = np.float32(0.9)
    y, _before, after, limited = pl.loudness_normalize(x, SR, -23.0)
    assert limited is True
    assert float(np.max(np.abs(y))) <= pl.PEAK_CEIL + 1e-4
    assert math.isfinite(after)


def test_resample_changes_length_proportionally():
    x = sine_sweep_free(seconds=1.0, sr=44100)
    y = pl.resample(x, 44100, SR)
    assert abs(y.size / x.size - SR / 44100) < 0.01


def test_dc_offset_removed():
    x = sine_sweep_free() + np.float32(0.25)
    y = pl.remove_dc_offset(x)
    assert abs(float(np.mean(y))) < 1e-4


def test_highpass_removes_rumble():
    t = np.arange(SR) / SR
    rumble = (0.5 * np.sin(2 * np.pi * 30.0 * t)).astype(np.float32)
    y = pl.highpass_filter(rumble, SR, cutoff_hz=75.0)
    # Compare steady-state (filtfilt has edge transients at the boundaries).
    steady = float(np.max(np.abs(y[2000:-2000])))
    assert steady < 0.05 * float(np.max(np.abs(rumble)))


def test_duration_filter_boundaries():
    ok_short, why = pl.duration_ok(0.29, 0.3, 15.0)
    assert not ok_short and why == pl.REASON_TOO_SHORT
    ok_long, why = pl.duration_ok(15.01, 0.3, 15.0)
    assert not ok_long and why == pl.REASON_TOO_LONG
    ok, why = pl.duration_ok(5.0, 0.3, 15.0)
    assert ok and why == ""


# ---------------------------------------------------------------------------
# Per-utterance processing (writes into tmp dirs only)
# ---------------------------------------------------------------------------

def _write_wav(path: Path, x: np.ndarray, sr: int = SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr, subtype="PCM_16")


def test_process_utterance_happy_path(tmp_path, params):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _write_wav(raw / "a.wav",
               with_silence_padding(sine_sweep_free(seconds=1.2), SR))
    row = preprocess.process_utterance(raw / "a.wav", out, SR, params)
    assert row["id"] == "a"
    expected_rel = (out / "a.wav").relative_to(preprocess.REPO_ROOT).as_posix()
    assert row["audio_path"] == expected_rel
    assert row["sample_rate"] == SR
    assert 1.0 <= row["duration_s"] <= 1.5
    assert row["clipped_frac"] == pytest.approx(0.0, abs=1e-6)
    assert abs(row["lufs_after"] + 23.0) < 1.5
    assert Path(out / "a.wav").exists()


def test_process_utterance_rejects_corrupt(tmp_path, params):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "bad.wav").write_bytes(b"RIFFnot-a-wav\x00\x01\x02garbage")
    out = tmp_path / "out"
    with pytest.raises(preprocess.Rejected) as ei:
        preprocess.process_utterance(raw / "bad.wav", out, SR, params)
    assert ei.value.reason == pl.REASON_CORRUPT


def test_process_utterance_rejects_too_short(tmp_path, params):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_wav(raw / "tiny.wav", sine_sweep_free(seconds=0.1), SR)
    with pytest.raises(preprocess.Rejected) as ei:
        preprocess.process_utterance(raw / "tiny.wav", out, SR, params)
    assert ei.value.reason == pl.REASON_TOO_SHORT


def test_process_utterance_flags_clipped_beyond_threshold(tmp_path, params):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_wav(raw / "hot.wav", clipped_signal(frac_clipped=0.05), SR)
    with pytest.raises(preprocess.Rejected) as ei:
        preprocess.process_utterance(raw / "hot.wav", out, SR, params)
    assert ei.value.reason == pl.REASON_CLIPPED
    # Same signal passes when the rejection threshold is relaxed.
    relaxed = dict(params, max_clipped_frac=0.5)
    row = preprocess.process_utterance(raw / "hot.wav", out, SR, relaxed)
    assert row["clipped_frac"] >= 0.04


def test_process_utterance_all_silence_rejected(tmp_path, params):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_wav(raw / "sil.wav", np.zeros(int(2.0 * SR), dtype=np.float32), SR)
    with pytest.raises(preprocess.Rejected) as ei:
        preprocess.process_utterance(raw / "sil.wav", out, SR, params)
    assert ei.value.reason == pl.REASON_ALL_SILENCE


def test_process_utterance_long_clip_rejected(tmp_path, params):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_wav(raw / "long.wav", sine_sweep_free(seconds=16.0), SR)
    with pytest.raises(preprocess.Rejected) as ei:
        preprocess.process_utterance(raw / "long.wav", out, SR, params)
    assert ei.value.reason == pl.REASON_TOO_LONG


# ---------------------------------------------------------------------------
# End-to-end: run_pipeline + validate_manifest on a synthetic raw dir
# ---------------------------------------------------------------------------

def _build_synthetic_raw(root: Path) -> Path:
    raw = root / "slice"
    (raw / "wavs").mkdir(parents=True)
    utts = {
        "LJ000-0001": ("in the quiet village of stonebridge", 1.2),
        "LJ000-0002": ("the sun rose over the eastern hills", 1.5),
        "LJ000-0003": ("IN THE QUIET VILLAGE OF STONEBRIDGE!", 1.1),  # dup text
        "LJ000-0004": ("too short", 0.12),                            # short
        "LJ000-0005": ("this one is clipped hot", 1.3),               # clipped
    }
    lines = []
    for uid, (text, secs) in sorted(utts.items()):
        if uid == "LJ000-0004":
            x = sine_sweep_free(seconds=secs, sr=8000)
            _write_wav(raw / "wavs" / f"{uid}.wav", x, 8000)
        elif uid == "LJ000-0005":
            _write_wav(raw / "wavs" / f"{uid}.wav",
                       clipped_signal(seconds=secs, frac_clipped=0.08), SR)
        else:
            _write_wav(raw / "wavs" / f"{uid}.wav",
                       with_silence_padding(sine_sweep_free(seconds=secs), SR))
        lines.append(f"{uid}|{text}|{text}")
    # Corrupt wav that DOES have a transcript entry -> reaches the loader and
    # fails there (CORRUPT_AUDIO, not MISSING_TRANSCRIPT).
    lines.append("LJ000-0006|broken bytes|broken bytes")
    (raw / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (raw / "wavs" / "LJ000-0006.wav").write_bytes(b"\x00\x01broken")
    return root


def test_end_to_end_pipeline_then_validate(tmp_path):
    raw_root = _build_synthetic_raw(tmp_path / "rawdir")

    res = preprocess.run_pipeline(
        raw_dir=tmp_path / "rawdir" / "slice", name="synthetic",
        out_dir=tmp_path / "data" / "manifests",
        processed_root=tmp_path / "data" / "processed",
        params={
            "target_sr": SR, "lufs_target": -23.0, "min_dur": 0.3,
            "max_dur": 15.0, "max_clipped_frac": 0.001,
            "clip_abs_threshold": pl.CLIP_ABS_THRESHOLD,
            "trim_top_db": pl.TRIM_TOP_DB_DEFAULT, "trim_pad_ms": pl.TRIM_PAD_MS,
            "highpass_hz": 75.0,
        },
        dataset="synthetic", license_id="CC0-fixture", source_url="fixture",
        download_state_path=None)

    counts = res["counts"]
    assert counts["seen"] == 6
    assert counts["kept"] == 2                      # LJ000-0001, LJ000-0002
    assert counts["rejected"] == 4
    assert counts["by_reason"] == {
        pl.REASON_CORRUPT: 1,                       # broken wav
        pl.REASON_CLIPPED: 1,                       # LJ000-0005
        pl.REASON_DUPLICATE: 1,                     # LJ000-0003
        pl.REASON_TOO_SHORT: 1,                     # LJ000-0004
    }

    rows = [json.loads(l) for l in
            open(res["kept"], encoding="utf-8").read().splitlines() if l]
    assert [r["id"] for r in rows] == ["LJ000-0001", "LJ000-0002"]
    for r in rows:
        assert set(preprocess.ROW_FIELDS) == set(r.keys())
        assert (tmp_path / r["audio_path"]).exists()

    meta = json.loads(Path(res["meta"]).read_text(encoding="utf-8"))
    assert meta["dataset"] == "synthetic"
    assert meta["license"] == "CC0-fixture"
    assert meta["download_sha256"] is None
    assert meta["transcript_report"]["non_ascii_transcripts"] == 0
    assert meta["created"]  # timestamp allowed here only

    # Integrity gate passes on fresh output...
    ok, errors = validate_manifest.validate(tmp_path / "data" / "manifests",
                                            "synthetic", repo_root=tmp_path)
    assert ok, errors

    # ...and catches tampering: rewrite one processed wav's bytes.
    victim = tmp_path / rows[0]["audio_path"]
    original = victim.read_bytes()
    victim.write_bytes(original[:-4] + b"\x01\x02\x03\x04")
    ok, errors = validate_manifest.validate(tmp_path / "data" / "manifests",
                                            "synthetic", repo_root=tmp_path)
    assert not ok
    assert any("sha256 mismatch" in e for e in errors)


def test_validate_counts_mismatch_fails(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir(parents=True)
    meta = {"dataset": "x", "counts": {"seen": 5, "kept": 99, "rejected": 0}}
    (mdir / "ghost.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (mdir / "ghost.jsonl").write_text("", encoding="utf-8")  # empty manifest
    ok, errors = validate_manifest.validate(mdir, "ghost", repo_root=tmp_path)
    assert not ok
    assert any("counts.kept" in e for e in errors)


def test_determinism_rerun_byte_identical(tmp_path):
    """Same inputs twice into the SAME dirs -> byte-identical manifests/wavs."""
    _build_synthetic_raw(tmp_path / "rawdir")
    kwargs = dict(
        raw_dir=tmp_path / "rawdir" / "slice", name="det",
        out_dir=tmp_path / "data" / "manifests",
        processed_root=tmp_path / "data" / "processed",
        params={"target_sr": SR, "lufs_target": -23.0, "min_dur": 0.3,
                "max_dur": 15.0, "max_clipped_frac": 0.001,
                "clip_abs_threshold": pl.CLIP_ABS_THRESHOLD,
                "trim_top_db": pl.TRIM_TOP_DB_DEFAULT,
                "trim_pad_ms": pl.TRIM_PAD_MS, "highpass_hz": 75.0},
        dataset="synthetic", license_id="CC0", source_url="fixture",
        download_state_path=None)

    import shutil

    def snapshot():
        return {
            "kept": Path(tmp_path / "data" / "manifests" / "det.jsonl").read_bytes(),
            "rej": Path(tmp_path / "data" / "manifests" /
                        "det.rejected.jsonl").read_bytes(),
            "wav": pl.sha256_file(tmp_path / "data" / "processed" / "det" /
                                  "LJ000-0001.wav"),
            "rows": Path(tmp_path / "data" / "manifests" /
                         "det.jsonl").read_text(encoding="utf-8").count("\n"),
        }

    preprocess.run_pipeline(**kwargs)
    s1 = snapshot()
    shutil.rmtree(tmp_path / "data")
    preprocess.run_pipeline(**kwargs)
    s2 = snapshot()
    assert s1["kept"] == s2["kept"]
    assert s1["rej"] == s2["rej"]
    assert s1["wav"] == s2["wav"]
    assert s1["rows"] == 2


def test_download_member_id_parser():
    from download_ljspeech import _member_id
    assert _member_id("LJSpeech-1.1/wavs/LJ001-0001.wav") == "LJ001-0001"
    assert _member_id("LJSpeech-1.1/metadata.csv") is None
    assert _member_id("wavs/LJ001-0001.flac") is None
