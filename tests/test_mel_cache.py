"""Tests for the E0018 persistent log-Mel disk cache (mel_cache.py).

Proves the opt-in contract:
- roundtrip IDENTITY vs both datasets' direct computation (max-abs-diff == 0,
  torch.equal);
- atomicity: interrupted writers leave only *.tmp-* files which are never read
  nor promoted; truncated/corrupt promoted .npy is rejected and recomputed;
- stale-version entries are never reused, are recomputed, and old-version
  files stay on disk for GC;
- hit/miss/computed/stale counters behave;
- determinism: independent cache instances (and disk reloads) produce bitwise
  identical mels and identical keys/filenames.

All audio fixtures are synthesized in-process (stdlib ``wave``), so the suite
needs no corpus and runs in seconds.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.training.mel_cache import MelDiskCache  # noqa: E402
from caedral_voice.training.mel_dataset import PilotDataset  # noqa: E402
from caedral_voice.training.regressor_dataset import RegressorDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures: deterministic PCM16 wav writers + tiny datasets
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, sr: int = 22050, seconds: float = 0.6,
               seed: int = 0) -> None:
    """Deterministic mono PCM16 wav (two-tone, no RNG)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        v = 0.4 * math.sin(2 * math.pi * (220 + 11 * seed) * t) \
            + 0.2 * math.sin(2 * math.pi * 3.0 * t + seed)
        s = int(max(-1.0, min(1.0, v)) * 32767)
        frames += struct.pack("<h", s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


class _TinyCorpus:
    """One wav + manifest/phoneme rows shaped for both dataset classes."""

    def __init__(self, root: Path, sr_source: int = 22050, seed: int = 0):
        self.root = root
        self.rel = f"data/processed/t/utt{seed}.wav"
        self.wav = root / self.rel
        _write_wav(self.wav, sr=sr_source, seconds=0.6, seed=seed)
        self.manifest_row = {"id": f"v{seed}_0001", "audio_path": self.rel,
                             "duration_s": 0.6}
        self.phone_row = {"id": f"v{seed}_0001",
                          "phoneme_string": "a b"}
        self.speaker_index = {f"v{seed}": 0}


def _pilot(corpus: _TinyCorpus, disk_cache=None) -> PilotDataset:
    return PilotDataset(
        [corpus.manifest_row], [corpus.phone_row], corpus.speaker_index,
        sample_rate=24000, n_mels=80, hop=256, n_fft=1024,
        repo_root=corpus.root, disk_cache=disk_cache)


def _regressor(corpus: _TinyCorpus, disk_cache=None) -> RegressorDataset:
    return RegressorDataset(
        [corpus.manifest_row], [corpus.phone_row], None,
        corpus.speaker_index, sample_rate=24000, n_mels=80, hop=256,
        n_fft=1024, repo_root=corpus.root, synthetic_uniform_durations=True,
        disk_cache=disk_cache)


# --------------------------------------------------------------------------- #
# identity vs direct computation
# --------------------------------------------------------------------------- #
def test_roundtrip_identity_vs_direct_pilot(tmp_path):
    corpus = _TinyCorpus(tmp_path)
    cache = MelDiskCache(root=tmp_path / "cache")

    ds_direct = _pilot(corpus)
    ds_cached = _pilot(corpus, disk_cache=cache)

    mel_direct = ds_direct[0]["mel"]
    mel_cached = ds_cached[0]["mel"]
    assert isinstance(mel_cached, torch.Tensor)
    assert mel_cached.shape[0] == 80
    assert torch.equal(mel_direct, mel_cached)          # bitwise
    assert float((mel_direct - mel_cached).abs().max()) == 0

    # standalone API agrees too
    arr = cache.get_or_compute(corpus.wav, expected_sr=24000)
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
    assert arr.shape == (80, mel_direct.shape[1])
    assert float(np.abs(arr - mel_direct.numpy()).max()) == 0
    # ds_cached[0] was the miss (computed + stored); this call hit the disk
    assert cache.stats["misses"] == 1 and cache.stats["hits"] == 1


def test_roundtrip_identity_vs_direct_regressor(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=1)
    cache = MelDiskCache(root=tmp_path / "cache")

    ds_direct = _regressor(corpus)
    ds_cached = _regressor(corpus, disk_cache=cache)

    it_direct = ds_direct[0]
    it_cached = ds_cached[0]
    assert torch.equal(it_direct["mel"], it_cached["mel"])
    # synthetic uniform durations derive from T_mel: identical mels => identical
    assert it_direct["duration_targets"] == it_cached["duration_targets"]

    # second pass through a FRESH dataset (RAM memo empty) hits the disk cache
    ds_warm = _regressor(corpus, disk_cache=cache)
    assert torch.equal(it_direct["mel"], ds_warm[0]["mel"])
    assert cache.stats["hits"] == 1 and cache.stats["computed"] == 1


def test_dataset_disk_cache_survives_ram_memo_clear(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=2)
    cache = MelDiskCache(root=tmp_path / "cache")
    ds = _pilot(corpus, disk_cache=cache)
    first = ds[0]["mel"].clone()
    ds._mel_cache.clear()                    # simulate epoch 2 after eviction
    second = ds[0]["mel"]
    assert torch.equal(first, second)
    assert cache.stats["hits"] == 1          # served by the DISK tier


# --------------------------------------------------------------------------- #
# atomicity / interruption safety
# --------------------------------------------------------------------------- #
def test_interrupted_writer_tmp_files_never_promoted_or_read(tmp_path):
    corpus = _TinyCorpus(tmp_path)
    root = tmp_path / "cache"
    cache = MelDiskCache(root=root)
    ref = cache.get_or_compute(corpus.wav, 24000)
    stem = cache.key_for(corpus.wav)[:32]

    # simulate a killed writer: orphan tmp files with garbage, no final files
    for p in root.glob(f"{stem}.*"):
        p.unlink()
    (root / f"{stem}.npy.tmp-999-deadbeef").write_bytes(b"\x00\x01partial")
    (root / f"{stem}.json.tmp-999-deadbeef").write_text('{"trunc')

    got = cache.get_or_compute(corpus.wav, 24000)
    assert np.array_equal(got, ref)
    assert np.isfinite(got).all()
    # orphan tmps were neither promoted nor consumed -- they just sit there
    assert (root / f"{stem}.npy.tmp-999-deadbeef").read_bytes() == b"\x00\x01partial"
    # proper entry exists alongside
    assert (root / f"{stem}.npy").is_file() and (root / f"{stem}.json").is_file()
    side = json.loads((root / f"{stem}.json").read_text(encoding="utf-8"))
    for field in ("sha256", "n_mels", "sr", "hop", "version", "created_utc"):
        assert field in side


def test_truncated_promoted_npy_rejected_and_recomputed(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=3)
    root = tmp_path / "cache"
    cache = MelDiskCache(root=root)
    ref = cache.get_or_compute(corpus.wav, 24000)
    stem = cache.key_for(corpus.wav)[:32]
    npy, js = root / f"{stem}.npy", root / f"{stem}.json"
    good_meta = js.read_text(encoding="utf-8")
    good_bytes = npy.read_bytes()

    # truncate the promoted array to half its bytes; sidecar stays "valid"
    npy.write_bytes(good_bytes[: len(good_bytes) // 2])
    cache2 = MelDiskCache(root=root)
    got = cache2.get_or_compute(corpus.wav, 24000)
    assert np.array_equal(got, ref)
    assert cache2.stats["invalid"] == 1
    assert cache2.stats["hits"] == 0
    # ...and the rewrite healed the file atomically
    assert (root / f"{stem}.npy").read_bytes() == good_bytes
    assert json.loads(js.read_text(encoding="utf-8")) == json.loads(good_meta)


def test_nan_entry_rejected_and_recomputed(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=4)
    root = tmp_path / "cache"
    cache = MelDiskCache(root=root)
    ref = cache.get_or_compute(corpus.wav, 24000)
    stem = cache.key_for(corpus.wav)[:32]
    arr = np.load(root / f"{stem}.npy")
    arr[0, 0] = np.nan
    np.save(root / f"{stem}.npy", arr)       # finite check must catch this
    cache2 = MelDiskCache(root=root)
    got = cache2.get_or_compute(corpus.wav, 24000)
    assert np.array_equal(got, ref) and np.isfinite(got).all()
    assert cache2.stats["invalid"] == 1


# --------------------------------------------------------------------------- #
# versioning
# --------------------------------------------------------------------------- #
def test_stale_version_never_reused_and_old_files_left_for_gc(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=5)
    root = tmp_path / "cache"

    v1 = MelDiskCache(root=root, version=1)
    arr_v1 = v1.get_or_compute(corpus.wav, 24000)
    stem_v1 = v1.key_for(corpus.wav)[:32]
    assert (root / f"{stem_v1}.npy").is_file()

    v2 = MelDiskCache(root=root, version=2)
    # legacy-layout simulation: old-format content sitting at the NEW version's
    # filenames with OLD sidecar metadata (what an in-place upgrade finds)
    stem_v2 = v2.key_for(corpus.wav)[:32]
    shutil.copyfile(root / f"{stem_v1}.npy", root / f"{stem_v2}.npy")
    shutil.copyfile(root / f"{stem_v1}.json", root / f"{stem_v2}.json")

    got = v2.get_or_compute(corpus.wav, 24000)
    assert isinstance(got, np.ndarray) and got.dtype == np.float32
    assert np.isfinite(got).all()
    assert v2.stats["stale_version"] == 1    # detected + rejected, NEVER reused
    assert v2.stats["hits"] == 0

    # recomputed sidecar now carries the new version...
    meta = json.loads((root / f"{stem_v2}.json").read_text(encoding="utf-8"))
    assert meta["version"] == 2 and meta["sha256"] == v2.key_for(corpus.wav)
    # ...and the v1 files remain untouched on disk for GC
    assert (root / f"{stem_v1}.npy").is_file()
    assert (root / f"{stem_v1}.json").is_file()

    report = v2.scan_stale()
    assert report["stale"] >= 1              # maintenance scan sees them too


def test_expected_sr_mismatch_raises_before_io(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=6)
    cache = MelDiskCache(root=tmp_path / "cache")
    with pytest.raises(ValueError):
        cache.get_or_compute(corpus.wav, expected_sr=22050)
    for k in ("hits", "misses", "computed", "stale_version", "invalid"):
        assert cache.stats[k] == 0           # nothing counted
    assert not (tmp_path / "cache").exists() or \
        not any((tmp_path / "cache").glob("*"))   # no files created


# --------------------------------------------------------------------------- #
# hit/miss accounting
# --------------------------------------------------------------------------- #
def test_hit_and_miss_counters(tmp_path):
    corpus_a = _TinyCorpus(tmp_path, seed=7)
    corpus_b = _TinyCorpus(tmp_path, seed=8)
    cache = MelDiskCache(root=tmp_path / "cache")

    cache.get_or_compute(corpus_a.wav, 24000)     # miss
    cache.get_or_compute(corpus_a.wav, 24000)     # hit
    cache.get_or_compute(corpus_b.wav, 24000)     # miss
    s = cache.stats
    assert s["misses"] == 2 and s["computed"] == 2
    assert s["hits"] == 1
    assert s["stale_version"] == 0 and s["invalid"] == 0
    assert s["bytes_written"] > 0


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_determinism_across_instances_and_runs(tmp_path):
    corpus = _TinyCorpus(tmp_path, seed=9)
    c1 = MelDiskCache(root=tmp_path / "c1")
    c2 = MelDiskCache(root=tmp_path / "c2")     # independent dir, same recipe

    a = c1.get_or_compute(corpus.wav, 24000)    # compute
    b = c2.get_or_compute(corpus.wav, 24000)    # independent compute
    assert np.array_equal(a, b)                 # bitwise across runs

    c3 = MelDiskCache(root=tmp_path / "c1")     # same dir -> disk reload
    d = c3.get_or_compute(corpus.wav, 24000)
    assert np.array_equal(a, d)
    assert c3.stats["hits"] == 1

    # keys/filenames stable across instances
    assert c1.key_for(corpus.wav) == c2.key_for(corpus.wav)
    stem = c1.key_for(corpus.wav)[:32]
    assert (tmp_path / "c1" / f"{stem}.npy").is_file()


def test_key_depends_on_audio_bytes_not_path(tmp_path):
    ca = _TinyCorpus(tmp_path, seed=10)
    cb = _TinyCorpus(tmp_path, seed=11)         # different tones -> different key
    cache = MelDiskCache(root=tmp_path / "cache")
    k_a = cache.key_for(ca.wav)
    k_b = cache.key_for(cb.wav)
    assert k_a != k_b

    # byte-identical copy under a DIFFERENT name shares one entry
    alias = tmp_path / "data/processed/other/renamed.wav"
    alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ca.wav, alias)
    assert cache.key_for(alias) == k_a
