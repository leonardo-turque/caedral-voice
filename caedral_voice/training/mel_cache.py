"""Persistent incremental log-Mel disk cache (experiment CV-G1-E0018).

Motivation: mel preprocessing of the 30,917-utterance clean100 corpus costs
~25-35 min of CPU and, with ``num_workers=0``, lives in RAM -- three process or
session failures have each thrown that work away. This module persists mels to
``data/cache/mel-v{VERSION}/`` so preprocessing is paid ONCE per
(feature-semantics, audio-bytes) pair and every later launch starts warm.

Exactness contract: ``get_or_compute`` reproduces the dataset recipe in
``mel_dataset.PilotDataset._load_mel`` / ``regressor_dataset.RegressorDataset``
*bitwise* on this box:

    torchaudio.load -> mean(channels) -> resample to target sr (torchaudio
    functional.resample defaults) -> MelSpectrogram(power=1.0) ->
    natural log(clamp_min(1e-5))  ==  float32 [80, T]

Unit tests prove max-abs-diff == 0 against both datasets' direct path, so
opt-in caching cannot change training data by even one ulp.

Layout / identity:
    root = data/cache/mel-v{VERSION}/            (default)
    key  = sha256( audio_file_bytes + b"|" + fingerprint )
    fingerprint = "melcache-v{VERSION}|sr=..|mels=..|hop=..|fft=..|power=1|floor=.."
    files: {key[:32]}.npy + {key[:32]}.json sidecar

Because VERSION participates in the key, bumping it moves every filename: a
stale-version entry can never be found/reused under a new version's key (old
files are simply orphaned for GC). ``scan_stale()`` walks a cache root and
reports entries whose sidecar version/config no longer matches (legacy layouts,
manual copies); ``get_or_compute`` itself additionally rejects any candidate
whose sidecar disagrees with the expected key/version (truncated-name
collision, doctored file), counts it stale/invalid, and recomputes.

Atomicity: the .npy is streamed to a uniquely-named ``*.tmp-*`` file, flushed,
fsynced, then ``os.replace``d into place; only afterwards is the sidecar json
written the same way. Sidecar presence therefore certifies a complete .npy;
an interrupted writer leaves only unreferenced tmp files which are never
promoted and never read.

Thread model: get_or_compute is safe to call concurrently from a small thread
pool (the E0018 warmer uses 2). Shared state (stats counters, log cadence) is
lock-guarded; audio decode + mel math releases the GIL inside torchaudio.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mel_dataset import MEL_FLOOR

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MelDiskCache:
    """sha256-keyed persistent log-Mel cache. See module docstring."""

    VERSION = DEFAULT_VERSION

    def __init__(
        self,
        root: str | Path | None = None,
        sample_rate: int = 24000,
        n_mels: int = 80,
        hop: int = 256,
        n_fft: int = 1024,
        log_floor: float = MEL_FLOOR,
        version: int | None = None,
        fsync: bool = True,
        log=None,                      # callable(str); default print(flush)
    ):
        self.version = int(version) if version is not None else int(self.VERSION)
        self.root = Path(root) if root is not None else (
            REPO_ROOT / "data" / "cache" / f"mel-v{self.version}")
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.log_floor = float(log_floor)
        self.fsync = bool(fsync)
        self._log_fn = log if log is not None else (
            lambda msg: print(msg, flush=True))

        # config fingerprint baked into EVERY key: any change here silently
        # moves all filenames instead of serving mismatched mels
        self.fingerprint = (
            f"melcache-v{self.version}|sr={self.sample_rate}"
            f"|mels={self.n_mels}|hop={self.hop}|fft={self.n_fft}"
            f"|power=1|floor={self.log_floor:g}")
        self.stats = {
            "hits": 0,             # valid sidecar + valid npy served from disk
            "misses": 0,           # lookups that ended in a fresh computation
            "computed": 0,         # mels actually computed (== misses)
            "stale_version": 0,    # candidate found but version/key metadata
                                   # disagreed -> recomputed, never reused
            "invalid": 0,          # corrupt/truncated npy or validation failure
                                   # (shape, dtype, non-finite) -> recomputed
            "bytes_written": 0,
            "read_errors": 0,      # unexpected read exceptions (also recomputed)
        }
        self._stats_lock = threading.Lock()
        self._processed_since_log = 0
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # keying
    # ------------------------------------------------------------------ #
    def key_for(self, audio_path: str | Path) -> str:
        """Full sha256 hex over the audio bytes plus the config fingerprint.

        Hashing file bytes (not paths) means renamed/moved-but-identical audio
        shares one cache entry; changed audio always gets a new entry.
        """
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"|")
        h.update(self.fingerprint.encode("utf-8"))
        return h.hexdigest()

    def _stem(self, key: str) -> str:
        return key[:32]

    def _npy_path(self, stem: str) -> Path:
        return self.root / f"{stem}.npy"

    def _json_path(self, stem: str) -> Path:
        return self.root / f"{stem}.json"

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get_or_compute(self, audio_path: str | Path,
                       expected_sr: int) -> np.ndarray:
        """Return the [n_mels, T] float32 log-Mel for ``audio_path``.

        ``expected_sr`` must equal this cache's configured target sample rate
        (the rate the returned mel is expressed at); a mismatch raises before
        any I/O so two pipeline configs can never cross-contaminate entries.
        """
        if int(expected_sr) != self.sample_rate:
            raise ValueError(
                f"expected_sr {expected_sr} != cache sample_rate "
                f"{self.sample_rate}: refusing to serve mels across configs")
        audio_path = Path(audio_path)
        key = self.key_for(audio_path)
        stem = self._stem(key)
        npy_path = self._npy_path(stem)
        json_path = self._json_path(stem)

        arr = self._try_load(key, npy_path, json_path)
        if arr is not None:
            self._bump("hits")
            return arr
        # miss (files absent, or rejected as stale/corrupt -- _try_load already
        # classified rejections into stale_version/invalid): compute + publish
        arr = self.compute(audio_path)
        self._write_entry(stem, key, arr, audio_path)
        self._bump("misses")
        return arr

    def compute(self, audio_path: str | Path) -> np.ndarray:
        """Reference log-Mel computation -- bitwise-identical to the datasets'
        ``_load_mel`` recipe (see module docstring). Thread-safe."""
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))   # (C, T) float32
        wav = wav.mean(dim=0)                        # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.sample_rate)
        mel_fn = getattr(self._tls(), "mel_fn", None)
        if mel_fn is None:
            mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop, n_mels=self.n_mels, power=1.0,
            )
            self._tls().mel_fn = mel_fn
        mel = mel_fn(wav)                            # (n_mels, T) power=1
        mel = torch.log(mel.clamp_min(self.log_floor))
        out = mel.numpy().astype(np.float32, copy=True)
        return out

    def scan_stale(self) -> dict:
        """One pass over the cache root: classify every *.json sidecar as
        current or stale (version/config differs from this instance).
        Returns {'entries': n, 'stale': n, 'stale_examples': [...] }.
        Maintenance helper -- per-item lookups never need it because VERSION
        participates in the key."""
        entries = stale = 0
        examples: list[str] = []
        for jp in self.root.glob("*.json"):
            entries += 1
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                stale += 1
                continue
            if (int(meta.get("version", -1)) != self.version
                    or int(meta.get("sr", -1)) != self.sample_rate
                    or int(meta.get("n_mels", -1)) != self.n_mels
                    or int(meta.get("hop", -1)) != self.hop
                    or int(meta.get("n_fft", -1)) != self.n_fft):
                stale += 1
                if len(examples) < 5:
                    examples.append(jp.name)
        return {"entries": entries, "stale": stale,
                "stale_examples": examples}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _tls(self):
        tls = getattr(self, "_tls_storage", None)
        if tls is None:
            tls = threading.local()
            self._tls_storage = tls
        return tls

    def _try_load(self, key: str, npy_path: Path,
                  json_path: Path) -> np.ndarray | None:
        """Validate-and-load a candidate entry; None means recompute.
        Any rejection increments exactly one stats bucket."""
        if not (npy_path.is_file() and json_path.is_file()):
            return None                       # plain miss (caller counts it)
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            self._bump("invalid")
            return None
        if (int(meta.get("version", -1)) != self.version
                or str(meta.get("sha256", "")) != key):
            # wrong version (or full-key mismatch: truncated-name collision /
            # doctored sidecar) -> NEVER reuse; old file left for GC until the
            # atomic rewrite replaces same-named content
            self._bump("stale_version")
            return None
        if (int(meta.get("sr", -1)) != self.sample_rate
                or int(meta.get("n_mels", -1)) != self.n_mels
                or int(meta.get("hop", -1)) != self.hop
                or int(meta.get("n_fft", -1)) != self.n_fft):
            self._bump("stale_version")
            return None
        try:
            arr = np.load(npy_path, allow_pickle=False)
        except Exception:
            self._bump("invalid")
            return None
        if (not isinstance(arr, np.ndarray) or arr.dtype != np.float32
                or arr.ndim != 2 or arr.shape[0] != self.n_mels
                or not np.isfinite(arr).all()):
            self._bump("invalid")
            return None
        return arr

    def _write_entry(self, stem: str, key: str, arr: np.ndarray,
                     audio_path: Path) -> None:
        """Atomic publish: tmp -> flush -> fsync -> os.replace, .npy first,
        sidecar last (sidecar presence == complete array)."""
        self.root.mkdir(parents=True, exist_ok=True)
        suffix = f".tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
        npy_path = self._npy_path(stem)
        json_path = self._json_path(stem)

        tmp_npy = self.root / f"{stem}.npy{suffix}"
        with open(tmp_npy, "wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())
        os.replace(tmp_npy, npy_path)

        meta = {
            "sha256": key,
            "n_mels": self.n_mels,
            "sr": self.sample_rate,
            "hop": self.hop,
            "n_fft": self.n_fft,
            "version": self.version,
            "frames": int(arr.shape[1]),
            "audio_file": Path(audio_path).name,
            "created_utc": _utc_now_iso(),
        }
        tmp_json = self.root / f"{stem}.json{suffix}"
        with open(tmp_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, indent=2) + "\n")
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())
        os.replace(tmp_json, json_path)

        with self._stats_lock:
            self.stats["bytes_written"] += int(arr.nbytes)

    def _bump(self, name: str) -> None:
        with self._stats_lock:
            self.stats[name] += 1
            if name == "misses":
                self.stats["computed"] += 1
            self._processed_since_log += 1
            processed = self.stats["hits"] + self.stats["misses"]
            due = self._processed_since_log >= 1000
            if due:
                self._processed_since_log = 0
        if due:
            self._log_fn(
                f"[melcache] {processed} items processed "
                f"(hits={self.stats['hits']} computed={self.stats['computed']} "
                f"stale={self.stats['stale_version']} "
                f"invalid={self.stats['invalid']}) "
                f"elapsed={time.perf_counter() - self._t0:.1f}s")

    def __repr__(self) -> str:
        return (f"MelDiskCache(root={str(self.root)!r}, v{self.version}, "
                f"sr={self.sample_rate}, n_mels={self.n_mels}, hop={self.hop}, "
                f"fft={self.n_fft})")
