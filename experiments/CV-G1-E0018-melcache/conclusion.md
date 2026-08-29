# CV-G1-E0018 — Persistent incremental mel disk cache (MELCACHE-01)

**Date:** 2026-08-23 · **Status:** COMPLETE — cache WARM (30,917/30,917 entries, 0 errors)
**Directive:** founder — mel preprocessing of 30,917 utts (~25–35 min CPU) had been
repeated 3× after process/session failures killed in-RAM caches. Make it pay-once.

## Design

New module `caedral_voice/training/mel_cache.py` (no existing file's semantics touched):

* **Identity**: `key = sha256(audio_file_bytes + b"|" + fingerprint)` where the
  fingerprint pins the full recipe (`melcache-v{VERSION}|sr|mels|hop|fft|power=1|floor=1e-5`).
  Hashing *bytes* (not paths) means renamed-but-identical audio shares one entry and any
  audio change gets a new one. Files: `{key[:32]}.npy` + sidecar `.json`
  `{sha256, n_mels, sr, hop, n_fft, version, frames, created_utc}`.
* **Versioning**: `MelDiskCache.VERSION = 1`, default root `data/cache/mel-v{VERSION}/`.
  VERSION participates in the key, so a bump moves every filename — stale entries can
  never be found under a new version's keys (old files orphaned on disk for GC);
  `scan_stale()` classifies leftovers in one pass; a candidate whose sidecar disagrees
  with the expected key/version is counted `stale_version`, never reused, atomically replaced.
* **Exactness contract**: `compute()` reproduces the datasets' `_load_mel` recipe bitwise:
  `torchaudio.load → mean(channels) → functional.resample(→24 kHz) → MelSpectrogram(power=1.0)
  → ln(clamp_min(1e-5))`. Proven below on synthetic AND real corpus wavs.
* **Atomicity / interruption safety**: `.npy` streamed to a unique `*.tmp-{pid}-{tid}-{uuid}`
  file → flush → fsync → `os.replace`; sidecar written last the same way. Sidecar presence
  certifies a complete array; a killed writer leaves only unreferenced tmp files that are
  never read nor promoted.
* **Validation on read**: sidecar key/version/config match, `allow_pickle=False` load,
  dtype float32, `shape[0]==80`, all-finite; any failure ⇒ `invalid`, recompute, heal file.
* **Ops**: lock-guarded hit/miss/stale/invalid/computed/bytes counters, log line per 1000
  items, thread-safe (`get_or_compute` is safe from a small pool).

## Opt-in integration (r1 untouched)

Minimal patch, **32 insertions, 0 deletions** across three files:

* `mel_dataset.PilotDataset` / `regressor_dataset.RegressorDataset`: trailing keyword
  param `disk_cache=None` stored + a 5-line branch at the top of `_load_mel` that runs
  only when a cache was passed; default path byte-identical to pre-E0018.
* `train_regressor.py`: `--mel-cache-dir` flag (default None), constructs `MelDiskCache`,
  threads it into `RegressorDataset`, appends `mel_cache_stats` to the fit summary.

The running r1 worker (PID 11140, `--exp-id r1`) already holds its code in memory; disk
edits cannot affect it. Nothing about its inputs (`data/cache/` did not exist before)
changed.

## Evidence

| Gate | Result |
|---|---|
| Identity vs direct compute, PilotDataset | [MEASURED] `torch.equal` true, max-abs-diff == 0.0 |
| Identity vs direct compute, RegressorDataset (+ identical synthetic durations) | [MEASURED] max-abs-diff == 0.0 |
| Identity on REAL clean100 wav (22050→24k resample path) | [MEASURED] max-abs-diff == 0.0, shape (80,153) float32 |
| Determinism across independent instances/runs | [MEASURED] bitwise equal arrays, identical keys/filenames |
| Atomicity: orphan tmp files | [MEASURED] never read/promoted, entry computed correctly alongside |
| Atomicity: truncated promoted .npy / NaN entry | [MEASURED] rejected (`invalid`), recomputed, file healed |
| Stale version rejection | [MEASURED] v2 lookup of v1 content: `stale_version=1`, recomputed, v1 files left for GC, `scan_stale` reports them |
| Hit/miss accounting | [MEASURED] miss+hit+second-file: misses=2 hits=1 computed=2 bytes>0; rerun of warmed set = 12/12 hits, 0 computes |
| `expected_sr` mismatch guard | [MEASURED] ValueError raised before I/O, counters untouched |
| Default-path regression | [MEASURED] full existing suite **131 passed** (excl. new file); new suite **11 passed** |

## Cache warming (next-launch readiness) — DONE

`warm_cache.py` walks `librittsr-clean100.jsonl` (30,917 utts) through `MelDiskCache`
with a hard-capped 2-thread pool and Windows BELOW_NORMAL_PRIORITY_CLASS (+background
mode) beside the healthy r1 worker (PID 11140 verified alive before/during/after).
Detached run completed 2026-08-23T03:39Z:

* [MEASURED] **30,917/30,917 entries written in 7.9 min** (65.4 utt/s at 2
  below-normal threads), **0 errors, 0 stale, 0 invalid**, 4.77 GB / 4.6 GiB on disk;
  final stale-scan: 30,917 entries, 0 stale.
* [MEASURED] rerun proof: 300/300 pure disk hits, computed_now=0 — a cold launch now
  pays load-from-disk instead of decode+resample+mel.
* Resume-friendly: an interrupted warm simply re-runs as hits.

Note: 7.9 min measured here vs the historical ~25–35 min figure for in-pipeline
preprocessing — the warmer hashes bytes and computes mels only (no manifest rebuild,
no dataloader contention); treat the historical number as the conservative bound.

## Next-launch flag

```
--mel-cache-dir data/cache/mel-v1
```

(any directory works; the conventional layout is `data/cache/mel-v{MelDiskCache.VERSION}`)
