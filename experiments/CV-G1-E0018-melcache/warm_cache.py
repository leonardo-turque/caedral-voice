"""CV-G1-E0018 -- warm the persistent mel disk cache for the next launch.

Standalone filler: walks a processed-wav manifest and pushes every utterance
through MelDiskCache.get_or_compute so later training launches start with a
warm mel tier instead of paying ~25-35 min of CPU per cold process.

Politeness contract (r1 training worker must keep priority):
* hard-capped thread pool (default 2, --threads);
* Windows BELOW_NORMAL_PRIORITY_CLASS + background mode via ctypes (no-op on
  other platforms);
* resume-friendly: reruns are near-no-ops (every entry is a disk hit).

Usage:
    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0018-melcache/warm_cache.py \
        [--manifest librittsr-clean100] [--threads 2] [--limit N]

Progress: warm.progress.json rewritten every --progress-every items; log lines
every 250 items + MelDiskCache's own every-1000 lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.training.mel_cache import MelDiskCache  # noqa: E402


def _drop_priority() -> str:
    """Best-effort OS-level deprioritization; returns a description."""
    if sys.platform != "win32":
        try:
            import os
            os.nice(10)
            return "nice+10"
        except Exception:
            return "unavailable"
    try:
        import ctypes
        BELOW_NORMAL = 0x4000
        PROCESS_MODE_BACKGROUND_BEGIN = 0x00100000
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ok1 = bool(ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL))
        # best-effort: also request background mode (I/O + scheduling)
        ctypes.windll.kernel32.SetPriorityClass(
            h, BELOW_NORMAL | PROCESS_MODE_BACKGROUND_BEGIN) or ok1
        return "below-normal" + ("+background" if ok1 else "")
    except Exception:
        return "failed (running at normal priority)"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=str, default="librittsr-clean100",
                   help="manifest name under data/manifests/ (default clean100)")
    p.add_argument("--threads", type=int, default=2,
                   help="worker threads, HARD-CAPPED at 2 unless overridden")
    p.add_argument("--root", type=str, default=None,
                   help="cache root (default data/cache/mel-v1)")
    p.add_argument("--limit", type=int, default=None,
                   help="only warm the first N rows (smoke/testing)")
    p.add_argument("--progress-every", type=int, default=25,
                   help="rewrite progress json every N completions")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    threads = max(1, min(args.threads, 2)) if args.limit is None \
        else max(1, args.threads)          # smoke runs may use more
    t0 = time.perf_counter()

    import os
    print(f"[warm] pid={os.getpid()} priority={_drop_priority()} "
          f"threads={threads}", flush=True)

    mpath = REPO / "data" / "manifests" / f"{args.manifest}.jsonl"
    rows = [json.loads(l) for l in mpath.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"[warm] manifest {mpath.name}: {total} utts", flush=True)

    cache = MelDiskCache(root=args.root, sample_rate=24000, n_mels=80,
                         hop=256, n_fft=1024)
    cache.stats["hits"] = 0               # fresh counters for THIS invocation
    cache.stats["misses"] = 0
    print(f"[warm] cache root {cache.root} (v{cache.version})", flush=True)

    prog_path = Path(__file__).parent / "warm.progress.json"
    done = errors = 0
    lock_print = __import__("threading").Lock()

    def work(row):
        path = REPO / row["audio_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = cache.get_or_compute(path, expected_sr=24000)
        assert arr.dtype.name == "float32" and arr.shape[0] == 80
        return row["id"]

    def write_progress():
        el = time.perf_counter() - t0
        rate = done / el if el > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else None
        prog = {
            "experiment": "CV-G1-E0018-melcache",
            "manifest": args.manifest,
            "total": total,
            "done": done,
            "percent": round(100.0 * done / total, 2) if total else 100.0,
            "errors": errors,
            "cache_stats": dict(cache.stats),
            "elapsed_s": round(el, 1),
            "utts_per_s": round(rate, 2),
            "eta_s": round(remaining, 1) if remaining is not None else None,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = prog_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(prog, indent=2), encoding="utf-8")
        tmp.replace(prog_path)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:                     # noqa: BLE001
                errors += 1
                with lock_print:
                    print(f"[warm] ERROR {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % args.progress_every == 0 or done == total:
                write_progress()
            if done % 250 == 0:
                s = cache.stats
                pct = 100.0 * done / total
                eta = (time.perf_counter() - t0) / done * (total - done)
                print(f"[warm] {done}/{total} ({pct:.1f}%) "
                      f"computed={s['computed']} hits={s['hits']} "
                      f"errors={errors} eta_min={eta / 60:.1f}", flush=True)

    write_progress()
    s = cache.stats
    print(f"[warm] DONE {done}/{total} errors={errors} "
          f"computed_now={s['computed']} hits_now={s['hits']} "
          f"stale={s['stale_version']} invalid={s['invalid']} "
          f"wall={(time.perf_counter() - t0) / 60:.1f} min "
          f"bytes={s['bytes_written']}", flush=True)
    scan = cache.scan_stale()
    print(f"[warm] stale-scan: {scan['entries']} entries, "
          f"{scan['stale']} stale", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
