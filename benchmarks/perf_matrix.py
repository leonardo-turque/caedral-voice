"""PERF-MATRIX: MEASURED RTF / TTFA / peak-RSS / steady-RAM vs thread count.

Founder requirement for V1: measured (never estimated) RTF, TTFA, peak RSS
and steady RAM at torch intra-op thread counts {1,2,4,8}. This harness runs
against any checkpointed run dir; the full sweep belongs on V1 checkpoints,
tiny local validations are fine anytime.

Protocol per thread count (fresh engine each time):
  1. CaedralEngine.from_run(threads=N) built fresh (load timed separately);
     the previous engine is unloaded (del + gc.collect()) beforehand.
  2. EXPLICIT untimed warmup synth first -> one-time costs (g2p_en NLTK LSTM
     ~12 s cold, vocoder load, lazy kernel paths) are excluded from every
     number reported below. Cold first-synth does NOT include engine warmup.
  3. Per sentence:
       - first TIMED synth  -> 'cold_first_synth_wall' (per-input lazy costs
         only: new tensor shapes / allocator growth for this input);
       - WARM_REPEATS timed synths -> warm wall + RTF (median reported);
       - audio_seconds from the produced waveform; RTF = wall / audio.
  4. TTFA, two flavors:
       - 'ttfa_upper_bound': warm full-sentence synth wall (an UPPER BOUND --
         the whole clip must exist before the caller sees anything);
       - 'ttfa_streaming_estimate': TRUE first-chunk estimate following the
         RUNTIME-01 design: sentence-split the text, synthesize ONLY chunk 1,
         report wall-to-first-chunk-audio. Implemented HERE in the harness
         (no engine change needed); becomes obsolete if the engine grows a
         real incremental decoder, which would replace this estimate.
  5. Unload (del + gc) -> steady RSS after collection.

Memory accounting honesty: this harness intentionally runs all thread
configs in ONE process (engine rebuild isolates model state, not the
allocator). Win32 PeakWorkingSetSize is therefore process-LIFETIME and is
reported ONCE at the end, never attributed to a config. Per-config peaks
use the harness sampler (benchmarks.harness.measure), with the config's
entry baseline recorded next to it.

Output schema: each thread config carries a 'comparison_row' mirroring the
fields of benchmarks/results/natgate/comparison-template.json (rtf /
ram_mb / size_mb, quality slots null) so the final V1 table is a join.

Usage:
  python benchmarks/perf_matrix.py --run-dir experiments/pilot-tiny-15k \
      [--checkpoint PATH] [--sentences-json PATH] [--threads 1,2,4,8] \
      [--warm-repeats 3] [--limit N] [--out results.json]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.harness.measure import measure, rtf, win_process_peaks  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402

DEFAULT_SENTENCES = REPO / "benchmarks" / "preview-sentences.json"
DEFAULT_OUT_DIR = REPO / "benchmarks" / "results" / "perf-matrix"

# Keys of a sentences-json dict that hold directives/prose, not sentences.
_META_KEYS = {"purpose", "rules", "version", "notes", "protocol"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sentences(path: Path) -> list[dict]:
    """Accept either a flat list of strings or a preview-sentences-style
    dict {category: str | [str, ...]} (meta keys skipped)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[dict] = []
    if isinstance(data, list):
        for i, text in enumerate(data, 1):
            items.append({"id": f"s_{i:03d}", "category": "adhoc",
                          "text": str(text)})
        return items
    if isinstance(data, dict):
        for category, val in data.items():
            if category in _META_KEYS:
                continue
            texts = val if isinstance(val, list) else [val]
            n = 0
            for entry in texts:
                if not isinstance(entry, str) or not entry.strip():
                    continue
                n += 1
                items.append({"id": f"{category}_{n:03d}",
                              "category": str(category), "text": entry})
        return items
    raise ValueError(f"unsupported sentences-json structure in {path}")


def split_sentences(text: str) -> list[str]:
    """Approximate sentence splitter for chunk-boundary estimates.

    Deliberately naive (regex on .!? + whitespace); good enough to define
    chunk boundaries for TTFA estimation on benchmark prose, not a production
    chunker (that lives in the RUNTIME-01 longform layer).
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _mb(num_bytes: float) -> float:
    return round(num_bytes / 1e6, 1)


def run_thread_config(
    n_threads: int,
    sentences: list[dict],
    *,
    run_dir: Path,
    checkpoint: Path | None,
    warm_repeats: int,
    log=print,
) -> dict:
    """Build a fresh engine at ``n_threads`` intra-op threads and measure."""
    import psutil
    import torch

    from caedral_voice.inference.engine import WARMUP_TEXT, CaedralEngine

    gc.collect()
    proc = psutil.Process()

    t0 = time.perf_counter()
    engine = CaedralEngine.from_run(run_dir, checkpoint, threads=n_threads,
                                    warmup=True)
    load_seconds = time.perf_counter() - t0

    # Explicit untimed warmup synth (second pass; from_run(warmup=True) does
    # the first). Guarantees one-time frontend/vocoder/kernel costs are paid
    # before ANY timing below, independent of from_run defaults.
    voice = engine.voices[0]
    engine.synth(WARMUP_TEXT, voice["index"])

    baseline_rss = proc.memory_info().rss
    log(f"  threads={n_threads}: engine loaded+warmed in {load_seconds:.1f}s "
        f"(voice={voice['speaker_name']} idx={voice['index']}); "
        f"baseline RSS {_mb(baseline_rss)} MB")

    sent_records: list[dict] = []
    peak_seen = baseline_rss
    ttfa_ub_all: list[float] = []
    ttfa_stream_all: list[float] = []

    for sent in sentences:
        text = sent["text"]

        # -- cold first timed synth (engine warmup deliberately EXCLUDED --
        #    paid in step 2 above; this captures per-input lazy costs).
        audio_cold, perf_cold = measure(engine.synth, text, voice["index"])
        cold_audio_s = len(audio_cold) / engine.sample_rate
        peak_seen = max(peak_seen, perf_cold.peak_rss_bytes)

        # -- warm repeats
        warm_rows = []
        for _ in range(warm_repeats):
            audio_w, perf_w = measure(engine.synth, text, voice["index"])
            peak_seen = max(peak_seen, perf_w.peak_rss_bytes)
            w_audio_s = len(audio_w) / engine.sample_rate
            warm_rows.append({
                "wall_seconds": round(perf_w.wall_seconds, 4),
                "cpu_seconds": round(perf_w.cpu_seconds, 4),
                "audio_seconds": round(w_audio_s, 3),
                "rtf": round(rtf(perf_w.wall_seconds, w_audio_s), 4),
            })
        warm_medians = {
            "wall_seconds": statistics.median(r["wall_seconds"] for r in warm_rows),
            "audio_seconds": statistics.median(r["audio_seconds"] for r in warm_rows),
            "rtf": statistics.median(r["rtf"] for r in warm_rows),
        }

        # -- TTFA (a) upper bound = warm full-synth wall
        ttfa_upper = warm_medians["wall_seconds"]
        ttfa_ub_all.append(ttfa_upper)

        # -- TTFA (b) TRUE streaming estimate (RUNTIME-01 path):
        #    sentence-split, synthesize ONLY chunk 1, time to first audio.
        chunks = split_sentences(text)
        if len(chunks) >= 2:
            c1 = chunks[0]
            engine.synth(c1, voice["index"])          # untimed chunk-1 warmup
            _, perf_c1 = measure(engine.synth, c1, voice["index"])
            peak_seen = max(peak_seen, perf_c1.peak_rss_bytes)
            ttfa_stream: float | None = perf_c1.wall_seconds
            stream_method = "sentence_split_chunk1_synth"
        else:
            ttfa_stream = None
            stream_method = "single_sentence_no_chunk_boundary"
        if ttfa_stream is not None:
            ttfa_stream_all.append(ttfa_stream)

        sent_records.append({
            "id": sent["id"],
            "category": sent["category"],
            "chars": len(text),
            "n_sentence_chunks": len(chunks),
            "cold_first_synth": {
                "definition": "first TIMED synth of this input after the "
                              "explicit engine warmup (excludes engine load "
                              "and one-time warmup costs)",
                "wall_seconds": round(perf_cold.wall_seconds, 4),
                "cpu_seconds": round(perf_cold.cpu_seconds, 4),
                "audio_seconds": round(cold_audio_s, 3),
                "rtf": round(rtf(perf_cold.wall_seconds, cold_audio_s), 4),
                "perf": perf_cold.to_dict(),
            },
            "warm": {
                "repeats": warm_repeats,
                "runs": warm_rows,
                "wall_median_seconds": round(warm_medians["wall_seconds"], 4),
                "audio_seconds_median": round(warm_medians["audio_seconds"], 3),
                "rtf_median": round(warm_medians["rtf"], 4),
                "rtf_min": round(min(r["rtf"] for r in warm_rows), 4),
                "rtf_max": round(max(r["rtf"] for r in warm_rows), 4),
            },
            "ttfa": {
                "upper_bound_seconds": round(ttfa_upper, 4),
                "upper_bound_definition": "warm FULL-sentence synth wall "
                                          "(whole clip ready before first "
                                          "byte reaches the caller)",
                "streaming_method": stream_method,
                "streaming_estimate_seconds": (
                    round(ttfa_stream, 4) if ttfa_stream is not None else None
                ),
                "streaming_definition": "wall-to-first-chunk-audio: sentence-"
                                        "split text, warm chunk-1 shapes, then "
                                        "time synthesizing chunk 1 alone "
                                        "(RUNTIME-01 design path)",
                "chunk1_chars": len(chunks[0]) if chunks else 0,
            },
        })
        log(f"    [{sent['id']}] audio={warm_medians['audio_seconds']:.2f}s "
            f"rtf_warm={warm_medians['rtf']:.3f} "
            f"ttfa_ub={ttfa_upper:.3f}s "
            f"ttfa_stream="
            f"{format(ttfa_stream, '.3f') if ttfa_stream is not None else 'n/a'}s")

    # -- unload + steady-state memory
    del engine
    gc.collect()
    steady_rss = proc.memory_info().rss
    log(f"    steady RSS after unload+gc: {_mb(steady_rss)} MB")

    summary = {
        "rtf_warm_median": round(
            statistics.median(r["warm"]["rtf_median"] for r in sent_records), 4),
        "ttfa_upper_bound_seconds_median": round(statistics.median(ttfa_ub_all), 4),
        "ttfa_streaming_estimate_seconds_median": (
            round(statistics.median(ttfa_stream_all), 4)
            if ttfa_stream_all else None),
        "ram_mb_baseline": _mb(baseline_rss),
        "ram_mb_peak": _mb(peak_seen),
        "ram_mb_steady_after_gc": _mb(steady_rss),
    }
    return {
        "threads": n_threads,
        "torch_num_threads_after_load": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "engine_load_seconds": round(load_seconds, 2),
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes_sampler_max": int(peak_seen),
        "steady_rss_bytes_after_gc": int(steady_rss),
        "rss_note": "peaks from the harness 5 ms sampler within THIS process "
                    "(allocator state carries across configs; compare each "
                    "config against its own baseline_rss_bytes). Win32 "
                    "PeakWorkingSetSize is process-lifetime and reported once "
                    "at top level, never per config.",
        "sentences": sent_records,
        "summary": summary,
        "comparison_row": {
            "model": None,           # filled by caller (needs model_id)
            "status": "measured",
            "dnsmos_median": None,   # natgate join fills quality slots
            "dnsmos_min": None,
            "wer_median": None,
            "rtf": summary["rtf_warm_median"],
            "ram_mb": summary["ram_mb_peak"],
            "size_mb": None,         # filled by caller (checkpoint size)
            "detail": {
                "threads": n_threads,
                "rtf_definition": (
                    "median over sentences of median warm RTF "
                    f"({warm_repeats} timed repeats/sentence; excludes engine "
                    "load and explicit warmup)"),
                "ram_note": "max sampler peak RSS across measured runs in a "
                            "shared process; see rss_note on the config",
                "ttfa_note": "see per-sentence ttfa block: upper bound "
                             "(full synth) and RUNTIME-01 streaming estimate "
                             "(chunk-1-only synth)",
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="training run dir (checkpoints/, phoneme_vocab.json)")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit .pt path (default: last.pt > best.pt > "
                         "newest step_*.pt, engine resolution order)")
    ap.add_argument("--sentences-json", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--warm-repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N sentences (validation runs)")
    ap.add_argument("--model-id", default=None,
                    help="row label for comparison_row (default: "
                         "caedral-<run_dir name>)")
    ap.add_argument("--out", default=None,
                    help=f"output json (default: {DEFAULT_OUT_DIR}/"
                         "run-<timestamp>.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    sentences_path = Path(args.sentences_json)

    sentences = load_sentences(sentences_path)
    if args.limit:
        sentences = sentences[: args.limit]
    if not sentences:
        raise SystemExit("no sentences to measure")
    threads_list = [int(t) for t in str(args.threads).split(",") if t.strip()]
    if not threads_list:
        raise SystemExit("no thread counts requested")

    out_path = (Path(args.out) if args.out else
                DEFAULT_OUT_DIR / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the checkpoint exactly as the engine will, for sha256/size
    # embedding even though the engine resolves independently per build.
    from caedral_voice.inference.engine import CaedralEngine  # resolution only
    ckpt_resolved = CaedralEngine.from_run  # noqa: F841  (import side effects kept minimal)
    from caedral_voice.inference.engine import _resolve_checkpoint
    ckpt_path = _resolve_checkpoint(run_dir, checkpoint)
    ckpt_sha = sha256_file(ckpt_path)
    ckpt_size_mb = _mb(ckpt_path.stat().st_size)
    model_id = args.model_id or f"caedral-{run_dir.name}"

    print(f"PERF-MATRIX  run={run_dir.name}  ckpt={ckpt_path.name} "
          f"({ckpt_size_mb} MB)  threads={threads_list}  "
          f"sentences={[s['id'] for s in sentences]}")

    configs = []
    for i, n in enumerate(threads_list):
        print(f"[{i + 1}/{len(threads_list)}] thread config = {n}")
        cfg = run_thread_config(
            n, sentences, run_dir=run_dir, checkpoint=checkpoint,
            warm_repeats=args.warm_repeats)
        cfg["comparison_row"]["model"] = f"{model_id} (threads={n})"
        cfg["comparison_row"]["size_mb"] = ckpt_size_mb
        configs.append(cfg)

    native = win_process_peaks()
    results = {
        "schema_version": 1,
        "benchmark": "perf_matrix",
        "purpose": "Measured RTF / TTFA / peak-RSS / steady-RAM per torch "
                   "intra-op thread count for CaedralEngine; comparison_row "
                   "objects mirror the natgate comparison-template.json "
                   "fields (rtf / ram_mb / size_mb) for a plain join.",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "machine": capture(),
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_size_mb": ckpt_size_mb,
        "model_id": model_id,
        "sample_rate_hz": 24000,
        "warm_repeats": args.warm_repeats,
        "sentences_source": str(sentences_path),
        "sentences": sentences,
        "thread_configs": configs,
        "process_lifetime": {
            "native_peak_rss_bytes": native[0] if native else None,
            "note": "Win32 PeakWorkingSetSize covers this ENTIRE process "
                    "(all thread configs); valid per-config only when each "
                    "config runs in a fresh process.",
        },
        "protocol_notes": [
            "fresh engine per thread config; previous engine unloaded via "
            "del + gc.collect() before the next build",
            "explicit untimed warmup synth precedes all timing; "
            "'cold_first_synth' therefore excludes engine warmup by design",
            "intra-op threads only (torch.set_num_threads); inter-op pool "
            "left at library default; synthesis is serialized by the engine "
            "lock, so N threads = parallelism INSIDE ops, not concurrent "
            "requests",
            "RTF = wall / audio_seconds, <1.0 faster than real time",
            "TTFA has two flavors per sentence: full-synth upper bound and "
            "harness-side sentence-split chunk-1 streaming estimate "
            "(RUNTIME-01 design path)",
        ],
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # console digest
    print("\nsummary (warm RTF / TTFA-ub / TTFA-stream / peak MB):")
    for cfg in configs:
        s = cfg["summary"]
        ts = s["ttfa_streaming_estimate_seconds_median"]
        print(f"  threads={cfg['threads']:>2}: rtf={s['rtf_warm_median']:.3f}  "
              f"ttfa_ub={s['ttfa_upper_bound_seconds_median']:.3f}s  "
              f"ttfa_stream="
              f"{format(ts, '.3f') + 's' if ts is not None else 'n/a'}  "
              f"peak={s['ram_mb_peak']}MB  "
              f"steady={s['ram_mb_steady_after_gc']}MB")


if __name__ == "__main__":
    main()
