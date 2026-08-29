"""Baseline benchmark: Kokoro-82M v1.0 via kokoro-onnx (Apache-2.0).

Produces the program's first MEASURED CPU numbers against the frozen corpus.
Every result row embeds machine baseline + corpus version + git commit so no
number can ever circulate without provenance.

Usage:
  python benchmarks/run_baseline_kokoro.py [--items id1,id2,...] [--voices af_heart,am_michael]
  [--threads N]
Results: benchmarks/results/kokoro-v1.0/run-<timestamp>.jsonl (+ .meta.json)

Notes:
- Thread count is applied via OMP/ORT environment variables BEFORE onnxruntime
  loads; it cannot be changed after session creation.
- kokoro_onnx.create_stream is an async generator; the TTFA probe runs under
  asyncio and reports time-to-first-nonempty-chunk.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.harness.measure import measure, rtf  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402

MODEL = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "kokoro" / "voices-v1.0.bin"
CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
RESULTS = REPO / "benchmarks" / "results" / "kokoro-v1.0"

DEFAULT_ITEMS = ["conv_001", "num_001", "acr_005", "het_002", "para_001", "long_001"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_corpus_items(ids: list[str]) -> list[dict]:
    data = json.loads(io.open(CORPUS, encoding="utf-8").read())
    by_id = {
        item["id"]: {"id": item["id"], "text": item["text"], "category": cat["id"]}
        for cat in data["categories"]
        for item in cat["items"]
    }
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown corpus item ids: {missing}")
    return [by_id[i] for i in ids]


async def stream_ttfa(engine, text: str, voice: str, speed: float) -> dict:
    first_chunk_at = None
    n_chunks = 0
    total_samples = 0
    sr_last = None
    t0 = time.perf_counter()
    async for chunk_samples, chunk_sr in engine.create_stream(
        text, voice=voice, speed=speed, lang="en-us"
    ):
        if first_chunk_at is None and len(chunk_samples):
            first_chunk_at = time.perf_counter() - t0
        n_chunks += 1
        total_samples += len(chunk_samples)
        sr_last = chunk_sr
    return {
        "ttfa_seconds": round(first_chunk_at, 4) if first_chunk_at else None,
        "stream_chunks": n_chunks,
        "stream_total_audio_seconds": round(total_samples / sr_last, 3),
        "stream_total_wall_seconds": round(time.perf_counter() - t0, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS))
    ap.add_argument("--voices", default="af_heart,am_michael")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/intra-op threads (0=library default); must precede ORT load")
    ap.add_argument("--stream-only", action="store_true",
                    help="skip per-item synthesis; only run the TTFA probe")
    args = ap.parse_args()

    if args.threads > 0:
        for var in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY"):
            os.environ[var] = str(args.threads) if var == "OMP_NUM_THREADS" else "PASSIVE"
        # onnxruntime reads this only at first session creation.
        os.environ["ORT_INTRA_OP_NUM_THREADS"] = ""
        del os.environ["ORT_INTRA_OP_NUM_THREADS"]

    from kokoro_onnx import Kokoro

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows_path = RESULTS / f"run-{stamp}.jsonl"
    meta_path = RESULTS / f"run-{stamp}.meta.json"

    t0 = time.perf_counter()
    engine = Kokoro(str(MODEL), str(VOICES))
    load_seconds = time.perf_counter() - t0

    header = {
        "baseline": "kokoro-82M-v1.0-onnx",
        "license": "Apache-2.0 (code+weights)",
        "machine": capture(),
        "corpus_version": json.loads(io.open(CORPUS, encoding="utf-8").read())["version"],
        "git_commit": git_commit(),
        "omp_num_threads": args.threads or None,
        "model_load_seconds_inproc": round(load_seconds, 4),
        "speed_param": args.speed,
        "api_note": "warm RTF excludes one-time model load; cold process wall measured externally",
    }
    meta_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    voices = [v.strip() for v in args.voices.split(",")]
    items = load_corpus_items([i.strip() for i in args.items.split(",")])
    print(f"model loaded in {load_seconds:.2f}s; benchmarking {len(items)} items x {len(voices)} voices")

    with io.open(rows_path, "w", encoding="utf-8") as out:
        if args.stream_only:
            items = [next(i for i in items if i["id"] == "para_001")]
        for voice in voices:
            if not args.stream_only:
                _, warm_perf = measure(
                    lambda: engine.create("Warmup sentence for cache priming.",
                                          voice=voice, speed=args.speed, lang="en-us")
                )
                for item in items:
                    result, perf = measure(
                        lambda t=item["text"]: engine.create(t, voice=voice,
                                                             speed=args.speed, lang="en-us")
                    )
                    samples, sr = result
                    audio_s = len(samples) / sr
                    row = {
                        "item_id": item["id"],
                        "category": item["category"],
                        "chars": len(item["text"]),
                        "voice": voice,
                        "audio_seconds": round(audio_s, 3),
                        "sample_rate": int(sr),
                        "rtf_warm": round(rtf(perf.wall_seconds, audio_s), 4),
                        **{f"perf_{k}": v for k, v in perf.to_dict().items()},
                        "warmup_wall_seconds": round(warm_perf.wall_seconds, 4),
                    }
                    out.write(json.dumps(row) + "\n")
                    print(f"[{voice}] {item['id']}: {audio_s:.2f}s audio, "
                          f"{perf.wall_seconds:.2f}s wall, RTF {row['rtf_warm']}")

        for voice in voices:
            probe_text = next(i["text"] for i in items if i["id"] == "para_001") \
                if any(i["id"] == "para_001" for i in items) else items[0]["text"]
            probe = measure(
                lambda v=voice, t=probe_text: asyncio.run(stream_ttfa(engine, t, v, args.speed))
            )[0]
            row = {"item_id": "STREAM_TTFA_para_001", "category": "streaming",
                   "voice": voice, "probe_chars": len(probe_text), **probe}
            out.write(json.dumps(row) + "\n")
            print(f"[{voice}] TTFA {probe['ttfa_seconds']}s over "
                  f"{probe['stream_chunks']} chunks")

    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
