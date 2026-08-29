"""Baseline benchmark: Piper TTS via official Windows binary (GPL-3.0).

ISOLATION POLICY (per docs/research licensing findings): Piper is GPL-3.0 and
must never become a linked dependency of Caedral code. This driver therefore
treats piper.exe strictly as an external black-box process — exactly how we
would ship it in production if ever needed.

Measurement honesty:
- Each synthesis is a fresh subprocess: wall time INCLUDES process spawn +
  model load. Rows record it as `cold_subprocess` (first run) vs
  `warm_subprocess` (immediate repeat; OS file cache hot, spawn still paid).
- Child peak RSS is sampled from outside via psutil (no GIL blindness).

Usage:
  python benchmarks/run_baseline_piper.py [--items id1,id2] [--voices lessac,ryan]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import psutil  # noqa: E402
import soundfile as sf  # noqa: E402

from benchmarks.harness.measure import rtf  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402

PIPER_EXE = REPO / "models" / "piper" / "piper" / "piper.exe"
CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
RESULTS = REPO / "benchmarks" / "results" / "piper"

DEFAULT_ITEMS = ["conv_001", "num_001", "acr_005", "het_002", "para_001", "long_001"]
VOICE_MODELS = {
    "lessac": REPO / "models" / "piper" / "en_US-lessac-medium.onnx",
    "ryan": REPO / "models" / "piper" / "en_US-ryan-medium.onnx",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_corpus_items(ids):
    data = json.loads(io.open(CORPUS, encoding="utf-8").read())
    by_id = {
        item["id"]: {"id": item["id"], "text": item["text"], "category": cat["id"]}
        for cat in data["categories"] for item in cat["items"]
    }
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown corpus item ids: {missing}")
    return [by_id[i] for i in ids]


def synth_once(exe: Path, model: Path, text: str, wav_out: Path) -> tuple[float, int]:
    """Run one isolated synthesis; return (wall_seconds, child_peak_rss_bytes)."""
    t0 = time.perf_counter()
    proc = psutil.Popen(
        [str(exe), "--model", str(model), "--output_file", str(wav_out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    peak = [0]

    def sample():
        while proc.poll() is None:
            try:
                rss = proc.memory_info().rss
                if rss > peak[0]:
                    peak[0] = rss
            except Exception:
                break
            time.sleep(0.005)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        proc.wait(timeout=1200)
    finally:
        th.join(timeout=1.0)
    wall = time.perf_counter() - t0
    return wall, peak[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS))
    ap.add_argument("--voices", default="lessac,ryan")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows_path = RESULTS / f"run-{stamp}.jsonl"
    meta_path = RESULTS / f"run-{stamp}.meta.json"

    header = {
        "baseline": "piper-en_US-medium (2023.11.14-2 binaries)",
        "license": "GPL-3.0 (binaries+voices MIT/CC?) — ISOLATED SUBPROCESS ONLY",
        "machine": capture(),
        "corpus_version": json.loads(io.open(CORPUS, encoding="utf-8").read())["version"],
        "git_commit": git_commit(),
        "measurement_note": "wall includes process spawn + model load each call",
    }
    meta_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    items = load_corpus_items([i.strip() for i in args.items.split(",")])
    print(f"benchmarking {len(items)} items x {args.voices} (isolated subprocess)")

    with io.open(rows_path, "w", encoding="utf-8") as out:
        for voice in args.voices.split(","):
            model = VOICE_MODELS[voice.strip()]
            for item in items:
                with tempfile.TemporaryDirectory() as td:
                    wav1 = Path(td) / "cold.wav"
                    wall_cold, peak = synth_once(PIPER_EXE, model, item["text"], wav1)
                    info = sf.info(wav1)
                    audio_s = info.frames / info.samplerate
                    # immediate repeat = OS caches warm, spawn still included
                    wav2 = Path(td) / "warm.wav"
                    wall_warm, _ = synth_once(PIPER_EXE, model, item["text"], wav2)
                    row = {
                        "item_id": item["id"],
                        "category": item["category"],
                        "chars": len(item["text"]),
                        "voice": voice,
                        "audio_seconds": round(audio_s, 3),
                        "sample_rate": int(info.samplerate),
                        "rtf_cold_subprocess": round(rtf(wall_cold, audio_s), 4),
                        "rtf_warm_subprocess": round(rtf(wall_warm, audio_s), 4),
                        "wall_cold_seconds": round(wall_cold, 4),
                        "wall_warm_seconds": round(wall_warm, 4),
                        "child_peak_rss_bytes": peak,
                    }
                    out.write(json.dumps(row) + "\n")
                    print(f"[{voice}] {item['id']}: {audio_s:.2f}s audio | "
                          f"cold {row['rtf_cold_subprocess']} / warm {row['rtf_warm_subprocess']}")

    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
