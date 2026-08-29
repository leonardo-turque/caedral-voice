"""Baseline benchmark: KittenTTS mini/nano (Apache-2.0) via kittentts pip package.

Requires espeak-ng at runtime (GPL) — provided WITHOUT system install via the
espeakng-loader wheel, wired into phonemizer before engine init. This is
acceptable ONLY because KittenTTS is a measured BASELINE, never a Caedral
dependency (see docs/research/TEXT-FRONTEND.md GPL findings).

No streaming API exists -> TTFA recorded as null.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Wire pip-provided espeak-ng into phonemizer BEFORE any backend construction.
import espeakng_loader  # noqa: E402
import phonemizer.backend.espeak.wrapper as _wrapper  # noqa: E402

_wrapper.EspeakWrapper.set_library(espeakng_loader.get_library_path())
_wrapper.EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

from benchmarks.harness.measure import measure, rtf  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402

CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
RESULTS = REPO / "benchmarks" / "results" / "kittentts"

DEFAULT_ITEMS = ["conv_001", "num_001", "acr_005", "het_002", "para_001", "long_001"]


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
    return [by_id[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS))
    ap.add_argument("--voices", default="expr-voice-2-m,expr-voice-3-f")
    args = ap.parse_args()

    from kittentts import KittenTTS

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows_path = RESULTS / f"run-{stamp}.jsonl"
    meta_path = RESULTS / f"run-{stamp}.meta.json"

    t0 = time.perf_counter()
    engine = KittenTTS()  # downloads ONNX weights from HF hub on first use
    load_seconds = time.perf_counter() - t0

    header = {
        "baseline": "kittentts (nano-class ONNX)",
        "license": "Apache-2.0 (weights+code); runtime dep espeak-ng GPL via espeakng-loader",
        "machine": capture(),
        "corpus_version": json.loads(io.open(CORPUS, encoding="utf-8").read())["version"],
        "git_commit": git_commit(),
        "model_init_seconds_incl_download": round(load_seconds, 4),
        "streaming": None,
    }
    meta_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    items = load_corpus_items([i.strip() for i in args.items.split(",")])
    print(f"init {load_seconds:.2f}s; benchmarking {len(items)} items x {args.voices}")

    with io.open(rows_path, "w", encoding="utf-8") as out:
        for voice in [v.strip() for v in args.voices.split(",")]:
            _, _ = measure(lambda: engine.generate("Warmup sentence.", voice=voice))
            for item in items:
                result, perf = measure(lambda t=item["text"], v=voice: engine.generate(t, voice=v))
                audio_s = len(result) / 24000
                row = {
                    "item_id": item["id"],
                    "category": item["category"],
                    "chars": len(item["text"]),
                    "voice": voice,
                    "audio_seconds": round(audio_s, 3),
                    "sample_rate": 24000,
                    "rtf_warm": round(rtf(perf.wall_seconds, audio_s), 4),
                    **{f"perf_{k}": v for k, v in perf.to_dict().items()},
                }
                out.write(json.dumps(row) + "\n")
                print(f"[{voice}] {item['id']}: {audio_s:.2f}s audio, "
                      f"{perf.wall_seconds:.2f}s wall, RTF {row['rtf_warm']}")

    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
