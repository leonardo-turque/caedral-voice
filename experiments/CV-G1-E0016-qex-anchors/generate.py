"""ANCHOR-QEX (CV-G1-E0016): render missing teacher anchors for qex_002..qex_006.

Prerequisite task flagged by docs/benchmarks/PROSODY-EVAL.md §1 (ANCHOR-QEX row):
Tier C prosody gating currently covers qex_001 only. This renders the remaining
five questions_exclamations items with the NATURALNESS-GATE anchor systems,
following benchmarks/run_natgate.py exactly:
  - Kokoro-82M v1.0 (Apache-2.0), in-process kokoro-onnx, voices af_heart +
    am_michael, defaults speed=1.0 lang=en-us, same headroom rule.
  - Piper en_US-lessac-medium + en_US-ryan-medium (GPL-3.0) via ISOLATED
    piper.exe subprocess per clip (black-box child only; cold+warm like
    natgate, warm clip saved).

RAM policy: engines run in SEPARATE process invocations (--stage kokoro then
--stage piper); the Kokoro ORT session is fully unloaded before Piper spawns.

Output: audio/<teacher>-<voice>-<item>.wav  (+ gen-meta.json, references.json)

Usage:
  python experiments/CV-G1-E0016-qex-anchors/generate.py --stage kokoro
  python experiments/CV-G1-E0016-qex-anchors/generate.py --stage piper
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf

from benchmarks.harness.system_info import capture

CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
KOKORO_MODEL = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES_BIN = REPO / "models" / "kokoro" / "voices-v1.0.bin"

EXP = Path(__file__).resolve().parent
AUDIO = EXP / "audio"

ITEMS = ["qex_002", "qex_003", "qex_004", "qex_005", "qex_006"]
KOKORO_VOICES = ["af_heart", "am_michael"]
PIPER_VOICES = ["lessac", "ryan"]
HEADROOM_CEIL = 0.98  # identical to run_natgate.py


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
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


def generate_kokoro() -> list[dict]:
    from kokoro_onnx import Kokoro

    items = load_corpus_items(ITEMS)
    t0 = time.perf_counter()
    engine = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES_BIN))
    load_seconds = time.perf_counter() - t0

    rows: list[dict] = []
    for voice in KOKORO_VOICES:
        engine.create("Warmup.", voice=voice, speed=1.0, lang="en-us")  # prime
        for item in items:
            wall0 = time.perf_counter()
            samples, sr = engine.create(item["text"], voice=voice,
                                        speed=1.0, lang="en-us")
            wall = time.perf_counter() - wall0
            peak = float(np.abs(samples).max()) if len(samples) else 0.0
            gain_applied = None
            if peak > HEADROOM_CEIL:
                samples = samples * (HEADROOM_CEIL / peak)
                gain_applied = round(HEADROOM_CEIL / peak, 6)
            name = f"kokoro-{voice}-{item['id']}.wav"
            sf.write(AUDIO / name, samples, sr, subtype="PCM_16")
            rows.append({
                "file": name,
                "teacher": "kokoro",
                "voice": voice,
                "item_id": item["id"],
                "chars": len(item["text"]),
                "audio_seconds": round(len(samples) / sr, 3),
                "sample_rate": int(sr),
                "peak_pre_gain": round(peak, 4),
                "gain_applied": gain_applied,
                "wall_warm_seconds": round(wall, 4),
            })
            print(f"[kokoro/{voice}] {item['id']}: {rows[-1]['audio_seconds']}s "
                  f"peak {peak:.3f}")
    meta_load = {"model_load_seconds_inproc": round(load_seconds, 4)}
    del engine
    gc.collect()
    print(f"[kokoro] wrote {len(rows)} clips; model unloaded (gc)")
    return rows, meta_load


def generate_piper() -> list[dict]:
    from benchmarks.run_baseline_piper import PIPER_EXE, VOICE_MODELS, synth_once

    items = load_corpus_items(ITEMS)
    rows: list[dict] = []
    for voice in PIPER_VOICES:
        model = VOICE_MODELS[voice]
        for item in items:
            with tempfile.TemporaryDirectory() as td:
                wav_cold = Path(td) / "cold.wav"
                wall_cold, peak_rss = synth_once(PIPER_EXE, model, item["text"],
                                                 wav_cold)
                wav_warm = Path(td) / "warm.wav"
                wall_warm, _pk2 = synth_once(PIPER_EXE, model, item["text"],
                                             wav_warm)
                info = sf.info(wav_warm)
                name = f"piper-{voice}-{item['id']}.wav"
                wav_data, wav_sr = sf.read(wav_warm, dtype="float32")
                peak = float(np.abs(wav_data).max()) if len(wav_data) else 0.0
                gain_applied = None
                if peak > HEADROOM_CEIL:
                    wav_data = wav_data * (HEADROOM_CEIL / peak)
                    gain_applied = round(HEADROOM_CEIL / peak, 6)
                    sf.write(AUDIO / name, wav_data, wav_sr, subtype="PCM_16")
                else:
                    shutil.copyfile(wav_warm, AUDIO / name)
                rows.append({
                    "file": name,
                    "teacher": "piper",
                    "voice": voice,
                    "item_id": item["id"],
                    "chars": len(item["text"]),
                    "audio_seconds": round(info.frames / info.samplerate, 3),
                    "sample_rate": int(info.samplerate),
                    "peak_pre_gain": round(peak, 4),
                    "gain_applied": gain_applied,
                    "rtf_cold_subprocess": None,
                    "rtf_warm_subprocess": None,
                    "wall_cold_seconds": round(wall_cold, 4),
                    "wall_warm_seconds": round(wall_warm, 4),
                    "child_peak_rss_bytes": int(peak_rss),
                    "_rtf_cold_raw": round(wall_cold / max(info.frames / info.samplerate, 1e-9), 4),
                    "_rtf_warm_raw": round(wall_warm / max(info.frames / info.samplerate, 1e-9), 4),
                })
                print(f"[piper/{voice}] {item['id']}: "
                      f"{rows[-1]['audio_seconds']}s audio")
    print(f"[piper] wrote {len(rows)} clips (each synthesis an isolated child)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["kokoro", "piper"], required=True)
    args = ap.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)

    if args.stage == "kokoro":
        rows, extra = generate_kokoro()  # type: ignore[misc]
        meta = {
            "stage": "generate-kokoro",
            "experiment": "CV-G1-E0016-qex-anchors",
            "task": "ANCHOR-QEX",
            "baseline": "kokoro-82M-v1.0-onnx",
            "license": "Apache-2.0",
            "protocol": ["benchmarks/run_natgate.py", "docs/benchmarks/PROSODY-EVAL.md"],
            "items": ITEMS,
            "voices": KOKORO_VOICES,
            "defaults": {"speed": 1.0, "lang": "en-us"},
            "headroom_ceil": HEADROOM_CEIL,
            "machine": capture(),
            "git_commit": git_commit(),
            "measured_at": datetime.now(timezone.utc).isoformat(),
            **extra,
            "rows": rows,
        }
    else:
        rows = generate_piper()
        for r in rows:  # promote RTF fields
            r["rtf_cold_subprocess"] = r.pop("_rtf_cold_raw")
            r["rtf_warm_subprocess"] = r.pop("_rtf_warm_raw")
        meta = {
            "stage": "generate-piper",
            "experiment": "CV-G1-E0016-qex-anchors",
            "task": "ANCHOR-QEX",
            "baseline": f"piper-en_US-{{{','.join(PIPER_VOICES)}}}-medium",
            "license": "GPL-3.0 - ISOLATED SUBPROCESS ONLY (black-box child)",
            "protocol": ["benchmarks/run_natgate.py", "docs/benchmarks/PROSODY-EVAL.md"],
            "items": ITEMS,
            "voices": PIPER_VOICES,
            "defaults": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
            "machine": capture(),
            "git_commit": git_commit(),
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "measurement_note": "wall includes process spawn + model load each call",
            "rows": rows,
        }

    (EXP / f"gen-meta-{args.stage}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # refresh combined reference map for the scorer ({stem: text})
    refs_path = EXP / "references.json"
    refs: dict[str, str] = {}
    if refs_path.exists():
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
    for r in rows:
        text = next(i["text"] for i in load_corpus_items([r["item_id"]]))
        refs[Path(r["file"]).stem] = text
    refs_path.write_text(json.dumps(refs, indent=2), encoding="utf-8")

    print(f"meta -> gen-meta-{args.stage}.json; references.json updated")


if __name__ == "__main__":
    main()
