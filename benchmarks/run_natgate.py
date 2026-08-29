"""Naturalness-gate runner (GATE-NAT-01).

Converts the V1 bar "naturalness competitive with medium modern TTS"
(docs/MASTER-PLAN.md) into a MEASURED numeric gate anchored on locally
measured baselines:

  - Kokoro-82M v1.0 (Apache-2.0): 10 diverse corpus sentences x 2 voices
    (af_heart, am_michael) at library defaults = 20 anchor clips.
  - Piper en_US-lessac-medium (GPL-3.0, isolated subprocess only): same
    10 sentences = 10 second-anchor clips.

Scoring is done by scripts/classify_preview.py signals (dual-ASR minWER,
DNSMOS ovrl via speechmos, rate plausibility), run as a SEPARATE process
per batch so exactly one ASR/MOS process is resident at any time; models
are unloaded between stages via process exit + gc.

The gate itself (frozen before any Caedral V1 audio exists):
  PASS iff  median(DNSMOS ovrl) >= K_median - 0.30
       AND  min(DNSMOS ovrl)    >= K_median - 0.60
where K_median = median DNSMOS ovrl of the 20 Kokoro anchor clips on this
machine. See docs/benchmarks/NATURALNESS-GATE.md for exact numbers and
the full protocol.

Usage:
  python benchmarks/run_natgate.py generate --engine kokoro|piper
  python benchmarks/run_natgate.py compare     # anchors + gate + template
  python benchmarks/run_natgate.py table       # markdown table from template
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf

from benchmarks.harness.measure import measure, rtf, win_process_peaks
from benchmarks.harness.system_info import capture

CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
NATGATE = REPO / "benchmarks" / "results" / "natgate"
KOKORO_MODEL = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES_BIN = REPO / "models" / "kokoro" / "voices-v1.0.bin"

# GATE-NAT-01 frozen protocol subset: one sentence per category across ten
# categories, all >=3 words (WER reliability floor in classify_preview.py),
# excluding edge_cases/longform/dialogue which have their own protocols.
GATE_SENTENCES = [
    "conv_001",   # short_conversational
    "num_001",    # numbers_dates_currency
    "acr_005",    # acronyms_initialisms (known dual-ASR hyphen artifact; fair to all systems)
    "het_002",    # heteronyms ("wind" noun vs verb)
    "pho_001",    # difficult_phonemes (tongue twister)
    "qex_001",    # questions_exclamations
    "frm_001",    # formal_prose
    "nm_001",     # names_places
    "tech_001",   # technical_programming
    "emo_001",    # emotional_expressive
]
GATE_VOICES_KOKORO = ["af_heart", "am_michael"]
GATE_VOICE_PIPER = ["lessac"]
# Peak ceiling for written clips; clips above it get scaled down (headroom
# so the scorer's resampler cannot overshoot past speechmos's [-1, 1] limit).
HEADROOM_CEIL = 0.98

SCORES_KOKORO = NATGATE / "scores-kokoro.json"
SCORES_PIPER = NATGATE / "scores-piper.json"
GEN_META_KOKORO = NATGATE / "gen-meta-kokoro.json"
GEN_META_PIPER = NATGATE / "gen-meta-piper.json"
TEMPLATE_PATH = NATGATE / "comparison-template.json"


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


def generate_kokoro() -> None:
    """20 clips in-process (kokoro-onnx) at defaults speed=1.0 lang=en-us."""
    from kokoro_onnx import Kokoro

    out_dir = NATGATE / "kokoro"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_corpus_items(GATE_SENTENCES)

    t0 = time.perf_counter()
    engine = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES_BIN))
    load_seconds = time.perf_counter() - t0

    rows: list[dict] = []
    for voice in GATE_VOICES_KOKORO:
        engine.create("Warmup.", voice=voice, speed=1.0, lang="en-us")  # cache prime
        for item in items:
            result, perf = measure(
                lambda t=item["text"]: engine.create(t, voice=voice,
                                                     speed=1.0, lang="en-us"))
            samples, sr = result
            # Kokoro float output can exceed [-1, 1]; PCM_16 write would clip
            # to full scale and the scorer's 24k->16k sinc resample then
            # overshoots past +/-1.0, which speechmos rejects. Give any
            # hot clip explicit headroom instead (only rescales samples that
            # are already out of nominal range).
            peak = float(np.abs(samples).max()) if len(samples) else 0.0
            gain_applied = None
            if peak > HEADROOM_CEIL:
                samples = samples * (HEADROOM_CEIL / peak)
                gain_applied = round(HEADROOM_CEIL / peak, 6)
            name = f"natgate-kokoro-{voice}-{item['id']}.wav"
            sf.write(out_dir / name, samples, sr, subtype="PCM_16")
            audio_s = len(samples) / sr
            rows.append({
                "file": name,
                "item_id": item["id"],
                "voice": voice,
                "chars": len(item["text"]),
                "audio_seconds": round(audio_s, 3),
                "sample_rate": int(sr),
                "peak_pre_gain": round(peak, 4),
                "gain_applied": gain_applied,
                "wall_warm_seconds": round(perf.wall_seconds, 4),
                "rtf_warm": round(rtf(perf.wall_seconds, audio_s), 4),
            })
            print(f"[{voice}] {item['id']}: {audio_s:.2f}s audio, RTF {rows[-1]['rtf_warm']}")

    native_peak, _cur = win_process_peaks()

    meta = {
        "stage": "generate-kokoro",
        "baseline": "kokoro-82M-v1.0-onnx",
        "license": "Apache-2.0",
        "protocol": "docs/benchmarks/NATURALNESS-GATE.md",
        "sentences": GATE_SENTENCES,
        "voices": GATE_VOICES_KOKORO,
        "defaults": {"speed": 1.0, "lang": "en-us"},
        "machine": capture(),
        "git_commit": git_commit(),
        "model_load_seconds_inproc": round(load_seconds, 4),
        # Win32 PeakWorkingSetSize of this fresh process = upper bound incl.
        # python+ORT overhead; consistent with earlier ~1.25 GB measurement.
        "process_native_peak_rss_bytes": native_peak,
        "rows": rows,
    }
    GEN_META_KOKORO.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    del engine
    gc.collect()
    print(f"wrote {len(rows)} clips -> {out_dir}; meta -> {GEN_META_KOKORO.name}")


def generate_piper() -> None:
    """10 clips, each synthesis an isolated piper.exe subprocess.

    Reuses benchmarks/run_baseline_piper.py synth_once verbatim (cold+warm;
    the warm clip is saved). GPL isolation policy respected: black-box child
    only, one child at a time.
    """
    from benchmarks.run_baseline_piper import PIPER_EXE, VOICE_MODELS, synth_once

    out_dir = NATGATE / "piper"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_corpus_items(GATE_SENTENCES)

    rows: list[dict] = []
    for voice in GATE_VOICE_PIPER:
        model = VOICE_MODELS[voice]
        for item in items:
            with tempfile.TemporaryDirectory() as td:
                wav_cold = Path(td) / "cold.wav"
                wall_cold, peak_rss = synth_once(PIPER_EXE, model, item["text"], wav_cold)
                wav_warm = Path(td) / "warm.wav"
                wall_warm, _peak2 = synth_once(PIPER_EXE, model, item["text"], wav_warm)
                info = sf.info(wav_warm)
                name = f"natgate-piper-{voice}-{item['id']}.wav"
                # piper.exe emits full-scale audio by design; apply the same
                # headroom rule as kokoro so the scorer's 22k->16k resample
                # cannot overshoot past speechmos's [-1, 1] limit.
                wav_data, wav_sr = sf.read(wav_warm, dtype="float32")
                peak = float(np.abs(wav_data).max()) if len(wav_data) else 0.0
                gain_applied = None
                if peak > HEADROOM_CEIL:
                    wav_data = wav_data * (HEADROOM_CEIL / peak)
                    gain_applied = round(HEADROOM_CEIL / peak, 6)
                    sf.write(out_dir / name, wav_data, wav_sr, subtype="PCM_16")
                else:
                    shutil.copyfile(wav_warm, out_dir / name)
                audio_s = info.frames / info.samplerate
                rows.append({
                    "file": name,
                    "item_id": item["id"],
                    "voice": voice,
                    "chars": len(item["text"]),
                    "audio_seconds": round(audio_s, 3),
                    "sample_rate": int(info.samplerate),
                    "peak_pre_gain": round(peak, 4),
                    "gain_applied": gain_applied,
                    "rtf_cold_subprocess": round(rtf(wall_cold, audio_s), 4),
                    "rtf_warm_subprocess": round(rtf(wall_warm, audio_s), 4),
                    "child_peak_rss_bytes": peak_rss,
                })
                print(f"[{voice}] {item['id']}: {audio_s:.2f}s audio | "
                      f"cold/warm RTF {rows[-1]['rtf_cold_subprocess']}/{rows[-1]['rtf_warm_subprocess']}")

    meta = {
        "stage": "generate-piper",
        "baseline": "piper-en_US-lessac-medium",
        "license": "GPL-3.0 - ISOLATED SUBPROCESS ONLY",
        "protocol": "docs/benchmarks/NATURALNESS-GATE.md",
        "sentences": GATE_SENTENCES,
        "voices": GATE_VOICE_PIPER,
        "defaults": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
        "machine": capture(),
        "git_commit": git_commit(),
        "measurement_note": "wall includes process spawn + model load each call",
        "rows": rows,
    }
    GEN_META_PIPER.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} clips -> {out_dir}; meta -> {GEN_META_PIPER.name}")


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rows", [])
    return data


def summarize_scores(path: Path) -> dict:
    rows = _load_rows(path)
    if not rows:
        raise SystemExit(f"no score rows in {path}")
    ovrl = [r["dnsmos_ovrl"] for r in rows]
    wers = [r["min_wer"] for r in rows if r.get("min_wer") is not None]
    cps = [r["chars_per_sec"] for r in rows if r.get("chars_per_sec") is not None]
    return {
        "n_clips": len(rows),
        "dnsmos_ovrl_values_sorted": sorted(round(v, 3) for v in ovrl),
        "dnsmos_median": round(statistics.median(ovrl), 3),
        "dnsmos_min": round(min(ovrl), 3),
        "dnsmos_max": round(max(ovrl), 3),
        "wer_median": None if not wers else round(statistics.median(wers), 4),
        "wer_max": None if not wers else round(max(wers), 4),
        "rate_chars_per_sec_median": None if not cps else round(statistics.median(cps), 2),
        "rate_plausible_fraction": round(sum(
            1 for r in rows if r.get("rate_plausible")) / len(rows), 3),
    }


def compare() -> None:
    machine = capture()
    kokoro = summarize_scores(SCORES_KOKORO)
    piper = summarize_scores(SCORES_PIPER)

    gen_k = _load_rows(GEN_META_KOKORO)
    gen_p = _load_rows(GEN_META_PIPER)
    meta_k = json.loads(GEN_META_KOKORO.read_text(encoding="utf-8"))
    rtf_k = round(statistics.median([r["rtf_warm"] for r in gen_k]), 4)
    rtf_p_warm = round(statistics.median([r["rtf_warm_subprocess"] for r in gen_p]), 4)
    rtf_p_cold = round(statistics.median([r["rtf_cold_subprocess"] for r in gen_p]), 4)
    ram_p_bytes = int(statistics.median([r["child_peak_rss_bytes"] for r in gen_p]))
    ram_k_bytes = meta_k.get("process_native_peak_rss_bytes")

    def size_mb(*paths: Path) -> float:
        total = sum(p.stat().st_size for p in paths if p.exists())
        return round(total / (1024 * 1024), 1)

    K = kokoro["dnsmos_median"]
    gate = {
        "anchor_model": "kokoro-82M-v1.0 (20 clips)",
        "anchor_dnsmos_median": K,
        "median_threshold": round(K - 0.30, 3),
        "min_floor": round(K - 0.60, 3),
        "formula": ("PASS iff median(DNSMOS ovrl) >= anchor_median - 0.30 AND "
                    "min(DNSMOS ovrl) >= anchor_median - 0.60"),
        "frozen_before_caedral_v1_audio": True,
    }

    template = {
        "schema_version": 1,
        "purpose": ("Comparison table the final Caedral V1 benchmark fills in. "
                    "Each row: one system scored under the frozen GATE-NAT-01 "
                    "protocol (same 10 sentences, classify_preview signals, this "
                    "machine). null = pending local measurement."),
        "protocol_doc": "docs/benchmarks/NATURALNESS-GATE.md",
        "scoring_signals": ("scripts/classify_preview.py (DNSMOS ovrl primary; "
                            "dual-ASR minWER informational; rate plausibility)"),
        "machine": machine,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "gate": gate,
        "rows": {
            "caedral-v1": {
                "model": "caedral-v1",
                "status": "pending",
                "dnsmos_median": None,
                "dnsmos_min": None,
                "wer_median": None,
                "rtf": None,
                "ram_mb": None,
                "size_mb": None,
            },
            "kokoro": {
                "model": "kokoro-82M-v1.0-onnx (af_heart+am_michael)",
                "status": "measured",
                "dnsmos_median": kokoro["dnsmos_median"],
                "dnsmos_min": kokoro["dnsmos_min"],
                "wer_median": kokoro["wer_median"],
                "rtf": rtf_k,
                "ram_mb": None if ram_k_bytes is None else round(ram_k_bytes / (1024 * 1024)),
                "size_mb": size_mb(KOKORO_MODEL, KOKORO_VOICES_BIN),
                "detail": {
                    "n_clips": kokoro["n_clips"],
                    "dnsmos_max": kokoro["dnsmos_max"],
                    "wer_max": kokoro["wer_max"],
                    "rate_chars_per_sec_median": kokoro["rate_chars_per_sec_median"],
                    "rate_plausible_fraction": kokoro["rate_plausible_fraction"],
                    "rtf_definition": "median warm RTF over 20 clips (excludes model load)",
                    "ram_note": "Win32 PeakWorkingSetSize of dedicated generation process (upper bound)",
                },
            },
            "piper": {
                "model": "piper-en_US-lessac-medium (isolated subprocess)",
                "status": "measured",
                "dnsmos_median": piper["dnsmos_median"],
                "dnsmos_min": piper["dnsmos_min"],
                "wer_median": piper["wer_median"],
                "rtf": rtf_p_warm,
                "ram_mb": round(ram_p_bytes / (1024 * 1024), 1),
                "size_mb": size_mb(REPO / "models" / "piper" / "en_US-lessac-medium.onnx"),
                "detail": {
                    "n_clips": piper["n_clips"],
                    "role": "second anchor point; sanity-checks the scorer across two very different systems",
                    "dnsmos_max": piper["dnsmos_max"],
                    "wer_max": piper["wer_max"],
                    "rate_chars_per_sec_median": piper["rate_chars_per_sec_median"],
                    "rate_plausible_fraction": piper["rate_plausible_fraction"],
                    "rtf_cold_subprocess_median": rtf_p_cold,
                    "rtf_definition": "median warm-subprocess RTF over 10 clips (spawn paid every call)",
                    "ram_note": "median child peak RSS sampled externally via psutil",
                },
            },
            "kittentts": {
                "model": "kittentts-nano (Apache-2.0; espeak-ng GPL runtime dep)",
                "status": "pending natgate run",
                "dnsmos_median": None,
                "dnsmos_min": None,
                "wer_median": None,
                "rtf": None,
                "ram_mb": None,
                "size_mb": None,
            },
        },
    }
    TEMPLATE_PATH.write_text(json.dumps(template, indent=2), encoding="utf-8")

    gate_summary = {
        "task": "GATE-NAT-01",
        "measured_at": template["measured_at"],
        "git_commit": template["git_commit"],
        "anchors": {"kokoro": kokoro, "piper": piper},
        "gate": gate,
        "artifacts": {
            "clips": ["benchmarks/results/natgate/kokoro/",
                      "benchmarks/results/natgate/piper/"],
            "scores": [SCORES_KOKORO.name, SCORES_PIPER.name],
            "generation_meta": [GEN_META_KOKORO.name, GEN_META_PIPER.name],
            "comparison_template": TEMPLATE_PATH.name,
        },
    }
    (NATGATE / "gate-summary.json").write_text(
        json.dumps(gate_summary, indent=2), encoding="utf-8")

    print(json.dumps(gate, indent=2))
    render_table(TEMPLATE_PATH)
    print(f"wrote {TEMPLATE_PATH.name} + gate-summary.json")


def render_table(template_path: Path) -> str:
    tpl = json.loads(Path(template_path).read_text(encoding="utf-8"))
    lines = [
        "| Model | DNSMOS med | DNSMOS min | WER med | RTF | RAM MB | Size MB |",
        "|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v: "-" if v is None else v
    for row in tpl["rows"].values():
        lines.append(
            f"| {row['model']} | {fmt(row['dnsmos_median'])} | {fmt(row['dnsmos_min'])} "
            f"| {fmt(row['wer_median'])} | {fmt(row['rtf'])} | {fmt(row['ram_mb'])} "
            f"| {fmt(row['size_mb'])} |")
    table = "\n".join(lines)
    print(table)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="synthesize anchor clips")
    gen.add_argument("--engine", choices=["kokoro", "piper"], required=True)
    sub.add_parser("compare", help="aggregate scores into gate + comparison template")
    tbl = sub.add_parser("table", help="print markdown table from template")
    tbl.add_argument("--template", default=str(TEMPLATE_PATH))
    args = ap.parse_args()
    if args.cmd == "generate":
        {"kokoro": generate_kokoro, "piper": generate_piper}[args.engine]()
    elif args.cmd == "compare":
        compare()
    else:
        render_table(Path(args.template))


if __name__ == "__main__":
    main()
