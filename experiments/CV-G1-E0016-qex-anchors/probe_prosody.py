"""ANCHOR-QEX prosody probes: PROSODY-EVAL §2 P1 + P3 implemented inline.

docs/benchmarks/PROSODY-EVAL.md carries these detectors as formulas only (no
implementation file exists in-repo); this script is the minimal faithful
implementation, used identically on every clip (§8.5: anchors and candidates
must share one code path).

P1 rise flag  B = max(F0 final 700 ms) / median(F0 utterance);  rise iff B >= 1.15.
   Front-end: mono float -> librosa.effects.split(top_db=40) trim ->
   librosa.pyin(fmin=60, fmax=400, frame_length=1024, hop_length=256).
   Joint diagnostic C = linear fit of F0 (semitones rel. utterance median) over
   final 800 ms voiced frames [st/s] — LOGGED ONLY, must never gate.
   Windows with <3 voiced frames -> None (calibration footnote convention).
P3 pause accounting: 10 ms frames RMS dB; silence = frame <
   max(P90_frame_dB - 35, -50 dBFS); internal runs (leading/trailing excluded);
   P150 = runs >=150 ms, P350 = runs >=350 ms. Expected boundaries E and long
   marks L come from the REAL frontend token stream (caedral_voice.text.g2p:
   ',;:'->'|', '.!?'->'‖', adjacent-dup collapsed, trailing stripped).
   Pass rule (upper-bound): P150 <= E+2 AND no internal silence >700 ms AND
   (L>=1 => at least one run >=200 ms).

Runs over the 20 new CV-G1-E0016 clips plus the 3 existing qex_001 natgate
anchor clips (same code path for cross-experiment comparability).
Output: prosody.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.text.g2p import g2p

EXP = Path(__file__).resolve().parent
NATGATE = REPO / "benchmarks" / "results" / "natgate"
CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"

NEW_CLIPS = sorted(str(p) for p in (EXP / "audio").glob("*.wav"))
LEGACY_QEX001 = [
    NATGATE / "kokoro" / "natgate-kokoro-af_heart-qex_001.wav",
    NATGATE / "kokoro" / "natgate-kokoro-am_michael-qex_001.wav",
    NATGATE / "piper" / "natgate-piper-lessac-qex_001.wav",
]

RISE_FLAG_THRESHOLD = 1.15
FINAL_WINDOW_B_S = 0.7
FINAL_WINDOW_C_S = 0.8
MIN_VOICED_WINDOW = 3


def corpus_text(item_id: str) -> str:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    for cat in data["categories"]:
        for item in cat["items"]:
            if item["id"] == item_id:
                return item["text"]
    raise SystemExit(f"unknown item {item_id}")


def expected_pauses(text: str) -> tuple[int, int]:
    """E = count(|)+count(‖), L = count(‖) from the real G2P token stream."""
    tokens = g2p(text).phoneme_tokens
    e = tokens.count("|") + tokens.count("‖")
    l = tokens.count("‖")
    return e, l


def p1_rise(y_trim: np.ndarray, sr: int) -> dict:
    f0, vflag, vprob = librosa.pyin(
        y_trim, fmin=60, fmax=400, frame_length=1024, hop_length=256, sr=sr)
    t = librosa.times_like(f0, sr=sr, hop_length=256)
    valid = ~np.isnan(f0)
    vfrac = float(valid.mean())
    out: dict = {"voiced_fraction": round(vfrac, 4)}
    if not valid.any():
        out.update({"median_f0_hz": None, "B": None, "C_slope_st_per_s": None})
        return out
    med = float(np.nanmedian(f0))
    out["median_f0_hz"] = round(med, 2)
    dur = len(y_trim) / sr

    sel_b = t >= max(dur - FINAL_WINDOW_B_S, 0.0)
    vals_b = f0[sel_b & valid]
    if len(vals_b) >= MIN_VOICED_WINDOW:
        out["B"] = round(float(vals_b.max()) / med, 4)
        out["B_window_max_f0_hz"] = round(float(vals_b.max()), 2)
    else:
        out["B"] = None
        out["B_window_max_f0_hz"] = None

    sel_c = t >= max(dur - FINAL_WINDOW_C_S, 0.0)
    tt = t[sel_c & valid]
    ss = 12.0 * np.log2(f0[sel_c & valid] / med)
    if len(tt) >= MIN_VOICED_WINDOW:
        out["C_slope_st_per_s"] = round(float(np.polyfit(tt, ss, 1)[0]), 3)
    else:
        out["C_slope_st_per_s"] = None
    return out


def p3_pauses(y: np.ndarray, sr: int, E: int, L: int) -> dict:
    fl = int(round(0.010 * sr))
    n = int(np.ceil(len(y) / fl))
    yp = np.pad(y.astype(np.float64), (0, n * fl - len(y)))
    frames = yp.reshape(n, fl)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = 20.0 * np.log10(rms + 1e-12)
    thr = max(float(np.percentile(db, 90)) - 35.0, -50.0)
    silent = db < thr

    # internal silence runs: leading/trailing excluded, duration = frames x 10 ms
    runs_ms: list[float] = []
    i = 0
    while i < n:
        if silent[i]:
            j = i
            while j + 1 < n and silent[j + 1]:
                j += 1
            if i > 0 and j < n - 1:
                runs_ms.append(round((j - i + 1) * fl / sr * 1000.0, 1))
            i = j + 1
        else:
            i += 1

    p150 = sum(1 for d in runs_ms if d >= 150.0)
    p350 = sum(1 for d in runs_ms if d >= 350.0)
    max_internal_ms = max(runs_ms) if runs_ms else 0.0
    long_pause_ok = True if L == 0 else any(d >= 200.0 for d in runs_ms)
    rule_pass = bool(p150 <= E + 2 and max_internal_ms <= 700.0 and long_pause_ok)
    return {
        "expected_internal_boundaries_E": E,
        "long_marks_L": L,
        "silence_threshold_db": round(thr, 2),
        "internal_silence_runs_ms": runs_ms,
        "P150": int(p150),
        "P350": int(p350),
        "max_internal_silence_ms": float(max_internal_ms),
        "long_pause_realized": (None if L == 0 else long_pause_ok),
        "p3_rule_pass": rule_pass,
    }


def main() -> None:
    rows = []
    clips = [(p, "CV-G1-E0016", Path(p).stem) for p in NEW_CLIPS]
    clips += [(str(p), "natgate-legacy-qex_001", p.stem.replace("natgate-", ""))
              for p in LEGACY_QEX001]

    for path, origin, stem in clips:
        t0 = time.perf_counter()
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # parse teacher-voice-item from stem (legacy stems end voice-item too)
        parts = stem.split("-")
        item_id = parts[-1]
        voice = parts[-2]
        teacher = parts[-3] if len(parts) >= 3 else "kokoro"

        text = corpus_text(item_id)
        E, L = expected_pauses(text)

        intervals = librosa.effects.split(wav, top_db=40)
        y_trim = wav[intervals[0][0]:intervals[-1][1]] if len(intervals) else wav

        p1 = p1_rise(y_trim, sr)
        p3 = p3_pauses(wav, sr, E, L)

        row = {
            "file": Path(path).name,
            "path": str(Path(path).relative_to(REPO)),
            "origin": origin,
            "teacher": teacher,
            "voice": voice,
            "item_id": item_id,
            "text": text,
            "audio_seconds": round(len(wav) / sr, 3),
            "trimmed_seconds": round(len(y_trim) / sr, 3),
            "sample_rate": int(sr),
            "probe_seconds_wall": round(time.perf_counter() - t0, 2),
            **p1,
            **p3,
        }
        rows.append(row)
        b = row["B"]
        c = row["C_slope_st_per_s"]
        print(f"{stem}: B={b} C={c} vfrac={row['voiced_fraction']} "
              f"E={E} L={L} P150={row['P150']} P350={row['P350']} "
              f"maxSil={row['max_internal_silence_ms']}ms "
              f"pass={row['p3_rule_pass']} ({row['probe_seconds_wall']}s)")

    payload = {
        "experiment": "CV-G1-E0016-qex-anchors",
        "task": "ANCHOR-QEX",
        "detector_source": ("docs/benchmarks/PROSODY-EVAL.md §2 formulas; no "
                            "in-repo implementation existed — minimal inline "
                            "implementation, identical code path for all clips"),
        "rise_flag_threshold": RISE_FLAG_THRESHOLD,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "rows": rows,
    }
    (EXP / "prosody.json").write_text(json.dumps(payload, indent=2),
                                      encoding="utf-8")
    print(f"wrote prosody.json ({len(rows)} clips)")


if __name__ == "__main__":
    main()
