"""Automatic perceptual classification of preview WAVs (EVAL-CLASSIFY-01).

Audit gate: no checkpoint gets quality claims without this audit. Every number
reported below is MEASURED from the audio file itself (dual-ASR WER, repetition,
rate, spectral flatness, DNSMOS); nothing is asserted from training metrics.

Classes (in decision order):
  NOISE           - near-silence or spectrally flat output (vocoder noise floor)
  NATURAL         - candidate natural speech; ALWAYS requires human confirmation
  INTELLIGIBLE    - content matches the reference within tolerance
  PARTIAL_SPEECH  - speech-like fragments recognized, content not reliable
  BABBLE          - fluent-sounding garbage / wrong content

Signals
  1. dual-ASR minWER : faster-whisper small.en int8 and base.en int8 transcribe
                       the wav independently; each scored against the reference
                       text; we keep the minimum (minWER).
  2. repetition      : fraction of duplicated word 3-/4-grams in the better
                       transcript (looping/garbage indicator). n>=3 avoids
                       false positives on legit 2-word repetition in English
                       prose (e.g. anaphora "she would X ... she would Y").
  3. rate plausibility: chars_per_sec = len(reference text or transcript) /
                        audio_seconds (phones_per_sec proxy).
  4. spectral flatness: numpy-only proxy (geometric/arithmetic mean of frame
                       power spectra, voiced-gated). Speech ~0.05-0.35;
                       broadband noise approaches 1.0.
  5. DNSMOS ovrl     : speechmos dnsmos at 16 kHz mono.

THRESHOLDS (chosen from LibriTTS-R native-roundtrip calibration, MEASURED:
clean vocos roundtrip ovrl~3.5, flatness<0.2; step-1000 previews ovrl~1.0):
  SILENCE_RMS            = 1e-4   -> NOISE
  FLATNESS_NOISE_MAX     = 0.50   -> NOISE (mean voiced-frame flatness above this)
  NATURAL                : minWER <= 0.10 AND rep < 0.02 AND dnsmos_ovrl >= 3.2
                           AND rate not extreme  -> label NATURAL, but it is a
                           CANDIDATE ONLY: human confirmation still required.
  INTELLIGIBLE           : minWER <= 0.30
  PARTIAL_SPEECH         : minWER <= 0.60 OR (rate anomaly AND >=2 words
                           recognized) OR (no reference AND any words
                           recognized -- intelligibility cannot be AUDITED
                           without a reference, so the label is capped here).
  else                   : BABBLE
  RATE plausible band    : 5..30 chars/s; EXTREME outside 3..50 chars/s.
  References < 3 words get a short_ref caveat (WER noisy on such clips).

Usage:
  python scripts/classify_preview.py --audio f.wav [g.wav ...] \
      [--reference-texts benchmarks/preview-sentences.json | any json mapping]
      [--json-out results.json]

--reference-texts accepts either a section->sentence(s) mapping (as
benchmarks/preview-sentences.json; sections are matched against the trailing
tokens of the filename, e.g. "...-spk0-neutral.wav" -> "neutral") or a direct
{filename-or-stem: text} mapping.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]

# --- thresholds (see module docstring) -------------------------------------
SILENCE_RMS = 1e-4
FLATNESS_NOISE_MAX = 0.50
NATURAL_MAX_WER = 0.10
NATURAL_MAX_REP = 0.02
NATURAL_MIN_DNSMOS = 3.2
INTELLIGIBLE_MAX_WER = 0.30
PARTIAL_MAX_WER = 0.60
RATE_PLAUSIBLE = (5.0, 30.0)   # chars per second
RATE_EXTREME = (3.0, 50.0)     # outside this blocks NATURAL
SECTION_KEYS = ("neutral", "conversational", "expressive")

THRESHOLDS = {
    "silence_rms": SILENCE_RMS,
    "flatness_noise_max": FLATNESS_NOISE_MAX,
    "natural_max_wer": NATURAL_MAX_WER,
    "natural_max_rep_fraction": NATURAL_MAX_REP,
    "natural_min_dnsmos_ovrl": NATURAL_MIN_DNSMOS,
    "intelligible_max_wer": INTELLIGIBLE_MAX_WER,
    "partial_max_wer": PARTIAL_MAX_WER,
    "rate_plausible_chars_per_sec": list(RATE_PLAUSIBLE),
    "rate_extreme_chars_per_sec": list(RATE_EXTREME),
}

_MODELS: dict[str, object] = {}


# --- signal extraction ------------------------------------------------------
def load_models() -> dict[str, object]:
    """Lazy-load the two faster-whisper models once (small.en/base.en, int8)."""
    if "small.en" not in _MODELS:
        from faster_whisper import WhisperModel
        _MODELS["small.en"] = WhisperModel("small.en", device="cpu",
                                           compute_type="int8")
        _MODELS["base.en"] = WhisperModel("base.en", device="cpu",
                                          compute_type="int8")
    return _MODELS


def transcribe(path: Path, model) -> str:
    segments, _info = model.transcribe(
        str(path), beam_size=5, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True)
    return " ".join(s.text.strip() for s in segments).strip()


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(ref_words: list[str], hyp_words: list[str]) -> float:
    """Standard Levenshtein WER over normalized word lists."""
    if not hyp_words:
        return 1.0
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        cur = [i]
        for j, h in enumerate(hyp_words, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[len(hyp_words)] / max(len(ref_words), 1)


def repetition_fraction(words: list[str]) -> float:
    """Fraction of word n-grams (n=3,4) that occur more than once."""
    best = 0.0
    for n in (3, 4):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        dup = sum(c for c in counts.values() if c > 1)
        best = max(best, dup / len(grams))
    return best


def spectral_flatness(wav: np.ndarray, sr: int,
                      n_fft: int = 1024, hop: int = 512) -> float:
    """Mean voiced-frame spectral flatness (geometric/arithmetic power ratio)."""
    x = wav.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)
    mag2 = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(mag2), axis=1)) / np.mean(mag2, axis=1)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if frame_rms.max() <= 0:
        return 1.0
    voiced = frame_rms > 0.05 * frame_rms.max()
    return float(flat[voiced].mean()) if voiced.any() else float(flat.mean())


def dnsmos_ovrl(wav: np.ndarray, sr: int) -> float:
    """speechmos DNSMOS overall MOS; resamples to the required 16 kHz mono."""
    import torch
    import torchaudio
    from speechmos import dnsmos
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(
            wav.astype(np.float32)), sr, 16000).numpy()
    out = dnsmos.run(np.ascontiguousarray(wav, dtype=np.float32), 16000)
    row = out if isinstance(out, dict) else out.iloc[0].to_dict()
    return float(row.get("ovrl_mos", row.get("ovrl", float("nan"))))


# --- reference handling -----------------------------------------------------
def load_reference_map(path: Path) -> dict[str, str]:
    """Flatten a reference JSON into {key: text} (sections may be lists)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    refs: dict[str, str] = {}
    for k, v in data.items():
        if k in ("purpose", "rules") or not isinstance(k, str):
            continue
        if isinstance(v, list):
            if v and isinstance(v[0], str):
                refs[k] = v[0]
        elif isinstance(v, str):
            refs[k] = v
    return refs


def resolve_reference(refs: dict[str, str], wav_path: Path) -> str | None:
    """Exact filename, then stem, then trailing filename tokens (sections)."""
    name, stem = wav_path.name, wav_path.stem
    if name in refs:
        return refs[name]
    if stem in refs:
        return refs[stem]
    for tok in reversed(stem.split("-")):
        if tok in SECTION_KEYS and tok in refs:
            return refs[tok]
    return None


# --- classification ---------------------------------------------------------
def classify_file(path: Path, reference_text: str | None = None,
                  models: dict | None = None) -> dict:
    path = Path(path)
    notes: list[str] = []
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur_s = len(wav) / sr

    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    flatness = spectral_flatness(wav, sr)
    ovrl = dnsmos_ovrl(wav, sr)

    models = models or load_models()
    hyps = {m: transcribe(path, mod) for m, mod in models.items()}
    ref_words = norm_words(reference_text) if reference_text else []

    wers, best = {}, None
    for m, txt in hyps.items():
        w = word_error_rate(ref_words, norm_words(txt)) if ref_words else None
        wers[m] = w
        if best is None or (w is not None and w < wers[best]):
            best = m
    min_wer = wers[best] if (ref_words and best) else None
    hyp_words = norm_words(hyps[best]) if best else []
    rep = repetition_fraction(hyp_words)

    chars_basis = reference_text if reference_text else (hyps[best] if best else "")
    cps = len(chars_basis) / dur_s if dur_s > 0 else float("inf")
    rate_plausible = RATE_PLAUSIBLE[0] <= cps <= RATE_PLAUSIBLE[1]
    rate_extreme = not (RATE_EXTREME[0] <= cps <= RATE_EXTREME[1])
    if rate_extreme:
        notes.append(f"rate extreme ({cps:.1f} chars/s)")
    if not ref_words:
        notes.append("no reference text: capped below INTELLIGIBLE")
    elif len(ref_words) < 3:
        notes.append("short_ref (<3 words): WER noisy")
    if best:
        notes.append(f"best ASR: {best}")

    # decision ladder (order matters; see docstring)
    if rms <= SILENCE_RMS:
        label = "NOISE"
        notes.append("near-silence")
    elif flatness >= FLATNESS_NOISE_MAX:
        label = "NOISE"
        notes.append("spectral flatness extreme")
    elif (min_wer is not None and min_wer <= NATURAL_MAX_WER
          and rep < NATURAL_MAX_REP and ovrl >= NATURAL_MIN_DNSMOS
          and not rate_extreme):
        label = "NATURAL"
        notes.append("CANDIDATE ONLY - human confirmation required")
    elif min_wer is not None and min_wer <= INTELLIGIBLE_MAX_WER:
        label = "INTELLIGIBLE"
    elif (min_wer is not None and min_wer <= PARTIAL_MAX_WER) \
            or (rate_extreme and len(hyp_words) >= 2) \
            or (not ref_words and len(hyp_words) >= 1):
        label = "PARTIAL_SPEECH"
    else:
        label = "BABBLE"

    return {
        "file": path.name,
        "path": str(path),
        "class": label,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "audio_seconds": round(dur_s, 3),
        "sample_rate": sr,
        "rms": round(rms, 6),
        "spectral_flatness": round(flatness, 4),
        "dnsmos_ovrl": round(ovrl, 3),
        "reference_given": bool(reference_text),
        "wer_small_en": None if wers.get("small.en") is None
                        else round(wers["small.en"], 4),
        "wer_base_en": None if wers.get("base.en") is None
                       else round(wers["base.en"], 4),
        "min_wer": None if min_wer is None else round(min_wer, 4),
        "hypothesis": hyps.get(best, ""),
        "rep_fraction": round(rep, 4),
        "hyp_words": len(hyp_words),
        "chars_per_sec": round(cps, 2) if cps == cps else None,
        "rate_plausible": bool(rate_plausible),
        "notes": notes,
        "thresholds": THRESHOLDS,
    }


def format_line(r: dict) -> str:
    wer = "-" if r["min_wer"] is None else f"{r['min_wer']:.2f}"
    tag = r["class"] + ("*" if r["class"] == "NATURAL" else "")
    return (f"[{tag}] {r['file']} | minWER={wer} rep={r['rep_fraction']:.3f} "
            f"dnsOvrl={r['dnsmos_ovrl']:.2f} "
            f"rate={r['chars_per_sec']}c/s flat={r['spectral_flatness']:.2f} "
            f"dur={r['audio_seconds']}s | {'; '.join(r['notes'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--reference-texts", default=None,
                    help="JSON mapping filenames/stems/sections -> text")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    refs = load_reference_map(Path(args.reference_texts)) \
        if args.reference_texts else {}
    models = load_models()
    results = []
    for a in args.audio:
        p = Path(a)
        res = classify_file(p, resolve_reference(refs, p), models=models)
        print(format_line(res))
        results.append(res)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2),
                                       encoding="utf-8")
        print(f"wrote {args.json_out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
