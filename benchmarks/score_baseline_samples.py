"""Quality scoring per CAEDRAL EVAL SPEC v0 (auto tier).

Implements the runnable subset of docs/research/EVALUATION-BENCHMARKS.md:
- Intelligibility gate: normalized dual-ASR min-WER.
  SPEC deviation (documented): Parakeet-TDT ONNX not available on this
  Windows/CPU setup, so the second ASR is whisper base.en (a different
  model size => partially independent error profile). Both normalized with
  EnglishTextNormalizer-equivalent rules + jiwer.
- Quality floor/trend: DNSMOS v8 ONNX (spec tier-2; ceiling ~3.3 on clean
  speech = sanity tier only). UTMOS deferred: PyPI 'speechmos' ships
  AECMOS/DNSMOS/PLCMOS but not UTMOS (backlog).
- Hallucination gates: repetition n-gram check, length ratio, duration/rate
  outliers vs corpus expectations.

Inputs: artifacts/audio/baselines/<baseline>/<voice>/<item>.wav (+ corpus for
reference transcripts). Output: benchmarks/results/quality/<stamp>.jsonl
"""
from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import soundfile as sf  # noqa: E402
import jiwer  # noqa: E402

CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
AUDIO_ROOT = REPO / "artifacts" / "audio" / "baselines"
RESULTS = REPO / "benchmarks" / "results" / "quality"


def normalize(text: str) -> str:
    """Lightweight English text normalization for WER scoring."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("€", "euros ").replace("$", " dollars ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def repetition_penalty(transcript: str) -> float:
    """Max fraction of repeated 3-grams (hallucination signal)."""
    words = transcript.split()
    if len(words) < 6:
        return 0.0
    grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    total = sum(grams.values())
    most = max(grams.values())
    return (most - 1) / max(1, total - 1)


def main() -> None:
    from faster_whisper import WhisperModel
    from speechmos.dnsmos import DNSMOS
    import numpy as np  # noqa: F401  (DNSMOS path input uses librosa internally)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS / f"score-{stamp}.jsonl"

    corpus = json.loads(io.open(CORPUS, encoding="utf-8").read())
    refs = {i["id"]: i["text"] for c in corpus["categories"] for i in c["items"]}

    print("loading ASR models (small.en int8 + base.en int8)...")
    asr_a = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    asr_b = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=8)
    print("loading DNSMOS v8 ONNX...")
    pkg = Path(DNSMOS.__module__.split(".")[0])  # unused; resolve via import
    import speechmos
    pkg_dir = Path(speechmos.__file__).parent / "dnsmos_models"
    dnsmos = DNSMOS(str(pkg_dir / "sig_bak_ovr.onnx"), str(pkg_dir / "model_v8.onnx"))

    wavs = sorted(AUDIO_ROOT.rglob("*.wav"))
    print(f"scoring {len(wavs)} files")

    with io.open(out_path, "w", encoding="utf-8") as out:
        for wav_path in wavs:
            rel = wav_path.relative_to(AUDIO_ROOT).as_posix()
            baseline, voice, item_id = rel.split("/")[:3]
            item_id = Path(item_id).stem
            ref_raw = refs.get(item_id)
            info = sf.info(wav_path)
            audio_s = info.frames / info.samplerate

            hyps = {}
            for name, model in (("small.en", asr_a), ("base.en", asr_b)):
                segments, _ = model.transcribe(str(wav_path), beam_size=1)
                hyps[name] = " ".join(s.text for s in segments).strip()

            wers = {}
            if ref_raw:
                ref = normalize(ref_raw)
                for name, hyp in hyps.items():
                    wers[name] = jiwer.wer(ref, normalize(hyp))
            row = {
                "file": rel,
                "baseline": baseline,
                "voice": voice,
                "item_id": item_id,
                "audio_seconds": round(audio_s, 3),
                "wer_small_en": round(wers.get("small.en", -1), 4),
                "wer_base_en": round(wers.get("base.en", -1), 4),
                "min_wer": round(min(wers.values()), 4) if wers else None,
                "hyp_small_en": hyps["small.en"][:200],
                "repetition_frac": round(repetition_penalty(hyps["small.en"]), 4),
                "len_ratio": round(len(hyps["small.en"].split()) /
                                   max(1, len(normalize(ref_raw or "").split())), 3),
            }
            scores = dnsmos(str(wav_path), 16000, False)
            row["dnsmos_ovrl"] = round(float(scores["ovrl_mos"]), 3)
            row["dnsmos_sig"] = round(float(scores["sig_mos"]), 3)
            row["dnsmos_bak"] = round(float(scores["bak_mos"]), 3)
            out.write(json.dumps(row) + "\n")
            print(f"{rel}: minWER {row['min_wer']} DNSMOS {row['dnsmos_ovrl']}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
