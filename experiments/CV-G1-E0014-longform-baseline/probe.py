"""CV-G1-E0014-longform-baseline: first measured long-form numbers (V1 before).

Runs the LONGFORM-DESIGN.md v0 chunking policy (MAX_CHARS=300 sentence-guarded
greedy packing, HARD=420 overflow rule) over corpus item ``long_001``
(benchmarks/corpus/eval-corpus-v1.0.json, 5687 chars) and synthesizes every
chunk through the current diagnostic prototype:

    CaedralEngine (caedral_voice.inference) on the newest
    experiments/pilot-tiny-15k/checkpoints/step_*.pt, voice '103'
    (internal index 0), noise_scale = engine default 0.25 (E0009),
    torch threads = 4 (CLI default).

Chunks are concatenated with flat 120 ms silences (task-specified baseline
pause; the design §1.2 document-global pause table is post-V1 work) and
written to artifacts/audio/longform-baseline/long_001-spk103.wav plus
per-chunk wavs for independent scoring.

Measurements (all MEASURED here; evidence labels per charter §9):
  - wall-clock synthesis time and effective RTF vs produced audio duration;
  - dual-ASR minWER PER CHUNK (faster-whisper small.en/base.en int8, beam 5;
    scripts/classify_preview.py helpers reused verbatim) against each chunk's
    own reference = the chunk text itself;
  - per-chunk minWER population variance vs the design stability gate
    (<= 0.0025) and the catastrophic-chunk gate (no chunk minWER > 0.15);
  - repetition-fraction per chunk (<= 0.02) and document-wide (<= 0.01),
    compression ratio (<= 1.5), trailing-loop check (design §3 gates);
  - waveform clipping check on the concatenated document audio;
  - ONE control: the same pipeline on a 2-sentence text (short-path check).

Usage:
  .venv/Scripts/python.exe experiments/CV-G1-E0014-longform-baseline/probe.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))          # classify_preview helpers

import numpy as np                                  # noqa: E402
import soundfile as sf                              # noqa: E402

CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
RUN_DIR = REPO / "experiments" / "pilot-tiny-15k"
OUT_DIR = REPO / "artifacts" / "audio" / "longform-baseline"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

MAX_CHARS = 300                                     # design v0 soft target
HARD_MAX = 420                                      # design v0 never-exceed
PAUSE_S = 0.120                                     # task-specified baseline gap
VOICE_NAME = "103"                                  # internal index 0 (speaker_index)
DOC_ID = "long_001"

CONTROL_TEXT = (
    "The lighthouse keeper trimmed the wick at dusk. "
    "Fog swallowed the harbor before the lamp was even lit."
)

# ---------------------------------------------------------------- chunker --
# Abbreviation guard: the design-doc initial list exactly (Mr., Dr., e.g.,
# i.e., etc.), plus structural rules for digit-dot ("1992.") and
# single-capital initials ("J. F. Kennedy").
ABBR = {"mr", "dr", "e.g", "i.e", "etc"}


def _is_abbrev_guard(text: str, i: int, j: int) -> bool:
    """True when the [.!?]+ run at text[i:j] must NOT end a sentence."""
    run = text[i:j]
    if run != ".":                                   # !/? always terminate
        return False
    prev = text[i - 1] if i > 0 else ""
    if prev.isdigit():                               # digits + '.' rule
        return True
    if prev.isupper() and (i - 2 < 0 or not (text[i - 2].isalnum()
                                             or text[i - 2] == "'")):
        return True                                  # single capital initial
    m = re.search(r"[A-Za-z][A-Za-z.]*", text[:i][::-1])
    if m:
        word = m.group(0)[::-1].lower().rstrip(".")
        return word in ABBR
    return False


def _split_words(text: str, limit: int) -> list[str]:
    """Greedy word-boundary packing into pieces each <= limit."""
    words = text.split()
    if not words:
        return []
    pieces, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= limit:
            cur = cur + " " + w
        else:
            pieces.append(cur)
            cur = w
    pieces.append(cur)
    return pieces


def _split_overflow(sentence: str, hard: int = HARD_MAX) -> list[str]:
    """§1.1(4) overflow fallback: split at clauses (`,` `;` `:`), then words.

    Guarantees every returned piece is <= hard (a pathological single token
    longer than `hard` is emitted as its own piece and reported by the caller's
    length check).
    """
    if len(sentence) <= hard:
        return [sentence]
    clauses = re.split(r"(?<=[,;:])\s+", sentence)
    pieces: list[str] = []
    cur: str | None = None
    lock = False
    for c in clauses:
        if cur is None:
            cur = c
        elif lock:
            pieces.append(cur)
            cur = c
            lock = False
        elif len(c) > hard:
            pieces.append(cur)
            cur = c
            lock = True        # word-split this clause alone; no re-merging
        elif len(cur) + 1 + len(c) <= hard:
            cur = cur + " " + c
        else:
            pieces.append(cur)
            cur = c
    if cur is not None:
        pieces.append(cur)
    final: list[str] = []
    for p in pieces:
        final.extend(_split_words(p, hard) if len(p) > hard else [p])
    return final


def _sentence_units(paragraph: str) -> list[str]:
    """Sentences of ONE paragraph (never crossing \\n\\n), overflow-split."""
    sentences: list[str] = []
    start, i, n = 0, 0, len(paragraph)
    while i < n:
        ch = paragraph[i]
        if ch in ".!?":
            j = i + 1
            while j < n and paragraph[j] in ".!?":
                j += 1
            if (j >= n or paragraph[j].isspace()) and not _is_abbrev_guard(
                    paragraph, i, j):
                sentences.append(paragraph[start:j].strip())
                start = j
            i = j
        else:
            i += 1
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    units: list[str] = []
    for s in sentences:
        units.extend(_split_overflow(s))
    return units


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """LONGFORM-DESIGN v0 §1.1 splitter: pure function of text, deterministic.

    Paragraphs (\\n\\n+) are always chunk boundaries; sentences are detected
    within a paragraph under the abbreviation guards; greedy packing merges
    consecutive sentences while chars <= MAX_CHARS (300); the overflow
    fallback keeps every chunk <= HARD_MAX (420).
    """
    out: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        cur = ""
        for unit in _sentence_units(para):
            if not cur:
                cur = unit
            elif len(cur) + 1 + len(unit) <= max_chars:
                cur = cur + " " + unit
            else:
                out.append(cur)
                cur = unit
        if cur:
            out.append(cur)
    return out


# --------------------------------------------------------- corpus helpers --
def load_corpus_item(item_id: str) -> dict:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    for cat in data["categories"]:
        for it in cat.get("items", []):
            if it.get("id") == item_id:
                return it
    raise KeyError(f"corpus item {item_id!r} not found in {CORPUS.name}")


def chunker_unit_check(name: str, text: str,
                       band: tuple[int, int] | None = None
                       ) -> tuple[dict, list[str]]:
    chunks = chunk_text(text)
    lens = [len(c) for c in chunks]
    check = {
        "n_chunks": len(chunks),
        "char_min": min(lens),
        "char_mean": round(statistics.fmean(lens), 1),
        "char_max": max(lens),
        "all_chunks_le_420": bool(max(lens) <= HARD_MAX),
    }
    if band is not None:
        lo, hi = band
        check["band"] = [lo, hi]
        check["pass"] = bool(lo <= len(chunks) <= hi and check["all_chunks_le_420"])
    else:
        check["pass"] = check["all_chunks_le_420"]
    return check, chunks


# ------------------------------------------------------------- synthesis ---
def synthesize_chunks(engine, chunks: list[str], voice_index: int,
                      out_wav: Path, chunk_dir: Path,
                      pause_s: float = PAUSE_S) -> tuple[list[dict], np.ndarray, dict]:
    """Synthesize chunks, concat with flat pauses, write wavs, time it all."""
    sr = engine.sample_rate
    pause = np.zeros(int(round(pause_s * sr)), dtype=np.float32)
    parts: list[np.ndarray] = []
    rows: list[dict] = []
    t_loop = time.perf_counter()
    for k, text in enumerate(chunks):
        t0 = time.perf_counter()
        wav = np.asarray(engine.synth(text, voice_index), dtype=np.float32)
        wall = time.perf_counter() - t0
        stats = engine.last_stats
        parts.append(wav)
        if k < len(chunks) - 1:
            parts.append(pause)
        cpath = chunk_dir / f"chunk_{k:03d}.wav"
        sf.write(str(cpath), wav, sr)
        rows.append({
            "i": k,
            "text": text,
            "chars": len(text),
            "phonemes": int(stats.n_phonemes),
            "audio_s": round(len(wav) / sr, 3),
            "synth_wall_s": round(wall, 3),
            "wav": str(cpath.relative_to(REPO)),
        })
    doc = np.concatenate(parts)
    sf.write(str(out_wav), doc, sr)
    wall_total = time.perf_counter() - t_loop
    audio_s = len(doc) / sr
    timing = {
        "synth_loop_wall_s": round(wall_total, 3),
        "audio_duration_s": round(audio_s, 3),
        "eff_rtf": round(wall_total / max(audio_s, 1e-9), 4),
    }
    return rows, doc, timing


# --------------------------------------------------------------- scoring ---
def _has_trailing_loop(words: list[str]) -> bool:
    return len(words) >= 3 and words[-1] == words[-2] == words[-3]


def score_chunk_rows(rows: list[dict], models: dict) -> None:
    """Dual-ASR minWER + repetition + compression per chunk, in place."""
    from classify_preview import (
        transcribe,
        norm_words,
        word_error_rate,
        repetition_fraction,
    )
    for row in rows:
        path = REPO / row["wav"]
        hyps = {m: transcribe(path, mod) for m, mod in models.items()}
        ref_w = norm_words(row["text"])
        wers = {m: word_error_rate(ref_w, norm_words(h))
                for m, h in hyps.items()}
        best = min(wers, key=wers.get)
        hyp_w = norm_words(hyps[best])
        row["wer_small_en"] = round(wers["small.en"], 4)
        row["wer_base_en"] = round(wers["base.en"], 4)
        row["min_wer"] = round(min(wers.values()), 4)
        row["best_asr"] = best
        row["hyp_transcript"] = hyps[best]
        row["ref_words"] = len(ref_w)
        row["hyp_words"] = len(hyp_w)
        row["compression_ratio"] = round(len(hyp_w) / max(len(ref_w), 1), 4)
        row["rep_fraction"] = round(repetition_fraction(hyp_w), 4)
        row["trailing_loop"] = _has_trailing_loop(hyp_w)
        print(f"[asr] chunk {row['i']:03d} minWER={row['min_wer']:.3f} "
              f"(small={row['wer_small_en']:.3f} base={row['wer_base_en']:.3f}) "
              f"rep={row['rep_fraction']:.3f} comp={row['compression_ratio']:.2f}",
              flush=True)


def doc_wide_metrics(rows: list[dict]) -> dict:
    from classify_preview import norm_words, repetition_fraction
    hyp_w = norm_words(" ".join(r["hyp_transcript"] for r in rows))
    ref_w = norm_words(" ".join(r["text"] for r in rows))
    return {
        "rep_fraction_docwide": round(repetition_fraction(hyp_w), 4),
        "compression_ratio_docwide": round(
            len(hyp_w) / max(len(ref_w), 1), 4),
        "trailing_loop_docwide": _has_trailing_loop(hyp_w),
    }


def wer_stats_and_gates(rows: list[dict], docwide: dict,
                        trivial_variance_note: bool = False) -> tuple[dict, dict]:
    minw = [r["min_wer"] for r in rows]
    mean = statistics.fmean(minw)
    pop_var = sum((x - mean) ** 2 for x in minw) / len(minw)
    stats = {
        "mean": round(mean, 4),
        "median": round(statistics.median(minw), 4),
        "min": round(min(minw), 4),
        "max": round(max(minw), 4),
        "population_variance": round(pop_var, 6),
    }
    worst_rep = max(r["rep_fraction"] for r in rows)
    worst_comp = max(r["compression_ratio"] for r in rows)
    gates = {
        "wer_variance_le_0_0025": {
            "value": stats["population_variance"],
            "target": 0.0025,
            "pass": bool(pop_var <= 0.0025),
            **({"note": "trivially 0 for N=1 (control); not a stability test"}
               if trivial_variance_note else {}),
        },
        "no_chunk_min_wer_gt_0_15": {
            "worst_chunk_min_wer": stats["max"],
            "pass": bool(max(minw) <= 0.15),
        },
        "rep_fraction_per_chunk_le_0_02": {
            "worst": worst_rep,
            "pass": bool(all(r["rep_fraction"] <= 0.02 for r in rows)),
        },
        "rep_fraction_docwide_le_0_01": {
            "value": docwide["rep_fraction_docwide"],
            "pass": bool(docwide["rep_fraction_docwide"] <= 0.01),
        },
        "compression_ratio_le_1_5": {
            "worst_per_chunk": worst_comp,
            "docwide": docwide["compression_ratio_docwide"],
            "pass": bool(worst_comp <= 1.5
                         and docwide["compression_ratio_docwide"] <= 1.5),
        },
        "no_trailing_loop": {
            "any": bool(any(r["trailing_loop"] for r in rows)
                        or docwide["trailing_loop_docwide"]),
            "pass": not bool(any(r["trailing_loop"] for r in rows)
                             or docwide["trailing_loop_docwide"]),
        },
    }
    return stats, gates


def clipping_check(doc_wav: np.ndarray) -> dict:
    if doc_wav.size == 0:
        return {"peak_abs": 0.0, "n_samples_ge_0_999": 0, "flag": False}
    peak = float(np.max(np.abs(doc_wav)))
    n_clip = int(np.count_nonzero(np.abs(doc_wav) >= 0.999))
    return {
        "peak_abs": round(peak, 4),
        "n_samples_ge_0_999": n_clip,
        "flag": bool(peak >= 0.999),
    }


# ------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser(
        description="CV-G1-E0014 long-form baseline probe")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (CLI default 4)")
    ap.add_argument("--selftest", action="store_true",
                    help="run chunker unit checks only, no model/ASR load")
    args = ap.parse_args()

    t_run0 = time.perf_counter()

    # 1. corpus + chunker unit checks (fail fast, before any model load)
    text = load_corpus_item(DOC_ID)["text"]
    para_text = load_corpus_item("para_001")["text"]
    check_long, chunks = chunker_unit_check(DOC_ID, text, band=(25, 31))
    check_para, _ = chunker_unit_check("para_001", para_text)
    checks = {"long_001": check_long, "para_001": check_para}
    print(f"[chunker] long_001 -> {check_long['n_chunks']} chunks "
          f"(band 25-31, design [24,32]); chars min/mean/max = "
          f"{check_long['char_min']}/{check_long['char_mean']}/"
          f"{check_long['char_max']} (hard max 420); pass={check_long['pass']}")
    print(f"[chunker] para_001 -> {check_para['n_chunks']} chunks "
          f"(single-paragraph sanity, all<=420: "
          f"{check_para['all_chunks_le_420']})")
    if not check_long["pass"]:
        print("[chunker] FAIL: unit check band/limit violated; aborting "
              "before synthesis", flush=True)
        return 2
    if args.selftest:
        print("[selftest] chunker unit checks OK", flush=True)
        return 0

    # 2. engine on the newest pilot checkpoint (step_*.pt, latest step)
    ckpt = sorted((RUN_DIR / "checkpoints").glob("step_*.pt"))[-1]
    step = int(ckpt.stem.split("_")[1])
    from caedral_voice.inference.engine import CaedralEngine
    t0 = time.perf_counter()
    engine = CaedralEngine.from_run(RUN_DIR, ckpt, threads=args.threads)
    load_s = time.perf_counter() - t0
    voice_index = engine.resolve_voice(VOICE_NAME)
    voice_name = next(v["speaker_name"] for v in engine.voices
                      if v["index"] == voice_index)
    print(f"[engine] {ckpt.name} (step {step}) voice={voice_name} "
          f"index={voice_index} threads={args.threads} "
          f"load+warmup={load_s:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_dir = OUT_DIR / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    out_wav = OUT_DIR / "long_001-spk103.wav"

    # 3. long-form synthesis
    rows, doc_wav, timing = synthesize_chunks(
        engine, chunks, voice_index, out_wav, chunk_dir)
    print(f"[synth] long_001: {timing['audio_duration_s']}s audio, "
          f"wall {timing['synth_loop_wall_s']}s, eff RTF {timing['eff_rtf']}",
          flush=True)

    # 4. control: same pipeline, 2-sentence text (short path)
    ctrl_chunks = chunk_text(CONTROL_TEXT)
    ctrl_dir = OUT_DIR / "control"
    ctrl_dir.mkdir(exist_ok=True)
    ctrl_out = OUT_DIR / "control-2sent-spk103.wav"
    ctrl_rows, ctrl_wav, ctrl_timing = synthesize_chunks(
        engine, ctrl_chunks, voice_index, ctrl_out, ctrl_dir)
    ctrl_short_path_ok = len(ctrl_chunks) == 1
    print(f"[control] {len(ctrl_chunks)} chunk(s) (expected 1), "
          f"{ctrl_timing['audio_duration_s']}s audio, "
          f"wall {ctrl_timing['synth_loop_wall_s']}s", flush=True)

    # 5. dual-ASR scoring
    print("[asr] loading faster-whisper small.en + base.en (int8)...",
          flush=True)
    from classify_preview import load_models
    models = load_models()
    score_chunk_rows(rows, models)
    score_chunk_rows(ctrl_rows, models)
    docwide = doc_wide_metrics(rows)
    stats, gates = wer_stats_and_gates(rows, docwide)
    ctrl_docwide = doc_wide_metrics(ctrl_rows)
    ctrl_stats, ctrl_gates = wer_stats_and_gates(
        ctrl_rows, ctrl_docwide, trivial_variance_note=True)

    results = {
        "experiment": "CV-G1-E0014-longform-baseline",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": "MEASURED",
        "note_chunker_target": (
            "task said 'para_001' but the 28±3 unit check and the mission "
            "(long_001-spk103.wav, 5687 chars) both define long_001; "
            "para_001 (387 chars) is unit-checked as splitter sanity"),
        "checkpoint": {"path": str(ckpt.relative_to(REPO)), "step": step},
        "engine": {
            "threads": args.threads,
            "noise_scale": 0.25,
            "noise_scale_source": "engine default (E0009)",
            "voice": {"name": voice_name, "index": voice_index},
            "sample_rate": engine.sample_rate,
            "engine_load_incl_warmup_s": round(load_s, 1),
        },
        "chunker": {
            "policy": "LONGFORM-DESIGN v0 section 1.1 (inline implementation)",
            "max_chars": MAX_CHARS,
            "hard_max": HARD_MAX,
            "pause_between_chunks_s": PAUSE_S,
            "unit_checks": checks,
        },
        "longform": {
            "item": DOC_ID,
            "chars": len(text),
            "words": len(text.split()),
            "n_chunks": len(chunks),
            "audio_path": str(out_wav.relative_to(REPO)),
            "timing": timing,
            "per_chunk": rows,
            "min_wer_stats": stats,
            "doc_wide": docwide,
            "gates": gates,
            "clipping": clipping_check(doc_wav),
        },
        "control": {
            "text": CONTROL_TEXT,
            "n_chunks": len(ctrl_chunks),
            "short_path_ok": ctrl_short_path_ok,
            "audio_path": str(ctrl_out.relative_to(REPO)),
            "timing": ctrl_timing,
            "per_chunk": ctrl_rows,
            "min_wer_stats": ctrl_stats,
            "doc_wide": ctrl_docwide,
            "gates": ctrl_gates,
            "clipping": clipping_check(ctrl_wav),
        },
        "total_runtime_s": round(time.perf_counter() - t_run0, 1),
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    g = gates
    all_pass = all(v.get("pass", True) for v in g.values())
    print(f"[summary] chunks={len(chunks)} "
          f"wall={timing['synth_loop_wall_s']}s "
          f"audio={timing['audio_duration_s']}s eff_rtf={timing['eff_rtf']} "
          f"minWER med={stats['median']} min={stats['min']} "
          f"max={stats['max']} var={stats['population_variance']} "
          f"rep_doc={docwide['rep_fraction_docwide']} "
          f"clip_peak={results['longform']['clipping']['peak_abs']} "
          f"gates={'ALL PASS' if all_pass else 'FAIL: ' + ','.join(
              k for k, v in g.items() if not v.get('pass', True))}",
          flush=True)
    print(f"[summary] control: chunks={len(ctrl_chunks)} "
          f"short_path_ok={ctrl_short_path_ok} "
          f"minWER={ctrl_stats['min']}", flush=True)
    print(f"[done] results -> {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
