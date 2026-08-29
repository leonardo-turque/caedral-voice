#!/usr/bin/env python
"""Teacher corpus builder -- experiment CV-G1-E0012-teacher-gen.

Implements docs/training/TEACHER-DATA-ENGINE.md (founder-authorized):
Kokoro-82M v1.0 (Apache-2.0 code + weights) RE-SPEAKS LibriTTS-R
train-clean-100 transcripts; the audio is training data ONLY, never shipped.

Pair construction ("ours on both sides"):
  * target side  = OUR 80-band ln-mel (24 kHz / hop 256 / power mel /
    natural log / floor 1e-5), computed at train time from the stored PCM16
    wav by caedral_voice/training/mel_dataset.py. This script validates every
    clip through that exact chain inline (never stores mels).
  * input side   = OUR frontend phonemes, already frozen in
    data/manifests/librittsr-clean100.phonemes.jsonl, joined by source_id
    (the teacher's own phoneme chain is never used as model input).

Resumable & multi-worker safe:
  * work is sharded contiguously (--shard i --num-shards n);
  * each shard appends only to its own files (manifest/rejected/meta), so
    no two processes ever write the same handle;
  * the merged views data/manifests/teacher-kokoro.jsonl (+ .rejected.jsonl,
    .meta.json) are rebuilt atomically from shard files on every progress
    tick -- crash-safe, torn-line tolerant;
  * restart skips ids already present in the shard manifest (unless the wav
    is missing, which is regenerated).

Quality gates (inline): reject clips < min-dur or > max-dur, clipped_frac
above threshold, MAS-infeasible rows (actual mel frames < phoneme count),
non-finite mels, synthesis errors. Eval-corpus firewall per TEACHER doc S3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------- #
# teacher slate -- FROZEN, TEACHER-DATA-ENGINE.md S1 (10 voices, 5F/5M,
# 7 US / 3 UK); verified against the voices bin at runtime. Prefixes t00..t09
# make mel_dataset.py's id.split("_")[0] assign speaker slot == NN unchanged.
# --------------------------------------------------------------------------- #
TEACHER_SLATE = [
    ("af_heart", "F / US"), ("af_bella", "F / US"), ("af_nicole", "F / US"),
    ("bf_alice", "F / UK"), ("bf_emma", "F / UK"), ("am_adam", "M / US"),
    ("am_fenrir", "M / US"), ("am_michael", "M / US"), ("am_puck", "M / US"),
    ("bm_george", "M / UK"),
]
MODEL_PATH = REPO / "models" / "kokoro" / "kokoro-v1.0.onnx"
VOICES_PATH = REPO / "models" / "kokoro" / "voices-v1.0.bin"

TARGET_SR = 24000          # kokoro v1.0 native output rate
N_MELS, N_FFT, HOP = 80, 1024, 256   # OUR canonical mel geometry


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _early_thread_env(default_threads: int) -> None:
    """Set OMP/MKL caps BEFORE any heavy import (torch/onnxruntime read these
    at load/session time). Scans argv directly because argparse runs later."""
    n = default_threads
    if "--threads" in sys.argv:
        i = sys.argv.index("--threads")
        try:
            n = int(sys.argv[i + 1])
        except (IndexError, ValueError):
            pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)


_early_thread_env(3)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(s: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed form used for
    dedupe keys and eval-corpus firewall comparison."""
    return _PUNCT_RE.sub(" ", s.casefold()).strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp" + str(os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl_tolerant(path: Path) -> list[dict]:
    """Lines -> dicts; silently drops a torn/partial trailing line (crash
    mid-append) so resume scans never die on our own crash debris."""
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


class AppendJsonl:
    """Append-only writer; flush per line so another process rebuilding the
    merged view never sees more than one partial line after a hard kill."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# corpus selection
# --------------------------------------------------------------------------- #
def build_work_items(rows: list[dict], eval_norm_set: set[str]) -> list[dict]:
    """First-N rows -> dedup/filter/firewall -> deterministic work items.

    Order-preserving dedupe on normalized-text sha256; drop texts longer than
    300 chars (chunk safety); reject eval-corpus text matches (S3 firewall --
    training on benchmark text would invalidate every quality number).
    """
    items: list[dict] = []
    seen_hash: set[str] = set()
    for g, r in enumerate(rows):        # g = global position -> voice slot
        text = (r.get("transcript_norm") or "").strip()
        src_id = r["id"]
        base = {"g": g, "source_id": src_id, "transcript": text,
                "n_phones": int(r.get("n_phones") or 0)}
        if not text:
            items.append({**base, "skip": "empty_transcript"})
            continue
        norm = normalize_text(text)
        if not norm:
            items.append({**base, "skip": "empty_transcript"})
            continue
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in seen_hash:
            items.append({**base, "skip": "duplicate_text"})
            continue
        if len(text) > 300:
            items.append({**base, "skip": "too_long_text"})
            continue
        if norm in eval_norm_set:
            items.append({**base, "skip": "eval_firewall"})
            continue
        seen_hash.add(h)
        items.append(base)
    return items


def load_eval_corpus_texts(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_text(item["text"])
        for cat in data["categories"] for item in cat["items"]
    }


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def save_wav_pcm16(path: Path, samples, sr: int) -> None:
    """float mono [-1, 1] -> 24 kHz PCM16 wav, atomic (tmp + replace)."""
    import numpy as np
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    ints = (x * 32767.0).astype("<i2")
    tmp = path.with_name(path.name + ".tmp" + str(os.getpid()))
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())
    os.replace(tmp, path)


class MelValidator:
    """The exact PilotDataset._load_mel chain (mel_dataset.py), used as an
    inline target-side sanity check: wav -> 24 kHz -> power mel [80, T] ->
    natural log with floor 1e-5. Single transform instance reused."""

    def __init__(self):
        import torch
        import torchaudio
        self.torch = torch
        self.torchaudio = torchaudio
        from caedral_voice.training.mel_dataset import MEL_FLOOR
        self.mel_floor = MEL_FLOOR
        self._fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, power=1.0)

    def frames_and_finite(self, wav_path: Path) -> tuple[int, bool]:
        wav, sr = self.torchaudio.load(str(wav_path))
        if sr != TARGET_SR:
            wav = self.torchaudio.functional.resample(wav, sr, TARGET_SR)
        mel = self._fn(wav.mean(dim=0))
        mel = self.torch.log(mel.clamp_min(self.mel_floor))
        return int(mel.shape[1]), bool(self.torch.isfinite(mel).all())


def rms_dbfs_proxy(samples) -> float:
    """UNCALIBRATED RMS power dBFS (no K-weighting/gating). Stored in the
    lufs_before/lufs_after fields as an honest placeholder; meta records the
    semantics. Real loudnorm stays a train-time preprocess.py concern."""
    import math
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    p = float((x * x).mean()) if x.size else 0.0
    return round(10.0 * math.log10(p + 1e-12), 3)


# --------------------------------------------------------------------------- #
# merged-view rebuild
# --------------------------------------------------------------------------- #
def rebuild_merged(num_shards: int, out_manifest: Path, out_rejected: Path,
                   out_meta: Path, static_meta: dict) -> None:
    """Rebuild teacher-kokoro{,.rejected}.jsonl and .meta.json from shard
    files. Manifest dedupes by id keeping the LAST row (mirrors how
    mel_dataset builds ph_by_id). All writes atomic."""
    rows_by_id: dict[str, dict] = {}
    rej_rows: list[dict] = []
    totals = {
        "seen": 0, "processed": 0, "kept": 0,
        "rejected_by_reason": {}, "audio_seconds": 0.0,
        "per_voice_counts": {}, "wall_seconds": 0.0, "synth_seconds": 0.0,
    }
    for i in range(num_shards):
        smeta_p = out_manifest.parent / f"{out_manifest.stem}.meta.shard{i}.json"
        if smeta_p.is_file():
            try:
                sm = json.loads(smeta_p.read_text(encoding="utf-8"))
                totals["seen"] += int(sm.get("seen", 0))
                totals["processed"] += int(sm.get("processed", 0))
                totals["kept"] += int(sm.get("kept", 0))
                totals["audio_seconds"] += float(sm.get("audio_seconds", 0.0))
                totals["wall_seconds"] += float(sm.get("wall_seconds", 0.0))
                totals["synth_seconds"] += float(sm.get("synth_seconds", 0.0))
                for k, v in (sm.get("rejected") or {}).items():
                    totals["rejected_by_reason"][k] = \
                        totals["rejected_by_reason"].get(k, 0) + int(v)
                for k, v in (sm.get("per_voice_counts") or {}).items():
                    totals["per_voice_counts"][k] = \
                        totals["per_voice_counts"].get(k, 0) + int(v)
            except (json.JSONDecodeError, OSError):
                pass
        for r in read_jsonl_tolerant(out_manifest.parent /
                                     f"{out_manifest.stem}.shard{i}.jsonl"):
            rows_by_id[r["id"]] = r
        rej_rows.extend(read_jsonl_tolerant(
            out_manifest.parent / f"{out_manifest.stem}.rejected.shard{i}.jsonl"))

    buf = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                  for r in rows_by_id.values())
    atomic_write(out_manifest, buf)
    atomic_write(out_rejected,
                 "".join(json.dumps(r, ensure_ascii=False) + "\n"
                         for r in rej_rows))

    audio_s = totals["audio_seconds"]
    meta = dict(static_meta)
    meta["last_update"] = utcnow()
    meta["totals"] = {
        **totals,
        "audio_hours": round(audio_s / 3600.0, 4),
        "rows_in_manifest": len(rows_by_id),
        "rejected_rows_logged": len(rej_rows),
        "rtf_effective": (
            round(totals["synth_seconds"] / audio_s, 4) if audio_s > 0 else None),
    }
    atomic_write(out_meta, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(
        REPO / "data/manifests/librittsr-clean100.phonemes.jsonl"))
    ap.add_argument("--out-manifest", default=str(
        REPO / "data/manifests/teacher-kokoro.jsonl"))
    ap.add_argument("--out-dir", default=str(
        REPO / "data/processed/teacher-kokoro"))
    ap.add_argument("--eval-corpus", default=str(
        REPO / "benchmarks/corpus/eval-corpus-v1.0.json"))
    ap.add_argument("--exp-dir", default=str(
        REPO / "experiments/CV-G1-E0012-teacher-gen"))
    ap.add_argument("--limit", type=int, default=16000,
                    help="use only the FIRST N rows of the input manifest")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3,
                    help="onnxruntime intra-op + torch threads")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--max-clipped-frac", type=float, default=0.001)
    args = ap.parse_args()

    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"bad shard spec {args.shard}/{args.num_shards}")

    # thread caps BEFORE torch/onnxruntime imports (env already primed at
    # module import); ORT gets exact numbers via SessionOptions below.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    import numpy as np                      # noqa: F401  (warm the imports)
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    import torch
    torch.set_num_threads(args.threads)

    out_manifest = Path(args.out_manifest)
    man_dir = out_manifest.parent
    stem = out_manifest.stem                 # e.g. teacher-kokoro
    out_dir = Path(args.out_dir)
    exp_dir = Path(args.exp_dir)
    for d in (man_dir, out_dir, exp_dir):
        d.mkdir(parents=True, exist_ok=True)

    shard_man = man_dir / f"{stem}.shard{args.shard}.jsonl"
    shard_rej = man_dir / f"{stem}.rejected.shard{args.shard}.jsonl"
    shard_meta_p = man_dir / f"{stem}.meta.shard{args.shard}.json"
    prog_path = exp_dir / f"shard{args.shard}.progress.json"

    def log(msg: str) -> None:
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
              f"s{args.shard}: {msg}", flush=True)

    log(f"threads={args.threads} limit={args.limit} "
        f"shard={args.shard}/{args.num_shards} pid={os.getpid()}")

    # ---- inputs ---------------------------------------------------------- #
    rows = read_jsonl_tolerant(Path(args.input))[: args.limit]
    eval_norm = load_eval_corpus_texts(Path(args.eval_corpus))
    items_all = build_work_items(rows, eval_norm)
    lo = len(items_all) * args.shard // args.num_shards
    hi = len(items_all) * (args.shard + 1) // args.num_shards
    mine = items_all[lo:hi]
    log(f"input rows={len(rows)} filtered={len(items_all)} "
        f"my range=[{lo},{hi}) n={len(mine)}")

    # ---- engine ---------------------------------------------------------- #
    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"])
    engine = Kokoro.from_session(session, str(VOICES_PATH))
    available = set(engine.get_voices())
    missing = [v for v, _ in TEACHER_SLATE if v not in available]
    if missing:
        raise SystemExit(f"slate voices missing from voices bin: {missing}")
    log(f"engine loaded ({time.perf_counter()-t0:.1f}s); slate verified "
        f"{len(TEACHER_SLATE)}/{len(available)} voices present")

    validator = MelValidator()

    static_meta = {
        "dataset": stem,
        "experiment": "CV-G1-E0012-teacher-gen",
        "design_doc": "docs/training/TEACHER-DATA-ENGINE.md",
        "git_commit": git_commit(),
        "created": utcnow(),
        "text_source": {
            "manifest": str(Path(args.input).relative_to(REPO)),
            "first_n_rows": args.limit,
            "corpus": "LibriTTS-R train-clean-100",
            "license": "CC-BY-4.0 (text rights cleared by the real-data track)",
        },
        "teacher": {
            "engine": "kokoro-onnx 0.6.1",
            "model": MODEL_PATH.name,
            "model_sha256": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
            "voices_bin": VOICES_PATH.name,
            "voices_bin_sha256": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
            "license": "Apache-2.0 (code + weights)",
            "authorization": "founder approved teacher models 'temporariamente durante desenvolvimento' (program record)",
            "usage_statement": "Teacher audio is TRAINING DATA ONLY and is never shipped; only its statistical influence on from-scratch Caedral weights persists.",
        },
        "slate": [{"slot": f"t{i:02d}", "voice": v, "cell": c}
                  for i, (v, c) in enumerate(TEACHER_SLATE)],
        "params": {
            "speed": args.speed,
            "target_sr": TARGET_SR, "n_mels": N_MELS, "hop": HOP, "n_fft": N_FFT,
            "pcm": "PCM16 mono", "min_dur_s": args.min_dur,
            "max_dur_s": args.max_dur,
            "max_clipped_frac": args.max_clipped_frac,
            "max_text_chars": 300,
            "shards": args.num_shards, "threads_per_worker": args.threads,
            "round_robin_rule": "voice = SLATE[global_item_index % 10]",
        },
        "field_semantics": {
            "lufs_note": "lufs_before/lufs_after carry an UNCALIBRATED RMS power dBFS proxy (no K-weighting); audio is raw synthesis output, no loudnorm applied at generation time.",
            "sha256_audio": "sha256 of the stored PCM16 wav bytes",
            "sha256_transcript": "sha256 of the normalized transcript text",
            "source_id": "LibriTTS-R utterance id; joins to librittsr-clean100.phonemes.jsonl for the OUR-frontend input side",
        },
        "storage_layout": {
            "wavs": "data/processed/teacher-kokoro/<id>.wav",
            "shard_files": [f"{stem}.shard{{i}}.jsonl / .rejected.shard{{i}}.jsonl / .meta.shard{{i}}.json"],
            "merged_views": [str(out_manifest.name), f"{stem}.rejected.jsonl",
                             f"{stem}.meta.json"],
        },
    }

    # ---- resume state ---------------------------------------------------- #
    done: dict[str, dict] = {}
    for r in read_jsonl_tolerant(shard_man):
        done[r["id"]] = r
    rej_fh = AppendJsonl(shard_rej)
    man_fh = AppendJsonl(shard_man)

    stats = {"seen": len(mine), "processed": 0, "kept": 0, "rejected": {},
             "audio_seconds": 0.0, "wall_seconds": 0.0, "synth_seconds": 0.0,
             "per_voice_counts": {}}
    started = time.perf_counter()

    def reject(item: dict, reason: str, detail: str = "") -> None:
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
        rej_fh.append({"id": item.get("id"), "source_id": item["source_id"],
                       "teacher_voice": item.get("voice"),
                       "reason": reason, "detail": detail})

    def tick(force: bool = False) -> None:
        stats["wall_seconds"] = round(time.perf_counter() - started, 1)
        if force or stats["processed"] > 0:
            atomic_write(shard_meta_p, json.dumps(
                {**stats, "shard": args.shard, "pid": os.getpid(),
                 "done": False, "updated": utcnow()},
                ensure_ascii=False, indent=2))
            prog_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(prog_path, json.dumps({
                "shard": args.shard, "pid": os.getpid(),
                "processed": stats["processed"], "kept": stats["kept"],
                "audio_hours": round(stats["audio_seconds"] / 3600.0, 4),
                "elapsed_h": round(stats["wall_seconds"] / 3600.0, 4),
                "updated": utcnow()}, indent=2))
            rebuild_merged(args.num_shards, out_manifest,
                           man_dir / f"{stem}.rejected.jsonl",
                           man_dir / f"{stem}.meta.json", static_meta)

    # ---- main loop ------------------------------------------------------- #
    wav_bytes_total = 0
    try:
        for item in mine:
            g, src_id, text = item["g"], item["source_id"], item["transcript"]
            slot = g % len(TEACHER_SLATE)
            voice = TEACHER_SLATE[slot][0]
            tid = f"t{slot:02d}_{src_id}"
            item["id"], item["voice"] = tid, voice

            if "skip" in item:                     # corpus-level filters
                reject(item, item["skip"])
                stats["processed"] += 1
                continue

            wav_path = out_dir / f"{tid}.wav"
            prior = done.get(tid)
            if prior is not None and wav_path.is_file():
                continue                            # resumable skip
            if prior is not None and not wav_path.is_file():
                log(f"WARN {tid}: manifest row without wav -- regenerating")

            t_syn = time.perf_counter()
            try:
                samples, sr = engine.create(
                    text, voice=voice, speed=args.speed, lang="en-us")
            except Exception as exc:                # noqa: BLE001
                reject(item, "synth_error", repr(exc)[:200])
                stats["processed"] += 1
                continue
            stats["synth_seconds"] += time.perf_counter() - t_syn
            duration = len(samples) / sr

            # ---- inline quality gates ------------------------------------- #
            if duration < args.min_dur:
                reject(item, "too_short_audio", f"{duration:.2f}s")
                stats["processed"] += 1
                continue
            if duration > args.max_dur:
                reject(item, "too_long_audio", f"{duration:.2f}s")
                stats["processed"] += 1
                continue
            x = np.abs(np.asarray(samples, dtype=np.float32))
            clipped_frac = float((x > 0.99).mean())
            if clipped_frac > args.max_clipped_frac:
                reject(item, "clipped_audio", f"frac={clipped_frac:.5f}")
                stats["processed"] += 1
                continue

            if sr != TARGET_SR:                     # unexpected; normalize
                import torchaudio
                samples = torchaudio.functional.resample(
                    torch.tensor(samples), sr, TARGET_SR).numpy()
                sr = TARGET_SR

            save_wav_pcm16(wav_path, samples, sr)

            # ---- target-side validation through OUR mel chain -------------- #
            frames, finite = validator.frames_and_finite(wav_path)
            if not finite:
                reject(item, "mel_nonfinite")
                wav_path.unlink(missing_ok=True)
                stats["processed"] += 1
                continue
            if frames < item["n_phones"]:
                reject(item, "mas_infeasible",
                       f"frames {frames} < phones {item['n_phones']}")
                wav_path.unlink(missing_ok=True)
                stats["processed"] += 1
                continue

            try:
                ap_str = str(wav_path.relative_to(REPO))
            except ValueError:                  # outputs outside the repo
                ap_str = str(wav_path)
            row = {
                "id": tid,
                "audio_path": ap_str.replace("\\", "/"),
                "transcript": text,
                "duration_s": round(duration, 6),
                "sample_rate": sr,
                "lufs_before": rms_dbfs_proxy(samples),
                "lufs_after": rms_dbfs_proxy(samples),
                "clipped_frac": clipped_frac,
                "sha256_audio": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                "sha256_transcript": sha256_text(text),
                "source_id": src_id,
                "teacher_voice": voice,
            }
            man_fh.append(row)

            stats["kept"] += 1
            stats["audio_seconds"] += duration
            stats["per_voice_counts"][voice] = \
                stats["per_voice_counts"].get(voice, 0) + 1
            stats["processed"] += 1
            done[tid] = row

            if stats["kept"] % 25 == 0:
                el = max(time.perf_counter() - started, 1e-6)
                rtf = stats["synth_seconds"] / max(stats["audio_seconds"], 1e-6)
                log(f"kept={stats['kept']} audio={stats['audio_seconds']/3600:.2f}h "
                    f"rej={sum(stats['rejected'].values())} "
                    f"rtf={rtf:.2f} rate={el and stats['audio_seconds']/el:.1f}x "
                    f"elapsed={el/3600:.2f}h")
                tick()

    except KeyboardInterrupt:
        log("interrupted -- finalizing")
    finally:
        tick(force=True)
        stats["done"] = True
        atomic_write(shard_meta_p, json.dumps(
            {**stats, "shard": args.shard, "pid": os.getpid(), "done": True,
             "updated": utcnow()}, ensure_ascii=False, indent=2))
        rebuild_merged(args.num_shards, out_manifest,
                       man_dir / f"{stem}.rejected.jsonl",
                       man_dir / f"{stem}.meta.json", static_meta)
        man_fh.close()
        rej_fh.close()
        log(f"DONE kept={stats['kept']} audio="
            f"{stats['audio_seconds']/3600:.3f}h "
            f"rejected={stats['rejected']}")


if __name__ == "__main__":
    main()
