"""Phase D preprocessing pipeline: raw audio dir -> frozen manifests.

Reads a directory of WAV files (+ transcripts from a LJSpeech-style
metadata.csv or sibling .txt files), normalizes every utterance, and writes:

  data/manifests/<name>.jsonl           kept rows (see ROW_FIELDS)
  data/manifests/<name>.rejected.jsonl  {id, reason, detail} rows
  data/manifests/<name>.meta.json       provenance, params, counts, report

Determinism contract:
- Files iterated in sorted order; first occurrence wins dedup.
- No timestamps in outputs except meta["created"].
- Processed audio is written to data/processed/<name>/ -- data/raw is never
  modified. Float values rounded to 6 decimals for byte-stable JSONL.

Usage:
  python data/scripts/preprocess.py --raw-dir data/raw/ljspeech-slice \
      --name ljspeech-slice [--target-sr 22050] [--min-dur 0.3] ...
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_lib as pl  # noqa: E402

ROW_FIELDS = [
    "id", "audio_path", "transcript", "duration_s", "sample_rate",
    "lufs_before", "lufs_after", "clipped_frac",
    "sha256_audio", "sha256_transcript",
]

DATASET_DEFAULT = "ljspeech"
LICENSE_DEFAULT = "Public Domain (CC0)"
SOURCE_URL_DEFAULT = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


# ---------------------------------------------------------------------------
# Transcript discovery
# ---------------------------------------------------------------------------

def load_transcripts(raw_dir: Path) -> dict[str, str]:
    """LJSpeech metadata.csv (id|norm|raw) preferred, else sibling .txt files.

    The *normalized* column (index 1) is used when present, matching the
    dataset's own normalization pass; callers still run pl.normalize_transcript
    over it for hashing/dedup.
    """
    transcripts: dict[str, str] = {}
    meta_csv = None
    for candidate in sorted(raw_dir.rglob("metadata.csv")):
        meta_csv = candidate
        break
    if meta_csv is not None:
        with open(meta_csv, "r", encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="|"):
                if not row:
                    continue
                uid = row[0].strip()
                text = row[1].strip() if len(row) > 1 else ""
                if len(row) > 2 and not text:
                    text = row[2].strip()
                transcripts[uid] = text
        return transcripts
    for txt in sorted(raw_dir.rglob("*.txt")):
        transcripts[txt.stem] = txt.read_text(encoding="utf-8").strip()
    return transcripts


def find_wavs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*.wav") if p.is_file())


# ---------------------------------------------------------------------------
# Per-utterance processing
# ---------------------------------------------------------------------------

class Rejected(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def process_utterance(wav_path: Path, out_dir: Path, sr_target: int,
                      params: dict, uid: str | None = None) -> dict:
    """Full per-clip chain. Returns a manifest row or raises Rejected.

    Writes the processed PCM_16 wav into `out_dir`. Never touches the input.
    """
    # Corrupt-file rejection: any load failure is fatal for this clip only.
    try:
        x_src, sr_src = pl.load_wav_mono(wav_path)
    except Exception as exc:  # soundfile psf errors, ValueError, OSError
        raise Rejected(pl.REASON_CORRUPT, f"{type(exc).__name__}: {exc}") from exc

    clipped_frac = pl.clipped_fraction(
        x_src, threshold=params["clip_abs_threshold"])
    if x_src.size == 0:
        raise Rejected(pl.REASON_ALL_SILENCE, "zero samples decoded")

    # DC offset + high-pass (survey practice #4), then resample (#2).
    x = pl.remove_dc_offset(x_src)
    x = pl.highpass_filter(x, sr_src, cutoff_hz=params["highpass_hz"])
    x = pl.resample(x, sr_src, sr_target)

    # Loudness normalization (#3): two-pass to target LUFS, peak-limited.
    y, lufs_before, lufs_after, _limited = pl.loudness_normalize(
        x, sr_target, target_lufs=params["lufs_target"])

    # Silence trim (#5) then duration filter (#9).
    y = pl.trim_silence(
        y, sr_target,
        top_db=params["trim_top_db"], pad_ms=params["trim_pad_ms"],
        frame_ms=pl.TRIM_FRAME_MS, hop_ms=pl.TRIM_HOP_MS)
    if y.size == 0:
        raise Rejected(pl.REASON_ALL_SILENCE, "no energy above trim floor")
    duration_s = round(pl.duration_seconds(y.size, sr_target), 6)
    ok, reason = pl.duration_ok(duration_s, params["min_dur"], params["max_dur"])
    if not ok:
        raise Rejected(reason, f"duration_s={duration_s}")

    # Clipping rejection (#6): measured on source samples pre-processing.
    if clipped_frac > params["max_clipped_frac"]:
        raise Rejected(pl.REASON_CLIPPED, f"clipped_frac={clipped_frac:.6f}")

    # Too quiet to meter even before normalization -> unusable for TTS.
    if not np.isfinite(lufs_before):
        raise Rejected(pl.REASON_TOO_QUIET, "non-finite integrated loudness")

    stem = uid if uid is not None else wav_path.stem
    out_path = out_dir / f"{stem}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(y, dtype=np.float32), sr_target,
             subtype="PCM_16")

    return {
        "id": stem,
        "audio_path": out_path.relative_to(REPO_ROOT).as_posix(),
        "duration_s": duration_s,
        "sample_rate": int(sr_target),
        "lufs_before": round(float(lufs_before), 6),
        "lufs_after": round(float(lufs_after), 6),
        "clipped_frac": round(clipped_frac, 6),
        "sha256_audio": pl.sha256_file(out_path),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_pipeline(raw_dir: Path, name: str, out_dir: Path, processed_root: Path,
                 params: dict, dataset: str = DATASET_DEFAULT,
                 license_id: str = LICENSE_DEFAULT,
                 source_url: str = SOURCE_URL_DEFAULT,
                 download_state_path: Path | None = None) -> dict:
    """Run the full pass; returns paths dict of everything written."""
    manifests_dir = out_dir
    manifests_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = processed_root / name
    processed_dir.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts(raw_dir)
    wavs = find_wavs(raw_dir)

    seen_chars: set[str] = set()
    nonascii_char_count = 0
    nonascii_transcripts = 0
    norm_hashes_seen: set[str] = set()

    kept_rows: list[dict] = []
    rejected_rows: list[dict] = []
    by_reason: dict[str, int] = {}

    def reject(uid: str, reason: str, detail: str = "") -> None:
        rejected_rows.append({"id": uid, "reason": reason, "detail": detail})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    for wav_path in wavs:  # sorted iteration -> deterministic dedup winner
        uid = wav_path.stem
        transcript = transcripts.get(uid)
        if transcript is None:
            reject(uid, pl.REASON_MISSING_TRANSCRIPT,
                   "no metadata.csv entry or sibling .txt")
            continue
        seen_chars.update(transcript)
        n_bad, has_bad = pl.transcript_nonascii_stats(transcript)
        nonascii_char_count += n_bad
        nonascii_transcripts += 1 if has_bad else 0

        norm_hash = pl.sha256_text(pl.normalize_transcript(transcript))
        if norm_hash in norm_hashes_seen:
            reject(uid, pl.REASON_DUPLICATE, "hash of normalized transcript "
                   "already emitted (sorted-order first occurrence kept)")
            continue

        try:
            row = process_utterance(
                wav_path, processed_dir, params["target_sr"], params, uid=uid)
        except Rejected as rej:
            reject(uid, rej.reason, rej.detail)
            continue
        norm_hashes_seen.add(norm_hash)
        row["transcript"] = transcript
        row["sha256_transcript"] = norm_hash
        kept_rows.append({k: row[k] for k in ROW_FIELDS})

    download_sha256 = None
    download_provenance = None
    if download_state_path is not None and download_state_path.exists():
        state = json.loads(download_state_path.read_text(encoding="utf-8"))
        download_sha256 = state.get("sha256")
        download_provenance = state.get("provenance")

    counts = {
        "seen": len(wavs),
        "kept": len(kept_rows),
        "rejected": len(rejected_rows),
        "by_reason": dict(sorted(by_reason.items())),
    }
    meta = {
        "dataset": dataset,
        "license": license_id,
        "source_url": source_url,
        "download_sha256": download_sha256,
        "download_sha256_provenance": download_provenance,
        "pipeline_version": pl.PIPELINE_VERSION,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "params": dict(sorted(params.items())),
        "counts": counts,
        "transcript_report": {
            "chars": sorted(seen_chars),
            "n_unique_chars": len(seen_chars),
            "non_ascii_char_count": nonascii_char_count,
            "non_ascii_transcripts": nonascii_transcripts,
            "unique_norm_hashes": len(norm_hashes_seen),
        },
        "processed_dir": (processed_root / name).relative_to(REPO_ROOT).as_posix(),
    }

    keep_path = manifests_dir / f"{name}.jsonl"
    rej_path = manifests_dir / f"{name}.rejected.jsonl"
    meta_path = manifests_dir / f"{name}.meta.json"

    def dump_jsonl(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    dump_jsonl(keep_path, kept_rows)
    dump_jsonl(rej_path, rejected_rows)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    return {"kept": keep_path, "rejected": rej_path, "meta": meta_path,
            "counts": counts}


def build_params(args: argparse.Namespace) -> dict:
    return {
        "target_sr": args.target_sr,
        "lufs_target": args.lufs_target,
        "min_dur": args.min_dur,
        "max_dur": args.max_dur,
        "max_clipped_frac": args.max_clipped_frac,
        "clip_abs_threshold": pl.CLIP_ABS_THRESHOLD,
        "trim_top_db": args.trim_top_db,
        "trim_pad_ms": pl.TRIM_PAD_MS,
        "highpass_hz": args.highpass_hz,
        "resampler": "soxr-HQ" if pl._soxr is not None else "scipy-resample_poly",
        "pipeline_version": pl.PIPELINE_VERSION,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True,
                        help="raw dataset dir (read-only; never modified)")
    parser.add_argument("--name", required=True,
                        help="manifest basename, e.g. ljspeech-slice")
    parser.add_argument("--manifests-dir", default=None,
                        help=f"default: {REPO_ROOT / 'data' / 'manifests'}")
    parser.add_argument("--processed-root", default=None,
                        help=f"default: {REPO_ROOT / 'data' / 'processed'}")
    parser.add_argument("--download-state", default=None,
                        help="download_ljspeech.py state json carrying the "
                             "archive sha256/provenance")
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--license", default=LICENSE_DEFAULT)
    parser.add_argument("--source-url", default=SOURCE_URL_DEFAULT)
    parser.add_argument("--target-sr", type=int, default=pl.TARGET_SR_DEFAULT)
    parser.add_argument("--lufs-target", type=float, default=pl.LUFS_TARGET_DEFAULT)
    parser.add_argument("--min-dur", type=float, default=pl.MIN_DURATION_S_DEFAULT)
    parser.add_argument("--max-dur", type=float, default=pl.MAX_DURATION_S_DEFAULT)
    parser.add_argument("--max-clipped-frac", type=float,
                        default=pl.CLIP_REJECT_FRAC_DEFAULT,
                        help="reject clips whose source clipped-sample fraction "
                             "exceeds this (survey #6 precedent: ~0.001)")
    parser.add_argument("--trim-top-db", type=float, default=pl.TRIM_TOP_DB_DEFAULT)
    parser.add_argument("--highpass-hz", type=float, default=pl.HIGHPASS_HZ_DEFAULT)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    manifests_dir = (Path(args.manifests_dir) if args.manifests_dir
                     else REPO_ROOT / "data" / "manifests")
    processed_root = (Path(args.processed_root) if args.processed_root
                      else REPO_ROOT / "data" / "processed")
    dl_state = Path(args.download_state) if args.download_state else (
        REPO_ROOT / "data" / "downloads" / "LJSpeech-1.1.download.json")

    result = run_pipeline(
        raw_dir=raw_dir, name=args.name, out_dir=manifests_dir,
        processed_root=processed_root, params=build_params(args),
        dataset=args.dataset, license_id=getattr(args, "license"),
        source_url=args.source_url, download_state_path=dl_state)

    c = result["counts"]
    print(f"[preprocess] {args.name}: seen={c['seen']} kept={c['kept']} "
          f"rejected={c['rejected']}")
    for reason, n in sorted(c["by_reason"].items()):
        print(f"  {reason}: {n}")
    print(f"[preprocess] wrote {result['kept']}")
    print(f"[preprocess] wrote {result['rejected']}")
    print(f"[preprocess] wrote {result['meta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
