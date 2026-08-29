"""Manifest integrity gate: re-hash audio, re-check durations, verify counts.

Reads data/manifests/<name>.jsonl (+ .rejected.jsonl, .meta.json), then for
every kept row:
  - resolves audio_path (repo-relative) and confirms existence
  - recomputes sha256 and compares to row["sha256_audio"]
  - loads file headers via soundfile: duration within tolerance of
    row["duration_s"], sample_rate matches row["sample_rate"]
  - recomputes sha256 of the transcript text and compares
  - checks id uniqueness across rows
Cross-checks meta counts (seen == kept + rejected, per-reason totals) against
the actual JSONL line counts.

Exits 0 only when everything matches; exits 1 listing mismatches. This is the
gate downstream runs must pass before reading a manifest.

Usage:
  python data/scripts/validate_manifest.py --name ljspeech-slice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_lib as pl  # noqa: E402

DURATION_TOL_S = 0.02   # PCM16 write + header rounding tolerance
MAX_PRINTED_ERRORS = 20


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate(manifests_dir: Path, name: str,
             repo_root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    errors: list[str] = []
    keep_path = manifests_dir / f"{name}.jsonl"
    rej_path = manifests_dir / f"{name}.rejected.jsonl"
    meta_path = manifests_dir / f"{name}.meta.json"

    if not keep_path.exists():
        return False, [f"missing manifest: {keep_path}"]
    if not meta_path.exists():
        return False, [f"missing meta: {meta_path}"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = load_jsonl(keep_path)
    rejected = load_jsonl(rej_path)

    # -- counts ------------------------------------------------------------
    counts = meta.get("counts", {})
    if counts.get("kept") != len(rows):
        errors.append(f"counts.kept={counts.get('kept')} != {len(rows)} rows in "
                      f"{keep_path.name}")
    if rej_path.exists() and counts.get("rejected") != len(rejected):
        errors.append(f"counts.rejected={counts.get('rejected')} != "
                      f"{len(rejected)} rows in {rej_path.name}")
    by_reason = counts.get("by_reason", {})
    reason_totals: dict[str, int] = {}
    for r in rejected:
        reason_totals[r["reason"]] = reason_totals.get(r["reason"], 0) + 1
    if dict(sorted(by_reason.items())) != dict(sorted(reason_totals.items())):
        errors.append(f"counts.by_reason {by_reason} != observed {reason_totals}")
    if "seen" in counts and counts["seen"] != counts.get("kept", len(rows)) \
            + counts.get("rejected", len(rejected)):
        errors.append(f"counts.seen {counts['seen']} != kept+rejected")

    # -- per-row integrity --------------------------------------------------
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        where = f"{keep_path.name}:{i + 2}"  # headerless; line number
        rid = row.get("id")
        if not rid:
            errors.append(f"{where}: missing id")
            continue
        if rid in seen_ids:
            errors.append(f"{where}: duplicate id {rid}")
        seen_ids.add(rid)

        rel = row.get("audio_path", "")
        audio = repo_root / rel
        if not audio.exists():
            errors.append(f"{where}: audio missing: {rel}")
            continue

        got_hash = pl.sha256_file(audio)
        if got_hash != row.get("sha256_audio"):
            errors.append(f"{where}: sha256 mismatch for {rel}: "
                          f"manifest={row.get('sha256_audio')} "
                          f"actual={got_hash}")

        info = None
        try:
            import soundfile as sf
            info = sf.info(str(audio))
        except Exception as exc:
            errors.append(f"{where}: unreadable audio {rel}: {exc}")
        if info is not None:
            dur = info.frames / float(info.samplerate)
            if abs(dur - float(row.get("duration_s", -1))) > DURATION_TOL_S:
                errors.append(f"{where}: duration mismatch for {rel}: "
                              f"manifest={row.get('duration_s')} actual={dur:.6f}")
            if int(row.get("sample_rate", -1)) != info.samplerate:
                errors.append(f"{where}: sample_rate mismatch for {rel}: "
                              f"manifest={row.get('sample_rate')} "
                              f"actual={info.samplerate}")

        transcript = row.get("transcript")
        if transcript is None:
            errors.append(f"{where}: missing transcript field")
        else:
            thash = pl.sha256_text(pl.normalize_transcript(transcript))
            if thash != row.get("sha256_transcript"):
                errors.append(f"{where}: sha256_transcript mismatch for id {rid}")

    ok = not errors
    return ok, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True,
                        help="manifest basename, e.g. ljspeech-slice")
    parser.add_argument("--manifests-dir", default=None,
                        help=f"default: {REPO_ROOT / 'data' / 'manifests'}")
    args = parser.parse_args(argv)

    manifests_dir = (Path(args.manifests_dir) if args.manifests_dir
                     else REPO_ROOT / "data" / "manifests")
    ok, errors = validate(manifests_dir, args.name)
    if ok:
        print(f"[validate] OK: {args.name} passed all integrity checks")
        return 0
    print(f"[validate] FAILED: {len(errors)} error(s)")
    for err in errors[:MAX_PRINTED_ERRORS]:
        print(f"  - {err}")
    if len(errors) > MAX_PRINTED_ERRORS:
        print(f"  ... and {len(errors) - MAX_PRINTED_ERRORS} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
