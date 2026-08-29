"""Download LJSpeech-1.1 and extract a deterministic slice.

Stages (each skipped when its outputs already exist -> resumable):
  1. download  data/downloads/LJSpeech-1.1.tar.bz2   (HTTP Range resume)
  2. verify    sha256 vs the official torchaudio/HF-published hash; a mismatch
               aborts unless --accept-measured-sha256 records the file hash as
               MEASURED-at-download with explicit provenance marking
  3. extract   first N utterances (sorted by ID) into data/raw/ljspeech-slice/
               with a filtered metadata.csv so the slice is self-contained

The archive is treated as immutable input: it is only read, never modified,
and extraction writes into data/raw/ljspeech-slice/ (created fresh by this
script only; preprocess.py never touches data/raw).

Usage:
  python data/scripts/download_ljspeech.py [--n 400]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

URL_DEFAULT = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
# Official sha256 published in torchaudio's LJSpeech dataset config and
# Hugging Face keithito/lj_speech download_checksums (both retrieved 2026-08-22).
EXPECTED_SHA256 = "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5"
EXPECTED_SIZE = 2748572632  # bytes, per HF download_checksums metadata

ARCHIVE_NAME = "LJSpeech-1.1.tar.bz2"


def downloads_dir() -> Path:
    return REPO_ROOT / "data" / "downloads"


def slice_dir(n: int) -> Path:
    return REPO_ROOT / "data" / "raw" / "ljspeech-slice"


def state_path() -> Path:
    return downloads_dir() / "LJSpeech-1.1.download.json"


# ---------------------------------------------------------------------------
# Stage 1: download (with HTTP Range resume of the .part file)
# ---------------------------------------------------------------------------

def _stream_to_file(resp, fh, part: Path, start_size: int, total: int | None):
    written = start_size
    report_at = written + (200 << 20)
    for chunk in resp.iter_content(chunk_size=1 << 20):
        if not chunk:
            continue
        fh.write(chunk)
        written += len(chunk)
        if written >= report_at:
            pct = f"{100.0 * written / total:.1f}" if total else "?"
            print(f"  {part.name}: {written} bytes ({pct}%)", flush=True)
            report_at = written + (200 << 20)


def stage_download(url: str) -> dict:
    """Return the download state dict (sha256/provenance filled by verify)."""
    ddir = downloads_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    final = ddir / ARCHIVE_NAME
    state_p = state_path()

    if final.exists():
        # Archive already on disk: never re-download; verification (stage 2)
        # will hash whatever bytes are here.
        prev = {}
        if state_p.exists():
            prev = json.loads(state_p.read_text(encoding="utf-8"))
        print(f"[download] stage 1 skipped: {final} present "
              f"({final.stat().st_size} bytes)")
        return {"url": url, "size": final.stat().st_size,
                "sha256": prev.get("sha256"),
                "provenance": prev.get("provenance")}

    import requests

    part = final.with_suffix(final.suffix + ".part")
    headers = {}
    start_size = 0
    if part.exists():
        start_size = part.stat().st_size
        headers["Range"] = f"bytes={start_size}-"
        print(f"[download] resuming {part.name} from byte {start_size}")
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if start_size and resp.status_code == 200:
            # Server ignored Range; restart cleanly.
            start_size = 0
            part_len = 0
        elif resp.status_code == 206:
            part_len = None
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                part_len = int(cr.rsplit("/", 1)[1])
        elif resp.status_code == 200:
            part_len = None
        else:
            raise RuntimeError(f"HTTP {resp.status_code} from {url}")
        total = part_len
        clen = resp.headers.get("Content-Length")
        mode = "ab" if start_size else "wb"
        with open(part, mode) as fh:
            _stream_to_file(resp, fh, part, start_size,
                            total if total else (int(clen) + start_size if clen else None))
    size = part.stat().st_size
    if EXPECTED_SIZE and size != EXPECTED_SIZE and total is None:
        print(f"[download] WARNING: size {size} != expected {EXPECTED_SIZE}")
    part.replace(final)
    print(f"[download] stage 1 done: {final} ({size} bytes)")

    # Fresh bytes -> force re-verification even if an old state file existed.
    if state_p.exists():
        state_p.unlink()
    return {"url": url, "size": size, "sha256": None, "provenance": None}


# ---------------------------------------------------------------------------
# Stage 2: sha256 verification / provenance recording
# ---------------------------------------------------------------------------

def stage_verify(archive: Path, state: dict, accept_measured: bool) -> dict:
    if state.get("sha256") and state.get("provenance"):
        print(f"[verify] stage 2 skipped: already recorded "
              f"({state['provenance']})")
        return state

    import pipeline_lib as pl  # reuse streaming hasher

    print(f"[verify] hashing {archive.name} ...", flush=True)
    measured = pl.sha256_file(archive)
    size = archive.stat().st_size
    if measured == EXPECTED_SHA256:
        provenance = ("VERIFIED-official-sha256 (torchaudio LJSpeech config / "
                      "HF keithito/lj_speech download_checksums)")
        print(f"[verify] OK: {measured}")
    elif accept_measured or not EXPECTED_SHA256:
        provenance = ("MEASURED-at-download (did NOT match official published "
                      "hash -- provenance flagged, do not treat as verified)")
        print(f"[verify] WARNING: hash mismatch. expected={EXPECTED_SHA256} "
              f"measured={measured}\n[verify] recording as MEASURED-at-download.")
    else:
        raise SystemExit(
            f"FATAL: sha256 mismatch for {archive}\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  measured {measured}\n"
            f"Refusing to use unverified archive. Re-download, or pass "
            f"--accept-measured-sha256 to record the measured hash with "
            f"MEASURED-at-download provenance.")

    state.update({"sha256": measured, "size": size, "provenance": provenance,
                  "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                      timespec="seconds")})
    sp = state_path()
    sp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
    print(f"[verify] stage 2 done: wrote {sp}")
    return state


# ---------------------------------------------------------------------------
# Stage 3: deterministic slice extraction
# ---------------------------------------------------------------------------

def _member_id(name: str) -> str | None:
    """'LJSpeech-1.1/wavs/LJ001-0001.wav' -> 'LJ001-0001'."""
    parts = name.split("/")
    if len(parts) == 3 and parts[1] == "wavs" and parts[2].endswith(".wav"):
        return parts[2][: -len(".wav")]
    return None


def _scan_archive(tf: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], str]:
    """Single sorted-order scan: collect wav members + metadata.csv content."""
    wav_members: list[tarfile.TarInfo] = []
    metadata_csv: str | None = None
    for member in tf:  # tar members iterated deterministically
        if not member.isfile():
            continue
        uid = _member_id(member.name)
        if uid is not None:
            wav_members.append(member)
        elif member.name.endswith("metadata.csv"):
            fh = tf.extractfile(member)
            metadata_csv = io.TextIOWrapper(fh, encoding="utf-8").read()
    if metadata_csv is None:
        raise SystemExit("FATAL: metadata.csv not found in archive")
    wav_members.sort(key=lambda m: m.name)  # deterministic selection order
    return wav_members, metadata_csv


def stage_extract(archive: Path, n: int, archive_sha256: str | None) -> Path:
    out = slice_dir(n)
    marker = out / ".slice_state.json"
    if marker.exists():
        prev = json.loads(marker.read_text(encoding="utf-8"))
        if prev.get("n") == n and prev.get("archive_sha256") == archive_sha256:
            n_wavs = sum(1 for p in out.rglob("*.wav"))
            if n_wavs >= n and (out / "metadata.csv").exists():
                print(f"[extract] stage 3 skipped: {out} complete "
                      f"({n_wavs} wavs, n={n})")
                return out
            print(f"[extract] incomplete slice dir found; resuming per-file")

    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] scanning archive (full bz2 pass) ...", flush=True)
    selected_ids: list[str]
    with tarfile.open(archive, "r:bz2") as tf:
        wav_members, metadata_csv = _scan_archive(tf)
        selected_ids = [m.name.split("/")[-1][:-4] for m in wav_members[:n]]
        id_set = set(selected_ids)

        # Filtered metadata keeps the slice self-contained.
        kept_lines = []
        for line in metadata_csv.splitlines():
            uid = line.split("|", 1)[0].strip()
            if uid in id_set:
                kept_lines.append(line)
        missing = [i for i in selected_ids
                   if i not in {l.split("|", 1)[0].strip() for l in kept_lines}]
        if missing:
            raise SystemExit(
                f"FATAL: {len(missing)} selected ids lack transcripts, "
                f"first={missing[:3]}")

        (out / "metadata.csv").write_text("\n".join(kept_lines) + "\n",
                                          encoding="utf-8")
        wavs_out = out / "wavs"
        wavs_out.mkdir(exist_ok=True)
        n_wavs_done = sum(1 for p in wavs_out.glob("*.wav")
                          if p.stat().st_size > 0)
        if n_wavs_done:
            print(f"[extract] resuming: {n_wavs_done}/{len(selected_ids)} "
                  f"already on disk", flush=True)
        print(f"[extract] extracting {len(selected_ids)} wavs ...", flush=True)
        extracted = 0
        for member in wav_members[:n]:
            uid = member.name.split("/")[-1]
            dst_path = wavs_out / uid
            if dst_path.exists() and dst_path.stat().st_size > 0:
                continue  # per-file resume: never redo completed members
            src = tf.extractfile(member)
            with open(dst_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if extracted % 100 == 0:
                print(f"  {extracted} new ({extracted + n_wavs_done}"
                      f"/{len(selected_ids)})", flush=True)

    marker.write_text(json.dumps(
        {"n": n, "archive_sha256": archive_sha256,
         "ids_first": selected_ids[0], "ids_last": selected_ids[-1],
         "extracted": len(selected_ids)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[extract] stage 3 done: {out} ({len(selected_ids)} utterances)")
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=400,
                        help="slice size: first N utterances sorted by ID")
    parser.add_argument("--url", default=URL_DEFAULT)
    parser.add_argument("--accept-measured-sha256", action="store_true",
                        help="if the hash mismatches the official value, record "
                             "the measured hash as MEASURED-at-download instead "
                             "of aborting")
    args = parser.parse_args(argv)

    archive = downloads_dir() / ARCHIVE_NAME
    state = stage_download(args.url)
    state = stage_verify(archive, state, args.accept_measured_sha256)
    stage_extract(archive, args.n, state.get("sha256"))
    print(f"[done] slice ready at {slice_dir(args.n)} "
          f"(run preprocess.py on it next)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
