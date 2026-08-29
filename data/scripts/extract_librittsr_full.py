"""Extract ALL train-clean-100 speakers from the LibriTTS-R archive.

DATA-FULL-01: scales raw training data from the 10-speaker slice to the full
~100 h / ~247-speaker train-clean-100 subset.

Output layout (book dirs flattened, mirroring extract_librittsr_slice.py):
  data/raw/librittsr-full/<SPK>/<utt>.wav            24 kHz restored audio
  data/raw/librittsr-full/<SPK>/<utt>.txt            normalized transcript
      (archive's <utt>.normalized.txt is written as <utt>.txt so the
       preprocess.py sibling-*.txt discovery finds it by wav stem)

Resumability:
  - Speakers whose dir contains a .complete marker are skipped entirely.
  - Within an incomplete speaker, files already on disk are not rewritten.
  - A speaker is marked .complete only when its on-disk file set exactly
    matches the set observed for it in this archive pass, so an interrupted
    run never leaves a falsely-complete marker behind.

Single sequential gzip pass (no separate scan pass): the expected per-speaker
file sets are accumulated while streaming. Read-only on the archive, so it can
run concurrently with other readers of the same tar.gz.
"""
from __future__ import annotations

import argparse
import tarfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "data" / "downloads" / "train_clean_100.tar.gz"
OUT = REPO / "data" / "raw" / "librittsr-full"

PROGRESS_EVERY_S = 30.0


def spk_of(member_name: str) -> str | None:
    """Speaker ID for an archive member, or None if not train-clean-100 data.

    Layout: LibriTTS_R/train-clean-100/<SPK>/<book>/<file>.<ext>
    (same detection rule as extract_librittsr_slice.py).
    """
    parts = member_name.split("/")
    if len(parts) < 5:
        return None
    if not parts[1].startswith("train"):
        return None
    return parts[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    archive: Path = args.archive
    out: Path = args.out
    assert archive.exists(), f"missing {archive}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    last_report = t0
    n_wav = n_txt = n_skipped_files = 0
    # per touched speaker: {filename -> member}, built during the same pass
    expected: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    skipped_spk: set[str] = set()

    print(f"streaming {archive.name} -> {out}", flush=True)
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            spk = spk_of(m.name)
            if spk is None or m.isdir():
                continue
            if spk in skipped_spk:
                continue
            spk_dir = out / spk
            if (spk_dir / ".complete").exists():
                skipped_spk.add(spk)
                continue

            fname = Path(m.name).name
            # NOTE: Path.suffix of "<utt>.normalized.txt" is ".txt" (last
            # suffix only), so match on the full name, not on suffix.
            if fname.endswith(".wav"):
                dst_name = fname
            elif fname.endswith(".normalized.txt"):
                dst_name = fname[: -len(".normalized.txt")] + ".txt"
            else:
                continue  # .trans.txt and book-level files not needed
            expected[spk][dst_name] = m

            dst = spk_dir / dst_name
            if dst.exists():
                n_skipped_files += 1
                continue
            spk_dir.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if ext == ".wav":
                dst.write_bytes(src.read())
                n_wav += 1
            else:
                dst.write_text(src.read().decode("utf-8"), encoding="utf-8")
                n_txt += 1

            now = time.perf_counter()
            if now - last_report >= PROGRESS_EVERY_S:
                last_report = now
                print(f"  ... {n_wav} wavs written, {len(expected)} speakers "
                      f"touched, {now - t0:.0f}s elapsed", flush=True)

    completed: list[str] = []
    incomplete: list[str] = []
    for spk in sorted(expected):
        spk_dir = out / spk
        on_disk = {p.name for p in spk_dir.iterdir() if p.name != ".complete"}
        want = set(expected[spk])
        missing = want - on_disk
        if missing:
            incomplete.append(spk)
            print(f"  INCOMPLETE {spk}: {len(missing)} file(s) missing "
                  f"(e.g. {sorted(missing)[:3]})", flush=True)
            continue
        (spk_dir / ".complete").write_text("ok\n", encoding="utf-8")
        completed.append(spk)

    dt = time.perf_counter() - t0
    print(f"done in {dt:.0f}s: {n_wav} wavs + {n_txt} txts written, "
          f"{n_skipped_files} files already present (resume skips), "
          f"{len(skipped_spk)} speakers skipped (complete), "
          f"{len(completed)} speakers completed this run, "
          f"{len(incomplete)} incomplete -> {out}", flush=True)
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
