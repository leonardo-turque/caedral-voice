"""Extract a deterministic multi-speaker LibriTTS-R slice.

Selects the first N speakers (sorted by ID) from train_clean_100 and extracts
their wavs + normalized transcripts into data/raw/librittsr-slice/.
Single sequential gzip pass; resumable by skipping complete speaker dirs.
"""
from __future__ import annotations

import argparse
import io
import tarfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "data" / "downloads" / "train_clean_100.tar.gz"
OUT = REPO / "data" / "raw" / "librittsr-slice"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=10)
    args = ap.parse_args()

    assert ARCHIVE.exists(), f"missing {ARCHIVE}"
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("scanning archive for speakers ...", flush=True)
    members_by_spk: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for m in tf:
            parts = m.name.split("/")
            # LibriTTS-R layout: LibriTTS_R/train-clean-100/<SPK>/<book>/<file>.<ext>
            if len(parts) >= 5 and not m.isdir():
                spk = parts[2] if parts[1].startswith("train") else parts[1]
                members_by_spk[spk].append(m)
    selected = sorted(members_by_spk.keys())[: args.speakers]
    print(f"{len(members_by_spk)} speakers found; selecting first "
          f"{len(selected)}: {selected}", flush=True)

    wanted = set(selected)
    done_wavs = 0
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for spk in sorted(wanted):
            spk_dir = OUT / spk
            marker = spk_dir / ".complete"
            if marker.exists():
                continue
            spk_dir.mkdir(parents=True, exist_ok=True)
        for m in tf:
            parts = m.name.split("/")
            if len(parts) < 5 or m.isdir():
                continue
            spk = parts[2] if parts[1].startswith("train") else parts[1]
            if spk not in wanted:
                continue
            fname = Path(m.name).name
            stem, ext = Path(fname).stem, Path(fname).suffix
            # flatten book dirs: keep only wav + normalized transcript
            if ext == ".wav":
                dst = OUT / spk / fname
                if not dst.exists():
                    src = tf.extractfile(m)
                    dst.write_bytes(src.read())
                    done_wavs += 1
            elif fname.endswith(".normalized.txt"):
                dst = OUT / spk / f"{stem}.normalized.txt"
                if not dst.exists():
                    src = tf.extractfile(m)
                    dst.write_text(src.read().decode("utf-8"), encoding="utf-8")
    for spk in sorted(wanted):
        (OUT / spk / ".complete").write_text("ok\n", encoding="utf-8")
    print(f"extracted {done_wavs} wavs in {time.perf_counter()-t0:.0f}s -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
