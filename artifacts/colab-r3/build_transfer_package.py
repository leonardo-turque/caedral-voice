"""Build the colab-r3 transfer package (run locally, once per code state).

Creates, inside artifacts/colab-r3/:
  code-package.zip    caedral_voice/, configs/, data/manifests/**,
                      data/scripts/**, the CTC-aligner scripts, the regressor
                      launcher, benchmarks/harness/, requirements-colab.txt
  teacher-kokoro.zip  data/processed/teacher-kokoro/** +
                      data/manifests/teacher-kokoro{,.phonemes}.jsonl +
                      data/manifests/teacher-kokoro.durations.jsonl
  transfer_manifest.json   SHA256 + byte sizes of both zips

It then patches the SHA256 manifest INTO COLAB_BOOTSTRAP.py between the
TRANSFER_MANIFEST-BEGIN/END sentinels (single source of truth for Colab) and
prints the founder upload checklist.

Usage: .venv/Scripts/python.exe artifacts/colab-r3/build_transfer_package.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "colab-r3"
CODE_ZIP = OUT / "code-package.zip"
TEACHER_ZIP = OUT / "teacher-kokoro.zip"
MANIFEST = OUT / "transfer_manifest.json"
BOOTSTRAP = OUT / "COLAB_BOOTSTRAP.py"

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
_ZIP_EPOCH = (2026, 8, 23, 0, 0, 0)


def os_join(*parts: str) -> str:
    return "/".join(parts)


CODE_TREES = ["caedral_voice", "configs", os_join("data", "manifests"),
              os_join("data", "scripts"), os_join("benchmarks", "harness")]
CODE_FILES = [
    "requirements-colab.txt",
    os_join("experiments", "CV-G1-E0011-ctc-aligner", "train_aligner.py"),
    os_join("experiments", "CV-G1-E0011-ctc-aligner", "write_durations.py"),
    os_join("experiments", "CV-G1-E0011-ctc-aligner", "validate_aligner.py"),
    os_join("experiments", "CV-G1-E0013-regressor", "train_regressor.py"),
]
TEACHER_MANIFESTS = [
    os_join("data", "manifests", "teacher-kokoro.jsonl"),
    os_join("data", "manifests", "teacher-kokoro.phonemes.jsonl"),
    os_join("data", "manifests", "teacher-kokoro.durations.jsonl"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def add_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> int:
    info = zipfile.ZipInfo(arcname.replace("\\", "/"), date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with open(path, "rb") as fh:
        data = fh.read()
    zf.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6)
    return len(data)


def build_code_zip() -> tuple[int, int]:
    n_files = total = 0
    with zipfile.ZipFile(CODE_ZIP, "w", allowZip64=True) as zf:
        for tree in CODE_TREES:
            root = REPO / Path(*tree.split("/"))
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if any(part in EXCLUDE_DIRS for part in path.parts):
                    continue
                if path.suffix == ".pyc":
                    continue
                arc = path.relative_to(REPO).as_posix()
                total += add_file(zf, path, arc)
                n_files += 1
        for rel in CODE_FILES:
            if rel == "requirements-colab.txt":
                src = OUT / rel          # authored next to this script
            else:
                src = REPO / Path(*rel.split("/"))
            if not src.is_file():
                sys.exit(f"FATAL: required file missing: {src}")
            total += add_file(zf, src, rel)
            n_files += 1
    return n_files, total


def build_teacher_zip() -> tuple[int, int]:
    n_files = total = 0
    proc_root = REPO / "data" / "processed" / "teacher-kokoro"
    wavs = sorted(p for p in proc_root.rglob("*.wav") if p.is_file())
    if not wavs:
        sys.exit(f"FATAL: no teacher wavs under {proc_root}")
    with zipfile.ZipFile(TEACHER_ZIP, "w", allowZip64=True) as zf:
        for path in wavs:
            arc = path.relative_to(REPO).as_posix()
            total += add_file(zf, path, arc)
            n_files += 1
        for rel in TEACHER_MANIFESTS:
            src = REPO / Path(*rel.split("/"))
            if not src.is_file():
                sys.exit(f"FATAL: required manifest missing: {src}")
            total += add_file(zf, src, rel)
            n_files += 1
    return n_files, total


def patch_bootstrap(manifest: dict) -> bool:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(# === TRANSFER_MANIFEST-BEGIN \(auto-patched by "
        r"build_transfer_package\.py\) ===\n).*?"
        r"(\n# === TRANSFER_MANIFEST-END ===)",
        re.DOTALL)
    payload = ("TRANSFER_MANIFEST: dict = "
               + json.dumps(manifest, indent=4, sort_keys=True))
    new_text, n = pattern.subn(lambda m: m.group(1) + payload + m.group(2),
                               text)
    if n != 1:
        print(f"WARN: bootstrap sentinels not found/unique (n={n}); "
              "bootstrap will fall back to transfer_manifest.json")
    BOOTSTRAP.write_text(new_text, encoding="utf-8", newline="\n")
    return n == 1


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] building {CODE_ZIP.name} ...", flush=True)
    n1, b1 = build_code_zip()

    print(f"[2/4] building {TEACHER_ZIP.name} (3.7 GB of wavs, be patient) ...",
          flush=True)
    n2, b2 = build_teacher_zip()

    print("[3/4] hashing zips + writing transfer_manifest.json ...",
          flush=True)
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            CODE_ZIP.name: {"sha256": sha256_file(CODE_ZIP),
                            "bytes": CODE_ZIP.stat().st_size},
            TEACHER_ZIP.name: {"sha256": sha256_file(TEACHER_ZIP),
                               "bytes": TEACHER_ZIP.stat().st_size},
        },
        "contents": {
            "code_files": n1,
            "teacher_wavs": n2,
            "librittsr_manifest_rows": 30917,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    print("[4/4] embedding manifest into COLAB_BOOTSTRAP.py ...", flush=True)
    patched = patch_bootstrap(manifest)

    print("\n=== PACKAGE READY ===")
    for z in (CODE_ZIP, TEACHER_ZIP):
        print(f"{z.name}\n  bytes : {z.stat().st_size:,}\n"
              f"  sha256: {manifest['files'][z.name]['sha256']}")
    print(f"\npatched into COLAB_BOOTSTRAP.py: {patched}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print("\nFOUNDER UPLOAD CHECKLIST -- drag ALL THREE files from "
          "artifacts/colab-r3/ into Google Drive folder MyDrive/caedral/:")
    print("  1. code-package.zip")
    print("  2. teacher-kokoro.zip")
    print("  3. COLAB_BOOTSTRAP.py")
    print("Wait until Drive finishes uploading BOTH zips (no 'uploading' "
          "badge), then follow COLAB_BOOTSTRAP_CELL.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
