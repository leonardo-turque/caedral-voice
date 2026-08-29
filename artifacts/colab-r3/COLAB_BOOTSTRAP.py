"""COLAB_BOOTSTRAP.py -- CAEDRAL VOICE colab-r3 single-file orchestrator.

Run INSIDE a Colab GPU session (see COLAB_BOOTSTRAP_CELL.md). Stages:

  0. environment: CUDA detect, GPU name/VRAM, CUBLAS determinism env
  1. transfer:    unzip founder zips from Drive, SHA256-verify vs the
                  embedded TRANSFER_MANIFEST (patched in by
                  build_transfer_package.py at package build time)
  2. deps:        pip install -r requirements-colab.txt (quiet)
  3. corpus:      download LibriTTS-R train_clean_100.tar.gz (resumable),
                  MD5-verify (full official hash), extract flattened layout
  4. preprocess:  reproduce data/processed/librittsr-clean100 with the repo's
                  deterministic pipeline (parallel, resumable), then verify
                  EVERY kept wav's sha256 against the shipped manifest rows
  5. smoke:       load CaedralRegressor, one forward/backward on cuda
  6. amp sanity:  N=50 steps FP32 vs AMP-FP16 on real batches; AMP is enabled
                  for the real run only if the divergence rule below passes
  7. benchmark:   200 measured steps -> sec/step median, samples/sec, VRAM
                  peak, dataloader wait; compared against the local CPU r2
                  baseline
  8. cutover:     only if every gate passes print "COLAB R3 VALIDATED -- SAFE
                  TO CUT OVER" and exec full training (20k steps,
                  resume-capable) with an ETA reporter thread and Drive
                  checkpoint persistence. On ANY gate failure: print failure
                  diagnosis and exit non-zero WITHOUT touching local r3 --
                  the local CPU r3 run on the founder's machine keeps running
                  regardless of anything this script does or doesn't do.

Idempotence: every stage drops a marker under <repo>/.colab-r3-state/, so a
crashed/interrupted run resumes where it left off. If
experiments/colab-r3/checkpoints/last.pt exists the gates are skipped and the
training subprocess resumes from it automatically (CAEDRAL_FORCE_GATES=1
forces a re-run of all gates).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Must be set before torch creates its first cuBLAS handle so deterministic
# algorithms are actually enforceable on CUDA (mirrors training/seed.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  (Colab preinstalls it)

# --------------------------------------------------------------------------- #
# configuration (edit here if your Drive layout differs)
# --------------------------------------------------------------------------- #
DRIVE_CAEDRAL = Path("/content/drive/MyDrive/caedral")
TEACHER_ZIP = DRIVE_CAEDRAL / "teacher-kokoro.zip"
CODE_ZIP = DRIVE_CAEDRAL / "code-package.zip"
REPO = Path("/content/caedral-voice")
STATE = REPO / ".colab-r3-state"
R3_DRIVE = Path("/content/drive/MyDrive/caedral-r3/checkpoints")

LIBRITTS_URL = "https://openslr.elda.org/resources/141/train_clean_100.tar.gz"
LIBRITTS_MD5 = "6df668d8f5f33e70876bfa33862ad02b"   # official SLR141 md5sum.txt
LIBRITTS_BYTES = 8129817509                          # measured locally, DATA-PILOT-ACQUISITION.md section 1
LIBRITTS_TARBALL = Path("/content/train_clean_100.tar.gz")

TOTAL_STEPS = 20000
CKPT_EVERY = 1000
PREVIEW_READY_FRACTION = 0.6        # operational heuristic: previews cut from checkpoints at >=60% of steps
ETA_INTERVAL_S = 600                # ETA reporter cadence (10 min)
PERSIST_POLL_S = 20                 # Drive persistence watcher cadence

FOUNDER_TZ = "America/Sao_Paulo"

# Local baseline (experiments/r2/metrics.jsonl): median 1.5095 s/step over its
# full 15k-step window on the founder's 6-thread CPU; planning figure ~1.7.
R2_CPU_BASELINE_SEC_PER_STEP = 1.51

# Frozen preprocessing facts of librittsr-clean100 (data/manifests/*.meta.json)
EXPECTED_PREPROCESS = {
    "seen": 33232, "kept": 30917, "rejected": 2315,
    "by_reason": {"DUPLICATE_TRANSCRIPT": 588, "TOO_LONG": 1705,
                  "TOO_QUIET": 9, "TOO_SHORT": 13},
}

# --- AMP sanity rule (documented contract) ---------------------------------
AMP_N_STEPS = 50
AMP_RULE_DOC = (
    "AMP-FP16 is enabled for the real run iff, over N identical-seed steps on "
    "identical batches: (a) every FP16 loss is finite, (b) median relative "
    "loss divergence vs the FP32 twin stays < 5%, and (c) final-step relative "
    "divergence stays < 25%. Otherwise the run proceeds in FP32 (still "
    "validated). Divergence = |L16-L32| / max(1e-8, |L32|).")
AMP_MEDIAN_REL_MAX = 0.05
AMP_FINAL_REL_MAX = 0.25

BENCH_WARMUP_STEPS = 10
BENCH_MEASURED_STEPS = 200
# Cutover requires the GPU to actually beat the machine that keeps running.
BENCH_MUST_BEAT_CPU_BASELINE = True

# === TRANSFER_MANIFEST-BEGIN (auto-patched by build_transfer_package.py) ===
TRANSFER_MANIFEST: dict = {
    "contents": {
        "code_files": 67,
        "librittsr_manifest_rows": 30917,
        "teacher_wavs": 15501
    },
    "files": {
        "code-package.zip": {
            "bytes": 18675618,
            "sha256": "baab591c7f96acab753ec41b24aa8004889900c37b31a01c43d00d27669dd9f7"
        },
        "teacher-kokoro.zip": {
            "bytes": 3270946734,
            "sha256": "6dade3b7951f437e619488b160c3b4994658733c908b650916720eb01cf7de3e"
        }
    },
    "generated_utc": "2026-08-23T13:23:39Z"
}
# === TRANSFER_MANIFEST-END ===


def _manifest() -> dict:
    """Embedded manifest first; fallback to a JSON file next to this script."""
    if TRANSFER_MANIFEST:
        return TRANSFER_MANIFEST
    for cand in (Path(__file__).resolve().parent / "transfer_manifest.json",
                 REPO / "transfer_manifest.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> bool:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def stage(n: int, title: str) -> None:
    print(f"\n=== STAGE {n}: {title} " + "=" * max(4, 58 - len(title)),
          flush=True)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def md5_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def marker(name: str) -> Path:
    return STATE / name


def marker_ok(name: str, force_env: str = "CAEDRAL_FORCE_GATES") -> bool:
    return marker(name).is_file() and os.environ.get(force_env) != "1"


# --------------------------------------------------------------------------- #
# stage 0 -- environment
# --------------------------------------------------------------------------- #
def stage0_env() -> tuple[bool, str]:
    stage(0, "environment")
    if not torch.cuda.is_available():
        record("cuda_available", False,
               "torch.cuda.is_available() is False -- Runtime > Change runtime "
               "type > GPU, then rerun")
        return False, "no cuda"
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / (1 << 30)
    record("cuda_available", True,
           f"{props.name}, {vram_gib:.1f} GiB VRAM, cc{props.major}."
           f"{props.minor}, torch {torch.__version__}")
    print(f"[env] python={sys.version.split()[0]} "
          f"platform={sys.platform}", flush=True)
    print(f"[env] runtime CPU count: {os.cpu_count()}", flush=True)
    return True, f"cuda:{props.name}"


# --------------------------------------------------------------------------- #
# stage 1 -- transfer + integrity
# --------------------------------------------------------------------------- #
def stage1_transfer() -> bool:
    stage(1, "transfer verification (Drive zips)")
    man = _manifest()
    files = man.get("files", {})
    if not files:
        return record("transfer_manifest", False,
                      "TRANSFER_MANIFEST empty and no fallback "
                      "transfer_manifest.json found -- rebuild the package "
                      "with build_transfer_package.py")

    DRIVE_CAEDRAL.mkdir(parents=True, exist_ok=True)
    ok_all = True
    targets = [
        (CODE_ZIP, "code-package.zip", REPO),
        (TEACHER_ZIP, "teacher-kokoro.zip", REPO),
    ]
    for path, key, dest in targets:
        spec = files.get(key)
        if not path.is_file():
            ok_all &= record(f"present:{key}", False,
                             f"{path} missing -- drag it into Drive "
                             f"MyDrive/caedral/ first (see COLAB_BOOTSTRAP_CELL.md)")
            continue
        size = path.stat().st_size
        ok_size = (spec.get("bytes") == size)
        print(f"  hashing {key} ({size:,} B)...", flush=True)
        digest = sha256_file(path)
        ok_hash = (spec.get("sha256") == digest)
        ok_all &= record(f"sha256:{key}", ok_size and ok_hash,
                         f"{'match' if ok_hash else 'MISMATCH '
                            '(re-upload the zip)'}"
                         + ("" if ok_size else f"; bytes {size} != manifest "
                                               f"{spec.get('bytes')}"))
        if ok_hash and ok_size:
            marker("hash_" + key.replace("-", "_")).write_text(digest)
            # unzip only once per verified artifact
            stamp = STATE / ("unzipped_" + key.replace("-", "_"))
            if not stamp.exists():
                print(f"  unzipping {key} -> {dest} ...", flush=True)
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(dest)
                stamp.write_text("ok")
            else:
                print(f"  unzip skipped (marker present)", flush=True)

    # cross-check the code zip's own copy of the manifest, when readable
    inner = REPO / "transfer_manifest.json"
    if inner.is_file() and files:
        try:
            same = json.loads(inner.read_text(encoding="utf-8")) == man
            record("inner_manifest_agrees", same,
                   "code-package carries an identical manifest copy"
                   if same else "embedded vs inner manifest differ -- rebuild")
            ok_all &= same
        except Exception as exc:  # noqa: BLE001
            record("inner_manifest_agrees", False, f"unreadable: {exc}")
    return ok_all


# --------------------------------------------------------------------------- #
# stage 2 -- dependencies
# --------------------------------------------------------------------------- #
def stage2_deps() -> bool:
    stage(2, "dependencies (pip -q -r requirements-colab.txt)")
    req = REPO / "requirements-colab.txt"
    if not req.is_file():
        return record("requirements", False, f"{req} missing")
    if marker_ok("deps_ok"):
        return record("pip_install", True, "skipped (marker present)")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                          "-r", str(req)])
    ok = rc == 0
    if ok:
        # NLTK model data for the g2p_en fallback (phonemes ship frozen in the
        # code zip, so this is belt-and-braces; never fatal).
        try:
            import nltk
            for pkg in ("cmudict", "averaged_perceptron_tagger",
                        "averaged_perceptron_tagger_eng"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        marker("deps_ok").write_text("ok")
    return record("pip_install", ok, f"exit code {rc}")


# --------------------------------------------------------------------------- #
# stage 3 -- LibriTTS-R download + extract (flattened layout)
# --------------------------------------------------------------------------- #
RAW_OUT = REPO / "data" / "raw" / "librittsr-full"


def _download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming download at {have:,} B", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", 200)
        if have and status != 206:
            # mirror ignored the Range header -> body starts at byte 0;
            # appending would corrupt the file, so start over cleanly
            print("  server ignored Range; restarting download from 0",
                  flush=True)
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        last = time.time()
        got = have
        with open(part, mode) as fh:
            while True:
                blk = resp.read(1 << 22)
                if not blk:
                    break
                fh.write(blk)
                got += len(blk)
                if time.time() - last > 30:
                    last = time.time()
                    pct = f"{got / total * 100:5.1f}%" if total else "?"
                    print(f"  download {pct} ({got:,} B)", flush=True)
    os.replace(part, dest)


def _extract_librittsr(archive: Path, out_dir: Path) -> int:
    """Flatten train-clean-100 members to <spk>/<utt>.wav|<utt>.txt.

    Mirrors data/scripts/extract_librittsr_full.py's layout contract (the
    archive's <utt>.normalized.txt becomes <utt>.txt so preprocess.py's
    sibling-.txt discovery keys by wav stem); resume-safe by file existence.
    """
    n_wav = 0
    t_last = time.time()
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            parts = m.name.split("/")
            if len(parts) < 5 or m.isdir() or not parts[1].startswith("train"):
                continue
            fname = parts[-1]
            dst_name = None
            if fname.endswith(".wav"):
                dst_name = fname
            elif fname.endswith(".normalized.txt"):
                dst_name = fname[: -len(".normalized.txt")] + ".txt"
            if dst_name is None:
                continue
            spk_dir = out_dir / parts[2]
            dst = spk_dir / dst_name
            if dst.exists():
                if dst_name.endswith(".wav"):
                    n_wav += 1
                continue
            spk_dir.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            data = src.read()
            dst.write_bytes(data)
            if dst_name.endswith(".wav"):
                n_wav += 1
                if time.time() - t_last > 30:
                    t_last = time.time()
                    print(f"  extracted {n_wav} wavs so far...", flush=True)
    return n_wav


def stage3_corpus() -> bool:
    stage(3, "LibriTTS-R train-clean-100 (download + MD5 + extract)")
    need_download = not LIBRITTS_TARBALL.is_file()
    if LIBRITTS_TARBALL.is_file() and \
            md5_file(LIBRITTS_TARBALL) == LIBRITTS_MD5:
        print("  tarball already verified on disk", flush=True)
    else:
        need_download = True
    if need_download:
        print(f"  downloading {LIBRITTS_URL} (~7.6 GiB)...", flush=True)
        try:
            _download(LIBRITTS_URL, LIBRITTS_TARBALL)
        except Exception as exc:  # noqa: BLE001
            return record("librittsr_download", False,
                          f"{type(exc).__name__}: {exc}")
    size = LIBRITTS_TARBALL.stat().st_size
    digest = md5_file(LIBRITTS_TARBALL)
    ok_dl = (digest == LIBRITTS_MD5)
    record("librittsr_md5", ok_dl,
           f"{digest} ({size:,} B)" + ("" if ok_dl else
           f" != official {LIBRITTS_MD5} -- delete {LIBRITTS_TARBALL} and rerun"))

    stamp = STATE / "corpus_extracted"
    if not stamp.exists():
        print(f"  extracting -> {RAW_OUT} (flattened <spk>/<utt>)...",
              flush=True)
        RAW_OUT.mkdir(parents=True, exist_ok=True)
        n = _extract_librittsr(LIBRITTS_TARBALL, RAW_OUT)
        stamp.write_text(str(n))
    n_wav = sum(1 for p in RAW_OUT.rglob("*.wav"))
    ok_ex = record("librittsr_extract", n_wav == EXPECTED_PREPROCESS["seen"],
                   f"{n_wav} raw wavs (expected {EXPECTED_PREPROCESS['seen']})")
    return ok_dl and ok_ex


# --------------------------------------------------------------------------- #
# stage 4 -- deterministic preprocessing + sha256 audit
# --------------------------------------------------------------------------- #
def _preprocess_params():
    """Exact parameter set recorded in librittsr-clean100.meta.json (all
    values are the repo pipeline defaults; imported live to avoid drift)."""
    sys.path.insert(0, str(REPO / "data" / "scripts"))
    import pipeline_lib as pl  # noqa: PLC0415
    return pl, {
        "target_sr": pl.TARGET_SR_DEFAULT,          # 22050
        "lufs_target": pl.LUFS_TARGET_DEFAULT,      # -23.0
        "min_dur": pl.MIN_DURATION_S_DEFAULT,       # 0.3
        "max_dur": pl.MAX_DURATION_S_DEFAULT,       # 15.0
        "max_clipped_frac": pl.CLIP_REJECT_FRAC_DEFAULT,
        "clip_abs_threshold": pl.CLIP_ABS_THRESHOLD,
        "trim_top_db": pl.TRIM_TOP_DB_DEFAULT,
        "trim_pad_ms": pl.TRIM_PAD_MS,
        "highpass_hz": pl.HIGHPASS_HZ_DEFAULT,
        "resampler": "soxr-HQ" if pl._soxr is not None
                     else "scipy-resample_poly",
        "pipeline_version": pl.PIPELINE_VERSION,
    }


def _worker_process(args):
    """Pool worker: one utterance through preprocess.process_utterance.

    Runs in a fork()ed child that never touches CUDA (audio DSP only), the
    same pattern torch DataLoader workers use."""
    wav_path, out_dir, sr, params, uid = args
    import preprocess as pp  # data/scripts on sys.path via fork inheritance
    try:
        row = pp.process_utterance(wav_path, out_dir, sr, params, uid=uid)
        return (uid, None, row)
    except pp.Rejected as rej:
        return (uid, (rej.reason, rej.detail), None)
    except Exception as exc:  # noqa: BLE001
        return (uid, ("PROCESSING_ERROR", f"{type(exc).__name__}: {exc}"), None)


def stage4_preprocess() -> bool:
    stage(4, "deterministic preprocessing (librittsr-clean100)")
    if marker_ok("preprocess_ok") :
        return record("preprocess", True, "skipped (verified marker present)")

    pl, params = _preprocess_params()
    import preprocess as pp  # noqa: PLC0415
    manifests_dir = REPO / "data" / "manifests"
    processed_dir = REPO / "data" / "processed" / "librittsr-clean100"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # shipped ground truth (byte-frozen locally)
    ship_rows = [json.loads(l) for l in
                 (manifests_dir / "librittsr-clean100.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
    expect_sha = {r["id"]: r["sha256_audio"] for r in ship_rows}
    expect_row = {r["id"]: r for r in ship_rows}

    transcripts = pp.load_transcripts(RAW_OUT)
    wavs = pp.find_wavs(RAW_OUT)
    seen_n = len(wavs)

    # ---- plan the compute set ------------------------------------------
    # A wav needs computing iff it has a transcript AND its output file does
    # not already exist with the manifest's sha256 (idempotent resume).
    # Dedup is NOT decided here on purpose: run_pipeline claims a text hash
    # only AFTER a clip passes audio QC, so a QC-rejected first occurrence
    # must not shadow a later duplicate. The pool below merely WRITES wavs;
    # the sequential assembly walk afterwards replays the exact original
    # keep/reject order and audits bytes against the frozen manifest.
    plan: list[tuple[Path, str]] = []
    reuse = 0
    for wav_path in wavs:                       # sorted order
        uid = wav_path.stem
        if transcripts.get(uid) is None:
            continue                            # handled in assembly walk
        out_wav = processed_dir / f"{uid}.wav"
        exp_sha = expect_sha.get(uid)
        if (exp_sha and out_wav.is_file()
                and pl.sha256_file(out_wav) == exp_sha):
            reuse += 1
            continue
        plan.append((wav_path, uid))
    print(f"  plan: seen={seen_n} reuse_verified={reuse} "
          f"compute={len(plan)} (workers={max(1, os.cpu_count() or 2)})",
          flush=True)

    comp_rejected: dict[str, tuple[str, str]] = {}
    if plan:
        import multiprocessing as mp
        ctx = mp.get_context("fork")  # Linux Colab: cheap, inherits sys.path
        jobs = [(w, processed_dir, params["target_sr"], params, u)
                for (w, u) in plan]
        with ctx.Pool(processes=max(1, os.cpu_count() or 2)) as pool:
            for i, (uid, rej, _row) in enumerate(
                    pool.imap_unordered(_worker_process, jobs, chunksize=32)):
                if rej is not None:
                    comp_rejected[uid] = rej
                if (i + 1) % 2000 == 0:
                    print(f"  processed {i + 1}/{len(jobs)}", flush=True)

    # ---- sequential assembly walk: exact run_pipeline order + byte audit --
    rejected: list[dict] = []
    kept_rows: list[dict] = []
    mismatches: list[str] = []
    norm_hashes_seen: set[str] = set()

    def reject(uid, reason, detail=""):
        rejected.append({"id": uid, "reason": reason, "detail": detail})

    for wav_path in wavs:
        uid = wav_path.stem
        transcript = transcripts.get(uid)
        if transcript is None:
            reject(uid, pl.REASON_MISSING_TRANSCRIPT,
                   "no metadata.csv entry or sibling .txt")
            continue
        nh = pl.sha256_text(pl.normalize_transcript(transcript))
        if nh in norm_hashes_seen:
            reject(uid, pl.REASON_DUPLICATE,
                   "hash of normalized transcript already emitted")
            continue
        if uid in comp_rejected:
            reason, detail = comp_rejected[uid]
            reject(uid, reason, detail)
            continue
        out_wav = processed_dir / f"{uid}.wav"
        actual = pl.sha256_file(out_wav) if out_wav.is_file() else None
        exp = expect_sha.get(uid)
        if exp is None or exp != actual:
            # freshly computed yet differing, missing, absent-from-manifest,
            # or unaccounted -- preprocessing drifted either way: not safe
            mismatches.append(uid)
            continue
            # freshly computed yet differing, missing, or absent from the
            # frozen manifest -- preprocessing drifted either way: not safe
            mismatches.append(uid)
            continue
        norm_hashes_seen.add(nh)                # hash claimed on KEEP only
        kept_rows.append({k: expect_row[uid][k] for k in pp.ROW_FIELDS})

    counts = {"seen": seen_n, "kept": len(kept_rows),
              "rejected": len(rejected)}
    exp = EXPECTED_PREPROCESS
    ok_counts = (counts == {"seen": exp["seen"], "kept": exp["kept"],
                            "rejected": exp["rejected"]})
    record("preprocess_counts", ok_counts,
           f"{counts} expected seen={exp['seen']} kept={exp['kept']} "
           f"rejected={exp['rejected']}")

    ok_sha = not mismatches
    record("audio_sha256_audit", ok_sha,
           f"{len(kept_rows) - len(mismatches)}/{exp['kept']} kept wavs match "
           "the frozen manifest sha256_audio"
           + ("" if ok_sha else f"; FIRST MISMATCHES: {mismatches[:5]} "
              "-- preprocessing drifted (check pinned versions in "
              "requirements-colab.txt); NOT safe to train"))

    if ok_counts and ok_sha:
        def dump(path, rows):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=True) + "\n")
        dump(manifests_dir / "librittsr-clean100.jsonl", kept_rows)
        dump(manifests_dir / "librittsr-clean100.rejected.jsonl", rejected)
        meta = json.loads((manifests_dir / "librittsr-clean100.meta.json")
                          .read_text(encoding="utf-8"))
        meta["counts"] = counts
        meta["colab_rerun"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                               "sha256_audited": True}
        (manifests_dir / "librittsr-clean100.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        marker("preprocess_ok").write_text(json.dumps(counts))
        print("[note] .phonemes.jsonl ships frozen inside code-package.zip -- "
              "phonemization is NOT rerun on Colab", flush=True)
    return ok_counts and ok_sha


# --------------------------------------------------------------------------- #
# shared model/batch helpers (smoke, AMP sanity, benchmark)
# --------------------------------------------------------------------------- #
def _build_dataset_and_loaders(seed: int = 1234):
    sys.path.insert(0, str(REPO))
    from caedral_voice.model.regressor import (  # noqa: PLC0415
        CaedralRegressor, CaedralRegressorConfig)
    from caedral_voice.training.mel_dataset import (  # noqa: PLC0415
        LengthBucketBatchSampler, PhonemeVocab)
    from caedral_voice.training.mel_cache import MelDiskCache  # noqa: PLC0415
    from caedral_voice.training.regressor_dataset import (  # noqa: PLC0415
        RegressorDataset, build_speaker_index, collate_regressor, load_jsonl,
        stratified_split)
    from caedral_voice.training.seed import set_determinism  # noqa: PLC0415

    set_determinism(seed, deterministic_algorithms=True, num_threads=None)
    manifests = REPO / "data" / "manifests"

    def jl(name):
        return [json.loads(l) for l in
                (manifests / name).read_text(encoding="utf-8").splitlines()
                if l.strip()]

    audio, phones = jl("librittsr-clean100.jsonl"), \
        jl("librittsr-clean100.phonemes.jsonl")
    durs = jl("librittsr-clean100.durations.jsonl")
    spk = build_speaker_index(audio)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phones])
    cache = MelDiskCache(REPO / "data" / "cache" / "mel-v1", sample_rate=24000,
                         n_mels=80, hop=256, n_fft=1024)
    ds = RegressorDataset(audio, phones, durs, spk, sample_rate=24000,
                          n_mels=80, hop=256, vocab=vocab, repo_root=REPO,
                          disk_cache=cache)
    tr_idx, va_idx = stratified_split(ds.rows, 0.05)
    tr_smp = LengthBucketBatchSampler(tr_idx, ds.est_frames, max_frames=2000,
                                      max_utts=16, bucket_size=32,
                                      shuffle=True, base_seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=tr_smp,
                                         collate_fn=collate_regressor,
                                         num_workers=2, pin_memory=True)
    cfg = CaedralRegressorConfig()
    cfg.n_speakers = max(cfg.n_speakers, len(spk))
    return ds, loader, CaedralRegressor(cfg), vocab.size, cfg


def _one_step(model, optimizer, batch_x, scaler=None, device="cuda"):
    from caedral_voice.training.trainer import Trainer  # noqa: PLC0415
    x = Trainer._unpack_batch(batch_x)[0] if isinstance(batch_x, dict) \
        else batch_x
    x = _to_dev(x, device)
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model.losses_for_training(
                mel=x["mel"], phoneme_ids=x["phoneme_ids"],
                durations=x["duration_targets"], speaker_id=x["speaker_ids"],
                style_id=0, mel_lengths=x["mel_lengths"],
                phoneme_lengths=x["phoneme_lengths"])
    else:
        out = model.losses_for_training(
            mel=x["mel"], phoneme_ids=x["phoneme_ids"],
            durations=x["duration_targets"], speaker_id=x["speaker_ids"],
            style_id=0, mel_lengths=x["mel_lengths"],
            phoneme_lengths=x["phoneme_lengths"])
    loss = out["loss"]
    lv = float(loss.detach())
    if not math.isfinite(lv):
        return lv, False
    (scaler.scale(loss) if scaler is not None else loss).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if scaler is not None:
        scaler.unscale_(optimizer)
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        if scaler is not None:
            scaler.update()
        return lv, False
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return lv, True


def _to_dev(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_dev(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dev(v, device) for v in obj)
    return obj


# --------------------------------------------------------------------------- #
# stage 5 -- smoke
# --------------------------------------------------------------------------- #
def stage5_smoke(model) -> bool:
    stage(5, "device smoke (1 forward/backward on cuda)")
    dev = next(model.parameters()).device
    assert str(dev).startswith("cuda"), f"model ended up on {dev}"
    g = torch.Generator().manual_seed(7)
    B, T_y, T_x, C = 2, 300, 60, 80
    mel = torch.rand(B, T_y, C, generator=g).to(dev)
    ids = torch.randint(2, 40, (B, T_x), generator=g).to(dev)
    durs = torch.full((B, T_x), T_y // T_x, dtype=torch.long, device=dev)
    spk = torch.zeros(B, dtype=torch.long, device=dev)
    xl = torch.full((B,), T_x, dtype=torch.long, device=dev)
    yl = torch.full((B,), T_y, dtype=torch.long, device=dev)
    out = model.losses_for_training(mel=mel, phoneme_ids=ids, durations=durs,
                                    speaker_id=spk, style_id=0,
                                    mel_lengths=yl, phoneme_lengths=xl)
    out["loss"].backward()
    grads_finite = all(bool(torch.isfinite(p.grad).all())
                       for p in model.parameters() if p.grad is not None)
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / (1 << 30)
    ok = math.isfinite(float(out["loss"])) and grads_finite
    return record("forward_backward_cuda", ok,
                  f"loss={float(out['loss']):.4f} grads_finite={grads_finite} "
                  f"peak_vram={peak_gib:.2f} GiB")


# --------------------------------------------------------------------------- #
# stage 6 -- AMP sanity (FP32 twin vs AMP-FP16 twin, same batches/seed)
# --------------------------------------------------------------------------- #
def stage6_amp_sanity(loader, model_cfg_fn) -> bool:
    stage(6, f"AMP sanity ({AMP_N_STEPS} steps FP32 vs AMP-FP16)")
    print(f"  RULE: {AMP_RULE_DOC}", flush=True)
    from caedral_voice.model.regressor import CaedralRegressor  # noqa: PLC0415
    from caedral_voice.training.seed import set_determinism  # noqa: PLC0415

    batches = []
    it = iter(loader)
    while len(batches) < AMP_N_STEPS:
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)

    set_determinism(1234, deterministic_algorithms=True, num_threads=None)
    m32 = model_cfg_fn().cuda(); o32 = torch.optim.AdamW(
        m32.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
    set_determinism(1234, deterministic_algorithms=True, num_threads=None)
    m16 = model_cfg_fn().cuda(); o16 = torch.optim.AdamW(
        m16.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
    sc = torch.amp.GradScaler("cuda", enabled=True)

    rel, finite16, l32s, l16s = [], True, [], []
    for i, b in enumerate(batches):
        l32, _ = _one_step(m32, o32, b, None)
        l16, _ = _one_step(m16, o16, b, sc)
        l32s.append(l32); l16s.append(l16)
        if not (math.isfinite(l16) and math.isfinite(l32)):
            finite16 = False
            break
        rel.append(abs(l16 - l32) / max(1e-8, abs(l32)))
        if (i + 1) % 10 == 0:
            print(f"  step {i+1}/{AMP_N_STEPS}: L32={l32:.4f} "
                  f"L16={l16:.4f} rel={rel[-1]:.4f}", flush=True)

    med_rel = statistics.median(rel) if rel else float("inf")
    fin_rel = rel[-1] if rel else float("inf")
    ok = (finite16 and med_rel < AMP_MEDIAN_REL_MAX
          and fin_rel < AMP_FINAL_REL_MAX)
    verdict = (f"finite16={finite16} median_rel={med_rel:.4f} "
               f"(< {AMP_MEDIAN_REL_MAX}) final_rel={fin_rel:.4f} "
               f"(< {AMP_FINAL_REL_MAX}) -> AMP {'ENABLED' if ok else 'DISABLED (FP32 run)'}")
    marker("amp_decision").write_text(
        json.dumps({"amp_ok": ok, "median_rel": med_rel, "final_rel": fin_rel,
                    "rule": AMP_RULE_DOC}))
    return record("amp_sanity", True, verdict)
    # NOTE: returns True even when AMP disabled -- a failed AMP comparison must
    # NOT block cutover; it only means the run goes out in FP32.


# --------------------------------------------------------------------------- #
# stage 7 -- benchmark
# --------------------------------------------------------------------------- #
def stage7_benchmark(loader, model) -> dict:
    stage(7, f"benchmark ({BENCH_WARMUP_STEPS} warmup + "
             f"{BENCH_MEASURED_STEPS} measured steps)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)

    def next_batch():
        nonlocal it
        t0 = time.perf_counter()
        while True:
            try:
                b = next(it)
                return b, time.perf_counter() - t0
            except StopIteration:
                it = iter(loader)

    step_times, wait_times, utts = [], [], 0
    done = 0
    while done < BENCH_WARMUP_STEPS + BENCH_MEASURED_STEPS:
        b, wait = next_batch()
        bs = int(b["x"]["mel_lengths"].shape[0])
        t0 = time.perf_counter()
        _one_step(model, opt, b, None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        done += 1
        if done > BENCH_WARMUP_STEPS:
            step_times.append(dt)
            wait_times.append(wait)
            utts += bs
        if (done - BENCH_WARMUP_STEPS) % 50 == 0 and done > BENCH_WARMUP_STEPS:
            print(f"  measured {done - BENCH_WARMUP_STEPS}/"
                  f"{BENCH_MEASURED_STEPS}", flush=True)

    med = statistics.median(step_times)
    mean = statistics.mean(step_times)
    samples_per_s = utts / sum(step_times)
    vram_peak = torch.cuda.max_memory_allocated() / (1 << 30)
    total = sum(step_times) + sum(wait_times)
    wait_frac = sum(wait_times) / total if total else 0.0
    proj_h = med * TOTAL_STEPS / 3600
    cpu_proj_h = R2_CPU_BASELINE_SEC_PER_STEP * TOTAL_STEPS / 3600
    print(f"  sec/step: median={med:.3f} mean={mean:.3f} "
          f"| samples/sec={samples_per_s:.1f} utts/s "
          f"| VRAM peak={vram_peak:.2f} GiB "
          f"| dataloader wait={wait_frac * 100:.1f}% of cycle", flush=True)
    print(f"  projection: {TOTAL_STEPS} steps ~ {proj_h:.1f} h GPU "
          f"(local CPU r2 baseline {R2_CPU_BASELINE_SEC_PER_STEP:.2f} s/step "
          f"~ {cpu_proj_h:.1f} h; speedup ~"
          f"{R2_CPU_BASELINE_SEC_PER_STEP / med:.1f}x)", flush=True)

    beats = (med < R2_CPU_BASELINE_SEC_PER_STEP
             if BENCH_MUST_BEAT_CPU_BASELINE else True)
    record("benchmark_beats_cpu_baseline", beats,
           f"GPU {med:.3f} s/step vs local CPU {R2_CPU_BASELINE_SEC_PER_STEP:.2f} s/step")
    return {"sec_step_median": med, "sec_step_mean": mean,
            "samples_per_s": samples_per_s, "vram_peak_gib": vram_peak,
            "dataloader_wait_frac": wait_frac}


# --------------------------------------------------------------------------- #
# stage 8 -- cutover gate + full training exec
# --------------------------------------------------------------------------- #
def _reporter_loop(proc, child_started, stop_evt):
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(FOUNDER_TZ)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    metrics_path = REPO / "experiments" / "colab-r3" / "metrics.jsonl"
    while not stop_evt.wait(ETA_INTERVAL_S):
        try:
            rows = []
            if metrics_path.is_file():
                with open(metrics_path, encoding="utf-8") as fh:
                    lines = fh.readlines()[-4000:]
                rows = [json.loads(l) for l in lines if l.strip()]
            tr = [r for r in rows if r.get("type") == "train"]
            if not tr:
                continue
            per = [(b["wall_s"] - a["wall_s"]) / (b["step"] - a["step"])
                   for a, b in zip(tr, tr[1:]) if b["step"] > a["step"]]
            recent = per[-100:]
            spd = statistics.median(recent) if recent else float("nan")
            step_now = tr[-1]["step"]
            elapsed = time.time() - child_started
            remain = max(0, TOTAL_STEPS - step_now) * spd
            eta = datetime.now(tz) + timedelta(seconds=remain)
            prev_eta = datetime.now(tz) + timedelta(
                seconds=max(0, int(TOTAL_STEPS * PREVIEW_READY_FRACTION)
                            - step_now) * spd)
            vram = ""
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                if q.returncode == 0:
                    vram = f" vram_used={int(q.stdout.strip())} MiB"
            except Exception:  # noqa: BLE001
                pass
            print(f"[eta] step {step_now}/{TOTAL_STEPS} "
                  f"({step_now/TOTAL_STEPS*100:.1f}%) | "
                  f"sec/step(med100)={spd:.3f} | elapsed={elapsed/3600:.2f}h | "
                  f"finish~{eta.strftime('%Y-%m-%d %H:%M')} {FOUNDER_TZ} | "
                  f"preview-ready(~{int(TOTAL_STEPS*PREVIEW_READY_FRACTION)} steps)"
                  f"~{prev_eta.strftime('%m-%d %H:%M')}{vram}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[eta] reporter error: {exc}", flush=True)


def _persist_loop(stop_evt, flush_evt):
    src_dir = REPO / "experiments" / "colab-r3" / "checkpoints"
    src_metrics = REPO / "experiments" / "colab-r3" / "metrics.jsonl"
    R3_DRIVE.mkdir(parents=True, exist_ok=True)
    last_sig = best_sig = m_sig = None

    def atomic_copy(src: Path, dst: Path):
        tmp = dst.with_name(dst.name + ".tmp-colab")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    while True:
        try:
            last = src_dir / "last.pt"
            best = src_dir / "best.pt"
            if last.is_file():
                sig = (last.stat().st_mtime_ns, last.stat().st_size)
                if sig != last_sig:
                    if R3_DRIVE.joinpath("last.pt").is_file() and last_sig:
                        atomic_copy(R3_DRIVE / "last.pt",
                                    R3_DRIVE / "prev-last.pt")
                    atomic_copy(last, R3_DRIVE / "last.pt")
                    last_sig = sig
                    print("[persist] synced last.pt -> Drive", flush=True)
            if best.is_file():
                sig = (best.stat().st_mtime_ns, best.stat().st_size)
                if sig != best_sig:
                    atomic_copy(best, R3_DRIVE / "best.pt")
                    best_sig = sig
                    print("[persist] synced best.pt -> Drive", flush=True)
            if src_metrics.is_file():
                sig = src_metrics.stat().st_size
                if sig != m_sig:
                    atomic_copy(src_metrics, R3_DRIVE / "metrics.jsonl")
                    m_sig = sig
            for stray in R3_DRIVE.glob("step_*.pt"):
                stray.unlink()          # policy: latest+previous-good+best only
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] error: {exc}", flush=True)
        if stop_evt.wait(PERSIST_POLL_S) or flush_evt.is_set():
            break


def stage8_train(amp_ok: bool) -> int:
    ckpt = REPO / "experiments" / "colab-r3" / "checkpoints" / "last.pt"
    resuming = ckpt.is_file()
    print("\n" + "=" * 66, flush=True)
    print("COLAB R3 VALIDATED -- SAFE TO CUT OVER", flush=True)
    print("=" * 66, flush=True)
    print("[gate note] local CPU r3 on the founder's machine is untouched and "
          "keeps running no matter what happens here.", flush=True)

    cmd = [sys.executable, "experiments/CV-G1-E0013-regressor/train_regressor.py",
           "--manifests", "librittsr-clean100,teacher-kokoro",
           "--durations", "data/manifests/librittsr-clean100.durations.jsonl",
           "--steps", str(TOTAL_STEPS), "--exp-id", "colab-r3",
           "--mel-cache-dir", "data/cache/mel-v1",
           "--threads", str(os.cpu_count()), "--device", "cuda",
           "--num-workers", "2", "--ckpt-every", str(CKPT_EVERY)]
    if amp_ok:
        cmd.append("--amp")
    if resuming:
        print(f"[resume] {ckpt} exists -> launcher auto-resumes exactly "
              "(E0003)", flush=True)

    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(REPO / "colab-r3-train.log", "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=log_fh, stderr=subprocess.STDOUT)
    started = time.time()
    stop_evt, flush_evt = threading.Event(), threading.Event()
    threading.Thread(target=_reporter_loop, args=(proc, started, stop_evt),
                     daemon=True).start()
    threading.Thread(target=_persist_loop, args=(stop_evt, flush_evt),
                     daemon=True).start()

    rc = None
    try:
        while rc is None:
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        print("\n[interrupt] SIGINT -> trainer finalize-saves checkpoint...",
              flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            rc = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        flush_evt.set()
        stop_evt.set()
        time.sleep(PERSIST_POLL_S + 5)  # let the watcher push the final state
        print("[interrupt] checkpoints flushed to Drive "
              f"{R3_DRIVE}; rerun this bootstrap to resume.", flush=True)
        return 130
    stop_evt.set()
    log_fh.close()
    if rc == 0:
        flush_evt.set()
        time.sleep(PERSIST_POLL_S + 5)
        print(f"[done] training complete; final artifacts on Drive at {R3_DRIVE}",
              flush=True)
    else:
        print(f"[failed] trainer exited rc={rc}; see colab-r3-train.log; "
              "rerun the bootstrap to auto-resume from last.pt", flush=True)
    return rc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    STATE.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap] start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
          flush=True)

    ok_cuda, gpu = stage0_env()
    if not ok_cuda:
        _fail()
        return 2
    if not stage1_transfer():
        _fail()
        return 3
    if not stage2_deps():
        _fail()
        return 4
    if not stage3_corpus():
        _fail()
        return 5
    if not stage4_preprocess():
        _fail()
        return 6

    # gates below are skipped ONLY when a validated run already left checkpoints
    force = os.environ.get("CAEDRAL_FORCE_GATES") == "1"
    ckpt = REPO / "experiments" / "colab-r3" / "checkpoints" / "last.pt"
    if ckpt.is_file() and not force:
        print("\n[gates] last.pt exists -> skipping sanity/benchmark gates "
              "(they passed on the run that wrote it); training will RESUME. "
              "Set CAEDRAL_FORCE_GATES=1 to rerun gates anyway.", flush=True)
        amp_dec = STATE / "amp_decision"
        amp_ok = bool(amp_dec.is_file()
                      and json.loads(amp_dec.read_text())["amp_ok"])
        return stage8_train(amp_ok)

    print("\n[gates] building dataset/loaders for smoke+AMP+benchmark...",
          flush=True)
    ds, loader, model, vocab_n, model_cfg = _build_dataset_and_loaders()
    model = model.cuda()
    usable = len(ds)
    record("dataset_built", usable > 0,
           f"usable utterances={usable} vocab={vocab_n} "
           f"(local reference: 30917 usable, vocab 43)")

    if not stage5_smoke(model):
        _fail()
        return 7
    model.zero_grad(set_to_none=True)

    def fresh_model():
        # twin gate models MUST carry the corpus-expanded speaker table
        # (same n_speakers as the real run) or speaker-id lookups would index
        # out of range on CUDA
        import dataclasses  # noqa: PLC0415
        from caedral_voice.model.regressor import (  # noqa: PLC0415
            CaedralRegressor, CaedralRegressorConfig)
        cfg2 = CaedralRegressorConfig(
            **{f.name: getattr(model_cfg, f.name)
               for f in dataclasses.fields(model_cfg)})
        return CaedralRegressor(cfg2)

    if not stage6_amp_sanity(loader, fresh_model):
        _fail()
        return 8
    bench = stage7_benchmark(loader, model)

    print("\n=== GATE SUMMARY " + "=" * 48, flush=True)
    all_ok = True
    for name, ok, detail in CHECKS:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not all_ok:
        _fail(bench)
        return 9
    amp_ok = json.loads(marker("amp_decision").read_text())["amp_ok"]
    print(f"\n[decision] AMP for the real run: {amp_ok}", flush=True)
    return stage8_train(amp_ok)


def _fail(bench: dict | None = None) -> None:
    print("\n" + "!" * 66, flush=True)
    print("CUTOVER REFUSED -- FAILURE DIAGNOSIS", flush=True)
    print("!" * 66, flush=True)
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAIL {name}: {detail}", flush=True)
    print("\nRemediation hints:", flush=True)
    print(" - missing/mismatched zips: finish uploading BOTH zips to "
          "MyDrive/caedral/ then rerun", flush=True)
    print(" - md5/sha failures: delete the bad file (/content/... or Drive) "
          "and rerun; downloads are resumable", flush=True)
    print(" - preprocessing drift: confirm requirements-colab.txt pins were "
          "installed (stage 2) and rerun with CAEDRAL_FORCE_GATES=1", flush=True)
    print("\nIMPORTANT: nothing here touched the local CPU r3 run on the "
          "founder's machine -- it keeps running regardless.", flush=True)
    if bench:
        print(f"benchmark context: {json.dumps(bench)}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[bootstrap] interrupted before/during launch; nothing "
              "outside /content and MyDrive was modified.", flush=True)
        raise SystemExit(130)
