# ARCH-PORT-01 — Windows-to-Arch Portability Audit

Experiment: CV-G1-E0019-archport · 2026-08-23 · local light audit (read-only sweep + minimal fixes)

Scope audited: `caedral_voice/**`, `data/scripts/**`,
`experiments/CV-G1-E0013-regressor/train_regressor.py`, `benchmarks/harness/*`.
`research/firecrawl_client.py` excluded per mission (allowed to stay
Windows-specific; not on any training-core import path — verified).

Search patterns applied to every in-scope file: drive letters (`X:\`),
backslash-only path joins / backslash string literals, `os.system` /
`subprocess` with `shell=True`, PowerShell/`cmd.exe`/`Start-Process`
invocation, `winreg`, `ctypes.windll`, `msvcrt`, `os.startfile`, `sys.platform`
branches, `platform.system()` branches, psutil Windows-only fields,
case-mismatched imports, hardcoded `data\` literals vs pathlib/os.path.

## Verdict

**Training core (caedral_voice/**, train_regressor.py): CLEAN — 0 blockers.**
Every path is built with `pathlib.Path`; every open/write passes explicit
`encoding="utf-8"`; atomic writes use `os.replace` + `os.fsync` (POSIX-safe);
the one OS-dependent literal (`trainer.py`) uses `os.sep`. All intra-repo
imports are lowercase and match filenames exactly (case-sensitive-FS safe).
Manifests (`data/manifests/*.jsonl`) store repo-relative **forward-slash**
audio paths and the `data/processed/...` tree is all-lowercase — transfers to
a case-sensitive filesystem unchanged.

The bring-up risk is **environmental**, not code: venv layout
(`.venv/Scripts` vs `.venv/bin`), torch wheel flavor (CPU vs CUDA), and system
packages (libsndfile for soundfile). Handled by
`scripts/arch/bootstrap-dev.sh` + `requirements-arch.txt`.

## Findings

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| F1 | benchmarks/harness/measure.py:62-104 | `win_process_peaks()` uses `ctypes.windll`/`ctypes.wintypes` (Win32 `GetProcessMemoryInfo`). Import of `ctypes.wintypes` fails on Linux but sits inside the function's try/except → returns None; `native_peak_rss_bytes`/`native_post_rss_bytes` become `null` in Linux benchmark JSONs. Never crashes. | note | none needed (graceful degradation by design); optionally rename fields or add `sys.platform` guard upstream if JSON-schema consumers require stable keys |
| F2 | experiments/CV-G1-E0018-melcache/warm_cache.py:45-56 | `ctypes.windll.kernel32.SetPriorityClass` for deprioritization. Already dual-platform: `sys.platform != "win32"` branch takes `os.nice(10)` (line 38-44). Outside strict audit scope dirs; listed because r3 bring-up may run the warmer. | note | none (already portable) |
| F3 | experiments/CV-G1-E0013-regressor/train_regressor.py:6 (docstring) | Documents launch as `.venv/Scripts/python.exe` — Windows venv layout only. Code itself is cwd-independent and portable; Linux equivalent is `.venv/bin/python`. Same pattern in LAUNCH_READY. | note | documented in RESUME-R3-LINUX.md / bootstrap-dev.sh instead of editing (worker has this file loaded) |
| F4 | experiments/r1/run_manifest.json:9, experiments/r3/run_manifest.json:9 | `durations_source` embeds an absolute Windows path (`C:\Users\...`). Cosmetic log field written fresh at each launch; on Linux it will record a POSIX path automatically. No code reads it back. | note | none |
| F10 | data/cache/mel-v1 (E0018 cache, 6.1 GB) | Verified portable for transfer: entry keys are sha256(audio bytes + config fingerprint), filenames are hex stems, sidecars are UTF-8 JSON, npy is platform-neutral binary. `scan_stale()` on a copied cache reports 0 stale. The live r3 run uses this cache (`--mel-cache-dir data/cache/mel-v1`), so RESUME-R3-LINUX.md transfers it to skip the ~25-35 min warm. | pass | — |
| F5 | caedral_voice/training/mel_dataset.py:174,251; mel_cache.py:179; regressor_dataset.py:253 | `import torchaudio` is lazy (function-local). Portable, but easy to miss when freezing deps → torchaudio MUST be in requirements-arch.txt or first training step raises ModuleNotFoundError. | warn | added torchaudio==2.6.0 to requirements-arch.txt |
| F6 | environment (not code) | `pip install torch` on Linux defaults to the CUDA build (+~2 GB nvidia-* wheels). This box trains on torch 2.6.0+**cpu**; an unintended CUDA build changes numerics/behavior. | warn | bootstrap-dev.sh pins the CPU wheel index by default; GPU build is an explicit opt-in flag (founder decision) |
| F7 | data/scripts/pipeline_lib.py:29-34 | `soxr` imported under try/except with scipy.signal fallback — portable, but Arch needs `libsndfile` (system package) for `soundfile`, and scipy/pyloudnorm are hard deps of this module. Not needed for resume; needed only if re-preprocessing corpora on Arch. | note | optional section in requirements-arch.txt (commented); libsndfile in bootstrap pacman list is justified by this module + tests that write wavs |
| F8 | data/scripts/download_ljspeech.py:89 | `import requests` (lazy) — third-party dep used only by dataset download. | note | listed commented-out in requirements-arch.txt optional section |
| F9 | whole scope | Case-sensitivity trap scan: all intra-repo imports lowercase, exact filename match (`mel_cache`, `mel_dataset`, `regressor_dataset`, `trainer`, `seed`, model modules). No `data\` literals anywhere in scope; no drive letters outside research/firecrawl_client.py (excluded). | pass | — |

## Training-core import verification (deps evidence)

Grepped actual imports of everything reachable from
`train_regressor.py` + the three gate test files:

| Package | Imported by | Venv version (pin basis) |
|---|---|---|
| torch | trainer, seed, mel_dataset, regressor_dataset, model/*, train_regressor, tests | 2.6.0 (+cpu) |
| torchaudio | mel_dataset._load_mel (lazy), regressor_dataset._load_mel (lazy), mel_cache.compute (lazy) | 2.6.0 |
| numpy | seed, mel_cache, tests/test_mel_cache | 2.5.2 |
| pytest | three gate test files only | 9.1.1 |

NOT imported by the training core or gate tests (kept out of the install set):
psutil (only benchmarks/harness + inference/cli), soundfile/scipy/pyloudnorm/
soxr (only data/scripts pipeline), requests (download script),
g2p_en/nltk (**inference and phonemization only** — training consumes
frozen phoneme strings from `*.phonemes.jsonl`; verified: nothing under
`caedral_voice/training/` or `train_regressor.py` reaches `caedral_voice.text`).
