# DATA-FULL-01 — Full LibriTTS-R train-clean-100 extraction + frozen manifest

Date: **2026-08-22**. Status: **COMPLETE**. Evidence: [MEASURED] throughout
(script logs under `logs/`, counts re-derivable from the frozen manifests).

## Headline

Training corpus scaled **1.66 h (10-spk smoke) → 44.11 h kept / 30,917 utts /
247 speakers** (`librittsr-clean100`), phonemized and integrity-validated.
Raw yield: 33,232 wavs, 53.55 h, all 247 train-clean-100 speakers.

## Stage results

| Stage | Result | Wall |
|---|---|---|
| Extract (`data/scripts/extract_librittsr_full.py`) | 247/247 speakers `.complete`; 33,232 wav + 33,233 txt written; 0 resume-skips (fresh run); archive streamed once | 316 s |
| Preprocess (`preprocess.py --name librittsr-clean100`) | seen=33,232 kept=**30,917** rejected=2,315 | ~30 min |
| Phonemize (`phonemize_manifest.py`) | 30,917 rows @ 555 utt/s; sources: lexicon 481,389, g2pen 6,028, morphology 1,125, rule_fallback 1,535 — g2pen share **1.23%** | 56 s |
| Validate (`validate_manifest.py`) | **OK** — sha256/duration/count/integrity checks pass on every kept row | ~2 min |

Rejections (2,315 total): TOO_LONG 1,705 · DUPLICATE_TRANSCRIPT 588 ·
TOO_QUIET 9 · TOO_SHORT 13. Kept hours = 44.11 h vs raw 53.55 h (**82.4%**
yield: silence trim + >15 s clip removal + exact-duplicate text dedup).
Archive has 33,233 transcripts but 33,232 wavs (one upstream transcript-only
entry; harmless — preprocess iterates wavs).

## Artifacts

- Manifests: `data/manifests/librittsr-clean100{.jsonl,.rejected.jsonl,.meta.json,.phonemes.jsonl}`
- Processed audio: `data/processed/librittsr-clean100/` (22.05 kHz PCM_16, −23 LUFS)
- Raw tree (resumable, per-speaker `.complete` markers): `data/raw/librittsr-full/`
- Speaker stats + 15-voice shortlist: `speaker_stats.json`
- Provenance stamp in meta: archive sha256 `6f0cef…2547`, provenance
  VERIFIED-official-md5 (OpenSLR SLR141), license CC-BY-4.0, source
  https://www.openslr.org/141/

## Disk

Free before 96.3 GB → after 80.1 GB: **+16.2 GB total** (raw tree 9.4 GB,
processed+manifests 6.9 GB). Well inside budget (processed ≈10–11 GB est.).

## Speaker stats / 15-voice candidates [MEASURED]

247 speakers; kept hours per speaker 0.02–0.47 h. **29 speakers ≥ 20 min.**
Selection rule (deterministic): qualifying speakers sorted alphabetically,
15 taken evenly spaced across the range (`idx=round(i·(N−1)/14)`).

Candidates (kept hours / utts):
1088 (.448/346) · 1263 (.343/214) · 2416 (.365/271) · 3879 (.385/250) ·
412 (.374/215) · 460 (.339/227) · 5322 (.352/246) · 6147 (.389/345) ·
6415 (.371/275) · 7067 (.370/285) · 7190 (.353/318) · 7505 (.358/168) ·
7800 (.343/260) · 8123 (.380/237) · 8419 (.334/241)

Per-speaker kept-hours distribution: min 0.0003, median 0.168, max 0.469.

**Implication for Phase J voice targets:** max single-speaker kept audio in
train-clean-100 is **~28 min (spk 6209)** — the DATA-PILOT-ACQUISITION §5
"≥60 min kept per catalog voice" target is **unreachable from clean-100
alone**; voice plans must pool multiple speakers per codename or draw on
clean-360 later. Diversity refinement of this shortlist happens elsewhere.

## Incident log (for future runs)

1. Two aborted launches: PowerShell `Start-Process -ArgumentList` does not
   quote elements containing spaces → argparse `unrecognized arguments`.
   Fix: detached background bash with explicit quoting.
2. One full preprocess pass ran with **all 33,232 utts rejected as
   MISSING_TRANSCRIPT**: extractor wrote `<utt>.normalized.txt` unchanged
   because `Path.suffix` returns only the last suffix (`.txt`), making the
   intended rename a no-op. Fixed extractor to match on full name; renamed
   33,233 files on disk; deleted stale manifests; clean rerun. The fixed
   script now writes `<utt>.txt` directly.

Total mission wall time ≈ 75 min including the false-start rerun (~58 min net
stages: extract 5.3 + preprocess 30 + phonemize 1 + validate 2 + scans/docs).
Nothing committed (per task instruction).
