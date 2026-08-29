# CAEDRAL VOICE — PILOT Acquisition & Merge Plan (DATA-PILOT-01)

Owner: DATA agent. Written: **2026-08-22**. Status: **plan + verification only — NO downloads performed by this task.**
Basis: `docs/data/DATASET-SURVEY.md` + `docs/data/LICENSE-MATRIX.md` (license verdicts verified there 2026-08-22; cited here as [S#], not re-researched). Voice targets per `configs/voices/README.md` (G1 slate).
Web evidence: every URL used is registered in `docs/research/WEB-SOURCE-INDEX.md`; cached copies under `research/cache/`.

## 0. Component summary

| # | Component | Subset for PILOT | Download size | Status |
|---|---|---|---|---|
| C1 | LibriTTS-R (SLR141) [S3] | train-clean-100 **+ dev-clean/dev-other/test-clean/test-other** + doc + failed-restoration list | 8.1 G (have) + ~4.5 G (to get) | archive verified vs official MD5 |
| C2 | LJSpeech v1.1 [S1] | full corpus (13,100 clips / 23:55:17) | 2.56 GiB (have) | sha256 VERIFIED match |
| C3 | Common Voice EN 26.0 [S7][S8] | validated-only deterministic slice, target ≈20 h | 0.5–0.7 GiB extracted — but requires gated full-bundle pass (see §3.4) | method defined; execution DEFERRED (disk gate) |
| C4 | LibriTTS-P annotations [S29] | repo `data/` CSVs (style/identity prompts overlay) | <0.1 GB | to fetch |

Total new downloads to finish the pilot set (excluding deferred C3): **≈4.6 GB**. Archives on disk after that: **≈14.6 GB** — consistent with the survey's "~16 GB PILOT".

---

## 1. Verified local state (measured this session, 2026-08-22)

Free disk **C:** = **90.25 GiB** (96,900,816,896 B free; used 414,084,833,280 B). All budget math in §4 uses this number.

| Local file | Bytes | SHA256 (computed locally) | Official checksum | Result |
|---|---|---|---|---|
| `data/downloads/train_clean_100.tar.gz` | 8,129,817,509 (7.57 GiB) | `6f0cef20fb1e72928f21f00711d8cda7acb2049fd1cef78c98e0c2db77de2547` | OpenSLR publishes **MD5 only** (`md5sum.txt`, SLR141): `6df668d8f5f33e70876bfa33862ad02b` | local MD5 == official MD5 → **authentic** |
| `data/downloads/LJSpeech-1.1.tar.bz2` | 2,748,572,632 (2.56 GiB) | `be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5` | sha256 published via torchaudio LJSPEECH config / HF `keithito/lj_speech` download_checksums (recorded in `data/downloads/LJSpeech-1.1.download.json`) | **exact match → authentic** |

**Checksum policy note (important):** OpenSLR does *not* publish sha256 sums for SLR141 files — the official page ships a single `md5sum.txt` (509 B). "sha256 where published" therefore resolves to: LJSpeech → official sha256 (above); LibriTTS-R → official MD5 + our locally computed sha256 recorded as the content-addressed manifest ID (per survey §Deterministic manifest design); CV EN 26.0 → official sha256 published in cv-dataset release JSON (§3.3).

Existing SMOKE artifacts (`data/raw/librittsr-slice/`, `data/raw/ljspeech-slice/`, `data/processed/librittsr-smoke-10spk/`) are separate from PILOT and untouched by this plan.

---

## 2. C1 — LibriTTS-R (SLR141), CC BY 4.0 [S3]

24 kHz Miipher-restored multi-speaker backbone of the pilot. Page facts verified today from `https://www.openslr.org/141`.

### 2.1 URLs, mirrors, sizes, checksums

Mirror bases (identical layout `<base>/resources/141/<file>`):
- US primary: `https://openslr.trmal.net/resources/141/`
- EU: `https://openslr.elda.org/resources/141/`
- CN: `https://openslr.magicdatatech.com/resources/141/`
- (`https://www.openslr.org/resources/141/<file>` resolves to a near mirror.)

| File | Size (page label) | MD5 (official `md5sum.txt`) | In pilot? |
|---|---|---|---|
| `train_clean_100.tar.gz` | 8.1G | `6df668d8f5f33e70876bfa33862ad02b` | HAVE (verified §1) |
| `dev_clean.tar.gz` | 1.3G | `2c1f5312914890634cc2d15783032ff3` | download |
| `dev_other.tar.gz` | 975M | `62d3a80ad8a282b6f31b3904f0507e4f` | download |
| `test_clean.tar.gz` | 1.2G | `4d373d453eb96c0691e598061bbafab7` | download |
| `test_other.tar.gz` | 1.0G | `dbc0959d8bdb6d52200595cabc9995ae` | download |
| `doc.tar.gz` | 318K | `dad90f1b357625ce93d695b930977f7f` | download |
| `libritts_r_failed_speech_restoration_examples.tar.gz` | 106K | `43b616c18f317893cdf113af4d968de8` | download (QC metadata: utterances where restoration failed) |

Not in pilot: `train_clean_360.tar.gz` (28G), `train_other_500.tar.gz` (46G) — Phase B scale-up.
Procedure per file: download → verify MD5 against table → record byte size + locally computed sha256 into the acquisition ledger before extraction.

### 2.2 Layout after extraction
`data/raw/libritts_r/{train-clean-100,dev-clean,dev-other,test-clean,test-other}/<spk>/<utt>/{<utt>.wav,<utt>.trans.txt,<utt>.normalized.txt}` + `data/raw/libritts_r/doc/`. License/attribution: CC BY 4.0 — cite Koizumi et al. 2023 (LibriTTS-R) and Zen et al. 2019 (LibriTTS); see attribution block in §6.2.

## 3. C2/C3/C4

### 3.2 LJSpeech v1.1 — public domain [S1]
Official host is keithito.com only: `https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2` (page label 2.6 GB; exact bytes + sha256 in §1; v1.1 current). **No byte-identical mirror exists**: OpenSLR SLR13 is *not* LJSpeech (it is the RWCP Sound Scene Database — checked today, common misconception); Hugging Face `keithito/lj_speech` is a repacked conversion, not the archive. Any re-download must come from `data.keithito.com` and must reproduce sha256 `be1a3045…e8aa5`. Audio: 22.05 kHz 16-bit mono WAV; transcripts.csv (ID|raw|normalized). PD — no attribution required (cite Ito & Johnson 2017 as courtesy).

### 3.3 Common Voice EN Scripted Speech 26.0 — CC0 + no-speaker-identification condition [S7][S8]

Verified facts (cv-dataset repo, release `cv-corpus-26.0-2026-06-12`):
- Bundle (whole locale, monolithic): **94,639,372,950 B ≈ 88.14 GiB**, published sha256 `6809228e6ab506d18f6a1ebc830056450f8266c8f513d6038bdb0fc88a49e6cb`.
- EN stats: validated 1,902,564 clips / **2,784.88 h** (validDurationSecs 10,025,595.594), avg clip 5.27 s, 100,172 users; total incl. non-validated 3,780.95 h.
- Archive layout: `cv-corpus-26.0-2026-06-12/en/clips/*.mp3` + `validated.tsv` (+ train/dev/test/invalidated/other/reported TSVs, `clip_durations.tsv`). Fields include `client_id`, `path`, `sentence_id`, `sentence`, `accents`.
- Distribution reality (all checked today): downloads are **Mozilla Data Collective (MDC)-exclusive since Oct 2025** (dataset page/datasheet `mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu` [S7]; API key + per-dataset terms acceptance required; SDK `pip install datacollective`, env `MDC_API_KEY`, `download_dataset()` is **whole-bundle only** — no partial/file selection, resumable chunks). The old Hugging Face mirrors are **emptied stubs** (`mozilla-foundation/common_voice_17_0` tree = `.gitattributes` + README only). The v26.0 EN **delta bundle contains zero validated clips** (8,647 clips, all bucket `other`, 15.19 h) — useless for TTS.

**Consequence:** true server-side "validated shards" do not exist upstream. Selection must therefore be client-side deterministic, executed inside a one-shot streaming pass (§3.4). This is the documented deviation from the survey's "shard selection" wording: same outcome (~20 h validated slice), different mechanism, because the upstream no longer offers shards of any kind.

#### Deterministic slice definition (freeze now, execute later)

```
SEED        = "caedral-cv-en26-pilot-v1"            # bump version to change slices; never reuse
TARGET      = 72,000 s (≈20 h) kept audio
THRESHOLD   = 132477473368460272                    # floor(2^64 · 72000/10025595.594); yields exactly 20.0 h expected
RULE        : rows = validated.tsv sorted lexicographically by path
              keep row iff int(sha256(f"{SEED}|{path}").hexdigest()[:16], 16) < THRESHOLD
SPEAKER CAP : group kept rows by client_id; if >40 kept clips for one client_id,
              retain the 40 with smallest hash value (prevents power-contributor domination)
DURATION    : join clip_durations.tsv for exact seconds; report achieved total at freeze
```

Expected yield ≈18–21 h across ≈hundreds of speakers; accent mix reported at freeze from the `accents` field (US/England/India/Africa/Scotland etc.) — no re-weighting in pilot.

### 3.4 CV execution plan (DEFERRED behind explicit gates — do not run casually)
Gates: (a) free disk ≥ **95 GiB** on target volume (or an external drive ≥100 GiB) — current 90.25 GiB fails this; (b) human steps done once: MDC account, accept CV terms, create API key stored ONLY in user env var `MDC_API_KEY` (never committed/logged — same policy as FIRECRAWL key).
Pass: `download_dataset(<dataset-id>)` → resume-safe stream to scratch → verify sha256 == `6809228e…49e6cb` and byte size → extract **only** paths passing the RULE above plus all TSV metadata files (single sequential tar pass; gzip has no random access) → delete bundle immediately. Peak disk = 88.14 + ~0.7 GiB; est. transfer 4–10 h at 25–50 Mbps + ~30 min extraction. Compliance while using: CC0; condition = do not attempt to identify speakers → `client_id` stays internal to manifests, never published, never used for identity anchors (matrix row 5).

### 3.5 LibriTTS-P annotation layer — CC BY 4.0 [S29]
Repo `github.com/line/LibriTTS-P`, directory `data/`:
`df1_en.csv`, `df2_en.csv`, `df3_en.csv` (annotator speaker prompts) · `excluded_spk_list.txt` (spk_ids with conflicting gender — EXCLUDE from voice candidates) · `unannotated_spk_list.txt` (excluded by restoration-failure filter) · `style_prompt_candidates_v230922.csv` · `metadata_w_style_prompt_tags_v230922.csv` (per-utterance style tags + measured F0/rate/loudness stats; `invalid=1` rows dropped).
Fetch: `git clone --depth 1 https://github.com/line/LibriTTS-P` (or raw.githubusercontent per file). No upstream checksums → record local sha256 per file at fetch. Size well under 0.1 GB. Joins to LibriTTS-R audio by `item_name`/`spk_id`.

---

## 4. Disk budget (free now: 90.25 GiB)

| Phase | Adds (GiB) | Running total freed-check |
|---|---|---|
| On disk now (2 archives, verified) | 10.13 | 90.25 free |
| + LibriTTS-R dev/test/doc/failed-list downloads | +4.55 | archives 14.68 |
| + LibriTTS-P data | +0.01 | 14.69 |
| Extract C1+C2 sequentially, **delete nothing yet** | +20.5 (LJR ≈11 + dev/test ≈6 + LJ ≈3.6) | ≈35.2 used by data |
| Processed store: 24 kHz normalized masters (WAV) + manifests for ≈105 h | ≤+18 | steady-state ≈53 |
| Optional: FLAC processed masters instead | ≈+9 | steady-state ≈44 |

**Peak during processing ≈ 53–58 GiB → leaves ≥32 GiB margin against 90.25. Safe.**
C3 (CV pass) needs its own window with ≥95 GiB free or external volume — schedule after pilot training starts, or free ~5 GiB by deleting the two largest *extracted* trees once processed masters exist. Never let the CV scratch share the volume with everything else extracted simultaneously.

## 5. Speaker-balance targets — 4 provisional catalog voices (`configs/voices/README.md`)

All four anchor on distinct LibriTTS-R train-clean-100 speakers (identity class COND→LEAN-YES per matrix; counsel sign-off before any named-identity launch, per survey open item 4). Targets apply to **kept** audio in frozen manifests:

| Codename | Presentation / character | Primary speaker target | Constraints |
|---|---|---|---|
| codename-f01 | F, warm narration | ≥60 min kept (floor 45, hard min 30 = Phase J criterion) | mid F0 band; speaking_rate normal; reject rate <15% |
| codename-f02 | F, energetic/conversational | ≥60 min kept (same floors) | F0 mean separated ≥3 st from f01; higher `speaking_rate`/`energy` tags in LibriTTS-P |
| codename-m01 | M, professional/serious | ≥60 min kept (same floors) | low-mid F0; stable loudness (low LUFS σ) |
| codename-m02 | M, calm documentary | ≥60 min kept (same floors) | F0 separated ≥3 st from m01; slow `speaking_rate` tag |

Selection procedure (Phase J, deterministic shortlist then human listen):
1. Candidate pool = clean-100 speakers minus `excluded_spk_list.txt` ∪ `unannotated_spk_list.txt`, with ≥45 min raw audio and ≥85% valid rows in `metadata_w_style_prompt_tags_v230922.csv`.
2. Rank per voice by style-tag fit (table above) + lowest clip-reject projection; enforce pairwise F0-mean separation ≥3 st (use LibriTTS-P `raw_f0_mean`) and accent breadth across the four (mix US/UK/non-US readers where pool allows).
3. Human listening gate at Phase K/N decides final IDs; auto-metrics cannot close identity quality (EVALUATION-BENCHMARKS finding).
Corpus-level balance: overall kept multi-speaker mix within 55/45 F/M either way; LJSpeech (PD, female) is a fifth, non-catalog reference voice for pipeline validation only — it does not count toward the four.

## 6. Deterministic slices & merge plan

### 6.1 Split rules (regenerate byte-identical, never re-shuffle)
- ID = sha256 of content bytes (survey manifest spec). Splits derive from dataset-official partitions first, hash ranges second:
- **LibriTTS-R:** outer split = official subset (train-clean-100 / dev-* / test-*). Inside train-clean-100 carve a 2% internal holdout: `int(sha256(id).hex[:8],16) % 50 == 0` → `pilot-eval` pool (~25 min) excluded from training; official dev/test reserved for reporting only.
- **LJSpeech:** r = `int(sha256(id).hex[:8],16) % 1000`; test r<10, dev 10≤r<20, else train (≈131/131 clips).
- **CV:** slice rule §3.3 IS the selection; within it, same %1000 rule for dev/test carve-outs.
- Cross-split disjointness enforced on normalized text + speaker (survey step 10). Frozen ID lists committed to `data/manifests/` at freeze; CI regenerates and diffs.

### 6.2 Manifest & license ledger
JSONL fields exactly per survey §Deterministic manifest design (`id, sha256, relpath, dataset, dataset_version, license_id, source_url, speaker, duration_ms, sr, lufs_int, tp_dbtp, snr_est, wer_est, text_raw, text_norm, lang, split`). Ledger rows: LibriTTS-R → CC BY 4.0 [S3]; LJSpeech → PD [S1]; CV EN 26.0 → CC0 + no-identification condition [S7][S8]; LibriTTS-P → CC BY 4.0 [S29].
Attribution block shipped with any model touching CC BY 4.0 data: LibriTTS-R © Google/OpenSLR contributors, CC BY 4.0, Koizumi et al. 2023; derived from LibriTTS, Zen et al. 2019; LibriTTS-P Kawamura et al. 2024, CC BY 4.0. Merge order: LJR → LJ → (later) CV; dedup on normalized-text hash + MinHash near-dup before freeze.

## 7. Post-processing hour estimate (survey pipeline steps 1–11; CPU 8-core, parallel per-file)

| Stage | Volume | Assumption | Wall hours |
|---|---|---|---|
| Untar + checksum ledger | 13 GB archives | disk-bound | 0.5 |
| Resample→24 kHz, 2-pass loudnorm (−23 LUFS), DC/HPF | ~105 h audio | ~15× RT/core ×8 cores | 1–2 |
| Silero VAD segmentation 3–12 s + trim | ~105 h | ~100× RT/core | 0.5 |
| Quality screens: clipping, SNR≥32 dB, optional DNSMOS | ~105 h | DNSMOS ~3× RT/core | 4–6 |
| MFA forced alignment | ~105 h | ~15× RT/core + model setup | 2–4 |
| ASR WER screen (reject >20–25%) | ~105 h | faster-whisper int8 ~6× RT/core | 3–6 |
| Dedup + split lists + manifest freeze | — | trivial | 0.2 |
| **Total** | | | **≈12–19 h** (budget 24 h) |

CV pass adds ≈3–4 h when executed. Single-machine figures; re-estimate if core count differs.

## 8. Execution checklist (next sessions)
1. Download 6 small LibriTTS-R packages (§2.1 order: doc → failed-list → dev_clean → test_clean → dev_other → test_other); MD5-verify each; record bytes+sha256.
2. Fetch LibriTTS-P data dir; record hashes; build exclusion/style tables.
3. Extract C1 subsets sequentially; extract LJSpeech (its earlier extract attempt left empty logs — redo and verify wavs count = 13,100).
4. Run processing pipeline §7; freeze manifests + split lists (§6).
5. Voice candidate shortlist per §5 → Phase J/K gates.
6. Schedule gated CV pass (§3.4) once disk gate passes; integrate slice; re-freeze.

## 9. Sources
Survey/matrix citations reused: [S1] keithito.com/LJ-Speech-Dataset · [S3] openslr.org/141 · [S7] MDC CV 26.0 EN datasheet · [S8] github.com/common-voice/cv-dataset · [S29] line/LibriTTS-P. New primary fetches this session (cached + indexed in `docs/research/WEB-SOURCE-INDEX.md`): SLR141 `md5sum.txt`; openslr.org/13 (negative check: not LJSpeech); cv-dataset scripted-speech README + `cv-corpus-26.0-2026-06-12.json` + delta JSON; MDC api-reference + datacollective-python README/docs; HF API listing mozilla-foundation datasets + emptied `common_voice_17_0` tree. Local measurements: Get-FileHash over both archives; PSDrive free-space query.
