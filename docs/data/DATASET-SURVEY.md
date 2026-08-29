# CAEDRAL VOICE — English Speech Dataset Survey (commercial TTS)

Maintained by: RESEARCHER E. Verification date: **2026-08-22** (all license verdicts verified from official pages/cards this date; see Sources and companion `LICENSE-MATRIX.md`, which shares the [S#] numbering).
Scope: proprietary, CPU-first, multi-voice **English** TTS intended for commercial release. Research only — nothing downloaded.

## How to read verdicts

- **Train OK**: usable to train models we sell (license-level question).
- **Voice-ID OK**: usable to define a released VOICE IDENTITY (stricter: needs documented speaker consent/publicity-rights position). License permission is NOT consent; US state right-of-publicity laws (e.g., Tennessee ELVIS Act 2024) and EU AI Act Art. 50 transparency duties apply regardless of dataset license.
- **Provenance risk** (for voice-identity use): LOW / MEDIUM / HIGH / EXTREME.
- `[REPORTED]` = secondary-source figure. `[UNVERIFIED]` = official page not confirmed.

## Tier A — Commercial-clean AND plausible voice-identity anchors

| Dataset | License | Hours | Speakers | Gender/Accents | Audio | Transcription | Style | Provenance risk |
|---|---|---|---|---|---|---|---|---|
| LJSpeech v1.1 [S1] | Public domain | 23.92 h (13,100 clips, 1.1–10.1 s) | 1 (Linda Johnson, LibriVox reader) | F / US | 22.05 kHz 16-bit WAV mono; source was 128 kbps MP3 (codec artifacts) | Manual match + QA pass; raw+normalized text | Plain read only | LOW |
| Hi-Fi-TTS (SLR109) [S22] | CC BY 4.0 | 291.6 h | 10 native EN (6F/4M), 17.7–58 h each | mixed / mostly US+UK readers | 44.1 kHz WAV; SNR>=32 dB (clean >=40 dB), >=13 kHz bandwidth, zero-WER ASR screen | ASR-verified vs Gutenberg texts | Plain read | LOW-MED (named LibriVox readers, PD lineage) |
| LibriTTS-R (SLR141) [S3] | CC BY 4.0 | ~585 h | 2,456 | disclosed via LibriTTS IDs / global accents | 24 kHz, Miipher-restored quality | Original + normalized text; sentence-split | Read, some expressiveness in "other" | MED at scale (PD LibriVox lineage; per-reader consent undocumented) |
| CMU Arctic [S33] | Permissive CMU notice ("any purpose (commercial or otherwise)") | ~1,150 utts x 18 single-speaker sets (~20–35 min/speaker) | 18 incl. pro talent bdl(M)/slt(F)/clb(F) | US + Scottish/Indian/Canadian/German/Israeli etc. | 16 kHz WAV (+EGG) | Gutenberg out-of-copyright scripts | Neutral read | LOW (explicit broad grant, 2003); dated audio |

Identity-anchor shortlist: LJSpeech; Hi-Fi-TTS clean-subset readers 92/6097/9017; selected LibriTTS-R speakers; CMU Arctic bdl/slt/clb as legacy voices. Counsel sign-off recommended before any named-speaker identity launch.

## Tier B — Commercial-training workhorses (do NOT anchor released identities on them)

| Dataset | License | Hours | Speakers | Notes |
|---|---|---|---|---|
| LibriTTS (SLR60) [S2] | CC BY 4.0 | ~585 h | 2,456 | Superseded by LibriTTS-R (same IDs). Tarballs total ~83 GB. |
| MLS English (SLR94) [S9][S10] | CC BY 4.0 | ~44.5 kh (abstract) | ~5,574 [REPORTED via S31] | Machine-aligned ASR-grade transcripts; 16 kHz [REPORTED]; 2.4 TB FLAC / 651 GB Opus download — subset deliberately. |
| Common Voice EN 26.0 [S7][S8] | CC0 + no-speaker-identification condition | 3,781 h rec / 2,785 h validated; 2.58 M clips | 100,172 | Global accents (major asset). MP3, crowd mic quality, avg clip 5.27 s. Identity anchoring conflicts with ToS condition -> NO. |
| Libriheavy [S28] | Apache-2.0 | ~50 kh (large subset) | thousands [UNVERIFIED count] | Punctuation+casing+context transcripts (rare, valuable for TTS); LibriVox lineage. |
| LibriTTS-P [S29] | CC BY 4.0 | 373,868 utts over LibriTTS-R | 2,443 | Style/identity prompt annotations — pairs with Tier A/B audio for style control. |
| HiFiTTS-2 [S31] | CC BY 4.0 (metadata-only card) | 36.7 kh @22.05 kHz / 31.7 kh @44.1 kHz | 5,013 / 4,631 | You rebuild audio from LibriVox via NVIDIA SDP pipeline; strong PD lineage; <=20 s segments. |

## Tier C — Conditional, gated, or unresolved (legal review before ANY training use)

| Dataset | Status |
|---|---|
| VCTK v0.92 [S4][S5][S6][S42] | Official `license_text.txt` = CC BY 4.0; several catalogs say ODC-By 1.0 (both attribution-only). 110 EN speakers, various accents, hemi-anechoic chamber, 48 kHz release (96 kHz capture), DPA 4035 + Sennheiser MKH 800 mics, 10.94 GB zip, ~400 sentences/speaker. Newspaper prompt texts carry separate Herald & Times permission. Speaker-consent scope for synthesis undocumented -> fine as auxiliary TRAINING diversity, RISKY for identities. Resolve notice discrepancy before shipping. |
| GigaSpeech [S11][S12] | Repo badge Apache-2.0 but raw README has NO license section; access gated by Google Form + emailed instructions with unpublished terms. 33 kh total / 10 kh human-transcribed; audiobook/podcast/YouTube mix; 16 kHz Opus. Third-party underlying rights unresolved -> treat as UNVER until form terms reviewed; NO identities ever. |
| People's Speech [S13][S14] | CC-BY subsets + separate CC-BY-SA (`_sa`) subsets; 30+ kh EN from archive.org (gov proceedings, speeches, oral histories); 16 kHz parquet. Documented v1.0 misalignment; transcripts often pre-existing/ASR; almost all US-accented. Usable per-subset with compliance tracking; NO identities (uploaders' publicity rights). |
| RAON-OpenTTS (KRAFTON, 2026) [S30] | Code Apache-2.0; Pool 615 kh / Core 510 kh @16 kHz aggregated from 11 public corpora — reportedly incl. Emilia (CC-BY-NC!) and YouTube-Commons [REPORTED]. Aggregate license cannot cure upstream NC defects; data cards UNVERIFIED. Avoid pending per-component audit. |
| GLOBE [S32] | 535 h @24 kHz, 23,519 speakers, 164 accents (Common Voice-derived) — excellent accent-diversity candidate, but NO license stated on project page; HF card unchecked. UNVERIFIED — hold. |
| Blizzard Challenge sets [REPORTED] | Historically challenge/research terms; per-release agreements — UNVERIFIED, not pursued. |
| VoxPopuli | EP-footage reuse terms UNVERIFIED this session — excluded. |
| TITW (2025) [S26] | See Tier E. |

## Tier D — Non-commercial / research-only (evaluation & style research ONLY)

| Dataset | License | Contents | Use for us |
|---|---|---|---|
| Expresso [S19][S20][S21] | CC BY-NC 4.0 (confirmed) | 47 h, 4 NA-EN actors (2M/2F), 48 kHz/24-bit, 26 styles (read 7 styles + 25 improvised dialogue styles, singing) | Internal NC evaluation of expressiveness targets/benchmarks; never in shipped weights without Meta negotiation |
| ESD [S23] | Signed research-only license (email zhoukun@u.nus.edu) | 10 EN + 10 ZH speakers x 350 parallel utts x 5 emotions; >29 h studio | Parallel emotional references; internal eval only |
| RAVDESS [S24] | CC BY-NC-SA 4.0; paid commercial license available from licensor | 24 pro actors (12F/12M), 8 emotions x 2 intensities, 48 kHz WAV, 10x-rated validity | Emotion-classifier calibration; commercial path = purchase |
| CREMA-D [S25] | ODbL 1.0 + DbCL 1.0 | 91 actors (48M/43F) x 12 sentences x 6 emotions x 4 levels; 7,442 clips; WAV/MP3/FLV | Share-alike + consent questions -> counsel review even for training |
| JVS/JENTZ-class [REPORTED] | No "JENTZ" corpus found; JVS = Japanese versatile speech (research), JVNV emotional — all Japanese-only | No English versions exist | N/A for English program |

## Tier E — High/EXTREME provenance risk (avoid entirely for our purposes)

| Dataset | Why dangerous |
|---|---|
| Emilia core (101.7 kh; EN 46.8 kh) [S15][S16][S17][S18] | Card states CC BY-NC 4.0 despite sidebar cc-by-4.0 tag; "copyright remains with the original owners"; indemnification demanded from users; access terminable; no consent process. F5-TTS had to downgrade checkpoints to NC because of it. The 2024 "Sound Check" audit found ~35% of audio datasets potentially copyright-infringing — Emilia is the canonical example. |
| Emilia-YODAS portion (113.9 kh; EN 92.2 kh) [S15][S27] | CC BY 4.0 label but YouTube-derived; uploader self-reported CC status; platform-ToS exposure; speaker consent absent. |
| YODAS/YODAS2 [S27] | CC BY 3.0 claimed with good-faith CC-only collection + takedown channel, yet caption quality varies wildly and consent/publicity rights remain unaddressed. |
| TITW [S26] | Stated CC BY 4.0 BUT underlying speech copyright explicitly remains with original video owners (VoxCeleb1 interviews/social media); WhisperX auto-transcripts; author-gated HF access. Train-at-your-own-risk even for research. |
| Any scraped/podcast aggregate (GigaSpeech dev/test crawls, People's Speech dirty subsets, RAON components) | Same failure mode: license label != rights clearance. |

## Consented studio-data purchase path (flagship voices)

- Defined.ai: 80 h en-US professional female actor, scripted+spontaneous, 44.1 kHz WAV; DLA permits commercializing trained models, forbids reselling data, and contains a clause restricting outputs of recognizable contributor voice/content — must be negotiated/read against voice-cloning intent [S34][S35].
- SpeechOcean/DataoceanAI King-Line King-TTS-*: single-speaker 10–23 h studio sets (44.1/48 kHz, phoneme+prosody annotated) and multi-speaker conversational sets (e.g., 135 h / 140 spk); quote-based commercial licensing; review contract per SKU [S36].
- LDC (Switchboard/Fisher): nonmembers are NON-COMMERCIAL ONLY; commercial rights require for-profit membership or corpus-specific licenses [S37][S38]. Poor fit vs vendors.
- Direct-to-talent buyout (Voices.com/Voice123/agencies): cleanest chain — work-for-hire + explicit cloning/model-release clauses; budget guidance: foundation voices typically need 100–300+ finished hours [REPORTED].
- Recommendation: flagship voices = purchased consented recordings (or Tier A anchors after counsel review); open data = pretraining/diversity only.

## Preprocessing best practices (CPU-friendly pipeline order)

1. **Ingest & verify**: checksum every archive (MD5 provided by OpenSLR et al.); record source URL + version + license per file.
2. **Resample**: unify to one working rate (24 kHz for our stack; keep 44.1 kHz masters for Hi-Fi-TTS-class data); soxr-quality resampler.
3. **Loudness normalization**: two-pass loudnorm to EBU R128 target (-23 LUFS integrated, TP <= -1 dBTP) or a consistent program target (-20…-26 LUFS); consistency across datasets matters more than the absolute number. Log measured LUFS per clip.
4. **DC offset removal + high-pass** ~70–80 Hz.
5. **VAD segmentation**: Silero VAD [S41] (LICENSE file confirmed MIT) into 3–12 s chunks; trim leading/trailing silence with 10–50 ms padding.
6. **Clipping/noise rejection**: drop clips with >~0.1% samples at |x|>=0.99 FS; estimate SNR, enforce >=32 dB (Hi-Fi-TTS precedent; >=40 dB for "clean") [S22]; optional DNSMOS-style scoring (Emilia/TITW precedent) [S15][S26].
7. **Forced alignment**: MFA (MIT code, CC-BY-4.0 models) [S39][S40] for word timestamps; flag clips where alignment confidence/deviation outliers.
8. **ASR transcript validation**: re-decode with a strong ASR model; compute WER vs transcript; reject WER >~20–25% (zero-WER screens used by Hi-Fi-TTS) [S22]; catches misaligned/mistranscribed data (People's Speech v1.0 lesson [S14]).
9. **Duration/rate filters**: keep ~0.5–15 s (HiFiTTS-2 caps at 20 s [S31]); chars/sec within ~8–25; drop >3-sigma duration/LUFS/F0 outliers per speaker.
10. **Dedup**: exact hash of normalized text; MinHash/n-gram near-dup detection; enforce cross-split text+speaker disjointness (Libriheavy dev/test design [S28]).
11. **Freeze**: emit manifests + split lists, commit hashes; downstream runs read-only.

## Deterministic manifest design

- ID = `sha256(file bytes)` (content-addressed; survives renames/copies).
- JSONL fields: `{id, sha256, relpath, dataset, dataset_version, license_id, source_url, speaker, duration_ms, sr, lufs_int, tp_dbtp, snr_est, wer_est, text_raw, text_norm, lang, split}`.
- Splits: deterministic function of the content hash (e.g., sha256 hex prefix ranges) OR frozen ID lists committed to git; NEVER random-shuffle after edits; regenerate byte-identical manifests in CI.
- Keep a per-dataset LICENSES ledger mapping dataset -> license -> conditions -> source URL (this document).

## SMOKE (<2 GB) and PILOT (~10–30 GB) recommendations (122 GB free disk respected)

SMOKE — ~1.8 GB downloads, tests 3 sources, 2 licenses, multi-speaker:
| Component | Subset | Est. size |
|---|---|---|
| LJSpeech v1.1 | 4 h slice (~8k clips) | ~0.45 GB |
| LibriTTS-R | train-clean-100 first ~5 h | ~0.87 GB |
| Common Voice EN 26.0 | validated shards ~2 h (MP3 ~23 MB/h) | ~0.05 GB |
| Manifests/checksums | — | <0.01 GB |

PILOT — ~16 GB downloads (leaves >100 GB for features/experiments):
| Component | Subset | Why | Est. size |
|---|---|---|---|
| LibriTTS-R | train-clean-100 (8.1 GB tar) + dev/test (4 pkgs, ~4.5 GB) | 24 kHz restored multi-speaker backbone, CC BY 4.0 | ~12.6 GB |
| Common Voice EN 26.0 | validated subset ~20 h via shard selection | accent breadth, CC0 | ~0.5 GB |
| LJSpeech | full v1.1 | PD identity-safe anchor voice | 2.6 GB |
| LibriTTS-P | annotation CSVs | style prompts overlay | <1 GB |
| Deferred: Hi-Fi-TTS full 41 GB tarball, MLS-EN, Libriheavy-large | Phase B (disk planning) | scale-up after pilot gates | n/a now |
Excluded by policy: Emilia*, YODAS*, TITW, GigaSpeech (until terms reviewed), RAON (until component audit), Expresso/ESD/RAVDESS/CREMA-D (NC/research-only).

## Open items needing a human/legal decision

1. GigaSpeech form-gated dataset terms (obtain + review before any use).
2. VCTK governing instrument: CC BY 4.0 vs ODC-By 1.0 notice reconciliation.
3. Defined.ai DLA "recognizable contributor voice" clause vs flagship-clone intent.
4. Whether any LibriVox-derived identity anchor needs supplemental counsel opinion (right-of-publicity).
5. RAON-OpenTTS component audit; GLOBE HF license check. (Silero-VAD LICENSE resolved as MIT.)
6. Budget approval for purchased consented flagship recordings (Defined.ai/SpeechOcean/custom buyout).

## Sources

All verified 2026-08-22 (same list as `docs/data/LICENSE-MATRIX.md`):
- [S1] http://keithito.com/LJ-Speech-Dataset/ — The LJ Speech Dataset — Keith Ito — 2026-08-22
- [S2] https://www.openslr.org/60 — LibriTTS (SLR60) — OpenSLR — 2026-08-22
- [S3] https://www.openslr.org/141/ — LibriTTS-R (SLR141) — OpenSLR — 2026-08-22
- [S4] https://datashare.ed.ac.uk/handle/10283/3443 — CSTR VCTK Corpus v0.92 — Univ. Edinburgh DataShare — 2026-08-22
- [S5] https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text.txt — VCTK license text (CC BY 4.0) — Univ. Edinburgh — 2026-08-22
- [S6] https://catalogue.yorvoice.york.ac.uk/catalogue/M4IACQQQ — VCTK entry (ODC-BY 1.0 listing) — YorVoice — 2026-08-22
- [S7] https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu — CV Scripted 26.0 EN datasheet — Mozilla — 2026-08-22
- [S8] https://github.com/common-voice/cv-dataset — cv-dataset index — Mozilla — 2026-08-22
- [S9] https://www.openslr.org/94 — Multilingual LibriSpeech (SLR94) — OpenSLR — 2026-08-22
- [S10] https://arxiv.org/abs/2012.03411 — MLS paper abstract — Pratap et al. — 2026-08-22
- [S11] https://github.com/SpeechColab/GigaSpeech — GigaSpeech repo — SpeechColab — 2026-08-22
- [S12] https://raw.githubusercontent.com/SpeechColab/GigaSpeech/main/README.md — GigaSpeech README — SpeechColab — 2026-08-22
- [S13] https://github.com/mlcommons/peoples-speech — People's Speech repo — MLCommons — 2026-08-22
- [S14] https://huggingface.co/datasets/mlcommons/peoples_speech — People's Speech card — MLCommons/HF — 2026-08-22
- [S15] https://huggingface.co/datasets/amphion/Emilia-Dataset — Emilia card — Amphion/OpenMOSS — 2026-08-22
- [S16] https://arxiv.org/html/2407.05361 — Emilia paper — He et al. — 2026-08-22
- [S17] https://github.com/SWivid/F5-TTS/issues/2 — F5-TTS NC downgrade statement — SWivid — 2026-08-22
- [S18] https://arxiv.org/html/2410.13114 — Sound Check audit paper — 2024 — 2026-08-22
- [S19] https://ai.meta.com/research/publications/expresso-a-benchmark-and-analysis-of-discrete-expressive-speech-resynthesis/ — Expresso publication — Meta AI — 2026-08-22
- [S20] https://speechbot.github.io/expresso/ — Expresso project page — Meta AI — 2026-08-22
- [S21] https://huggingface.co/datasets/ylacombe/expresso — Expresso mirror card (CC BY-NC 4.0) — HF — 2026-08-22
- [S22] https://openslr.org/109/ — Hi-Fi Multi-Speaker English TTS Dataset (SLR109) — OpenSLR/NVIDIA — 2026-08-22
- [S23] https://github.com/HLTSingapore/Emotional-Speech-Data — ESD repo (research-only) — NUS/SUTD — 2026-08-22
- [S24] https://zenodo.org/records/1188976 — RAVDESS (CC BY-NC-SA 4.0) — Zenodo — 2026-08-22
- [S25] https://github.com/CheyneyComputerScience/CREMA-D — CREMA-D (ODbL 1.0/DbCL 1.0) — Cheyney Univ. — 2026-08-22
- [S26] http://www.jungjee.com/titw/ + https://arxiv.org/html/2409.08711 — TITW page + paper — Jung et al. — 2026-08-22
- [S27] https://huggingface.co/datasets/espnet/yodas + https://arxiv.org/html/2406.00899v1 — YODAS card + paper — ESPnet — 2026-08-22
- [S28] https://github.com/k2-fsa/libriheavy + https://arxiv.org/html/2309.08105 — Libriheavy — k2-fsa — 2026-08-22
- [S29] https://github.com/line/LibriTTS-P + https://arxiv.org/html/2406.07969v1 — LibriTTS-P — LINE Corp. — 2026-08-22
- [S30] https://github.com/krafton-ai/RAON-OpenTTS — RAON-OpenTTS — KRAFTON — 2026-08-22
- [S31] https://huggingface.co/datasets/nvidia/hifitts-2 + ISCA Interspeech 2025 langman25 — HiFiTTS-2 — NVIDIA — 2026-08-22
- [S32] https://globecorpus.github.io/ — GLOBE Corpus page — Wang, Song & Jha — 2026-08-22
- [S33] http://festvox.org/cmu_arctic/index.html — CMU ARCTIC + COPYING — CMU/Festvox — 2026-08-22
- [S34] https://defined.ai/datasets/english-female-voice-actor — Defined.ai listing — 2026-08-22
- [S35] https://defined.ai/data-license-agreement — Defined.ai DLA — 2026-08-22
- [S36] https://kingline.speechocean.com / https://www.dataoceanai.com — King-Line listings — SpeechOcean/DataoceanAI — 2026-08-22
- [S37] https://www.ldc.upenn.edu/data-management/using/licensing — LDC licensing — LDC — 2026-08-22
- [S38] https://www.ldc.upenn.edu/data-management/using-data/user-agreements — LDC user agreements — LDC — 2026-08-22
- [S39] https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner — MFA (MIT) — 2026-08-22
- [S40] https://github.com/MontrealCorpusTools/mfa-models — MFA models (CC-BY-4.0 label) — 2026-08-22
- [S41] https://github.com/snakers4/silero-vad — Silero VAD (MIT labeling; badge caveat) — 2026-08-22
- [S42] https://huggingface.co/datasets/confit/vctk-full — VCTK mirror COPYING (ODC-By claim) — 2026-08-22
