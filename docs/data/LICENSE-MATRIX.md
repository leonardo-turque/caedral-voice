# CAEDRAL VOICE — Dataset License Matrix (English TTS, commercial release)

Maintained by: RESEARCHER E (datasets, licensing & data quality). Verification date for ALL rows: **2026-08-22**.
Scope: commercial-use analysis for a proprietary, CPU-first, multi-voice English TTS. Research only — nothing downloaded.

## Verdict legend

- **Comm-train**: may we train models (that we sell) on it?
  - `YES` — permissive/public-domain; conditions limited to attribution/notices.
  - `COND` — conditional; see Conditions column. Do NOT proceed without satisfying every condition.
  - `NO` — license or terms bar commercial use.
  - `UNVER` — could not verify license from an authoritative source; treat as NO until verified.
- **Voice-ID OK**: may a released VOICE IDENTITY (a flagship cloneable voice) be defined primarily from it? Stricter than training: asks whether the speaker's consent/publicity-rights position is documented. `YES` / `LEAN-YES` / `RISKY` / `NO`.
- **[REPORTED]** = figure/claim seen only in secondary sources this session. **[UNVERIFIED]** = not confirmed from the official page.

## Master matrix

| # | Dataset | License (official source) | Comm-train | Voice-ID | Key conditions / caveats | Source | Verified |
|---|---------|---------------------------|------------|----------|--------------------------|--------|----------|
| 1 | LJSpeech v1.1 | Public domain ("no restrictions on its use") | YES | LEAN-YES | None. Single speaker = Linda Johnson (LibriVox reader; LibriVox dedications are PD). Source audio was 128 kbps MP3 (codec artifacts). | [S1] | 2026-08-22 |
| 2 | LibriTTS (SLR60) | CC BY 4.0 | YES (attribute) | COND→LEAN-YES | Attribution + citation of Zen et al. 2019. Derived from LibriVox (PD) + Project Gutenberg. | [S2] | 2026-08-22 |
| 3 | LibriTTS-R (SLR141) | CC BY 4.0 | YES (attribute) | COND→LEAN-YES | Same identities/audio as LibriTTS, Miipher-restored. Attribution required. | [S3] | 2026-08-22 |
| 4 | VCTK v0.92 | CC BY 4.0 per official `license_text.txt`; several catalogs (YorVoice, HF mirror COPYING) list **ODC-By 1.0** — discrepancy flagged | YES (attribute) | RISKY | Reconcile which license text governs before shipping. Newspaper prompt texts included "with permission from Herald & Times Group" (text copyright separate). Speaker consent scope for synthesis/cloning undocumented. | [S4][S5][S6][S42] | 2026-08-22 |
| 5 | Mozilla Common Voice EN 26.0 | CC0-1.0 + condition: users agree NOT to attempt to identify speakers | YES (keep validated clips; honor no-identification condition) | NO | No-identification condition conflicts with selecting/broadcasting a specific speaker's voice. Crowd-recorded, variable mic quality; MP3. | [S7][S8] | 2026-08-22 |
| 6 | Multilingual LibriSpeech, English (SLR94) | CC BY 4.0 | YES (attribute) | COND→LEAN-YES | LibriVox-derived. Huge (2.4 TB FLAC / 651 GB Opus) — plan subset downloads. Transcripts machine-aligned (ASR-grade, not studio-grade). | [S9][S10] | 2026-08-22 |
| 7 | GigaSpeech | Repo sidebar: Apache-2.0. Raw README contains **no license section**; access gated via Google Form + emailed instructions (**additional unpublished terms**) | COND / partly UNVER | NO | Underlying audio = third-party audiobooks, podcasts, YouTube. No rights disclaimer found in README. Obtain and review form/e-mail terms before ANY use. | [S11][S12] | 2026-08-22 |
| 8 | People's Speech | CC-BY family (2.0/2.5/3.0/4.0) + separate CC-BY-SA subsets (`*_sa`) | YES per-subset (CC-BY only; SA subsets impose share-alike) | NO | archive.org-sourced (gov proceedings, speeches, oral histories); known alignment defects in v1.0; transcripts often pre-existing/ASR. 16 kHz. | [S13][S14] | 2026-08-22 |
| 9 | Emilia (core, 101.7 kh) | **CC BY-NC 4.0** per card terms (HF sidebar tag `cc-by-4.0` inconsistent); access contract requires indemnifying authors; access terminable | NO | NO | Card: "copyright remains with the original owners of the videos or audio". In-the-wild scraping; no speaker consent process; F5-TTS was forced to downgrade its checkpoints to CC-BY-NC over exactly this. | [S15][S16][S17][S18] | 2026-08-22 |
| 10 | Emilia-YODAS portion (113.9 kh) | CC BY 4.0 (raw from espnet/yodas2) | COND | NO | License says yes; provenance says careful: YouTube-derived, uploader-consent unknown, platform-ToS risk. | [S15][S27] | 2026-08-22 |
| 11 | Expresso (Meta) | **CC BY-NC 4.0** (confirmed via HF mirror + Meta publication page; official card is gated) | NO (internal non-commercial evaluation only) | NO | 4 professional actors; superb expressiveness benchmark. Commercial license would need direct negotiation with Meta. | [S19][S20][S21] | 2026-08-22 |
| 12 | Hi-Fi-TTS (SLR109) | CC BY 4.0 | YES (attribute) | LEAN-YES | LibriVox+Gutenberg derived; clean subset readers are named (92 Cori Samuel, 6097 Phil Benson, 9017 John Van Stan). 44.1 kHz, SNR-screened. | [S22] | 2026-08-22 |
| 13 | HiFiTTS-2 (NVIDIA) | CC BY 4.0 (card); **metadata-only** release — you pull LibriVox audio yourself via their SDP pipeline | YES (attribute) | LEAN-YES | Same PD LibriVox lineage as MLS. Verify final audio files carry/inherit LibriVox PD terms when building locally. | [S31] | 2026-08-22 |
| 14 | ESD (Emotional Speech Database) | Custom signed license: "research purpose" ONLY; signed agreement emailed to zhoukun@u.nus.edu | NO | NO | Gated (Google Drive after signature). 10 EN speakers, 5 emotions. | [S23] | 2026-08-22 |
| 15 | RAVDESS | CC BY-NC-SA 4.0 (Zenodo); paid commercial licenses sold by Affective Data Science Lab (ravdess@gmail.com) | NO (unless purchased) | NO (unless purchased) | ShareAlike additionally infects derivatives. 24 actors (12F/12M), 48 kHz. | [S24] | 2026-08-22 |
| 16 | CREMA-D | ODbL 1.0 + DbCL 1.0 | COND | NO | ODbL share-alike obligations on derivative databases; actor consent for speech synthesis not documented; access-form tracking. Counsel review required before any commercial training. | [S25] | 2026-08-22 |
| 17 | TITW (Interspeech 2025) | Stated CC BY 4.0 BUT "copyright of the underlying human speech files remains with the original video owners"; gated HF access with author-approved terms | COND / high risk | NO | VoxCeleb1-derived in-the-wild interviews/social media; WhisperX auto-transcripts. Review gate terms; underlying-rights problem persists regardless of stated license. | [S26] | 2026-08-22 |
| 18 | YODAS / YODAS2 (espnet) | CC BY 3.0 (card metadata + paper) | COND | NO | Only YouTube videos already carrying CC licenses were collected, with takedown e-mail — but CC status is uploader self-reported; captions vary in quality. | [S27] | 2026-08-22 |
| 19 | Libriheavy | Apache-2.0 (repo + HF) | YES (Apache conditions) | COND→LEAN-YES | LibriVox-derived (labeled Libri-Light); punctuation+casing+context transcripts. | [S28] | 2026-08-22 |
| 20 | LibriTTS-P | CC BY 4.0 (annotations; audio = LibriTTS-R under CC BY 4.0) | YES (attribute) | COND→LEAN-YES | Style/identity prompt layer over LibriTTS-R; 373,868 utts / 2,443 speakers. | [S29] | 2026-08-22 |
| 21 | RAON-OpenTTS (KRAFTON) | Code Apache-2.0. Data/model cards on HF **not yet checked** → data license UNVERIFIED | UNVER (do not use yet) | NO | Aggregates 11 public EN corpora incl. reportedly Emilia (CC-BY-NC!) and YouTube-Commons [REPORTED]. An aggregate license cannot cure upstream NC/rights defects — per-component audit mandatory. | [S30] | 2026-08-22 |
| 22 | GLOBE | **No license stated** on project page; HF card unchecked | UNVER | UNVER | Common Voice-derived (favorable lineage) but do NOT assume CC0 passthrough. Check HF `MushanW/GLOBE_V2` card before any use. | [S32] | 2026-08-22 |
| 23 | CMU Arctic | Permissive CMU notice: "free for use for any purpose (commercial or otherwise)" with light attribution/modification conditions (BSD-style) | YES | LEAN-YES | Professional voice talent, explicit broad grant (2003). Legacy limitation: ~1,150 utts/speaker, 16 kHz. | [S33] | 2026-08-22 |
| 24 | JVS / "JENTZ"-class | No corpus named "JENTZ" exists [REPORTED]; JVS = Japanese Versatile Speech, free for research, Japanese only | N/A — no English version exists | N/A | Japanese-language corpora (JVS ~30 h, 100 pro speakers; JVNV emotional). Out of scope for English program. | [REPORTED, S-refs in survey] | 2026-08-22 |

## Consented / purchasable studio-data paths (commercial by design)

| Vendor/product | Offering (as advertised) | Commercial terms snapshot | Source | Verified |
|---|---|---|---|---|
| Defined.ai — English Female Voice Actor | 80 h en-US, professional actor, scripted + spontaneous, 44.1 kHz/16-bit WAV | Data License Agreement: perpetual, non-exclusive; commercialization of models trained on the data permitted; data must not be resold/redistributed; restriction that models must not output recognizable contributor voice/content as standalone assets — READ CAREFULLY against voice-cloning intent; explicit contributor consent, GDPR, ISO 27001/27701/42001 claimed | [S34][S35] | 2026-08-22 |
| SpeechOcean / DataoceanAI "King-Line" King-TTS-* | e.g. King-TTS-230 (19.4 h US male, 48 kHz), -052 (22.7 h male 48 kHz/24-bit), -013 (10.6 h female 44.1 kHz), -065 (24.7 h, 50 speakers, prosody-annotated), -313 (135 h conversation, 140 spk), -030 (British female 12.6 h) | Quote-based commercial licensing; studio recordings with phoneme/prosody annotations. Exact contract terms not public — request and legal-review per SKU | [S36] | 2026-08-22 |
| LDC (Switchboard, Fisher, Blizzard-class corpora) | Telephone/meeting/studio speech, professionally curated | **Nonmembers (incl. for-profit orgs) may use LDC data for NON-commercial research/education ONLY.** Commercial rights require for-profit LDC membership (or corpus-specific license). Some corpora carry separate superseding agreements | [S37][S38] | 2026-08-22 |
| RAVDESS commercial license | Emotional speech/song, 24 actors | Paid commercial license directly from licensor (see row 15) | [S24] | 2026-08-22 |
| Direct-to-talent studio buyout (Voices.com/Voice123/agency contracts) | Custom flagship voices, any spec (48 kHz+, target hours, emotion grids) | Cleanest chain: work-for-hire/buyout + explicit voice-cloning and model-release clauses. Cost varies; typical vendor TTS foundation-voice engagements quoted at 100–300+ finished hours [REPORTED] | practice note, no single URL | 2026-08-22 |

## Preprocessing tool licenses

| Tool | Role | License (as verified) | Source | Verified |
|---|---|---|---|---|
| Montreal Forced Aligner (code) | Word-level forced alignment | MIT | [S39] | 2026-08-22 |
| mfa-models (acoustic models/dictionaries) | Alignment models | CC-BY-4.0 (repo-level label) | [S40] | 2026-08-22 |
| Silero VAD | VAD segmentation | MIT (LICENSE file verified: "MIT License", Silero Team) | [S41] | 2026-08-22 |
| whisperX | ASR + alignment | [UNVERIFIED] — not checked this session; verify before pipeline inclusion | — | — |
| aeneas | Legacy alignment | [UNVERIFIED] — believed AGPL-family [REPORTED]; avoid in proprietary pipelines unless confirmed | — | — |

## Items deliberately left UNVERIFIED (needs a human/legal call or later session)

1. GigaSpeech form-gated dataset terms (the operative license may differ from the repo badge).
2. RAON-OpenTTS data/weights HF cards + per-component license audit (contains reportedly-NC components).
3. GLOBE license on HF (`MushanW/GLOBE_V2`).
4. VCTK: which instrument governs — CC BY 4.0 (`license_text.txt`) vs ODC-By 1.0 (catalog listings). Either way attribution-only, but ship the correct notice.
5. Blizzard Challenge releases: historically challenge/research-use [REPORTED]; not verified — check per-release agreements before considering.
6. VoxPopuli: European Parliament footage reuse terms not verified this session — excluded from matrix.
7. Defined.ai DLA clause restricting outputs of "recognizable contributor voice" vs our flagship-voice goal — needs contract interpretation.
8. ~~Silero-VAD LICENSE file~~ RESOLVED: LICENSE file confirmed MIT ("MIT License", Copyright 2020-present Silero Team).

## Sources

All verified 2026-08-22.
- [S1] http://keithito.com/LJ-Speech-Dataset/ — The LJ Speech Dataset — Keith Ito — 2026-08-22
- [S2] https://www.openslr.org/60 — LibriTTS (SLR60) — OpenSLR (JHU CLSP) — 2026-08-22
- [S3] https://www.openslr.org/141/ — LibriTTS-R (SLR141) — OpenSLR — 2026-08-22 (page details via search index of official page)
- [S4] https://datashare.ed.ac.uk/handle/10283/3443 — CSTR VCTK Corpus v0.92 — University of Edinburgh DataShare — 2026-08-22
- [S5] https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text.txt — VCTK v0.92 license text (CC BY 4.0) — University of Edinburgh — 2026-08-22
- [S6] https://catalogue.yorvoice.york.ac.uk/catalogue/M4IACQQQ — VCTK catalogue entry (lists ODC-BY 1.0) — YorVoice, Univ. of York — 2026-08-22
- [S7] https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu — Common Voice Scripted Speech 26.0 English datasheet — Mozilla — 2026-08-22
- [S8] https://github.com/common-voice/cv-dataset — cv-dataset release index — Mozilla Common Voice — 2026-08-22
- [S9] https://www.openslr.org/94 — Multilingual LibriSpeech (SLR94) — OpenSLR — 2026-08-22
- [S10] https://arxiv.org/abs/2012.03411 — MLS: SSL for 8 languages (abstract) — Pratap et al., Meta AI — 2026-08-22
- [S11] https://github.com/SpeechColab/GigaSpeech — GigaSpeech repository — SpeechColab — 2026-08-22
- [S12] https://raw.githubusercontent.com/SpeechColab/GigaSpeech/main/README.md — GigaSpeech README (no license section found) — SpeechColab — 2026-08-22
- [S13] https://github.com/mlcommons/peoples-speech — People's Speech pipelines repo — MLCommons — 2026-08-22
- [S14] https://huggingface.co/datasets/mlcommons/peoples_speech — People's Speech dataset card — MLCommons on HF — 2026-08-22
- [S15] https://huggingface.co/datasets/amphion/Emilia-Dataset — Emilia / Emilia-Large dataset card — Amphion/OpenMOSS — 2026-08-22
- [S16] https://arxiv.org/html/2407.05361 — Emilia paper — He et al. — 2026-08-22
- [S17] https://github.com/SWivid/F5-TTS/issues/2 — F5-TTS license downgrade to CC-BY-NC (maintainer statement) — SWivid — 2026-08-22
- [S18] https://arxiv.org/html/2410.13114 — Sound Check: Auditing Audio Datasets — 2024 audit (~35% potentially infringing) — 2026-08-22
- [S19] https://ai.meta.com/research/publications/expresso-a-benchmark-and-analysis-of-discrete-expressive-speech-resynthesis/ — Expresso publication page — Meta AI — 2026-08-22
- [S20] https://speechbot.github.io/expresso/ — Expresso project page — Meta AI — 2026-08-22
- [S21] https://huggingface.co/datasets/ylacombe/expresso — Expresso mirror card (CC BY-NC 4.0 statement) — HF — 2026-08-22
- [S22] https://openslr.org/109/ — Hi-Fi Multi-Speaker English TTS Dataset (SLR109) — OpenSLR/NVIDIA — 2026-08-22
- [S23] https://github.com/HLTSingapore/Emotional-Speech-Data — ESD repository (research-only license statement) — NUS/SUTD — 2026-08-22
- [S24] https://zenodo.org/records/1188976 — RAVDESS record (CC BY-NC-SA 4.0; commercial licenses via licensor) — Livingston & Russo, Zenodo — 2026-08-22
- [S25] https://github.com/CheyneyComputerScience/CREMA-D — CREMA-D repository (ODbL 1.0 + DbCL 1.0) — Cheyney Univ. — 2026-08-22
- [S26] http://www.jungjee.com/titw/ and https://arxiv.org/html/2409.08711 — TITW database project page + paper — Jung et al., Interspeech 2025 — 2026-08-22
- [S27] https://huggingface.co/datasets/espnet/yodas and https://arxiv.org/html/2406.00899v1 — YODAS card + paper (CC BY 3.0; CC-licensed YouTube sourcing) — ESPnet/CMU/UTokyo — 2026-08-22
- [S28] https://github.com/k2-fsa/libriheavy and https://arxiv.org/html/2309.08105 — Libriheavy repo + paper (Apache-2.0) — k2-fsa — 2026-08-22
- [S29] https://github.com/line/LibriTTS-P and https://arxiv.org/html/2406.07969v1 — LibriTTS-P repo + paper (CC BY 4.0) — LINE Corp. — 2026-08-22
- [S30] https://github.com/krafton-ai/RAON-OpenTTS — RAON-OpenTTS repository (code Apache-2.0; data cards unchecked) — KRAFTON — 2026-08-22
- [S31] https://huggingface.co/datasets/nvidia/hifitts-2 and https://www.isca-archive.org/interspeech_2025/langman25_interspeech.pdf — HiFiTTS-2 card (CC BY 4.0, metadata-only) + Interspeech 2025 paper — NVIDIA — 2026-08-22
- [S32] https://globecorpus.github.io/ — GLOBE Corpus project page (no license stated) — Wang, Song & Jha — 2026-08-22
- [S33] http://festvox.org/cmu_arctic/index.html — CMU ARCTIC databases + COPYING — CMU LTI / Festvox — 2026-08-22
- [S34] https://defined.ai/datasets/english-female-voice-actor — English Female Voice Actor Dataset listing — Defined.ai — 2026-08-22
- [S35] https://defined.ai/data-license-agreement — Defined.ai Data License Agreement — Defined.ai — 2026-08-22
- [S36] https://kingline.speechocean.com / https://www.dataoceanai.com — King-Line King-TTS product listings — SpeechOcean/DataoceanAI — 2026-08-22
- [S37] https://www.ldc.upenn.edu/data-management/using/licensing — LDC Licensing Terms — Linguistic Data Consortium — 2026-08-22
- [S38] https://www.ldc.upenn.edu/data-management/using-data/user-agreements — LDC User Agreements — LDC — 2026-08-22
- [S39] https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner — MFA repository (MIT) — Montreal Corpus Tools — 2026-08-22
- [S40] https://github.com/MontrealCorpusTools/mfa-models — MFA models repository (CC-BY-4.0 label) — Montreal Corpus Tools — 2026-08-22
- [S41] https://github.com/snakers4/silero-vad — Silero VAD repository (MIT labeling; badge inconsistency noted) — snakers4 — 2026-08-22
- [S42] https://huggingface.co/datasets/confit/vctk-full — VCTK mirror COPYING (states ODC-By v1.0) — HF mirror — 2026-08-22
