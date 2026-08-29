# WEB SOURCE INDEX — shared acquisition registry

Rule: every Firecrawl-fetched page gets a row here. Agents MUST check this
index (+ research/cache/) BEFORE requesting any URL. One agent fetches;
everyone reuses the artifact.

Columns: url · track · retrieved_at · local artifact · verification status.

## Acquired sources

| url | track | retrieved_at | local artifact | status |
|---|---|---|---|---|
| https://docs.firecrawl.dev/features/search.md | infra/firecrawl | 2026-08-22 | research/cache/*search*.md | verified 200 |
| https://docs.firecrawl.dev/features/scrape.md | infra/firecrawl | 2026-08-22 | research/cache/*scrape*.md | verified 200 |
| https://docs.firecrawl.dev/features/map.md | infra/firecrawl | 2026-08-22 | research/cache/*map*.md | verified 200 |
| https://docs.firecrawl.dev/features/crawl.md | infra/firecrawl | 2026-08-22 | research/cache/*crawl*.md | verified 200 |
| https://docs.firecrawl.dev/features/batch-scrape.md | infra/firecrawl | 2026-08-22 | research/cache/*batch-scrape*.md | verified 200 |
| https://docs.firecrawl.dev/features/extract.md | infra/firecrawl | 2026-08-22 | research/cache/*extract*.md | verified 200 |
| https://docs.firecrawl.dev/rate-limits.md | infra/firecrawl | 2026-08-22 | research/cache/*rate-limits*.md | verified 200 |
| https://docs.firecrawl.dev/api-reference/errors.md | infra/firecrawl | 2026-08-22 | research/cache/*errors*.md | verified 200 |
| https://onnx.ai/onnx/operators/ | runtime/onnx | 2026-08-22 | research/cache/https-onnx.ai-onnx-operators-38c115364dad88a3.md | verified 200 |
| https://onnx.ai/onnx/operators/onnx__STFT.html | runtime/onnx | 2026-08-22 | (indexed via operators/ page; not separately scraped) | referenced |
| https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*gemelo-ai-vocos-main-LICENSE*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/charactr/vocos-mel-24khz | licenses/decoder | 2026-08-22 | research/cache/*charactr-vocos-mel-24khz.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*NVIDIA-BigVGAN-main-LICENSE*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x | licenses/decoder | 2026-08-22 | research/cache/*nvidia-bigvgan_v2_24khz_100band_256x.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/hubertsiuzdak/snac/main/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*hubertsiuzdak-snac-main-LICE*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/hubertsiuzdak/snac_24khz | licenses/decoder | 2026-08-22 | research/cache/*huggingface.co-hubertsiuzdak-snac_24khz.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*jishengpeng-WavTokenizer-mai*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/novateur/WavTokenizer | licenses/decoder | 2026-08-22 | research/cache/*novateur-WavTokenizer.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*ParallelWaveGAN--54fb5e565172b3b3.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/README.md | licenses/decoder | 2026-08-22 | research/cache/*ParallelWaveGAN--93db1de80a80c639.md | verified 200 (LIC-DEC-01) |
| https://raw.githubusercontent.com/jik876/hifi-gan/master/LICENSE | licenses/decoder | 2026-08-22 | research/cache/*jik876-hifi-gan-master-LICEN*.md | verified 200 after 429 retry (LIC-DEC-01) |
| https://raw.githubusercontent.com/jik876/hifi-gan/master/README.md | licenses/decoder | 2026-08-22 | research/cache/*jik876-hifi-gan-master-READM*.md | verified 200 after 429 retry (LIC-DEC-01) |
| https://huggingface.co/api/models/charactr/vocos-mel-24khz | licenses/decoder | 2026-08-22 | research/cache/*api-models-charactr-vocos-mel-24khz*.md | verified 200 — card license field carrier [F4] |
| https://huggingface.co/api/models/nvidia/bigvgan_v2_24khz_100band_256x | licenses/decoder | 2026-08-22 | research/cache/*api-models-nvidia-bigvgan_v2*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/api/models/hubertsiuzdak/snac_24khz | licenses/decoder | 2026-08-22 | research/cache/*api-models-hubertsiuzdak-snac_24khz*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/api/models/novateur/WavTokenizer | licenses/decoder | 2026-08-22 | research/cache/*api-models-novateur-WavTokenizer*.md | verified 200 (LIC-DEC-01) |
| https://huggingface.co/api/models/charactr/vocos-encodec-24khz | licenses/decoder | 2026-08-22 | research/cache/*api-models-charactr-vocos-encodec-24khz*.md | verified 200 (LIC-DEC-01) |
| https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/contents/egs/ljspeech/voc1/conf | licenses/decoder | 2026-08-22 | research/cache/*ParallelWaveGAN-conte*.md | verified 200 (LIC-DEC-01) |
| https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/git/trees/master?recursive=1 | licenses/decoder | 2026-08-22 | research/cache/*ParallelWaveGAN-git-t*.md | verified 200, truncated:false (LIC-DEC-01) |
| https://www.openslr.org/141 | data/librittsr | 2026-08-22 | research/cache/https-www.openslr.org-141-dc8c398d68193cff.md | verified 200 (fetched prior session; indexed at use, DATA-PILOT-01) |
| https://openslr.trmal.net/resources/141/md5sum.txt | data/librittsr | 2026-08-22 | research/cache/https-openslr.trmal.net-resources-141-md5sum.txt-e2f7940e52354b07.md | verified 200 (DATA-PILOT-01) |
| https://keithito.com/LJ-Speech-Dataset/ | data/ljspeech | 2026-08-22 | research/cache/https-keithito.com-LJ-Speech-Dataset-684f7ca8f92cbd93.md | verified 200 [S1] (DATA-PILOT-01) |
| https://www.openslr.org/13 | data/negative-check | 2026-08-22 | research/cache/https-www.openslr.org-13-e8d3250f49e04f21.md | verified 200 — SLR13 is RWCP, NOT LJSpeech (DATA-PILOT-01) |
| https://github.com/common-voice/cv-dataset | data/commonvoice | 2026-08-22 | research/cache/https-github.com-common-voice-cv-dataset-a2549fa1db65eb8f.md | verified 200 [S8] (DATA-PILOT-01) |
| https://raw.githubusercontent.com/line/LibriTTS-P/main/README.md | data/librittsp | 2026-08-22 | research/cache/https-raw.githubusercontent.com-line-LibriTTS-P-main-README.-fe1e9ba1bae85dfe.md | verified 200 [S29] (DATA-PILOT-01) |
| https://huggingface.co/api/datasets?author=mozilla-foundation&search=common_voice&limit=100 | data/commonvoice | 2026-08-22 | research/cache/https-huggingface.co-api-datasets-author-mozilla-foundation--cfd79a86f12ed7dc.md | verified 200 — only cv_17_0 + cv_13_0 remain, MDC-exclusive notices (DATA-PILOT-01) |
| https://api.github.com/repos/common-voice/cv-dataset/contents/datasets | data/commonvoice | 2026-08-22 | research/cache/https-api.github.com-repos-common-voice-cv-dataset-contents--d93cb9d43e2f1778.md | verified 200 (DATA-PILOT-01) |
| https://mozilladatacollective.com/api-reference | data/commonvoice | 2026-08-22 | research/cache/https-mozilladatacollective.com-api-reference-9d9f6c3bcd77a87b.md | verified 200 (DATA-PILOT-01) |
| https://raw.githubusercontent.com/Mozilla-Data-Collective/datacollective-python/main/README.md | data/commonvoice | 2026-08-22 | research/cache/https-raw.githubusercontent.com-Mozilla-Data-Collective-data-25ee078008dbd3ee.md | verified 200 (DATA-PILOT-01) |
| https://api.github.com/repos/common-voice/cv-dataset/contents/datasets/scripted-speech | data/commonvoice | 2026-08-22 | research/cache/https-api.github.com-repos-common-voice-cv-dataset-contents--fdd421fe04593f38.md | verified 200 (DATA-PILOT-01) |
| https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/README.md | data/commonvoice | 2026-08-22 | research/cache/https-raw.githubusercontent.com-common-voice-cv-dataset-main-f2f3d67c5aac08a2.md | verified 200 (DATA-PILOT-01) |
| https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json | data/commonvoice | 2026-08-22 | research/cache/https-raw.githubusercontent.com-common-voice-cv-dataset-main-1ce43733a5d07604.md | verified 200 — EN size+sha256+stats source (DATA-PILOT-01) |
| https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-26.0-delta-2026-06-12.json | data/commonvoice | 2026-08-22 | research/cache/https-raw.githubusercontent.com-common-voice-cv-dataset-main-5522fd2290dfed2d.md | verified 200 (DATA-PILOT-01) |
| https://huggingface.co/api/datasets/mozilla-foundation/common_voice_17_0/tree/main | data/commonvoice | 2026-08-22 | research/cache/https-huggingface.co-api-datasets-mozilla-foundation-common_-36856bb075bc7cfe.md | verified 200 — tree EMPTY except .gitattributes+README (DATA-PILOT-01) |
| https://Mozilla-Data-Collective.github.io/datacollective-python/ | data/commonvoice | 2026-08-22 | research/cache/https-Mozilla-Data-Collective.github.io-datacollective-pytho-d1ec76bb17390c94.md | verified 200 (DATA-PILOT-01) |
| https://github.com/coqui-ai/TTS/discussions/1703 | conv/recipes | 2026-08-22 | research/cache/https-github.com-coqui-ai-TTS-discussions-1703-9a5fc1deba705038.md | verified 200 (ARCH-CONV-01) |
| https://github.com/jaywalnut310/glow-tts/issues/16 | conv/recipes | 2026-08-22 | research/cache/https-github.com-jaywalnut310-glow-tts-issues-16-c6315a69f510a60f.md | verified 200 (ARCH-CONV-01) |
| https://github.com/coqui-ai/TTS/discussions/1734 | conv/recipes | 2026-08-22 | research/cache/https-github.com-coqui-ai-TTS-discussions-1734-2ba64d46224565f0.md | verified 200 (ARCH-CONV-01) |
| https://github.com/coqui-ai/TTS/discussions/3377 | conv/recipes | 2026-08-22 | research/cache/https-github.com-coqui-ai-TTS-discussions-3377-81727617e57cf7da.md | verified 200 (ARCH-CONV-01) |
| https://github.com/coqui-ai/TTS/discussions/2615 | conv/recipes | 2026-08-22 | research/cache/https-github.com-coqui-ai-TTS-discussions-2615-4cd4a81a7ca307c3.md | verified 200 (ARCH-CONV-01) |
| https://github.com/rhasspy/piper/discussions/706 | conv/recipes | 2026-08-22 | research/cache/https-github.com-rhasspy-piper-discussions-706-94307f6ff4e919df.md | verified 200 (ARCH-CONV-01) |
| https://arxiv.org/abs/2005.11129 | conv/recipes | 2026-08-22 | research/cache/https-arxiv.org-abs-2005.11129-b35acedeeaa748ef.md | verified 200 (ARCH-CONV-01) |
| https://ar5iv.labs.arxiv.org/html/2005.11129 | conv/recipes | 2026-08-22 | research/cache/https-ar5iv.labs.arxiv.org-html-2005.11129-a3a299a05404de47.md | verified 200 — full text w/ training details (ARCH-CONV-01) |
| https://ar5iv.labs.arxiv.org/html/2106.06103 | conv/recipes | 2026-08-22 | research/cache/https-ar5iv.labs.arxiv.org-html-2106.06103-bbc916d919665825.md | verified 200 — VITS full text (ARCH-CONV-01) |
| https://arxiv.org/html/2309.03199v2 | conv/recipes | 2026-08-22 | research/cache/https-arxiv.org-html-2309.03199v2-3ced150bf53321b9.md | verified 200 (ARCH-CONV-01) |
| https://arxiv.org/html/2104.05557v2 | conv/recipes | 2026-08-22 | research/cache/https-arxiv.org-html-2104.05557v2-f2c7dcc8fce86aa4.md | verified 200 (ARCH-CONV-01) |
| https://openreview.net/pdf?id=0NQwnnwAORi | conv/recipes | 2026-08-22 | research/cache/https-openreview.net-pdf-id-0NQwnnwAORi-60b1eb33ca261c9e.md | verified 200 — RAD-TTS text layer (ARCH-CONV-01) |
| https://arxiv.org/html/2206.00208v3 | conv/recipes | 2026-08-22 | research/cache/https-arxiv.org-html-2206.00208v3-35fbfbd63a068c7e.md | verified 200 — AdaVITS (ARCH-CONV-01) |
| https://github.com/Plachtaa/VITS-fast-fine-tuning | conv/recipes | 2026-08-22 | research/cache/https-github.com-Plachtaa-VITS-fast-fine-tuning-1fe7c3fcdf94327e.md | verified 200 (ARCH-CONV-01) |
| https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html | conv/recipes | 2026-08-22 | research/cache/https-k2-fsa.github.io-icefall-recipes-TTS-vctk-vits.html-b3292826a13a3577.md | verified 200 (ARCH-CONV-01) |
| https://www.emergentmind.com/topics/styletts2 | conv/recipes | 2026-08-22 | research/cache/https-www.emergentmind.com-topics-styletts2-60f1b942c47d39a9.md | verified 200 (ARCH-CONV-01) |
| https://coqui-tts.readthedocs.io/en/latest/training/finetuning.html | conv/recipes | 2026-08-22 | research/cache/https-coqui-tts.readthedocs.io-en-latest-training-finetuning-75788edf1227ce3c.md | verified 200 (ARCH-CONV-01) |
| https://sls.csail.mit.edu/publications/2020/JeffLai_ICASSP-2020.pdf | conv/recipes | 2026-08-22 | research/cache/https-sls.csail.mit.edu-publications-2020-JeffLai_ICASSP-202-751125add963753a.md | verified 200 (ARCH-CONV-01) |
| https://huggingface.co/hexgrad/Kokoro-82M | conv/recipes | 2026-08-22 | research/cache/https-huggingface.co-hexgrad-Kokoro-82M-464e108a8a04d874.md | verified 200 (ARCH-CONV-01) |
| https://arxiv.org/html/2512.17356v1 | conv/recipes | 2026-08-22 | research/cache/https-arxiv.org-html-2512.17356v1-7746888fa3c44b33.md | verified 200 — context only, big-model regime (ARCH-CONV-01) |
| https://huggingface.co/docs/transformers/en/model_doc/vits | conv/recipes | 2026-08-22 | research/cache/https-huggingface.co-docs-transformers-en-model_doc-vits-c4df8255bac192e4.md | verified 200 — JS shell, no noise_scale content usable (ARCH-CONV-01) |
| https://raw.githubusercontent.com/shivammehta25/Matcha-TTS/main/configs/model/matcha_ljspeech.yaml | conv/recipes/negative | 2026-08-22 | research/cache/https-raw.githubusercontent.com-shivammehta25-Matcha-TTS-mai-ab90996300a31bdf.md | HTTP 404 dead path (ARCH-CONV-01) |
| https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/tts/configs/vits_config.json | conv/recipes/negative | 2026-08-22 | research/cache/https-raw.githubusercontent.com-coqui-ai-TTS-dev-TTS-tts-con-af1f3a69763c7250.md | HTTP 404 dead path (ARCH-CONV-01) |
| https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py | conv/recipes/negative | 2026-08-22 | research/cache/https-github.com-coqui-ai-TTS-blob-dev-TTS-tts-configs-vits_-014dcb8fe7bfba0c.md | verified 200 but JS shell only (ARCH-CONV-01) |
| https://github.com/shivammehta25/Matcha-TTS/blob/main/configs/model/matcha_ljspeech.yaml | conv/recipes/negative | 2026-08-22 | research/cache/https-github.com-shivammehta25-Matcha-TTS-blob-main-configs--0248cba49f8107d2.md | verified 200 but JS shell only (ARCH-CONV-01) |
| https://cdn.jsdelivr.net/gh/coqui-ai/TTS@dev/TTS/tts/configs/vits_config.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-coqui-ai-TTS-dev-TTS-tts-configs-v-f9ec80e86f3eea65.md | verified 200 — NO EMA field found (ARCH-CONV-01 negative check) |
| https://cdn.jsdelivr.net/gh/coqui-ai/TTS@dev/TTS/tts/configs/shared_configs.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-coqui-ai-TTS-dev-TTS-tts-configs-s-2a39464b08a377b5.md | verified 200 — NO ema_decay found (ARCH-CONV-01 negative check) |
| https://cdn.jsdelivr.net/gh/huawei-noah/Speech-Backbones@main/Grad-TTS/train.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-huawei-noah-Speech-Backbones-main--74fa8cb81cc88da7.md | verified 200 — NO EMA found (ARCH-CONV-01 negative check) |
| https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/matcha/utils/model.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-shivammehta25-Matcha-TTS-main-matc-339ad9d950130bfa.md | verified 200 — glow-tts helpers only, no EMA (ARCH-CONV-01) |
| https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/matcha/train.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-shivammehta25-Matcha-TTS-main-matc-48c47c35dcd32bc2.md | verified 200 — no EMA in visible portion (ARCH-CONV-01) |
| https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/configs/model/matcha_ljspeech.yaml | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-shivammehta25-Matcha-TTS-main-conf-221d708507fd4d3e.md | file not found in repo (ARCH-CONV-01) |
| https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/matcha/utils/ema.py | conv/recipes/negative | 2026-08-22 | research/cache/https-cdn.jsdelivr.net-gh-shivammehta25-Matcha-TTS-main-matc-d8a31a1ff00ce98c.md | file not found in repo (ARCH-CONV-01) |
| https://github.com/resemble-ai/Resemblyzer | tools/speaker-embed | 2026-08-23 | research/cache/https-github.com-resemble-ai-Resemblyzer-d434f10637b5dc97.md | verified 200 (CV-G1-E0015 / SPK-DISTINCT-01) |
| https://raw.githubusercontent.com/resemble-ai/Resemblyzer/master/LICENSE | tools/speaker-embed | 2026-08-23 | research/cache/https-raw.githubusercontent.com-resemble-ai-Resemblyzer-mast-27973b9f31e8affd.md | verified 200 — Apache-2.0, © 2019 Resemble.AI (CV-G1-E0015) |
| https://pypi.org/project/Resemblyzer/ | tools/speaker-embed | 2026-08-23 | research/cache/https-pypi.org-project-Resemblyzer-23578344ce538bec.md | verified 200 — PyPI license field "Apache Software License" (CV-G1-E0015) |

## Findings from acquired sources

| date | finding | source row |
|---|---|---|
| 2026-08-22 | UTMOS available as unofficial PyPI pkg `utmos` (CPU/CUDA/MPS) — resolves EVAL SPEC backlog item | search result pypi.org/project/utmos |
| 2026-08-22 | ONNX spec has NO ISTFT op; STFT exists @opset 17, DFT @17/20. Local MEASURED check (venv): ORT 1.29.0 registry registers STFT-17 + DFT with working CPU kernels, ISTFT nowhere (637 ai.onnx + 123 com.microsoft ops scanned); torch 2.6 export of IstftHead fails on unsupported `aten::complex` | operators/ row + local venv measurement (RUNTIME-02) |
| 2026-08-22 | LIC-DEC-01 dossier done (docs/data/DECODER-LICENSES.md): Vocos/BigVGAN-v2/SNAC/WavTokenizer all MIT code+weights verified from primary artifacts; jik876 HiFi-GAN Drive checkpoints have NO stated license (PROHIBITED); kan-bayashi PWG LICENSE confirmed MIT but master tree contains NO iSTFTNet/iSTFTNet2 code — survey's implementation-home premise wrong | licenses/decoder rows above |
| 2026-08-22 | OpenSLR SLR141 publishes MD5-only (md5sum.txt; no official sha256). Local train_clean_100.tar.gz (7.57 GiB) MD5 `6df668d8…ad02b` matches official → authentic; local sha256 recorded as content ID (DATA-PILOT-01, docs/data/DATA-PILOT-ACQUISITION.md §1–2) | md5sum.txt row + local Get-FileHash measurement |
| 2026-08-22 | LJSpeech: keithito.com is the sole canonical host (OpenSLR SLR13 = RWCP, not LJSpeech); local tar.bz2 sha256 matches torchaudio-published official value (DATA-PILOT-01 §1, §3.2) | keithito + openslr/13 rows + local measurement |
| 2026-08-22 | CV EN SCS 26.0: monolithic per-locale bundle 94,639,372,950 B, sha256 `6809228e…49e6cb` published in cv-dataset JSON; distribution MDC-exclusive since Oct 2025; HF mozilla-foundation mirrors emptied (tree check); MDC SDK downloads whole bundle only (no partial/file selection); v26.0 EN delta contains ZERO validated clips (all bucket=other) → validated-subset acquisition requires client-side deterministic selection inside a gated full-bundle pass (DATA-PILOT-01 §3.3–3.4) | cv-dataset README+26.0 JSON+delta JSON, HF tree, MDC api-reference+SDK rows |
| 2026-08-22 | ARCH-CONV-01 (docs/architecture/CONVERGENCE-RECIPES.md): Glow-TTS-class convergence — alignment usable in few-k steps, first intelligible ~10–20k steps on 10–45h data, final quality 240k–1M steps ⇒ our babble @1–1.6k steps is on schedule; best-evidenced low-resource accelerator is pretrain→finetune staging (SC-GlowTTS LJSpeech-290k→11-spk-70k; AdaVITS 800k→2000-step adaptation; Kokoro-82M proves <100h→production at ~500 A100-h) | coqui #1703, RAD-TTS openreview, Glow-TTS/VITS/Matcha/SC-GlowTTS/AdaVITS papers, Kokoro HF card rows |
| 2026-08-22 | Negative check: EMA weights NOT shipped in coqui VITS config, coqui shared_configs, Grad-TTS train.py, or Matcha train/utils (probed via jsdelivr raw) → EMA labeled ESTIMATED in ARCH-CONV-01, not REPORTED | conv/recipes/negative jsdelivr rows |
| 2026-08-23 | Resemblyzer 0.1.4 (GE2E d-vector speaker encoder) is **Apache-2.0, not MIT** as sometimes assumed — verified from primary LICENSE + PyPI; permissive/SHIP-SAFE with attribution+NOTICE. Windows install note: pip build of `webrtcvad` needs MSVC; use prebuilt `webrtcvad-wheels` + `pip install --no-deps resemblyzer` (CV-G1-E0015) | tools/speaker-embed rows above |

## Pending acquisitions (blocked or queued)

| url | reason |
|---|---|
| https://onnxruntime.ai/docs/reference/operators.html | HTTP 404 (page moved); superseded by local ORT registry measurement |
