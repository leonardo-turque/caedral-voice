# CAEDRAL VOICE — Evaluation Suite & Benchmark Research

- **Author:** RESEARCHER G ("evaluation & benchmarks"), CAEDRAL VOICE program
- **Date:** 2026-08-22
- **Scope:** Reproducible auto-evaluation design (CPU-only at eval time) + audit of which third-party quality signals are credible in August 2026.
- **Status conventions:** every load-bearing claim is labeled **[REPORTED]** (number/fact from a primary source), **[ESTIMATED]** (our extrapolation from documented specs), or **[HYPOTHESIS]** (our design proposal, not yet validated). Third-party ranking claims additionally carry **[REPORTED BY THIRD PARTY]**.

---

## 0. Executive summary

1. **No single automated metric is trustworthy for frontier TTS in 2026.** Predicted-MOS tools saturate (score std across strong systems as low as 0.07–0.12 [15]), fail out-of-domain [12][14], and are nearly blind to prosodic errors humans hear clearly [14]. The credible stack is *multi-signal*: intelligibility (dual-ASR WER) + predicted MOS (as a coarse regression alarm only) + speaker-embedding distance + prosody/artifact statistics + hallucination detectors.
2. **Everything worth running runs on CPU.** DNSMOS ships as a 2.1 MB ONNX graph [8]; UTMOS-strong is packaged fairseq-free with torch+torchaudio only [3]; ECAPA-TDNN and Resemblyzer are tens-of-MB models [40][42]; Whisper large-v3-turbo has maintained INT8/ONNX and GGML-class CPU paths [27][28]. Full per-clip cost is seconds-to-tens-of-seconds [ESTIMATED].
3. **Licensing traps:** NISQA/NISQA-TTS **weights** are CC BY-NC-SA 4.0 (non-commercial) even though the code is MIT [6] — unusable for us. WavLM SV weights are CC BY-SA 3.0 [41] — avoid embedding in the product; ECAPA-TDNN (Apache-2.0) and Resemblyzer (Apache-2.0) are clean [40][42].
4. **Public arenas are directional only.** TTS Arena v2 and Artificial Analysis' Speech Arena both use Bradley–Terry/Elo-style aggregation over blind votes; known confounds include vendor-default voice identity, rater quality, and vote-distribution sensitivity [47][49][51].
5. **Human MOS cannot be automated and must stay a scheduled manual program** (Section 6): P.800-style ACR with ~24 naive raters for absolute quality; MUSHRA (BS.1534, 15–20 trained raters) once systems are close [54][55].

---

## 1. Predicted-MOS estimators: inventory, validation, licenses, CPU

### 1.1 UTMOS / UTMOSv2 / SpeechMOS packaging

**UTMOS** (UTokyo-SaruLab, VoiceMOS Challenge 2022 winner; Interspeech 2022) predicts synthetic-speech MOS from raw waveform: SSL features (wav2vec 2.0 base pretrained on LibriSpeech, plus an XLSR-53-based phoneme encoder) → frame-level BLSTM scoring → ensemble of strong/weak learners with listener/domain embeddings and a contrastive loss [1]. Validation on the BVCC-based main track: utterance-level MSE 0.165, utterance SRCC 0.897, system SRCC 0.936 (1st place in 10/16 metrics); OOD (Chinese) track system SRCC 0.988 (1st) [1][REPORTED].

- Official code: `sarulab-speech/UTMOS22`, **MIT** [2]. Requires fairseq for SSL backbones — heavier dependency chain.
- Deployment-friendly repackaging: **tarepan/SpeechMOS**, **MIT**, loads `utmos22_strong` via `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`, dependencies limited to Python ≥3.8 + torch + torchaudio, fairseq-free reimplementation of Saeki (2022) [3][REPORTED].
- Alternative: **Blinorot/UTMOS-PyTorch** (unofficial, fairseq-free, TorchScript variant; recommends batch size 1 because padding shifts scores slightly) [4][REPORTED].
- **UTMOSv2** (`sarulab-speech/UTMOSv2`, **MIT**): successor system; 1st place in 7/16 metrics, VoiceMOS Challenge 2024 Track 1 [5][REPORTED]. In one 2026 study its correlation with overall naturalness was weak (SRCC 0.118) but improved to 0.464 by averaging predictions over multiple generated samples — i.e., *sample-aggregated* UTMOS is meaningfully more reliable than single-shot [15][REPORTED].

**CPU feasibility:** wav2vec-2.0-base-scale backbone (~100M params class); per-clip inference on a modern desktop CPU is sub-second to ~1–2 s for a 10 s clip [ESTIMATED from architecture; no published CPU RTF found]. TorchScript/TorchHub paths avoid fairseq entirely [3][4].

**Validation status 2025–2026 (the honest part):**
- TTSDS2 (SSW 2025), validated against one of the largest TTS listening tests to date (200 raters, 20 recent cloning models, 4 domains incl. noisy/wild/kids): **UTMOS showed no correlation with human scores in out-of-domain settings**; among MOS predictors only SQUIM held up (ρ≈0.57–0.67) [12][REPORTED].
- A 2026 perturbation study: predictors track acoustic degradation well (SHEET-BV system SRCC 0.964, UTMOS 0.929) but are **insensitive to prosodic errors** — swapping a speaker's accent dropped human MOS by 1.84 while no model moved >0.1; models carry F0 biases humans don't have and miss speaking-rate/F0-variability changes humans do notice [14][REPORTED].
- Score saturation: across strong contemporary systems, cross-system std was 0.12 (UTMOSv2) and 0.07 (DNSMOS) — metric bias exceeds true gaps [15][REPORTED].

**Verdict:** keep UTMOS-strong as a *coarse regression alarm* (catches broken audio: noise, robotic buzz, clipping collapse) and log it always; never treat ±0.1–0.2 UTMOS deltas as evidence of quality change [HYPOTHESIS grounded in 12][14][15].

### 1.2 NISQA / NISQA-TTS

NISQA (Mittag, Naderi, Chehadi, Möller — **Quality and Usability Lab, TU Berlin**, not Dortmund) v2.0: deep CNN + self-attention on mel features; outputs MOS plus four dimensions (noisiness, coloration, discontinuity, loudness) [6]. **NISQA-TTS** (v1.0 weights, `nisqa_tts.tar`) is specifically trained to estimate naturalness of synthetic speech from TTS/VC systems (Interspeech 2020, Mittag & Möller) [6][REPORTED].

- **Licensing split — critical:** code **MIT** (© 2021 Gabriel Mittag); **model weights (`nisqa.tar`, `nisqa_tts.tar`, …) are CC BY-NC-SA 4.0 — non-commercial** [6][7][REPORTED]. For a proprietary/commercial TTS program this effectively excludes NISQA-TTS unless counsel approves internal-eval use; there is no permissive retrain-free path.
- CPU: small CNN-scale model, trivially CPU-fast [ESTIMATED].
- Verdict: **exclude from the shipped eval stack on license grounds**; note that its dimension scores would have been diagnostically useful [HYPOTHESIS].

### 1.3 DNSMOS (and why it's a floor-check, not a judge)

DNSMOS (Reddy et al., ICASSP 2021; Microsoft DNS Challenge) predicts ITU-T P.835-style dimensions **SIG / BAK / OVR** plus a P.808 mode from log-mel spectrograms [8][11]. Ships as ready-to-run **ONNX** graphs inside `microsoft/DNS-Challenge`: `sig_bak_ovr.onnx` (~2.1 MB), `model_v8.onnx`, plus a personalized variant; local runner `dnsmos_local.py` uses plain `onnxruntime` at 16 kHz over ~9 s segments [8][REPORTED]. License: repo code **MIT** (`LICENSE-CODE`), content/data **CC-BY-4.0** [8].

**Limits for clean speech (documented):**
- Training MOS distribution is heavily skewed toward 3–4 [11][REPORTED]; empirically the effective ceiling is ≈3.3: users report LibriTTS (~25k files) averaging **3.23**, synthetic clean clips 3.28, CHiME4 simulated 3.31 [9][REPORTED].
- It is gameable: a MetricGAN optimized on DNSMOS scored highest on the metric but **lowest in real listening tests**; systems scored above clean references; a system that destroyed its input still got inflated BAK/OVRL [10][REPORTED].
- Saturation across strong TTS systems: cross-system std 0.07 [15][REPORTED].

**Verdict:** excellent *sanity floor* (ms-level cost, catches hiss/noise/hum and catastrophic failure) and useful for denoiser/postfilter regressions; useless for separating two good voices. Never optimize against it [HYPOTHESIS grounded in 10].

### 1.4 SQUIM (torchaudio)

Torchaudio-SQUIM (ICASSP 2023) reference-free-estimates classic objectives (**PESQ, STOI, SI-SDR**) and a NORESQA-style **MOS using a non-matching reference** (any clean speech clip as NMR) [16]. Distributed inside torchaudio (**BSD-2-Clause** project license; checkpoints shipped with the library) [16][REPORTED]. CPU-runnable (small conv/transformer heads on SSL features) [ESTIMATED].

Validation: in TTSDS2's out-of-domain torture test, **SQUIM was the best-performing MOS predictor** (ρ≈0.57 alone, up to 0.67 combined with CMOS) [12][REPORTED]. Caveat: NMR choice affects scores — fix one canonical NMR clip in-repo for reproducibility [HYPOTHESIS].

### 1.5 SHEET family / MOS-Bench

SHEET ("Speech Human Evaluation Estimation Toolkit", Interspeech 2025, Huang et al.): open-source SSQA toolkit, **MIT**, torch.hub interface (`torch.hub.load("unilight/sheet:v0.2.5", "sheet_ssqa")`), pretrained models addressed like `bvcc/sslmos-wavlm_large/1337`, HuggingFace-hosted weights [17][REPORTED]. Variants relevant to us: **SHEET-BV** (SSL-MOS, WavLM-Large encoder, trained on BVCC only) — best-in-class on acoustic degradation (system SRCC 0.964 vs 0.750 for the multi-dataset **SHEET-MB**, trained on MOS-Bench's 8 datasets) [14][17][18][REPORTED]. MOS-Bench itself is the reference generalization study for SSQA models [18].

Trade-off: WavLM-Large backbone makes this the heaviest CPU item in the MOS drawer (multi-second per clip [ESTIMATED]); still fully CPU-feasible for batch/nightly use.

### 1.6 Newer 2025–2026 predictors (watchlist)

| Predictor | Venue | Idea | Status for us |
|---|---|---|---|
| FUSE-MOS | Interspeech 2025 | fuses wav2vec 2.0 + Whisper-encoder latents; posterior MOS + uncertainty | research-grade; SOTA on VCC2018/BVCC [20][REPORTED]; checkpoint availability unverified |
| DistilMOS | arXiv 2026 | layer-wise self-distillation vs catastrophic forgetting; MOS + SSL-token prediction | beats SSL baselines in/out of domain [21][REPORTED]; not packaged for eval use |
| MSR-UTMOS | AudioMOS Challenge 2025 Tr.3 | sampling-rate-independent conv front-end; 16/24/48 kHz | 1st in utterance MSE [22][REPORTED]; useful if we evaluate 24/48 kHz output directly |
| SALF-MOS | arXiv 2025 | small stacked-conv speaker-agnostic model | claims SOTA MSE/LCC/SRCC/KTAU [23][REPORTED]; unverified independently |
| WhisperBert | arXiv 2026 | Whisper-feature MOS head | lower absolute error than UTMOS (MSE 0.161 vs 0.242) though UTMOS leads correlation [24][REPORTED] |

None displaces UTMOS-strong + SQUIM for a production stack yet; revisit after checkpoints mature [HYPOTHESIS]. Also noted: `flexthink/ttseval-wavlm-transformer` (Apache-2.0, SpeechBrain, SOMOS-trained, Pearson 0.904 on SOMOS test) as an alternative permissively-licensed predictor [19][REPORTED].

### 1.7 MOS-predictor summary table

| Predictor | Trained for | License (code / weights) | CPU | Credibility 2026 |
|---|---|---|---|---|
| UTMOS-strong | synthetic MOS (BVCC) | MIT / MIT | yes, ~≤2 s/clip [EST] | good in-domain; fails OOD & prosody [12][14] |
| UTMOSv2 | VoiceMOS 2024 | MIT / MIT | yes [EST] | weak solo (SRCC 0.118), better averaged over samples [15] |
| NISQA-TTS | synthetic naturalness | MIT / **CC BY-NC-SA (NC!)** | yes | decent historically; **license-blocked** [6] |
| DNSMOS | noisy telecom speech | MIT / CC-BY-4.0 (content) | trivial (ONNX, ms) | floor-check only; ceiling ≈3.3; gameable [9][10] |
| SQUIM-MOS | enhanced speech + NMR | BSD-2 (torchaudio) | yes | best MOS predictor OOD in TTSDS2 [12] |
| SHEET-BV | BVCC (WavLM-Large) | MIT / MIT (HF-hosted) | yes, heavier [EST] | top acoustic-degradation tracker; prosody-blind [14] |
| ttseval-wavlm-transformer | SOMOS | Apache-2.0 / Apache-2.0 | yes [EST] | Pearson 0.904 SOMOS test [19] |

---

## 2. ASR-WER intelligibility testing (eval-time CPU only)

### 2.1 The 2026 open-ASR landscape

The reference point is the **Open ASR Leaderboard** (HuggingFace `hf-audio/open_asr_leaderboard` + its benchmark paper: 60+ systems, 10 datasets, standardized normalization, WER + RTFx) [29][30]. Key English averages [REPORTED, paper Table 3 and HF cards]:

| Model (open unless noted) | Avg WER ↓ | RTFx ↑ | License |
|---|---|---|---|
| NVIDIA Canary-Qwen-2.5B | **5.63** (#1) | 418 | unverified — check before adoption |
| IBM Granite Speech 3.3 8B | 5.74 | 145 | permissive [REPORTED] |
| IBM Granite Speech 3.3 2B | 6.00 | 260 | permissive [REPORTED] |
| MS Phi-4-Multimodal-Instruct | 6.02 | 151 | MIT (model card) [REPORTED] |
| **NVIDIA Parakeet TDT 0.6B v2** | 6.05 | 3386 | **CC-BY-4.0** [31] |
| Aqua Voice Avalon (closed) | 6.24 | – | closed |
| NVIDIA Parakeet TDT 0.6B v3 (25 langs) | 6.32 | 3333 | **CC-BY-4.0** [32] |
| NVIDIA Canary-1B-v2 | 7.15 | 749 | **CC-BY-4.0** (older Canary-1B was CC-BY-NC-4.0) [33][34] |
| openai/whisper-large-v3-turbo | mean 7.83; LS-clean-class ≈2% | 200 (GPU) | **MIT (code + weights)** [26][27] |

Leaderboard repo activity continues through Mar–Jul 2026 with Qwen3-ASR (0.6B/1.7B) entries added [29][REPORTED].

### 2.2 Which ASR for *our* evaluation

TTS-intelligibility WER needs high accuracy on **clean synthetic speech** — a much easier task than the leaderboard's noisy-domain mix. All candidates above are near-ceiling on clean audio; what actually differs for us is license + CPU cost:

- **Whisper large-v3-turbo** (809M params; decoder pruned 32→4 layers; ~8× faster than large-v3 on GPU): MIT, best-documented CPU paths — Microsoft Olive ONNX INT8 k-quant CPU recipes with built-in WER eval [27][28][REPORTED]; GGML-family runtimes run it quantized on CPU (third-party benchmark: Q8_0 turbo ≈2.01% vs fp16 large-v3 ≈1.82% on LibriSpeech test-clean [REPORTED via third-party tooling docs]). Whisper small (~244M) ≈3.4% LS test-clean per the original paper tables [25][REPORTED] — adequate for fast CI loops.
- **Parakeet TDT 0.6B v2/v3**: CC-BY-4.0, 600M params, TDT streaming-friendly architecture; GPU RTFx ~3300+ but the model is small enough that ONNX-exported CPU inference is practical (expect single-digit× real time) [31][32][ESTIMATED]. Different architecture family from Whisper → valuable as the **second opinion** in dual-ASR checks (§7).
- Canary-1B-v2: CC-BY-4.0, 978M FastConformer+Transformer, strong punctuation/capitalization (helps raw-text comparison) [33][REPORTED]; heavier on CPU.

**Recommendation:** primary judge = whisper large-v3-turbo (Q8 GGML/ONNX); secondary = Parakeet v2 ONNX for disagreement/min-WER logic; whisper small reserved for rapid inner-loop CI [HYPOTHESIS].

### 2.3 Normalization protocol (do not improvise this)

Standard = OpenAI Whisper `EnglishTextNormalizer`, applied **identically to reference and hypothesis** before WER/CER (12 documented steps: bracket/parenthetical removal, filler-word removal, contraction expansion, number/currency normalization, British→American spelling, symbol stripping, whitespace collapse; OpenAI measured up to 50% WER reduction from normalization alone on some corpora) [25][26][REPORTED].

- Implementations: `whisper/normalizers/english.py` in openai/whisper (**MIT**) [26]; mirrored in HuggingFace Transformers (`english_normalizer.py`) [26]; standalone PyPI **`whisper-normalizer` v0.1.15 (MIT**, maintainer kurianbenoy; provides BasicTextNormalizer + EnglishTextNormalizer) [36][REPORTED].
- Pair with **jiwer** for WER/CER computation (de-facto standard; license to be pinned at adoption — repo states MIT/Apache-family; verify file before vendoring) [REPORTED].
- Caveats from "What is Lost in Normalization?" (EMNLP 2024): English is fine (−5.1% absolute WER legitimately removed), but the same normalizer produces pathological results for Indic languages (e.g., Malayalam −152%) due to character-splitting inflating word counts [35][REPORTED]. Irrelevant to our English-only scope, but do not reuse the pipeline verbatim if CAEDRAL ever adds languages.
- CosyVoice 2's protocol note: punctuation excluded before WER when comparing against unprompted references — our transcripts should be normalized, not stripped ad hoc [44][REPORTED].

### 2.4 CPU feasibility summary [ESTIMATED]

| Path | Model size | Per 10 s clip (desktop CPU) | Notes |
|---|---|---|---|
| whisper.cpp-class turbo Q8 | 809M | ~5–20 s | batch nightly use |
| whisper small (any runtime) | 244M | ~1–3 s | inner CI loop |
| Parakeet v2 ONNX | 600M | ~3–10 s | second-opinion ASR |
| Canary-1B-v2 | 978M | ~10–30 s | optional; skip by default |

Nightly batch of ~500 clips ≈ 2–6 h single-machine; parallelize across cores or split ASR models across two workers [ESTIMATED].

---

## 3. Speaker-similarity metrics (voice distinctiveness)

### 3.1 Embedder inventory

| Embedder | Dims | EER (VoxCeleb1-O) | License | Threshold notes |
|---|---|---|---|---|
| ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | 192 | 0.80% (with s-norm), 0.90% without [40][REPORTED] | **Apache-2.0** [40] | SpeechBrain `verify_batch` default cosine threshold **0.25** [40][REPORTED] |
| WavLM-base-plus-SV (`microsoft/wavlm-base-plus-sv`) | – | strong SV head [41] | code **MIT**, weights **CC BY-SA 3.0** (UniSpeech root LICENSE) [41] | HF example threshold **0.86**, explicitly dataset-dependent [41][REPORTED] |
| Resemblyzer (GE2E speaker encoder) | 256 | legacy-grade | **Apache-2.0** [42] | no fixed threshold; recommends 5–30 s enrollment and calibration [42][REPORTED] |
| WavLM-large finetuned SV (`wavlm_large_finetune.pth`) | – | the de-facto TTS-paper embedder [43][37] | UniSpeech family, CC-share-alike terms [41] | used by F5-TTS & seed-tts-eval SIM |
| RawNet3 | – | – | permissive (OSS) | strongest single factor (ρ=0.73) in TTSDS2 [12][REPORTED] |

### 3.2 What papers actually report (calibration anchors)

Absolute cosine values are **embedder-specific and not comparable across embedders** (ECAPA default accept-threshold 0.25 vs WavLM example 0.86 illustrates the scale mismatch [40][41]). What transfers is *distance to the ground-truth anchor* computed with the same embedder:

- F5-TTS (ACL 2025), WavLM-large-ft embedder, LibriSpeech-PC test-clean: **SIM-o = 0.66** vs ground truth self-similarity **0.69** [43][REPORTED].
- CosyVoice 2 (arXiv 2412.10117), ERes2Net embedder, LS test-clean: **SS = 0.745–0.751** vs human-recording anchor **0.697** [44][REPORTED].
- seed-tts-eval reports SIM ×100 (test-en 65.9 for CosyVoice 2) with WavLM-large embeddings [37][44][REPORTED].
- F5-TTS eval code notes some LibriSpeech speakers vary voice per character → low GT-GT similarity in places; always compute your own GT anchor on *your* data rather than importing paper constants [43][REPORTED].
- TTSDS2 found speaker-similarity factors (RawNet3, X-vector) correlate better with human judgment than any MOS predictor (ρ≈0.6–0.73) [12][REPORTED] — this is why SIM earns gate status.

### 3.3 Protocol recommendation for CAEDRAL [HYPOTHESIS]

1. Primary embedder **ECAPA-TDNN** (clean Apache-2.0, light CPU). Secondary **Resemblyzer** (256-d GE2E, also Apache-2.0) for cross-checking; avoid WavLM SV weights in anything product-adjacent (CC BY-SA).
2. Report three numbers per sample: `sim(gen, ref_prompt)`; `anchor = mean sim(ref_take_i, ref_take_j)` over ground-truth takes of the same script/speaker; `delta = sim − anchor`. A delta ≥ −0.05 means "indistinguishable from re-recording variance" at typical anchor spreads of 0.68–0.75 [ESTIMATED from 43][44 ranges].
3. Use the ECAPA 0.25 threshold only for *verification-style* flags (wrong-voice detection), never as a quality score.
4. For zero-shot distinctiveness tracking (our own voice vs corpus drift), maintain a fixed probe set of enrolled speakers + impostor pool and log both genuine and impostor score distributions monthly.

---

## 4. Prosody & signal-level objective metrics worth logging

### 4.1 Aligned-to-reference metrics (only when we own the reference take)

The classic FastSpeech2-lineage set — **F0 RMSE** (the single most-used objective prosody metric in a systematic review covering 40 studies), **MCD** (14 studies), **V/UV voicing-decision error**, plus GPE, FFE, duration-RMSE and energy-RMSE — requires time-aligned generated/reference pairs of the same text [45][46][REPORTED]. Useful when CAEDRAL records its own reference takes; meaningless for arbitrary prompts. The same review notes these frame-level metrics miss within-speaker variability across pitch/loudness conditions [46][REPORTED].

### 4.2 Reference-free logging stats (every clip, always)

Computed with pyworld/librosa-class tools, all CPU-trivial [HYPOTHESIS on exact choices]:

- **F0:** mean, median, std, 10th–90th percentile range, per-phrase contour slope; voiced-frame fraction. Rationale: predictors are F0-biased in ways humans aren't [14], so we must see F0 drift ourselves.
- **Energy:** RMS mean/std, dynamic range (dB), crest factor.
- **Pause statistics** via lightweight VAD (silero-vad/webrtcvad class): pause count per minute, median pause duration, leading/trailing silence ratio, longest pause.
- **Speaking rate & long-form consistency:** words/sec from ASR word timestamps (chars/sec fallback); coefficient of variation of words/sec across sliding 30 s windows for long generations — humans notice rate drift that MOS predictors miss [14].
- **Spectral artifact proxies:** clipping rate (fraction of samples at |x| > 0.99 full scale), DC offset (|mean sample|), silence ratio, spectral-flatness outliers (hiss/buzz proxy), high-frequency energy ratio.

These stats double as regression detectors and as features if we later train an internal quality head [HYPOTHESIS].

---

## 5. Third-party public rankings — all claims below [REPORTED BY THIRD PARTY]

### 5.1 TTS Arena v2 (HuggingFace `TTS-AGI/TTS-Arena-V2`)

Blind pairwise "which sounds more natural" voting, ranked with an Elo-like algorithm inspired by LMSYS Chatbot Arena; names revealed after voting. Its launch post explicitly argues WER is unreliable and small-panel MOS can't separate similar models [47][48]. **Criticisms (documented):**
- Runs on **vendor-default voices**, conflating model capability with voice-selection skill — voice identity is a known significant score factor since at least Hinterleitner et al., QoMEX 2014 [51][REPORTED].
- Open public voting without rater admission gates/QC trap items (contrast vetted native-speaker panels) [51].
- Elo/Bradley-Terry sensitivity to vote distribution; ignores latency, cost, cloning fidelity [47][51].

### 5.2 Artificial Analysis Speech Arena

Bradley-Terry-style linear-regression "Quality Elo" over blind votes, 95% CIs, minimum-vote gate, ≥3 s listen requirement before voting [49]. Notably runs two arenas: **Provider Voice** (each model's own 8 voices: 2M/2F × US/UK) and a **Controlled Voice Arena** using cloned professional voices specifically to de-confound voice identity from model quality [49] — direct evidence the field recognizes criticism 5.1's confound. Also tracks price/1M chars and median latency (measured 4× daily). Current top rows (Aug 2026): Sonic 3.6 (Cartesia) 1285; Qwen-Audio-3.0-TTS-Plus (Alibaba) 1240; Simba 3.2 (SpeechifyAI) 1240; Luna TTS (VUI Labs) 1221; ElevenLabs v3 Conversational 1220 [50][REPORTED BY THIRD PARTY]. Treat as directional marketing-adjacent signal; methodology page publishes an independence statement [49].

### 5.3 Objective-ish public benchmarks

- **TTSDS / TTSDS2 leaderboard** (`ttsds/benchmark`, ttsdsbenchmark.com): multi-factor distributional score (prosody, speaker identity, intelligibility, environment, general SSL distance); TTSDS correlated 0.60–0.83 with human evals across eras vs inconsistent 0.05–0.85 for MOS predictors; TTSDS2 (validated on a 200-rater study) is the only metric >0.5 Spearman across all 12 domain-score pairs (mean ρ=0.67); quarterly dataset refresh across 14 languages; code `ttsds/ttsds` open source [12][13][53][REPORTED].
- **seed-tts-eval** (ByteDance): de-facto standard pair — Whisper-large-v3 WER (EN) / Paraformer-zh CER (ZH) + WavLM SIM; test-en/zh/hard splits [37]. Reproducibility caution: independent replications diverge from official numbers (CosyVoice2 test-en 2.57 official vs ≈3.24 reproduced after weight/code updates) [37][44][REPORTED]; maintainers' issue tracker itself debates model-based-metric fairness (#19).
- **Balacoon speech-generation leaderboard**: argues arenas dominate human-judgment space but objective metrics remain necessary; aggregates intelligibility/naturalness/SIM alternatives [52].

### 5.4 How we use these

Directional context only, never gating input. Log quarterly snapshots; investigate only when our internal signals and ≥2 external sources move together [HYPOTHESIS].

---

## 6. Human MOS protocol (executable later; cannot be automated)

Standards anchors:
- **ITU-T P.800 ACR**: 5-point absolute category rating by naive listeners sampled from the normal user population (telecom lineage); Annex B/F guidance: pair-comparison minimum 6 listeners, preferably 12+; speech material needs ≥2 male + ≥2 female talkers; P.800.2 additionally prohibits cross-language comparisons of results [54][REPORTED]. Common practice in TTS papers: ~24–32 raters [REPORTED as practice norm].
- **ITU-R BS.1534 MUSHRA**: for intermediate-quality differences — exactly our regime once near parity. Panel of **15–20 trained raters** ("data from no more than 20 assessors are often sufficient" under tight control); ≤12 stimuli per trial including mandatory hidden reference and anchors; post-hoc screening per BS.2300 [55][REPORTED]. General subjective-assessment floor per BS.1284: 10 expert or 20 non-expert listeners [56][REPORTED].

Concrete plan for CAEDRAL [HYPOTHESIS]:
1. **Quarterly ACR-MOS gate** (absolute quality): 24 naive raters × (8 systems-or-checkpoints × 10 texts) within-subject, randomized; headphones required; one high anchor (human recording from our reference corpus) + one low anchor (8 kbps-codec speech); per-rater hidden-reference/trap QC with predeclared exclusion rules; report per-system mean ± 95% CI; never compare against other-language panels.
2. **Checkpoint MUSHRA** (relative deltas during development): 15 trained raters (internal panel calibrated on BS.1116-style practice), ≤9 conditions + hidden ref + low anchor per trial, focus on prosody/naturalness deltas our auto metrics are blind to [14].
3. Crowdsourcing option (Prolific-class panels) only with rater gates: audio-device screen, trap rejection ≥1 strike-out, per-rater cap, pay above median.
4. Sample-size sanity: with ~24 raters and within-subject design, detectable difference ≈0.15–0.25 MOS at typical TTS-test variances [ESTIMATED]; MUSHRA resolves finer deltas due to reference anchoring [ESTIMATED].

---

## 7. Hallucination / repetition detection for generative TTS (alignment-free)

Failure taxonomy from the ECCD study (2026): mispronunciations, substitutions, omissions, **repetitions**, unintended continuations — worst on long, repetitive, phonetically hard utterances [39][REPORTED].

Detector set — all CPU, all alignment-free [HYPOTHESIS unless cited]:
1. **WER spikes vs running baseline**: flag clips whose normalized WER exceeds e.g. 4× the corpus median or an absolute band.
2. **Dual-ASR disagreement / min-WER**: run Whisper-turbo + Parakeet v2; take min-WER as the estimate. Justification: Whisper itself hallucinates on repetitive/short audio (documented artifacts incl. spurious "ご視聴ありがとうございました" tails), so single-ASR spikes conflate ASR and TTS failure; J-HARD-TTS-Eval institutionalized dual-ASR min-CER for exactly this reason [38][REPORTED].
3. **Repetition n-gram checks on ASR output**: max repeated 4-gram count, compression ratio `len(hyp)/len(ref)` (>1.5 flags looping), trailing-loop detection (last tokens repeated ≥k times consecutively).
4. **Multi-sample dispersion**: generate N=3 seeds per prompt; report WER best/avg/worst. J-HARD showed huge best/worst gaps on repetition-heavy text (e.g., 11.81% best vs 79.90% worst for one frontier model's repetition subset) — single-sample eval under-reports instability [38][REPORTED].
5. **Duration/rate outliers**: audio duration vs expected duration (chars/sec outside ≈[2, 8] band), truncated-continuation detection via VAD trailing-silence ratio.
6. Optional diagnostic (not gating): alignment-support shortfall at first-error boundaries has been proposed as a hallucination-onset signal [39][REPORTED].

---

## Sources

Format: URL · title · project/paper · date · license.

**MOS predictors**

1. https://arxiv.org/abs/2204.02152 · "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022" (Saeki et al., Interspeech 2022) · paper · 2022 · n/a
2. https://github.com/sarulab-speech/utmos22 · UTMOS official implementation (incl. `strong/` SSL checkpoints note) · UTokyo-SaruLab · 2022–2024 · MIT
3. https://github.com/tarepan/SpeechMOS · fairseq-free UTMOS-strong repackaging (`utmos22_strong`, torch.hub; deps torch+torchaudio only) · tarepan · maintained 2024+ · MIT (verified on page)
4. https://huggingface.co/Blinorot/UTMOS-PyTorch · unofficial fairseq-free/TorchScript UTMOS · Blinorot · 2024 · see card (unofficial)
5. https://github.com/sarulab-speech/UTMOSv2 · UTMOSv2 (VoiceMOS Challenge 2024 Track 1) · UTokyo-SaruLab · 2024 · MIT
6. https://github.com/gabrielmittag/NISQA/ · NISQA v2.0 + NISQA-TTS weights (`nisqa_tts.tar`) · G. Mittag, Quality & Usability Lab TU Berlin · code 2021, push Dec 2024 · code MIT; weights CC BY-NC-SA 4.0
7. https://pypi.org/project/nisqa/ · nisqa PyPI package v2.0.post2 · Mittag et al. · 2024 · same split (code MIT / weights CC BY-NC-SA)
8. https://github.com/microsoft/DNS-Challenge · DNSMOS ONNX models (`DNSMOS/DNSMOS/sig_bak_ovr.onnx`, `model_v8.onnx`), `dnsmos_local.py` · Microsoft · ICASSP 2021 model, repo active · code MIT (`LICENSE-CODE`), content CC-BY-4.0
9. https://github.com/microsoft/DNS-Challenge/issues/189 · "Question about DNSMOS on clean speech" (LibriTTS ≈3.23 ceiling reports) · community issue · 2022–2023 · n/a
10. https://arxiv.org/html/2403.11732 · "Hallucination in Perceptual Metric-Driven Speech Enhancement Networks" · paper · 2024 · n/a
11. https://arxiv.org/html/2010.15258v2 · "DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric…" (Reddy & Cutler) · paper · 2020/2021 · n/a
12. https://www.isca-archive.org/ssw_2025/minixhofer25_ssw.pdf (also arXiv:2506.19441) · "TTSDS2: Robust and Objective Evaluation of Human-Quality Synthetic Speech" · Minixhofer et al., SSW 2025 · 2025 · paper
13. https://arxiv.org/abs/2407.12707 · "TTSDS — Text-to-Speech Distribution Score" · Minixhofer et al. · 2024 · paper
14. https://arxiv.org/html/2606.19951 · "Investigating Human-Model Discrepancies in Speech Quality Assessment via Acoustic and Prosodic Perturbations" · paper · 2026 · n/a
15. https://arxiv.org/html/2603.24430v1 · "Iterate to Differentiate: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation" · paper · 2026 · n/a
16. https://arxiv.org/abs/2304.01448 (+ tutorial https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html) · TorchAudio-SQUIM reference-less quality/intelligibility · Kamo et al., ICASSP 2023 · 2023 · BSD-2-Clause (torchaudio)
17. https://github.com/unilight/sheet (paper arXiv:2505.15061, Interspeech 2025) · SHEET SSQA toolkit, torch.hub `sheet_ssqa`, e.g. `bvcc/sslmos-wavlm_large/1337` · Huang et al. · 2025 · MIT (verified on page)
18. https://arxiv.org/abs/2411.03715 · MOS-Bench: generalization of subjective speech quality assessment models · 2024 · paper
19. https://huggingface.co/flexthink/ttseval-wavlm-transformer · WavLM+transformer MOS predictor trained on SOMOS (Pearson 0.904 SOMOS test) · flexthink/SpeechBrain ecosystem · 2023 · Apache-2.0
20. https://www.isca-archive.org/interspeech_2025/hoq25_interspeech.pdf · FUSE-MOS (wav2vec 2.0 × Whisper-encoder fusion, posterior MOS + uncertainty) · Hoq et al., Interspeech 2025 · 2025 · paper
21. https://arxiv.org/html/2601.13700v1 · DistilMOS: layer-wise self-distillation for MOS prediction · 2026 · preprint
22. https://arxiv.org/html/2507.14647v1 · MSR-UTMOS multi-sampling-rate system, AudioMOS Challenge 2025 Track 3 · 2025 · paper
23. https://arxiv.org/html/2506.02082 · SALF-MOS small stacked-conv MOS predictor · 2025 · preprint
24. https://arxiv.org/html/2604.08562v1 · WhisperBert vs UTMOS benchmarking · 2026 · preprint

**ASR & normalization**

25. https://cdn.openai.com/papers/whisper.pdf · "Robust Speech Recognition via Large-Scale Weak Supervision" (normalizer Appendix C; WER tables incl. small.en) · Radford et al./OpenAI · 2022 · n/a
26. https://github.com/openai/whisper (normalizer: whisper/normalizers/english.py; mirrored in HuggingFace Transformers english_normalizer.py) · Whisper code+weights · 2022+ · MIT (code & model weights, per README)
27. https://huggingface.co/openai/whisper-large-v3-turbo · model card: 809M params, decoder 32→4 layers, ~8× faster than large-v3, mean WER 7.83, RTFx 200.19 · OpenAI · Oct 2024 · MIT
28. https://github.com/microsoft/olive-recipes/tree/main/openai-whisper-large-v3-turbo/cpu · CPU ONNX INT8-k-quant optimization path with WER/RTFx eval · Microsoft Olive · 2024+ · MIT (repo)
29. https://huggingface.co/spaces/hf-audio/open_asr_leaderboard (repo https://github.com/huggingface/open_asr_leaderboard) · live leaderboard, refreshed through Mar–Jul 2026 incl. Qwen3-ASR entries · HuggingFace audio team · ongoing · n/a
30. https://arxiv.org/abs/2510.06961 · "Open ASR Leaderboard" benchmark paper (60+ systems, 10 datasets, WER+RTFx tables) · HF · 2025 · paper
31. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 · Parakeet TDT 0.6B v2 (EN; avg WER 6.05, RTFx 3386 on OASR-LB) · NVIDIA · 2025 · CC-BY-4.0 (commercial OK, per card)
32. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · Parakeet TDT 0.6B v3 (25 EU langs; avg WER 6.32, RTFx 3333) · NVIDIA · 2025 · CC-BY-4.0
33. https://huggingface.co/nvidia/canary-1b-v2 · Canary-1B-v2 (978M FastConformer; avg WER 7.15, RTFx 749; LS-clean 2.18%) · NVIDIA · 2025 · CC-BY-4.0 (older Canary-1B: CC-BY-NC-4.0)
34. https://arxiv.org/pdf/2509.14128 · Canary-1B-v2 / Parakeet-TDT-0.6B-v3 technical report · NVIDIA · 2025 · paper
35. ACL Anthology 2024.emnlp-main.607 · "What is Lost in Normalization?" (normalization pitfalls; EN −5.1% abs vs Indic-language pathology) · EMNLP 2024 · 2024 · paper
36. https://pypi.org/project/whisper-normalizer/ · standalone Whisper-style normalizers (Basic/English/Malayalam), v0.1.15 · kurianbenoy · 2024–2025 · MIT (verified on page)
37. https://github.com/BytedanceSpeech/seed-tts-eval · Seed-TTS-Eval protocol (Whisper-large-v3 WER + WavLM SIM; test-en/zh/hard; issue #19 on metric fairness; replication-divergence notes) · ByteDance · 2024+ · code BSD-style (check repo); benchmarks n/a
38. https://github.com/Parakeet-Inc/J-HARD-TTS-Eval · robustness benchmark: dual-ASR min-CER, Repetition subset, best/avg/worst over 5 sampled runs; documents Whisper hallucination on repetitive audio · Parakeet Inc. · 2025–2026 · see repo
39. https://arxiv.org/html/2608.00722 · "Experience-Calibrated Contrastive Decoding" (speech-hallucination taxonomy; WER reductions up to 55.6% on SeedTTS-Eval settings) · 2026 · preprint

**Speaker similarity**

40. https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb · ECAPA-TDNN VoxCeleb embeddings (192-d; EER 0.80% VoxCeleb1-O w/ s-norm; verify threshold 0.25 default) · SpeechBrain · 2021+ · Apache-2.0
41. https://huggingface.co/microsoft/wavlm-base-plus-sv (license root: https://github.com/microsoft/UniSpeech/blob/main/LICENSE) · WavLM SV head (example threshold 0.86, dataset-dependent) · Microsoft · 2022 · code MIT (src/), weights CC BY-SA 3.0
42. https://github.com/resemble-ai/Resemblyzer · GE2E voice encoder, 256-d embeddings, 5–30 s enrollment guidance · Resemble AI · 2018+, maintained · Apache-2.0 (verified on page)
43. https://arxiv.org/abs/2410.06885 (eval code: https://github.com/SWivid/F5-TTS src/f5_tts/eval/) · F5-TTS (ACL 2025): SIM-o 0.66 vs GT 0.69 via `wavlm_large_finetune.pth`; Whisper-large-v3 WER; caveats on LibriSpeech GT similarity · SWivid · 2024/2025 · code MIT (per repo)
44. https://arxiv.org/html/2412.10117v1 · CosyVoice 2 (SS 0.745 vs human 0.697 via ERes2Net; WER protocol notes; replication divergence data) · Alibaba FunAudioLLM · 2024/2025 · paper

**Prosody metrics**

45. https://arxiv.org/abs/2006.04558 · FastSpeech 2 (objective eval: MCD + F0 RMSE lineage) · Ren et al. · 2020/2021 · paper
46. https://journals-sol.sbc.org.br/index.php/jbcs/article/download/5468/3317 · "The Evaluation of Prosody in Speech Synthesis: A Systematic Review" (F0 RMSE most used, 40 studies; MCD 14; V/UV; GPE/FFE/duration/energy RMSE; frame-level limits) · JBCS · 2024 · paper

**Rankings / arenas**

47. https://huggingface.co/blog/arena-tts · "TTS Arena: Benchmarking Text-to-Speech Models in the Wild" (methodology rationale; anti-WER/small-MOS argument) · HF + VB· 2023 · n/a
48. https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2 · TTS Arena V2 live space (Elo-like blind pairwise voting) · TTS-AGI · ongoing · n/a
49. https://artificialanalysis.ai/text-to-speech/methodology · Speech Arena methodology (Bradley-Terry-style linear-regression Elo, 95% CI, Provider-Voice vs Controlled-Voice arenas, price/latency tracking, independence statement) · Artificial Analysis · ongoing · n/a
50. https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice · current Elo standings (Aug 2026 snapshot cited in body) · Artificial Analysis · Aug 2026 · n/a
51. https://voicearena.com/tts-methodology · methodology criticism compilation (voice-identity confound w/ Hinterleitner QoMEX 2014 precedent; rater gates; ITU-T P.800.2 cross-language prohibition; Bradley-Terry + bootstrap CIs) · Voice Arena · 2025–2026 · n/a
52. https://balacoon.com/blog/tts_leaderboard/ · "Speech Generation Evaluation and Leaderboard" (arenas vs objective-metric argument; seed-tts-eval/TTSDS aggregation) · Balacoon · 2024–2025 · n/a
53. https://huggingface.co/spaces/ttsds/benchmark (site https://ttsdsbenchmark.com; code https://github.com/ttsds/ttsds) · TTSDS leaderboard, quarterly refresh, 14 languages · ttsds org · 2024+ · code Apache-2.0-class (verify at adoption)

**Human evaluation standards**

54. https://www.itu.int/rec/T-REC-P.800 · ITU-T P.800 "Methods for subjective determination of transmission quality" (ACR; pair-comparison min 6/preferably 12+ listeners; ≥2M+2F talkers) · ITU-T · 08/1996 · standard (purchase/free view)
55. https://www.itu.int/rec/R-REC-BS.1534 · ITU-R BS.1534-3 MUSHRA ("no more than 20 assessors often sufficient"; ≤12 signals/trial; hidden ref + anchors mandatory) · ITU-R · 10/2015 · standard
56. https://www.itu.int/rec/R-REC-BS.1284 · ITU-R BS.1284 (10 expert / 20 non-expert listener floors) · ITU-R · · standard

---

## THE CAEDRAL EVAL SPEC v0

Design constraints: **CPU-only at eval time** (no GPU in the eval environment), fully scriptable, deterministic given fixed model checkpoints, English-only, proprietary program → permissive licenses only for anything shipped or embedded.

### S1. Metric set (auto-runnable on CPU)

Tier G — **Gate metrics**, run on every eval batch (per clip):

| # | Metric | Tool | Gate use [HYPOTHESIS thresholds] |
|---|---|---|---|
| G1 | `wer_min` — normalized WER, dual-ASR min | whisper large-v3-turbo Q8 (GGML/ONNX) + Parakeet-TDT-0.6B-v2 ONNX; `EnglishTextNormalizer` both sides; jiwer | fail if >4% on clean single-take set, or >2× corpus median |
| G2 | `utmos_mean/std` | `utmos22_strong` via SpeechMOS torch.hub | regression alarm: mean <3.6 fails; deltas <0.2 are noise |
| G3 | `dnsmos_sig/bak/ovr` | DNSMOS ONNX + onnxruntime | floor check: OVR <3.0 flags degraded render |
| G4 | `sim_ecapa` (+`anchor_delta`) | ECAPA-TDNN (SpeechBrain); cosine gen-vs-ref; anchor = GT-take self-sim when ref available | delta < −0.08 flags voice drift |
| G5 | artifact flags | numpy/soundfile/VAD: clipping rate (>0.01%), DC offset (>1e−3 FS), silence ratio (>0.5), leading/trailing silence | any hard flag fails |
| G6 | hallucination set | ASR output analysis: repeated-4gram ratio, hyp/ref length ratio (>1.5), chars/sec outside [2,8], dual-ASR disagreement >10 pts | any flag routes to human review |

Tier T — **Trend metrics**, weekly/nightly batch:

| # | Metric | Tool | Purpose |
|---|---|---|---|
| T1 | `squim_mos` (fixed NMR clip committed to repo) | torchaudio SQUIM | OOD-robust MOS trend [12] |
| T2 | `sheet_bv_mos` | SHEET torch.hub (`bvcc/sslmos-wavlm_large/*`) | strongest acoustic-degradation tracker; heavy CPU ok in batch |
| T3 | prosody suite | pyworld/librosa/VAD: F0 mean/std/range/slope, voiced fraction, energy std+dynamic range, pause count/min + median pause dur, words/sec CV over sliding windows | long-form rate & prosody drift |
| T4 | multi-sample dispersion | N=3 seeds/prompt → WER best/avg/worst | instability detection (J-HARD convention [38]) |
| T5 | `sim_resemblyzer` | Resemblyzer 256-d | cross-check of G4 embedder family |

Explicitly **excluded**: NISQA/NISQA-TTS (CC BY-NC-SA weights [6]); WavLM SV embeddings anywhere product-adjacent (CC BY-SA weights [41]); optimizing any loss against DNSMOS/UTMOS (documented gameability [10]); single-sample repetition-subset WER without dispersion stats [38].

### S2. Tool list with licenses

| Tool / asset | Role | License | CPU cost [EST] |
|---|---|---|---|
| openai whisper large-v3-turbo (Q8 GGML/ONNX path) | primary ASR judge | MIT (code+weights) [26][27][28] | ~5–20 s/10 s clip |
| nvidia parakeet-tdt-0.6b-v2 (ONNX export) | secondary ASR | CC-BY-4.0 [31] | ~3–10 s/clip |
| openai whisper small | inner-loop CI ASR | MIT [25] | ~1–3 s/clip |
| tarepan/SpeechMOS `utmos22_strong` | predicted MOS | MIT [3] | ~≤2 s/clip |
| microsoft/DNS-Challenge DNSMOS ONNX + onnxruntime | quality floor | code MIT, models CC-BY-4.0 content [8] | ms/clip |
| torchaudio SQUIM (objective + MOS w/ fixed NMR) | robust MOS trend | BSD-2-Clause [16] | ~1 s/clip |
| unilight/sheet SHEET-BV | batch MOS tracker | MIT [17] | seconds–tens of s/clip (WavLM-Large) |
| speechbrain spkrec-ecapa-voxceleb | speaker sim | Apache-2.0 [40] | ~0.2 s/clip |
| resemble-ai/Resemblyzer | speaker sim cross-check | Apache-2.0 [42] | ~0.1 s/clip |
| whisper-normalizer PyPI or openai/whisper normalizer | text normalization | MIT [26][36] | trivial |
| jiwer | WER/CER computation | pin license file at adoption before vendoring [REPORTED] | trivial |
| pyworld / librosa / soundfile / numpy / silero-vad-class VAD | prosody + artifacts | MIT / ISC / BSD-2 / BSD-3 / MIT-class | trivial |

Total per-clip gate cost ≈ 15–40 s CPU; nightly 500-clip batch ≈ 2–6 h on one machine, parallelizable across cores and across the two ASR workers [ESTIMATED].

### S3. Scoring script interface sketch

```
# caedral_eval.py — single entrypoint, stdlib+pip only, no GPU required
usage:
  python -m caedral_eval.score --wav gen.wav --transcript prompt.txt \
         [--ref ref.wav] [--ref-speaker-takes dir/] [--n-samples 3] \
         [--asr both|turbo|small] [--out report.json] [--strict]

output JSON schema (v0):
{
  "meta": {"wav": str, "duration_s": float, "sr": int,
           "tool_versions": {...}, "checkpoint_ids": {...}, "timestamp": iso},
  "intelligibility": {
      "wer_turbo": float, "wer_parakeet": float|null, "wer_min": float,
      "cer_min": float|null, "insertions": int, "deletions": int,
      "substitutions": int},
  "predicted_quality": {"utmos": {"mean": float, "std": float},
                        "dnsmos": {"sig": float, "bak": float, "ovr": float},
                        "squim_mos": float|null, "sheet_bv": float|null},
  "speaker_similarity": {"ecapa_cosine": float|null,
                         "anchor_cosine": float|null,     # GT-take self-sim
                         "anchor_delta": float|null,
                         "resemblyzer_cosine": float|null},
  "prosody": {"f0_mean_hz": float, "f0_std_hz": float, "f0_p10_p90_range_hz": float,
              "voiced_fraction": float, "energy_std_db": float, "dynamic_range_db": float,
              "pauses_per_min": float, "median_pause_s": float,
              "words_per_sec": float, "rate_cv_longform": float|null},
  "artifacts": {"clipping_rate": float, "dc_offset_fs": float,
                "silence_ratio": float, "leading_silence_s": float,
                "trailing_silence_s": float, "crest_factor": float},
  "hallucination": {"max_repeat_4gram": int, "hyp_ref_length_ratio": float,
                    "chars_per_sec": float, "dual_asr_disagreement_wer_pts": float|null,
                    "n_sampled_runs": int, "wer_best": float|null,
                    "wer_avg": float|null, "wer_worst": float|null},
  "flags": ["G1_WER_HIGH", ...],           # empty == pass
  "gate": "PASS" | "FAIL" | "REVIEW"
}
exit codes: 0 pass, 1 gate fail, 2 input error.
```

Corpus-level aggregation script consumes a directory of these JSONs → `summary.json` (means, medians, flag rates, per-prompt-set breakdowns) + a run manifest hash so every number is reproducible from `(code commit, checkpoint ids, fixed NMR/probe assets)`.

### S4. What CANNOT be automated — human evaluation plan (see §6)

Predicted MOS, embedding similarity, and WER cannot substitute for listening tests: predictors are prosody-blind [14], saturate near the frontier [15], and arenas carry identity confounds [51]. Therefore:

- **Quarterly ACR-MOS gate** (absolute quality): ~24 naive raters, within-subject randomized design, high/low anchors, hidden-reference QC traps with predeclared exclusion, per-system mean ± 95% CI, P.800-style conduct [54]. Detectable difference ≈0.15–0.25 MOS [ESTIMATED].
- **Checkpoint MUSHRA** during development (relative deltas): 15 trained raters, ≤9 conditions + hidden reference + low anchor per trial per BS.1534 [55]; this is the only instrument that resolves the prosody/naturalness differences our auto metrics provably miss [14].
- Budget one engineer-day per quarter to build stimuli (randomization, anchors, traps) and one week wall-clock for panel turnaround; results feed the same `summary.json` schema as auto metrics so gates compare like with like [HYPOTHESIS].

— END OF SPEC v0 — revisions require re-running the full Tier-G suite plus one human MUSHRA round on the changed voices.


