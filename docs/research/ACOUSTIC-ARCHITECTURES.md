# CAEDRAL VOICE — Acoustic-Model Architecture Research (Researcher C)

**Date:** 2026-08-22 · **Scope:** acoustic-model architecture classes for CPU-first batch-1 English TTS (x86-64 AVX2/VNNI, no AVX-512; 15–150M params) vs quality, with analytical cost modeling.
**Labels used throughout:** [REPORTED] = stated in cited source; [ESTIMATED] = derived here by arithmetic from labeled assumptions; [HYPOTHESIS] = untested judgment.
**Vocoder/codec-decoder cost is EXCLUDED from all RTFs unless noted** (noted separately where known).

---

## 0. Executive summary

1. **Two architecture families dominate the CPU-feasible design space**: (a) non-autoregressive (NAR) duration-predicting mel models (FastSpeech2 → VITS → StyleTTS2/Kokoro lineage) and (b) compact NAR flow-matching mel models (Matcha-TTS / ZipVoice-Distill lineage). Both do their work in a few big compute-bound BLAS passes instead of one memory-bound pass per token — the structural reason they win wall-clock at batch-1 on CPUs despite similar or higher total FLOPs.
2. **AR codec-token LMs are feasible only at low codec frame rate.** Feasibility is governed by tokens-per-second-of-audio, not model size alone: 12.5 Hz (Mimi) is ~6–15× cheaper per second of audio than 75 Hz (EnCodec-class) or ~195 Hz (WavTokenizer as configured for OuteTTS). [REPORTED rates][1][2][3][25]
3. **Diffusion acoustic models have been displaced by flow matching** as of Aug 2026 — every recent flagship (F5-TTS, ZipVoice, CosyVoice 2, Qwen-Audio-3.0-TTS) uses conditional flow matching; head-to-head ablations show FM ≈ diffusion quality at far lower sampling cost. [REPORTED][7][15][23][37][38]
4. **Small-data (<100 h) evidence favors the StyleTTS2-lineage NAR class**: Kokoro-82M was trained on <100 h (v0.19) to "few hundred hours" (v1.0) for ~$400–1000 of A100 time. [REPORTED][12] Codec LMs sit at the opposite extreme (60k–100k+ h). [REPORTED][19][24]
5. **Top-2 Phase F candidates** (Section 12): **C1 — Kokoro/StyleTTS2-lineage decoder-only NAR, 60–90M**; **C2 — compact OT-CFM Matcha/ZipVoice-Distill-lineage NAR, 25–60M acoustic + ~14M vocoder, distilled to NFE ≤ 4**.

---

## 1. Method and confidence discipline

- Facts were verified against papers and official repos (arXiv, GitHub, Hugging Face model cards, ISCA archive); community benchmarks (llama.cpp, ONNX Runtime reports) are treated as anecdotal but directionally reliable and are labeled [REPORTED] with source.
- All cost arithmetic is shown in Section 10 and calibrated against two independent reported CPU RTF data points (ZipVoice-Distill, F5-TTS Base) — the calibration closes to within ±35%, which we consider trustworthy enough to rank paradigms before building anything. [ESTIMATED]
- We never quote a benchmark number we did not find in a source.

---

## 2. Paradigm map

| # | Class | Canonical systems | Output | Steps per utterance |
|---|-------|-------------------|--------|---------------------|
| A | NAR duration-predicting mel | FastSpeech2, VITS, StyleTTS2, Kokoro, Piper | 80-mel frames | 1 (+ vocoder) |
| B | AR codec-token LM | VALL-E(-2), CSM, Orpheus, OuteTTS, Moshi | discrete codec tokens | 1/token (sequential) |
| C | NAR flow matching mel | Voicebox→E2→F5, Matcha, ZipVoice, CosyVoice-2's CFM | mel frames | NFE × (×2 if CFG) |
| D | NAR diffusion mel (legacy) | Grad-TTS, DiffSinger, NaturalSpeech 2/3 | mel/latent | 10s of NFE, curved paths |
| E | SSM/hybrid backbones | SMAM, VALL-M, MamTra | any above | backbone swap |

---

## 3. Class A — NAR duration-predicting mel models (FastSpeech2 → VITS → StyleTTS2/Kokoro)

### 3.1 Mechanics and why this parallelizes on CPU

All members share: text encoder over phonemes → explicit or inferred duration predictor expands phonemes to mel-frame counts → one transformer/conv stack processes **all T mel frames simultaneously** → decoder/vocoder synthesizes waveform. There is no sequential dependence across frames at synthesis time.

- FastSpeech 2 introduced variance adapters (duration/pitch/energy) on mel targets. [REPORTED][18]
- VITS replaced explicit alignment with learned monotonic alignment (MAS) inside a VAE+GAN; 36.3M params, GPU RTF 0.074. [REPORTED][17][5-table]
- StyleTTS 2 added style vectors via a small diffusion prior, prosody (F0/energy) predictors, iSTFTNet decoder, and WavLM-SLM adversarial training; hidden dim 512, style dim 128, two-stage training. Full training setup ≈217M components; inference-relevant generative path ≈147–161M (community component breakdown; paper states no official total). [REPORTED][10][11]
- **Kokoro-82M is a StyleTTS2-derived, decoder-only release** ("no diffusion, no encoder released"): CNN text encoder (k=5), prosody/duration predictor, PL-BERT-derived phoneme representations, ISTFTNet decoder producing 24 kHz audio directly (the iSTFT head generates magnitude+phase, so the "decoder" subsumes the vocoder). Verified from the official model card and its `models.py`/`istftnet.py`. [REPORTED][12][13][36]

**CPU economics of "one big BLAS pass":** a batched forward pass over T≈94·seconds frames issues large GEMMs that run compute-bound near sustained peak (~hundreds of GFLOPS on 8 threads), whereas an AR decode step is a GEMV streaming all weights once per token at memory bandwidth (~tens of GB/s ⇒ effective single-digit-to-low-tens of GFLOPS-equivalent). Per frame, NAR therefore converts FLOPs into wall-clock roughly 3–15× more efficiently than AR even when total FLOPs are comparable or larger. [ESTIMATED — derivation and crossover math in Section 10.4] This matches reported endpoints: Piper (VITS-class, ONNX) achieves RTF ≈0.19 on a Raspberry Pi 5 and real-time on a Pi 4 at low tiers [REPORTED][29]; Kokoro-82M ONNX runs RTF 0.51–0.69 on a 4-core Xeon 8272CL [REPORTED][28]; FastSpeech2 GPU RTF 0.010 (41.2M). [REPORTED][5-table]

Reference points within Class A:

| System | Params (inference path) | Decoder / vocoder | Reported speed | Notes |
|---|---|---|---|---|
| FastSpeech 2 | 41.2M | external vocoder | RTF 0.010 GPU [6] | variance adapters; needs external aligner or MAS |
| VITS | 36.3M | built-in HiFi-GAN-style | RTF 0.074 GPU [6] | alignment-free MAS+VAE |
| StyleTTS 2 | ~147–161M | iSTFTNet inside decoder | human-level MOS claims [10] | style diffusion + WavLM-SLM adversarial stage; two-stage training complexity |
| **Kokoro-82M** | 82M | ISTFTNet (no separate vocoder) | CPU RTF 0.51–0.69 @4 cores [28] | StyleTTS2-derived, decoder-only distillation of the class |
| Piper voices | ~1–30M tiers (x_low…high) | VITS decoder | RTF ≈0.19 Pi 5 [29] | proof the class scales *down* gracefully |

The Kokoro row is the existence proof that this exact lineage reaches high subjective quality (UTMOS 4.44 in independent CPU benchmarking [REPORTED][28]) at our parameter scale and license constraints (Apache-2.0 weights).

Notable quantization datapoint: INT8 Kokoro was *slower* than FP16/FP32 on CPU (RTF 4.5× vs 1.4–1.5×) due to dequantization overhead in conv-heavy graphs — INT8 is not automatically a win outside GEMM-dominated paths. [REPORTED][28]

### 3.2 Controllability & limits

Explicit duration/F0/energy predictors give direct knobs (rate, pitch); style vectors give voice/timbre control (StyleTTS2 needs the style encoder or diffusion prior at inference; Kokoro ships fixed voice embeddings instead). [REPORTED][10][11][12]
Weaknesses: sentence-level synthesis only by default (streaming must be engineered, Section 8); expressiveness ceiling below AR-LM zero-shot cloning without large data.

---

## 4. Class B — AR codec-token language models

### 4.1 Codec frame rates (verified)

| Codec | Rate | Token layout | Notes | Source |
|---|---|---|---|---|
| EnCodec 24 kHz mono | **75 Hz** [REPORTED] | RVQ, strides (2,4,5,8)=320 samples; bitrates 1.5–24 kbps; optional LM compresses up to 40% further | MusicGen/AudioGen token grid; interleaving K codebooks multiplies AR steps ×K | [1][2] |
| Mimi (Kyutai, Moshi/CSM) | **12.5 Hz** [REPORTED] | RVQ Q=8, ~1.1 kbps @24 kHz; first codebook semantically distilled from WavLM; fully causal, 80 ms streaming latency | Chosen explicitly to bring audio tokens near text-token rates and cut AR steps | [3][4] |
| SNAC 24 kHz | **12 / 23 / 47 Hz across 3 codebooks ≈ 82 codes/s** [REPORTED] | Multi-scale RVQ, coarse level ~12 Hz designed for LM context efficiency; codec itself 19.8M params | Orpheus flattens to 7 tokens/frame ⇒ ~82 tok/s of audio | [5][24] |
| WavTokenizer (OuteTTS config) | **~195 tok/s of audio** [REPORTED: llama.cpp PR metadata "8192 tokens ≈ 42 s"] | single codebook + iFFT-based decoder | Highest per-audio-second AR cost of the set | [25] |

**Why frame rate transforms feasibility:** an AR LM pays `t_token` once per emitted token. Tokens/s-of-audio therefore multiplies wall-clock directly — a 6× lower frame rate is exactly a 6× RTF reduction at fixed model size (Section 10.3). This is why Kyutai built Mimi at 12.5 Hz ("chosen to be closer to text token rates… reduce autoregressive steps") [REPORTED][3], and why CSM-1B pairs a Llama-3.2-1B backbone with Mimi plus a small depth-AR head over 32 RVQ levels per frame step. [REPORTED][26]

### 4.2 KV-cache memory math [ESTIMATED — formula shown]

`KV_bytes/token = 2 · n_layers · n_kv_heads · d_head · bytes_per_elt`

Worked examples:
- 80M AR LM: L=12, d_model=512 (8 heads × 64), GQA with 2 KV heads, FP32 KV → 2·12·2·64·4 = **12.3 KB/token**. At 82 tok/s × 60 s context (SNAC-rate): 4,920 tokens → **60 MB**. Trivial.
- Same model without GQA (8 KV heads): **245 MB** per minute of generated audio context — still fits RAM, but decode re-reads the whole cache every step, so attention time grows linearly with context; GQA cuts that bandwidth term 4×. [ESTIMATED][33]
- 150M AR LM: L=16, d=768 (12×64), 4 KV heads → 32.8 KB/token; 120 s @ 82 Hz → 393 MB (GQA) vs 1.57 GB (full MHA). Fine at Mimi's 12.5 Hz (49 MB/120 s); painful at high frame rate.

**RAM totals are never the binding constraint for this size class**: 80M INT8 weights ≈ 80 MB + KV tens–hundreds MB + codec decoder (~20 MB for SNAC [REPORTED][5]) ⇒ well under 1 GB. Wall-clock, not memory, decides. [ESTIMATED]

### 4.3 Reported CPU throughput anchors

- Llama-class 7B Q4_0 on desktop Ryzen: 16.7 t/s at 8 threads (llama-bench reference table); performance is memory-bandwidth-bound — ~5 threads saturate dual-channel RAM. [REPORTED][27]
- Qwen2.5-0.5B Q4_K_M: ~95 t/s on i7-13700K desktop; 14.2–47.9 t/s on Ryzen 7 5800H laptop (1–8 threads). [REPORTED][27]
- No published CPU t/s found specifically for OuteTTS-500M or CSM-1B generation; realtime for OuteTTS would require ~195 t/s (=8192/42). [REPORTED gap noted][25]

Scaling anchor [ESTIMATED]: if t_token were purely weight-streaming, 0.5B→95 t/s implies ~500 t/s at 80M INT8; measured small-model behavior includes ~ms-level fixed overheads per token, so we predict **150–250 t/s at 30–80M INT8 on a modern 8-thread desktop** (derivation in 10.3).

### 4.4 Failure modes and mitigations

Documented failure classes: token repetition/stuttering loops, word transpositions/omissions, and infinite silence under greedy decoding. [REPORTED][22][26]
Mitigations with evidence:
- **Repetition-Aware Sampling** (VALL-E 2): monitors repetition ratio over a window and resamples when it exceeds threshold; enables very low top-p without infinite loops; contributed to human-parity zero-shot results. [REPORTED][19]
- **top-p/top-k combos + repetition penalty** as used by CosyVoice 2 (its "RAS") and Orpheus (temp 0.6, top-p 0.8, rep penalty 1.3). [REPORTED][23][24]
- **Alignment structure in training**: ELLA-V phoneme-interleaved sequence reordering kills repetitions/transpositions/omissions and detects abnormally long phones. [REPORTED][22]
- **Post-training distribution steering**: GOAT (GFlowNet subtrajectory balance) >50% WER reduction on hallucination-prone subsets; attention-guidance/OAS training reduces stability hallucinations. [REPORTED][20][21]
- **Grouped code modeling** (VALL-E 2): model groups of codec frames as one step — trades effective frame rate down 2–4×, directly relevant to our CPU budget. [REPORTED][19]

### 4.5 Deployed examples

| System | Backbone | Codec / rate | Scale | CPU status |
|---|---|---|---|---|
| VALL-E / VALL-E 2 | Transformer, AR+NAR two-stage | EnCodec-class 75 Hz | 60k h train; not released [19] | — |
| CSM-1B (Sesame) | Llama-3.2-1B + depth-AR head over 32 RVQ levels | Mimi @12.5 Hz [26] | 1B backbone; audio head 96.1M in GGUF | GGUF split-runtime exists (llama.cpp + separate codec runtime); Mimi convs don't quantize below Q8_0 well [REPORTED][26] |
| Orpheus (Canopy) | Llama-3.2-3B | SNAC flattened ≈82 tok/s [24] | 100k+ h; 150M/400M/1B variants announced, unreleased | runs via llama.cpp/GGUF community quants |
| OuteTTS v0.3 | Qwen2 500M/1B, GQA (14q/2kv) | WavTokenizer ≈195 tok/s [25] | trained on modest public corpora | native llama.cpp `llama-tts` support |
| Moshi (Kyutai) | 7B dialogue LM | Mimi @12.5 Hz | foundation scale | real-time full-duplex on GPU-class hardware [REPORTED][3] |

Pattern worth noting for us: **the only codec-LM TTS systems that run comfortably at batch-1 CPU are either (a) small-backbone + low-frame-rate codec (OuteTTS-500M is borderline), or (b) huge models nobody expects to run on CPU.** None of the published small-data successes exist in this class — its economics assume internet-scale pretraining. [ESTIMATED interpretation of REPORTED facts above]

---

## 5. Class C — Flow matching for speech (Voicebox → E2 → F5 → Matcha → ZipVoice)

### 5.1 Mechanics

Conditional flow matching (CFM) learns a velocity field whose ODE maps noise→mel along (near-)straight probability paths; sampling = N Euler steps, each one full network pass over all T frames (NAR). OT-CFM's straight paths are why Matcha stays intelligible at NFE=2 (WER 2.34 vs 2.09 at NFE=10). [REPORTED][6]

### 5.2 NFE in practice and solver costs

| System | Params | NFE used | CFG | RTF (GPU) | RTF (CPU) |
|---|---|---|---|---|---|
| E2-TTS | 333M | 32 | yes | 0.68 (3090) | — [REPORTED][7] |
| F5-TTS Base | 336M | 16 / 32 | yes (sway sampling helps fewer steps) | 0.15 / 0.31 (3090) | **37.28 @32NFE** [REPORTED][7][15] |
| F5-TTS Small | ~149M (dim 768, depth 18, heads 12) | 16–32 typical | yes | — | — [REPORTED][8] |
| Matcha (MAT) | 18.2M | 2 / 4 / 10 | yes | 0.015 / 0.019 / 0.038 | — [REPORTED][6] |
| ZipVoice | 123M (Zipformer) | 16 | yes | 0.0557 | 9.553 [REPORTED][15] |
| ZipVoice-Distill | 123M | 8 / 4 | **eliminated by distillation** | 0.0233 / 0.0125 | **2.418 / 1.220** [REPORTED][15] |

CPU solver cost = NFE × (CFG multiplier) × one NAR pass (Section 10.2). Euler is first-order — no adaptive-step overhead, perfectly BLAS-friendly. The ZipVoice table is the single most important public datapoint for this program: **a 123M flow-matching model distilled to 4 NFE without CFG runs at CPU RTF 1.22** (described as near-real-time even on a single CPU thread). [REPORTED][15]

### 5.3 Few-step distillation evidence

- **FlashSpeech**: adversarial **consistency training from scratch** (no diffusion teacher needed), 1–2 sampling steps, ≈5% of prior zero-shot systems' inference time; quality plateaus at NFE=2 (UTMOS 4.00). [REPORTED][16]
- **ZipVoice flow distillation**: reduces NFE 16→4 *and* removes CFG evaluation overhead — a double win specific to CPU. [REPORTED][15]
- CosyVoice 2 runs its production streaming stack at NFE=10 with CFG β=0.7. [REPORTED][23]
[ESTIMATED] For our sizes, plan on NFE 4–8 undistilled, distill to NFE 2–4 + CFG-free once quality is anchored.

---

## 6. Class D — Diffusion acoustic models, status Aug 2026

**Verdict: displaced from the frontier by flow matching; retained only in legacy/production-legacy systems and in StyleTTS2's small style-diffusion component (dropped in Kokoro).** Evidence:
- 2026 literature explicitly frames the shift: "more recent studies have shifted toward flow matching" (WavTTS, naming Diff-TTS/Grad-TTS/ProDiff as the prior paradigm). [REPORTED][38]
- Head-to-head ablation (ProsodyFlow, COLING 2025): swapping FM for diffusion changes CMOS by only −0.04 — equal quality, worse efficiency. [REPORTED][37]
- Production/frontier systems 2025–2026 are FM-based: F5-TTS, ZipVoice, CosyVoice 2 (chunk-aware CFM), Qwen-Audio-3.0-TTS ("flow-matching acoustic renderer"). [REPORTED][7][15][23][39]
- F5 paper frames its RTF gains as "greatly improved compared to state-of-the-art diffusion-based TTS." [REPORTED][7]
- Frontier is moving past mel entirely (waveform-native FM, BareWave). [REPORTED][40]
Implication for us: **do not invest in a diffusion acoustic head**; if iterative refinement is needed, CFM with distillation dominates it on every axis we care about (steps, straightness, CPU BLAS shape). [HYPOTHESIS grounded in the above]

---

## 7. Efficient backbone choices inside these classes

- **Transformer vs Conformer vs hybrid conv/attention.** The winning pattern at small scale is *conv-local + attention-global hybrids*: Matcha = 1D-CNN residual U-Net + Transformer blocks [REPORTED][6]; F5 = ConvNeXt-V2 text refiner feeding a 22-layer DiT [REPORTED][7]; StyleTTS2/Kokoro = CNN text encoder (k=5) + LSTM duration/prosody predictor + iSTFTNet conv decoder with only a small transformer for style [REPORTED][10][11][12]. Pure Conformer is common in ASU-style TTS but brings no CPU advantage we could verify; the conv-hybrid pattern dominates the small fast models.
- **Positional encoding:** RoPE is used in both Matcha (rotary pos. embeddings) and F5; learned/relative encodings elsewhere. RoPE is now default and costs nothing at inference. [REPORTED][6][7][34]
- **GQA/MQA for AR:** OuteTTS-500M ships 14 query heads / **2 KV heads** — GQA in a production TTS LM. Benefit is KV size and per-step attention bandwidth (Section 4.2 math: 4× reduction in the worked example). For NAR classes GQA is irrelevant (no decode loop). [REPORTED][25][ESTIMATED effect size][33]
- **Mamba-class / SSM:** research exists but no mainstream TTS deployment found as of Aug 2026. SMAM (Interspeech 2025): 25–26M-param Mamba streaming acoustic model over RVQ codes, CER 2.73%, MOS 4.02, **CPU RTF ~0.70 with sub-70 ms latency** — the strongest existence proof [REPORTED][30]. Speech Slytherin: bidirectional Mamba ≈ or > self-attention in encoders, unidirectional slightly worse for joint text+speech; efficiency crossover only beyond ~15 s utterances at typical token rates [REPORTED][31]. MamTra: hybrid Mamba-Transformer initialized from pretrained weights, comparable quality [REPORTED][32]. Tooling maturity and small-data behavior unproven → treat as watchlist, not Phase F. [HYPOTHESIS]
- **Decoder note:** iSTFTNet-style heads (magnitude/phase + inverse STFT) replace the vocoder inside the decoder at modest cost — this is why Kokoro needs no external vocoder while Matcha-class mel models add ~13.9M params of HiFi-GAN/Vocos. [REPORTED][6][12][13][14]

---

## 8. Streaming compatibility per class

| Class | Native streaming? | Evidence / mechanism |
|---|---|---|
| A: NAR deterministic | No — sentence-level by construction | Chunking must respect duration/expansion consistency; standard practice is per-sentence synthesis (Piper streams by splitting text into sentences). [REPORTED][29] [HYPOTHESIS: prosodic-boundary chunked NAR with cross-chunk context is feasible but unpublished at our scale] |
| B: AR codec LM | **Yes — native** | Emits frames sequentially; Moshi full-duplex at ~160 ms theoretical / ~200 ms practical latency; Mimi codec itself causal with 80 ms latency. [REPORTED][3] |
| C: NAR flow matching | Engineered streaming possible | CosyVoice 2 "chunk-aware flow matching": treat the N stacked CFM steps as one deep causal net with non-causal/full-causal/chunk-M/chunk-2M masks; claims first-packet ≈150 ms — but a real-world issue reports 3–4 s first-chunk latency in practice. [REPORTED][23] |
| D: diffusion | Poor (same machinery as C, more steps) | Superseded anyway. |

Streaming ranking: AR > chunked-FM > chunked-NAR-deterministic. If CAEDRAL VOICE's product surface requires <300 ms first audio on CPU, that pushes toward AR-at-low-frame-rate or aggressively engineered chunked CFM; if batch/near-batch generation suffices (stated deployment target), NAR wins outright. [ESTIMATED]

Latency budget arithmetic for a hypothetical streaming NAR [ESTIMATED]: first-audio latency = time to buffer one prosodic clause (~2–4 s of text) + synthesis pass (Class A: RTF 0.04–0.19 ⇒ 80–760 ms) + vocoder + playback priming ⇒ **~0.5–1.5 s to first audio**, versus AR at Mimi rate: one 80 ms codec frame generated in ≤ t_token (≤9 ms at our sizes) + look-ahead ⇒ **~100–250 ms** achievable (Moshi demonstrates ~200 ms practical [REPORTED][3]; SMAM sub-70 ms acoustic latency on CPU [REPORTED][30]). This ~5–10× gap is the entire streaming argument for Class B — and it is the *only* axis on which Class B beats compact NAR at our scale.

---

## 8b. Notation used in this document

- `P` = model parameters; `f_mel` = mel/acoustic frame rate per second of audio; `f_codec` = codec token rate per second of audio; `NFE` = number of neural function evaluations (ODE solver steps); `k_CFG` = classifier-free-guidance multiplier (2 when CFG doubles passes); `F_eff` = effective sustained throughput (GFLOPS); `BW_eff` = effective memory bandwidth; `t_ovh` = fixed per-token decode overhead; `RTF` = real-time factor (compute time / audio duration; <1 = faster than real time); `T` = number of frames in an utterance.

---

## 9. Trainability on SMALL data (<100 h licensed)

Known data points:
- **Kokoro v0.19: trained on <100 h, 1 language, 10 voices, ≈$400 of A100 time; v1.0: "few hundred hours," total ≈$1000.** Caveat: card lists synthetic audio from closed TTS providers among sources — so "<100 h licensed" reproducibility has an asterisk. [REPORTED][12]
- VITS/FastSpeech2 class reaches intelligible speech on LJSpeech (~24 h); Piper ships hundreds of such voices across 30+ languages. [REPORTED][17][18][29]
- StyleTTS 2 zero-shot adaptation demonstrated on LibriTTS ~245 h (vs VALL-E's 60k h). [REPORTED][10]
- Zero-shot flow-matching flagships need ~100k h to reach their reported quality (F5: Emilia 95–100k h on 8×A100; ZipVoice ~100k h). Single-speaker/small-corpus FM training is plausible but unpublished at <100 h. [REPORTED][7][15][HYPOTHESIS]
- Codec LMs are the hungriest: VALL-E 60k h; Orpheus 100k+ h; CSM/Moshi foundation-scale. [REPORTED][19][24][3]

Data-hunger ranking [ESTIMATED from the above]: deterministic NAR (FS2/VITS/Kokoro-lineage, esp. with PL-BERT-style pretrained phoneme representations) **<** compact single/multi-speaker OT-CFM **<** zero-shot FM/DiT **<** AR codec LMs.

Tabulated for our <100 h licensed constraint:

| Class | Intelligible at <100 h? | Evidence strength |
|---|---|---|
| A: FS2/VITS/StyleTTS2-lineage | **Yes — demonstrated** (LJSpeech 24 h; Kokoro <100 h) | strong [12][17][29] |
| C′: compact OT-CFM, few voices | Likely — Matcha-class trained on LJSpeech-scale in the paper | moderate [6] [HYPOTHESIS for multi-voice] |
| C″: zero-shot FM/DiT cloning | No (100k h class); single-speaker subset untested publicly | strong for the 100k h claim [7][15] |
| B: codec-token LM | No published <100 h success found | strong absence-of-evidence [19][24] |

Caveat that matters for CAEDRAL: Kokoro's sub-100 h run mixed licensed and *synthetic* audio from closed TTS providers [REPORTED][12]; a strictly-licensed reproduction should budget the full <100 h for real recordings plus a pretrained phoneme frontend, and treat "Kokoro-at-<100 h with licensed-only data" as an open question Phase F can settle cheaply. [HYPOTHESIS]

---

## 10. Analytical cost model (all content [ESTIMATED]; formulas and arithmetic shown)

### 10.1 Hardware assumptions (explicit)

| Symbol | Meaning | Value used | Derivation |
|---|---|---|---|
| clk | sustained all-core clock | 4.0 GHz | modern 8 P-core desktop |
| peak FP32/core | 2 FMA units × 8 lanes × 2 flops × clk | **128 GFLOPS** | 32 flops/cycle × 4.0e9 |
| peak FP32, 8 cores | ×8 | **1024 GFLOPS** | AVX2 FMA, no AVX-512 |
| peak INT8 (VNNI) | ≈4× FP32 | 4096 GOPS | VPDPBUSD: 8×4 MACs/instr/unit |
| F_eff (H8) | effective throughput, batched NAR passes, 8 threads, mixed real-world graph | **400 GFLOPS** central, band 250–550 | calibrated below |
| F_eff (H1) | same, single thread | **80 GFLOPS** central, band 50–110 | calibrated below |
| BW_eff | effective weight-streaming bandwidth, batch-1 decode | **60 GB/s** central, band 40–70 | dual-channel DDR5-5600 ≈ 89.6 GB/s peak; llama.cpp reports ~5 threads saturate RAM [REPORTED][27] |
| t_ovh | fixed per-token overhead (sampling, launches, non-GEMV ops) | **3 ms** | fitted below |

**Calibration against reported CPU RTFs** (mel rate assumed f_mel = 93.75 Hz = 24 kHz/hop-256, typical for this family [ESTIMATED]):
- ZipVoice-Distill 123M, NFE=4, CFG eliminated [REPORTED][15]: FLOPs per second-of-audio = 2 × 123e6 × 93.75 × 4 = **92.3 GFLOPs**. Reported CPU RTF 1.220 ⇒ implied sustained throughput = 92.3 / 1.220 = **75.6 GFLOPS**.
- F5-TTS Base 336M, NFE=32, CFG ×2 [REPORTED][7][15]: 2 × 336e6 × 93.75 × 32 × 2 = **4.032 TFLOPs**/s-audio. Reported CPU RTF 37.284 ⇒ implied **108.2 GFLOPS** (or 54.1 if its published CPU run disabled CFG).
- The two independent points imply 54–108 GFLOPS/thread-equivalent → our H1=80 central is mid-band; H8=400 assumes 5× thread scaling typical of mixed graphs. **The model closes to ±35% on independent data — adequate for paradigm ranking, inadequate for final tuning (Phase F must re-measure).**

### 10.2 Class A/C: NAR passes (compute-bound)

`RTF_acoustic = NFE × k_CFG × 2·P·f_mel / F_eff`, with f_mel = 93.75 Hz; k_CFG ∈ {1 (distilled/no CFG), 2}.

Class A (single deterministic pass, NFE=1):

| Params | FLOPs per s-audio | RTF @ H8 (400G) | RTF @ H1 (80G) |
|---|---|---|---|
| 30M | 2×30e6×93.75 = 5.63 G | **0.014** | 0.070 |
| 80M | 15.0 G | **0.038** | 0.19 |
| 150M | 28.1 G | **0.070** | 0.35 |

Class C (OT-CFM, distilled, k_CFG=1):

| Params | NFE=4 FLOPs/s-audio | RTF @H8 | RTF @H1 | NFE=8 @H8 |
|---|---|---|---|---|
| 30M | 22.5 G | **0.056** | 0.28 | 0.11 |
| 80M | 60.0 G | **0.15** | 0.75 | 0.30 |
| 150M | 112.5 G | **0.28** | 1.41 | 0.56 |

Undistilled + CFG doubles again (80M @ NFE8+CFG ⇒ RTF 0.60 H8 — still viable; matches ZipVoice's measured advantage from killing both terms [REPORTED][15]).
Second-order term: self-attention adds ≈ 4·T²·d·L FLOPs per pass; at T=938 frames (10 s), d=512, L=8 ⇒ ≈1.4×10¹⁰, i.e., co-dominant with the dense term only beyond ~10 s utterances; conv-hybrid/local-window backbones suppress it. Ignore below 10 s. [ESTIMATED]
Sanity cross-check: Kokoro-class observed CPU RTF 0.51–0.69 on 4 modest cores [REPORTED][28] exceeds our transformer-only estimate because its wall-clock is dominated by the iSTFTNet conv decoder op mix — a reminder that **decoder/vocoder ops, not the acoustic transformer, often set NAR floor**. Vocoder cost excluded above; budget +13.9M params (Vocos/HiFi-GAN class) or fold into an iSTFTNet decoder. [REPORTED][6][14]

### 10.3 Class B: AR decode (memory-bound)

`t_token = t_ovh + P_active_bytes / BW_eff`; `tokens_generated_per_s = 1/t_token`; `RTF_acoustic = f_codec / tokens_per_s`.

Overhead fit: Qwen2.5-0.5B Q4 (~350 MB active) at 95 t/s ⇒ t_token = 10.5 ms; bandwidth term 350 MB/60 GB/s = 5.8 ms ⇒ t_ovh ≈ 4.2 ms. Cross-check 7B Q4_0 (4.2 GB) at 16.7 t/s: predicted 4.2+70 = 74 ms vs actual 59.9 ms ⇒ use t_ovh = 3 ms, BW = 60 GB/s, carry ±40% error bars (anchors bracket the truth).

Predicted generation speed (FP32 / INT8 weights):

| Params | active bytes FP32/INT8 | t_token | tok/s FP32 | tok/s INT8 |
|---|---|---|---|---|
| 30M | 120 / 30 MB | 5.0 / 3.5 ms | 200 | 286 |
| 80M | 320 / 80 MB | 8.3 / 4.3 ms | 120 | 231 |
| 150M | 600 / 150 MB | 13.0 / 5.5 ms | 77 | 182 |

Resulting acoustic-only RTF (INT8):

| Codec rate | 30M | 80M | 150M |
|---|---|---|---|
| **12.5 Hz (Mimi)** | 0.044 | **0.054** | 0.069 |
| 75 Hz (EnCodec, flat RVQ level-0 only) | 0.26 | 0.32 | 0.41 |
| 82 Hz (SNAC flattened) | 0.29 | 0.36 | 0.45 |
| 195 Hz (WavTokenizer cfg.) | 0.68 | 0.84 | **1.07 (not realtime)** |
| 600 Hz (EnCodec 75 Hz × 8 codebooks interleaved) | 2.1 | **2.6 — infeasible** | — |

Caveats: (i) RVQ depth-AR (CSM-style) adds sub-steps per frame on a *smaller* head — cost sits between flat and interleaved; (ii) KV-cache reads add linearly with context (Section 4.2) — matters past ~1 min; (iii) quality-preserving sampling (top-p + rep-penalty + RAS) adds minor per-token cost.

### 10.4 Paradigm crossover equation

NAR-FM beats AR when `NFE·k·2·P·f_mel / F_eff < f_codec·(P_bytes/BW_eff + t_ovh)`.

At P=80M, f_mel=93.75, F_eff=400 GFLOPS, BW=60 GB/s, t_ovh=3 ms:
- LHS per NFE unit = 2×80e6×93.75/400e9 = **37.5 ms/s-audio**
- vs 12.5 Hz INT8 AR: 12.5×(80e6/60e9 + 0.003) = **54.2 ms** ⇒ parity at **NFE ≈ 1.4**
- vs 12.5 Hz FP32 AR: 12.5×(5.33+3 ms) = 104 ms ⇒ parity at NFE ≈ 2.8
- vs 82 Hz INT8 AR: 355 ms ⇒ parity at NFE ≈ 9.5
- vs 82 Hz FP32 AR: 82×(5.33+3) = 683 ms ⇒ parity at NFE ≈ 18

**Readings:** (1) against any high-frame-rate codec LM, compact flow matching at NFE ≤ 8 wins by ≥2× regardless of size — both sides scale linearly in P, so the conclusion is size-invariant; (2) against a 12.5 Hz INT8 AR LM, NAR economics are roughly a wash at NFE 2–4 — **the tiebreakers become data appetite (Section 9), failure modes (4.4), and streaming needs (8)**, not compute; (3) NAR keeps a robustness edge: its performance depends on GFLOPS utilization (tunable via threads/batching) while AR is pinned to RAM bandwidth, which we cannot upgrade. [ESTIMATED]

### 10.5 RAM summary (weights + KV, GiB-scale)

| Config | Weights FP32/INT8 | KV cache example | Peak RAM est. |
|---|---|---|---|
| 80M NAR + 14M vocoder | 356 / 89 MB | none | < 0.6 GB incl. activations |
| 150M NAR DiT + vocoder | 614 / 154 MB | none | < 1 GB |
| 80M AR + Mimi/SNAC decoder | 99.8 / 20 MB (INT8) | 60–245 MB/min ctx (GQA on/off) | < 0.6 GB |
All trivially feasible ⇒ **RAM never discriminates in the 15–150M class; wall-clock does.** [ESTIMATED]

### 10.6 Lower-bound size check (15M, the floor of our range)

Class A at 15M: FLOPs/s-audio = 2×15e6×93.75 = 2.81 G ⇒ RTF 0.007 (H8) / 0.035 (H1). Class C′ at 15M NFE=4: 11.3 G ⇒ RTF 0.028 / 0.14. Class B INT8: t_token ≈ 3.4 ms ⇒ 294 tok/s ⇒ RTF 0.043 @Mimi-rate / 0.28 @SNAC-flat. Conclusion: **at 15M everything is fast; quality, not compute, sets the size floor** — consistent with Piper shipping usable sub-30M voices and Matcha's 18.2M beating 41.2M FastSpeech2 on MOS [REPORTED][6][29]. Size should be chosen by quality-per-parameter under pilot data, with 80M as the default midpoint where both NAR candidates have proven anchor points (Kokoro 82M; ZipVoice-class headroom to 123M+). [ESTIMATED]

---

## 11. Scored decision matrix

Scores 1–5 (5 = best). "CPU RTF outlook" = acoustic-only RTF at our sizes per Section 10; "Risk" scores *low* risk high.

| Architecture class | CPU RTF outlook | RAM | Streaming | Controllability | Small-data trainability | Risk | **Total /30** |
|---|---|---|---|---|---|---|---|
| **A. NAR deterministic mel, StyleTTS2/Kokoro lineage** | 5 | 5 | 3 | 4 | 5 | 5 | **27** |
| **C′. Compact OT-CFM mel (Matcha/ZipVoice-Distill), NFE≤8→2–4 distilled** | 4 | 5 | 3 | 4 | 4 | 4 | **24** |
| B-lowfps. AR codec LM @12.5 Hz (Mimi/CSM-style) | 4 | 4 | 5 | 4 | 1 | 2 | **20** |
| C″. Large DiT-FM zero-shot (F5-class 150–336M) | 2 | 3 | 3 | 5 (in-context cloning) | 2 | 3 | **18** |
| Hybrid: AR semantic tokens + chunked-FM local (CosyVoice-2-style) | 3 | 3 | 4 | 5 | 2 | 2 (complexity) | **19** |
| B-highfps. AR codec LM @75–195 Hz | 2 | 4 | 5 | 3 | 1 | 2 | **17** |
| D. Diffusion acoustic head | 2 | 3 | 2 | 3 | 2 | 2 | **14** |
| E. SSM/Mamba backbone swap | n/a (backbone) | 4 | 4 | — | 2 | 2 (tooling immaturity) | watchlist |

Justification anchors: A's RTF/RAM/small-data scores rest on Kokoro <100 h + CPU RTF 0.51–0.69 [REPORTED][12][28] and Piper on a Pi [REPORTED][29]; C′ rests on Matcha quality at NFE≤4 and ZipVoice-Distill CPU RTF 1.22 single-thread-equivalent [REPORTED][6][15]; B-lowfps loses only on data hunger (100k+ h class) and documented instability mitigations [REPORTED][19][24]; D is displaced (Section 6).

---

## 12. Top-2 recommendations for Phase F prototyping

### Candidate 1 (primary): "CAEDRAL-NAR" — Kokoro/StyleTTS2-lineage decoder-only NAR
- **Parameter budget (~82M inference-path, mirroring the proven Kokoro point):** CNN text encoder ≈5–8M; PL-BERT-style phoneme frontend frozen or lightly tuned ≈6M (or external); duration+prosody predictor ≈10–16M; optional style encoder ≈8–14M (drop for single-voice); iSTFTNet decoder ≈35–50M (upsample rates ~[10,6] @24 kHz, gen_istft hop 256). Ceiling 90M.
- **Why:** fastest CPU wall-clock of any credible quality point; explicit duration/F0 knobs; strongest small-data evidence (<100 h, $400 A100) [REPORTED][12]; ONNX-first deployment path already demonstrated; INT8 caution for conv-heavy parts noted. [REPORTED][28]
- **Risks:** expressiveness ceiling vs AR-LM; multi-voice consistency at small data unproven for us; Kokoro's synthetic-data asterisk means we must validate with licensed-only audio.

### Candidate 2 (parallel track): "CAEDRAL-CFM" — compact OT-CFM flow-matching mel
- **Parameter budget:** acoustic U-Net (1D conv residual blocks + light transformer, RoPE) 25–60M + Vocos-class vocoder ≈14M; target ≤75M total; train NFE 8–16 → distill to NFE 2–4 CFG-free (ZipVoice-Distill/FlashSpeech recipe) [REPORTED][15][16].
- **Why:** best quality-per-FLOP trajectory (Matcha MOS 3.84 @18M [REPORTED][6]); CPU story already measured at 123M (RTF 1.22 @NFE4) [REPORTED][15]; natural upgrade path to zero-shot cloning later (F5-small lineage exists at ~149M [REPORTED][8]); alignment-free MAS training removes external aligner dependency.
- **Risks:** vocoder adds a component to maintain; streaming requires CosyVoice-2-style chunked-CFM engineering (real-world first-chunk latency reports are mixed) [REPORTED][23].

**Watchlist:** Mimi-rate AR codec LM if the program pivots to full-duplex conversation; SSM backbones pending tooling maturity. **Rejected:** diffusion heads (Section 6).

### What Phase F must measure first
1. **Calibrate silicon constants on target hardware**: sustained batched GEMM GFLOPS (8 threads, T≈1000-frame batches) and single-stream GEMV bandwidth + fixed per-token overhead — validates/refutes Section 10 within its ±40% band before any model is trained. [ESTIMATED]
2. **End-to-end RTF of two reference implementations**: Matcha-18M at NFE {2,4,8} ±CFG vs Kokoro-ONNX-82M, same box, FP32 vs INT8 (Kokoro datapoint says INT8 may lose [REPORTED][28]).
3. **Small-data convergence race**: WER (ASR-scored intelligibility) after identical ≤100 h pilot budget on both candidates — tests the Section 9 ranking under CAEDRAL's licensing constraint.

---

## Sources

1. Défossez, Copet, Synnaeve, Adi — "High Fidelity Neural Audio Compression" (EnCodec), Meta AI, arXiv:2210.13438, Oct 2022. https://arxiv.org/abs/2210.13438 · https://github.com/facebookresearch/encodec
2. EnCodec frame-rate/bitrates summary — paper PDF (strides 2,4,5,8 ⇒ 75 Hz @24 kHz). https://arxiv.org/pdf/2210.13438
3. Défossez et al. — "Moshi: a speech-text foundation model for real-time dialogue" (Mimi codec, 12.5 Hz, 1.1 kbps, WavLM distillation, causal/80 ms), Kyutai, arXiv:2410.00037, Oct 2024. https://arxiv.org/abs/2410.00037 · https://kyutai.org/Moshi.pdf · https://github.com/kyutai-labs/moshi
4. Kyutai — "Neural audio codec explainer" (Mimi adoption by CSM, VoXtream, LFM2-Audio). https://kyutai.org/codec-explainer/
5. Siuzdak et al. — "SNAC: Multi-Scale Neural Audio Codec", arXiv:2410.14411, Oct 2024; snac_24khz card (12/23/47 Hz, 0.98 kbps, 19.8M params). https://arxiv.org/abs/2410.14411 · https://huggingface.co/hubertsiuzdak/snac_24khz
6. Mehta et al. — "Matcha-TTS: A fast TTS architecture with conditional flow matching" (18.2M, OT-CFM, NFE 2/4/10 table incl. FS2 41.2M & VITS 36.3M RTFs, RoPE U-Net, 13.9M vocoder separate), arXiv:2309.03199, Sep 2023. https://arxiv.org/html/2309.03199 · https://github.com/shivammehta25/Matcha-TTS
7. Chen et al. — "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" (336M DiT, NFE 16/32 RTF 0.15/0.31 on 3090; E2-TTS 333M NFE32 RTF 0.68; sway sampling; ConvNeXt-V2 refiner; Emilia 95–100k h), arXiv:2410.06885, ACL 2025. https://arxiv.org/abs/2410.06885 · https://aclanthology.org/2025.acl-long.313/
8. SWivid/F5-TTS — SHARED.md checkpoints (v1 Base config dim1024/depth22; Small config dim768/depth18/heads12; community Apache/CC-BY Small checkpoints). https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md
9. Eskimez et al. — "E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Text-to-Speech," Microsoft, arXiv:2406.18009, 2024 (params/NFE/RTF as reported in [7]'s comparison). https://arxiv.org/abs/2406.18009
10. Li et al. — "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models," NeurIPS 2023, arXiv:2306.07691 (config hidden 512, style 128, iSTFTNet decoder, WavLM SLM discriminator, two-stage training, LibriTTS ~245 h adaptation). https://arxiv.org/abs/2306.07691 · https://github.com/yl4579/StyleTTS2
11. StyleTTS2 GitHub issue #288 — community component parameter breakdown (~217M total incl. discriminators; ~147–161M inference path). https://github.com/yl4579/StyleTTS2/issues/288
12. hexgrad — Kokoro-82M model card (StyleTTS 2 + ISTFTNet, "decoder only: no diffusion, no encoder released"; v0.19 <100 h/$400, v1.0 few hundred hours/$1000 A100; Apache-2.0; permissive/synthetic data disclosure), Jan 2025. https://huggingface.co/hexgrad/Kokoro-82M
13. Kaneko et al. — "iSTFTNet: Fast and High-Quality Neural Vocoder with Inverse Short-Time Fourier Transform," CVPR 2022, arXiv:2203.02395. https://arxiv.org/abs/2203.02395
14. Siuzdak, Graczyk — "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis," arXiv:2306.00814, 2023. https://arxiv.org/abs/2306.00814
15. Zhu et al. (k2-fsa) — "ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching" (123M Zipformer; CPU RTF table: F5 37.284, ZipVoice 9.553, Distill 2.418 @8NFE / 1.220 @4NFE; flow distillation kills CFG), arXiv:2506.13053, Jun 2025. https://arxiv.org/abs/2506.13053 · https://github.com/k2-fsa/ZipVoice
16. Ye et al. — "FlashSpeech: Efficient Zero-Shot Speech Synthesis" (adversarial consistency training from scratch, NFE 1–2, ~5% prior inference time), arXiv:2404.14700. https://arxiv.org/abs/2404.14700 · https://flashspeech.github.io/
17. Kim et al. — "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," ICML 2021, arXiv:2106.06103 (LJSpeech training; params via [6]). https://arxiv.org/abs/2106.06103
18. Ren et al. — "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," arXiv:2006.04558, 2020 (params via [6]). https://arxiv.org/abs/2006.04558
19. Chen et al. — "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers" (Repetition-Aware Sampling; Grouped Code Modeling; not released), Microsoft, arXiv:2406.05370, Jun 2024. https://arxiv.org/abs/2406.05370
20. "GOAT: Great Milking GFlowNets for TTS Hallucination Mitigation"/GFlOwNet-guided distribution alignment, EMNLP 2025, arXiv:2508.15442 (>50% WER cut on hallucination subsets; CosyVoice2-RAS baseline). https://arxiv.org/abs/2508.15442
21. "Eliminating Stability Hallucinations in LLM-based TTS via Attention Guidance" (OAS/Viterbi metric), arXiv:2509.19852. https://arxiv.org/abs/2509.19852
22. Song et al. — "ELLA-V: Stable Neural Codec Language Modeling with Alignment-Guided Sequence Reordering," AAAI 2025, arXiv:2401.07333 (repetition/transposition/omission/infinite-silence failure taxonomy). https://arxiv.org/abs/2401.07333
23. Du et al. — "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models" (chunk-aware flow matching masks, NFE=10, β=0.7, ~150 ms claimed first packet; mel 50 Hz), arXiv:2412.10117, Dec 2024. https://arxiv.org/abs/2412.10117 · https://funaudiollm.github.io/cosyvoice2 · practical latency report: https://github.com/FunAudioLLM/CosyVoice/issues/367
24. Canopy Labs — Orpheus TTS (Llama-3.2-3B backbone, SNAC 24 kHz, 7 tokens/frame ≈82 tok/s, 100k+ h English, rep-penalty settings; planned 150M/400M/1B unreleased), 2025. https://huggingface.co/canopylabs/orpheus-3b-0.1-ft · https://unsloth/orpheus-3b GGUF cards
25. edwko/OuteTTS repo + llama.cpp PR #11287 (v0.3 500M/1B: Qwen2 arch, 24L/d896/14q/2kv heads; Q4_K_M 378.94 MiB; ~8192 tokens ≈ 42 s audio; WavTokenizer decoder in llama-tts). https://github.com/edwko/OuteTTS · https://github.com/ggml-org/llama.cpp/pull/11287 · https://github.com/ggml-org/llama.cpp/pull/10784
26. Sesame AI Labs — CSM-1B (Llama-3.2-1B backbone + Mimi 12.5 Hz, depth-AR audio head over 32 RVQ levels; ggml-org GGUF notes 96.1M audio head; hans00 split-runtime card: Mimi conv kernels don't quantize below Q8_0 well), Mar 2025. https://github.com/SesameAILabs/csm · https://huggingface.co/ggml-org/sesame-csm-1b-GGUF · https://huggingface.co/hans00/CSM-1B-GGUF
27. llama.cpp performance references: johannesgaessler performance page ("memory bandwidth is the dominant factor"); llama-bench README (7B Q4_0 4.05→16.71 t/s 1→8 threads); 200lz lab (Qwen2.5-0.5B Q4_K_M 14.2–47.9 t/s on Ryzen 7 5800H); runaiathome (i7-13700K ~95 t/s @Q4); mgroeber.de legacy-hardware benchmarks. https://johannesgaessler.github.io/llamacpp_performance · https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench · https://github.com/200lz/llm-inference-optimization-lab
28. Kokoro CPU benchmark reports: heyneo.com multi-model benchmark (Xeon 8272CL 4-core, Kokoro ONNX mean RTF 0.641, PyTorch 0.665); NemesisNet self-host report (Xeon E5620: FP16 169 MB RTF-equiv 1.4× faster-than-realtime scale, INT8 slower due to dequant); efemaer gist (32-vCPU c6a.8xlarge ≈5× realtime). https://heyneo.com/blog/kokoro-supertonic-inflect-nano-pocket-tts-cpu-benchmark
29. Rhasspy/Piper — VITS ONNX TTS (ARM-optimized; Pi 5 RTF ≈0.19, short-sentence 0.53 s; realtime medium voices on Pi 5); ktomanek/edge_tts_comparison benchmark; project moved to OHF-Voice/piper1-gpl (Oct 2025 read-only). https://github.com/rhasspy/piper · https://github.com/OHF-Voice/piper1-gpl · https://github.com/ktomanek/edge_tts_comparison
30. Lee et al. — "Efficient Streaming TTS Acoustic Model with Mamba Framework" (SMAM, 25–26M, RVQ MLM/INR decoding, CPU RTF ≈0.70, sub-70 ms latency), Interspeech 2025. https://www.isca-archive.org/interspeech_2025/lee25h_interspeech.pdf
31. "Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis," arXiv:2407.09732 (VALL-M comparisons; ~15 s efficiency crossover). https://arxiv.org/abs/2407.09732
32. "MamTra: Hybrid Mamba-Transformer Backbone for Speech Synthesis," arXiv:2603.12342, Mar 2026. https://arxiv.org/abs/2603.12342
33. Ainslie et al. — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245, 2023. https://arxiv.org/abs/2305.13245
34. Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864
35. Gu, Dao — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023. https://arxiv.org/abs/2312.00752
36. yl4579/PL-BERT — phoneme-level BERT used by Kokoro pipeline. https://github.com/yl4579/PL-BERT
37. ProsodyFlow ablation (flow matching vs diffusion CMOS −0.04), COLING 2025. https://aclanthology.org/2025.coling-main.518/
38. WavTTS — "studies have shifted toward flow matching" landscape statement, arXiv:2606.03455, Jun 2026. https://arxiv.org/abs/2606.03455
39. Qwen-Audio-3.0-TTS — production flow-matching acoustic renderer, arXiv:2607.23938, Jul 2026. https://arxiv.org/abs/2607.23938
40. BareWave — waveform-native flow-matching TTS, arXiv:2606.09048, Jun 2026. https://arxiv.org/abs/2606.09048

*End of document — Researcher C, 2026-08-22.*
