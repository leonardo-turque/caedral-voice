# VOCODERS & OUTPUT DECODERS for CPU-First TTS (Researcher D2)

Status: IN PROGRESS (skeleton — filling incrementally)
Date: 2026-08-22
Scope: highest-quality audio decoder compatible with strict CPU deployment (x86-64 AVX2/FMA, no GPU), for mel-based AND codec-token acoustic models.

## 0. Executive summary
- **Safest licensed high-quality picks (G1):** Vocos-class iSTFT-head decoder retrained in-house on Caedral voices (MIT code+weights precedent [8][9]), with iSTFTNet2 as the minimal-param recipe (0.79M params, 0.018 single-thread CPU RTF reported [13]). SNAC (MIT) is the license-clean codec-token fallback [25]. BigVGAN-v2 is now fully MIT including weights [3][4] — quality king but no credible CPU evidence.
- **Biggest licensing traps:** EnCodec pretrained weights CC-BY-NC 4.0 despite MIT code [17]; OpenVPI NSF-HiFiGAN weights CC-BY-NC-SA 4.0 (non-commercial AND share-alike) [5][6]; jik876 HiFi-GAN checkpoints carry NO stated license at all [2].
- **CPU RTF reality:** only the iSTFT-head family has controlled single-thread evidence (0.018-0.053 [13]); Vocos streaming measured 0.14 on CPU [11]; Kokoro proves whole-TTS with iSTFTNet head runs 1.8x real-time on CPU via ONNX [33]; BigVGAN/DAC/Mimi/SNAC/WavTokenizer have zero published CPU numbers.
- **Recommendation shape:** mel -> iSTFT head at 24 kHz for G1; codec-token path (SNAC or WavTokenizer) as G2 experiment; measure everything chunked in Phase G because batch RTF does not predict streaming throughput [29].

## 1. GAN vocoders

### 1.1 HiFi-GAN v1 [REPORTED]
- Architecture: ConvNeXt-ish stack of transposed-conv upsamplers + multi-receptive-field fusion (MPD + MSD discriminators). 22.05 kHz flagship; V100 RTF 167.9x real-time generation claimed in abstract [1]. Code repo `jik876/hifi-gan` is **MIT** (verified GitHub sidebar/LICENSE listing) [2].
- **Checkpoint licensing is an open trap**: README links pretrained models on Google Drive with NO explicit license stated anywhere in the README [2]. Treat official checkpoints as unlicensed-for-commercial unless written permission obtained. Community forks retrain under MIT code but each set of weights carries its own terms.
- CPU signal: the same paper reports the small-footprint variant runs 13.4x faster than real time **on CPU** [2] — one of very few vendor-published CPU RTF claims in this space. Params: full ~14M (V1), small variants down to ~0.9-2M depending on config.
- Derivatives: VITS/StyleTTS2 embed their own HiFi-GAN heads (usually MIT-inherited); quality ceiling lower than BigVGAN-v2 but far cheaper on CPU.

### 1.2 HiFi-GAN v2 / improved variants
- There is no single canonical "v2" open repo; the name circulates for (a) the improved discriminator/upsampler recipes inside commercial stacks (e.g., Google papers) and (b) community retunes. Practical consequence for Caedral: treat "HiFi-GAN v2" as a recipe family, not a licensed artifact. License verdict must be made per-implementation. [HYPOTHESIS - naming convention, verified absence of single canonical repo]

### 1.3 BigVGAN / BigVGAN-v2 — LICENSE RELICENSED TO MIT (verified)
- **Code**: `NVIDIA/BigVGAN` LICENSE file = MIT, "Copyright (c) 2024 NVIDIA CORPORATION" (read directly from raw LICENSE at main) [3].
- **Checkpoints**: HF card `nvidia/bigvgan_v2_24khz_100band_256x` lists **License: mit** [4]. The v1 checkpoints were historically CC-BY-NC-SA and were relicensed to MIT together with the v2 release (2024); current cards are MIT. Historical NC status matters only if you pin old snapshots/mirrors.
- Specs: v2 24 kHz = 100-band mel, fmax 12k, 112M params (256x variant), trained on large-scale compilation [4]; other v2 variants at 22.05k/44.16k/48k (40-band/128-band etc.). Anti-aliased Multi-Periodicity Composition (snake activation) gives strong periodicity modeling — top-tier MOS among open vocoders.
- CPU cost: heavy. GPU speed table shows 21.5-240x RTF on A100/3080/2080Ti only; no official CPU numbers [3][4]. 112M params with snake activations = expensive on AVX2/FMA (snake = x + sin^2(x)/a, adds transcendentals per element). [ESTIMATED] CPU RTF likely >> 1 for the 112M variant on a desktop core; the smaller 22k v1 (~14M) configs are borderline-viable. Phase G must measure.
- Fused CUDA kernel ("1.5-3x faster on A100") is GPU-only — irrelevant to Caedral CPU target [4].

### 1.4 NSF-HiFiGAN (source-filter HiFi-GAN) — COMMERCIAL TRAP CONFIRMED
- The widely-used pretrained NSF-HiFiGAN singing/tts vocoders released by the OpenVPI/DiffSinger community (`openvpi/vocoders`, `openvpi/SingingVocoders`) are **CC BY-NC-SA 4.0** — stated verbatim in every release notice ("Pretrained models are released under the Attribution-NonCommercial-ShareAlike 4.0 International license") and in bundled NOTICE.txt files [5][6]. Checkpoints: 44.1 kHz hop512 128-bin (2022-12-11), finetune 2024-02-19 (pitch range to C6), PC-NSF-HiFiGAN 2025-02-27.
- Distinction: Xin Wang's *code* repos at nii-yamagishilab (project-NN-Pytorch-scripts, TSNetVocoder) are BSD-3-Clause; pre-trained checkpoints hosted via those projects (Zenodo) may carry separate terms [7]. **Verdict: any NSF-HiFiGAN weights path requires either NC-safe use or full self-training from scratch on owned data using BSD/MIT code — training code is free, weights are not.**

### 1.5 Vocos — MIT, iSTFT head, best-documented CPU story
- Code + both official checkpoints are **MIT** (repo LICENSE "MIT License, Copyright (c) 2023 Charactr Inc."; HF cards `charactr/vocos-mel-24khz` and `charactr/vocos-encodec-24khz` confirm MIT) [8][9].
- Architecture: predicts STFT spectral coefficients and resynthesizes via **inverse STFT** — no upsampling transposed convs, single forward pass. Params: mel-24khz 13.5M; encodec-24khz 7.9M [9]. This is directly relevant twice over: it is BOTH a cheap mel decoder AND a published EnCodec-token decoder head.
- CPU numbers [REPORTED]:
  - Open Laboratory summary: ~169.6x real-time throughput on CPU => **CPU RTF ~= 0.006**; claims up to 13x faster than HiFi-GAN and ~70x faster than BigVGAN [10].
  - Community streaming/causal fork benchmark: **RTF 0.14 on CPU** (16 kHz streaming, 114 ms end-to-end latency), GPU RTF 0.035 [11]. Streaming chunking costs ~23x vs offline figure — expected overhead of causal/chunked inference.
  - Spread between 0.006 (offline, batched, possibly multi-core) and 0.14 (streaming fork, different SR/hardware) is wide; treat both as [REPORTED] third-party, verify in Phase G.

### 1.6 iSTFTNet / iSTFTNet2 — the iSTFT-head design pattern
- Original iSTFTNet (ICASSP 2022, NTT): replace transposed-conv upsampling with fewer upsamples + shared iSTFT head — cuts FLOPs massively at near-equal MOS [12].
- **iSTFTNet2** (Interspeech 2023, NTT; NOT 2024): 1D temporal CNN + 2D frequency CNN operating on freq-downsampled spectrograms. Vendor-published single-thread CPU benchmark on **Intel Core i7-12700H**, LJSpeech [13]:
  | Model | MOS | CPU RTF | Params |
  |---|---|---|---|
  | HiFi-GAN V2 (their baseline) | 4.20 | 0.053 | 0.93M |
  | iSTFTNet-C8C8I4 | 4.12 | 0.029 | 0.89M |
  | iSTFTNet2-Base | 4.24 | **0.021** | 0.85M |
  | iSTFTNet2-Small | 4.22 | **0.018** | 0.79M |
  => ~3x faster than a compact HiFi-GAN at equal-or-better MOS, sub-1M params. This is the strongest *controlled* CPU RTF evidence found anywhere in this survey [REPORTED].
- Implementation home: `kan-bayashi/ParallelWaveGAN` (also hosts original iSTFTNet); repo is MIT per GitHub labeling [14] — LICENSE text not read directly yet; verify before shipping.
- Naming note: the "HiFi-GAN v2" in these papers = the improved recipe inside kan-bayashi's ParallelWaveGAN ecosystem, resolving the ambiguous name from 1.2 for TTS contexts.
- CPU advantage rationale [ESTIMATED]: iSTFT heads move upsampling into the linear STFT overlap-add, which is FFT-based (O(N log N)) instead of transposed conv (O(N*k)); transcendentals avoided except final windowing — ideal for AVX2/FMA where sin/cos throughput is the bottleneck of snake-type activations.

### 1.7 HiFi-GAN derivative ecosystem (license-relevant)
- VITS, StyleTTS 2, and most open TTS stacks embed their own HiFi-GAN/iSTFTNet decoder heads trained end-to-end; when the surrounding stack is permissively licensed (e.g., Kokoro's released decoder, Apache-2.0 [33]), the head weights inherit that license. This is the practical route by which permissive decoder weights actually reach production — standalone vocoder checkpoints are far messier.
- Consequence for Caedral: prefer decoders trained inside Caedral's own training run (fully-owned weights under MIT/Apache code lineage) over any third-party checkpoint whose weights license is ambiguous. The only third-party checkpoints this survey found clean AND CPU-relevant are: Vocos mel/encodec (MIT [9]), SNAC (MIT [25]), WavTokenizer (MIT [26][27]), BigVGAN-v2 family (MIT [4]), DAC (MIT [20]), Mimi (CC-BY 4.0 [23]).

## 2. Neural codec decoders

### 2.1 EnCodec (Meta) — WEIGHTS ARE THE TRAP
- **Code**: `facebookresearch/encodec` was originally CC-BY-NC 4.0 and was relicensed to **MIT** on 2023-04-26 (commit 349b729, author adefossez) [15]. HF card/API metadata carries no license tag at all [16].
- **Weights**: released through `facebookresearch/audiocraft` under **CC-BY-NC 4.0**, stated verbatim: "The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the LICENSE_weights file" [17]. Community reading of issue #98 concurs: official checkpoints non-commercial; train-your-own with MIT code is the commercial path [18]. **Verdict: COMMERCIAL TRAP — do not ship facebook encodec_24khz/48khz weights in Caedral.**
- Specs [REPORTED]: causal 24 kHz mono model at 1.5/3/6/12/24 kbps with n_q = 2/4/8/16/32 codebooks; non-causal 48 kHz stereo music model at 3-24 kbps; 1-s chunks with 1% overlap for streaming; 24 kHz model frame rate 75 Hz (paper) [19].
- Artifact character [REPORTED]: at <=6 kbps known buzzy/metallic artifacts on speech; Multi-Band Diffusion exists as an expensive (diffusion) decoder alternative — not CPU-viable.

### 2.2 DAC / Descript Audio Codec — MIT INCLUDING WEIGHTS (verified)
- README states verbatim: "Weights are released as part of this repo under MIT license"; code repo MIT [20]. Checkpoints at **16, 24, 44.1 kHz**; 44.1 k default compresses to ~8 kbps ("~90x compression") [20].
- Architecture (paper: "High-Fidelity Audio Compression with Improved RVQGAN"): improved RVQGAN — periodic activation (snake), factorized/low-dim codebook projection, L2-normalized codes, random quantizer dropout training [21]. Transformers port `descript/dac_24khz` ~74.7M params F32 [22] (HF license field unfilled there — rely on GitHub statement).
- CPU cost signal [REPORTED]: no official RTF published anywhere found. Params comparable to EnCodec-class SEANet decoders; high frame rate (~86 Hz class at 44.1k, hop 512) makes token stream heavy for an LM front-end even if decode itself is fast. Good fidelity reputation in AudioLM community as EnCodec drop-in [REPORTED].

### 2.3 Mimi (Kyutai/Moshi) — CC-BY 4.0, streaming-native
- Weights at `kyutai/mimi` are **CC-BY 4.0** (README `cc-by-4.0`); Moshi repo: Python code MIT, Rust Apache-2.0, model weights CC-BY 4.0 [23]. Commercial-friendly (attribution + no share-alike trap).
- Specs: 24 kHz audio -> **12.5 Hz tokens, 1.1 kbps**, fully **streaming** with **80 ms frame latency**; RVQ of 16 codebooks (Transformers docs); first codebook semantically distilled from WavLM; adversarial-only (EBEN-style) training [23][24].
- CPU cost concern [HYPOTHESIS/ESTIMATED]: Mimi embeds small Transformers in BOTH encoder and decoder (to mix semantic+acoustic info). Sequential attention over frames raises per-frame latency vs pure-conv decoders; at only 12.5 fps the per-frame compute is amortized, so net CPU cost may still be modest — must be measured in Phase G. Token rate 12.5/s is the cheapest LM front-end of all codecs surveyed (fewer tokens = cheaper acoustic model).

### 2.4 SNAC — MIT, multi-scale, TTS-proven
- `hubertsiuzdak/snac_24khz` card lists **License: mit** [25]. Variants: 24 kHz/0.98 kbps/19.8M params (speech), 32 kHz & 44 kHz/54.5M params (music/SFX). Mono only.
- Token structure: hierarchical multi-scale RVQ — coarse tokens at **12 Hz**, mid **23 Hz**, fine **47 Hz** (shapes like [12,24,48] per second) [25]. Multi-scale lets an LM predict cheap coarse structure + cheap fine refinement instead of flat 9-codebook stacks.
- Deployment evidence [REPORTED]: card notes use in 100+ HF Spaces incl. Orpheus-TTS and mini-omni — real open-source TTS deployments run SNAC decoding on commodity CPUs daily; strongest field-proven codec-token path for speech TTS after Vocos/encodec pairing.
- Card states "primarily trained on speech data... recommended use case is speech synthesis" [25].

### 2.5 WavTokenizer — MIT, 40 tok/s, iSTFT head (Vocos-style)
- GitHub `jishengpeng/WavTokenizer` MIT (~1.3k stars, ICLR 2025); HF `novateur/WavTokenizer` weights open-source [26][27]. Canonical speech checkpoint: **small-600-24k-4096** (LibriTTS, 40 tok/s, single codebook 4096) [26].
- Architecture: EnCodec-style SEANet encoder + single-codebook VQ + **Vocos-style decoder with iSTFT head** — the swiss-ai Transformers conversion lists ~76.3M params for large-unify; small variant far lighter [28].
- Why it matters: it is a codec-token interface whose DECODER is literally the cheapest mel-decoder family (iSTFT head). Single codebook at 40 tok/s = minimal LM-side sequence length. Caveats [REPORTED]: quality concentrated in speech domain (LibriTTS-trained small model); medium/large checkpoints were "coming soon" as of card writing — verify current availability; some community reports note instability/artifacts vs Mimi/SNAC at equal rates [REPORTED - weaker sourcing, verify].

### 2.6 Decoder-side CPU cost ranking [ESTIMATED]
Cheap->expensive, reasoning shown:
1. Vocos-encodec head (7.9M, conv + iSTFT) — smallest.
2. SNAC 19.8M conv decoder, but 47+12 Hz multi-rate streams are trivially cheap.
3. WavTokenizer small (SEANet+iSTFT; small variant est. <15M decoder).
4. EnCodec/DAC 24k SEANet-class decoders (~50-75M full stack, decoder share roughly half).
5. Mimi (conv + decoder Transformer; 12.5 fps amortizes, unknown net).
6. Anything diffusion-based (MBD) — disqualified for CPU G1.
All figures pending Phase G measurement; parameter counts cited above where available.

### 2.7 Artifact character (decoder-side) [REPORTED + HYPOTHESIS]
- EnCodec-class at <=6 kbps on speech: buzzy/metallic artifacts, smearing of sibilants — widely reported in AudioLM community; Multi-Band Diffusion exists specifically as an artifact-repair decoder, which itself signals the baseline decoder's ceiling [19].
- SNAC multi-scale: coarse 12 Hz tokens carry prosody, fine 47 Hz tokens carry periphery; artifact profile reportedly milder than flat RVQ at similar total rate because bandwidth is spent hierarchically [REPORTED - card + adoption evidence, no formal MOS found].
- Mimi: adversarial-only (EBEN-style) training tends to produce "natural-but-slightly-different" timbre rather than metallic artifacts; WavLM distillation anchors phonetic content [REPORTED from architecture description]. Risk for a multi-voice brand: EBEN-style decoders can subtly shift speaker identity — must be ABX-tested on Caedral voices [HYPOTHESIS].
- WavTokenizer single-codebook: quantization bottleneck concentrated in one 4096-entry codebook; speech-domain checkpoints (LibriTTS) may degrade on out-of-domain voices/music [REPORTED - training data from card; degradation HYPOTHESIS].
- GAN mel decoders (Vocos/iSTFTNet2): artifacts manifest as occasional pitch-period errors on low F0 and phase discontinuities at chunk boundaries when streaming — the chunk-boundary issue is the one Caedral must engineer against (windowing/overlap policy) [HYPOTHESIS grounded in 3.1 padding-artifact evidence].

## 3. Streaming vocoder techniques / TTFA implications

### 3.1 What actually breaks in naive streaming [REPORTED]
- Fast *batch* inference does NOT predict streaming throughput: limited parallelism per short chunk, inter-frame dependency management, and parameter-loading overhead dominate — shown comparatively by Yoneyama et al., Interspeech 2025 ("Comparative Analysis of Fast and High-Fidelity Neural Vocoders for Low-Latency Streaming Synthesis") [29]. Any Phase G benchmark must measure CHUNKED inference, not whole-utterance RTF.
- Symmetric/zero padding in non-causal convs causes spectral distortion when naively chunked; transposed convs cause checkerboard artifacts. Fixes seen in the wild: causal convolutions with left-only padding + pixel-shuffle upsampling ("Causal Shuffle Vocoder", Conan system) [30].

### 3.2 Lookahead budgets [REPORTED]
- MS-Wavehax (multi-stream Wavehax): **single-frame lookahead nearly matches non-causal quality** while achieving the highest throughput under <80 ms latency conditions, at only 2.4% of HiFi-GAN V1's model size; measured with ONNX Runtime on AMD EPYC (CPU) [29]. Chunk-size sweep: chunk 8 adds ~70 ms extra latency vs chunk 1 [29].
- StreamingVocos (community replication of Vocos with all-causal CNNs, adjustable chunk = k mel frames, hop 20 ms @16 kHz): chunk=5 (100 ms buffer) -> CPU ~14 ms processing/chunk, **114 ms end-to-end latency, RTF 0.14**; GPU <25 ms E2E at chunk=1 [11]. This gives an empirical lookahead->latency curve for one iSTFT-head decoder.
- Conan Causal Shuffle Vocoder: fast setting 37 ms total latency (20 ms chunks); full setting 140 ms (80 ms chunks + 2 right-context chunks); vocoder module alone ~6 ms on A100 [30]. Right-context frames are exactly the "lookahead budget" knob.
- EnCodec-style codecs: causal mode exists for 24 kHz model; official streaming uses 1-s chunks with 1% overlap [19] — fine for file decode, NOT for conversational TTFA without re-chunking work.

### 3.3 TTFA implications for Caedral [ESTIMATED]
- TTFA (time to first audio) ~= acoustic-model first-frame latency + vocoder buffering delay + vocoder compute of first chunk. For an iSTFT-head decoder at hop 256/24 kHz (10.7 ms/frame), a 4-frame lookahead + 8-frame chunk costs ~128 ms of buffer before synthesis even starts; single-frame lookahead cuts that to ~32 ms at modest quality cost (per MS-Wavehax evidence).
- Practical budget proposal [HYPOTHESIS]: 1 frame lookahead + chunk 4-8, iSTFT head -> vocoder-side TTFA contribution 50-120 ms on desktop AVX2 hardware, consistent with StreamingVocos's measured 114 ms E2E on CPU.

### 3.4 Concrete chunking math at Caedral G1 settings [ESTIMATED]
Assumptions: 24 kHz, mel hop 256 (94 Hz frames, 10.7 ms/frame), iSTFT head with window 1024 / overlap-add.
| Lookahead | Chunk | Buffering delay | First-chunk compute (at RTF 0.05) | Vocoder TTFA share |
|---|---|---|---|---|
| 0 | 1 frame | 10.7 ms | ~0.5 ms | ~11 ms |
| 1 | 1 frame | 21.4 ms | ~0.5 ms | ~22 ms |
| 1 | 4 frames | 53.5 ms | ~2 ms | ~56 ms |
| 2 | 8 frames | 107 ms | ~4 ms | ~111 ms |
| EnCodec-official-style | 1 s @1% overlap | ~1000 ms | ~50 ms | ~1050 ms |
Takeaways: (a) the vocoder is never the compute bottleneck for TTFA — buffering policy is; (b) official-codec "streaming" modes built around ~1 s chunks are unusable for conversational TTFA and force custom re-chunked decode wrappers; (c) single-frame lookahead costs almost nothing in buffer time and buys near-non-causal quality per [29], making it the default Phase G configuration to try first.

## 4. Sample-rate scaling cost (22.05k vs 24k vs 44.1k) [ESTIMATED]

Reasoning chain (no single source isolates SR scaling; this is derived):
- Decoder compute scales roughly linearly in output samples for conv layers (FLOPs ∝ samples × channels × kernel) => 44.1k is ~2.0x the per-second FLOPs of 22.05k, ~1.84x that of 24k, BEFORE architecture changes.
- Hop sizes are set relative to frame rate (typically 200-600 samples): 22.05k@hop256 = 86 Hz frames; 24k@hop256 = 94 Hz; 44.1k@hop512 = 86 Hz. Same frame rate means same number of LM/acoustic tokens; the multiplier hits only the decoder's per-frame work.
- Real architectures do NOT scale purely linearly because high-SR models add sub-band splits or deeper stacks: NSF-HiFiGAN 44.1k community checkpoints pair with hop512/128-bin mels (bigger mel input too); DAC 44.1k targets ~8 kbps at 90x compression. Empirical anchors from the literature: HiFi-GAN V1 small config = 13.4x real-time on CPU at 22.05k [2]; iSTFTNet2 = 0.018 RTF single-thread at 22.05k [13]; StreamingVocos 0.14 RTF streaming at 16k [11].
- Conclusion [ESTIMATED]: staying at 24 kHz buys ~1.9x cheaper decoding than 44.1k at equal architecture, and 44.1k additionally forces larger mels/hops or sub-band designs. For Caedral G1 (CPU-first), 24 kHz mono is the correct target; treat 44.1k as a G2 "premium tier" via separate checkpoint, not a shared stack. 22.05k vs 24k is effectively neutral (~10% delta) — choose based on training data, not CPU fear.

Concrete multiplier table at equal architecture [ESTIMATED]:
| Target SR | vs 24 kHz decode cost | Typical hop / frame rate | Notes |
|---|---|---|---|
| 16 kHz | ~0.67x | 256 -> 62.5 Hz | StreamingVocos evidence point [11]; telephony band only |
| 22.05 kHz | ~0.92x | 256 -> 86 Hz | HiFi-GAN/iSTFTNet2 literature default [2][13] |
| 24 kHz | 1.00x (baseline) | 256 -> 94 Hz | Codec ecosystem native rate (EnCodec/DAC/Mimi/SNAC/WavTokenizer all 24k-capable) |
| 44.1 kHz | ~1.84x | 512 -> 86 Hz | NSF-HiFiGAN/OpenVPI and DAC 44.1 territory; needs bigger mel or sub-band design |
| 48 kHz | ~2.00x | varies | EnCodec 48k stereo music model only; no speech-TTS advantage |
Interpretation: SR choice is really an ecosystem choice. 24 kHz sits exactly on the codec world's common denominator AND on modern TTS training-data norms, while costing half of 44.1k. The only reason to leave it is a premium fidelity SKU with its own checkpoint line.

## 5. Reported/community CPU RTF numbers [all REPORTED unless noted]

Vocoder-only:
| System | CPU RTF | Hardware / conditions | Source |
|---|---|---|---|
| Vocos (offline) | ~0.006 (~169.6x RT) | Open Laboratory summary, conditions unstated | [10] |
| StreamingVocos (streaming) | 0.14 | CPU, chunk=5 frames, 16 kHz | [11] |
| iSTFTNet2-Small | **0.018** | single thread, i7-12700H, LJSpeech 22.05k | [13] |
| iSTFTNet2-Base | 0.021 | same | [13] |
| HiFi-GAN V2 (kan-bayashi baseline) | 0.053 | same | [13] |
| HiFi-GAN small config | "13.4x faster than real time" (RTF~0.075) | vendor paper, Intel Core i7 2.6 GHz | [2] |
| ParallelWaveGAN (WaveRNN) | 1.6 -> 0.46 (residual layers cut 30->10); 0.734 on 16-thread Xeon Gold 6154 | community issue #28 | [31] |
| Subband WaveRNN | 0.064 single-threaded | i7-8750H @2.2 GHz, Interspeech 2022 | [32] |
| BigVGAN v1/v2 | none published for CPU | GPU-only official tables | [3][4] |

Pipeline-level (acoustic model + vocoder together):
| System | CPU RTF | Notes | Source |
|---|---|---|---|
| Kokoro-82M (StyleTTS2+iSTFTNet head), ONNX | 0.571 | full TTS incl. decoder; PyTorch 0.787; Apache-2.0 weights | [33] |
| Inflect-Nano-v1 | 0.138 | 4.6M params, lower MOS 3.48 UTMOS | [33] |
| Supertonic-3 (5-step) | 0.316 | ~99M, diffusion-steps tradeoff | [33] |
| Matcha-TTS MAT-2 | 0.015 incl. vocoding | hardware per paper (GPU-class); shows flow-matching step count dominates | [34] |

Reading of the table: the iSTFT-head family is the only one with controlled single-thread evidence at RTF<=0.02; GAN transposed-conv decoders cluster 0.05-0.15 streaming; heavy snake-activation models (BigVGAN-v2 112M) have no credible CPU story at all. Kokoro proves an entire 82M mel-TTS with iSTFT head runs at 1.8x real-time on ordinary CPUs via ONNX — a ceiling Caedral should beat comfortably given a smaller acoustic model.

### 5.1 CPU runtime/deployment notes for AVX2/FMA
- Evidence base: every credible CPU number in this survey came through either ONNX Runtime (MS-Wavehax streaming study [29], Kokoro benchmark [33]) or plain PyTorch CPU (StreamingVocos [11]). No vendor publishes AVX2-specific kernels; the fused CUDA kernel class (BigVGAN [4]) has no CPU analogue.
- Op-level concerns [ESTIMATED]: (a) snake/sin activations in BigVGAN/DAC cost transcendentals per element — on AVX2, scalar sin/cos throughput can dominate; (b) transposed convs usually lower to im2col+GEMM (good for FMA) but with poor arithmetic intensity at small batch=1; (c) iSTFT heads end in FFT + overlap-add, which is memory-bound and cheap; (d) Conv1D with kernel 3 and dilations exports cleanly to ONNX; pixel-shuffle upsampling (Conan [30]) also lowers to cheap reshapes.
- INT8 quantization of vocoder decoders is largely unexplored publicly [REPORTED - absence]; GAN decoders are known to be quality-sensitive to quantization in community reports [REPORTED - weak]. Plan Phase G as FP32 first; treat INT8 as an optimization experiment, not a dependency.
- Threading: vocoder decode at chunk sizes of tens of ms has little intra-op parallelism per call; expect single-core performance to dominate TTFA, with threading useful only for offline/throughput modes — consistent with iSTFTNet2 publishing single-thread numbers as its headline [13] and MS-Wavehax's finding that batch speed != streaming speed [29].

## 6. Mel-output vs codec-token decoder decision inputs

For mel-based acoustic model:
- Pros: mature tooling; decoder quality ceiling highest (BigVGAN-v2 MIT if CPU budget allows; Vocos/iSTFTNet2 otherwise); no quantization floor — artifacts are fully under decoder's control; phase handled implicitly or via iSTFT prediction.
- Cons: mel must be computed from predicted features anyway (cheap); but the LM/acoustic side must model continuous features (regression/flow/diffusion) rather than discrete tokens — this cost lives upstream of D2 scope.
- Field proof: Kokoro-82M ships exactly this (mel -> iSTFTNet magnitude+phase head, no separate vocoder) at Apache-2.0 and runs 1.8x RT on CPU [33]. Matcha-TTS likewise uses lightweight decoder + separate vocoding at negligible added RTF [34].

For codec-token acoustic model:
- Pros: discrete tokens suit autoregressive LMs (Caedral multi-voice roadmap); tiny token rates available (Mimi 12.5 Hz, WavTokenizer 40/s, SNAC coarse 12 Hz) shrink sequence length dramatically vs waveform/mel frames (86-94 Hz).
- Cons: quantization ceiling — voice timbre ("proprietary multi-voice" brand asset!) is partially destroyed by codebooks trained on other speakers' data; artifact character (buzzy/metallic at low kbps for EnCodec-class [19]) is audible on clean studio speech even when inaudible on noisy data; retraining the codec on owned voices = extra GPU training stage that a mel path avoids entirely.
- License-clean token paths verified here: SNAC (MIT) [25], WavTokenizer (MIT) [26][27], Mimi (CC-BY 4.0) [23], DAC (MIT) [20], Vocos-encodec head (MIT, but pairs with NC EnCodec tokens unless you train your own codec) [9].
- Hybrid option [HYPOTHESIS]: WavTokenizer-style design — single codebook 40 tok/s + Vocos/iSTFT decoder — gives LM-friendly discreteness while keeping decoder CPU cost at mel-path level; its main risk is speech-domain-only training coverage of open checkpoints.

Decision input summary: choose mel+iSTFT-head for G1 (quality ceiling + zero extra training stages + best CPU evidence); keep codec-token path as G2 experiment gated on Phase G measurements of SNAC/WavTokenizer decode cost and a proprietary-voice codec fine-tune budget.

### 6.1 Quick-reference decision table

| Criterion | Mel -> iSTFT-head (Vocos/iSTFTNet2 class) | Codec tokens -> conv decoder (SNAC/DAC) | Codec tokens -> streaming codec (Mimi) |
|---|---|---|---|
| Quality ceiling on owned voices | highest (no quantization floor) | bounded by codebook fit; needs own fine-tune for brand voices [HYPOTHESIS] | good at 1.1 kbps but timbre-shift risk (EBEN) [HYPOTHESIS] |
| LM-side sequence cost upstream | none (continuous features) | 12-47 tok/s multi-rate — very cheap | 12.5 tok/s — cheapest flat rate |
| Decoder CPU evidence | best (0.006-0.14 RTF reported [10][11][13]) | none published; params suggest cheap [ESTIMATED] | unknown (Transformer in decoder) |
| Streaming maturity | proven via causal forks [11], single-frame lookahead results [29] | unproven publicly for chunked decode | native, 80 ms frames [23] |
| Extra training stages needed | 1 (decoder with acoustic model) | 2 (+ codec fine-tune on Caedral voices) | 2 (same) |
| License-clean today | yes (MIT) | yes (MIT) | yes (CC-BY 4.0 weights) |

## 7. Comparison matrix

Legend: quality = community/MOS signal; RTF signal = best CPU evidence found (offline unless noted). License shown as code / weights.

| Decoder | Family | Quality signal | CPU RTF signal | Params | SR (native) | License code / weights | Streaming-friendliness |
|---|---|---|---|---|---|---|---|
| HiFi-GAN V1 | GAN transposed-conv | high (canonical baseline) | small cfg 13.4x RT [REPORTED]; full cfg fails RT on weak cores [31] | 0.9M small / ~14M full | 22.05k | MIT / **UNSTATED (Google Drive)** [2] | poor: non-causal, padding artifacts when chunked [30] |
| BigVGAN-v2 | GAN + snake act. | top open MOS | none published; [ESTIMATED] RTF >> 1 at 112M on desktop CPU | 112M (24k 256x) | 22.05/24/44.16/48k | **MIT / MIT** (2024 relicense, verified) [3][4] | poor; CUDA kernel GPU-only |
| Vocos (mel) | iSTFT head GAN | near-HiFiGAN | **0.006 offline / 0.14 streaming** [10][11] | 13.5M | 24k | MIT / MIT [8][9] | good: causal forks, chunked iSTFT, 114 ms E2E CPU [11] |
| Vocos (EnCodec head) | iSTFT head | good (codec repair) | same family | 7.9M | 24k | MIT / MIT — but EnCodec tokens NC [9][17] | good (same) |
| iSTFTNet2 | iSTFT head GAN | MOS 4.22-4.24 vs 4.20 baseline | **0.018-0.021 single-thread** [13] | 0.79-0.85M | 22.05k | MIT (repo labeling; verify text) / train-your-own [14] | good pattern; causalizable [ESTIMATED] |
| NSF-HiFiGAN (OpenVPI) | source-filter GAN | singing-grade F0 control | none found | n/a | 44.1k | BSD-3 code / **CC-BY-NC-SA weights** [5][6][7] | poor |
| EnCodec 24k | SEANet RVQ codec | artifacts <=6 kbps [REPORTED] | none published; SEANet-class [ESTIMATED] | ~50-75M class | 24k/48k | MIT / **CC-BY-NC 4.0 weights** [15][17] | causal 24k model exists; official streaming = 1 s chunks [19] |
| DAC | improved RVQGAN | high fidelity, AudioLM drop-in | none found | ~74.7M (24k port) [22] | 16/24/44.1k | **MIT / MIT** (verbatim) [20] | non-causal primary |
| Mimi | conv+Transformer codec | SOTA at 1.1 kbps speech | none; [ESTIMATED] modest (12.5 fps amortization) vs Transformer risk | n/a (unverified) | 24k | MIT+Apache code / **CC-BY 4.0 weights** [23] | **excellent**: streaming-native, 80 ms frame [23] |
| SNAC | multi-scale RVQ | good speech (Orpheus-TTS field use) [25] | none direct; 19.8M conv [ESTIMATED] cheap | 19.8M (24k) | 24/32/44k | **MIT / MIT** [25] | good: multi-rate tokens, conv decoder |
| WavTokenizer small-600 | SEANet enc + Vocos-style iSTFT dec | good speech at 40 tok/s [26] | decoder = iSTFT family [ESTIMATED] cheap | large 76.3M [28]; small <15M [ESTIMATED] | 24k | MIT / MIT [26][27] | good (iSTFT head, causalizable) |

## 8. Recommended Caedral G1 candidates + Phase G falsification experiments

### 8.1 Candidates (ranked)
1. **Vocos `charactr/vocos-mel-24khz` as reference decoder; retrain in-house on Caedral voices** — MIT code + weights; best-documented CPU story (0.006 offline / 0.14 streaming [10][11]); 13.5M params; 24 kHz matches G1 target. License verdict: **SHIP-SAFE** (MIT both; retraining under MIT code produces fully-owned weights).
2. **iSTFTNet2-style head trained from scratch (kan-bayashi/ParallelWaveGAN lineage, MIT)** — sub-1M params, single-thread RTF 0.018 evidence [13], MOS at parity with compact HiFi-GAN; integrates as decoder-only head inside Caedral's own acoustic model (the Kokoro pattern, Apache-2.0 precedent [33]). License verdict: **SHIP-SAFE** (verify LICENSE text at pin time).
3. **SNAC 24 kHz if/when codec-token path is activated** — MIT weights, speech-trained, multi-rate tokens (12/23/47 Hz) shrink LM sequence length, field-proven in Orpheus-TTS [25]. License verdict: **SHIP-SAFE**. Runner-up: Mimi (CC-BY 4.0, streaming-native 12.5 Hz) if ultra-low token rate matters more than conv-only decode [23].

Avoid for commercial G1: EnCodec pretrained weights (CC-BY-NC [17]); OpenVPI NSF-HiFiGAN weights (CC-BY-NC-SA [5][6]); jik876 HiFi-GAN Google Drive checkpoints (no stated license [2]). BigVGAN-v2 is license-clean (MIT) but CPU-unproven — premium/offline tier candidate only, pending measurement.

### 8.2 Phase G falsification experiments
1. **RTF sweep (falsifies the iSTFT-head ranking):** ONNX Runtime, AVX2/FMA pinned, single-thread + 4-thread, on the target desktop CPU class: vocos-mel-24khz, snac_24khz decoder-only, WavTokenizer-small-600 decoder-only, bigvgan_v2_24khz_100band_256x, HiFi-GAN small. Offline RTF + chunked RTF at chunk {1,4,8} frames x lookahead {0,1,2}. Falsification: if SNAC or WavTokenizer decode beats Vocos, or BigVGAN-v2 streams at RTF<0.3, the G1 recommendation changes.
2. **TTFA budget test (falsifies 3.3):** causalized-Vocos 1-frame-lookahead vs chunked non-causal; measure vocoder-only first-audio contribution; hypothesis <150 ms on desktop AVX2. Fail => revisit chunk/lookahead budget or move to Mimi-style streaming-native codec.
3. **Mimi decode cost (falsifies 2.6 ranking):** decode 60 s stream via Transformers vs Kyutai Rust stack; if per-chunk p95 < 30 ms on target CPU, Mimi jumps to candidate 3.
4. **Quality gate on proprietary voices:** UTMOS + internal ABX at 24 kHz: Vocos(retrained) vs iSTFTNet2-style(in-house) vs SNAC(own fine-tune). If iSTFTNet2-class head loses >=0.2 MOS to Vocos-class, pay its 13.5M-param cost; else ship the smaller head.
5. **License pinning audit (process, not experiment):** record exact commit hashes for every adopted repo; re-verify LICENSE/LICENSE_weights files at pin time. Precedent: BigVGAN relicensed NC->MIT (2024) and EnCodec code NC->MIT (2023-04-26) while weights stayed NC [3][15][17] — licenses move in both directions, and old mirrors may still serve NC-era artifacts.

## 9. Known unknowns & risks
- No official CPU benchmark exists for BigVGAN-v2, DAC, Mimi, SNAC, WavTokenizer decoders — every codec-decoder CPU number in this document is an estimate; Phase G exp. 1 is the only way to settle the ranking.
- Mimi decoder parameter count/structure detail not extracted from sources read; its Transformers implementation may differ from Kyutai's Rust runtime in speed.
- WavTokenizer medium/large checkpoint availability has shifted over time ("coming soon" at card writing); small-600 speech model is the stable artifact.
- jik876 HiFi-GAN Google Drive checkpoints have no stated license — using them commercially is legally undefined even though the code is MIT. Do not build on them.
- License drift risk is real and bidirectional (BigVGAN NC->MIT 2024; EnCodec code NC->MIT 2023 while EnCodec weights stayed NC): pin commits and re-read LICENSE files at each release cut.
- Subband WaveRNN datapoint [32] came from search aggregation without a captured URL — treat as directional only.
- Community Vocos "13x faster than HiFi-GAN / 70x faster than BigVGAN" figures [10] are marketing-grade summaries without hardware disclosure; the defensible numbers are iSTFTNet2's controlled table [13] and StreamingVocos's methodology [11].
- CC-BY 4.0 (Mimi) has no share-alike clause but requires attribution; decide whether Kyutai attribution in product docs is acceptable brand-wise before committing.
- CC-BY-NC-SA (OpenVPI NSF-HiFiGAN) is doubly toxic for Caedral: non-commercial AND share-alike would force derivative weights open.

## Sources

| # | URL | Title | Project | Date | License (as verified) |
|---|---|---|---|---|---|
| [1] | https://arxiv.org/abs/2010.05601 | HiFi-GAN: Generative Adversarial Networks for Efficient and Synthesizer-Conditional High Fidelity Speech Synthesis | HiFi-GAN (Kong et al., NeurIPS 2020) | 2020-10 | paper (code MIT via repo) |
| [2] | https://github.com/jik876/hifi-gan | jik876/hifi-gan (README: V100 167.9x; small cfg 13.4x RT on CPU; checkpoints on Google Drive, no license stated) | HiFi-GAN official | 2020 | MIT (code only) |
| [3] | https://github.com/NVIDIA/BigVGAN (+ LICENSE at main) | NVIDIA/BigVGAN; LICENSE = "MIT License, Copyright (c) 2024 NVIDIA CORPORATION" | BigVGAN/BigVGAN-v2 | 2024 (relicense) | MIT |
| [4] | https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x | bigvgan_v2_24khz_100band_256x card (License: mit; 24k/100band/112M) | BigVGAN-v2 | 2024 | MIT (weights) |
| [5] | https://openvpi.github.io/vocoders/ | OpenVPI DiffSinger Community Vocoder releases (NSF-HiFiGAN 2022-12-11, 2024-02-19; PC-NSF 2025-02-27) | OpenVPI | 2022-2025 | CC-BY-NC-SA 4.0 (weights) |
| [6] | https://github.com/openvpi/vocoders/releases | openvpi/vocoders release notices ("Pretrained models are released under the Attribution-NonCommercial-ShareAlike 4.0 International license") | OpenVPI | 2022-2025 | CC-BY-NC-SA 4.0 (weights) |
| [7] | https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts | Xin Wang NSF/NSF-HiFiGAN code (BSD-3-Clause); pretrained vocoders via Zenodo carry own terms | nii-yamagishilab | ongoing | BSD-3-Clause (code) |
| [8] | https://github.com/gemelo-ai/vocos (+ LICENSE) | gemelo-ai/vocos; LICENSE = "MIT License, Copyright (c) 2023 Charactr Inc." | Vocos (Charactr) | 2023-06 | MIT |
| [9] | https://huggingface.co/charactr/vocos-mel-24khz | vocos-mel-24khz card (MIT; 13.5M, LibriTTS 1M iters); sibling vocos-encodec-24khz (7.9M, DNS Challenge) | Vocos | 2023 | MIT (code + weights) |
| [10] | https://openlaboratory.com/models/vocos/ | Open Laboratory model summary (~169.6x RT on CPU => RTF ~0.006; 13x faster than HiFi-GAN, ~70x than BigVGAN) | Vocos | n.d. | n/a (third-party benchmark) |
| [11] | https://github.com/warisqr007/vocos | StreamingVocos (causal CNN fork; CPU RTF 0.14, 114 ms E2E @16k, chunk=5) | community | 2024-2025 | inherits MIT (verify) |
| [12] | https://arxiv.org/abs/2203.02395 | iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform | iSTFTNet (NTT, ICASSP 2022) | 2022-03 | paper |
| [13] | https://arxiv.org/abs/2308.07117 (+ https://www.isca-archive.org/interspeech_2023/kaneko23_interspeech.html) | iSTFTNet2: Faster and More Lightweight iSTFT-Based Neural Vocoder Using 1D-2D CNN (single-thread i7-12700H RTF table) | iSTFTNet2 (NTT, Interspeech 2023) | 2023-08 | paper |
| [14] | https://github.com/kan-bayashi/ParallelWaveGAN | ParallelWaveGAN (hosts iSTFTNet + HiFi-GAN v1/v2 configs) | kan-bayashi | ongoing | MIT (repo labeling; verify LICENSE text) |
| [15] | https://github.com/facebookresearch/encodec (relicense commit 349b729, 2023-04-26) | facebookresearch/encodec (CC-BY-NC 4.0 -> MIT for code) | EnCodec (Meta) | 2023-04 | MIT (code) |
| [16] | https://huggingface.co/facebook/encodec_24khz (+ /api/models/facebook/encodec_24khz) | EnCodec 24k card — no license field/tag anywhere | EnCodec | 2022-2023 | UNSTATED on HF (weights NC via [17]) |
| [17] | https://github.com/facebookresearch/audiocraft (LICENSE_weights) | audiocraft: "models weights ... released under the CC-BY-NC 4.0 license as found in the LICENSE_weights file" | AudioCraft (EnCodec/MusicGen/MBD weights) | 2023 | CC-BY-NC 4.0 (weights) |
| [18] | https://github.com/facebookresearch/audiocraft/issues/98 | Issue #98 "LICENSE_WEIGHTS [CC-BY-NC 4.0]" — community interpretation | AudioCraft | 2023 | n/a (discussion) |
| [19] | https://arxiv.org/abs/2210.13438 | High Fidelity Neural Audio Compression (EnCodec paper: 24k causal model, 1.5-24 kbps, 75 Hz frames; 1-s chunks, 1% overlap) | EnCodec (Meta) | 2022-10 | paper |
| [20] | https://github.com/descriptinc/descript-audio-codec | Descript Audio Codec — "Weights are released as part of this repo under MIT license"; 16/24/44.1 kHz, ~8 kbps, ~90x compression | DAC (Descript) | 2023 | MIT (code + weights, verbatim) |
| [21] | https://arxiv.org/abs/2306.06546 | High-Fidelity Audio Compression with Improved RVQGAN | DAC paper | 2023-06 | paper |
| [22] | https://huggingface.co/descript/dac_24khz | dac_24khz Transformers port (74.7M F32; license field unfilled "[More Information Needed]") | DAC port | 2024 | rely on [20] |
| [23] | https://huggingface.co/kyutai/mimi + https://github.com/kyutai-labs/moshi | Mimi card (cc-by-4.0; 12.5 Hz, 1.1 kbps, 80 ms streaming frame); Moshi repo (Py MIT, Rust Apache-2.0, weights CC-BY 4.0) | Kyutai Mimi/Moshi | 2024 | CC-BY 4.0 (weights) |
| [24] | https://huggingface.co/docs/transformers/en/model_doc/mimi | Transformers Mimi docs (RVQ 16 codebooks, causal streaming encoder) | HF Transformers | 2024-2025 | docs |
| [25] | https://huggingface.co/hubertsiuzdak/snac_24khz | snac_24khz card (License: mit; 0.98 kbps, 19.8M; token rates 12/23/47 Hz; speech-recommended; used by Orpheus-TTS, mini-omni) | SNAC | 2024 | MIT (weights) |
| [26] | https://huggingface.co/novateur/WavTokenizer | WavTokenizer cards (small-600-24k-4096: LibriTTS, 40 tok/s, open-source) | WavTokenizer | 2024 | MIT |
| [27] | https://github.com/jishengpeng/WavTokenizer | WavTokenizer repo (MIT, ~1.3k stars, ICLR 2025) | WavTokenizer | 2024-2025 | MIT (code) |
| [28] | https://huggingface.co/swiss-ai/wavtokenizer-large-unify-40token | Transformers-format conversion (SEANet enc + Vocos-style iSTFT dec, 76.3M large; bit-exact vs original) | WavTokenizer port | 2025 | MIT (verify at pin) |
| [29] | https://www.isca-archive.org/interspeech_2025/yoneyama25_interspeech.pdf | Comparative Analysis of Fast and High-Fidelity Neural Vocoders for Low-Latency Streaming Synthesis (MS-Wavehax: 2.4% of HiFi-GAN V1 size, 1-frame lookahead ~= non-causal; ONNX on AMD EPYC) | Interspeech 2025 (Yoneyama et al.) | 2025-08 | paper |
| [30] | https://arxiv.org/html/2507.14534v4 | Conan: Chunkwise Online Voice Conversion (Causal Shuffle Vocoder: causal conv + pixel shuffle; 37 ms fast / 140 ms full latency) | Conan | 2025-07 | paper |
| [31] | https://github.com/kan-bayashi/ParallelWaveGAN/issues/28 | Community RTF reports (1.6 i7 laptop; 0.46 with residual 30->10; 0.734 on 16-thread Xeon Gold 6154; HiFi-GAN V1 not real-time on low-end single-core EPYC 7551) | community | ~2019-2021 | n/a |
| [32] | (URL not captured in search aggregation — verify before external citation) | Subband WaveRNN, RTF 0.064 single-thread i7-8750H, Interspeech 2022 [REPORTED, weakly sourced] | academic | 2022 | n/a |
| [33] | https://heyneo.com/blog/kokoro-supertonic-inflect-nano-cpu-tts-benchmark + https://huggingface.co/hexgrad/Kokoro-82M | CPU TTS benchmark (Kokoro-82M ONNX RTF 0.571; PyTorch 0.787) + Kokoro card (StyleTTS2 + iSTFTNet, Apache-2.0) | Kokoro / heyneo | 2025 | Apache-2.0 (Kokoro weights) |
| [34] | https://arxiv.org/abs/2309.03199 | Matcha-TTS (OT-CFM; MAT-2 RTF 0.015 incl. vocoding) | Matcha-TTS | 2023-09 | paper (code MIT) |

### Verification notes
- Licenses read from primary artifacts (LICENSE files, LICENSE_weights, HF card license fields, verbatim README quotes) wherever possible; secondary aggregations flagged.
- Unresolved/weak items: [14] LICENSE text not read directly; [32] URL not captured; WavTokenizer medium/large checkpoint availability changed over time — re-verify; Mimi decoder parameter count not found — measure in Phase G.
- No downloads or installs performed; all findings via web verification on 2026-08-22.
