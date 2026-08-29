# STATE-OF-THE-ART — Open/Public TTS Landscape, August 2026

**CAEDRAL VOICE — Research A2 (TTS landscape & baselines). Survey date: 2026-08-22.**

Status: COMPLETE (desk survey, 2026-08-22). Companion to `BASELINE-MATRIX.md` (shared source numbering).

## 1. Architecture classes

### 1.1 Autoregressive codec-LM ("speech language models")

The dominant paradigm of 2025–2026. An LLM backbone (often a repurposed open LLM: Llama-3B in Orpheus [39], Llama-3.2-1B in Sesame CSM-1B [41], Qwen2.5-0.5B in Spark-TTS [37], Qwen2-0.5B in NeuTTS Air [21], Qwen3-0.6B in OuteTTS [45], Llama-0.5B-class in Chatterbox [16][17], MiniCPM-4 in VoxCPM [23]) generates discrete audio-codec tokens next-token-style; a decoder/vocoder renders waveform. Codecs vary: SNAC, Mimi (Kyutai), Descript Audio Codec, BiCodec (single-stream, factorized VQ), NeuCodec (50 Hz single codebook). Token rate is the CPU-critical number: Dia's ~86 tokens/second of audio [28] vs NeuCodec's 50 Hz [21] vs VoxCPM's 12.5 Hz [25] — an order-of-magnitude spread that directly determines feasible CPU streaming.

Properties: natural zero-shot cloning from reference audio; emergent paralinguistics when trained on expressive data (Dia's `(laughs)` tags [28], Orpheus emotion tags [39], Chatterbox `[laugh]`/`[sigh]` [18]). Costs: slow token-by-token generation (RTF ≈ 2.15 even on an A4000 GPU for Dia at release [28]), hallucination/disfluency risk, and heavy memory. The 2025–2026 trend is **distillation to single-step or few-step decoders** to fix exactly this: Chatterbox-Turbo (10 steps → 1) and Chatterbox-Nano [17][18].

### 1.2 Non-autoregressive mel + duration predictor (the VITS lineage)

The workhorse class of 2023–2024: predict duration → condition a generative mel decoder → vocoder. VITS itself (end-to-end variance adapter + normalizing flow + HiFi-GAN decoder) still powers Piper's hundreds of voices [4][6]. StyleTTS2 added diffusion-based prosody/style sampling plus adversarial training with WavLM features, achieving then-SOTA naturalness and zero-shot cloning [7]; its architecture (minus diffusion) became Kokoro, the current small-model quality king [1][2]. MeloTTS is a BERT-conditioned VITS-family variant explicitly marketed for CPU real-time [12]. Matcha-TTS replaced the decoder with conditional flow matching but kept the duration predictor [14][15]. This class remains the best quality-per-CPU-cycle trade-off: no iterative sampling at inference (Kokoro, Piper) or few-step ODE only (Matcha, 5–10 Euler steps [14]).

### 1.3 Flow-matching

Conditional flow matching / rectified flow — regress a velocity field and integrate a short ODE — overtook both denoising-diffusion and normalizing flows for mel synthesis after E2/F5-TTS (Oct 2024) [11]. F5-TTS pairs it with a DiT backbone and predicted duration for strong zero-shot cloning with ~32 NFEs, reducible via sway sampling [9][10][11]. Matcha-TTS demonstrated optimal-transport CFM at tiny scale (~18M params) [15]. Flow matching also appears as the *decoder stage* inside hybrids (CosyVoice2/3 conditional flow-matching decoder [30][31]; IndexTTS2 [26]; Zonos components [43]). For CPU, flow-matching sits in the middle: fewer evaluations than diffusion, but still iterative — F5 on desktop CPU is plausibly slower than real time [ESTIMATED].

### 1.4 Diffusion

Largely displaced as a primary paradigm by flow matching, but alive inside systems: StyleTTS2's style/prosody SDE sampler [7], MegaTTS3's 0.45B Diffusion Transformer over WaveVAE latents (25 Hz @ 24 kHz) with impressive quality-per-param and accent-intensity control [35][36], and local DiT decoders in tokenizer-free hybrids like VoxCPM [23][24]. Diffusion's fixed-step iteration cost keeps pure-diffusion TTS off the CPU critical path.

### 1.5 Hybrids

Two hybrid families matter. (a) **AR-LM + generative decoder**: CosyVoice3 (LLM over supervised semantic tokens + flow-matching decoder) [30][31], IndexTTS2 (GPT-style semantic tokens + flow decoder, multi-modal emotion control) [26][27], VoxCPM (TSLM prosody planner + RALM acoustic LM guiding a local DiT, "tokenizer-free" via FSQ bottleneck) [23][24]. These dominate quality leaderboards but are GPU-first. (b) **SSM/hybrid backbones**: Zonos-v0.1's 1.6B hybrid SSM/Transformer with explicit emotion conditioning [43] — architecturally interesting, commercially stalled.

## 2. What changed 2024–2026

1. **LLMs ate TTS, then dieting began.** Early 2025: labs bolted speech codecs onto open LLM backbones (Orpheus, CSM-1B, Zonos, Spark-TTS, Chatterbox — all Feb–Apr 2025 [16][37][39][41][43]). Mid/late 2025 through 2026: the same labs shipped *small distilled* versions (Chatterbox-Turbo/Nano [17][18], NeuTTS-Nano/2E [22]) because token-rate × params made the big AR models unusable for real products.
2. **Zero-shot cloning became table stakes** for frontier-class systems — every 2025 flagship clones from a reference clip (F5 5–10 s [10], NeuTTS Air 3 s [21], Chatterbox-Nano 5 s [18]). The counter-trend matters for Caedral: fixed-voicebank systems (Kokoro, Piper, KittenTTS) still own the "many consistent product voices" niche, because cloning UX and per-product voice consistency remain unsolved.
3. **Emotion control matured from labels → tags → conditioning.** Parler-TTS natural-language description prompts (2024) [47] → Orpheus/Dia inline paralinguistic tags [28][39] → Chatterbox exaggeration knob + Turbo tags [17][18] → IndexTTS2 multi-modal emotion reference audio/video/vectors [26]. Expressiveness is no longer a closed-API differentiator.
4. **The license landscape fragmented sharply.** The 2024 norm (Apache/MIT code *and* weights: Kokoro, MeloTTS, Matcha, Parler) eroded: F5 weights CC-BY-NC [9][10]; IndexTTS custom registration-required weight license [26][27]; Fish Audio moved everything behind a research-only license in Mar 2026 [32][33]; Neuphonic split Apache (Air) vs paid-commercial (Nano/2E) within one family [21][22]; even Piper flipped MIT→GPL-3.0 moving to OHF [4][5][6]. **Permissive code + encumbered weights is now the default trap.**
5. **On-device became an explicit market segment.** Kokoro proved small models could win quality contests [2]; by 2026 vendors ship CPU-first SKUs with published CPU numbers (Chatterbox-Nano 3× RT on 8 cores [18], NeuTTS Air GGUF on Raspberry Pi-class hardware [21], KittenTTS ≤25–80 MB models [19]). GGUF/llama.cpp quantization culture migrated from LLMs into speech.
6. **Training-cost collapse at the small end.** hexgrad documented ~$1k for Kokoro v1.0 and ~$110 for v1.1-zh [2][3]: competitive small English voices are now a fine-tuning exercise, not a research program. This validates a proprietary-training strategy like Caedral's.
7. **Consolidation around three runtimes.** ONNX Runtime (Piper, KittenTTS, Kokoro-ONNX), llama.cpp/GGUF (NeuTTS Air), PyTorch+CUDA (all GPU-first systems). A Caedral CPU runtime story must target names from this short list.

## 3. CPU-relevant frontier

**The practical frontier as of Aug 2026 spans ~15M–550M parameters**, in three tiers:

- **Tier A — embedded (<100 MB, fixed voices):** KittenTTS 15–80M ONNX at 24 kHz [19][20]; Piper voices ~1–30M each [4][6]. Comfortably real-time; quality = "clear synthetic," not premium.
- **Tier B — laptop-class (100M–600M): where a commercial multi-voice CPU product lives today.** Kokoro-82M [1][2][3] (best quality-per-MB found in this survey); Chatterbox-Nano 110M [18] (cloning + tags); OuteTTS-0.6B Apache variant [45]; NeuTTS Air Q4 GGUF ≈527 MB [21] (cloning via GGUF). Plausibly RTF ≤ 0.5 on 8 modern cores [ESTIMATED] — must be measured, since almost nobody publishes CPU numbers.
- **Tier C — desktop/workstation, batch-only:** Spark-TTS 0.5B [37], CSM-1B [41], VoxCPM-0.5B (RTF 0.17 *on an RTX 4090* [23]) — feasible offline, unrealistic for streaming without exotic optimization.

**Open engineering problems to assume unsolved upstream:** (i) batch-1 latency on CPU for AR models — unpublished anywhere we looked; (ii) consistent *multi-voice* identity from one model without cloning references (Kokoro's pretrained style vectors are the best public example); (iii) Windows-first deployment — nearly all projects are Linux-dev-first; only piper1-gpl ≥1.5.0 and KittenTTS explicitly test Windows; (iv) int8/int4 quality retention for mel-decoder models is far less documented than LLM quantization.

**Strategic read for Caedral:** the public frontier gives away permissive *components* (architectures: StyleTTS2/Kokoro lineage; codec patterns: BiCodec/NeuCodec; training recipes) but leaves the CPU-multi-voice-product integration unsolved — exactly the gap a proprietary program can occupy. Benchmark Tier B against Kokoro and Chatterbox-Nano; treat NC- or registration-licensed systems (F5 weights, IndexTTS, Fish/OpenAudio) as architecture references only.

## Sources

Numbering matches BASELINE-MATRIX.md so citations are shared across both documents. All accessed/verified 2026-08-22 unless noted.

1. https://github.com/hexgrad/kokoro — kokoro inference library; hexgrad; Apache-2.0; created Jan 2025.
2. https://huggingface.co/hexgrad/Kokoro-82M — model card; hexgrad; v1.0 2025-01-27; 82M params; StyleTTS2+ISTFTNet decoder-only; ~$1k training cost; Apache-2.0.
3. https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh — model card; hexgrad; v1.1-zh Feb 2025; ~$110 training cost; Apache-2.0.
4. https://github.com/OHF-Voice/piper1-gpl — Piper successor; Open Home Foundation; GPL-3.0; v1.5.0 Jul 17 2026.
5. OHF-Voice/piper1-gpl release notes v1.5.0 — Windows MSVC fixes, libpiper, espeak-ng bump. [REPORTED]
6. https://github.com/rhasspy/piper — original repo; archived; MIT; ONNX Runtime VITS voices.
7. https://github.com/yl4579/StyleTTS2 — official repo; MIT code; extra disclosure/consent terms on pre-trained weights.
8. yl4579/StyleTTS2 Issue #37 — weight-license clarification by author. https://github.com/yl4579/StyleTTS2/issues/37
9. https://github.com/SWivid/F5-TTS — official repo; MIT code; active through Jul 2026 (tag 1.1.22).
10. SWivid/F5-TTS SHARED.md + HF cards — Emilia-trained weights CC-BY-NC-4.0.
11. arXiv:2410.06885 — F5-TTS paper; Chen et al., SJTU; Oct 2024.
12. https://github.com/myshell-ai/MeloTTS — MyShell.ai; MIT; CPU real-time claim.
13. https://huggingface.co/myshell-ai/MeloTTS-English — HF card; MIT weights.
14. https://github.com/shivammehta25/Matcha-TTS — official repo; MIT; OT-CFM + Euler ODE; ONNX export path.
15. arXiv:2309.03199 — Matcha-TTS paper; Mehta et al.; ICML 2024.
16. https://github.com/resemble-ai/chatterbox — Resemble AI; MIT; created Apr 23 2025.
17. https://www.resemble.ai/learn/models/chatterbox — vendor family page: sizes, Turbo single-step decoder, paralinguistic tags. [REPORTED, vendor]
18. Chatterbox Nano/Flash announcement (Jul 2026) + https://huggingface.co/ResembleAI/chatterbox-nano — Nano 110M; 3× realtime on 8-core CPU. [REPORTED, vendor]
19. https://github.com/KittenML/KittenTTS — KittenML; Apache-2.0; v0.8.1 Feb 24 2026; developer preview; ONNX CPU-only; model sizes.
20. https://huggingface.co/KittenML/kitten-tts-nano-0.8 — HF card; Apache-2.0.
21. https://github.com/neuphonic/neutts-air — Neuphonic; Apache-2.0 (Air); Qwen2-0.5B + NeuCodec 50 Hz; GGUF Q4/Q8; 3-s cloning.
22. Neuphonic NeuTTS-Nano / NeuTTS-2E pages — NeuTTS Open License v1.0 terms. [REPORTED — read license text before commercial use]
23. https://github.com/OpenBMB/VoxCPM — OpenBMB; Apache-2.0; RTF table; VoxCPM1.5 Dec 2025, VoxCPM2 Apr 2026.
24. arXiv:2509.24650 — VoxCPM technical report; ICLR 2026; Sep 29 2025.
25. https://huggingface.co/openbmb/VoxCPM-0.5B — HF card; Apache-2.0; 16 kHz output; 12.5 Hz token rate.
26. https://github.com/index-tts/index-tts — Bilibili; IndexTTS-2 Sep 8 2025; IndexTTS-2.5 Aug 10 2026; Apache-2.0 code / custom weight license.
27. index-tts Issue #228 — license ambiguity thread. https://github.com/index-tts/index-tts/issues/228
28. https://github.com/nari-labs/dia — Nari Labs; Apache-2.0 + research-intent caveat; 1.6B; DAC tokens; Dia2 Nov 19 2025.
29. https://huggingface.co/nari-labs/Dia-1.6B-0626 — HF card.
30. https://github.com/FunAudioLLM/CosyVoice (now QwenAudio/CosyVoice) — Alibaba; Apache-2.0; Fun-CosyVoice3-0.5B-2512 Dec 2025.
31. arXiv:2505.17589 — CosyVoice 3 paper; May 23 2025.
32. https://github.com/fishaudio/fish-speech/blob/main/LICENSE — Fish Audio Research License (updated Mar 7 2026); research/non-commercial only.
33. https://huggingface.co/fishaudio/s1-mini — CC-BY-NC-SA-4.0; plus fish.audio/developers commercial pricing. [REPORTED]
34. fishaudio/fish-speech Discussion #1051 — NC-outputs ambiguity. 
35. https://github.com/bytedance/MegaTTS3 — ByteDance; Apache-2.0; WaveVAE encoder withheld; academic-intent statement.
36. https://huggingface.co/ByteDance/MegaTTS3 — HF card; Apache-2.0.
37. https://github.com/SparkAudio/Spark-TTS — SparkAudio; Apache-2.0 incl. BiCodec; Qwen2.5-0.5B.
38. https://huggingface.co/SparkAudio/Spark-TTS-0.5B — HF card; Apache-2.0 checkpoint.
39. https://github.com/canopyai/Orpheus-TTS — Canopy Labs; Apache-2.0 (post Issue #24); Llama-3B backbone; SNAC tokens.
40. Orpheus/Baseten inference partnership announcement, May 2025. [REPORTED]
41. https://github.com/SesameAILabs/csm — Sesame AI Labs; Apache-2.0 code; CSM-1B Mar 13 2025; Llama 3.2 1B; Mimi codec.
42. https://huggingface.co/sesame/csm-1b — gated HF checkpoint.
43. https://github.com/Zyphra/Zonos — Zyphra; Apache-2.0; Feb 7 2025; 1.6B hybrid SSM/Transformer; emotion conditioning.
44. Zyphra Zonos-v0.1 blog + HF card. [REPORTED]
45. OuteAI model cards: OuteTTS-1.0-0.6B (Apache-2.0) vs Llama-OuteTTS-1.0-1B (CC-BY-NC-SA-4.0); library github.com/edwko/OuteTTS Apache-2.0.
46. edwko/OuteTTS README — speaker creation, cloning, GGUF support. [REPORTED]
47. https://github.com/huggingface/parler-tts — Hugging Face; Apache-2.0 fully-open release; Mini Aug 2024 / Large Dec 2024; dormant since Dec 10 2024.
48. https://huggingface.co/parler-tts — org page; description-conditioned models.
