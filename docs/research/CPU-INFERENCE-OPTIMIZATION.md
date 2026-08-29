# CPU Inference & Quantization Playbook — Caedral Voice

**Program:** CAEDRAL VOICE (proprietary CPU-first English TTS) · **Researcher:** F (CPU inference & quantization)
**Date:** 2026-08-22 · **Target hardware:** x86-64 client CPUs, AVX2 + AVX-VNNI typical, no AVX-512; Windows/Linux; wheel-only deployments (no local compilers).

**Labels:** `[REPORTED]` = measured/claimed by a third party (cite given) · `[ESTIMATED]` = our estimate derived from cited data · `[HYPOTHESIS]` = plausible, unverified.
**Citations:** inline `[n]` → `## Sources` at end. Entries marked `[URL-UNCONFIRMED]` had the source located but exact URL not yet re-verified at write time.

---

## 1. ONNX Runtime (ORT) CPU EP tuning

### 1.1 Threading model and defaults

- ORT creates an **intra-op pool** (ops parallelized internally) and, only in `ORT_PARALLEL` execution mode, an **inter-op pool** (ops run concurrently). Default `intra_op_num_threads=0` means "one thread per **physical** core" — hyperthreads are not counted by default, and threads are affinitized to physical cores automatically [1]. Setting the value explicitly **disables** the automatic affinity placement [1].
- Default execution mode is `ORT_SEQUENTIAL` [1]. For batch-1 speech graphs (mostly a single dependency chain: frontend → acoustic model → vocoder), `ORT_PARALLEL` rarely helps and can hurt; the docs explicitly warn it "could also hurt performance on some models without many branches" [1]. Keep SEQUENTIAL unless profiling shows a wide graph.
- **Batch-1 guidance [ESTIMATED from reported scaling data]:** intra-op scaling for small conv/transformer nets saturates early. Sherpa-onnx's official Raspberry Pi 4 measurements show piper-VITS RTF improving 0.774 → 0.482 → 0.390 → 0.357 for 1→4 threads — i.e., ~2.2x from 4x the threads, with the last thread buying only 9% [44]. On desktop cores expect the knee around 2–4 intra threads for a small acoustic model; measure per stage (acoustic vs vocoder separately — the vocoder is usually the more thread-scalable conv stack).

### 1.2 Spinning, backoff, and affinity (the hybrid-CPU section)

ORT worker threads **spin-wait** for work by default (server builds); this burns cycles and power and is the main source of "ORT uses 100% CPU doing nothing" reports [1][2].

Config keys (set via `session_options.add_session_config_entry(...)`) [1][2][4]:

| Key | Default | Notes |
|---|---|---|
| `session.intra_op.allow_spinning` | `1` (server pkgs) / `0` (client/on-device pkgs) | `0` = workers sleep instead of spin; lower latency jitter under co-located load, slightly slower wake-up [1][2] |
| `session.intra_op.spin_duration_us` | unset | Time-bounded spin window. `0` disables spinning; ~`500–2000` (µs) recommended for power-sensitive workloads [1][2] |
| `session.intra_op.spin_backoff_max` | `1` | Exponential `SpinPause()` backoff; must be power of two. Docs call it "particularly beneficial on hybrid (P-core/E-core) and mobile platforms" [1][2] |
| `session.intra_op_thread_affinities` | unset | Semicolon-separated per-thread logical-processor lists, e.g. `'0,1,2,3;4,5,6,7'` pins thread 0 to cores 0-3, etc. Normally leave unset; use to keep the pool off E-cores or inside one NUMA node [1] |

- **Officially benchmarked sweet spot:** `spin_duration_us="1000"` + `spin_backoff_max="8"` was "the most consistent best performer across models and thread counts" in ORT's own testing [1].
- **Hybrid P/E cores (Alder/Raptor/Meteor Lake):** ORT's docs do not count E-cores in the default thread count, but the OS is still free to schedule spinning workers onto E-cores, causing latency jitter [1][ESTIMATED]. Practical mitigations, in order of preference: (1) set `intra_op_num_threads` = P-core count and pin via `session.intra_op_thread_affinities` to P-core logical IDs; (2) use `spin_backoff_max=4` (the hybrid-friendly value per the config-key header) [2]; (3) on Windows, verify the power plan isn't parking P-cores `[ESTIMATED]`.
- **NUMA:** supported since ORT 1.14; confining the pool to one node gave "nearly 20 percent performance improvement" vs. spanning nodes in ORT's test [1]. Client parts are usually single-node; irrelevant there, relevant on Xeon workstations.
- **Latency variance / >64 logical cores:** ORT's troubleshooting page recommends `session.dynamic_block_base=4` to reduce variance from the dynamic cost model, and notes a lock-free task queue build flag (`--use_lock_free_queue`) for very high core counts [3].

### 1.3 Arena and memory patterns

- `enable_cpu_mem_arena` (default on): a BFC-style arena caches allocations across `Run()` calls; **it never shrinks by default** — the high-water mark is held for the process lifetime [8][9].
- **Memory patterns** (`enable_mem_pattern`): ORT pre-plans internal allocations per input shape. This **only works with static shapes and SEQUENTIAL mode**; variable-length audio inputs defeat it and cause re-planning [9][ESTIMATED from docs]. For Caedral: bucket/pad mel-frame counts to a small set of shapes (e.g., multiples of 256 frames) so patterns get reused; otherwise accept pattern-off and rely on the arena.
- **Arena shrinkage** is a **RunOption**, not a session option: `run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")` frees unused arena memory at the end of every `Run()` (only supported cadence). CPU arena must not be disabled if you shrink `cpu:0` [8][9]. Interaction with extend strategy: with `kNextPowerOfTwo` (default) only allocations beyond the initial chunk are shrink candidates — set `session.memory.initial_growth_chunk_size_bytes` to cover your typical utterance so only outlier-long inputs trigger growth [9]. Use shrinkage for co-located/desktop builds where RSS matters; leave it off for a dedicated serving box [ESTIMATED].
- Denormals: ORT exposes `session.set_denormal_as_zero` (Python `SessionOptions`) — audio DSP graphs can hit subnormal floats in filters/tails, which cost ~100x on x86 without FTZ/DAZ `[ESTIMATED]`; enable if profiling shows `__addsubpd`-style denormal stalls.

### 1.4 Graph optimizations and startup

- Optimization levels: `ORT_DISABLE_ALL` → `ORT_ENABLE_BASIC` → `ORT_ENABLE_EXTENDED` → `ORT_ENABLE_ALL` (default). CPU EP benefits: constant folding, redundancy elimination, conv+BN+ReLU fusion, attention/transformer fusions (Extended/All), and layout-sensitive kernel selection [5].
- **Offline optimization:** set `sess_options.optimized_model_filepath = "model.opt.onnx"` once on a dev machine, then ship the optimized graph and load it with `graph_optimization_level=ORT_DISABLE_ALL` to cut session-init time (maintainer-endorsed, CPU-EP-only caveat) [5][6]. Savings are real but **modest**: session creation still pays protobuf parse, topological sort, kernel resolution, and memory planning [7]. ORT's own format (`.ort` models) is the fuller version of this idea [55].
- **Do not create a session per request.** Session creation + first `Run()` (kernel selection, arena growth, autotuning) is one-time expensive work; the Kokoro ORT issue documents multi-second session-creation/first-run costs dominating naive benchmarks [14][7].

### 1.5 Known pitfalls that fake speedups (ORT-specific)

1. **Benchmarking session creation + warmup inside the timed region** (or excluding them when the server pays them per process). Separate cold/warm; see measurement protocol §(c) [14][7].
2. **Thread oversubscription with the HTTP framework:** `uvicorn --workers 4` × ORT `intra_op=8` = 32 spinning threads on 8 cores. Fix: one shared session (Run is thread-safe [12]) with `intra_op` sized to the concurrency budget (see §7).
3. **Hybrid-CPU variance misread as model variance:** E-core scheduling + spinning causes bimodal latencies. Pin threads / use backoff before concluding the model is unstable (§1.2).
4. **Dynamic shapes silently disabling mem patterns and re-triggering planning:** pad/bucket lengths [9][ESTIMATED].
5. **Comparing fp32 vs "int8" where the int8 path is dynamic quint8** — a known *slower* path on x86 (see §3.4; sherpa-onnx measured 3x *slower* [32]).

### 1.6 Windows specifics

- ORT ships CPU EP wheels for Windows with the same API; no compiler needed. Spinning defaults differ by package flavor (client/on-device builds default `allow_spinning=0`) [2] — check your wheel's behavior rather than assuming.
- High core-count Windows boxes (>64 logical) hit the lock-contention issue ORT addresses with the lock-free queue build; on typical client parts this is a non-issue [3].
- `[ESTIMATED]` Windows-specific operational items: use High Performance / Balanced-with-boost power plans (Windows parks cores aggressively on Balanced at low utilization, adding wake latency); timer resolution changes (`timeBeginPeriod`) affect sleep-based backoff granularity — relevant only if you disable spinning.

---

## 2. OpenVINO

### 2.1 Positioning: when it beats ORT

- OpenVINO's CPU plugin is built on its own JIT/oneDNN-derived kernel stack with explicit hybrid-CPU awareness: under the `LATENCY` hint it defaults to **`num_streams=1`, threads ≈ P-cores only (E-cores on a single socket may be added), hyperthreading disabled**; `ov::inference_num_threads` caps the total [15][16]. This is exactly the policy ORT makes you hand-tune (§1.2).
- `[REPORTED]` Head-to-head evidence, ASR-family models on Intel CPUs:
  - Lenovo Press (lp2448): Whisper-Large-v3 on Xeon — HF FP16 baseline vs OpenVINO FP16 = **+159% throughput, −85% TTFT** at unchanged WER (2.00% → 1.98% LibriSpeech); INT4 = +187% throughput, WER +0.01–0.05pp [53].
  - Community Whisper-on-CPU comparisons report PyTorch/OpenVINO ahead of plain ONNX Runtime CPU EP on the same model [REPORTED; URL-UNCONFIRMED Medium benchmark, see Sources note].
  - Counter-evidence for balance: on ARM64 Android, sherpa-onnx users found Supertonic-3 (int8) *slower* than piper VITS regardless of runtime — runtime choice doesn't rescue an unoptimized graph [REPORTED][52].
- Pattern across sources: OpenVINO tends to win most on **conv/RNN-heavy nets with many small kernels** (LSTM acoustic models, WaveRNN-style vocoders) where its fusion + thread scheduling shine, and on any model where you want INT8 done well out of the box (§2.3). For a transformer acoustic model already well-fused in ORT, gaps shrink to single-digit percent `[ESTIMATED]`. The honest method is our own A/B on Caedral's graphs with identical thread budgets (protocol §c).
- Interop note: ORT can load **OpenVINO as an execution provider**, and OpenVINO can read ONNX directly — so exporting once to ONNX keeps both runtimes available without maintaining two export paths.

### 2.2 License and distribution

- **Apache-2.0** (GitHub repo and PyPI package both state it) [18]. Commercial-friendly; no copyleft obligations on linked binaries beyond Apache terms.
- Wheels: `pip install openvino` — Windows/Linux/macOS, x86_64 and arm64; PyPI wheel is **Python API only**; C/C++ runtime via archive, WinGet, vcpkg, conda, npm if ever needed [19][18]. Requires Python ≥3.10 for the current line (2025.x/2026.x releases observed) [18][19].
- NNCF (the quantization tool, §2.3) is likewise Apache-2.0 [22].

### 2.3 Performance hints and the INT8 pipeline

- Use `ov::hint::performance_mode = LATENCY` for batch-1 interactive synthesis (maps to streams=1, P-core threads, HT off [16]); `THROUGHPUT` only for offline bulk jobs — it picks multiple CPU streams sized by the model's memory-bandwidth pressure [15][16]. Query back `{ OPTIMAL_NUMBER_OF_INFER_REQUESTS }` rather than guessing; run that many async requests for throughput mode [15].
- PTQ INT8: POT is deprecated since OpenVINO 2023.0; use **NNCF**: `nncf.quantize(model, calibration_dataset)` (~100–300 representative samples; MinMax + FastBiasCorrection under the hood), plus accuracy-aware flow (`nncf.quantize_with_accuracy_control`) when a metric must be held [20][21][22]. Accepts OV IR, PyTorch, TF2, ONNX inputs [21].
- Load-time: OpenVINO separates `read_model` from `compile_model`; compilation dominates startup. Cache compiled blobs or keep the process resident — same discipline as ORT optimized-graph shipping (§1.4).

### 2.4 Guidance for Caedral

Keep ORT as the default serving path (single dependency, pip-only, best-documented threading controls we need) and treat OpenVINO as the drop-in accelerator to evaluate per-model: same ONNX input, `LATENCY` hint, NNCF INT8 as its quantization story. Adopt OpenVINO wholesale only if it beats tuned ORT by >25% RTF on the full pipeline `[ESTIMATED threshold]`, because a second runtime adds wheel weight and a second set of threading pitfalls.

---

## 3. Quantization for conv/transformer speech models

### 3.1 Taxonomy and where each applies

| Method | What it does | Best for | CPU speed reality |
|---|---|---|---|
| **Dynamic INT8** | Weights int8 offline; activation scales computed **per forward pass** | Transformer MatMul/Linear blocks, RNNs | ORT docs recommend it for RNN/transformer [10]; but it is a known **slow path** for conv-heavy graphs — sherpa-onnx's int8 VITS (dynamic quint8) ran ~3x *slower* than fp32 with the same runtime [REPORTED][32] |
| **Static INT8** (calibrated) | Activation ranges fixed from calibration data (~100–1000 samples); QDQ or QOperator graph | CNNs / conv vocoders | The only int8 path that can beat fp32 on convs; needs calibration set + accuracy validation |
| **Weight-only INT8/INT4** | Only weights compressed; compute in fp32 after dequant | Memory-bound decode stages; disk/RAM-constrained installs | Saves 4x file size, near-fp32 speed if dequant is once-at-load or fused; sherpa-onnx's fp16 files behave this way (see §3.6) [REPORTED][33] |
| **QAT** | Fake-quant during training/fine-tune | When PTQ misses the quality bar | Costs a training pipeline, not a runtime cost |

ORT's own decision table: static for CNNs, dynamic for RNNs/transformers, QAT if both fail accuracy [10].

### 3.2 QDQ vs QOperator

- QOperator: dedicated `QLinearConv` / `MatMulInteger` nodes. QDQ: `DequantizeLinear(QuantizeLinear(x))` pairs around ordinary ops; quant params live in the graph and fusers fold them into adjacent ops.
- ORT's guidance: prefer **QDQ** (`QuantFormat.QDQ`, default since ORT 1.11) — x86-64 CPU kernels are optimized around QDQ; **S8S8 + QOperator is specifically called out as slow on x86-64 and to be avoided**; QOperator mainly serves other backends [10][11].
- Signedness on our target hardware: with AVX-VNNI use activations **QUInt8 × weights QInt8**, `reduce_range=False` [11]. Without VNNI (not our floor, but for old AMD), non-VNNI `VPMADDUBSW` saturation forces `reduce_range=True` or U8U8 [11]. oneDNN documents the same instruction-level story: VNNI's single `VPDPBUSD` computes four s8/u8 products exactly, vs. the saturating mad sequence without it [23].

### 3.3 Per-channel weights, and what actually degrades

- `per_channel=True` (weight granularity per output channel) improves accuracy wherever weight ranges vary per channel — typical for convs — at small throughput cost; default remains per-tensor for max throughput [10][11]. On AVX2/AVX-512, enabling per-channel may require `reduce_range=True` on some kernel paths [11].
- Debugging workflow when quality drops: ORT provides `create_weight_matching()`, `collect_activations()`, `modify_model_output_intermediate_tensors()` to rank offending tensors, then exclude them from quantization [11]. Run model optimization *before* quantization, not inside it [11].
- NNCF equivalent: MinMaxQuantization + FastBiasCorrection; bias correction measurably reduces PTQ drift layer-by-layer [22].

### 3.4 Measured evidence on vocoder/mel networks (the part that matters)

- **HiFi-GAN, dynamic int8 (OpenVINO issue #12964):** dynamically-quantized HiFi-GAN was ~**2x slower than FP32** on CPU *and* "quality deduced heavily" — the reporter concluded integer PTQ "is not fit for TTS vocoder" `[REPORTED]` [30]. Two independent failures (speed and quality) from the *dynamic* path on a transposed-conv GAN generator.
- **HiFi-GAN, TensorRT/modelopt attempt:** naive `trtexec --int8` gave ~2x speedup but QDQ placement gave none; maintainers steered toward QAT fine-tuning to hold quality `[REPORTED]` (GPU context, but confirms vocoders need care) [29].
- **VITS/piper, static-ish int8 via sherpa-onnx:** RTF 0.114 (fp32) → 0.320 (int8) at 75 MB → 22 MB, MacBook 1 thread; i.e., **int8 bought size, lost 3x speed** `[REPORTED]` [33]. Maintainer guidance verbatim: int8 preferred only "if memory matters" [32].
- **fp16 files as the practical middle:** same table shows fp16 = RTF 0.123 at 38 MB — near-fp32 speed, half the size, because ORT upconverts fp16→fp32 at load/compute (§3.6) [33].
- **Baseline quality anchor:** HiFi-GAN V1 MOS 4.27 vs ground truth 4.57 [31] — any MOS drop ≥0.05–0.1 from PTQ is audible territory for a product voice `[ESTIMATED]`.
- **Mel-spectrogram frontends / acoustic encoders** tolerate int8 far better than waveform generators in community experience (no headline degradation complaints for Matcha acoustic models; complaints concentrate on vocoders) `[ESTIMATED from absence-of-evidence across §6 sources]`. A Jetson/sherpa user went further: "do not int8-quantize the vocoder convs" — measured anti-remedy on ARM [URL-UNCONFIRMED HF model card, see Sources].
- **Net expectation for Caedral [ESTIMATED]:** static per-channel QDQ int8 on the *acoustic/text-encoder* stack: ≤0.02–0.05 MOS-equivalent drift, 1.5–2.5x speedup potential on MatMul-heavy parts with VNNI. On the *vocoder*: expect either audible artifacts or no speedup from naive PTQ; if the vocoder must be compressed, prefer (a) architecture-level shrink (§4), (b) fp16 storage + fp32 compute, or (c) QAT with a fine-tune on paired audio.

### 3.5 AVX-VNNI benefits (our actual hardware)

- AVX-VNNI (`VPDPBUSD`, 256-bit) is present on Alder-Lake-and-later P-cores and recent AMD Zen4+ (as AVX512-VNNI downclocked to 256-bit form); oneDNN exposes it as `AVX2_VNNI` and dispatches automatically [23].
- Measured int8-vs-fp32 geomeans with VNNI-class kernels: PyTorch X86 backend **2.97x geomean over FP32 across 69 models** on 4th-gen Xeon (vs 1.43x with the old FBGEMM backend) [24][25]; ONNX Runtime reported up to **2.9x** on quantized BERT with Intel DL Boost [REPORTED; URL-UNCONFIRMED ORT blog]. Client-core (non-AMX) reality is closer to 1.5–2.0x on GEMM-bound layers because AVX2-VNNI has half the width of AVX512-VNNI and there are no AMX tiles `[ESTIMATED]`.
- Benchmarking tip from oneDNN: pin ISA with `ONEDNN_MAX_CPU_ISA=AVX2_VNNI` to isolate VNNI's contribution on hybrid CPUs [23].

### 3.6 FP16-on-CPU viability

- Without AVX512-FP16 (Sapphire Rapids+, excluded by charter) there is **no native fp16 math**: ORT computes fp16 graphs with fp32 kernels plus casts ("Intel CPUs doesn't support fp16 actually") [13]; OpenVINO likewise converts f16→f32 internally unless native fp16 units exist [17]; PyTorch CPU does the same via F16C conversion [26].
- Consequences: (a) shipping fp16 *weights* is a **size win only** — compute stays fp32, so expect ≈fp32 RTF, which is exactly what sherpa measured (0.123 vs 0.114) [33][13]; (b) eager fp16 execution (cast per op) can be catastrophically slow — 5–14x slower than fp32 reported for transformers on consumer CPUs [27]; (c) even dequant/cast costs matter in hot loops — llama.cpp cut ~10% kernel time just by using F16C `vcvtph2ps` instead of a LUT [28].
- Verdict: fp16 is a **storage format**, not a compute format, on our target fleet. Never run an fp16-execution graph; do ship fp16-weight files where download/disk size matters.

### 3.7 Decision matrix for Caedral components `[ESTIMATED]`

| Component | Recommended compression | Rationale |
|---|---|---|
| Text frontend (G2P/BERT-ish encoder) | Static QDQ int8, per-channel, U8S8 | Transformer MatMuls = best case for VNNI int8; robust to PTQ |
| Acoustic model (conv/transformer hybrid) | Static QDQ int8 with calibration on real phoneme sequences; exclude first/last layers if drift | Main FLOPs pool; validate with ABX/MOS-lite |
| Vocoder | Architecture shrink (§4) > fp16-storage > QAT-int8. Avoid naive PTQ int8 | §3.4 evidence: degradation or slowdown |
| Duration/pitch predictors | int8 freely | Tiny, robust |

---

## 4. Distillation & pruning evidence for TTS (runtime-benefit techniques only)

Filter applied per charter: every item below reduces actual CPU work (fewer params/MACs/serial steps), not just disk size.

### 4.1 Sequence-level knowledge distillation (teacher → parallel student)

- **FastSpeech (the canonical result):** an autoregressive Transformer-TTS teacher generated phoneme durations *and* mel targets (sequence-level KD) to train the parallel FastSpeech student. Student MOS 3.84 ± 0.08 vs teacher 3.88 ± 0.09 (Tacotron2 3.86) — quality retained while mel generation got **269x faster** (38x end-to-end with WaveGlow) [34][35]. Ablation: removing KD costs −0.325 CMOS [35].
- Why it transfers to Caedral: KD converts an AR/attention teacher's quality into a *parallel, CPU-friendly* student and simultaneously produces the duration labels — it is the highest-evidence quality-preserving compression in TTS [34][35].

### 4.2 Architecture search / slim student designs

- **LightSpeech (ICASSP 2021, Microsoft):** GBDT-guided NAS over FastSpeech2: **27.0M → 1.8M params (15x), 12.5G → 0.76G MACs (16x), CPU RTF 6.1e-2 → 9.3e-3 (6.5x)** with CMOS on par with FastSpeech2; a hand-designed equal-size baseline lost −0.230 CMOS. The searched architecture is mostly depthwise-separable convs (kernels 5–25, hidden 256) — LSTMs were explicitly excluded for slow inference [36]. This is the strongest "small conv student ≈ full quality" data point for CPU TTS.
- **MB-iSTFT-VITS:** sherpa-onnx maintainers call it "an order of magnitude faster than raw VITS with the same quality" — replacing the STFT-based synthesis path with mel→iSTFT removes the most expensive vocoder work while keeping the VITS front half [REPORTED][32].
- **piper's model family** is the deployed proof of "small VITS students": x_low/low/medium/high tiers (x_low ≈ 7 MB-class, medium ≈ 63 MB-class files `[ESTIMATED]` from HF hosting) hitting RTF 0.36–0.77 on a Pi 4 [44][46].
- **Kokoro-82M** shows the quality end of small: 82M params, MOS-proxy (UTMOS) 4.44 in third-party CPU benchmarking, RTF ~0.57 ONNX on a 4-core Xeon [42] — i.e., you can be 20x smaller than LLM-TTS systems and still lead quality, but conv/attention cost keeps RTF ~5x worse than piper-tier students [42].

### 4.3 Structured channel pruning on conv vocoders

- Direct published evidence on HiFi-GAN channel pruning is thin; adjacent results: ℓ₁ filter pruning on AudioLDM's U-Net (which embeds a HiFi-GAN) reached **~83% parameter / 39% MACs reduction** with maintained audio quality [REPORTED, arXiv 2607.13330 `[URL-UNCONFIRMED]`]; structured pruning of conditional-GAN U-Net generators (L2 criteria + per-layer sensitivity) outperformed global pruning [REPORTED, arXiv 2206.14658]. HiFi-GAN itself ships V1/V2/V3 configs down to 0.92M params — the vendor-sanctioned way to shrink is *retraining the small config*, which beats post-hoc pruning on quality [31].
- Practical rule `[ESTIMATED]`: on CPU, only *structured* (whole-channel) pruning pays — unstructured sparsity needs sparse kernels we won't have wheel-only. Prefer (1) train-small (HiFi-GAN V2/V3 pattern), (2) KD into a slim student (§4.1–4.2), (3) structured channel pruning with fine-tune only if 1–2 miss the budget.

### 4.4 Low-rank tricks

- Low-rank factorization of convs (e.g., 1x7 + 7x1 separable factorizations, already standard in FastSpeech-style encoders) and low-rank linear replacements retain quality when ranks are chosen by SVD energy + fine-tune; LightSpeech's SepConv discovery is the TTS-specific validation that separable/low-rank operators carry TTS quality at 16x fewer MACs [36]. No TTS-specific post-hoc SVD paper with runtime numbers surfaced — treat deeper low-rank surgery as unproven `[HYPOTHESIS]` unless we run it.

---

## 5. Memory discipline

### 5.1 What memory actually looks like during synthesis

- Steady-state RSS ≈ weights (fp32: ~4 bytes/param; int8: ~1) + arena high-water mark + ORT runtime overhead. The arena grows to the **worst utterance seen** and holds it (§1.3) [8][9]. Peak during the first runs exceeds steady state because of kernel/workspace selection and pattern planning `[ESTIMATED from arena semantics 8][9]`.
- Consequence for a long-text server: one 10-minute document early in the process lifetime permanently inflates RSS unless arena shrinkage is on (RunOption `memory.enable_memory_arena_shrinkage="cpu:0"` [8][9]) or lengths are bucketed (§1.3).
- Community RAM data points: on-device TTS eval measured 21–833 MB across 18 mobile TTS models (piper-tier at the low end, Kokoro-tier near the top) `[REPORTED]` [49 `[URL-UNCONFIRMED]`]; Kokoro fp32 ONNX file is ~330 MB (int8 ~128 MB) [33] so resident footprint of that class is ~0.35–0.5 GB `[ESTIMATED]`.

### 5.2 Weight mmap and load-time engineering

- ORT loads models eagerly (protobuf → weights in RAM); it does not mmap weights by default, so "mmap the weights" tricks from llama.cpp land do not directly apply `[ESTIMATED]`. What ORT *does* offer: (a) ship the pre-optimized graph (`.onnx` with `ORT_DISABLE_ALL` reload, or ORT-format model) to skip transform time [5][6][55]; (b) keep external-data ONNX so the tensor data file can live on disk-backed pages the OS manages (helps cold-start page-in overlap) `[HYPOTHESIS — verify page-cache behavior on Windows]`.
- **Startup cost breakdown (dominance order) `[ESTIMATED from 7][14]`:** disk read of weights (I/O-bound, mitigable by OS cache after first run) < protobuf parse + graph transforms (CPU-bound, mitigable by pre-optimized model [6]) < kernel resolution + memory planning (unavoidable [7]) < **first `Run()` warmup** (arena growth + code paths; must be done before serving — see protocol §c). Kokoro's ORT thread documents session creation + first inference as the expensive part people mis-attribute to the runtime [14].
- Practical targets: warm the session with 2–3 representative utterances at boot (also stabilizes arena size); keep the service resident rather than per-invocation CLIs; measure cold vs warm separately (§c).

### 5.3 Arena sizing knobs

- `session.memory.initial_growth_chunk_size_bytes` — set ≈ steady utterance working set so `kNextPowerOfTwo` growth rarely triggers and shrinkage only reclaims outliers [9].
- Arena extend `kSameAsRequested` + shrinkage = minimal-RSS profile for desktop co-located builds, at some alloc overhead; default `kNextPowerOfTwo` without shrinkage = max-throughput profile for dedicated serving [9].

---

## 6. Real-world data points [REPORTED BY THIRD PARTY]

### 6.1 Piper (rhasspy → OHF-Voice/piper1-gpl)

- Architecture: VITS exported to ONNX (discriminator/posterior stripped), espeak-ng embedded for phonemization, ONNX Runtime C++ inference, streaming chunked output; voice tiers x_low/low/medium/high [46].
- Measured latency: Raspberry Pi 4, sherpa-onnx official RTF for `vits-piper-en_US-lessac-medium`: **0.774 / 0.482 / 0.390 / 0.357** at 1/2/3/4 threads [44]. Pi 5 community numbers: short sentence 0.53 s ± 0.00; RTF ≈ 0.19 claimed in a guide (≈5x realtime) `[REPORTED]` [45][URL-UNCONFIRMED guide]. M2 MacBook: same sentence 0.15 s [45].
- How they got real-time on Pi-class hardware (synthesis of sources): small student models trained *for* the tier [46], espeak-ng instead of a learned frontend (ms-scale, no BERT) [46][47-contrast], fp32 conv kernels that ORT executes well, sentence-chunked streaming so first-audio latency ≠ full-text latency [46].

### 6.2 Kokoro-82M

- Third-party CPU benchmark (Xeon 8272CL, 4 cores, 150 runs): ONNX RTF **0.571** (~1.8x realtime) vs PyTorch 0.787; wall-clock 0.65 s for 12 chars up to 68 s for 1712 chars; UTMOS 4.44 — best quality of the engines tested but ~4x slower than piper-tier on the Pi [42][45].
- AWS c6a.8xlarge: PyTorch ≈ ONNX ≈ 5x realtime with 32 intra/inter threads [43]. Community mixed-precision/quantized Kokoro ONNX builds circulate (e.g., yakhyo/kokoro-onnx) but carry no published quality gate [54]. Operational lessons from the ORT issue: reuse the session; exclude phonemization (`kokoro.create`) from runtime comparisons; non-ONNX frontend work is substantial [14].

### 6.3 MeloTTS

- README claims "fast enough for CPU real-time inference" [47]; user-measured reality diverges: ~12 s for a 10-word sentence on CPU reported in issue #258, blamed on its BERT encoder inside the VITS loop [48]. Lesson for Caedral: the *frontend* can dominate; a learned BERT frontend is a CPU liability unless distilled/int8'd (§3.7) `[ESTIMATED]`.

### 6.4 On-device writeups 2025–2026

- **VoicePing offline TTS eval (2025):** 18 models across Android/iOS on Galaxy S10, 4 threads: piper-VITS fastest open option (RTF ≈ 0.06–0.08), Kokoro slower-than-real-time on phone SoCs (up to 15 s/sentence), memory span 21–833 MB, Matcha+Vocos best speed/resource balance on iOS `[REPORTED][49 URL-UNCONFIRMED]`.
- **Supertonic / SupertonicTTS (2025):** 99M-param ONNX multilingual TTS, 44.1 kHz, demonstrated ~0.3x RTF airplane-mode on an e-reader; paper arXiv:2503.23108 [50][51]. Counterpoint on ARM phones: Supertonic-3 int8 slower than piper VITS in sherpa-onnx testing [52].
- **sherpa-onnx Android screen-reader thread:** ~500 ms synthesis delays with lessac-medium; mitigations = piper-tier models, more threads, sentence batching, big-core scheduling [52].

### 6.5 Cross-engine comparison snapshot (Pi 5, short sentence) [45]

| Engine | Time |
|---|---|
| Piper (low tier) | 0.53 s |
| Kokoro | 4.80 s |
| PocketTTS ONNX | ~7 s |

---

## 7. Threading & concurrency model for a small HTTP service

### 7.1 Process/thread layout

- One process, one shared `InferenceSession` per model stage; `Run()` is documented thread-safe ("Multiple threads are allowed to run this function") [12]. Sessions are safe after construction; only construction needs synchronization [12].
- Size the request-executor pool to the CPU budget and make ORT threads fit *inside* it. Two sane configurations:
  - **Latency-first (interactive, C=1–4):** executor pool = N concurrent requests allowed; each request's `Run` gets full `intra_op = min(physical_cores / expected_concurrency, 4)` threads `[ESTIMATED from scaling data §1.1/44]`. With spinning enabled this reserves cores; with `allow_spinning=0` it shares gracefully.
  - **Throughput-first (C=8–16):** `intra_op=1..2`, many concurrent `Run`s — total throughput rises because small batch-1 graphs parallelize poorly intra-op but perfectly across requests [44-scaling + ESTIMATED].
- Framework hygiene: if uvicorn/FastAPI, run synthesis in a bounded `ThreadPoolExecutor(max_workers=N)` sized to the above, never in unbounded `def` endpoints (anyio threadpool defaults will oversubscribe); avoid multiple worker processes unless you also divide `intra_op` accordingly (§1.5 pitfall 2).

### 7.2 Batching trade-offs at concurrency 1–16

- Dynamic micro-batching merges concurrent requests into one `Run`. For batch-1 speech nets the win is real but bounded: conv/GEMM efficiency improves with batch, yet queueing delay grows; at C≤2 batching usually loses to just running both requests concurrently `[HYPOTHESIS to verify]`; at C≥8 on one socket, batch-4/8 can cut per-utterance cost 20–40% for conv stacks `[ESTIMATED]`. Decide by measuring p99 *per-request* latency, not server throughput.
- Streaming responses (sentence-level chunks, piper-style) dominate batching for perceived latency: first-audio time stops depending on document length [46].

### 7.3 Safe-concurrency estimation method (do this, don't guess)

1. Fix SLO: e.g., p99 RTF ≤ 0.5 and p99 first-chunk ≤ 300 ms on reference hardware.
2. Load-test with an open-loop generator (wrk/k6/vegeta replaying realistic text-length distribution) stepping concurrency 1→2→4→8→16.
3. At each step record: throughput, p50/p99 end-to-end, p99 per-stage RTF, CPU utilization (should approach but not exceed ~90% of allocated cores), context switches/SPIN count proxy (CPU% minus useful work).
4. Safe concurrency = highest step meeting SLO with CPU < ~90%; provision pool = that number; re-verify after any ORT/runtime upgrade.
5. Guardrail math (Little's law sanity check): max sustainable concurrency ≈ arrival_rate × p99_latency_service_time; keep admitted concurrency below it.

---

## 8. Non-TTS transferable ideas

### 8.1 Speculative decoding for AR speech tokens — yes, with a twist

- Direct evidence exists now: **Speech Speculative Decoding (2025)** drafts with a small model, verifies in parallel, and adds a **tolerance factor β** because speech tokens are semantically continuous — many token swaps reconstruct perceptually identical audio, so strict rejection wastes valid drafts; quality metrics held across β settings [37]. **Principled Coarse-Grained acceptance (Apple, 2025)** formalizes this as rejection sampling over Acoustic Similarity Groups from the target embedding space, raising acceptance/throughput over both standard SD and naive relaxation while preserving intelligibility/speaker similarity [38].
- AR-ASR measurements give calibration for expected gains: restart acceptance 0.47–0.76, continuation acceptance 0.22–0.78 depending on audio conditioning; speedups 1.14–1.69x [REPORTED][URL-UNCONFIRMED arXiv 2608.12703].
- For Caedral `[HYPOTHESIS]`: if we ship any AR token stage, budget a draft model ~5–10% of target size and expect ~1.2–1.5x wall-clock via group-level acceptance; CPU economics favor it only if verification batches well (parallel verify = GEMM-shaped, our good case).

### 8.2 Early-exit

- Well-evidenced for LLMs, weakly for speech; duration/predictor heads already act as "exits". Skip unless a confidence head falls out of training for free `[HYPOTHESIS, deprioritized]`.

### 8.3 Cache the text frontend

- Phonemize/G2P results are deterministic per normalized text → memoize `(hash(normalized_text), voice_id) → phoneme IDs` in an LRU. piper's espeak-ng path is already ms-scale [46], but a learned frontend is not (MeloTTS's BERT frontend dominating its latency is the cautionary tale [48]). Zero-risk win; also cache prosody/duration predictions per sentence where speaker-independent.

### 8.4 Operator-fusion wins

- What ORT/OpenVINO fuse automatically: conv+BN+ReLU, conv sums, transformer attention/MatMul fusions at ORT_ENABLE_ALL [5]; NNCF folds QDQ pairs into adjacent ops during int8 conversion [21][22]. Manual wins left on the table: precomputing mel-filterbank as a constant-folded conv (do it once at export, not per request); fusing vocoder upsample+activation chains into fewer kernels via export-time rewriting `[ESTIMATED]`; replacing STFT/iSTFT with fixed conv equivalents when shapes allow.

### 8.5 INT8 KV cache (only if an AR decoder exists)

- Evidence from LLM serving: llama.cpp q8_0 KV halves cache RAM with negligible perplexity change but mixed speed (−1% to +9%) [40]; vLLM int8-KV PRs reported +15–20% throughput under load [39]; analysis shows gains come from memory-footprint→batching, not faster attention math [41]. For speech-token decoders with short contexts (seconds, not thousands of tokens), expect a **memory** win, not a speed win `[ESTIMATED]` — relevant only for long-form streaming or high concurrency.

### 8.6 Not worth porting (recorded to prevent re-litigation)

- Unstructured weight sparsity without sparse kernels (no wheel-only path); FP16 compute execution (§3.6); AVX-512-only code paths (charter excludes); speculative *mel-frame* generation without an AR structure to amortize `[HYPOTHESIS rejected for non-AR stacks]`.

---

## (a) Recommended default serving stack for Caedral

| Layer | Choice | License | Why |
|---|---|---|---|
| Runtime | **ONNX Runtime (CPU EP), pip wheel** | MIT | Wheel-only, Windows+Linux, documented thread/spin/affinity controls we need (§1); Run thread-safe for shared-session concurrency [12] |
| Session config baseline | SEQUENTIAL mode; `intra_op` per §7; `spin_duration_us=1000`, `spin_backoff_max=8` desktop / `4` hybrid; pre-optimized graph loaded with `ORT_DISABLE_ALL`; arena on, mem-pattern with bucketed shapes; shrinkage RunOption in co-located builds | — | §1.2–1.4, official recommended combo [1][2][6][8] |
| HTTP layer | Single process, bounded ThreadPoolExecutor (size = safe concurrency from §7.3), shared sessions, sentence-chunked streaming | — | §7 |
| Quantization toolkit | `onnxruntime.quantization` static QDQ (U8S8, per-channel weights, VNNI on) + ORT debug tooling; NNCF as cross-check | MIT / Apache-2.0 [11][22] | §3.2–3.4 |
| Alternate runtime (evaluate per-model, keep optional) | OpenVINO pip wheel, LATENCY hint | Apache-2.0 [18] | §2: adopt only if >25% RTF win over tuned ORT `[ESTIMATED threshold]` |
| Phonemizer/frontend | espeak-ng-class or distilled/int8 frontend + memoization cache (never an undistilled BERT in the hot path) | GPL (espeak-ng: process-boundary use if linked dynamically is avoided — check linking mode) | §6.1 vs §6.3 evidence |
| Model policy | Distilled/slim students (FastSpeech-KD/LightSpeech pattern); vocoder = trained-small config, fp16 storage, no naive int8 | — | §3.4, §4 |

License posture: MIT (ORT) + Apache-2.0 (OpenVINO/NNCF) are commercial-safe; espeak-ng is GPLv3 — keep it out-of-process or verify our distribution mode before embedding.

## (b) Ordered optimization checklist (profile-first discipline)

Run each step's measurement under protocol §(c). Never optimize without a profile attributing time to it.

1. **Instrument**: per-stage timers (frontend/acoustic/vocoder/audio-out), RSS sampler, RTF logger. Establish baseline p50/p99 cold+warm.
2. **Fix correctness of benchmarking**: session reuse, warmup excluded from steady metrics, cold/warm separated.
3. **Threading sweep**: intra_op ∈ {1,2,4,physical}; spinning variants (default, 1000µs+backoff8, off). Pick per deployment target (dedicated vs laptop).
4. **Hybrid-CPU hygiene**: pin to P-cores or cap threads at P-core count; re-run variance check (p99/p50 ratio).
5. **Graph level**: export-time constant folding (mel filterbank), optimized-model shipping, ORT_ENABLE_ALL; confirm fusions present (`onnxruntime.tools` / verbose log).
6. **Shape discipline**: bucket lengths; verify mem-pattern active; set initial_growth_chunk_size_bytes.
7. **Frontend**: cache G2P/phonemes; measure its share (<5% target).
8. **Quantize acoustic model**: static QDQ U8S8 per-channel; validate quality gate (ABX/MOS-lite ≥ threshold) *before* speed claims; exclude offender tensors via debug tools.
9. **Vocoder decision**: train-small or fp16-storage first; QAT only if the budget demands int8; never ship naive PTQ int8 vocoder without listening tests.
10. **A/B OpenVINO** on final graphs (same thread budget); adopt only on >25% RTF win.
11. **Memory pass**: peak vs steady RSS; enable shrinkage if co-located; re-check after long-document torture test.
12. **Concurrency certification**: §7.3 procedure; publish safe-concurrency number with the release.
13. **Only then**: exotic items (§8) behind flags, each with its own A/B.

## (c) Measurement protocol spec (comparability contract for all Caedral benchmarks)

**Hardware/process**
- Record CPU model, P/E core counts, RAM, OS build, power plan; pin nothing by default except where testing affinity itself (then document the exact `session.intra_op_thread_affinities` string).
- One process per configuration under test; fresh process per cold run.

**Cold vs warm**
- Cold: new process, first utterance timed end-to-end including session load. Report separately; never average into warm numbers.
- Warmup: 3 runs of a mid-length reference text, discarded.
- Warm: ≥30 runs per text-length bucket (short ≤50 chars, medium 200–400, long 800–1200) plus one torture case (>1500 chars); report p50/p99 and RTF per bucket.

**Threads**
- Sweep {1,2,4,N_physical} intra_op, inter_op fixed 1 (SEQUENTIAL); report the winning config alongside every latency figure — a number without its thread config is invalid.
- Spinning config recorded (`allow_spinning`, `spin_duration_us`, `spin_backoff_max`). Background load: none during microbenchmarks; for concurrency runs use §7.3 open-loop steps.

**Timing**
- Clock: `perf_counter_ns` around the ORT `run()` call (model time) AND end-to-end request wall time (service time); report both.
- RTF = audio_duration / wall_time, defined identically to sherpa-onnx so external numbers stay comparable [44].

**Memory**
- RSS sampling interval: **100 ms** (10 Hz) via psutil (Windows `GetProcessMemoryInfo` equivalent acceptable); additionally record `peak_working_set` at run end.
- Report: steady-state RSS (median over last 60 s idle), peak RSS during longest torture utterance, delta peak−steady. Arena shrinkage state recorded on/off.

**Quality gates attached to every perf claim**
- Any quantized/pruned/distilled artifact ships with: sample rate, test-set ABX or MOS-lite score vs fp32 baseline, and file size. Speed claims without a paired quality measurement are rejected in review.

**Environment hygiene**
- Disable OS updates/defender scans during runs; record ambient CPU% before start (<10% required); repeat any run whose p99/p50 > 3 (variance outlier → rerun, note hybrid-core suspicion).

---

## Sources

Format: [n] URL — title — publisher — date ("acc." = accessed).

[1] https://onnxruntime.ai/docs/performance/tune-performance/threading.html — ONNX Runtime Performance: Threading — Microsoft — acc. 2026-08-22
[2] https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h — Session options config keys (spinning defaults, backoff values) — Microsoft (source header) — acc. 2026-08-22
[3] https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html — Performance Troubleshooting (dynamic_block_base, >64-core lock-free queue) — Microsoft — acc. 2026-08-22
[4] https://github.com/microsoft/onnxruntime/issues/19968 — Disabling spinning via add_session_config_entry — GitHub/microsoft/onnxruntime — acc. 2026-08-22
[5] https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html — Graph optimizations (levels, online vs offline) — Microsoft — acc. 2026-08-22
[6] https://github.com/microsoft/onnxruntime/issues/13240 — Loading optimized model with ORT_DISABLE_ALL (maintainer guidance) — GitHub — acc. 2026-08-22
[7] https://github.com/microsoft/onnxruntime/issues/10131 — Session creation cost breakdown (deserialization, kernel resolution) — GitHub — acc. 2026-08-22
[8] https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h — Run options config keys (`memory.enable_memory_arena_shrinkage`) — Microsoft — acc. 2026-08-22
[9] https://github.com/microsoft/onnxruntime/pull/7284 — Memory arena shrinkage (dynamic-shape motivation, extend-strategy interaction) — GitHub — acc. 2026-08-22
[10] https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html — Quantization ONNX Runtime (dynamic vs static, QDQ/QOperator, signedness table) — Microsoft — acc. 2026-08-22
[11] https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md — Quantization tooling README (per-channel, reduce_range, debug tools) — Microsoft — acc. 2026-08-22
[12] https://github.com/microsoft/onnxruntime/issues/114 — "Is InferenceSession.Run thread-safe?" (maintainer: yes, no external sync) — GitHub — acc. 2026-08-22
[13] https://github.com/microsoft/onnxruntime/issues/10919 — fp16 computed with fp32 operators on CPU — GitHub — acc. 2026-08-22
[14] https://github.com/microsoft/onnxruntime/issues/23384 — Kokoro: ORT vs PyTorch CPU; session reuse; frontend cost — GitHub — 2024–2025, acc. 2026-08-22
[15] https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html — High-level Performance Hints (LATENCY/THROUGHPUT semantics) — Intel — acc. 2026-08-22
[16] https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html — Performance Hints and Thread Scheduling (CPU) — Intel — acc. 2026-08-22
[17] https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html — CPU device (f16→f32 internal conversion policy) — Intel — acc. 2026-08-22
[18] https://pypi.org/project/openvino/ — openvino on PyPI (License: Apache-2.0; platform support) — Intel — acc. 2026-08-22
[19] https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-windows.html — Install OpenVINO Runtime on Windows — Intel — acc. 2026-08-22
[20] https://docs.openvino.ai/2023.3/pot_introduction.html — POT deprecated since 2023.0; NNCF recommended — Intel — 2023
[21] https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html — Basic quantization flow (nncf.quantize) — Intel — acc. 2026-08-22
[22] https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md — NNCF PTQ usage (MinMaxQuantization, FastBiasCorrection, accuracy-aware) — Intel — acc. 2026-08-22
[23] https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html — INT8 computations (VNNI vs VPMADDUBSW saturation; ISA pinning) — Intel oneDNN — acc. 2026-08-22
[24] https://pytorch.org/blog/int8-quantization/ — INT8 Quantization for x86 CPU in PyTorch (2.97x geomean, 69 models) — PyTorch — acc. 2026-08-22
[25] https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-int8-inf-with-new-x86-backend.html — Accelerate PyTorch INT8 with the X86 backend — Intel — acc. 2026-08-22
[26] https://github.com/pytorch/pytorch/issues/97068 — CPU float16 performance optimization RFC (F16C convert-and-compute strategy) — PyTorch — acc. 2026-08-22
[27] https://github.com/huggingface/transformers/issues/34692 — fp16 on CPU 5–14x slower than fp32/bf16 — Hugging Face — acc. 2026-08-22
[28] https://github.com/ggerganov/llama.cpp/pull/6648 — F16C hardware F16→F32 conversion, 5–10% kernel gains — llama.cpp — acc. 2026-08-22
[29] https://github.com/NVIDIA/TensorRT/issues/4431 — HiFi-GAN INT8: trtexec --int8 2x but QDQ no speedup; QAT advised — NVIDIA — acc. 2026-08-22
[30] https://github.com/openvinotoolkit/openvino/issues/12964 — Dynamically quantized HiFi-GAN ~2x slower than FP32 with heavy quality degradation — OpenVINO — acc. 2026-08-22
[31] https://proceedings.neurips.cc/paper/2020/file/c5d736809766d46260d816d8dbc9eb44-Paper.pdf — HiFi-GAN: Generative Adversarial Networks for Efficient and Faithful Speech Synthesis (MOS table; V1/V2/V3 configs; 0.92M variant) — NeurIPS — 2020
[32] https://github.com/k2-fsa/sherpa-onnx/issues/575 — int8 TTS slower than fp32 (dynamic quint8 slow path; "if memory matters"; MB-iSTFT-VITS recommendation) — k2-fsa — acc. 2026-08-22
[33] https://github.com/k2-fsa/sherpa-onnx/issues/2460 — RTF/size table: piper fp32 0.114/75MB vs int8 0.320/22MB vs fp16 0.123/38MB; kokoro fp32/int8 — k2-fsa — acc. 2026-08-22
[34] https://arxiv.org/abs/1905.09263 — FastSpeech: Fast, Robust and Controllable Text to Speech (teacher-student sequence-level KD) — arXiv — 2019
[35] https://proceedings.neurips.cc/paper_files/paper/2019/file/f63f65b503e22cb970527f23c9ad7db1-Paper.pdf — FastSpeech (NeurIPS version: MOS table, KD ablation −0.325 CMOS, 269x mel speedup) — NeurIPS — 2019
[36] https://arxiv.org/pdf/2102.04040 — LightSpeech: Lightweight and Fast Text-to-Speech with Neural Architecture Search (27M→1.8M params, 6.5x CPU RTF, CMOS on par) — arXiv/ICASSP — 2021
[37] https://arxiv.org/html/2505.15380v1 — Accelerating Autoregressive Speech Synthesis Inference with Speech Speculative Decoding (tolerance factor β) — arXiv — 2025
[38] https://arxiv.org/html/2511.13732 — Principled Coarse-Grained Acceptance for speech speculative decoding (Acoustic Similarity Groups) — arXiv (Apple) — 2025
[39] https://github.com/vllm-project/vllm/pull/1507 — int8 KV cache: ~15% throughput improvement — vLLM — acc. 2026-08-22
[40] https://github.com/llama.cpp/pull/2969 — KV cache quantized to q8_0 (~50% cache RAM, ±0.005 PPL, mixed speed) — llama.cpp — acc. 2026-08-22
[41] https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079 — vLLM vs TensorRT-LLM #8: KV cache quantization (gains from footprint→batching) — SqueezeBits — acc. 2026-08-22
[42] https://heyneo.com/blog/kokoro-supertonic-inflect-nano-cpu-tts-benchmark — Kokoro vs Supertonic vs Inflect-Nano CPU TTS benchmark (RTF 0.571 ONNX, UTMOS 4.44) — heyneo.com — 2025/2026, acc. 2026-08-22
[43] https://gist.github.com/efemaer/23d9a3b949b751dde315192b4dcf0653 — Kokoro v1 benchmark: PyTorch/ONNX on CPU/GPU — GitHub Gist — acc. 2026-08-22
[44] https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html — VITS pretrained models, RTF on Raspberry Pi 4 (piper lessac-medium 0.774→0.357 at 1→4 threads) — k2-fsa docs — acc. 2026-08-22
[45] https://github.com/ktomanek/edge_tts_comparison — Edge TTS comparison: Piper 0.53 s on Pi 5, Kokoro 4.80 s, M2 0.15 s — GitHub — acc. 2026-08-22
[46] https://github.com/OHF-Voice/piper1-gpl — piper1-gpl: fast local neural TTS (VITS+ONNX Runtime+embedded espeak-ng, streaming chunks) — OHF-Voice — acc. 2026-08-22
[47] https://github.com/myshell-ai/MeloTTS — MeloTTS ("fast enough for CPU real-time inference" claim) — MyShell.ai/MIT — acc. 2026-08-22
[48] https://github.com/myshell-ai/MeloTTS/issues/258 — "inference time on cpu": ~12 s for 10 words; BERT encoder blamed — GitHub — 2025
[49] [URL-UNCONFIRMED] voiceping.net — Offline TTS eval: 18 on-device models, RTF/memory tables (piper RTF ≈0.06–0.08; Kokoro >1; 21–833 MB; Galaxy S10, 4 threads) — VoicePing — 2025
[50] https://github.com/supertone-inc/supertonic — Supertonic: on-device ONNX TTS, 99M params, 44.1 kHz, e-reader demo ~0.3x RTF — Supertone — acc. 2026-08-22
[51] https://arxiv.org/abs/2503.23108 — SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System — arXiv — 2025
[52] https://github.com/k2-fsa/sherpa-onnx/issues/1337 (+ #3813) — Android screen-reader ~500 ms latency mitigations; Supertonic-3 slower than piper VITS on ARM64 — k2-fsa — acc. 2026-08-22
[53] https://lenovopress.lenovo.com/lp2448-analyzing-the-performance-of-intel-xeon-cpus-for-automated-speech-recognition — Analyzing the Performance of Intel Xeon CPUs for Enterprise ASR (Whisper-Large-v3: OpenVINO FP16 +159% throughput at unchanged WER) — Lenovo Press — acc. 2026-08-22
[54] https://github.com/yakhyo/kokoro-onnx — Minimal kokoro-onnx inference with mixed-precision weights (unvalidated community quantizations) — GitHub — acc. 2026-08-22
[55] https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-model-runtime-optimization.html — ORT format models (pre-optimized model format) — Microsoft — acc. 2026-08-22

### URL-unconfirmed claims (located but exact URL not re-verified; do not cite externally without checking)
- Medium community Whisper CPU comparison (PyTorch/OpenVINO ahead of ORT CPU EP) — Medium — 2024/2025.
- HF model card "Luigi/matcha-zh-en-8k" (Jetson Nano; "do not int8-quantize the vocoder convs") — Hugging Face — acc. 2026-08-22.
- arXiv 2608.12703 — alignment drift in single-model speculative decoding for ASR (acceptance-rate tables) — 2026.
- Piper RTF ≈ 0.19 on Pi 5 guide — aivideosensei.com — 2025/2026.
- arXiv 2607.13330 — Efficient text-to-audio generation via pruning (83% params / 39% MACs) — 2026.
- arXiv 2206.14658 — Structured pruning of U-Net generators of conditional GANs — 2022.
- ONNX Runtime blog, BERT with Intel DL Boost up to 2.9x — Microsoft tech community — ca. 2021.
- Lenovo Press vLLM serving-capacity tables (Xeon 6980P/8568) — see [53] same document.

*End of document — Researcher F, 2026-08-22.*






