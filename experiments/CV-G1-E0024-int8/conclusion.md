# CV-G1-E0024 — INT8 dynamic quantization of the Kokoro ONNX model

Task QUANT-01 (2026-08-23). `onnxruntime.quantization.quantize_dynamic`
(weight_type=QInt8) on `models/kokoro/kokoro-v1.0.onnx` (310.45 MB fp32),
benchmarked against fp32 with the production engine path
(`CaedralVoiceKokoroEngine`, 4 intra-op threads, CPU EP, graph opt ENABLE_ALL,
fresh subprocess per model). Frozen sentence
`Hello. This is the first generation of Caedral Voice.`, voice=af_sky,
median of 10 warm reps after 1 discarded cold call, **2 rounds in opposite
order** to cancel thermal drift. Quality via repo-standard scorers
(`classify_preview`): dual-ASR minWER (faster-whisper small.en/base.en int8)
+ speechmos DNSMOS ovrl.

**Verdict: KILL — INT8 is a 10.1x latency regression.** Quantization itself
succeeded cheaply and audio stayed intelligible, but dynamic INT8 makes the
fused acoustic+vocoder transformer dramatically *slower* under ORT CPU EP.
The 3.5x size win does not buy back a 10x RTF loss; the fp32 graph remains the
production artifact.

## Measured results (`results.json`)

| Metric | FP32 | INT8 | Delta |
|---|---|---|---|
| Disk size | 310.45 MB | 87.69 MB | **-71.8%** (3.54x smaller) |
| RTF (median, n=20) | **0.575** | 5.810 | **10.1x SLOWER** (-90.1%) |
| RTF round medians | 1.858 s / 2.079 s wall | 18.94 / 18.84 s wall | order-stable, not thermal |
| Cold-call RTF | 0.554 | 6.455 | consistent regression |
| minWER (dual-ASR) | 0.1111 ("Kaedral") | 0.1111 ("Kydral") | identical error class |
| DNSMOS ovrl | 3.442 | 3.282 | -0.16 MOS |
| Class | INTELLIGIBLE | INTELLIGIBLE | both pass gates |

Quantization wall time: 10.35 s (well under the 15 min SKIP gate).

## Gate audit

- SKIP gate (quantize crash / >15 min): NOT triggered — clean in 10.35 s.
- KILL quality gates (WER > 0.3, DNSMOS < 2.8): NOT triggered — WER 0.111,
  DNSMOS 3.282. Audio is real speech of normal duration/flatness; the only
  transcript deviation is the invented proper-noun spelling "Caedral",
  present in BOTH arms (ASR artifact, not a quantization defect).
- KILL verdict applied anyway: the unstated but decisive requirement of
  "RTF improvement" failed catastrophically (-90%). A latency regression
  cannot ship regardless of size savings; verdict logic in
  `quant_bench.py` now kills when speedup < 0.98.

## Why it fails (mechanism)

Dynamic INT8 replaces MatMul/Gemm with DynamicQuantizeLinear + MatMulInteger.
Kokoro's per-layer matrices are small-to-medium and sequence-shaped; on this
CPU (Intel Family 6 Model 186, hybrid P/E cores) the runtime activation
quantization + int8 kernel + dequant overhead exceeds the fp32 FMA cost, and
the quantized rewrite forfeits whatever attention/Gemm fusion fp32 enjoys.
This matches the known profile: dynamic INT8 helps memory-bandwidth-bound
over-parameterized LLMs, not compute-bound TTS transformers at batch 1.

## Residual value

The 87.7 MB artifact is kept for reference only
(`kokoro-v1.0.int8.onnx`). If size ever becomes the binding constraint
(e.g. mobile), revisit with **static** QDQ quantization (calibrated, QDQ
format keeps fusion) or an int8-friendly export — do not reuse this file.

## Deliverables

| File | What |
|---|---|
| `quant_bench.py` | full pipeline: quantize -> RTF workers -> scorer -> verdict (rerunnable) |
| `results.json` | all measurements incl. per-rep walls |
| `kokoro-v1.0.int8.onnx` | quantized model, 87.69 MB (reference only) |
| `evalA_fp32.wav` / `evalA_int8.wav` | frozen-sentence renders used for scoring |
| `run.log` | parent orchestration log |

Reproduce: `.venv/Scripts/python.exe experiments/CV-G1-E0024-int8/quant_bench.py`
(delete `results.json`, `kokoro-v1.0.int8.onnx`, `*.wav` first for a clean run).
