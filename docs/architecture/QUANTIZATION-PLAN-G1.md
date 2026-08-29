# Quantization Plan G1 — Phase M experiment matrix

**Task:** CPU-QUANT-01 · **Status:** PLANNED (pre-Phase-M; execute after Phase L profile-first pass lands)
**Date:** 2026-08-22 · **Inputs:** `docs/research/CPU-INFERENCE-OPTIMIZATION.md` (cached research, cited as R-§n),
`docs/benchmarks/BASELINE-MEASURED.md`, `docs/architecture/ARCH-NOTES-G1.md`, `experiments/CV-G1-E0002` / `E0006`.
**Architecture (frozen direction):** CAEDRAL-NAR — phoneme text encoder + duration/pitch/energy
predictors + conditional flow decoder (MAS-trained, few-step) + iSTFT head.
Tiers: **Tiny 8.35M / Lite 31.6M / Quality 80.7M** [MEASURED, E0002].
**Target silicon:** i7-13620H-class client x86-64 — AVX2/FMA3, AVX-VNNI (per Intel ARK [REPORTED];
unverified by local tooling), **no AVX-512**, hybrid 6P+4E [MEASURED].

Every number carries a label per charter §9: MEASURED / REPORTED BY THIRD PARTY (hereafter REPORTED,
traceable via R-§n sources) / ESTIMATED / HYPOTHESIS. Nothing below is measured until Phase M runs it.

---

## 1. The four strategies, as concretely implemented

| Column | ORT implementation | What it buys | What it does NOT buy |
|---|---|---|---|
| S0 — fp32 baseline | Exported ONNX, offline-optimized graph (`optimized_model_filepath`), reloaded with `ORT_DISABLE_ALL`; session knobs per R-§a (SEQUENTIAL, spin 1000µs/backoff 8 desktop · 4 hybrid, bucketed shapes, arena on) | Reference for all deltas; zero quality risk | nothing |
| S1 — static INT8 QDQ, per-channel | `quantize_static` with `QuantFormat.QDQ`, activations **QUInt8** × weights **QInt8**, `per_channel=True`, `reduce_range=False`, MinMax calibration (~256 utterances); op scope starts MatMul/Gemm, Conv admitted per-op only after validation (§4) | Real RAM reduction (int8 weights resident) + the only int8 path that can beat fp32 on our graphs [R-§3.1/3.2] | Not automatic: conv-heavy cells carry the vocoder-int8 trap (§7) |
| S2 — weight-only INT8 | Two routes, both wheel-only: (a) **file-size route** — int8 initializers + DequantizeLinear feeding fp32 Gemm; ORT folds DQ-at-load ⇒ disk −75%, **runtime RSS ≈ fp32** [HYPOTHESIS — verify constant-folding keeps dequantized copy resident]; (b) **dynamic-int8 scoped strictly to MatMul/Gemm** (`quantize_dynamic`, MatMulInteger) — genuine resident int8 on those weights, but integer-compute path with per-pass activation quant [R-§3.1] | Distribution/download size; (b) adds modest RSS win | Speed ≈ neutral (±5–10%) [ESTIMATED]; (b) is forbidden on Conv/iSTFT cells (trap, §7) |
| S3 — weight-only INT4 | `onnxruntime.quantization.matmul_4bits_quantizer` → blockwise `MatMulNBits` (com.microsoft), MatMul/Gemm only; int4 stays packed resident | Largest size/RAM cut (−85–90% on covered weights) [ESTIMATED] | No native broad int4 kernels beyond MatMulNBits; dequant-in-kernel cost ≈ cancels bandwidth win at batch-1 ⇒ RTF neutral ±10% [ESTIMATED]; highest quality risk |

FP16 note (applies everywhere): without AVX-512-FP16 there is **no native fp16 compute** on our fleet;
fp16 files are a *storage* format only, ≈fp32 RTF [R-§3.6, REPORTED sherpa 0.123 vs 0.114]. FP16-storage
is the sanctioned size lever for any cell where int8/int4 fails its gate.

## 2. Component inventory and share model

Wall-time and param shares are placeholders until Phase M step M0 instruments per-stage timers
(R-§b step 1 discipline: never quantify blind).

| Component | Est. share of pipeline wall-time | Est. param share (tier-dependent) | Compute character | Quantization prior [R-§3.4/3.7] |
|---|---|---|---|---|
| C1 text encoder | 10–15% [ESTIMATED] | ~15–25% [ESTIMATED] | MatMul/attention-heavy | Best case for VNNI int8; robust to PTQ |
| C2 duration+pitch+energy predictors | 5–10% [ESTIMATED] | ~10–15% [ESTIMATED] | small MLP/convs, run once per utterance | "int8 freely" [R-§3.7] |
| C3 flow decoder (NFE × full-frame passes, 2–4 Euler steps) | 55–70% [ESTIMATED] | ~45–60% [ESTIMATED] | big dense GEMMs + some conv branches | Main FLOPs pool; validate per-op |
| C4 iSTFT head | 10–20% [ESTIMATED] (Kokoro lesson: decoder op mix sets the NAR floor [R-ACOUSTIC-§10.2]) | ~10–20% [ESTIMATED] | transposed/1-D convs + FFT overlap-add | **Vocoder-int8 trap territory — do NOT naive-PTQ** |

Share-model arithmetic used throughout (mid-shares, quantized pool ≈ 80% of wall time):
component-level int8 speedup 1.5–2.0× on GEMM-bound layers with AVX-VNNI [ESTIMATED from R-§3.5;
server-class 2.97× geomean does NOT transfer — half-width VNNI, no AMX] ⇒ whole-pipeline
**RTF ×0.60–0.73** (−27–40%) when C1+C2+C3-MatMul quantize cleanly and C4 stays fp32 [ESTIMATED].

## 3. Core deliverable — per-component strategy matrix

Expected deltas are **relative to the same tier's S0 fp32 arm**, same session config, protocol §c.
Negative RTF % = faster. Verdicts: **GO** (default adopt if gates pass) · **COND** (conditional,
see note) · **NO-GO** (do not spend an experiment slot without a named trigger).

### C1 — text encoder

| Strategy | Expected RTF (pipeline) | Expected RAM/file | Verdict |
|---|---|---|---|
| S0 fp32 | 1.00× ref | ref | GO (reference arm) |
| S1 static int8 QDQ pc | ×0.88–0.96 (component 1.5–2.0×) [ESTIMATED] | weights −~75% of C1 bytes; arena unchanged | **GO — first full-graph candidate** (transformer MatMuls = VNNI best case [R-§3.7]) |
| S2 weight-only int8 | ×0.95–1.05 [ESTIMATED] | file −75%; RSS −75% only via route-(b) [HYPOTHESIS] | COND — size lever for profiles C/D/E only |
| S3 weight-only int4 | ×0.95–1.10 [ESTIMATED] | file/RAM −85–90% of C1 GEMMs | COND — profile E only |

### C2 — duration/pitch/energy predictors

| Strategy | Expected RTF (pipeline) | Expected RAM/file | Verdict |
|---|---|---|---|
| S0 fp32 | 1.00× ref | ref | GO (reference) |
| S1 static int8 QDQ pc | ×0.93–0.97 (small stage, near-zero risk) [ESTIMATED] | −75% of tiny C2 bytes | **GO — cheapest cell, run first as pipeline shakedown** [R-§3.7 "int8 freely"] |
| S2 weight-only int8 | ×0.98–1.02 [ESTIMATED] | negligible absolute bytes | COND (bundle with C1/C3 size builds) |
| S3 weight-only int4 | ~1.0× [ESTIMATED] | negligible | **NO-GO** — risk not worth megabytes |

### C3 — flow decoder

| Strategy | Expected RTF (pipeline) | Expected RAM/file | Verdict |
|---|---|---|---|
| S0 fp32 | 1.00× ref | ref | GO (reference) |
| S1 static int8 QDQ pc, **MatMul/Gemm-scoped** | ×0.62–0.78 (dominant stage at 1.5–2.0×) [ESTIMATED] | −~50% of tier's total weights | **COND — the main prize; per-op validation mandatory**: admit Conv branches only cell-by-cell; exclude first flow block, last output projection, and any ConvTranspose/upsample on drift [R-§3.3 debug workflow] |
| S1 naive (all ops incl. conv branches) | unpredictable; trap says slowdown/artifacts possible [REPORTED R-§3.4] | — | **NO-GO** as a configuration (may be run once as a recorded negative) |
| S2 weight-only int8 | ×0.95–1.05 [ESTIMATED]; route-(b) may add small win on memory-bound GEMV-shaped passes (weight-stream floor 7.9→~2.0 ms/pass at 80M [MEASURED-derived ARCH-NOTES]) | file −75%; RSS −75% route-(b) | COND — size lever C/D/E |
| S3 weight-only int4 (MatMulNBits) | ×0.95–1.10 [ESTIMATED] | −85–90% of C3 GEMMs | COND — **profile E headline experiment** |

### C4 — iSTFT head

| Strategy | Expected RTF (pipeline) | Expected RAM/file | Verdict |
|---|---|---|---|
| S0 fp32 | 1.00× ref | ref | **GO — default forever unless a profile forces change** |
| S1 static int8 QDQ | expected **slower and/or audibly degraded** [REPORTED: HiFi-GAN dyn-int8 ~2× slower + heavy quality loss R-§3.4[30]; VITS int8 RTF 0.114→0.320 R-[32][33]; "do not int8-quantize the vocoder convs" R-§3.4] | — | **NO-GO (naive PTQ)**. Only legitimate route = QAT fine-tune on paired audio — out of Phase M scope; revisit only if profile D/E cannot close budget otherwise (documented trigger, not planned work) |
| S2 weight-only int8 / fp16-storage | ≈1.0× [REPORTED fp16 ≈fp32 RTF R-§3.6] | file −75%/−50% | COND — allowed for distribution-size only; compute stays fp32 |
| S3 weight-only int4 | n/a (conv head out of MatMulNBits scope) | — | **NO-GO — structural** |

## 4. Whole-tier projections (post-gate, recommended configs)

Config shorthand: **INT8-AC** = S1 on C1+C2+C3(MatMul/Gemm), C4 fp32.

| Tier | Config | File size fp32 → int8 → int4 [ESTIMATED params×bytes] | Expected warm RTF multiplier [ESTIMATED] | Expected peak-RSS effect [ESTIMATED] |
|---|---|---|---|---|
| Tiny 8.35M | INT8-AC | ~33 MB → ~8.5 MB → ~5 MB | ×0.65–0.80 | weights ~25 MB → ~6 MB resident (S1); torch-floor caveat: eager prototype measured 235 MB RSS [MEASURED E0002] — ONNX floor to be re-measured |
| Lite 31.6M | INT8-AC | ~126 MB → ~32 MB → ~17 MB | ×0.60–0.73 | weights ~126 MB → ~34 MB; prototype 334 MB RSS eager [MEASURED E0006] |
| Quality 80.7M | INT8-AC (profile A optional) | ~323 MB → ~81 MB → ~43 MB | ×0.60–0.73 | ~230 MB RSS saved vs fp32; prototype 528 MB [MEASURED E0002] |

Arena caveat: peak RSS at long inputs is dominated by the arena high-water mark, not weights
[R-§5.1]; bucketed shapes (`mem_pattern`) + `initial_growth_chunk_size_bytes` + shrinkage RunOption
are part of every arm's config so quantization deltas stay isolated [R-§1.3].

## 5. ORT toolchain commands (templates — verify exact flags against installed ORT wheel)

```bash
# 0) Pre-quantization hygiene — shape inference + optimization BEFORE the quantizer [R-§3.3]
python -m onnxruntime.quantization.quant_pre_process \
    --input caedral_lite.fp32.onnx --output caedral_lite.pre.onnx

# 1) Static INT8 QDQ per-channel (S1) — python API
python tools/quantize_qdq.py --in caedral_lite.pre.onnx --out caedral_lite.int8.onnx \
    --calib manifests/calib-phonemes-v1.jsonl --ops MatMul,Gemm   # Conv added per-op later
```

```python
# tools/quantize_qdq.py core
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      CalibrationMethod, QuantFormat, QuantType)

class PhonemeCalibReader(CalibrationDataReader):
    """Yields forward-input dicts over ~256 phoneme sequences spanning the
    length buckets {<=50, 200-400, 800-1200 chars} of eval-corpus-v1.0."""
    ...

quantize_static(
    model_input="caedral_lite.pre.onnx",
    model_output="caedral_lite.int8.onnx",
    calibration_data_reader=PhonemeCalibReader(),
    quant_format=QuantFormat.QDQ,            # QDQ, not QOperator — S8S8+QOperator is slow on x86-64 [R-§3.2]
    op_types_to_quantize=["MatMul", "Gemm"],
    per_channel=True,                        # accuracy where weight ranges vary [R-§3.3]
    reduce_range=False,                      # AVX-VNNI present -> full-range U8S8 [R-§3.2]
    activation_type=QuantType.QUInt8,        # U8 activations x S8 weights, VNNI VPDPBUSD [R-§3.2]
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    nodes_to_exclude=EXCLUDE_LIST,           # first flow block, final projection, offender list
)
```

```bash
# 2) Weight-only INT4 (S3) — blockwise MatMulNBits, MatMul/Gemm scope
python -m onnxruntime.quantization.matmul_4bits_quantizer \
    --input caedral_tiny.pre.onnx --output caedral_tiny.int4.onnx \
    --block_size 32        # try {32,64,128}; record quality per block size

# 3) Ship pre-optimized graph, reload inert (session-init cut) [R-§1.4]
python - <<'EOF'
import onnxruntime as ort
so = ort.SessionOptions(); so.optimized_model_filepath = "caedral_lite.int8.opt.onnx"
ort.InferenceSession("caedral_lite.int8.onnx", so)
EOF
# serving load: graph_optimization_level=ORT_DISABLE_ALL on the .opt.onnx

# 4) Numerical parity check before any benchmark: max-abs-diff of mel/waveform
#    fp32 vs quantized on 16 corpus items; investigate >1e-2 mel-scale outliers first.
```

Debug workflow when a gate fails [R-§3.3]: `create_weight_matching()`,
`collect_activations()`, `modify_model_output_intermediate_tensors()` to rank offending tensors →
add to `nodes_to_exclude` → requantize. One exclusion round maximum per experiment slot; deeper than
that = verdict "component does not survive PTQ", recorded honestly.

## 6. Quality gates (attached to every perf claim; speed claims without them are rejected [R-§c])

| Gate | Metric | Threshold | Harness |
|---|---|---|---|
| Q1 intelligibility | dual-ASR normalized minWER (whisper small.en + base.en int8, frozen scoring stack) | **Δ ≤ +0.02 absolute** vs same-tier fp32 arm, paired mean over frozen `benchmarks/corpus/eval-corpus-v1.0.json`; acr_005 scoring-rule note applied identically both arms | `benchmarks/score_baseline_samples.py` pattern |
| Q2 artifact floor | DNSMOS v8 ovrl (floor-check semantics; ceiling ≈3.3 known [R-EVAL-§1.3]) | **Δ ≥ −0.05** paired mean vs fp32; also report worst decile | DNSMOS ONNX runner |
| Q3 robustness | long-form torture >1500 chars completes; steady RTF within 15% of medium bucket | pass/fail | protocol §c torture case |
| Q4 human | internal MOS-lite panel (≥2 raters × ≥10 A/B pairs) | no perceived regression on shipped-default candidates | manual, scheduled |

Secondary alarm: UTMOS-strong (backlog item) logs always, never gates alone — saturation/prosody-blindness
documented [R-EVAL-§0/§1.1]. Gate evaluation runs cold+warm per protocol §c: threads reported with every
number, RSS sampled 10 Hz, p99/p50 variance check, fresh process per cold run.

## 7. Traps this plan is built around

1. **Vocoder-int8 trap (primary):** dynamically/naively int8'd GAN/iSTFT vocoders measured ~2–3×
   *slower* than fp32 with heavy quality loss [REPORTED R-§3.4: OpenVINO#12964 HiFi-GAN; sherpa-onnx
   #575/#2460 VITS]. Hence C4-S1 = NO-GO, C3 conv branches gated per-op, and all S2-dynamic routes
   scoped away from Conv.
2. **Fake-int8 speedups:** comparing fp32 against a dynamic-quint8 path that is actually slower
   [R-§1.5.5]. Mitigation: every arm timed under identical session/thread config, cold/warm split.
3. **Per-channel × reduce_range interaction:** per-channel may force reduce_range=True on some AVX2
   kernel paths [R-§3.3]; we pin reduce_range=False + VNNI and verify kernel selection in verbose log
   per arm; if a kernel silently needs reduce_range, that cell is re-gated.
4. **RSS mirage:** weight-only routes can leave dequantized fp32 copies resident after constant folding
   [HYPOTHESIS §1-S2a] — measure resident RSS, not just file size, before claiming memory wins.
5. **Session-per-request / warmup contamination** invalidating A/Bs [R-§1.5.1] — driver reuses one
   session per arm, 3 warmup runs discarded.

## 8. Experiment sequence (working tags QM-nn; assign CV-G1-E#### IDs from EXPERIMENT-INDEX at Phase M start)

| Tag | Content | Gate |
|---|---|---|
| QM-01 | Per-stage timer instrumentation on exported fp32 graphs → replace §2 share placeholders with MEASURED shares | shares published |
| QM-02 | S0 export + optimized-graph shipping + parity check, all three tiers | mel parity <1e-2 max-abs |
| QM-03 | S1 on C2 (shakedown) then C1 | Q1–Q3 |
| QM-04 | S1 on C3 MatMul/Gemm scope; per-op Conv admission rounds (max 1 exclusion round) | Q1–Q3 |
| QM-05 | C4 decision record: fp32 confirmed; fp16-storage size build | Q4 if promoted anywhere |
| QM-06 | S2 builds (routes a/b) for profiles C/D/E size targets | Q1–Q2 + RSS measurement |
| QM-07 | S3 MatMulNBits on Tiny (profile E headline); block-size sweep {32,64,128} | Q1–Q3, expectation set LOW |
| QM-08 | Freeze per-profile configs (§9) + concurrency certification per R-§7.3 | published safe-concurrency |

Negative results are recorded in EXPERIMENT-INDEX.md like any other (charter §9.4).

## 9. Go/no-go rules per deployment profile

Profiles per MASTER-PLAN §5. "Gates pass" = Q1 ∧ Q2 ∧ Q3 (Q4 for any shipped default).

| Profile | Tier/config under test | GO requires (all of) | NO-GO / fallback action |
|---|---|---|---|
| **A** Max quality (4–8 vCPU, ≤4 GB) | Quality 80.7M; S0 default; S1 INT8-AC challenger | gates pass ∧ warm RTF improves **≥10%** vs S0 ∧ peak RSS ≤700 MB | stay fp32 (fits 4 GB easily); WO-int8 file build still allowed for distribution |
| **B** Economical prod (2 vCPU, ≤2 GB) | Lite 31.6M INT8-AC (C4 fp32) | gates pass ∧ warm RTF ≤0.50 @2 thr ∧ peak RSS ≤400 MB | fallback chain: partial-int8 (C2+C1 only) → fp32 Lite; each step re-gated, measured minimum documented |
| **C** Ultra-low-cost (1 vCPU, ≤1 GB) | Lite INT8-AC (alt: Tiny fp32) | gates pass ∧ warm RTF **<1.0 @1 thr** ∧ peak RSS ≤600 MB | if C3 fails gates: ship C1+C2-int8 + fp32 C3 and accept smaller win; Tiny-fp32 alternate config |
| **D** Torture (1 vCPU, ≤500 MB) | Tiny 8.35M INT8-AC | gates pass ∧ RTF <1.0 @1 thr ∧ peak RSS ≤300 MB incl. arena | if RSS floor (runtime overhead) blocks: publish measured minimum (charter §9.5); fp16-storage size variant |
| **E** Extreme research (256–500 MB) | Tiny S3 int4 (+S1 remainder); optionally head-QAT spike | gates pass ∧ fits cap — **expectation set LOW upfront** | failure is a valid result: publish actual minimum + which gate failed; no silent shipping of failing artifacts ever |

Global rules: no quantized artifact ships a speed claim without paired Q1/Q2 numbers [R-§c];
every profile publishes the winning thread/spin config alongside its RTF; re-verify after any ORT
upgrade (kernel-selection sensitivity).

---

### Source map (internal — reuse cached research only, per task constraints)

- R-§n = `docs/research/CPU-INFERENCE-OPTIMIZATION.md` section n (its own Sources list carries the external URLs; key anchors: §3.1 taxonomy/dynamic-int8 slowness, §3.2 QDQ/U8S8/VNNI, §3.3 per-channel + debug tools, §3.4 vocoder-int8 evidence, §3.5 client-VNNI 1.5–2.0× estimate, §3.6 fp16-storage-only, §5.1 arena/RSS, §c measurement protocol, §a session-config baseline).
- R-EVAL-§n = `docs/research/EVALUATION-BENCHMARKS.md` (DNSMOS floor-check semantics, metric saturation).
- MEASURED items: `docs/benchmarks/BASELINE-MEASURED.md`, `docs/architecture/ARCH-NOTES-G1.md`,
  `experiments/CV-G1-E0002-nar-prototype/`, `experiments/CV-G1-E0006-mas-flow-decoder/`.

*End of document — CPU-QUANT-01, 2026-08-22.*
