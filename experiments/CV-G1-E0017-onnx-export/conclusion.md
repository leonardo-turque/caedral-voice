# CV-G1-E0017 — ONNX inference-graph export of the CaedralRegressor

Task ONNX-REG-01 (2026-08-23). Implements the RUNTIME-02 plan
(`docs/architecture/ONNX-EXPORT-PLAN-G1.md`) for the E0013 B-arm model:
`phoneme_ids + durations + speaker/style -> mel`, exported with **random-init
weights** so a trained checkpoint drops in later with zero graph changes.
Script: `export_regressor.py`; parity/latency rows: `results/parity-20260823T005011Z.jsonl`
(+ `.meta.json` with full system capture, git rev `019acb6`).

**Verdict: COMPLETE — all gates PASS.** Three static opset-17 graphs exported
and validated; no export failures, no unsupported ops.

## Deliverables

| File | What |
|---|---|
| `regressor-t{50,200,800}.onnx` | static phoneme-bucket graphs, opset 17, ~60.7 MB each (fp32 weights of 15.91 M params), 1384 nodes |
| `export_regressor.py` | wrapper + exporter + parity/latency harness (rerunnable) |

## Wrapper contract (`RegressorOnnxWrapper`)

Inputs (B=1): `phoneme_ids (1,T_b) int64` right-zero-padded; `speaker_id (1,)`
int64; `style_id (1,) int64` (neutral = last table row = `n_styles`);
`durations (1,T_b) int64` frames/phone, pad slots 0. Outputs:
`mel (1,t_mel,80) f32` with **dynamic** `t_mel` (= sum of clamped durations)
and auxiliary `log_duration (1,T_b)` (predictor output kept so the runtime can
close the loop without a second graph; not part of the mel path).

Hazard mapping vs the plan: H1 (MAS/training path) never referenced — only
`_encode / conditioning / duration_predictor / _decode`; H3 (`expand_tokens`
python loop) replaced **in-graph** by a loop-free construction over primitive
ops — `ends=cumsum(d); src[p]=sum(pos>=ends); h_exp=Gather(h0,src,axis=0)`
(CumSum-11/Range-11/GreaterOrEqual-12/Gather-13; integer-exact through ORT at
arbitrary lengths per pre-probe); H4 n/a (no flow); H5 no randomness in-graph;
H6 speaker/style are int64 tensors, `max_duration=50` baked as constant;
H7 RoPE exports as Range/Cos/Sin and SDPA decomposes to MatMul+Softmax — both
stay shape-generic in `t_mel` (validated below); H8 identical byte-level padded
inputs fed to both runtimes. Pitch/energy projections are excluded exactly as
in eager v0 (`use_pitch/use_energy == False` asserted).

Deliberate deviation from the plan's primary H3 mitigation: expansion is
in-graph rather than host-side, which collapses stages A+B into ONE artifact
per bucket (three files total). Cost: an O(T_x·t_mel) int64 comparison matrix
transiently (~20 MB at t800/full) and a B=1 contract. With exact-sum durations
at B=1 the `forward()` mask multiply is identity, so it is omitted.

## Parity [MEASURED] — torch eager fp32 CPU vs ORT CPUExecutionProvider, gate ≤ 3e-4

| Bucket | Case | t_mel | audio s | mel max-abs | mel mean-abs | mel p99 | log_dur max-abs | Gate |
|---|---|---|---|---|---|---|---|---|
| t50 | full (T_x=50) | 504 | 5.38 | **2.38e-06** | 4.09e-07 | 1.31e-06 | 5.96e-07 | PASS |
| t50 | partial (35 valid) | 345 | 3.68 | **2.13e-06** | 3.89e-07 | 1.28e-06 | 8.35e-07 | PASS |
| t200 | full (T_x=200) | 1187 | 12.66 | **2.24e-06** | 3.91e-07 | 1.28e-06 | 9.54e-07 | PASS |
| t200 | partial (120 valid) | 1157 | 12.34 | **2.35e-06** | 3.73e-07 | 1.22e-06 | 1.04e-06 | PASS |
| t800 | full (T_x=800) | 2050 | 21.87 | **2.74e-06** | 3.90e-07 | 1.27e-06 | 1.61e-06 | PASS |
| t800 | partial (450 valid) | 2694 | 28.74 | **2.41e-06** | 3.70e-07 | 1.19e-06 | 1.43e-06 | PASS |

Worst case is ~110x under the 3e-4 gate (the plan reserved 3e-4 for flow-stack
drift; this decoder needs far less headroom). The "partial" cases exercise the
pad-zero duration convention through the in-graph `where(d>0, clamp, d)` guard.
Additionally, wrapper-vs-`CaedralRegressor.forward` cross-check on identical
inputs: **max-abs 0.0 (bitwise), frame counts agree** on all six cases — the
gather expansion reproduces `repeat_interleave` ordering exactly.

## ORT session smoke + latency [MEASURED] (informational)

SessionOptions: CPUExecutionProvider, `intra_op_num_threads=4`, sequential,
graph-opt ALL; torch pinned to 4 threads; warmup 2 + median of 8, full case:

| Bucket | torch eager ms | ORT ms | speedup | RTF torch | RTF ORT |
|---|---|---|---|---|---|
| t50 | 434.1 | 311.1 | ×1.40 | 0.081 | 0.058 |
| t200 | 1419.4 | 824.0 | ×1.72 | 0.112 | 0.065 |
| t800 | 2220.5 | 2024.7 | ×1.10 | 0.102 | 0.093 |

ORT wins everywhere on this unoptimized comparison; gains shrink as T_y grows
(ORThreaded GEMM efficiency vs torch MK-DNN at long sequences). Full-pipeline
RTF benchmarking per bucket/thread-sweep remains M3 work.

## Failed ops / export issues

None — first-pass export succeeded for all three buckets. Notes for the record:

- One benign TracerWarning (`variance.py:65`, `style_id.shape[0] !=
  speaker_id.shape[0]`): a shape-equality Python bool resolved to a constant
  True under the B=1 contract; the traced branch is the direct embedding path.
- Op inventory is primitive-only: MatMul 58, Gather 46, LayerNormalization 18
  (opset-17 fused), Erf-based GELU 10, Softmax 7, Conv 3, plus the expansion
  ops CumSum/Range/GreaterOrEqual (1 each). No custom/composite domains;
  single opset import `{ai.onnx: 17}`; IR version default.

## Checkpoint drop-in contract

The wrapper holds `CaedralRegressor` as a submodule, so `model.load_state_dict(...)`
followed by re-running `torch.onnx.export(...)` reproduces these graphs bit-for-
bit modulo weights — no structural edits needed after training. Static axes:
phoneme bucket T_b baked per file; everything downstream of durations is dynamic
(`mel` axis `t_mel` verified symbolic in all three graphs).

## Limitations / follow-ups

- B=1 only; batching would need padded expansion + the omitted output mask.
- In-graph expansion transient memory O(T_x·t_mel) — fine at CPU budgets here,
  revisit if bucket 800 ever runs with pathological uniform durations.
- `speed` knob is intentionally host-side (scale durations before feeding).
- iSTFT/vocos stays external per plan Stage C decision (E0017 scope ends at mel).
