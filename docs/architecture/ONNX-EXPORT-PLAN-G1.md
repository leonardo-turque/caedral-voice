# ONNX Export & Validation Plan — CAEDRAL-NAR (G1)

Task RUNTIME-02. Target: `caedral_voice/model/*` (CV-G1-E0002 deterministic
path and CV-G1-E0006 flow path) to ONNX Runtime CPU, batch 1.
Evidence labels follow repo discipline: [MEASURED] / [ESTIMATED] / [HYPOTHESIS].

## 0. Verified environment + op-support facts

[MEASURED 2026-08-22, project venv]

| Fact | Value | Source |
|---|---|---|
| torch / torchaudio | 2.6.0 | `.venv` |
| onnxruntime | 1.29.0 (CPU) | `.venv` |
| onnx | 1.22.0 | `.venv` |
| ONNX spec FFT ops | STFT @17, DFT @17/20; **no ISTFT op exists in any opset** | onnx.ai/onnx/operators index (cached, WEB-SOURCE-INDEX) |
| ORT 1.29 registry scan (`get_all_operator_schema`) | 637 ai.onnx ops incl. STFT-17 and DFT with working CPU kernels (both executed in smoke graphs); `com.microsoft::Irfft/Rfft` exist; **ISTFT: NONE** | local smoke test |
| `torch.onnx.export(IstftHead, opset=18)` | FAILS: unsupported `aten::complex`; dynamo exporter blocked (`onnxscript` absent) | local probe |

Consequence: the iSTFT cannot ship as a single op anywhere; it must be
decomposed into real arithmetic or kept in torch (Section 3, H2).

## 1. Component-wise export order

Three stages, dependency order = risk order. Each stage has an export gate;
do not start stage N+1 until stage N passes its parity gate (Section 4).

```
phoneme ids ─► [A frontend] ─► h, s, log_d, m_p, logs_p ──(host: expand by durations,
               static buckets {50,200,800}                    build z from noise)──┐
                                                                                ▼
                              [B prosody+flow-inverse] ◄── h_exp, s, z ─────────┘
                              dynamic T_y ──► mel (T_y,80), f0_hz, energy_db
                                                                                ▼
                              [C iSTFT head] mel (1,80,T_y) ──► wav (1,T_y*256)
```

The length regulator (`expand_tokens`, variance.py) is **host-side numpy**
between A and B (H3). Sampling `z ~ N(m_e, ns·sigma_e)` is host-side too:
the graph never sees randomness (H5).

### Stage A — text encoder + variance (first)

Wrapper module (`InferenceFrontend(nn.Module)` to be added under
`runtime/export/`) tracing: phoneme_embed → encoder Transformer →
ConditioningProvider → DurationPredictor → flow_prior_proj (chunk m_p/logs_p).
**Static shapes per bucket**: T_x ∈ {50, 200, 800}; export one graph per
bucket (`frontend-t50.onnx`, `-t200`, `-t800`). Static shapes let ORT
constant-fold RoPE tables and pre-shape every GEMM.

Inputs/outputs (int64 ids; scalars as tensors):

| role | name | shape/dtype |
|---|---|---|
| in | phoneme_ids | (1, T_b) int64 |
| in | speaker_id | (1,) int64 |
| in | style_id | (1,) int64 (neutral = last row index) |
| out | h | (1, T_b, d_model) f32 |
| out | s | (1, speaker_dim+style_dim = 192) f32 |
| out | log_duration | (1, T_b) f32 |
| out | prior_mean m_p / prior_logscale logs_p | (1, T_b, 80) ×2 |

Gate A: parity ≤1e-4 max-abs on all outputs, integer-exact durations after
exp/clamp, at all three buckets (Section 4).

### Stage B — flow decoder inverse path (second)

Wrapper tracing: PitchPredictor(h_exp) → EnergyPredictor(h_exp) →
FlowDecoder.inverse(z). Only `inverse` is exported — the forward
(data→latent) direction and MAS live only in training (H1).

| role | name | shape/dtype |
|---|---|---|
| in | h_expanded | (1, T_y, d_model) f32, **dynamic T_y** |
| in | s | (1, 192) f32 |
| in | z | (1, 80, T_y) f32 (host-built: m_e + eps·exp(logs_e)·noise_scale) |
| out | mel | (1, T_y, 80) f32 (mask-multiplied equivalent handled host-side) |
| out | f0_hz / energy_db | (1, T_y) f32 (knobs applied host-side post-hoc, or fold shift/gain in as constant before export) |

Dynamic T_y is safe here because the whole stage is convolutional + matmul —
no attention, no position tables. The E0002 non-flow variant would instead be
a FiLM transformer decoder with dynamic RoPE (see H7 variant note).

Gate B: mel parity ≤3e-4 max-abs vs torch across lengths {37, 120, 500, 1500}
frames with identical z inputs.

### Stage C — iSTFT head (third; **decision: keep torch for v0**)

**v0 justification** [HYPOTHESIS → verify at Phase G gate]: the head owns
~2% of params and `torch.istft` is one native CPU call, so the hybrid
(ONNX A+B, torch head) already removes all Python-side model glue from the
hot path while avoiding hand-rolled DSP before trained weights exist; the
full-ONNX decomposition below is scheduled for v1 and is proven feasible by
kokoro-onnx shipping istft-decomposed Kokoro graphs (package present in venv).

v1 decomposition (all real arithmetic, opset-13-safe ops):

1. mag = exp(mag_head(x)); phase = atan2(im, re) — exports cleanly.
2. spec_re = mag·cos(phase), spec_im = mag·sin(phase); rebuild the full
   1024-bin spectrum from 513 onesided bins via Slice + Neg(imag) + Concat
   (conjugate-symmetric flip).
3. Inverse DFT per frame: either DFT-17 (`inverse=1, onesided=0`) — kernel
   MEASURED working on ORT 1.29 CPU — or two matmuls against baked cos/sin
   tables (1024×1024 f32 ≈ 8 MB initializers).
4. Overlap-add: reshape frames to (B, n_fft, T) and run ConvTranspose1d with
   weight W[c=n, 0, k=n] = hann_window[n], stride=hop_length → exact
   windowed OLA; second ConvTranspose over ones with W² gives the
   sum-of-squared-windows denominator; divide; crop n_fft/2 both ends,
   trim to T·hop (`center=True` semantics).

Gate C: waveform parity vs torch head ≤1e-3 max-abs + Pearson r ≥ 0.99999.

## 2. Opset + dynamic-axis table (per component)

Uniform **opset 17** everywhere (minimum for STFT/DFT; everything else used
is ≤13; ORT 1.29 supports far newer but kernel maturity favors 17).

| Graph file | Opset | Input axes | Dynamic axes | Notes |
|---|---|---|---|---|
| frontend-t{50,200,800}.onnx | 17 | ids (1,T_b) int64; spk/sty (1,) int64 | none — fully static | RoPE folded to constants |
| prosody-flow-inverse.onnx | 17 | h_exp (1,T_y,d_model); s (1,192); z (1,80,T_y) | T_y on all frame axes | convs+matmul only |
| istft-head.onnx (v1) | 17 | mel (1,80,T_y) | T_y (+ derived T_wav=T_y·256 output axis) | decomposed per Section 1-C |

Export settings: `opset_version=17`, legacy TorchScript exporter (dynamo
exporter unavailable — onnxscript not installed [MEASURED]),
`do_constant_folding=True`, no training/eval ambiguity (dropout=0 configs,
`.eval()` asserted).

## 3. Known export hazards for this architecture

| # | Hazard | Status / mitigation |
|---|---|---|
| H1 | MAS is TRAINING-only | `maximum_path` (mas.py) is a numpy DP with Python loops and `.tolist()` control flow — untraceable and never needed at inference. Export wraps ONLY `infer()` components; add an assert that `losses_for_training` symbols are absent from traced graphs. Same for `FlowDecoder.forward` (only `inverse` exported). |
| H2 | ISTFT op support | No ISTFT op in ONNX or ORT [MEASURED, Section 0]; `torch.onnx.export(IstftHead)` fails on `aten::complex`. Mitigation: v0 keeps head in torch; v1 uses the real-arithmetic decomposition (Section 1-C). Do NOT attempt STFT-op-based round-trips. |
| H3 | Length regulator `expand_tokens` | Python loop over batch, `.tolist()` (host sync), per-item `repeat_interleave`, pad-to-batch-max — untraceable. Primary mitigation: host-side expansion between graphs A and B (numpy `np.repeat`, piper/VITS-ONNX precedent). Fallback (if single-graph ever required): in-graph gather-index construction via cumsum + repeat_interleave decomposition — needs its own verification spike; do not assume it works. |
| H4 | `torch.linalg.solve_triangular` in Invertible1x1Conv.weight() | No ONNX symbolic. Weights are frozen at export: compute W and W_inv once in Python and bake both as constant initializers (C≤128 → negligible size). Never trace `weight()`. |
| H5 | `torch.randn_like` sampling in infer() | Non-exportable and breaks determinism/parity. Host-side: sample eps once, feed identical z to torch and ORT. |
| H6 | Python scalar args (speaker_id/style_id/speed/pitch_shift/energy_gain/noise_scale) | Convert to int64/f32 tensor inputs; knobs become either inputs or export-time constants. No `.item()`/float() in traced region. |
| H7 | RoPE tables built inside forward (`arange`/`outer`) | Exports to Range/Cos/Sin/Mul — fine, and constant-folds under static buckets. Variant note: the E0002 non-flow decoder with dynamic T_mel would need dynamic RoPE construction; acceptable ops, but re-validate if that variant ships. GQA `repeat_interleave` over heads is static-dim — safe. SDPA exports decomposed MatMul+Softmax at opset 17 — safe. |
| H8 | Bucket padding semantics | Encoder has no padding mask; padded positions influence nothing downstream because host-side expansion consumes true durations only. Parity runs must feed byte-identical padded id arrays to both runtimes. |

## 4. Numerical parity protocol (torch vs ORT)

Fixtures: seeded model per config (tiny/lite/quality), fixed input set of
true phoneme lengths L ∈ {10, 50, 120, 200, 500, 800}, B=1, zero-padded into
the containing bucket; identical speaker/style ids; identical z (same eps)
for flow comparisons. Reference = eager torch fp32 CPU eval; candidate =
ORT CPUExecutionProvider, graph opt ALL.

Per-component metrics: max-abs-diff (primary gate), mean-abs-diff, p99-abs
(recorded, not gated); waveform additionally Pearson r and log-mel L1 using
the project's own mel transform.

| Component output | Gate (max-abs-diff) | Rationale |
|---|---|---|
| h (encoder states) | ≤ 1e-4 | ~10 GEMM/LN layers of fp32 reorder drift |
| log_duration, m_p, logs_p | ≤ 1e-4 | shallow heads on gated h |
| durations after exp/round/clamp | integer-exact | rounding must agree |
| mel (flow inverse, 6 blocks) | ≤ 3e-4 | deep affine stack amplifies drift |
| f0_hz, energy_db | ≤ 1e-3 | exp() amplification on tracks |
| waveform (hybrid v0) | ≤ 1e-3 and r ≥ 0.99999 | OLA accumulation over 1024-pt frames |

Procedure: `runtime/check_parity.py` (to be created by implementer) writes
`results/onnx/parity-<stamp>.jsonl` + `.meta.json` (machine header via
harness `capture()` + ort version/providers). Escalation rule: on any gate
failure, bisect by dumping intermediates (per-block outputs of the flow) from
subgraph exports until the diverging op is isolated; record findings in the
parity JSONL row.

## 5. Benchmark harness extension points (benchmarks/harness)

No breaking changes needed — the harness was built runtime-agnostic:

- `benchmarks/harness/measure.py`: `measure(fn)` / `rtf(wall, audio_s)` wrap
  any callable — pass a closure around `sess.run(...)` unchanged;
  `PerfResult` fields (incl. Win32 `native_peak_rss_bytes`) apply verbatim.
- `benchmarks/harness/system_info.py`: `capture()` supplies the meta header;
  runner adds runtime keys itself: `{"runtime": "onnxruntime",
  "ort_version": ort.__version__, "providers": [...],
  "intra_op_num_threads": N}` next to `torch_num_threads`.
- Runner pattern to copy: `benchmarks/run_decoder_vocos.py` — one process
  per benchmark, threads arg in argv[1], warmup run before timing, outputs
  `benchmarks/results/onnx/<name>-<stamp>.jsonl` + `.meta.json`.
- Planned runners (implementer): `benchmarks/run_caedral_onnx.py` (RTF per
  bucket + full pipeline hybrid vs pure-torch baseline, thread sweep
  {1,2,4}); session options to sweep: intra_op_num_threads,
  execution_mode sequential, `ort.SessionOptions().graph_optimization_level`.

## 6. Milestones / exit criteria

| M | Deliverable | Exit criterion |
|---|---|---|
| M1 | Stage A graphs ×3 buckets + parity report | Gate A pass, durations integer-exact |
| M2 | Stage B graph + host glue (expansion, sampling) | Gate B pass at 4 lengths |
| M3 | Hybrid v0 end-to-end RTF measured | results/onnx rows vs torch baseline; RTF ≤ torch baseline [HYPOTHESIS: ORT wins on glue elimination alone] |
| M4 | v1 istft-head.onnx decomposition | Gate C pass; then re-run M3 end-to-end |

Open questions: whether bucket padding should switch to masked attention
before training freeze (model-level decision, affects export only through
input contract); whether quality config (GQA 16/8) needs the Attention fused
op at newer opset for speed (measure in M3 before deciding).
