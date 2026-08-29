# CV-G1-E0026-graph-opt — Conclusion

Date: 2026-08-23/24 · Windows 11, Intel Meteor Lake (16 logical CPUs) ·
onnxruntime 1.29.0 · onnx 1.22.0 · kokoro-v1.0.onnx (310.4 MB, 2464 nodes,
554 initializers) · frozen sentence "Hello. This is the first generation of
Caedral Voice.", voice=af_sky · fresh subprocess per config, 1 discarded cold
call, 2x10 warm reps pooled per config, two rounds in opposite order per
thread tier to cancel thermal drift (E0024 protocol).

## Verdicts

| Technique | Quality gate | RTF @t6 (pooled) | Verdict |
|---|---|---|---|
| Manual zero-loss cleanup (`artifacts/optimized/kokoro-conservative.onnx`: dedup 8 initializers + dead prune) | BITWISE IDENTICAL (max diff 0.0, 4 input cases incl. real frontend tokens) | 0.6042 vs fp32 0.6723 (+11% vs spiked baseline; clean-round pairing only ~+1.7%) | **KEEP as hygiene only — no real speed gain** |
| onnxsim canonicalization (`artifacts/optimized/kokoro-opt.onnx`) | FAIL: max abs diff up to 0.104 (r>=0.9964); ulp-level reassociation chaotically amplified through transformer+vocoder. Default pass set even breaks ORT load (Gemm_11 shape unification); had to disable transpose-fusion passes to load at all | 0.6072 | **KILL** |
| Static-shape buckets L100/L300/L512 (pure dim edits of the exact base, no resim) | FAIL strict gate vs natural render (max diff up to 0.43 from ORT plan changes under static shape declarations) | 0.858 / 2.04 / 3.03 vs dynamic-natural 0.567 (-34/-72/-81%) | **KILL decisively** |

## Why nothing moved

ORT at ENABLE_ALL already constant-folds and eliminates redundant ops at
load time; the graph census found no naive redundancy to remove (0 unused
initializers, 8 duplicate initializers totalling 52 bytes, 0
Identity/Dropout nodes, no Transpose->Transpose or Cast->Cast pairs, 12
unused LSTM state outputs). The 51 Sin ops are Gemm-fed Snake activations,
not foldable tables. There is no zero-loss graph-level headroom in this
export; session.run remains compute-bound in conv/attention kernels.

## Gate status

Target RTF <= 0.40 NOT met by any graph-level technique (best same-session
config 0.5669 raw dynamic @t6). Note: today's absolute RTFs (fp32 0.63-0.73
@t6) sit far above the recorded 0.4575 baseline purely from machine thermal
state (E0021 documented this scatter); same-session interleaved deltas are
the valid comparison. Reaching 0.40 needs kernel-level work (quantization
that preserves quality, thread/affinity tuning, or a faster backend), not
graph surgery.

## Incidental findings

- `encoder.bert.embeddings.position_embeddings.weight` is [512,128]:
  NO variant of this model can process >512 window positions. The planned
  LONG=600 bucket is architecturally impossible; LONG was capped at 512.
- A parallel session wrote a premature `results.json` stub mid-run
  (single-round cherry-picks, e.g. claiming opt@t6 "+3%"); it has been
  replaced by this experiment's authoritative pooled-median results.json.
- Bucket equivalence failures are execution-order noise amplified
  chaotically (uniform diff across the timeline), not semantic breaks -
  but they still fail the task's strict preservation rule.

## Deliverables

- `artifacts/optimized/kokoro-conservative.onnx` - bitwise-exact cleaned model (KEEP)
- `artifacts/optimized/kokoro-opt.onnx` - onnxsim v2 (KILL, kept for record)
- `artifacts/optimized/kokoro-static-L{100,300,512}.onnx` - buckets (KILL, for record)
- `results.json` (pooled RTF, deltas, targets, graph stats, equivalence, verdicts)
- Pipeline scripts: inspect_graph.py, optimize.py, make_conservative.py,
  repair_sim.py, make_buckets.py, equivalence.py, bench_worker.py,
  run_bench.py, finalize_results.py, diff_signature.py
