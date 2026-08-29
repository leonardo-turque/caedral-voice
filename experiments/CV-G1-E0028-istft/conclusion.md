# CV-G1-E0028-istft — Splitting the fused iSTFT out of the Kokoro ONNX graph

**Task:** ISTFT-SPLIT-01 · **Date:** 2026-08-23 · **Verdict: KILL (speed) / KEEP (finding)**

## TL;DR

The split is **mathematically exact** — a 3-line numpy replication of the
graph's iSTFT reproduces the fused render to float32 noise (max-abs-diff
<= 1.2e-5 on audio with absmax ~0.44, correlation = 1.000000000, identical
length, whisper transcripts identical). But it is **not faster**: direct ORT
kernel profiling shows the in-graph iSTFT tail costs only **8.1 ms/call**,
not the ~170 ms/call from the earlier operator-profile claim. Native numpy
iSTFT costs 10.3 ms and librosa 21 ms; interleaved ABAB wall-clock timing
puts split+native at **1.02x of full-graph time (no win)**. The real
STFT-family cost is the **forward STFT node: 305 ms/call (~10% of kernel
time)**, which feeds the source-envelope branch and cannot be removed without
changing model semantics.

Bonus finding: pruning the iSTFT tail **removes exactly the subgraph that
crashed OpenVINO in E0025** — the split model converts and executes on
OpenVINO CPU where the fused model dies. Numerics diverge under OV
(corr 0.957 vs ORT), so quality there remains unproven.

## Graph anatomy (what the "iSTFT" actually is)

- No `ISTFT` op exists in the graph. The vocoder tail is a primitive
  subgraph `/decoder/decoder/generator/istft/stft/*`: magnitude (`Exp`) x
  phase (`Sin` -> Cos/Sin) -> Concat real|imag `[1, 22, L]` ->
  **ConvTranspose** (inverse_basis `22x1x20`, stride=5, kernel=20 = inverse
  DFT + window overlap-add, n_fft=20, hop=5 @24 kHz) -> edge-normalization
  chain (Slice/Greater/**NonZero**/Gather/Div/**ScatterND**) -> Mul(4.0) ->
  Slice[10:-10] -> audio.
- The normalization chain only touches samples `[0:10]` and `[-10:]` of the
  reconstruction — precisely what the center-trim then discards. It is
  **provably dead code for the emitted audio** (verified numerically:
  `audio == 4.0 * overlap_add(...)[10:-10]`, maxdiff 4.7e-6).
- Window constants (`window_sum`, forward-STFT Hann, `inverse_basis`) are
  already constant-folded initializers — nothing is regenerated per call.
  The only per-call recomputable work (NonZero indices etc.) sits inside the
  dead chain (0.35 ms/call).

## Exact native implementation (KEEP as reference)

```python
spec = split_model_output[0]              # [1, 22, L]; ch 0..10 real, 11..21 imag
contrib = np.einsum("ck,cl->lk", basis[:, 0, :], spec)   # basis: ConvTranspose weight
y = np.zeros((L - 1) * 5 + 20, dtype=np.float32)
for k in range(20):
    y[k::5][:L] += contrib[:, k]
audio = 4.0 * y[10:-10]
```

librosa equivalent: `librosa.istft(spec_real+j*spec_imag, hop_length=5,
win_length=20, n_fft=20, window="hann", center=True) * 1.5` (the 1.5 is the
hann^2 COLA sum that librosa divides out but the graph's basis folds in).

Demonstrated end-to-end in `scripts/native_istft_render.py`
(`SplitRenderer.synth_tokens`; split graph built by `build_split_model()`,
never touching the original model file).

## Equivalence results (full graph vs split+native)

| Sentence | maxabsdiff | rel | corr | whisper |
|---|---|---|---|---|
| evalA | 4.68e-06 | 1.06e-05 | 1.0 | transcripts identical |
| avx | 9.21e-06 | ~2e-05 | 1.0 | transcripts identical |
| numbers | 1.21e-05 | ~3e-05 | 1.0 | punctuation-only ASR jitter (+0.125 WER) |
| natural | 7.66e-06 | ~2e-05 | 1.0 | transcripts identical |

librosa*1.5 path gives byte-similar diffs to the exact replication. The one
WER delta is whisper punctuation jitter on a >90 dB-SNR signal — ASR noise,
not audible difference.

## Timing (@4 threads, evalA, 3.95 s audio)

| Measurement | Time |
|---|---|
| Full fused graph sess.run (median, 8 reps) | 3.80 s |
| Split graph sess.run (median) | 3.94 s |
| Native numpy iSTFT | 10.3 ms |
| librosa iSTFT | 21.0 ms |
| In-graph iSTFT tail kernels (ORT profile) | **8.1 ms/call** |
| Dead normalization chain kernels | 0.35 ms/call |
| **Forward STFT node kernels** | **305 ms/call (~10%)** |
| ABAB interleaved grand ratio (split+native / full) | **1.02x (3 rounds: 1.02 / 0.89 / 1.04 — thermal noise, no win)** |

Wall-clock variance on this machine (2.2–3.1 s for identical work) dwarfs any
iSTFT effect; nothing here approaches the operator-profile's claimed 11.7% /
2547 ms — that figure aggregates the *forward* STFT plus multi-run summation.

## OpenVINO probe (bonus)

E0025 crashed inside `istft/stft/Expand`. The split model deletes that whole
subgraph: `ov.convert_model` 3.5 s, CPU compile 2.4 s, inference runs.
However OV's execution of the remaining graph diverges from ORT upstream of
the cut (spectrum maxdiff up to 47% of absmax on scattered frames); rendered
audio corr = 0.957, SNR 10 dB, transcript still correct on the test clip.
Execution blocker gone; quality gate not passed. Revisit only if OpenVINO is
pursued again.

## Decision

- **KILL** the split as a speed optimization: the in-graph iSTFT is already
  cheap (8 ms), native replication is not faster, wall-clock parity within
  noise.
- **KEEP** as documented findings: (1) exact native iSTFT formula +
  `scripts/native_istft_render.py` reference implementation (useful for any
  future backend that lacks these ops, e.g. OpenVINO/CoreML/WebView runtimes);
  (2) the dead normalization chain (0.35 ms) could be pruned in a future
  re-export; (3) if anyone attacks "STFT" cost next, target the **forward**
  STFT node (305 ms/call), not the iSTFT tail.

Artifacts: `results.json` (all numbers), `kokoro-v1.0-noistft.onnx` (split
graph copy), `wavs/` (A/B renders), `probe_shapes.py`, `verify_math.py`,
`run_experiment.py`, `rigor_and_openvino.py`. Original model untouched;
no tests affected (no existing test imports these paths).
