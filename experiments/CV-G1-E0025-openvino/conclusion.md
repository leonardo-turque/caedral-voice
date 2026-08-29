# CV-G1-E0025-openvino — OpenVINO as CPU backend for Kokoro ONNX

**Task:** OPENVINO-01 · **Date:** 2026-08-23 · **Verdict: SKIP** (blocker, not a tuning loss)

## TL;DR

OpenVINO installs cleanly on py3.13 and converts `models/kokoro/kokoro-v1.0.onnx`
to IR in ~4.4 s, but the IR **crashes at first inference** on the CPU plugin.
The failure is in the fused iSTFT vocoder (`/decoder/decoder/generator/istft/stft`),
where OpenVINO's shape inference resolves a broadcast inconsistently (dim 61 vs 19).
ONNX Runtime executes the identical graph fine. No RTF or quality numbers can be
honestly reported because OpenVINO never produced audio.

## What was tested (all probes)

| Probe | OpenVINO | Result |
|---|---|---|
| pip install (py3.13) | 2026.3.0 / 2025.4.0 | OK, ~16 s both |
| Convert dynamic → IR | 2026.3.0 | OK (4.4 s; xml 1.65 MB + bin 309.5 MB) |
| Compile IR on CPU @4 threads | 2026.3.0 | OK (2.39 s) |
| First inference (CPU) | 2026.3.0 | **CRASH** — `istft/stft/Expand`: dim 61 cannot numpy-broadcast to 19 |
| Static-shape conversion (tokens[1,61]) | 2026.3.0 | **CRASH inside ovc** — inconsistent Add shapes `[61,1]` vs `[19,1]` in same istft subgraph |
| Dynamic IR on GPU plugin (diagnostic) | 2026.3.0 | **CRASH** — Interpolate `linear_onnx` rejects rank-3 input |
| Downgrade probe, dynamic convert | 2025.4.0 | **CRASH at convert** — ONNX frontend rejects STFT-20 with dynamic signal shape |

## Why this is SKIP, not KILL

- Task rule: "If conversion fails: SKIP and report." Conversion here produces an
  IR that cannot execute on any local plugin — substantively a failed conversion.
- The RTF-worse-than-ORT KILL rule never came into play: there is no measurable
  RTF. Baseline ORT band for reference: RTF 0.62–0.82, DNSMOS 3.35–3.55
  (E0021/E0022). WER ±0.10 gate unreachable without an OpenVINO render.

## Root cause

Kokoro v1.0's fused iSTFT head derives frame geometry from data-dependent shapes
(duration-predictor output). ORT evaluates those shapes correctly at runtime;
OpenVINO's symbolic propagation disagrees across two branches of the same subgraph,
both during runtime shape validation (CPU) and during constant folding with static
inputs (ovc). Upstream OpenVINO limitation — not a Kokoro export bug, not a
Caedral frontend issue.

## Method note (reusable)

`bench_openvino.py` mirrors E0024's design: isolated subprocesses, the SAME
`kokoro_onnx.Kokoro` wrapper driving both backends through an OpenVINO session
shim (only the graph executor differs), production Caedral frontend +
DEFAULT_VOCAB, 10 warm reps after a discarded cold call @ 4 threads, two
opposite-order rounds, repo-standard dual-ASR minWER + DNSMOS scoring. It is
ready to produce the full A/B unchanged once an OpenVINO release executes this
graph (conversion takes ~5 s to redo).

## Decision

**Keep ONNX Runtime CPU EP as the production backend.** Re-check OpenVINO only
after a major version bump whose changelog mentions ONNX STFT/iSTFT support;
then re-run `bench_openvino.py`.
