# CV-G1-E0002 — CAEDRAL-NAR prototype: parameters, CPU latency, RSS (Phase F)

**Date:** 2026-08-22 · **Machine:** i7-13620H (6P+4E, 16 logical), Windows 11, py3.13.15, torch 2.6.0+cpu
**Implementation:** `caedral_voice/model/` (`backbone.py` pre-LN transformer w/ RoPE+GQA · `variance.py` FastSpeech2-class predictors + speaker d=128 / style(10+neutral) tables + FiLM per MULTI-SPEAKER-PROSODY-STYLE §8 · `istft_head.py` Vocos-class torch.istft head · `caedral_nar.py` wiring)
**Configs:** `configs/{tiny,lite,quality}.json` · **Raw results:** `results.jsonl` · **Runner:** `run_benchmark.py` (fresh process per config so Win32 PeakWorkingSetSize is attributable)
Untrained stubs — contracts measured: param counts, shapes/determinism/conditioning sensitivity (tests/test_model.py, 12 passed), batch-1 latency/RSS. All numbers below [MEASURED] unless tagged.

---

## HYPOTHESIS

**(H1, from ARCH-NOTES-G1 budget arithmetic + ACOUSTIC-ARCHITECTURES decision matrix row A):** a deterministic NAR mel model of the Kokoro/StyleTTS2 lineage suits this CPU — i.e., all three size points {8M, 30M, 80M} can be instantiated within ±15% parameter targets, run faster than real time at batch-1 for utterance-scale inputs in *unoptimized eager* PyTorch, and stay inside the Phase-M RAM envelopes (Tiny ≤ 200 MB, Lite ≤ 350 MB, Quality ≤ 700 MB peak). Secondary expectations from ARCH-NOTES: (a) weight-stream floor implies ~8 ms/pass for 82M fp32 — real passes should sit well above it (implementation-bound, not bandwidth-bound); (b) hybrid-core bimodality should show up in the {1,4}-thread sweep.

## METHOD

- **Params:** `count_params()` + top-level component breakdown vs targets {8M, 30M, 80M}, ±15% gate.
- **Latency:** batch-1 forward, median-of-20 after 5 warmups via `benchmarks.harness.measure`, phoneme lengths {50, 200, 800} × intraop threads {1, 4}. Durations come from the model's own (untrained) predictor — exp(pred)≈1 frame/phoneme at init, so decoder length ≈ phoneme length. To anchor production scaling (real phonemes ≈ 4–6 mel frames @ hop 256 / 24 kHz ≈ 93.75 Hz), an extra cell runs T=200 with uniform 4-frame durations → 800 mel frames (~8.5 s audio). Anchor cells adapt reps down if a single forward exceeds ~2 s (reps recorded per row; all spec cells ran the full 20).
- **RSS:** sampler-thread peak over pre-forward baseline + exact Win32 PeakWorkingSetSize, one fresh python process per config (baseline therefore includes import-torch + built weights; activations show up as peak-over-baseline).
- **Determinism/sensitivity gates (tests/test_model.py):** same-seed bitwise reproduction; speaker switch (0→7) and style switches (neutral→3→8) measurably change mel/pitch/energy at frozen durations; knobs transform tracks exactly (speed divides clamped durations; pitch_shift multiplies F0 by 2^(st/12); energy_gain adds dB).

## RESULT

**Parameter counts (±15% gate: all PASS)**

| Config | Total | Target | Δ | Decoder | Encoder | iSTFT head | 3×variance preds | Rest |
|---|---|---|---|---|---|---|---|---|
| tiny | 8.349 M | 8 M | +4.4% | 4.609 M | 1.448 M | 1.473 M | 0.742 M | 0.078 M |
| lite | 31.605 M | 30 M | +5.4% | 17.739 M | 6.306 M | 5.042 M | 2.368 M | 0.150 M |
| quality | 80.654 M | 80 M | +0.8% | 49.633 M | 12.996 M | 10.707 M | 7.095 M | 0.220 M |

Decoder = 55–62% of every config; conditioning tables are negligible (4.8 K); FiLM generators add ~0.4–2.4 M inside decoder/encoder buckets.

**Batch-1 forward latency (median ms; acoustic-only RTF in parens)**

| Config | thr | T=50 ph | T=200 ph | T=800 ph | 800-frame anchor (×4) |
|---|---|---|---|---|---|
| tiny | 1 | 19.4 (.036) | 56.0 (.026) | 228.6 (.026) | 185.0 (.022) |
| tiny | 4 | 29.4 (.055) | 59.9 (.028) | 133.8 (.015) | 122.7 (.014) |
| lite | 1 | 59.6 (.082) | 194.2 (.070) | 891.2 (.080) | 616.8 (.072) |
| lite | 4 | 48.2 (.067) | 123.8 (.044) | 592.4 (.053) | 285.3 (.033) |
| quality | 1 | 124.1 (.228) | 429.3 (.174) | 2181.4 (.226) | 1342.8 (.157) |
| quality | 4 | 78.8 (.145) | 225.5 (.092) | 1143.0 (.118) | 730.7 (.085) |

Worst case (quality, 800 phonemes ≈ 9.7 s audio): RTF 0.23 @1 thread, 0.12 @4 threads. Reference point on THIS machine: Kokoro-82M ONNX MEASURED RTF ≈ 0.33 (Phase C). Thread speedup at T=800: tiny 1.71×, lite 1.51×, quality 1.91×. At T=50, 4 threads are SLOWER than 1 (tiny 29.4 vs 19.4 ms; quality still wins: 78.8 vs 124.1 ms). Marginal cost ≈ 0.28 ms/phoneme (tiny, 1thr) → 2.7 ms/phoneme (quality, 1thr).

**Peak RSS (fresh process, fp32)**

| Config | Import-torch+weights baseline | Forward peak-over-baseline | Process native peak | Phase-M envelope |
|---|---|---|---|---|
| tiny | 218 MB | +16.9 MB | **235 MB** | ≤200 MB [HYPOTHESIS] — MISS (+18%) |
| lite | 306 MB | +25.9 MB | **334 MB** | ≤350 MB — pass |
| quality | 496 MB | +28.8 MB | **528 MB** | ≤700 MB — pass |

## INTERPRETATION

1. **H1 SUPPORTED on latency [MEASURED].** Every config clears real time at batch-1 in plain eager PyTorch, worst cell RTF 0.23 — under the trained-Kokoro reference RTF 0.33 measured on this same box, before any optimization (no graph compile, no fusion, no quantization). The decision-matrix row-A premise ("fastest CPU wall-clock of any credible quality point") survives first contact with our own prototype.
2. **Compute-bound, not bandwidth-bound [MEASURED].** The 82M weight-stream floor is ~8 ms/pass; the actual quality pass at 800 frames costs 730–2181 ms — 90–270× above it. Consistent with ARCH-NOTES-G1: headroom is implementation-side (fusion, compiled graphs, attention over long frame sequences), not parameter-side. Parameter cuts below 80M buy wall-clock almost linearly (quality→lite ≈ 2.4–3.7× at equal frame counts), so the size ladder is real.
3. **Hybrid-core bimodality confirmed [MEASURED].** 4 threads help large batches (≤1.9×) and hurt small ones (T=50 regressions up to +52% on tiny) — dispatch/thread-sync overhead plus P/E-core asymmetry. Production policy should pin threads per workload class (short utterances → 1–2 threads), as the calibration notes anticipated.
4. **RAM verdicts split [MEASURED].** Lite and Quality fit their Phase-M envelopes today. Tiny misses its 200 MB hypothesis — not because of weights (+32 MB) but the CPython+torch runtime floor (~215 MB before any model). Confirms the Phase-C pattern (Kokoro 1.25 GB working set): profile-C/D targets need a small-runtime deployment path (ONNX / free-threaded slim runner), not a smaller acoustic model.
5. **Expansion-factor caveat, quantified [MEASURED].** Untrained duration predictors sit at the 1-frame floor; the ×4 anchor shows true production-length cost: quality @800 mel frames = 1343 ms (RTF .157) @1thr / 731 ms (.085) @4thr. Scaling stays gentle (attention quadratic term still minor at ≤1k frames); at realistic 5–6× expansion for 800 phonemes (~4.8k frames ≈ 51 s audio) linear extrapolation gives RTF ≈ 0.9–1.1 @1thr — still near-real-time, and the first target for Phase-L optimization.
6. Stub-quality caveats: no perceptual claims whatsoever; ISTFT head verified for shapes+determinism only; conditioning sensitivity proven structurally (FiLM + predictor paths), not perceptually.

## KEEP / REJECT / INVESTIGATE

**KEEP**
- CAEDRAL-NAR skeleton as implemented (`caedral_voice/model/`): pre-LN + RoPE + GQA backbone, FiLM-conditioned variance adapter (speaker d=128 ⊕ style 10+neutral), Vocos-class iSTFT head; all three shipped configs within ±15% targets — freeze as the Phase-F reference wiring.
- Batch-1 measurement protocol (median-of-20/5-warmup via shared harness, fresh-process RSS) as the standard for all future model experiments.
- Knob semantics on tracks (speed/pitch_shift/energy_gain) with exact-transform tests.

**REJECT**
- 4-thread defaults for interactive short-utterance synthesis (measured slowdowns at T=50).
- Any plan to hit the Tiny RAM envelope with the CPython+torch runtime — rejected by measurement; needs small-runtime path instead.

**INVESTIGATE (next)**
- Phase G: train the iSTFT head + measure Vocos-class RTF on this box (ARCH-NOTES directive) before architecture freeze.
- torch.compile / ONNX-export pass over this exact wiring to quantify the 90–270× implementation gap vs the weight-stream floor; re-run this sweep post-compile.
- Re-measure at trained duration predictors (realistic 4–6× expansion, padded batching) and with pinned-affinity thread policy; add T=1600 cell for long-form.
- GQA ablation at quality scale (kv 8/16 here): latency delta vs quality risk before Phase G freeze.

*Reproduce:* `.venv/Scripts/python.exe -X utf8 experiments/CV-G1-E0002-nar-prototype/run_benchmark.py --parent` (writes results.jsonl fresh; ~6 min). Tests: `.venv/Scripts/python.exe -m pytest tests/test_model.py -q`.
