# Memory Breakdown — Kokoro inference engine (CV-G1-E0027)

MEASURED 2026-08-23 · Windows 11, Intel Model 186 (Meteor Lake), 16 logical CPUs ·
Python 3.13.15 · onnxruntime 1.29.0 · kokoro-onnx 0.6.1 ·
model `models/kokoro/kokoro-v1.0.onnx` (310.5 MB disk) · threads=4 · voice `af_heart`.
Scripts/data: `experiments/CV-G1-E0027-memory/{memory_probe.py, config_matrix.py,
memory-breakdown.json, results.json}`.

## Question

Warm RSS of the kokoro_onnx engine is reported around ~796 MB while the model
file is 310 MB on disk. Where does the other ~486 MB live?

## Component decomposition [MEASURED]

Ladder method: each stage N loads stages 0..N in a FRESH subprocess; marginal
cost = RSS(N) − RSS(N−1) across clean heaps. A single-process cumulative walk
reproduces the same boundaries within ±0.5 MB (`memory-breakdown.json`).

| Stage | Marginal RSS | Cumulative RSS | Commit | Notes |
|---|---|---|---|---|
| Python + numpy + onnxruntime + psutil imports | 51.2 | 51.2 | 515 | interpreter + native libs |
| ONNX session create (fused graph) | **+335.2** | 386.4 | 868 | 310.5 MB weights resident + ~25 MB ORT overhead (~8%) |
| kokoro_onnx wrapper + voices + tokenizer | +11.2 | 397.6 | 877 | voices-v1.0.bin (26.9 MB) is a lazy NpzFile: only the requested voice's 0.5 MB slice is materialized per call |
| TextFrontend ctor (cmudict lexicon + chain) | +78.8 | 476.4 | 955 | lexicon JSON + pipeline objects |
| g2p_en NLTK LSTM fallback | +0 (conditional) | — | — | NOT loaded in the default path since E0020: "caedral" has a pronunciation override that bypasses G2P. Standalone construction costs **+205 MB / 3.35 s** when a true OOV word falls through to neural fallback |
| One short synthesis (59 phonemes → 3.18 s audio) | +71.9 | 548.4 | 1054 | transient activation working set; peak ≈ steady for short sentences |

**Fixed floor after load + frontend warmup ≈ 476 MB.** Synthesis adds a
usage-dependent working set: +72 MB (short sentence, first pass) growing to
+145 MB after repeated runs (ORT memory-pattern buffers + BFC arena growth),
and further for longer utterances — E0021 sampled a 1044 MB peak across a
mixed-length workload. The ~796 MB operating point sits inside this band: it
is accounted for by the components above (imports + session + wrapper +
frontend + synthesis working set), not a leak. Note Windows commit charge is
~1.05–1.19 GB throughout — RSS understates reserved backing by ~500 MB.

## Session memory configs [MEASURED]

Fresh subprocess per config; warm RSS after warmup + 5 reps of a frozen
sentence ("Hello. This is first generation of Caedral Voice.", 3.03 s audio);
RTF = median wall / audio. Ambient CPU 10–38% (r1 training worker active), so
absolute RTFs are inflated vs the historical idle band (0.32–0.49 @t4);
contention affects all configs equally.

| Config | Warm RSS | Δ vs base | RTF median | Verdict |
|---|---|---|---|---|
| base (all defaults) | 620.8 | — | 0.676 | reference |
| `enable_mem_pattern=False` | 547.5 | −73.3 | 0.648 | **KEEP (secondary)** — free −73 MB if the arena must stay on |
| `enable_cpu_mem_arena=False` | **480.4** | **−140.4** | 0.658 | **KEEP (primary)** — lowest RSS, RTF within noise of base; subsumes the mem_pattern saving |
| arena extend `kSameAsRequested` (config entry `arena.extend_strategy`) | 620.7 | −0.1 | 0.642 | **KILL** — no measurable effect; the knob is not reachable through the CPU EP via the Python API in ORT 1.29 |
| graph opt `ORT_DISABLE_ALL` | 620.5 | −0.3 | 0.667 | **KILL** — zero RSS benefit; forfeits runtime fusion headroom |
| mem_pattern off + kSameAsRequested | 548.8 | −72.0 | 0.677 | **KILL** — no gain over mem_pattern_off alone |
| mem_pattern off + arena off + kSameAsRequested | 479.3 | −141.5 | **1.383** | **KILL** — RTF doubles: no pattern reuse AND no arena caching forces full allocation churn every run |

Savings do not stack: base's post-load growth (+73→+145 MB) is one shared
pool (pattern buffers + arena retention) that either big toggle collapses;
combining them only removes allocator caching and costs speed (combo_b).

Appendix at intra_op=8, ambient 32–39% CPU: base 621.3 MB / RTF 0.649;
cpu_arena_off 481.7 MB / RTF 0.559.

## Recommendation

Adopt `enable_cpu_mem_arena=False` on the production session options:
**−140 MB warm RSS (621→480 MB, −23%) at no measurable RTF cost**, and the
lowest-RSS configuration tested that preserves real-time-class throughput.
Strictly, no config met RTF ≤ 0.55 during this run because of training
contention; `cpu_arena_off` reached 0.559 @t8 under load and the idle band
implies comfortable margin. Re-benchmark the gate on an idle box before
locking the claim. Keep `enable_mem_pattern=True` alongside it (its saving is
already included in the arena-off number).
