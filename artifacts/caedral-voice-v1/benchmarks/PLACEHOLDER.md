# PLACEHOLDER — benchmarks/

## What lands here

All measured V1 performance/quality tables (final report sections source from
here; every number MEASURED on the i7-13620H, CPU-only):

1. `rtf-ram-ttfa@threads.json|md` — RTF, peak RSS, TTFA for the V1 model at
   OMP threads {1, 2, 4, default} x utterances {short conv_001, para_001,
   long_001}; fresh-process runs. Pattern + known bimodality caveat:
   `docs/benchmarks/BASELINE-MEASURED.md` ORT thread sweep.
2. `natgate-v1/` — GATE-NAT-01 run on V1: clips, `scores-caedral-v1.json`,
   filled row in `benchmarks/results/natgate/comparison-template.json`
   (DNSMOS median/min, WER med, RTF, RAM, size vs Kokoro/Piper/KittenTTS).
3. `longform-v1.json` — E0014 protocol re-run on V1: per-chunk dual-ASR
   minWER distribution, population variance (gate <= 0.0025), worst-chunk,
   eff RTF, wall-clock.
4. `speaker-similarity.json` — ECAPA-TDNN cosine gen-vs-reference per voice
   (+ anchor delta) per EVALUATION-BENCHMARKS G4.
5. `machine-baseline` pointer — hardware snapshot reference
   (`benchmarks/machine-baseline.json`, `benchmarks/results/cpu-calibration-*.json`).

## Source paths in repo

- Harnesses: `benchmarks/run_natgate.py`, `scripts/classify_preview.py`
  (DNSMOS ovrl primary + dual-ASR minWER informational),
  `benchmarks/run_baseline_kokoro.py` (thread-sweep pattern)
- Long-form probe: `experiments/CV-G1-E0014-longform-baseline/probe.py`
  (baseline numbers: variance 0.0173 FAIL anchor, eff RTF 0.220 @4 thr)
- Gate definitions: `docs/benchmarks/NATURALNESS-GATE.md`,
  `provenance/release-gates.json`

## Generating commands

```bash
# thread sweep on V1 (fresh process per thread count)
OMP_NUM_THREADS=1 .venv/Scripts/python.exe <runtime bench entry>   # repeat {2,4,default}
# naturalness gate
.venv/Scripts/python.exe benchmarks/run_natgate.py generate --engine caedral-v1
.venv/Scripts/python.exe scripts/classify_preview.py --audio <clips> --reference-texts ... --json-out ...
.venv/Scripts/python.exe benchmarks/run_natgate.py compare
# long-form
.venv/Scripts/python.exe experiments/CV-G1-E0014-longform-baseline/probe.py --model <V1 ckpt>
```

## Status

PENDING — produced by the **threads** sweep, **natgate**, and
**CV-G1-E0014** re-run on the final V1 model. Delete this file when the real
artifacts land.
