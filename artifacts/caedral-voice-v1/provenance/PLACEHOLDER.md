# PLACEHOLDER — provenance/

## What lands here

Everything needed to reproduce and audit the release:

1. `release-gates.json` — PRESENT (template). Founder gates with TODO markers;
   measured values written in-place as experiments land; every gate flips
   PENDING -> PASS/FAIL before ship.
2. `machine.json` — hardware/software snapshot (i7-13620H 10C/16T, Windows 11,
   CPU-only; torch/onnxruntime versions) — extend `benchmarks/machine-baseline.json`.
3. `git-provenance.json` — repo commit(s), dirty-file list at packaging,
   checkpoint sha256s.
4. `data-hashes.json` — dataset manifest hashes: `data/manifests/
   librittsr-clean100*` (44.11 h / 30,917 utts / 247 spk) and
   `data/manifests/teacher-kokoro.*` (+ per-row sha256 fields already carried).
5. `teacher-block.json` — Kokoro teacher meta per TEACHER-DATA-ENGINE §5:
   kokoro-onnx version, model + voices-bin sha256, per-voice counts, QA
   thresholds, eval-corpus firewall statement.
6. `eval-corpus-freeze.json` — pointer + hash of
   `benchmarks/corpus/eval-corpus-v1.0.json` (frozen; never edited to improve scores).

## Source paths in repo

- `benchmarks/machine-baseline.json`, `benchmarks/results/cpu-calibration-*.json`
- `data/manifests/*.meta.json`
- `experiments/CV-G1-E0013-regressor/run_manifest.json`

## Generating commands

```bash
python - <<'EOF'
import hashlib, json
for p in ["data/manifests/librittsr-clean100.meta.json", "data/manifests/teacher-kokoro.meta.json"]:
    print(p, hashlib.sha256(open(p,'rb').read()).hexdigest())
EOF
```

(plus git rev-parse HEAD and checkpoint hashing at packaging time)

## Status

PARTIAL — `release-gates.json` template present now; remaining files assembled
at packaging time. Replace this placeholder when provenance is complete.
