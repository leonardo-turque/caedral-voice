# CAEDRAL VOICE — MASTER PLAN

**Program:** Caedral Voice Generation 1 (English-only)
**Document status:** LIVING — updated at every phase checkpoint
**Last updated:** 2026-08-22 (Phase A)

---

## 1. Mission (condensed)

Create a proprietary Caedar-owned TTS stack: multi-voice, expressive, natural English
speech with **CPU-first production inference**, full training pipeline, reproducible
benchmarks, documented data provenance, and honest measurement throughout.

Success is **Pareto-multi-dimensional**: quality × RTF × RAM × model size.
A variant that matches medium-class quality at dramatically lower RAM/latency beats a
marginal-quality champion that needs a GPU.

## 2. Generation-1 scope

| Dimension | G1 decision |
|---|---|
| Language | English only (architecture stays language-extensible) |
| Voices | ≥ masc + fem identities; multiple timbres/personalities; catalog-ready config format |
| Styles | neutral, conversational, professional, energetic, calm, serious, warm, excited, soft, dramatic |
| Controls | rate, pitch tendency, energy, pauses, emphasis, expressiveness — where evidence justifies |
| Cloning | NOT a G1 goal (zero-shot cloning creates provenance/legal risk for commercial voices) |
| Deployment | CPU-only serving profiles A–E (below); GPU permitted for future large training |

## 3. Hard constraints

1. **CPU-first architecture.** No "GPU model with a CPU fallback" designs.
2. **Ownership.** Final artifact: Caedral configs, pipeline, runtime, weights.
   Public code/papers/datasets usable per license; no proprietary copying;
   no mislabeled third-party weights; no unlicensed or celebrity voice material.
3. **Scientific honesty.** Labels on every claim: `MEASURED` / `REPRODUCED FROM SOURCE` /
   `REPORTED BY THIRD PARTY` / `ESTIMATED` / `HYPOTHESIS`. No invented benchmarks,
   no hidden failures, no corpus edits to inflate scores.

## 4. Hardware baseline — MEASURED (Phase A, 2026-08-22)

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 26200, en-US, tz E. South America |
| CPU | Intel Core i7-13620H (Raptor Lake-H), 10 physical cores (6P+4E) / 16 threads, base 2.4 GHz |
| ISA notes | **MEASURED via CPUID (py-cpuinfo):** AVX ✓, AVX2 ✓, FMA3 ✓, BMI2 ✓, AVX-512 ✗ (absent), AVX-VNNI unverified by tooling (Raptor Lake-H supports it per Intel ARK [REPORTED]) — see `benchmarks/machine-baseline.json` |
| RAM | 23.67 GB installed; ~6 GB free at inspection time (other apps running) |
| GPU | Intel UHD Graphics (integrated) — **no CUDA / no discrete GPU available** |
| Disk | C: 475.9 GB total, **122.3 GB free** |
| Python | 3.13.15 (+ pip 26.2.1) — native Windows |
| Toolchain | git 2.45.1, Node present, Docker Desktop present, WSL present |
| Compilers | cl/gcc/clang/cmake/make all ABSENT — native C++/Rust builds require toolchain install first |
| Implication | Training = CPU PyTorch only (smoke + small pilot scale). Inference optimization path = ONNX Runtime / OpenVINO wheels (no local compilation needed). Full-scale training requires external GPU budget later. |

Consequences recorded now:
- All local training this generation = smoke/pilot scale unless external compute appears (§15 of charter applies: never fake a full run).
- Docker Desktop enables the §18 torture-lab via WSL2 cgroups if stable; fall back to Job Objects + affinity masks if not.
- Free RAM (~6 GB) and disk (122 GB) bound what we may download — SMOKE/PILOT discipline mandatory (§7).

## 5. Deployment targets (from program charter)

| Profile | vCPU | RAM cap | Role |
|---|---|---|---|
| A — Max CPU quality | 4–8 | ≤4 GB | flagship quality |
| B — Economical prod | 2 | ≤2 GB | default production |
| C — Ultra-low-cost | 1 | ≤1 GB | cheap VPS tier |
| D — Torture | 1 | ≤500 MB | research stretch goal |
| E — Extreme research | 1 | 256–500 MB | honesty experiment |

Rule: targets are *measured against*, never assumed achieved. Failures are documented
with the actual minimum found.

## 6. Roadmap & status

| Phase | Scope | Status |
|---|---|---|
| A | Environment & inventory; MASTER-PLAN | ✅ DONE 2026-08-22 |
| B | State-of-the-art research (8 parallel tracks) | ✅ DONE 2026-08-22 (10 docs, ~500 citations, Checkpoint B) |
| C | Benchmark harness + frozen eval corpus + baseline benchmarking | 🔄 IN PROGRESS (corpus v1.0 frozen; harness core done) |
| D | Data & licensing survey → manifests, preprocessing design | ⬜ |
| E | Text frontend / tokenizer experiments | ⬜ |
| F | Architecture search (candidates, estimates, prototypes) | ⬜ |
| G | Decoder/vocoder selection | ⬜ |
| H | Speaker/style/prosody system | ⬜ |
| I | Training system (reproducible, resumable, smoke-proven) | ⬜ |
| J | First pilot Caedral model + fixed samples | ⬜ |
| K | Quality iteration (targeted, ablation-driven) | ⬜ |
| L | CPU optimization (profile-first) | ⬜ |
| M | Quantization/compression variants | ⬜ |
| N | Voice catalog build-out | ⬜ |
| O | Torture/production benchmark matrix | ⬜ |
| P | Independent scientific audit | ⬜ |

## 7. Data strategy

Modes: **SMOKE** (code validation, minutes) → **PILOT** (training dynamics, hours) →
**FULL** (only after architecture earns it). Before any download >1 GB: report sizes,
free disk, necessity, and a pilot subset plan in the phase checkpoint. Raw data immutable;
manifests hashed and versioned in git. Dataset licensing gate: nothing enters training
without a row in `docs/data/LICENSE-MATRIX.md` clearing commercial use.

## 8. Experiment protocol

IDs `CV-G1-E####`, folder `experiments/CV-G1-E####/` containing config, dataset-manifest
ref, environment.txt, metrics, logs, checkpoints, samples, benchmark results, and a
`conclusion.md` stating HYPOTHESIS / METHOD / RESULT / INTERPRETATION /
KEEP-REJECT-INVESTIGATE. Index in `experiments/EXPERIMENT-INDEX.md`.
Checkpoints are resumable; the best-known checkpoint is never overwritten in place.

## 9. Honesty charter (operational rules)

1. Every number in docs carries an evidence label (§3).
2. Benchmark hardware always disclosed; same-machine CPU comparisons preferred.
3. The frozen corpus (`benchmarks/corpus/`) changes only via a logged, justified commit — never to improve scores.
4. Failed/negative experiments are recorded in EXPERIMENT-INDEX.md like any other.
5. If a target is unreachable, we publish the measured minimum instead.
6. Third-party claims are never reprinted as our measurements.

## 10. Architecture candidate hypotheses (to be FALSIFIED, not assumed — Phase F)

These are starting hypotheses only, each requiring Phase B/C/F evidence:

- **H1 (NAR mel + FastSpeech-style predictors):** duration/F0/energy-predicting
  transformer/conformer → mel → neural vocoder. Parallel decoding suits CPU BLAS well;
  explicit prosody control falls out naturally. Risk: expressivity ceiling vs modern AR systems.
- **H2 (flow-matching few-step decoder):** F5/Matcha-class latent flow matching,
  distilled to few NFEs. Strong quality trend 2024–2026. Risk: per-step cost × steps on CPU;
  AR-free but compute-dense matmuls.
- **H3 (AR codec-token LM):** low-frame-rate neural-codec LM (e.g., 12.5–25 Hz tokens),
  KV-cached AR decoding. Small models punch above weight; streaming trivial. Risk:
  CPU AR token throughput; hallucination/repetition failure modes; codec decoder cost.
- **H4 (hybrid):** H1 prosody front + H2/H3 decoder, distillation across variants for
  Quality/Balanced/Lite/Tiny family sharing one architecture.

Selection criteria (Phase F): CPU RTF @ batch-1, peak/steady RAM, artifact size,
trainability on pilot data, controllability surface, failure-mode profile.

## 11. Decision log (append-only)

| Date | Decision | Evidence basis |
|---|---|---|
| 2026-08-22 | Repo skeleton adopted per charter §21 (adapted) | charter |
| 2026-08-22 | ONNX Runtime chosen as primary optimization substrate candidate (no local compilers; wheel-based deployment) | MEASURED env (§4); revisit after Phase F/L |
| 2026-08-22 | Zero-shot cloning excluded from G1 scope; catalog voices from licensed data | charter §3 ownership rules |

## 12. Risk register (top items)

| Risk | Mitigation |
|---|---|
| No discrete GPU → full-scale training impossible locally | Honest smoke/pilot science + exact recipes for future runs (charter §15) |
| 122 GB free disk limits FULL datasets | SMOKE/PILOT gating, size reports before downloads |
| espeak-ng-class GPL frontend deps infect distribution licensing | Phase B track B evaluates Apache/MIT alternatives before commitment |
| Laptop free RAM (~6 GB) constrains parallel workloads | Serialize heavy jobs; measure, don't assume |
| Research agents return stale/wrong claims | Coordinator reconciliation pass + independent audit (Phase P) |

## 13. Status snapshot

- **Current best Caedral model metrics:** NONE EXISTS YET (honest null).
- **Baselines MEASURED locally (2026-08-22):** Kokoro-82M RTF 0.32–0.49 /
  1.25 GB peak; Piper-medium ×2 RTF 0.06–0.24 (isolated); KittenTTS-nano
  RTF 0.37–0.42, crashes on long-form. Ledger: docs/benchmarks/BASELINE-MEASURED.md.
- **Deferred baselines:** Chatterbox-Nano (v0.1.7 lacks class), F5-TTS (NC weights).
- **Pending in Phase C:** quality scoring of 30 persisted samples (EVAL SPEC v0);
  ORT thread sweep for profiles.
- **Preliminary architecture direction [HYPOTHESIS]:** CAEDRAL-NAR >
  CAEDRAL-CFM >> AR-codec-LM — falsify in Phases C/F before freeze.
- **Verification backlog:** 49 UNVERIFIED flags (docs/research/SOURCES.md).
- **Human/legal queue:** GigaSpeech terms, VCTK discrepancy, Defined.ai
  clause, LibriVox publicity opinion, studio-recording budget.
- **Open blockers:** none blocking Phase D/E.
