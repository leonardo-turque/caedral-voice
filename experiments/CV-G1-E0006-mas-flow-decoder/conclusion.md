# CV-G1-E0006 — Glow-lineage MAS training + flow decoder for CAEDRAL-NAR

**Date:** 2026-08-22 · **Machine:** i7-13620H (6P+4E, 16 logical), Windows 11, py3.13.15, torch 2.6.0+cpu
**Implements:** CV-G1-E0005 DECISION ("KEEP: MAS + flow decoder for G1 pilot") — pilot training (Phase J) needs NO external aligner.
**Code:** `caedral_voice/model/flow_decoder.py` · `caedral_voice/model/mas.py` · `caedral_voice/model/caedral_nar.py` (`use_flow` wiring)
**Tests:** `tests/test_flow_mas.py` (12 passed; full suite 104 passed incl. all E0002 regressions)
**Raw results:** `results.jsonl` · **Runner:** `run_benchmark.py`. All numbers below [MEASURED].

---

## HYPOTHESIS

A conditional normalizing-flow decoder (affine-coupling stack, Glow-TTS lineage) can replace the
E0002 deterministic decoder so that Monotonic Alignment Search supplies hard durations during
training, **without breaking the E0002 batch-1 CPU latency budget** — full training-loss
computation AND inference stay within 1.3x of the E0002 lite forward medians
(891.2 ms @1thr / 592.4 ms @4thr at 800 phonemes; 194.2 / 123.8 ms at 200).

## METHOD

- **Flow decoder** (`FlowDecoder`): per block an invertible 1x1 conv (LU-decomposed weight,
  W = P·L·Û, log|det W| = Σ log_diag exactly by construction, fixed random permutation at init)
  followed by channel-halved affine coupling whose conditioner is a WaveNet-ish gated residual
  CNN (tanh·sigmoid gating) with per-layer FiLM injection of the global speaker|style vector;
  tanh-bounded log-scales (±2). Zero-init conditioner output ⇒ the stack starts near a
  mel↔N(0,I) bijection. `forward(mel, cond) -> (z, per-frame logdet)`; `inverse(z, cond, vec,
  noise_scale)`; optional per-frame `cond` surface (B,T,C) supported and tested, unused by the
  pilot wiring (see simplification S1).
- **MAS** (`mas.maximum_path`): log-domain DP over the score matrix ln N(z_j; m_i, σ_i)
  (prior stats from `flow_prior_proj` heads on the encoder). Row-scan formulation — one cumsum +
  one running-max scan per phoneme row, O(T_x·T_y), float64, plus O(T_x) backtrack reading
  durations directly off chosen run-starts; mathematically the same optimum as the reference
  b_top/b_bot recursion. A literal nested-loop DP (`maximum_path_reference`) ships alongside and
  the tests assert BOTH return identical optima (durations bitwise, scores to 1e-6) on random
  ragged-batch matrices + known-solution synthetics.
- **Training step** (`losses_for_training`): one flow pass data→latent (speaker|style-conditioned
  only) ⇒ chunked score matrix ⇒ hard MAS path (no grad through search) ⇒
  `flow_loss = -(Σ lnN(z_j; m_{A(j)}, σ_{A(j)}) + Σ logdet_j)/T_y` recomputed WITH grad under the
  chosen alignment; `duration_loss` = log-normal prior NLL of MAS durations under the duration
  predictor (σ=1, per valid phoneme); pitch/energy placeholder losses = 0.
- **Inference** (`infer`): durations from the predictor (or teacher-forced override), prior stats
  expanded per frame, z = m_e + noise_scale·σ_e·ε, inverse flow, iSTFT head.
- **Latency protocol:** median-of-10 after 3 warmups (`benchmarks.harness.measure`), threads {1,4},
  lengths {200,800}, lite config + use_flow(flow_hidden=256, flow_blocks=6). Gated cells use the
  E0002 baseline geometry (mel frames == phonemes); see R3 for why an explicit forced-1-frame/
  phoneme inference cell was required. Three full runs executed; verdict identical in all.

## RESULT

**Budget gate: 8/8 gated cells PASS (gate ≤ 1.3x E0002 forward median). Final run:**

| Path | thr | T=200 ph | ratio | T=800 ph | ratio |
|---|---|---|---|---|---|
| training_loss | 1 | 80.8 ms | 0.42x | 418.7 ms | 0.47x |
| training_loss | 4 | 63.5 ms | 0.51x | 322.3 ms | 0.54x |
| inference (forced 1 fpp) | 1 | 87.8 ms | 0.45x | 424.3 ms | 0.48x |
| inference (forced 1 fpp) | 4 | 66.7 ms | 0.54x | 293.1 ms | 0.49x |

(reference: E0002 lite forward 194.2/891.2 ms @1thr, 123.8/592.4 @4thr.) Informational cells:
inference at this init's own predicted durations (262 frames @200 ph / 1042 @800 — untrained
predictor emits ~1.3 frames/phoneme here vs ~1.0 in E0002): 117.1/455.0 ms @1thr, 82.0/414.1 @4thr
— still inside every gate despite ~30% extra frames. Training-loss x4-expansion anchor
(200 ph → 800 frames): 192.4 ms @1thr / 207.9 @4thr vs E0002 forward-only anchor 616.8/285.3 ms.
Run-to-run variance on this mobile chip reached ~2x on individual cells across the day (first run
slower across the board); worst gated ratio observed in ANY of the three runs was 0.72 — the
verdict never approached the 1.3x line.

**Correctness [MEASURED]:** roundtrip error inverse(forward(mel)) = 0.0 (float64) / < 1e-6 fp32-scale;
flow log-det == autograd Jacobian slogdet to <1e-3 after parameter perturbation; fast MAS ==
textbook DP on 200 random trials (durations exact, scores ≤1e-6) + known uniform/skewed solutions;
same-seed losses bitwise-equal; 50-step Adam run on random tensors: flow_loss strictly decreasing
(< 0.9x start), grads reach flow/prior/duration/encoder params, all losses finite throughout.

## INTERPRETATION

1. **HYPOTHESIS SUPPORTED [MEASURED].** MAS+flow costs LESS than the regression decoder it
   replaces at equal geometry (0.42–0.54x): the conv-stack flow over ~T frames is cheaper than
   the d512 5-layer attention decoder, and MAS adds only an O(T_x·T_y) score matrix + row-scan DP
   (tens of ms at utterance scale) because text conditioning lives in the prior heads, not in the
   flow. Phase J trains self-contained — no torchaudio/MFA aligner dependency (E0005 decision
   executed end-to-end).
2. **The factorization choice is load-bearing (S1).** Feeding phoneme-expanded features into the
   flow as per-frame conditioning would force T_x flow passes per MAS step (~100x cost — dead on
   arrival against this budget). We implement the canonical Glow-TTS split instead: flow sees
   only speaker|style (FiLM); text enters through (m_p, σ_p) compared against latents. The
   per-frame `cond` API exists, works, and is tested for future non-MAS uses.
3. **Geometry honesty (S2/R3):** init-luck changed the untrained predictor's expansion (1.3 vs
   1.0 fpp), which alone produced a spurious gate miss on inference@200@4thr in run 1. Gates are
   judged on matched-geometry (forced 1 fpp) cells; predicted-duration cells are reported
   informationally. Lesson recorded: baseline comparisons must pin frame counts, not trust stub
   predictors.
4. **Param point shifts with the architecture (S3):** lite+flow is 18.9M — the flow replaces the
   17.7M decoder transformer rather than adding to it. That lands almost exactly on the Matcha-18M
   quality-per-FLOP anchor [REPORTED][ACOUSTIC §12] and reopens the question of whether
   "lite" should grow the flow (more blocks/hidden) to reclaim the freed budget in Phase J.
5. Stub-quality caveats as in E0002: correctness contracts are structural (invertibility, log-det,
   optimality, determinism, single-step learnability); no perceptual claims until trained.

## KEEP / REJECT / INVESTIGATE

**KEEP**
- `use_flow` wiring, `FlowDecoder`, `maximum_path` (row-scan + reference cross-check) frozen as
  the Phase J training surface: `losses_for_training` / `infer` contracts per module docstring.
- Matched-geometry gating rule for all future latency budgets (pin frame counts explicitly).

**REJECT**
- Text-conditioned-inside-the-flow MAS variants (per-candidate flow passes) — ≥100x over latency
  budget by construction; revisit only if a likelihood-based need arises that MAS cannot serve.

**INVESTIGATE (next)**
- Phase J pilot: real-mel training stability (tanh-bounded scales, zero-init coupling chosen for a
  benign start; watch duration-predictor/MAS feedback loop at 1.3-fpp-style drift).
- Reclaim the lite param budget: grow flow to ~30M total (deeper/wider couplings) and re-gate.
- torch.compile / ONNX pass over the flow path (invconv triangular solves are eager overhead);
  thread policy per workload class confirmed still relevant (4thr helps only at T≥800).

*Reproduce:* `.venv/Scripts/python.exe -X utf8 experiments/CV-G1-E0006-mas-flow-decoder/run_benchmark.py`
(~2 min) · Tests: `.venv/Scripts/python.exe -m pytest tests/test_flow_mas.py tests/test_model.py -q`

*Simplification ledger:* S1 Glow-TTS factorization (text via prior heads; flow conditioned on
speaker/style FiLM only) · S2 duration-prior σ fixed 1.0 · S3 `noise_scale` implemented as latent
temperature on the prior sample (z = m_e + ns·σ_e·ε); literal N(0, ns·I) sampling is text-blind
under S1 · S4 invconv initialized as a fixed random permutation (dense orthogonal QR init skipped);
couplings zero-init compensate · S5 pitch/energy training losses are placeholders (0) per charter.
