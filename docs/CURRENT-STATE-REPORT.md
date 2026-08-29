# CURRENT STATE REPORT — Caedral Voice Program

Date: 2026-08-22 · Machine: i7-13620H (10C/16T, AVX2/FMA3, no AVX-512), 23.7 GB RAM, Windows 11 [MEASURED]
Labels: MEASURED / IMPLEMENTED / PARTIAL / PLANNED / HYPOTHESIS / THIRD-PARTY BASELINE

## Program status

| Phase | State |
|---|---|
| A environment | COMPLETE |
| B research | COMPLETE — 10 docs, ~500 citations, 49 UNVERIFIED flags tracked |
| C benchmarks | COMPLETE — 3 third-party baselines perf+quality, thread sweep, eval stack |
| D data | PARTIAL — LibriTTS-R smoke manifest frozen+validated; LJSpeech slice extracting (detached); PILOT download PLANNED |
| E frontend | COMPLETE — own-chain TN+lexicon KEEP, neural fallback wired |
| F architecture | COMPLETE for G1 direction — NAR+flow chosen via E0002/E0005/E0006 |
| G decoder | DECIDED + MEASURED — Vocos-class (MIT), RTF measured on this box |
| H voices/styles | PREPARED — catalog schema written; conditioning implemented in model; identity validation PARTIAL (unit-level only) |
| I training system | COMPLETE — bitwise-exact resume proven |
| J pilot model | IN PROGRESS — pipeline being built by agent; long run NOT started |
| K–P | NOT STARTED |

## Experiments executed (experiments/EXPERIMENT-INDEX.md)

CV-G1-E0001 frontend bake-off · E0002 NAR prototype · E0003 training system ·
E0004 g2p_en fallback · E0005 alignment strategy · E0006 MAS+flow decoder.
All with conclusion.md + evidence labels. Six more planned (see DAG below).

## Research collected

~500 cited sources across 10 documents [MEASURED count]; Firecrawl migration
done (8 core pages cached, index live). Datasets analyzed: 24-row license
matrix; datasets downloaded: LJSpeech 2.6GB archive + LibriTTS-R clean-100
7.6GB archive (SMOKE slices extracted).

## Architecture decision state

- **Chosen [IMPLEMENTED as prototype]:** CAEDRAL-NAR — phoneme encoder +
  duration/pitch/energy predictors + conditional flow decoder (MAS-trained,
  no external aligner) + iSTFT head. Tiny 8.35M / Lite 31.6M / Quality 80.7M.
- Open decisions: quantization targets (Phase M), streaming chunk policy,
  final voice slate, style-table bootstrapping method (PLANNED docs queued).

## Capability inventory

| Capability | State |
|---|---|
| Text frontend/tokenizer | IMPLEMENTED (normalize/g2p/vocab; 44 tests) |
| Acoustic model | IMPLEMENTED (prototype, untrained; shapes+latency verified) |
| Speaker system | IMPLEMENTED (embedding table d=128 + FiLM; sensitivity tested unit-level) |
| Prosody knobs | IMPLEMENTED (speed/pitch/energy transforms in forward API) |
| Style system | PARTIAL (table exists; no style-labeled training data yet) |
| Training pipeline | IMPLEMENTED + PROVEN (bitwise resume; NaN guard; checkpoints) |
| Inference pipeline | PARTIAL (model.infer() exists; no packaged CLI/server yet) |
| Decoder/vocoder | THIRD-PARTY BASELINE (pretrained Vocos, MIT) — Caedral-trained head PLANNED |
| CPU runtime | PLANNED (ONNX export path next; torch eager today) |
| Quantization | PLANNED |
| Tests | IMPLEMENTED — 104 passing [MEASURED] |

## Current numbers

### Third-party baselines [THIRD-PARTY BASELINE, measured here]

| System | Warm RTF | Peak RSS | Quality (minWER / DNSMOS) |
|---|---|---|---|
| Kokoro-82M v1.0 | 0.32–0.49 | ~1.25 GB | 0–0.33 / 3.22–3.51 |
| Piper medium ×2 | 0.06–0.24 | small | 0–0.33 / 3.07–3.46 |
| KittenTTS nano | 0.37–0.42 | small | 0–0.36 / 2.87–3.39 |

### Caedral prototypes [MEASURED, UNTRAINED — no perceptual claims]

| Config | Params | fwd @800 phones | RSS |
|---|---|---|---|
| tiny | 8.35M | 229 ms@1thr / 134 ms@4thr | 235 MB |
| lite+flow (E0006) | ~32M | 419 ms@1thr / 293 ms@4thr | ~334 MB |

Model size on disk: n/a (no trained checkpoint yet). TTFA: n/a for Caedral;
baselines: Kokoro native streaming ineffective (measured), Piper subprocess
spawn ~0.5 s.

## DO WE CURRENTLY HAVE A CAEDRAL-GENERATED VOICE?

**NO.**

Every WAV produced so far came from THIRD-PARTY BASELINES (Kokoro/Piper/
KittenTTS) used for measurement. The Caedar model exists only as an untrained
prototype. Pretending otherwise would violate charter §26.

→ FIRST-CAEDRAL-VOICE track activated (docs/training/FIRST-CAEDRAL-VOICE.md):
earliest legitimate audio = tiny-config pilot checkpoint decoded through our
frontend→model→Vocos path, generated as soon as training produces its first
usable checkpoint. artifacts/latest-preview/ will hold the newest previews
with full provenance README, marked PROTOTYPE — NOT PRODUCTION QUALITY.
