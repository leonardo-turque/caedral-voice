# CHECKPOINT — PHASE C (Benchmark Harness + Baselines)

Date: 2026-08-22 · All numbers MEASURED locally on i7-13620H unless labeled.

## 1. Research performed
Baseline selection informed by Phase B BASELINE-MATRIX shortlist; eval stack
per CAEDRAL EVAL SPEC v0; thread-sweep design per CPU-INFERENCE-OPTIMIZATION.md.

## 2. Sources consulted
See docs/research/SOURCES.md; baseline-specific licenses verified during install
(kokoro-onnx Apache; piper GPL-3.0 → isolated subprocess; kittentts Apache with
GPL espeak runtime via espeakng-loader; faster-whisper MIT; DNSMOS models via
speechmos pkg).

## 3. Findings
- Kokoro-82M: RTF 0.32–0.49 (default threads), 0.38–0.60 @≤8 threads,
  peak RSS ~1.25 GB, load 1.4 s, long-form stable 339 s audio.
- Piper-medium ×2: RTF 0.06–0.24 warm (subprocess incl. spawn+load) — fastest.
- KittenTTS-nano: RTF 0.37–0.42; HARD CRASH on >~1k-char input (position-
  embedding ceiling). Nano ≠ fast.
- Quality (dual-ASR minWER + DNSMOS): Kokoro best (DNSMOS 3.22–3.51),
  Piper 3.07–3.46, Kitten 2.87–3.39; minWER ≤0.36 all; acr_005 WER artifact
  documented (ASR normalization, not TTS failure).
- Thread scaling weak (memory-bound + hybrid P/E cores); 1-vCPU profiles
  remain real-time viable.
- kokoro-onnx native streaming yields 1 chunk for typical inputs → TTFA
  must be built by caller-side pipelining.

## 4. Decisions made
- Piper benchmarked only as isolated subprocess (GPL policy) — precedent for
  any future GPL tool usage.
- Eval stack deviation documented: second ASR = whisper base.en (Parakeet
  unavailable on Windows/CPU); DNSMOS as quality floor; UTMOS backlog.
- Corpus freeze policy held: acr_005 artifact handled by scoring-rule note,
  not corpus edit.

## 5. Files created/changed
benchmarks/: run_baseline_kokoro.py, run_baseline_piper.py,
run_baseline_kittentts.py, persist_baseline_samples.py,
score_baseline_samples.py, results/** (perf+quality JSONLs, manifests).
docs/benchmarks/BASELINE-MEASURED.md (ledger). docs/EXECUTION-QUEUE.md.
docs/checkpoints/PHASE-C-BENCHMARKS.md (this file).

## 6. Experiments executed
Benchmark runs (not CV-G1-E#### experiments — those start with training):
3 baselines × 6-item subset × 2 voices; TTFA probes; thread sweep {1,2,4,8,default}.

## 7. Exact measured results
See docs/benchmarks/BASELINE-MEASURED.md tables + raw JSONLs under
benchmarks/results/ (provenance-embedded).

## 8. Failed experiments / incidents
- KittenTTS long_001 crash → recorded as architecture finding.
- chatterbox-tts v0.1.7 lacks nano class → deferred with reasons.
- speechmos PyPI pkg ≠ UTMOS → DNSMOS used, UTMOS backlog.
- kokoro-onnx API differences (providers/lang/async stream) → fixed in driver.

## 9. Remaining risks
Quality signals are auto-tier only — no human listening yet; DNSMOS ceiling
limits resolution between good systems; acr_005-type artifacts need scoring
rules; deferred baselines may change conclusions if benchmarked.

## 10. Disk usage
C: free ≈118.8 GB. models/ ≈ 0.5 GB (kokoro 337 MB + piper 127 MB + kitten).
Repo incl. venv ≈ 2.7 GB. No dataset downloads yet.

## 11. Current best model metrics
NONE (no Caedral model yet — honest null). Reference targets established:
beat-or-match Kokoro DNSMOS at ≤0.5 GB RAM, RTF ≤0.5 @1-2 threads.

## 12. Next phase
PHASE D — data program: verify subagent-built preprocessing toolkit, download
SMOKE sets (LJSpeech slice, LibriTTS-R clean-100 head, CV-EN shards),
produce hashed manifests, integrity-gate them. PILOT (~16 GB) only after
SMOKE pipeline proven.
