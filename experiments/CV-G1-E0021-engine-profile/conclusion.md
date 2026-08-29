# CV-G1-E0021-engine-profile — Conclusion

Date: 2026-08-23 · Machine: Windows 11, Intel Model 186 (Meteor Lake), 16 logical CPUs ·
kokoro-onnx 0.6.1 · onnxruntime 1.29.0 · Python 3.13.15 · model `kokoro-v1.0.onnx` (310.5 MB)

## Headline numbers

| Config | COLD first synth | WARM median (per sentence) | Warm RTF range | Peak RSS |
|---|---|---|---|---|
| intra_op=2 | 3.90 s (RTF 1.39) | evalA 5.05 s / propernoun 4.99 s / numbers 10.28 s | 1.72–1.81 | 1044 MB |
| intra_op=4 | 2.21 s (RTF 0.79) | evalA 2.53 s / propernoun 2.29 s / numbers 5.56 s | 0.83–0.93 | 1044 MB |

Startup-to-first-audio (fresh process, one short sentence): **6.58 s parent wall**
(child: model load alone 2.25 s; remainder is interpreter + heavy imports).
Baseline RSS after load 611–612 MB; synthesis adds ~433 MB transient peak.

## Phase breakdown (warm medians)

The v1.0 export fuses acoustic model and vocoder into ONE output (`audio`,
verified at runtime) — there is no separate vocoder decode phase:

| Phase | threads=2 | threads=4 |
|---|---|---|
| frontend_phonemize (Caedral TN+G2P+map) | 3.64 ms | 2.99 ms |
| frontend_cache_lookup (text-hash hit) | 21 µs | 23 µs |
| tokenize | 19 µs | 20 µs |
| **session_run_acoustic (fused graph)** | **4715 ms** | **2658 ms** |
| vocoder_decode_if_any | n/a (fused) | n/a (fused) |
| wav_serialize (WAV float32, in-memory) | 0.55 ms | 0.64 ms |
| create() total (is_phonemes=True) | 4734 ms | 2684 ms |

**99.6 % of synthesis wall is the fused ONNX session.run.** Frontend,
mapping, tokenization and WAV encoding are noise (<0.2 % combined). Every
optimization dollar lives in ORT execution: thread count, provider choice,
quantization, or graph-level changes.

## Findings

1. **threads=4 >> threads=2**: ~1.78x faster on the fused graph (RTF 1.8 -> 0.9).
   intra_op=2 cannot reach real time on this box today; 4 crosses RTF < 1.
   Default the canonical engine to >=4; probing 8 is the obvious E0022.
2. **Warm > cold in this run** (e.g. 5.05 s warm vs 3.90 s cold @t2): sustained
   synthesis outlives the boost-clock budget and throttles; the "cold" call
   rides initial boost. This plus ambient contention fully explains the
   historical RTF scatter (0.32–0.49 idle vs 2.5–10.8 loaded): **today's idle
   RTF is 0.83–0.93 @t4**, still above the historical idle band — treat any
   single-session RTF as thermal-state-dependent. Future benchmarks need a
   fixed thermal protocol (cooldown, interleaved configs, rep counts raised)
   before A/B claims.
3. **Frontend caching works as designed**: text-hash cache hits cost ~22 µs
   vs 3–4 ms full phonemize (~150x), and NLTK/g2p construction (5–8 s) plus
   model load (2.25 s) are one-time costs the long-lived engine amortizes.
4. `headline_rtf` in profile.json pairs pooled-wall median with one
   sentence's audio duration — per-sentence `rtf_median` rows are the
   authoritative RTFs quoted above.

## Deliverables

- `scripts/profile_kokoro.py` — parent/worker profiler (fresh subprocess per
  thread config; psutil RSS sampler; startup probe; writes this profile.json)
- `caedral_voice/inference/kokoro_engine.py` — `CaedralVoiceKokoroEngine`
  (`__init__(model_path, voices_path, threads)` · `.synth(text, voice_name,
  speed=1.0) -> float32 @24k` · `.voices` · `.sample_rate` · `.phonemize(text,
  use_cache=True)` · `.infer_from_phonemes()` · `.warmup()` · `.last_stats`)
- `caedral_voice/text/kokoro_map.py` — promoted char-level vocab mapping
  (`caedral_tokens_to_kokoro`, alias/stress logic identical to
  test_kokoro_vocab_path_v2.py)
- Tests: `tests/test_kokoro_map.py` (8, pure) · `tests/test_kokoro_engine.py`
  (6, real model; load-once fixture, warm<=cold reuse assert, audio contract,
  unknown-voice/speed raises, cache-hit) — all 14 pass.
