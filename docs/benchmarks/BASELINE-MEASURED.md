# MEASURED Baselines — Caedral CPU Benchmark Lab

Machine: i7-13620H (10C/16T, AVX2/FMA3, no AVX-512), Windows 11, 23.7 GB RAM.
Corpus: `benchmarks/corpus/eval-corpus-v1.0.json`. Full provenance (machine
snapshot, git commit, ORT settings) embedded in each `benchmarks/results/**`
meta file. All numbers below are **MEASURED locally** unless labeled otherwise.
Warm RTF excludes one-time load; audio at batch-1.

## Results (2026-08-22)

### Performance (MEASURED locally)

| Baseline | License | Params/class | Warm RTF | Peak RSS | Long-form | Notes |
|---|---|---|---|---|---|---|
| Kokoro-82M v1.0 ONNX | Apache-2.0 | 82M NAR+iSTFT | 0.32–0.49 | ~1.25 GB native peak | ✓ stable (339 s audio, RTF steady ~0.37) | load 1.4 s; native "stream" = 1 chunk for typical inputs → TTFA ≈ full gen |
| Piper en_US-medium ×2 | GPL-3.0 (**isolated subprocess only**) | ~20M VITS-class | 0.06–0.24 | small (see rows) | ✓ RTF 0.06 | fastest by far; spawn+load ~0.5 s first call |
| KittenTTS nano ONNX | Apache-2.0 (+ espeak-ng GPL runtime dep) | nano | 0.37–0.42 | small (see rows) | ✗ CRASH >~1 k-char input | ONNX `/bert/Expand` invalid shape = position-embedding ceiling; "nano" ≠ fast (4× slower than Piper) |

### Quality (MEASURED locally; auto tier of EVAL SPEC v0)

Scoring stack: faster-whisper small.en int8 + base.en int8 dual-ASR normalized
min-WER (documented deviation from spec's Parakeet second-ASR), DNSMOS v8
ONNX (sanity tier, clean-speech ceiling ≈3.3–3.5 observed here). 30 samples =
5 corpus items × 2 voices × 3 baselines.

| Baseline | min-WER range | DNSMOS ovrl range |
|---|---|---|
| Kokoro-82M | 0.00–0.33 | **3.22–3.51** |
| Piper-medium | 0.00–0.33 | 3.07–3.46 |
| KittenTTS-nano | 0.00–0.36 | 2.87–3.39 |

Notes: `acr_005` ("AVX2 and AVX-VNNI… AVX-512") scores WER≈0.33 for ALL
baselines including perfect-pronunciation ones — an ASR normalization artifact
(hyphenated acronyms), not a TTS failure; flagged for corpus-aware scoring
rules rather than corpus edits (freeze policy respected). Full rows:
benchmarks/results/quality/score-20260822T131103Z.jsonl. UTMOS remains a
backlog item (PyPI 'speechmos' lacks it).

## Key architectural lessons already extracted

1. **Caller-side chunking is mandatory** for long-form in every small model
   tested (KittenTTS crashes without it; Kokoro's native streaming doesn't
   chunk usefully). TTFA must come from our runtime's sentence pipelining.
2. **Small ≠ slow is false; architecture matters**: Piper (~20M) beats
   Kokoro (82M) AND KittenTTS (nano) by 2–6× on this CPU. The winning
   pattern per Researcher C (NAR, few parallel passes, iSTFT head) matches
   what actually measures fast.
3. **RSS discipline**: Kokoro's 1.25 GB working set exceeds profile B (2 GB)
   comfortably but not profile C (1 GB). A Caedral Lite/Tiny must target
   well under this.

## Baselines deferred (documented, honest)

| System | Reason |
|---|---|
| Chatterbox-Nano (110M, MIT) | `chatterbox-tts==0.1.7` exposes no nano class; `from_pretrained(device)` hardwired to default repo. Import cost alone 80 s (torch 2.6 + transformers 5.2 + gradio stack). Revisit via GitHub-main install as CV-G1-E#### later. Vendor claim "3× realtime @8 cores" remains REPORTED BY THIRD PARTY, unverified. Sources: [HF card](https://huggingface.co/ResembleAI/chatterbox-nano), [repo](https://github.com/resemble-ai/chatterbox), [vendor docs](https://resemble.ai/learn/models/chatterbox-nano) |
| F5-TTS (flow-matching ref) | weights CC-BY-NC (Emilia) → evidence-only value; heavy torch stack; flow-matching reference will instead come from ZipVoice-class if needed |

## ORT thread sweep (MEASURED, Kokoro-82M, af_heart, conv_001+para_001)

| OMP threads | RTF conv_001 | RTF para_001 |
|---|---|---|
| 1 | 0.5998 | 0.4551 |
| 2 | 0.6038 | 0.4624 |
| 4 | 0.5521 | 0.4741 |
| 8 | 0.5786 | 0.3815 |
| default (16) | 0.3512 | 0.3229 |

Findings: weak thread scaling beyond ~4 (memory-bound + hybrid P/E-core
scheduling per CPU-INFERENCE-OPTIMIZATION.md predictions); single-thread RTF
still <1.0 ⇒ 1-vCPU profiles C/D remain real-time-viable for the Kokoro class.
Default all-thread mode is fastest but risks the documented bimodality under
concurrency — revisit at Phase O.

## Pending (Phase C continuation)

- UTMOS predictor integration (PyPI gap documented above).
- Chatterbox-Nano + F5-TTS reference benchmarks (deferred, reasons in ledger).
