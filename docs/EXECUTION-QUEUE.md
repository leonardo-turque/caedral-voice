# EXECUTION QUEUE — resume instructions (keep current)

Last updated: 2026-08-22, during Phase C continuation after user directive:
"CONTINUE autonomously through ALL phases C→P. Checkpoints persist, never stop
the run. ≤3 concurrent local-heavy subagents; ONE heavy local model at a time;
monitor RSS; OOM = recorded result, reduce concurrency and continue; retry/replace
failed subagents."

## Immediate queue (in order)

1. **Phase C completion**
   a. Quality scoring: `benchmarks/score_baseline_samples.py` per EVAL SPEC v0
      (docs/research/EVALUATION-BENCHMARKS.md): dual-ASR normalized min-WER
      (deviation allowed+documented if Parakeet unavailable: two distinct Whisper
      variants) + UTMOS-class predictor (speechmos pkg, torch already installed).
      Input: artifacts/audio/baselines/** (30 wavs, manifest committed).
      Output: benchmarks/results/quality/*.jsonl + rows added to
      docs/benchmarks/BASELINE-MEASURED.md.
   b. Thread sweep: fresh-process runs of benchmarks/run_baseline_kokoro.py
      with OMP_NUM_THREADS={1,2,4,8} on items conv_001,para_001, voice af_heart.
      Record bimodal latencies / hybrid P-E core effects.
   c. CHECKPOINT C → docs/checkpoints/PHASE-C-BENCHMARKS.md + commit.
2. **Phase D**: verify subagent-built data toolkit (data/scripts/*), download
   SMOKE sets (LJSpeech slice ~0.45GB, LibriTTS-R clean-100 head ~0.9GB,
   CV-EN ~50MB shards) per docs/data/DATASET-SURVEY.md, run pipeline to
   manifests with hashes, commit manifests. PILOT (~16GB) only after SMOKE works.
   CHECKPOINT D.
3. **Phase E**: frontend experiments per docs/research/TEXT-FRONTEND.md test
   list (char-vs-phoneme ratio, CMUdict coverage/OOV, G2P bake-off incl.
   dev-only espeakng-loader comparison, heteronym torture, latency). Choose
   frontend. CHECKPOINT E.
4. **Phase F**: calibrate CPU (GEMV GFLOPS/bandwidth microbench); implement
   minimal NAR prototype (caedral_voice/model) w/ param counts + measured
   forward latency at {8M,30M,80M}; reject/confirm H1-H4 ranking.
   CHECKPOINT F.
5. **Phase G**: vocoder RTF measurements (Vocos MIT weights via HF,
   torch CPU; export ONNX if cheap) → decoder decision. CHECKPOINT G.
6. **Phases H–P**: per MASTER-PLAN §6; training system must prove resume
   (I); pilot model trained honestly on SMOKE data with MAS-style alignment
   or documented alternative (J); audit by FRESH subagent (P).

## Standing rules

- Every number labeled MEASURED / REPORTED BY THIRD PARTY / ESTIMATED / HYPOTHESIS.
- Commit after every meaningful artifact. EXPERIMENT-INDEX.md maintained.
- Models unloaded after benchmarks (del + gc; separate processes preferred).
- Never fake training scale; document compute reality per charter §15.

## UPDATE 2026-08-22 (research infra policy)

ALL web research now via research/firecrawl_client.py per
docs/research/FIRECRAWL-WORKFLOW.md. Cache-first: check WEB-SOURCE-INDEX.md +
research/cache/ before any fetch. OpenRouter search retired (one-time bootstrap
done). Key in USER env FIRECRAWL_API_KEY; never print/commit. If 401/403:
log pending URLs here and continue offline.

## V1 DIRECTIVE (founder, 2026-08-22 evening)

Intelligible prototypes = DIAGNOSTIC ONLY. DNSMOS ~2.0 is NOT product.
V1 bar: >=15 distinct product-quality voices, naturalness competitive with
medium modern TTS (numeric gate = match/beat Kokoro DNSMOS measured on THIS
machine), full benchmark table, independent reviewers, packaged
artifacts/caedral-voice-v1/ + final report. DO NOT STOP until gates pass.

Active arms:
- B-arm (primary quality path): CTC aligner (own IP) -> supervised durations ->
  mel-regressor trained on clean100 + Kokoro-teacher mels (Apache; founder-
  authorized teacher). E0009 proved chain fidelity; regression converges far
  faster than flow on CPU.
- A-arm (own-data control): clean100 flow retrain queued behind B-arm resources;
  provides honest from-real-data comparison for the final report.
- Inference sigma default now 0.25 (E0009 finding).

