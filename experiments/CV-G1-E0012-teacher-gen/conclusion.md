# CV-G1-E0012 — teacher-gen: synthetic teacher corpus build (TEACH-01 implementation)

Status: **RUNNING** (started 2026-08-22). Design:
[docs/training/TEACHER-DATA-ENGINE.md](../../docs/training/TEACHER-DATA-ENGINE.md)
(founder-authorized; teacher audio = training data only, never shipped).
Builder: [scripts/build_teacher_set.py](../../scripts/build_teacher_set.py).

## Setup

- Teacher engine: Kokoro-82M v1.0 ONNX via kokoro-onnx 0.6.1,
  `models/kokoro/kokoro-v1.0.onnx` sha256 `7d5df8ec…363a6c5`,
  `models/kokoro/voices-v1.0.bin` sha256 `bca610b8…f1fbf7d`.
  License Apache-2.0 (code + weights).
- Voice slate (frozen, TEACHER doc §1): t00 af_heart, t01 af_bella,
  t02 af_nicole, t03 bf_alice, t04 bf_emma (5F) / t05 am_adam, t06 am_fenrir,
  t07 am_michael, t08 am_puck, t09 bm_george (5M); round-robin by global item
  index; all 10 names verified present in the voices bin at runtime.
- Text source: first `--limit 16000` rows of
  `data/manifests/librittsr-clean100.phonemes.jsonl` (LibriTTS-R CC-BY-4.0),
  dedupe on normalized text, >300-char skip, eval-corpus firewall
  (`benchmarks/corpus/eval-corpus-v1.0.json` normalized match ⇒ reject).
- Pair construction ("ours on both sides"): input side = OUR frontend phonemes
  joined by `source_id`; target side = OUR 80-band ln-mel computed from the
  stored wav at train time (`caedral_voice/training/mel_dataset.py`). Every
  clip is validated inline through that exact chain (mel never stored).
- Output: `data/processed/teacher-kokoro/<id>.wav` (24 kHz PCM16 mono),
  manifest `data/manifests/teacher-kokoro.jsonl` (+ `.rejected.jsonl`,
  `.meta.json` with license/provenance block; shard files merged atomically).
- Execution: two detached workers (`--shard 0|1 --num-shards 2`, contiguous
  id-range split), 3 ORT/torch threads each (ORT SessionOptions +
  OMP/MKL env caps), BelowNormal priority, coexists with ALIGN-01 training.
- Inline quality gates: reject duration <1 s or >20 s, clipped_frac
  (>0.99 abs) >0.1%, MAS-infeasible (actual mel frames < n_phones),
  non-finite mel, synthesis error. Resumable: done ids skipped on restart;
  append-only shard jsonl flushed per line, torn-line-tolerant readers.

## Results (to fill on completion)

- [MEASURED] totals: see `data/manifests/teacher-kokoro.meta.json`
  `totals` block (kept / rejected-by-reason / audio_hours /
  per_voice_counts / effective RTF) — live-updated every 25 kept clips.
- Per-shard progress snapshots: `shard0.progress.json`, `shard1.progress.json`;
  full logs `shard0.log`, `shard1.log`.

| metric | value |
|---|---|
| kept utterances | TBD |
| kept audio hours | TBD (target ≥12 h this session) |
| rejection breakdown | TBD |
| per-voice counts | TBD |
| effective RTF (synth_s / audio_s) | TBD |
| disk used (172.8 MB/h) | TBD |

## Honesty notes

- `lufs_before/lufs_after` fields carry an uncalibrated RMS dBFS proxy; audio
  is raw synthesis output (no loudnorm at generation time).
- Durations in the manifest are for batch-sampler estimates only; loss-side
  durations come from MAS at train time exactly as for real data.
- Gate thresholds here are generation-time sanity bounds (task spec), not the
  §4 QA ASR/DNSMOS screens, which remain a post-generation concern.
