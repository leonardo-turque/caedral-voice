# CAEDRAL Voice Catalog — G1 configuration format

Charter §12: each released voice = a well-defined configuration over model
conditioning, never hardcoded random behavior. Placeholder names below are
internal codenames; final public naming happens before Phase N freeze.

## Schema (configs/voices/<voice_id>.json)

```json
{
  "voice_id": "codename-f01",
  "version": 1,
  "gender_presentation": "feminine",
  "speaker_embedding_index": 0,
  "default_style": "neutral",
  "allowed_styles": ["neutral", "warm", "professional", "calm"],
  "prosody_ranges": {
    "speed":     [0.85, 1.15],
    "pitch_shift_semitones": [-2.0, 2.0],
    "energy_gain": [0.9, 1.1]
  },
  "training_data_ref": "manifests/<dataset-manifest>.jsonl#speaker=1183",
  "identity_anchor": {
    "method": "training-speaker-id",
    "notes": "identity derives from licensed training speaker; no cloning"
  },
  "provenance": {
    "dataset": "LibriTTS-R clean-100 (CC-BY-4.0)",
    "consent_class": "public-domain-derived-read-speech"
  }
}
```

## Rules

1. `speaker_embedding_index` must exist in the trained table; catalog entries
   are validated against checkpoints at load time.
2. Prosody knobs are clamped to `prosody_ranges` at the inference API layer —
   the model never sees out-of-range values (Phase H design).
3. Every voice ships with provenance rows; voices without cleared provenance
   are BLOCKED from release builds by the packaging script (Phase N).
4. Long-form consistency defaults (per-voice chunk-overlap settings) live in
   this file too once Phase J/K establishes them.

## G1 candidate slate (provisional, evidence-gated)

| Codename | Presentation | Target character | Data source plan |
|---|---|---|---|
| codename-f01 | feminine | warm narration | LibriTTS-R female speaker |
| codename-f02 | feminine | energetic/conversational | LibriTTS-R female speaker |
| codename-m01 | masculine | professional/serious | LibriTTS-R male speaker |
| codename-m02 | masculine | calm documentary | LibriTTS-R male speaker |

Speaker selection criteria (Phase J): ≥30 min kept audio in manifest, low
rejected-rate, accent breadth across the four, distinct F0 stats measured on
the manifest. Final identity quality gate happens with human listening at
Phase K/N; auto-metrics cannot resolve identity distinctiveness alone
(EVALUATION-BENCHMARKS.md finding).
