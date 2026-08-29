# PRE-ARCH TOP-3 — Best Current Caedral Voice Previews

**STATUS: BEST CURRENT CAEDRAL VOICE PREVIEW — NOT V1, NOT PRODUCTION-READY.**

Generated 2026-08-23 from the teacher-augmented regressor r3 snapshot (step 9000,
val mel-L1 0.8487). Full pipeline: Caedral frontend (lexicon+g2pen) →
CaedralRegressor 15.94M supervised-duration → pretrained Vocos decode (MIT).

## Final voices

| Voice | Speaker | Persona | Sentence | DNSMOS | WER |
|---|---|---|---|---|---|
| voice-01.wav | 78 | male grave narrator (~95 Hz) | evalB | **3.323** | 0.125 |
| voice-02.wav | 2416 | female professional (~181 Hz) | evalA | **3.072** | 0.111 |
| voice-03.wav | 696 | female warm-bright (~170 Hz) | evalA | **3.069** | 0.111 |

Each `-eval/` folder holds both frozen-sentence renders for that voice
(broader mini-evaluation; both sentences classified INTELLIGIBLE for all
three finalists).

## Honest quality statement

- BEST CURRENT NATURALNESS SCORE (single clip): **DNSMOS 3.323** (voice-01)
- FROZEN V1 GATE: median ≥3.15 AND min ≥2.85 over the full 20-clip protocol
- **No sample is certified against the V1 gate** — the gate requires the full
  10-sentence × multi-voice protocol; these are 2-sentence showcases.
  Median across showcase clips ≈ 3.07, below the 3.15 release target.

Known weaknesses: residual metallic edge on some phonemes; prosody flatter
than teacher anchors; "Caedral" pronounced as an unknown proper noun in
evalA (WER artifact present in all renders).

Selection rationale: highest measured DNSMOS among 72 sweep candidates
(3 checkpoints × 6 speakers × 2 sentences × 3 speeds), with speaker-diversity
constraint (distinct F0/persona per pick). Sweep evidence:
candidates/sweep-results.jsonl.

Reproduction: see generation-manifest.json (checkpoint sha256 of the
frozen snapshot at copy time, config reference, speaker index, speed, texts,
output hashes).
