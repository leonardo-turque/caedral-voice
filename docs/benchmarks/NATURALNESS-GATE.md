# Naturalness Gate — Caedral V1 (GATE-NAT-01)

Status: **MEASURED + FROZEN** (2026-08-22). Converts the MASTER-PLAN V1 bar
"naturalness competitive with medium modern TTS" into numeric thresholds
anchored on baselines measured **on this machine**, before any Caedral V1
audio existed (`"frozen_before_caedral_v1_audio": true` in
`benchmarks/results/natgate/comparison-template.json`).

Machine: i7-13620H (10C/16T), Windows 11. Provenance (machine snapshot, git
commit, per-clip rows) embedded in `benchmarks/results/natgate/*.json`.

## THE GATE (frozen)

Caedral V1 passes the naturalness gate iff, under the protocol below:

```
median(DNSMOS ovrl) >= 3.15   (= anchor median 3.45 - 0.30)
min(DNSMOS ovrl)    >= 2.85   (= anchor median 3.45 - 0.60)
```

Anchor K = median DNSMOS ovrl of the 20 Kokoro anchor clips = **3.45**
(measured here, 2026-08-22). Margins: −0.30 median tolerance (~9% of scale,
roughly one just-noticeable-difference band and within the observed
cross-voice spread); −0.60 min floor tolerates a single weak clip without
failing the system. The min bound references K_median by design so one bad
anchor clip cannot loosen it.

## Protocol (frozen)

1. **Sentences** — exactly these 10 from `eval-corpus-v1.0.json`, one per
   category, all ≥3 words; edge_cases/longform/dialogue excluded (own
   protocols):

   | id | category | text |
   |---|---|---|
   | conv_001 | short_conversational | Hey, did you already eat, or should we grab something on the way? |
   | num_001 | numbers_dates_currency | The invoice totals $1,247.83, due within 30 days. |
   | acr_005 | acronyms_initialisms | The CPU supports AVX2 and AVX-VNNI instructions but not AVX-512. |
   | het_002 | heteronyms | The wind was too strong to wind the clock. |
   | pho_001 | difficult_phonemes | She sells sea shells by the sea shore, and the shells she sells are surely from seashores. |
   | qex_001 | questions_exclamations | Are you seriously telling me this whole thing was your idea? |
   | frm_001 | formal_prose | This report examines the quarterly performance of the portfolio across three asset classes, with particular attention to risk-adjusted returns under volatile market conditions. |
   | nm_001 | names_places | Siobhan met Rajesh and Yuki at the café on Worcester Boulevard. (corpus stores a mojibake byte for "café"; WER normalization strips it — no scoring impact) |
   | tech_001 | technical_programming | The compiler inlines the recursive function, then applies tail-call optimization before emitting intermediate representation. |
   | emo_001 | emotional_expressive | I can't believe you kept this secret from me for three entire years. |

2. **Clips** — a candidate system renders all 10 sentences with **two product
   voices at library defaults** (Kokoro anchor: `af_heart` + `am_michael`,
   speed 1.0 → 20 clips; Piper second anchor: `lessac` → 10 clips). Caedral V1
   will render its two V1 launch voices the same way.
3. **Headroom rule** — any clip whose float peak exceeds 0.98 is peak-scaled
   to 0.98 before writing PCM_16 (Kokoro float output can exceed ±1; Piper
   emits full-scale by design). Recorded per clip as `gain_applied`.
4. **Scoring** — `scripts/classify_preview.py` signals only. Primary:
   DNSMOS ovrl (speechmos, resampled to 16 kHz mono). Informational: dual-ASR
   minWER (faster-whisper small.en + base.en int8) vs corpus references,
   chars/s rate plausibility, repetition. No human ratings in this tier.
5. **RAM discipline** — generation and scoring run as separate processes;
   exactly one ASR/MOS process resident at a time; models unload between
   stages (process exit + gc).
6. **Statistics** — median/min/max over each system's full clip set (pooled
   across voices). Gate uses median and min vs the frozen Kokoro anchor.

## Anchors MEASURED 2026-08-22 (this machine)

| Anchor | n | DNSMOS median | DNSMOS min | DNSMOS max | WER med | WER max | rate plausible |
|---|---|---|---|---|---|---|---|
| Kokoro-82M v1.0 (af_heart+am_michael) | 20 | **3.450** | 3.197 | 3.550 | 0.00 | 0.33 | 20/20 |
| Piper en_US-lessac-medium (isolated subprocess) | 10 | 3.382 | 3.200 | 3.616 | 0.00 | 0.24 | 10/10 |

Full sorted distributions are in `benchmarks/results/natgate/gate-summary.json`
(Kokoro spans 3.197–3.55 across both voices; the two systems agree within
0.07 median, which sanity-checks the scorer across architectures).
WER max 0.33 on both = the known acr_005 dual-ASR hyphen artifact
(see docs/benchmarks/BASELINE-MEASURED.md), not a TTS failure.

Performance context (same runs): Kokoro warm RTF 0.505 median,
707 MB process peak, 337 MB on disk; Piper warm-subprocess RTF 0.303
(cold 0.347), 157 MB child peak RSS, 60 MB on disk.

## Comparison table

Schema lives in `benchmarks/results/natgate/comparison-template.json`
(`{model, dnsmos_median, dnsmos_min, wer_median, rtf, ram_mb, size_mb}` per
row). The final V1 benchmark fills `caedral-v1` (+ `kittentts`) in-place.
Current state:

| Model | DNSMOS med | DNSMOS min | WER med | RTF | RAM MB | Size MB |
|---|---|---|---|---|---|---|
| caedral-v1 | — | — | — | — | — | — |
| kokoro-82M-v1.0-onnx (af_heart+am_michael) | 3.45 | 3.197 | 0.0 | 0.5052 | 707 | 337.4 |
| piper-en_US-lessac-medium (isolated subprocess) | 3.382 | 3.2 | 0.0 | 0.3027 | 157.4 | 60.3 |
| kittentts-nano (pending natgate run) | — | — | — | — | — | — |

## Re-run instructions

```bash
# stage 1+2: synthesis (separate processes; kokoro exits before piper starts)
.venv/Scripts/python.exe benchmarks/run_natgate.py generate --engine kokoro
.venv/Scripts/python.exe benchmarks/run_natgate.py generate --engine piper

# stage 3: scoring (one ASR/MOS process per batch)
.venv/Scripts/python.exe scripts/classify_preview.py \
    --audio benchmarks/results/natgate/kokoro/*.wav \
    --reference-texts benchmarks/results/natgate/references.json \
    --json-out benchmarks/results/natgate/scores-kokoro.json
.venv/Scripts/python.exe scripts/classify_preview.py \
    --audio benchmarks/results/natgate/piper/*.wav \
    --reference-texts benchmarks/results/natgate/references.json \
    --json-out benchmarks/results/natgate/scores-piper.json

# stage 4: anchors + gate + comparison template (+ markdown table)
.venv/Scripts/python.exe benchmarks/run_natgate.py compare
```

Caedral V1 evaluation reuses steps 3–4 with its own clip directory and score
JSON, then checks `median >= 3.15 AND min >= 2.85`.

## Caveats (honest limits)

- DNSMOS v8 is an automatic proxy; on this stack clean speech ceilings around
  3.3–3.6, so absolute MOS claims are out of scope — only same-machine
  relative comparisons are valid.
- NATURAL labels from classify_preview are CANDIDATES; human listening
  confirmation remains required for release sign-off.
- The gate freezes scorer version, corpus sentences, voices-as-defaults and
  machine; changing any of these requires re-measuring the anchor too.
