# artifacts/latest-preview/ — CAEDRAL-GENERATED AUDIO

**FIRST CAEDRAL VOICE AUDIO — generated 2026-08-22.**

## Open this file first

```
C:\Users\leona\Downloads\caedral-voice\artifacts\latest-preview\caedral-preview-earliest.wav
```

## ⚠️ EARLY TRAINING CHECKPOINT — NOT QUALITY REPRESENTATIVE ⚠️

This audio comes from a checkpoint after **60 training steps (~40 s of
compute)**. Expect babble/noise-like output. It exists to prove the full
Caedral pipeline end-to-end and to let the founder track the model's
evolution. It will be replaced by `caedral-preview-current-best.wav`
as measurable improvement lands.

## Provenance (all MEASURED at generation)

| Field | Value |
|---|---|
| Checkpoint | experiments/CV-G1-E0007-pilot-training/smoke-a/checkpoints/step_000000060.pt |
| Architecture | CAEDRAL-NAR tiny, 8.73M params, MAS+flow decoder (CV-G1-E0006) |
| Frontend | Caedral lexicon+g2pen ladder (E0001/E0004), GPL-free |
| Decoder | pretrained Vocos charactr/mel-24khz (MIT) via 80→100 mel-band interpolation (approximation, documented) |
| Speakers | spk0 = LibriTTS-R 103 (feminine-coded), spk1 = 1034; style_id=0 neutral |
| Sample rate | 24 kHz |
| Inference RTF | 0.12–0.17 @4 threads [MEASURED] |
| Hardware | i7-13620H CPU-only, Windows 11 |
| git | see preview-provenance.json |

## Regenerate / refresh

```
.venv\Scripts\python.exe scripts\generate_caedral_preview.py \
  --run-dir experiments\CV-G1-E0007-pilot-training\<run> \
  --checkpoint <run>\checkpoints\<step>.pt --label current-best
```

## Files

- `caedral-preview-earliest.wav` — canonical earliest (copy of spk0-neutral)
- `caedral-preview-earliest-spk{0,1}-{neutral,conversational,expressive}.wav` — full set
- `caedral-preview-current-best.wav` — updated on measurable improvement
- `preview-provenance.json` — machine-readable provenance

Honesty rule: third-party baseline audio NEVER appears in this directory.
