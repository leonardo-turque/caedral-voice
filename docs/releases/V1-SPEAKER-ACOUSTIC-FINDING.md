# V1 FINDING — TECHNICAL SENTENCE SPEAKER-ACOUSTIC VARIATION

**Date:** 2026-08-23 · **Evidence type:** MEASURED

## Finding

The Caedral text frontend produces **byte-for-byte identical phoneme strings**
across all 4 production voices for the same input text. Despite identical
phoneme input, the Kokoro acoustic model renders technical sentences
(containing letter-spelled acronyms like "C P U" and "A V X") with varying
clarity depending on the voice embedding.

## Evidence

```
caedral-m1: ðə sˈiː pˈiː jˈuː səpˈɔːrts ... [identical]
caedral-m2: ðə sˈiː pˈiː jˈuː səpˈɔːrts ... [identical]
caedral-f1: ðə sˈiː pˈiː jˈuː səpˈɔːrts ... [identical]
caedral-f2: ðə sˈiː pˈiː jˈuː səpˈɔːrts ... [identical]
All phoneme strings identical: True
```

## Result

| Voice | Technical WER | Class |
|---|---|---|
| caedral-m1 (am_michael) | 0.222 | INTELLIGIBLE ✅ |
| caedral-m2 (am_echo) | 0.444 | PARTIAL_SPEECH ⚠️ |
| caedral-f1 (af_sky) | 0.444 | PARTIAL_SPEECH ⚠️ |
| caedral-f2 (af_nova) | 0.444 | PARTIAL_SPEECH ⚠️ |

## Conclusion

This is genuinely **speaker-acoustic behavior** in the Kokoro model — not a
frontend/pipeline bug. The am_michael voice embedding produces clearer
articulation of letter-spelled acronyms than am_echo/af_sky/af_nova.

## Impact for V1

Technical sentences remain PARTIAL_SPEECH for 3/4 voices.
All other sentence types (control, numbers, proper nouns, long-form) are
INTELLIGIBLE or NATURAL for all 4 voices.

## Future fix path (V2 scope)

Fine-tune or adapt the acoustic model on technical/acronym-heavy text so
all voices handle letter-spelled sequences clearly. This requires training
data with technical vocabulary and cannot be fixed through the shared
frontend pipeline.
