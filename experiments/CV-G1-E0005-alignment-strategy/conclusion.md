# CV-G1-E0005 — duration/alignment strategy decision for pilot training

Date: 2026-08-22. Owner: coordinator. Status: DECIDED (blocks Phase J design).

## HYPOTHESIS

A license-safe forced-alignment source exists for supervised duration
training of the G1 pilot model without external aligner installs.

## METHOD

Evaluated candidate sources against commercial-use rules:

| Option | License status | Verdict |
|---|---|---|
| torchaudio forced_align + WAV2VEC2_ASR_BASE_960H | repo LICENSE file HTTP 404 [MEASURED 2026-08-22]; no license statement found on HF card | **BLOCKED** — absent license ≠ permission |
| Montreal Forced Aligner | MIT but heavy Kaldi-based install; Windows risk | deferred (fallback) |
| MFA dictionary models | per-model licenses vary | not needed if MAS chosen |
| MAS-learned durations inside training (Glow-TTS/Matcha lineage) | no external artifacts; clean by construction | **CHOSEN** |

## RESULT

Adopt MAS-learned durations. Consequence: the CV-G1-E0002 prototype decoder
(plain transformer → mel regression) must be upgraded to a conditional
normalizing flow over mel (affine-coupling stack) so MAS loss has a
likelihood to maximize. This merges part of H2 (flow machinery) into H1 —
consistent with MASTER-PLAN §10 H4 hybrid hypothesis.

## INTERPRETATION

Supervised-duration training remains possible later ONLY with alignments
from permissively licensed models or consented studio data; revisit at
Phase K/N. The pilot model trains self-contained.

## KEEP / REJECT / INVESTIGATE

KEEP: MAS + flow decoder for G1 pilot.
REJECT: wav2vec2-960h alignment lineage (unverifiable license).
INVESTIGATE: flow-decoder CPU latency impact vs E0002 numbers (must stay
within ~1.3× of RTF budget; measure after implementation at Phase F/J seam).
