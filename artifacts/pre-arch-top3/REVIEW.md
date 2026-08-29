# PRE-ARCH TOP-3 — Adversarial Review (TOP3-REVIEW-01)

**Date:** 2026-08-23 · **Reviewer:** independent adversarial reviewer (did not generate these samples)
**Mandate:** try to REJECT. No softening.

**Method.** All 6 unique renders reviewed (main wavs are byte-identical to their eval-folder
counterparts — sha256-verified, and all 3 sha256 match `generation-manifest.json`).
Measurements: numpy/scipy FFT comb analysis, librosa pYIN (60–420 Hz and 40–900 Hz),
comb-filter HNR per 2048-sample frame, energy VAD, sample-diff spike detection,
faster-whisper `base.en` int8 min-WER over temperatures {0.0, 0.2, 0.4} with word timestamps,
resemblyzer speaker embeddings (cosine). Spectrogram/waveform inspection of all files.

**Provenance:** PASS — hashes match manifest; voice-01.wav ≡ voice-01-evalB.wav,
voice-02.wav ≡ voice-02-evalA.wav, voice-03.wav ≡ voice-03-evalA.wav.

## Verdict table (worst across the voice's two renders)

| # | Check | voice-01 (spk78) | voice-02 (spk2416) | voice-03 (spk696) |
|---|---|---|---|---|
| 1 | Metallic ringing / combing | PASS — valley depth 4.2–5.5 dB, p75 7.8; 0.4–0.8 peaks/kHz; HF ratio ≤0.007 | PASS — valley 4.9–7.9 dB, p75 8.9; 1.2–1.4 peaks/kHz | N/A — no voiced frames to average; spectrum is noise-textured |
| 2 | Robotic cadence | PASS — word-dur CV 0.41–0.44; pauses only at punctuation (280–360 ms) | PASS — CV 0.44–0.45; boundary pauses 360–500 ms | PASS — CV 0.31–0.42 |
| 3 | Repetitive rhythm | PASS — durations 0.08–0.52 s, CV 0.412 | PASS — CV 0.442 | PASS — CV 0.312 |
| 4 | Unstable pitch | PASS — jitter 1.15%, 0 octave jumps; but F0 75–77 Hz vs claimed ~95 Hz (−19%); p90 seg-residual 1.17 st | WARN — seg-residual 0.16–0.20 st (abnormally flat); p10 F0 91.5 Hz vs median 174.7 (octave excursions, 2–3 onsets) | **FAIL — pYIN voiced fraction 0.000 on BOTH renders (40–900 Hz); median HNR 3.2–3.6 dB; 16–21% of frames HNR>6 dB (voice-02: 52–55%). No measurable pitch for a "~170 Hz" voice** |
| 5 | Missing/swallowed words | **FAIL — showcase evalB: "Artificial" → "reficial" (WER 0.125)**; evalA clean except "Caedral"→"cadrel" (0.111) | **FAIL — showcase evalB: "Artificial" → "but official" (WER 0.250)**; evalA 0.222 ("kid roll", no content loss) | **FAIL — evalB: "Artificial" → "official", "ar" swallowed (WER 0.125)**; evalA 0.111 ("kidroll") |
| 6 | Silence gaps >400 ms | PASS — max 150 ms | PASS — max 240 ms | PASS — max 160 ms |
| 7 | Stretched syllables | PASS — 4.65–5.32 syl/s | PASS — 4.15–4.70 syl/s | PASS — 4.43–5.00 syl/s |
| 8 | Breath / clipping | PASS — peak 0.542–0.615, 0 clipped, floors −59…−67 dBFS | PASS — peak 0.313, floors −64…−71 dBFS | PASS — peak 0.299, floors −65…−70 dBFS |
| 9 | Clicks / dispersal | PASS — 10 spike hits all zero-crossing-dense HF fricative transients (e.g. 360 ms: −0.092→+0.073 at local RMS 0.084); no isolated discontinuities | PASS — 1 transient hit | PASS — 0–4 transient hits |
| 10 | Weird endings | **FAIL (evalB) — final-voiced F0 rises +3.1 st (uptalk on declarative "…and human."); tail 30 ms**; evalA −1.1 st OK | PASS — end drift −0.1/−0.3 st; decay −31/−33 dB | PASS on energy (−34 dB decay); F0 unmeasurable (see #4) |
| 11 | Identity across renders | PASS — 0.9453 | PASS — 0.9033 | PASS — 0.9485 |

## Systematic defect (all three voices)

The evalB sentence onset **"Artificial" is degraded in 7/7 tested renders** — the three
finalists plus r2/r3 sweep candidates for spk78, spk696, spk2416 at spd1.0 (whisper heard
"reficial" / "but official" / "official" / "the official"). This is a pipeline-level
(frontend/duration) defect at utterance onset, not a per-render fluke. The hero clip of
each voice opens with a damaged first word. Additionally, "Caedral" is never pronounced
recognizably (cadrel/kid roll/kidroll) — known, but present in every evalA render.

## Distinctness of the three identities (resemblyzer, voice-level mean embeddings)

| Pair | Cosine | < 0.85? |
|---|---|---|
| voice-01 vs voice-02 | 0.6619 | PASS |
| voice-01 vs voice-03 | 0.6275 | PASS |
| **voice-02 vs voice-03** | **0.8729** | **FAIL (per-file worst 0.8802, evalB pair)** |

**The trio is NOT genuinely distinct as a set** — the two female voices (spk2416, spk696)
sit at 0.873, above the 0.85 ceiling.

## Verdicts

- **voice-01 (spk78): REJECTED** — worst defect: showcase clip opens with "Artificial"
  rendered as *"reficial"* (min-WER over 3 decode temperatures), plus a +3.1 st rising
  terminal on the declarative evalB ending.
- **voice-02 (spk2416): REJECTED** — worst defect: "Artificial" rendered as *"but official"*
  (WER 0.25 on the showcase sentence); also not sufficiently distinct from voice-03 (0.873).
- **voice-03 (spk696): REJECTED** — worst defect: **zero pYIN-voiced frames on both renders
  (0.0% at 40–900 Hz), median HNR 3.2–3.6 dB** — the excitation is noise-dominated/hoarse;
  a "warm-bright ~170 Hz" persona with no measurable pitch is not a viable voice.

**Overall: none of the three is approvable as best-current preview.** Fix the systematic
sentence-onset defect, re-pick the third voice (spk696 fails at the excitation level), and
re-select for cross-voice distinctness <0.85 before re-freezing a top-3.
