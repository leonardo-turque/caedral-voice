# Architecture Notes G1 — measured-silicon constraints

Living doc; updated as phases land. Evidence labels inline.

## Silicon envelope (MEASURED 2026-08-22, benchmarks/results/cpu-calibration-*)

| Quantity | Value | Implication |
|---|---|---|
| Weight-stream bandwidth (GEMV) | ~40–42 GB/s | 80M fp32 params ≈ **7.9 ms/pass** floor |
| Wide parallel GEMM | 87–245 GFLOPS fp32 | NAR mel passes are cheap in FLOPs terms |
| Elementwise scan (numpy) | ~3.9 GB/s | activation transforms matter; fuse/avoid |
| Hybrid CPU caveat | 6P+4E cores | pin threads; expect bimodality (thread sweep) |

## Budget arithmetic for CAEDRAL-NAR candidates [ESTIMATED from MEASURED rates]

| Config | Full weight stream/pass | Passes/audio-second (est.) | Compute RTF floor |
|---|---|---|---|
| 8M Lite/Tiny | ~0.8 ms | ~8 | ~0.007 |
| 30M Balanced | ~3 ms | ~8 | ~0.024 |
| 82M Quality-class | ~8 ms | ~8 | ~0.064 |

Kokoro MEASURED RTF ≈ 0.33 ⇒ real systems sit ~5× above the compute floor
(attention overhead, sampling, frontend, Python glue). Our optimization
headroom is therefore implementation-side as much as parameter-side —
Phase L profile-first discipline applies.

## RAM targets vs baselines (MEASURED)

- Kokoro working set 1.25 GB ⇒ Profile C (1 GB) fails on Kokoro today.
- Caedral Quality target ≤ 700 MB peak; Lite ≤ 350 MB; Tiny ≤ 200 MB
  [HYPOTHESIS to be verified at Phase M].

## Decoder direction (Phase B convergence + Phase C measurements)

Small MIT-licensed iSTFT-head vocoder retrained in-house (Vocos/iSTFTNet2
class); vocoder compression by construction not int8 [REPORTED trap].
Phase G must measure Vocos-class RTF on THIS machine before freeze.
