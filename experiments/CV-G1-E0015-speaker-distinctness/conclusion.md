# CV-G1-E0015 — Speaker distinctness of the V1 slate (model-independent)

Task **SPK-DISTINCT-01** · Date 2026-08-22 · Status **COMPLETE**
Evidence label: **[MEASURED]** throughout (Resemblyzer 0.1.4 GE2E d-vector encoder,
256-d, CPU; independent of our TTS model). Scripts: `embed_speakers.py`,
`analyze_thresholds.py`. Data: `embeddings.json` (150 utt embeddings + anchors),
`similarity_matrix.json`, `thresholds.json`, `run_manifest.json`.

## Verdict

**PASS — no collapsed identities on the V1 slate**, with one caveat on how the
>0.80 flag bar must be read.

- 15-way leave-one-out nearest-centroid identification: **83/90 correct (92.2%)**
  [MEASURED]. Every error involves a pair that still separates ≥91.7% in binary
  centroid tests — no pair is confusable in both directions.
- Silhouette (cosine, utterance-level): **+0.249** — positive cluster structure,
  no merged clusters.
- No pair approaches identity level (~0.95+); inter-anchor cosine spans
  **0.566–0.880**, mean 0.729.

## Worst pairs (anchor-vs-anchor cosine)

| Pair | Sim | Binary centroid acc | LOO errors | Read |
|---|---|---|---|---|
| 696 ↔ 2416 | **0.8802** (max) | 100% | 2 | Highest sim yet perfectly separable |
| 6209 ↔ 7505 | 0.8719 | **91.7%** (lowest) | 2 | Behaviorally weakest: narrator vs rank-14 medium-professional |
| 6209 ↔ 78 | 0.8715 | 100% | 0 | Sim high, no behavioral confusion |
| 78 ↔ 8419 | 0.8650 | 100% | 0 | Same |

The >0.80 flag bar caught **25/105 pairs** — but under GE2E geometry 0.80 is
mid-distribution (impostor *mean* anchor sim 0.729), not a collapse line.
Interpretation: the flag list is an **attention list**, and the collapse-relevant
evidence is the verification/identification numbers above, which pass. Watch
6209↔7505 at the Phase K listening gate (both masculine, close register; also
the only pair with mutual LOO leakage).

## Proposed synth-vs-real-anchor V1 gate protocol

Anchor = L2-normalized mean of the 6 stored real reference embeddings per voice
(`embeddings.json:pilot_real_anchor` / `slate.*.mean_embedding`; pilot anchors
stored for all 10 trained indices). Calibrated on gate geometry (utterance vs
anchor): genuine mean **0.904** / min 0.752; impostor max 0.866, p99.9 0.863,
p99 0.837; **EER @ 0.81 (3.5%)**.

Proposal: **PASS ≥ 0.85** (measured FAR 0.32%), **review 0.80–0.85**, **FAIL
< 0.80** (>4.6% of other-speaker clips score here — inside the confusion band).
Caveat: calibrated on real-real geometry; vocoder artifacts may shift synth
scores down slightly, so the first synth run must also measure a real-vs-synth
impostor control and re-calibrate if needed. Pilot-group context: worst pair
103↔1183 @ 0.8923 (smoke-10spk were not spread-selected; not a slate gate item).

## License finding

Resemblyzer is **Apache-2.0, NOT MIT** as briefed — verified from the primary
LICENSE (© 2019 Resemble.AI) + PyPI metadata via the Firecrawl cache-first
policy (rows added to WEB-SOURCE-INDEX). Permissive; SHIP-SAFE with attribution +
NOTICE if ever bundled. Installed `--no-deps`; VAD dependency satisfied by
prebuilt `webrtcvad-wheels 2.0.14` (no MSVC toolchain needed).

## Cost

One model load; 150 utterances embedded in **55.7 s**; peak RSS **431 MB**
(includes torch runtime floor), model unloaded after (`run_manifest.json`).
Well inside the <15 min / RAM budget.
