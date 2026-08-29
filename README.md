# CAEDRAL VOICE

Proprietary, **CPU-first**, multi-voice, expressive English text-to-speech.
Generation 1 research & engineering program.

> **STATUS:** Phase A complete (environment baseline), Phase B (state-of-the-art
> research) in progress via parallel research agents. No trained Caedral model exists yet.
> See [`docs/MASTER-PLAN.md`](docs/MASTER-PLAN.md) for the living plan and
> [`docs/checkpoints/`](docs/checkpoints/) for phase-gate records.

## Mission

Build the scientific and engineering foundation for a Caedral-owned speech
synthesis family with these first-class constraints:

- **CPU inference** — no GPU required to serve. Quality is pursued *within* a CPU budget.
- **Multi-voice catalog** — distinct masculine/feminine identities, timbres, personalities.
- **Expressive prosody** — controllable style/rate/pitch/energy/pauses where feasible.
- **100% ownership** — Caedral architecture, training pipeline, runtime, weights;
  documented dataset provenance and dependency licenses; no repackaged third-party models.
- **Scientific honesty** — every number is labeled MEASURED / REPRODUCED FROM SOURCE /
  REPORTED BY THIRD PARTY / ESTIMATED / HYPOTHESIS. Failed experiments are recorded,
  never hidden.

## Repository layout

```
caedral-voice/
├── README.md                  # this file
├── LICENSES/                  # third-party license texts + attribution
├── docs/                      # MASTER-PLAN, checkpoints, research notes
│   ├── research/              # state-of-the-art surveys w/ sources
│   ├── data/                  # dataset survey + license matrix
│   └── checkpoints/           # per-phase gate records
├── research/                  # scratch studies feeding decisions
├── experiments/               # CV-G1-E#### experiment folders
├── configs/                   # reproducible model/training configs
├── data/
│   ├── manifests/             # hashed dataset manifests (in git)
│   └── scripts/               # deterministic preprocessing
├── caedral_voice/             # Python package: text, tokenizer, model,
│   │                          # speaker, prosody, decoder, training,
│   │                          # inference, optimization
├── runtime/                   # deployment-facing runtime (future native)
├── benchmarks/                # frozen eval corpus + harness
├── tests/                     # unit + integration tests
├── scripts/                   # developer utilities
└── artifacts/                 # generated outputs (audio gitignored)
```

## Quickstart

Not yet — nothing is installed or trained. The benchmark corpus and harness land in
Phase C; the first runnable Caedral pipeline lands around Phases E–J.

## Honesty charter

Every benchmark result in this repository carries an evidence label and a source.
We would rather publish *"high quality requires 730 MB"* than fake *"works in 500 MB"*.
See `docs/MASTER-PLAN.md §9` for the full rules.
