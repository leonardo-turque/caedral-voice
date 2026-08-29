# Vocos — MIT (decoder dependency, SHIPPED)

Summary of `docs/data/DECODER-LICENSES.md` §1 (LIC-DEC-01, verified 2026-08-22
from primary artifacts only). Summary + sources; see repo doc for verbatim quotes.

## What it covers

- **Code**: `gemelo-ai/vocos` — MIT License, Copyright (c) 2023 Charactr Inc.
- **Weights**: `charactr/vocos-mel-24khz` (13.5M, the G1 reference decoder) —
  MIT via the Hub card license field (`"cardData":{"license":"mit"}`,
  tag `license:mit`, repo sha `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`);
  the field governs the whole repo including `pytorch_model.bin`.
  Sibling checkpoint `charactr/vocos-encodec-24khz` is likewise MIT — but
  pairing it with Meta EnCodec tokens would import CC-BY-NC upstream; not our path.

## Grant and obligation

- Commercial use: **YES** (use/modify/publish/distribute/sell permitted).
- Sole redistribution duty: retain the MIT copyright notice ("Charactr Inc.")
  and the permission notice in copies or substantial portions.

## Release-cut duties

- Pin the exact commit/weights sha256 in provenance and RE-VERIFY the license
  at packaging time (drift precedent documented for BigVGAN/EnCodec).
- Ship the retained notices in `NOTICE.md`.

## Sources

- https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE (retrieved 2026-08-22T18:35:44Z)
- https://huggingface.co/api/models/charactr/vocos-mel-24khz (retrieved 2026-08-22T18:42:04Z)
- Local cache: `research/cache/`; dossier: `docs/data/DECODER-LICENSES.md`
