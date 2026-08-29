# data/lexicons — provenance and licensing

| File | What | Source | License | Verified |
|---|---|---|---|---|
| `cmudict-0.7b` | CMUdict 0.7b, 134,373 entries incl. variant pronunciations `WORD(N)` | `https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b` (mirror of cmusphinx svn r13083, 2015-07-15) | BSD-style permissive, (C) 1993-2015 Carnegie Mellon University; header in-file states redistribution conditions; CMU Sphinx distribution describes use "for any research or commercial purpose is completely unrestricted" (see docs/research/TEXT-FRONTEND.md §3.1) | 2026-08-22: header read from this copy |
| `cmudict-LICENSE` | The CMU Sphinx repo LICENSE text kept for attribution-on-redistribution duty | `https://raw.githubusercontent.com/cmusphinx/cmudict/master/LICENSE` | same as above | 2026-08-22 |

Note: the reduced `cmudict.dict` mirror (cmusphinx GitHub) was evaluated and
REJECTED — it collapses alternate pronunciations, which kills heteronym
handling. Keep using full `cmudict-0.7b`.

Redistribution rule: if Caedral ever ships this lexicon file, ship
`cmudict-LICENSE` alongside it.
