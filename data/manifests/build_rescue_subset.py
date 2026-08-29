"""Build CHALLENGER-A subset manifest: 3 strongest speakers from clean100."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEEP_SPEAKERS = {"78", "2416", "6415"}
src = REPO / "data" / "manifests" / "librittsr-clean100.jsonl"
pho = REPO / "data" / "manifests" / "librittsr-clean100.phonemes.jsonl"
out_m = REPO / "data" / "manifests" / "rescue-subset.jsonl"
out_p = REPO / "data" / "manifests" / "rescue-subset.phonemes.jsonl"

kept_ids = set()
with io.open(src, encoding="utf-8") as fin, \
        io.open(out_m, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        if row["id"].split("_")[0] in KEEP_SPEAKERS:
            fout.write(line)
            kept_ids.add(row["id"])

n_pho = 0
with io.open(pho, encoding="utf-8") as fin, \
        io.open(out_p, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        if row["id"] in kept_ids:
            fout.write(line)
            n_pho += 1
print(f"subset rows: {len(kept_ids)} manifest / {n_pho} phonemes")
