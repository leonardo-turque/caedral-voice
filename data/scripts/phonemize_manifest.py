"""Phonemize a dataset manifest into training-ready rows.

Uses the shippable frontend ladder (lexicon -> morphology -> g2p_en ->
letter rules; GPL espeak NEVER touched). Deterministic given manifest order.

Input : data/manifests/<name>.jsonl  (from preprocess.py)
Output: data/manifests/<name>.phonemes.jsonl
        {id, transcript_norm, phoneme_string, n_phones, g2pen_used}
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.text import TextFrontend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="manifest name under data/manifests/")
    args = ap.parse_args()

    src = REPO / "data" / "manifests" / f"{args.name}.jsonl"
    dst = src.with_suffix(".phonemes.jsonl")
    rows = [json.loads(l) for l in io.open(src, encoding="utf-8")]
    print(f"phonemizing {len(rows)} utterances from {src.name}")

    tf = TextFrontend("lexicon+g2pen")
    t0 = time.perf_counter()
    sources: Counter = Counter()
    with io.open(dst, "w", encoding="utf-8") as out:
        for row in rows:
            res = tf.process(row["transcript"])
            used_g2pen = any(w.source == "g2pen" for w in res.words)
            sources.update(w.source for w in res.words)
            out.write(json.dumps({
                "id": row["id"],
                "transcript_norm": row["transcript"],
                "phoneme_string": res.phoneme_string,
                "n_phones": res.phoneme_count,
                "g2pen_used": used_g2pen,
            }) + "\n")

    dt = time.perf_counter() - t0
    total_words = sum(sources.values())
    print(f"wrote {dst.name} in {dt:.1f}s "
          f"({len(rows)/dt:.0f} utt/s)")
    print("pronunciation sources:", dict(sources),
          f"| g2pen share: {sources['g2pen']/max(1,total_words):.2%}")


if __name__ == "__main__":
    main()
