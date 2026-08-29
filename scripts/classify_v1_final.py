"""Classify all V1 technical renders + produce final benchmark."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import classify_preview as cp  # noqa: E402

TECH_TEXT = "The CPU supports AVX2 instructions but not AVX-512."
voices = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
results = {}

for v in voices:
    f = REPO / "artifacts" / "caedral-voice-v1" / v / "technical.wav"
    if not f.exists():
        continue
    r = cp.classify_file(str(f), TECH_TEXT)
    results[v] = {"class": r.get("class"), "wer": r.get("min_wer"),
                  "dnsmos": r.get("dnsmos_ovrl")}
    print(f"{v}: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")

# also classify all other sentences for complete benchmark
for v in voices:
    vd = REPO / "artifacts" / "caedral-voice-v1" / v
    all_rows = {}
    for wf in sorted(vd.glob("*.wav")):
        sname = wf.stem
        if sname == "technical":
            continue
        # find reference text from render script SET dict (hardcode refs)
        refs = {
            "control": "Hello. This is the first generation of Caedral Voice.",
            "numbers": "The invoice totals $1,247.83, due within 30 days.",
            "propernoun": "Welcome to Caedral, where every voice tells a story.",
        }
        if sname == "longform":
            ref = ("The morning market opened early, as it always did on Saturdays. "
                   "Vendors arranged their stalls under striped awnings while the "
                   "streetlights were still on, and by seven o'clock the narrow road "
                   "smelled of bread, wet stone, and strong coffee.")
        else:
            ref = refs.get(sname, "")
        if not ref:
            continue
        try:
            r = cp.classify_file(str(wf), ref)
            all_rows[sname] = {"class": r.get("class"), "wer": r.get("min_wer"),
                               "dnsmos": r.get("dnsmos_ovrl")}
        except Exception:
            pass

    results[v] = {"technical": results.get(v, {}), "all": all_rows}
    print(f"{v} full: {json.dumps(all_rows)}")

json.dump(results, open(REPO / "artifacts/caedral-voice-v1/technical-audit.json", "w"),
          indent=2)
print("\nDONE")
