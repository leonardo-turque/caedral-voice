"""Final V1 audit: classify technical.wav for 4 voices + control/numbers regression."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import classify_preview as cp  # noqa: E402

SENTENCES = {
    "control": "Hello. This is the first generation of Caedral Voice.",
    "technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
}
voices = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
all_results = {}
for v in voices:
    vdir = REPO / "artifacts" / "caedral-voice-v1" / v
    vrows = {}
    for sname in ("control", "technical", "numbers"):
        f = vdir / f"{sname}.wav"
        if not f.exists():
            continue
        r = cp.classify_file(str(f), SENTENCES[sname])
        vrows[sname] = {"class": r.get("class"), "wer": r.get("min_wer"),
                        "dnsmos": r.get("dnsmos_ovrl")}
        print(f"{v}/{sname}: {r.get('class'):16s} WER={r.get('min_wer')} "
              f"DNSMOS={r.get('dnsmos_ovrl')}")
    all_results[v] = vrows

tech_wers = [all_results[v]["technical"]["wer"] for v in voices
             if "technical" in all_results[v]]
print(f"\ntechnical WERs: {tech_wers}")
json.dump(all_results,
          open(REPO / "artifacts/caedral-voice-v1/post-fix-audit.json", "w"),
          indent=2)
