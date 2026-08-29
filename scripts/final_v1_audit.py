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
    "propernoun": "Welcome to Caedral, where every voice tells a story.",
}
voices = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
all_results = {}
for v in voices:
    vdir = REPO / "artifacts" / "caedral-voice-v1" / v
    vrows = {}
    for sname, ref in SENTENCES.items():
        f = vdir / f"{sname}.wav"
        if not f.exists():
            continue
        r = cp.classify_file(str(f), ref)
        vrows[sname] = {"class": r.get("class"), "wer": r.get("min_wer"),
                        "dnsmos": r.get("dnsmos_ovrl")}
        print(f"{v}/{sname}: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")
    all_results[v] = vrows

# longform
lf_path = REPO / "artifacts/caedral-voice-v1/caedral-m1/longform.wav"
if lf_path.exists():
    lf_ref = ("The morning market opened early, as it always did on Saturdays. "
              "Vendors arranged their stalls under striped awnings while the "
              "streetlights were still on, and by seven o'clock the narrow road "
              "smelled of bread, wet stone, and strong coffee.")
    r = cp.classify_file(str(lf_path), lf_ref)
    print(f"M1 longform: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")

json.dump(all_results,
          open(REPO / "artifacts/caedral-voice-v1/final-audit.json", "w"), indent=2)
print("\nDONE")
