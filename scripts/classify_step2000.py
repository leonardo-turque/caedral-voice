import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import classify_preview as cp  # noqa: E402

refs = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}
rows = []
for f in sorted(Path("artifacts/rescue-comparison/step-2000").glob("*.wav")):
    parts = f.stem.split("-")
    who, spk, sname = parts[0], parts[1], parts[2]
    r = cp.classify_file(str(f), refs[sname])
    rows.append({"file": f.name, "who": who, "speaker": spk, "sentence": sname,
                 "class": r.get("class"), "wer": r.get("min_wer"),
                 "dnsmos": r.get("dnsmos_ovrl")})
    print(f"{who:9s} {spk} {sname}: {r.get('class'):16s} "
          f"WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")
json.dump(rows, open("artifacts/rescue-comparison/step-2000/classification.json",
                     "w"), indent=2)
