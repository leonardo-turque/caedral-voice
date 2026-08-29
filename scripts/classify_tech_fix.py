import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import classify_preview as cp  # noqa: E402

text = "The CPU supports AVX2 instructions but not AVX-512."
f = REPO / "artifacts" / "caedral-voice-v1" / "caedral-m1" / "technical.wav"
r = cp.classify_file(str(f), text)
result = {"class": r.get("class"), "wer": r.get("min_wer"),
          "dnsmos": r.get("dnsmos_ovrl")}
print(json.dumps(result))
json.dump(result, open(REPO / "artifacts/rescue-state/tech-fix-result.json", "w"),
          indent=2)
