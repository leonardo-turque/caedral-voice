import json
from pathlib import Path

rows = [json.loads(l) for l in open("data/manifests/librittsr-smoke-10spk.jsonl", encoding="utf-8")]
durs = [r["duration_s"] for r in rows]
spk = {Path(r["audio_path"]).parts[0] for r in rows}
import statistics
print(f"kept: {len(rows)} | hours: {sum(durs)/3600:.2f} | mean dur: {statistics.mean(durs):.1f}s | speakers: {len(spk)}")
