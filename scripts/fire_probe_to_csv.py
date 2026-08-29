"""Convert fire-test probe-results JSON(s) into raw-runs.csv rows."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "artifacts" / "fire-test"
SOURCES = [
    ("profile-aplus-1.4gb-1vcpu", BASE / "profile-aplus" / "probe-results.json"),
]

OUT = BASE / "raw-runs.csv"


def main() -> None:
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["profile", "phase", "rep", "wall_s", "audio_s", "rtf",
                     "realtime_multiplier", "cpu_time_s",
                     "nr_throttled_delta", "throttled_usec_delta",
                     "rss_before_mb", "rss_after_mb", "vmhwm_peak_mb",
                     "memory_current_mb", "success"])
        for label, path in SOURCES:
            if not path.exists():
                print(f"skip missing {path}")
                continue
            d = json.loads(path.read_text())
            for r in d.get("reps", []):
                wr.writerow([label, "generation", r["rep"], r["wall_s"],
                             r["audio_s"], r["rtf"], r["realtime_multiplier"],
                             r["cpu_time_s"], r["nr_throttled_delta"],
                             r["throttled_usec_delta"], r["rss_before_mb"],
                             r["rss_after_mb"], r["vmhwm_peak_mb"],
                             r["memory_current_mb"], r["success"]])
            c = d.get("cold", {})
            if c:
                wr.writerow([label, "cold-load", 0, "", "", "", "",
                             "", "", "", "", c.get("rss_after_warmup_mb"),
                             c.get("vmhwm_peak_mb"), "", c.get("load_succeeded")])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
