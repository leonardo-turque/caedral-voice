import re
import sys

sys.path.insert(0, "research")
from firecrawl_client import ensure_cached  # noqa: E402

r = ensure_cached("https://www.openslr.org/141")
print(r["status"], r.get("bytes", ""))
txt = open(r["cache"], encoding="utf-8").read()
links = sorted(set(re.findall(r"https?://[^\s\"')]+?\.tar\.gz", txt)))
for link in links[:12]:
    print(" ", link)
