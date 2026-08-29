"""E0020 smoke: new normalize() outputs + old-suite spot checks (ASCII-safe)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.text import TextFrontend, normalize  # noqa: E402

CASES = [
    "AVX2", "AVX-512", "CPU", "GPU", "GPT-5", "RTX 5090",
    "Python 3.13", "$29.99", "99.9%", "10GB", "3.5GHz", "3.13",
    "2026-08-23", "caedral.com", "support@caedral.com", "HTTP/2",
    "The CPU supports AVX2 instructions but not AVX-512.",
    "Use CaedralVoice today.",
    "I have 3 apples.", "Version 2.11.3 shipped", "Call 555-0142 now.",
    "between 9:30 AM and 5:00 PM", "by Q3 2026 inflation had cooled",
]

out = {c: normalize(c) for c in CASES}
print(json.dumps(out, indent=1, ensure_ascii=True))

fe = TextFrontend("lexicon")
r = fe.process("Hello caedral. The CPU supports AVX2 instructions but not AVX-512.")
summary = [
    {"tok": w.token, "sel": ascii(w.selected), "src": w.source} for w in r.words
]
print(json.dumps(summary, indent=0, ensure_ascii=True))
