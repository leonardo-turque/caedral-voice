"""V1 ASR MISMATCH REPORT (read-only) — exact word-level diffs for the frozen
founder-pack WAVs.

Targets the known open items:
  * caedral-m2 / caedral-f1 / caedral-f2 : 02-technical.wav
  * caedral-f1                            : 04-proper-names.wav
plus caedral-m1 02-technical as an INTELLIGIBLE control for contrast.

For each clip: raw transcripts from BOTH audit ASRs (faster-whisper small.en /
base.en int8), then a word-level alignment against the reference text so the
exact substitutions/deletions/insertions are visible.

Output: artifacts/caedral-voice-v1/asr-mismatch-report.json
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

V1_DIR = REPO / "artifacts" / "caedral-voice-v1"

TECH = "The CPU supports AVX2 instructions but not AVX-512."
NAMES = "Welcome to Caedral, where every voice tells a story."
TARGETS = [
    ("caedral-m1", "02-technical", TECH, "control-pass"),
    ("caedral-m2", "02-technical", TECH, "open-item"),
    ("caedral-f1", "02-technical", TECH, "open-item"),
    ("caedral-f2", "02-technical", TECH, "open-item"),
    ("caedral-f1", "04-proper-names", NAMES, "open-item"),
]

TAG = re.compile(r"[a-z0-9']+")


def norm(text: str) -> list[str]:
    return TAG.findall(text.lower())


def align(ref: list[str], hyp: list[str]) -> list[tuple]:
    """Word-level ops via difflib: ('equal'|'sub'|'del'|'ins', ref_words, hyp_words)."""
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(("ok", ref[i1:i2], hyp[j1:j2]))
        elif tag == "replace":
            ops.append(("SUB", ref[i1:i2], hyp[j1:j2]))
        elif tag == "delete":
            ops.append(("DEL", ref[i1:i2], []))
        elif tag == "insert":
            ops.append(("INS", [], hyp[j1:j2]))
    return ops


def main() -> int:
    import scripts.classify_preview as cp

    models = cp.load_models()
    rows = []
    for vid, sname, text, kind in TARGETS:
        voice_dir = f"caedral-{vid}" if not vid.startswith("caedral") else vid
        path = V1_DIR / voice_dir / f"{sname}.wav"
        hyps = {m: cp.transcribe(path, mod) for m, mod in models.items()}
        ref_w = norm(text)
        per_asr = {}
        best_name, best_ops, best_w = None, None, 1.1
        for mname, txt in sorted(hyps.items()):
            hyp_w = norm(txt)
            wers = cp.word_error_rate(ref_w, hyp_w)
            ops = align(ref_w, hyp_w)
            per_asr[mname] = {
                "transcript": txt, "wer": round(wers, 4),
                "ops": [{"op": o[0], "ref": o[1], "hyp": o[2]} for o in ops],
            }
            if wers < best_w:
                best_name, best_ops, best_w = mname, ops, wers

        # classify verdict on the better transcript's WER (min-WER convention)
        min_wer = min(p["wer"] for p in per_asr.values())
        cls = ("INTELLIGIBLE" if min_wer <= 0.30 else
               "PARTIAL_SPEECH" if min_wer <= 0.60 else "BABBLE")
        rows.append({
            "voice": voice_dir, "sentence": sname, "role": kind,
            "reference": text,
            "asr": per_asr,
            "best_asr": best_name,
            "min_wer": round(min_wer, 4),
            "class_at_freeze": cls,
        })
        print(f"\n=== {voice_dir} / {sname} ({kind}) — minWER={min_wer:.3f} ===")
        print(f"  REF: {text}")
        for mname in sorted(hyps):
            print(f"  {mname}: {hyps[mname]}")
        for o in best_ops:
            if o[0] != "ok":
                print(f"    [{o[0]}] ref={o[1]} -> hyp={o[2]}")

    out = V1_DIR / "asr-mismatch-report.json"
    payload = {
        "report": "v1-asr-mismatch",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "asr_pair": ["faster-whisper small.en int8", "faster-whisper base.en int8"],
        "note": ("Read-only audit of FROZEN wavs; classes recomputed from minWER "
                 "band only (no DNSMOS re-run); freeze-audit classes remain "
                 "authoritative."),
        "rows": rows,
    }
    json.dump(payload, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
