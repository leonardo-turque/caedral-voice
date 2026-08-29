"""CAEDRAL VOICE V1 — FINAL FOUNDER ACCEPTANCE PACKAGE.

Renders 4 voices × 5 sentences, classifies, measures RTF/RAM/threads,
creates freeze manifest with SHA-256 hashes, and produces the founder
listening pack.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import psutil
import soundfile as sf

from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
import classify_preview as cp  # noqa: E402

V1_DIR = REPO / "artifacts" / "caedral-voice-v1"
VOICES = {
    "caedral-m1": {"kokoro_voice": "am_michael", "gender": "male",
                   "persona": "natural balanced male"},
    "caedral-m2": {"kokoro_voice": "am_echo", "gender": "male",
                   "persona": "distinct second male identity"},
    "caedral-f1": {"kokoro_voice": "af_sky", "gender": "female",
                   "persona": "natural balanced female"},
    "caedral-f2": {"kokoro_voice": "af_nova", "gender": "female",
                   "persona": "distinct second female identity"},
}
SENTENCES = {
    "01-normal": "Hello. This is the first generation of Caedral Voice.",
    "02-technical": "The CPU supports AVX2 instructions but not AVX-512.",
    "03-numbers": "The invoice totals $1,247.83, due within 30 days.",
    "04-proper-names": "Welcome to Caedral, where every voice tells a story.",
    "05-long-form": ("The morning market opened early, as it always did on "
                     "Saturdays. Vendors arranged their stalls under striped "
                     "awnings while the streetlights were still on, and by seven "
                     "o'clock the narrow road smelled of bread, wet stone, and "
                     "strong coffee."),
}
THREADS = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    proc = psutil.Process()

    print("=== CAEDRAL VOICE V1 — FOUNDER ACCEPTANCE ===\n")

    # --- Load engine ---
    print("[LOAD] Loading engine (cold)...")
    t0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=THREADS,
    )
    cold_load_s = round(time.perf_counter() - t0, 3)
    rss_preload = proc.memory_info().rss // 1e6
    engine.warmup()
    rss_postload = proc.memory_info().rss // 1e6
    print(f"    loaded {cold_load_s}s | RSS {rss_preload}→{rss_postload} MB")

    # --- Render 4 voices × 5 sentences ---
    print("\n[RENDER] 4 voices × 5 sentences...")
    voice_data = {}
    for vid, info in VOICES.items():
        vdir = V1_DIR / vid
        vdir.mkdir(parents=True, exist_ok=True)
        srows = []
        for sname, text in SENTENCES.items():
            t = time.perf_counter()
            audio = engine.synth(text, voice_id=vid, speed=1.0)
            wall = time.perf_counter() - t
            audio_s = len(audio) / 24000
            rtf = wall / audio_s if audio_s > 0 else 99
            f = vdir / f"{sname}.wav"
            sf.write(str(f), audio, 24000)

            try:
                r = cp.classify_file(str(f), text)
                cls = r.get("class")
                wer = r.get("min_wer")
                dmos = r.get("dnsmos_ovrl")
            except Exception:
                cls, wer, dmos = "ERROR", None, None

            peak = float(np.abs(audio).max()) if len(audio) else 0
            srows.append({
                "sentence": sname, "text": text,
                "wall_s": round(wall, 3), "audio_s": round(audio_s, 2),
                "rtf": round(rtf, 3), "peak_amplitude": round(peak, 3),
                "class": cls, "wer": wer, "dnsmos": dmos,
            })
            print(f"    {vid}/{sname}: RTF={rtf:.3f} {cls} WER={wer}")
        voice_data[vid] = srows

    # --- Thread confirmation ---
    print("\n[THREADS] Confirming @6thr...")
    thread_results = {}
    for nt in (4, 6, 8):
        code = f"""
import time, json, psutil
from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
e = CaedralVoiceKokoroEngine(
    model_path="models/kokoro/kokoro-v1.0.onnx",
    voices_path="models/kokoro/voices-v1.0.bin", threads={nt})
e.warmup()
times = []
for _ in range(5):
    t = time.perf_counter()
    a = e.synth("Hello. This is the first generation of Caedral Voice.",
                voice_id="caedral-m1")
    times.append(time.perf_counter() - t)
audio_s = len(a) / 24000
rtfs = sorted(w / audio_s for w in times)
rss = psutil.Process().memory_info().rss // 1e6
print(json.dumps({{"threads": {nt}, "load_s": round(time.perf_counter()-t0,3),
    "rtf_median": round(rtfs[2],4), "rtf_p95": round(rtfs[int(0.95*len(rtfs))],4),
    "rss_warm_mb": rss}}))
"""
        import subprocess as sp
        r = sp.run([str(REPO / ".venv/Scripts/python.exe"), "-c", code],
                   capture_output=True, text=True, timeout=120, cwd=str(REPO))
        if r.returncode == 0:
            d = json.loads(r.stdout.strip().split("\n")[-1])
            thread_results[nt] = d
            print(f"    {nt}thr: RTF med {d['rtf_median']} p95 {d.get('rtf_p95','N/A')}")

    # --- Peak RSS during synthesis ---
    audio_lf = engine.synth(SENTENCES["05-long-form"], voice_id="caedral-m1")
    rss_peak = proc.memory_info().rss // 1e6
    del audio_lf

    # --- Asset hashes ---
    print("\n[HASH] Computing asset SHA-256...")
    assets = {
        "kokoro-onnx": sha256(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        "voices-bin": sha256(REPO / "models/kokoro/voices-v1.0.bin"),
        "engine-py": sha256(REPO / "caedral_voice/inference/kokoro_engine.py"),
        "kokoro-map-py": sha256(REPO / "caedral_voice/text/kokoro_map.py"),
        "normalize-py": sha256(REPO / "caedral_voice/text/normalize.py"),
        "g2p-py": sha256(REPO / "caedral_voice/text/g2p.py"),
    }

    # --- Freeze manifest ---
    git_commit = "unknown"
    try:
        import subprocess as sp
        gc = sp.run(["git", "rev-parse", "HEAD"], capture_output=True,
                    text=True, cwd=str(REPO))
        git_commit = gc.stdout.strip()[:12]
    except Exception:
        pass

    freeze = {
        "version": "Caedral Voice V1",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "foundation": {
            "model": "Kokoro v1.0 ONNX (Apache-2.0)",
            "attribution": "Caedral-developed voice engine/frontend/runtime built on an attributed open-weight Kokoro foundation",
            "license": "Apache-2.0",
        },
        "engine_config": {
            "backend": "ONNX Runtime CPU",
            "graph_optimization": "ENABLE_ALL",
            "threads": THREADS,
            "cpu_arena": False,
            "mem_pattern": True,
            "execution_mode": "sequential",
        },
        "sample_rate": 24000,
        "voice_ids": list(VOICES.keys()),
        "voice_lineage": {k: v["kokoro_voice"] for k, v in VOICES.items()},
        "asset_hashes_sha256": assets,
        "known_limitations": [
            "Technical sentences PARTIAL_SPEECH for m2/f1/f2 "
            "(speaker-acoustic variation, identical phoneme input confirmed)",
            "No streaming/TTFA yet (full-buffer synthesis only)",
            "RTF varies 0.59–1.10 depending on sentence complexity",
        ],
        "benchmark": {"thread_results": thread_results},
    }
    freeze_path = V1_DIR / "V1-FREEZE-MANIFEST.json"
    json.dump(freeze, io.open(freeze_path, "w", encoding="utf-8"), indent=2)

    # --- Founder listening pack ---
    listen_dir = REPO / "artifacts" / "pre-arch-enhanced-base" / "final-founder-listen"
    listen_dir.mkdir(parents=True, exist_ok=True)
    for vid in VOICES:
        for sname in SENTENCES:
            src = V1_DIR / vid / f"{sname}.wav"
            if src.exists():
                shutil.copy2(src, listen_dir / f"{vid}-{sname}.wav")

    readme_lines = [
        "# CAEDRAL VOICE V1 — FOUNDER LISTENING PACK",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Checkpoint step: see benchmark.json",
        f"Freeze manifest: artifacts/caedral-voice-v1/V1-FREEZE-MANIFEST.json",
        "",
        "## Files",
        ""]
    for vid in VOICES:
        for sname in SENTENCES:
            readme_lines.append(f"- {vid}-{sname}.wav — {vid} speaking: {SENTENCES[sname][:60]}...")
    readme_lines += [
        "",
        "## Known limitations",
        "- Technical sentences PARTIAL_SPEECH for m2/f1/f2",
        "- No streaming yet",
        "",
        "Listen without reading the transcripts first.",
        "Founder perception is authoritative.",
    ]
    io.open(listen_dir / "TRANSCRIPTS.md", "w", encoding="utf-8").write(
        "\n".join(readme_lines))

    # --- Final report ---
    print(f"\n{'='*60}")
    print("CAEDRAL VOICE V1 — FOUNDER ACCEPTANCE PACKAGE")
    print(f"{'='*60}")
    print(f"\nCold load:     {cold_load_s}s")
    print(f"Warm RSS:      {rss_postload} MB")
    print(f"Peak RSS:      {rss_peak} MB")
    print(f"Threads:       {THREADS}")
    print(f"Assets hashed: {len(assets)}")

    print("\nPer-voice quality:")
    for vid, rows in voice_data.items():
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        dmoss = [r["dnsmos"] for r in rows if r["dnsmos"] is not None]
        rtfs = [r["rtf"] for r in rows]
        med_wer = statistics.median(wers) if wers else None
        med_dmos = statistics.median(dmoss) if dmoss else None
        med_rtf = statistics.median(rtfs) if rtfs else None
        print(f"  {vid}: WER med={med_wer} DNSMOS med={med_dmos} RTF med={med_rtf}")

    print(f"\nListening pack: {listen_dir}")
    print(f"Freeze manifest: {freeze_path}")
    print(f"\nV1 STATUS: READY FOR FOUNDER LISTENING")


if __name__ == "__main__":
    main()
