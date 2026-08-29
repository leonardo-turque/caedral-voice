"""V1 FREEZE VERIFICATION (read-only) — verifies the frozen V1 state.

Checks, with zero writes outside artifacts/caedral-voice-v1/freeze-verification.json:
  1. Exactly ONE onnxruntime InferenceSession exists after engine load, and
     engine._kokoro.sess IS that session (kokoro_onnx.from_session reuse).
  2. All 20 founder-pack WAVs exist in artifacts/caedral-voice-v1/<voice>/,
     are valid 24 kHz mono float PCM, and match their listening-pack copies.
  3. The six asset SHA-256 hashes recomputed now match V1-FREEZE-MANIFEST.json
     (frozen assets unchanged since freeze).
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

V1_DIR = REPO / "artifacts" / "caedral-voice-v1"
LISTEN_DIR = REPO / "artifacts" / "pre-arch-enhanced-base" / "final-founder-listen"
VOICES = ["caedral-m1", "caedral-m2", "caedral-f1", "caedral-f2"]
SENTENCES = ["01-normal", "02-technical", "03-numbers",
             "04-proper-names", "05-long-form"]
ASSETS = {
    "kokoro-onnx": "models/kokoro/kokoro-v1.0.onnx",
    "voices-bin": "models/kokoro/voices-v1.0.bin",
    "engine-py": "caedral_voice/inference/kokoro_engine.py",
    "engine-py": "caedral_voice/inference/kokoro_engine.py",
    "kokoro-map-py": "caedral_voice/text/kokoro_map.py",
    "normalize-py": "caedral_voice/text/normalize.py",
    "g2p-py": "caedral_voice/text/g2p.py",
}
THREADS = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    report: dict = {"checks": {}}

    # --- Check 3a: asset hashes BEFORE any model load ------------------------
    manifest = json.loads((V1_DIR / "V1-FREEZE-MANIFEST.json").read_text())
    frozen_hashes = manifest["asset_hashes_sha256"]
    hash_rows = {}
    for key, rel in ASSETS.items():
        actual = sha256(REPO / rel)
        hash_rows[key] = {"expected": frozen_hashes.get(key),
                          "actual": actual, "match": actual == frozen_hashes.get(key)}
    report["checks"]["asset_hashes"] = hash_rows

    # --- Check 2: 20 WAVs + listening-pack parity ----------------------------
    wav_rows = {}
    n_ok = 0
    for vid in VOICES:
        for sname in SENTENCES:
            f = V1_DIR / vid / f"{sname}.wav"
            row: dict = {"exists": f.exists()}
            if f.exists():
                with wave.open(str(f), "rb") as w:
                    row.update({
                        "channels": w.getnchannels(), "sampwidth": w.getsampwidth(),
                        "rate": w.getframerate(),
                        "frames": w.getnframes(),
                        "seconds": round(w.getnframes() / w.getframerate(), 2),
                    })
                lp = LISTEN_DIR / f"{vid}-{sname}.wav"
                if lp.exists():
                    row["listening_pack_copy"] = (
                        sha256(f) == sha256(lp)) if f.stat().st_size <= 4_000_000 \
                        else "skipped-large"
                n_ok += 1
            wav_rows[f"{vid}/{sname}"] = row
    report["checks"]["wavs"] = {
        "count_expected": 20, "count_ok": n_ok, "detail": wav_rows}

    # --- Check 1: single shared ORT session ----------------------------------
    import numpy as np
    import onnxruntime as rt
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine

    def count_sessions() -> int:
        return sum(1 for o in gc.get_objects()
                   if isinstance(o, rt.InferenceSession))

    engine = CaedralVoiceKokoroEngine(
        model_path=str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
        voices_path=str(REPO / "models/kokoro/voices-v1.0.bin"),
        threads=THREADS,
    )
    sessions = [o for o in gc.get_objects()
                if isinstance(o, rt.InferenceSession)]
    wrapper_sess = getattr(engine._kokoro, "sess", None)
    audio_out = [o.name for o in engine._kokoro.sess.get_outputs()]

    _audio = engine.synth("Session check.", voice_id="caedral-m1")
    sessions_post_synth = count_sessions()

    report["checks"]["single_session"] = {
        "inference_sessions_in_process_after_load": len(sessions),
        "wrapper_holds_same_object": len(sessions) == 1
        and wrapper_sess is sessions[0],
        "fused_outputs": audio_out,
        "sessions_after_synth": sessions_post_synth,
    }

    ok_hashes = all(v["match"] for v in hash_rows.values())
    ok_wavs = (n_ok == 20
               and all(r.get("rate") == 24000 and r.get("channels") == 1
                       and isinstance(r.get("listening_pack_copy"), bool)
                       and r["listening_pack_copy"] for r in wav_rows.values()))
    ok_sess = (report["checks"]["single_session"]
               ["inference_sessions_in_process_after_load"] == 1
               and report["checks"]["single_session"]["wrapper_holds_same_object"]
               and sessions_post_synth == 1)

    report["verdict"] = {"asset_hashes": ok_hashes, "wavs_20": ok_wavs,
                         "single_shared_session": ok_sess}

    out = V1_DIR / "freeze-verification.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2)

    print(json.dumps(report["verdict"], indent=2))
    print(f"\nwrote {out}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
