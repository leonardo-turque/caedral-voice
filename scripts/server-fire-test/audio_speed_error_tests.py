#!/usr/bin/env python3
"""BASIC AUDIO (4 voices) + SPEED SWEEP + ERROR CONTRACT probes.

- Section A: 'Welcome to Caedral.' for all four voices -> WAV saved,
  RIFF/24000/mono/PCM16 verified, wall/RTF recorded.
- Section B: canonical short sentence on caedral-f1 at speeds inside the
  code-supported range [0.75, 1.0, 1.25]; output-duration ratio checked.
- Section C: deterministic error contract probes against the live service.

stdlib-only. Writes audio + JSON results into --out.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def call(url, path, body=None, secret="", timeout=420, raw=None):
    req = urllib.request.Request(
        url + path, method="POST" if (body is not None or raw is not None)
        else "GET")
    if secret:
        req.add_header("X-Internal-Secret", secret)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    elif raw is not None:
        data = raw
    else:
        data = None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return {"status": r.status, "headers": dict(r.headers),
                    "payload": r.read(), "wall": round(time.time() - t0, 3)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "headers": dict(e.headers),
                "payload": e.read(), "wall": round(time.time() - t0, 3)}
    except Exception as exc:  # noqa: BLE001
        return {"status": f"ERR:{type(exc).__name__}", "headers": {},
                "payload": b"", "wall": round(time.time() - t0, 3),
                "err": str(exc)}


def wav_props(payload: bytes):
    try:
        with wave.open(io.BytesIO(payload), "rb") as w:
            rate, ch, sw, frames = (w.getframerate(), w.getnchannels(),
                                    w.getsampwidth(), w.getnframes())
        return {"rate": rate, "channels": ch, "sampwidth": sw,
                "seconds": round(frames / rate, 3),
                "riff_ok": payload[:4] == b"RIFF"
                           and payload[8:12] == b"WAVE"}
    except Exception as exc:  # noqa: BLE001
        return {"parse_error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--secret", default="")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"voices": [], "speed_sweep": [], "errors": []}

    # --- A. four voices -------------------------------------------------------
    for voice in ("caedral-f1", "caedral-f2", "caedral-m1", "caedral-m2"):
        r = call(args.url, "/v1/audio/speech",
                 body={"model": "caedral-voice-1", "voice": voice,
                       "input": "Welcome to Caedral.", "speed": 1.0,
                       "response_format": "wav"},
                 secret=args.secret)
        props = wav_props(r["payload"]) if r["status"] == 200 else None
        fn = out_dir / f"welcome-{voice}.wav"
        fn.write_bytes(r["payload"])
        hdr = {k.lower(): v for k, v in r["headers"].items()}
        entry = {
            "voice": voice, "status": r["status"], "file": fn.name,
            "bytes": len(r["payload"]), "props": props,
            "gen_wall_s": hdr.get("x-caedral-wall-seconds"),
            "rtf_hdr": hdr.get("x-caedral-rtf"),
            "client_wall_s": r["wall"],
        }
        report["voices"].append(entry)
        print("VOICE " + json.dumps(entry), flush=True)

    # --- B. speed sweep (code range [0.25, 2.0]) ------------------------------
    base_text = "The harbour lights trembled softly on the evening swells."
    durs = {}
    for speed in (0.75, 1.0, 1.25):
        r = call(args.url, "/v1/audio/speech",
                 body={"model": "caedral-voice-1", "voice": "caedral-f1",
                       "input": base_text, "speed": speed,
                       "response_format": "wav"},
                 secret=args.secret)
        props = wav_props(r["payload"]) if r["status"] == 200 else None
        durs[speed] = props.get("seconds") if props else None
        entry = {"speed": speed, "status": r["status"], "props": props,
                 "gen_wall_s": {k.lower(): v for k, v in
                                r["headers"].items()
                                }.get("x-caedral-wall-seconds"),
                 "rtf_hdr": {k.lower(): v for k, v in
                             r["headers"].items()}.get("x-caedral-rtf")}
        report["speed_sweep"].append(entry)
        print("SPEED " + json.dumps(entry), flush=True)

    ok_durs = all(v and durs[0.75] and durs[1.0] and durs[1.25] for v in
                  (1,))
    if ok_durs and durs[1.0]:
        report["speed_ratio_slow_over_default"] = round(
            durs[0.75] / max(durs[1.0], 1e-9), 3)
        report["speed_ratio_fast_over_default"] = round(
            durs[1.25] / max(durs[1.0], 1e-9), 3)

    # --- C. error contract probes --------------------------------------------
    good_voice_body = {"model": "caedral-voice-1", "voice": "caedral-f1",
                       "input": "Hello.", "speed": 1.0,
                       "response_format": "wav"}

    def probe(name, body=None, raw=None, with_secret=True):
        r = call(args.url, "/v1/audio/speech", body=body, raw=raw,
                 secret=args.secret if with_secret else "")
        code = ""
        try:
            code = json.loads(r["payload"])["error"]["code"]
        except Exception:
            pass
        entry = {"probe": name, "status": r["status"], "code": code}
        report["errors"].append(entry)
        print("ERRPROBE " + json.dumps(entry), flush=True)

    b = dict(good_voice_body); b["model"] = "gpt-x"; probe("invalid_model", b)
    b = dict(good_voice_body); b["voice"] = "nope"; probe("invalid_voice", b)
    b = dict(good_voice_body); b["input"] = "   "; probe("empty_input", b)
    b = dict(good_voice_body); b["input"] = "word " * 600
    probe("input_too_large", b)
    b = dict(good_voice_body); b["speed"] = 9.9; probe("invalid_speed", b)
    b = dict(good_voice_body); b["response_format"] = "mp3"
    probe("invalid_format", b)
    b = dict(good_voice_body); b["input"] = "Привет, как дела?"
    probe("unsupported_language", b)
    probe("malformed_json", raw=b"{not json")
    probe("internal_auth_failed", body=good_voice_body, with_secret=False)

    (out_dir / "audio-speed-error-results.json").write_text(
        json.dumps(report, indent=2))
    print("REPORT_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
