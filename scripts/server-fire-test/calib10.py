#!/usr/bin/env python3
"""Calibrate a candidate text against a target audio duration (f1 @ 1.0)."""
import json
import sys
import urllib.request

secret = sys.argv[1]
text = sys.argv[2]
body = json.dumps({"model": "caedral-voice-1", "voice": "caedral-f1",
                   "input": text, "speed": 1.0,
                   "response_format": "wav"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8600/v1/audio/speech", data=body,
    headers={"Content-Type": "application/json",
             "X-Internal-Secret": secret}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    h = {k.lower(): v for k, v in r.headers.items()}
    print("chars=%d audio_s=%s wall=%s rtf=%s" % (
        len(text), h.get("x-caedral-audio-seconds"),
        h.get("x-caedral-wall-seconds"), h.get("x-caedral-rtf")))
