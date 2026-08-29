"""Production smoke test for the Caedral Voice V1 internal service.

Usage: python smoke_test.py [--url http://127.0.0.1:8600] [--secret SECRET]

Checks health/readiness, all four voices (valid RIFF/WAVE, 24 kHz mono,
non-empty), the deterministic error contract, unsupported-language rejection,
and overload behavior (burst > queue capacity must yield clean 429s and a
surviving process). Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

SHORT = "Hello from Caedral Voice."


def call(url: str, path: str, body: dict | None = None,
         secret: str = "", raw: bytes | None = None, timeout: float = 120):
    req = urllib.request.Request(url + path, method="POST" if body is not None
                                 or raw is not None else "GET")
    if secret:
        req.add_header("X-Internal-Secret", secret)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    elif raw is not None:
        data = raw
    else:
        data = None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def wav_props(payload: bytes) -> tuple[int, int, float] | None:
    """(rate, channels, seconds) parsed via stdlib wave."""
    try:
        with wave.open(io.BytesIO(payload), "rb") as w:
            rate, ch, frames = (w.getframerate(), w.getnchannels(),
                                w.getnframes())
        return rate, ch, frames / rate
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--secret", default="")
    args = ap.parse_args()
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # --- liveness / readiness -------------------------------------------------
    st, _, _ = call(args.url, "/healthz", secret=args.secret)
    check("healthz", st == 200, f"status={st}")
    for _ in range(60):
        st, _, _ = call(args.url, "/readyz", secret=args.secret)
        if st == 200:
            break
        time.sleep(2)
    check("readyz", st == 200, f"status={st}")

    # --- four voices ----------------------------------------------------------
    for voice in ("caedral-f1", "caedral-f2", "caedral-m1", "caedral-m2"):
        st, hdrs, payload = call(
            args.url, "/v1/audio/speech",
            body={"model": "caedral-voice-1", "voice": voice, "input": SHORT,
                  "speed": 1.0, "response_format": "wav"},
            secret=args.secret, timeout=300)
        props = wav_props(payload) if st == 200 else None
        ok = (st == 200 and payload[:4] == b"RIFF"
              and props is not None and props[0] == 24000
              and props[1] == 1 and props[2] > 0.5)
        check(f"voice {voice}", ok,
              f"status={st} rate/ch/dur={props}")

    # --- error contract -------------------------------------------------------
    cases = [
        ("invalid-voice",
         {"model": "caedral-voice-1", "voice": "nope", "input": SHORT}, 400),
        ("invalid-model",
         {"model": "gpt-x", "voice": "caedral-f1", "input": SHORT}, 400),
        ("empty-input",
         {"model": "caedral-voice-1", "voice": "caedral-f1", "input": "   "},
         400),
        ("oversized-input",
         {"model": "caedral-voice-1", "voice": "caedral-f1",
          "input": "word " * 600}, 400),
        ("unsupported-language",
         {"model": "caedral-voice-1", "voice": "caedral-f1",
          "input": "Привет, как дела?"}, 400),
        ("invalid-speed",
         {"model": "caedral-voice-1", "voice": "caedral-f1", "input": SHORT,
          "speed": 9.9}, 400),
        ("bad-format",
         {"model": "caedral-voice-1", "voice": "caedral-f1", "input": SHORT,
          "response_format": "mp3"}, 400),
    ]
    for name, body, want in cases:
        st, _, payload = call(args.url, "/v1/audio/speech", body=body,
                              secret=args.secret)
        code = ""
        try:
            code = json.loads(payload)["error"]["code"]
        except Exception:
            pass
        check(f"error {name}", st == want and bool(code),
              f"status={st} code={code}")

    st, _, payload = call(args.url, "/v1/audio/speech", raw=b"{not json",
                          secret=args.secret)
    check("error malformed-json", st == 400, f"status={st}")

    # --- overload behavior ----------------------------------------------------
    statuses: list[int] = []
    lock = threading.Lock()

    def burst():
        st, _, _ = call(
            args.url, "/v1/audio/speech",
            body={"model": "caedral-voice-1", "voice": "caedral-f1",
                  "input": "Overload burst request.", "speed": 1.0,
                  "response_format": "wav"},
            secret=args.secret, timeout=420)
        with lock:
            statuses.append(st)

    threads = [threading.Thread(target=burst) for _ in range(12)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    n429 = statuses.count(429)
    n200 = statuses.count(200)
    check("overload clean-429-or-serve",
          all(s in (200, 429) for s in statuses) and n200 >= 1,
          f"200={n200} 429={n429} other={len(statuses)-n200-n429}")
    st, _, _ = call(args.url, "/healthz", secret=args.secret)
    check("survives-overload", st == 200, f"post-burst status={st} "
          f"burst_wall={time.time()-t0:.1f}s")

    print(f"\nSMOKE TEST: {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
