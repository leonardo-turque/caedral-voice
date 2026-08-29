#!/usr/bin/env python3
"""OVERLOAD SURVIVAL TEST — exceed recommended throughput briefly.

Fires K concurrent short requests (default 12 > queue capacity 8) at the
internal service. Verifies: bounded queue (clean 429 with Retry-After),
no crash, health/ready PASS afterwards, and a normal request still succeeds.
stdlib-only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

SHORT = "The harbour lights trembled on the swells."


def call(url, path, body=None, secret="", timeout=420):
    req = urllib.request.Request(
        url + path, method="POST" if body is not None else "GET")
    if secret:
        req.add_header("X-Internal-Secret", secret)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    else:
        data = None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}", {}, b""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--secret", default="")
    args = ap.parse_args()

    body = {"model": "caedral-voice-1", "voice": "caedral-f1",
            "input": SHORT, "speed": 1.0, "response_format": "wav"}

    st, _, _ = call(args.url, "/healthz", secret=args.secret)
    pre_health = st
    results = []
    lock = threading.Lock()

    def one():
        s, h, p = call(args.url, "/v1/audio/speech", body=body,
                       secret=args.secret)
        code = ""
        try:
            code = json.loads(p)["error"]["code"]
        except Exception:
            pass
        retry_after = {k.lower(): v for k, v in h.items()}.get("retry-after")
        riff = p[:4] == b"RIFF" if isinstance(p, bytes) else False
        with lock:
            results.append({"status": s, "code": code,
                            "retry_after": retry_after, "riff": riff})

    t0 = time.time()
    ts = [threading.Thread(target=one) for _ in range(args.k)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0

    st_h, _, _ = call(args.url, "/healthz", secret=args.secret)
    st_r, _, _ = call(args.url, "/readyz", secret=args.secret)
    st_n, h_n, p_n = call(args.url, "/v1/audio/speech", body=body,
                          secret=args.secret)

    n200 = sum(1 for r in results if r["status"] == 200)
    n429 = sum(1 for r in results if r["status"] == 429)
    ra_on_429 = all(r["retry_after"] is not None
                    for r in results if r["status"] == 429)
    summary = {
        "burst_k": args.k,
        "statuses": {str(s): sum(1 for r in results
                                 if str(r["status"]) == str(s))
                     for s in {r["status"] for r in results}},
        "n_200": n200, "n_429": n429,
        "other": len(results) - n200 - n429,
        "retry_after_on_all_429": ra_on_429,
        "all_200_riff_ok": all(r["riff"] for r in results
                               if r["status"] == 200),
        "codes_seen": sorted({r["code"] for r in results if r["code"]}),
        "pre_health": pre_health,
        "post_health": st_h, "post_ready": st_r,
        "post_normal_request_status": st_n,
        "post_normal_riff": p_n[:4] == b"RIFF",
        "burst_wall_s": round(wall, 1),
    }
    print("OVERLOAD_SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
