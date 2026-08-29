#!/usr/bin/env python3
"""REAL SERVER FIRE TEST — sequential canonical-request driver.

Runs N sequential canonical ~30 s generations against the deployed internal
service and samples container cgroup limits/usage around every request.

stdlib-only. Reads canonical-input.txt next to this script.
Writes <label>-seq.csv and <label>-seq.json into --out dir.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

CONTAINER = "caedral-voice-api"


def post_speech(url: str, body: dict, timeout: float = 420.0,
                secret: str = "") -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/audio/speech", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    if secret:
        req.add_header("X-Internal-Secret", secret)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            # uvicorn lowercases response header names on the wire
            h = {k.lower(): v for k, v in resp.headers.items()}
            dur = None
            try:
                with wave.open(io.BytesIO(payload), "rb") as w:
                    rate, ch, sw, frames = (w.getframerate(), w.getnchannels(),
                                            w.getsampwidth(), w.getnframes())
                    dur = frames / rate
                fmt_ok = (rate == 24000 and ch == 1 and sw == 2
                          and payload[:4] == b"RIFF"
                          and payload[8:12] == b"WAVE")
            except Exception:
                fmt_ok, rate, ch, sw = False, None, None, None
            return {"status": resp.status, "submit": t0, "done": time.time(),
                    "e2e_s": round(time.time() - t0, 3),
                    "gen_s": float(h.get("x-caedral-wall-seconds", 0)),
                    "audio_s": float(h.get("x-caedral-audio-seconds", 0)),
                    "rtf_hdr": h.get("x-caedral-rtf"),
                    "bytes": len(payload),
                    "riff": payload[:4] == b"RIFF",
                    "fmt_ok": fmt_ok, "rate": rate, "ch": ch,
                    "sampwidth": sw, "dur_parsed_s": round(dur, 3) if dur else None}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "submit": t0, "done": time.time(),
                "e2e_s": round(time.time() - t0, 3), "gen_s": 0.0,
                "audio_s": 0.0, "rtf_hdr": None, "bytes": 0, "riff": False,
                "fmt_ok": False, "rate": None, "ch": None,
                "sampwidth": None, "dur_parsed_s": None}
    except Exception as exc:  # noqa: BLE001 — record honestly
        return {"status": f"ERR:{type(exc).__name__}", "submit": t0,
                "done": time.time(), "e2e_s": round(time.time() - t0, 3),
                "gen_s": 0.0, "audio_s": 0.0, "rtf_hdr": None, "bytes": 0,
                "riff": False, "fmt_ok": False, "rate": None, "ch": None,
                "sampwidth": None, "dur_parsed_s": None}


def get_json(url: str, path: str, timeout: float = 5):
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cg(container: str) -> dict:
    """cgroup v2 facts inside the container (authoritative limits/usage)."""
    out: dict = {}

    def rd(path):
        try:
            return subprocess.run(
                ["docker", "exec", container, "cat", path],
                capture_output=True, text=True, timeout=6).stdout.strip()
        except Exception:
            return ""

    out["memory.max"] = rd("/sys/fs/cgroup/memory.max")
    out["memory.swap.max"] = rd("/sys/fs/cgroup/memory.swap.max")
    out["memory.current"] = rd("/sys/fs/cgroup/memory.current")
    out["memory.peak"] = rd("/sys/fs/cgroup/memory.peak")
    stat = rd("/sys/fs/cgroup/cpu.max")          # "quota period"
    out["cpu.max"] = stat
    for line in rd("/sys/fs/cgroup/cpu.stat").splitlines():
        k, _, v = line.partition(" ")
        if k:
            out[k] = int(v) if v.isdigit() else v
    msw = rd("/sys/fs/cgroup/memory.events")
    for line in msw.splitlines():
        k, _, v = line.partition(" ")
        if k:
            out["mem_" + k] = int(v) if v.isdigit() else v
    return out


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--secret", default="")
    ap.add_argument("--out", default=".")
    ap.add_argument("--input", default="canonical-input.txt",
                    help="input text file resolved next to this script "
                         "unless it is an absolute/relative path")
    ap.add_argument("--warmup", type=int, default=0,
                    help="unrecorded warm-up requests before measurement")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        in_path = Path(__file__).with_name(args.input)
    text = in_path.read_text(encoding="utf-8").strip()
    body = {"model": "caedral-voice-1", "voice": "caedral-f1",
            "input": text, "speed": 1.0, "response_format": "wav"}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # readiness gate
    for _ in range(120):
        if (get_json(args.url, "/readyz") or {}).get("ready"):
            break
        print("[WAIT] not ready...", flush=True)
        time.sleep(3)

    env0 = cg(CONTAINER)
    print("CGROUP_START " + json.dumps(env0, default=str), flush=True)

    for w in range(args.warmup):
        r = post_speech(args.url, body, secret=args.secret)
        print(f"[WARMUP {w + 1}/{args.warmup}] status={r['status']} "
              f"e2e={r['e2e_s']}s", flush=True)

    csv_path = out_dir / f"{args.label}-seq.csv"
    rows_all = []
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["i", "status", "e2e_s", "gen_s", "audio_s", "rtf_hdr",
                     "bytes", "riff", "fmt_ok", "rate", "ch", "sampwidth",
                     "mem_current", "mem_peak", "throttled_usec_delta"])
        prev_thr = env0.get("throttled_usec", 0)
        for i in range(1, args.n + 1):
            r = post_speech(args.url, body, secret=args.secret)
            c = cg(CONTAINER)
            thr_now = c.get("throttled_usec", 0)
            dthr = thr_now - prev_thr if isinstance(thr_now, int) else None
            prev_thr = thr_now
            row = [i, r["status"], r["e2e_s"], r["gen_s"], r["audio_s"],
                   r["rtf_hdr"], r["bytes"], r["riff"], r["fmt_ok"],
                   r["rate"], r["ch"], r["sampwidth"],
                   c.get("memory.current"), c.get("memory.peak"), dthr]
            wr.writerow(row)
            f.flush()
            rows_all.append({"i": i, **r, "cg": {
                k: c.get(k) for k in ("memory.current", "memory.peak")},
                "throttled_usec_delta": dthr})
            print(f"[{i}/{args.n}] status={r['status']} e2e={r['e2e_s']}s "
                  f"gen={r['gen_s']}s audio={r['audio_s']}s "
                  f"rtf={r['rtf_hdr']} peak={c.get('memory.peak')}",
                  flush=True)
            time.sleep(2)

    env_end = cg(CONTAINER)
    ok = [r for r in rows_all if r["status"] == 200]
    e2e = [r["e2e_s"] for r in ok]
    gen = [r["gen_s"] for r in ok]
    rtfs = [r["e2e_s"] / r["audio_s"] for r in ok if r["audio_s"]]
    peaks = [int(r["cg"]["memory.peak"]) / 1e6 for r in ok
             if r["cg"]["memory.peak"]]
    summ = {
        "label": args.label, "n": args.n,
        "ok": len(ok),
        "statuses": {str(s): sum(1 for r in rows_all if r["status"] == s)
                     for s in {r["status"] for r in rows_all}},
        "all_fmt_ok": all(r["fmt_ok"] for r in ok),
        "audio_s_range": [min((r["audio_s"] for r in ok), default=None),
                          max((r["audio_s"] for r in ok), default=None)],
        "e2e_min": min(e2e) if e2e else None,
        "e2e_p50": statistics.median(e2e) if e2e else None,
        "e2e_p95": pct(e2e, 0.95),
        "e2e_max": max(e2e) if e2e else None,
        "gen_p50": statistics.median(gen) if gen else None,
        "gen_p95": pct(gen, 0.95),
        "rtf_p50": statistics.median(rtfs) if rtfs else None,
        "rtf_p95": pct(rtfs, 0.95),
        "peak_mem_mb_max": max(peaks) if peaks else None,
        "cpu_stat_start": {k: env0.get(k) for k in
                           ("nr_periods", "nr_throttled", "throttled_usec")},
        "cpu_stat_end": {k: env_end.get(k) for k in
                         ("nr_periods", "nr_throttled", "throttled_usec")},
        "mem_events_end": {k: env_end.get("mem_" + k) for k in
                           ("oom", "oom_kill")},
        "limits": {k: env0.get(k) for k in
                   ("memory.max", "memory.swap.max", "cpu.max")},
        "theory_rpm_serial": round(60.0 / statistics.median(e2e), 4)
        if e2e else None,
    }
    jp = out_dir / f"{args.label}-seq.json"
    jp.write_text(json.dumps(summ, indent=2))
    print("SUMMARY " + json.dumps(summ, default=str), flush=True)


if __name__ == "__main__":
    main()
