#!/usr/bin/env python3
"""Server-variant of production/loadgen.py (REAL VPS fire test driver).

Identical methodology to the frozen local fire-test tooling — RPM ladder at
25/50/75/100/125 % of theoretical serial capacity (~5 min windows,
early-overload skip) plus concurrency bursts x2/x4 — with one mechanical
fix for the deployed service: uvicorn lowercases response header names, so
header lookups are case-insensitive here. Also takes an explicit --out dir.

stdlib-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_speech(url: str, body: dict, timeout: float = 420.0,
                secret: str = "") -> dict:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Internal-Secret"] = secret
    req = urllib.request.Request(
        url + "/v1/audio/speech", data=data, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return {"status": resp.status, "submit": t0, "done": time.time(),
                    "e2e_s": round(time.time() - t0, 3),
                    "audio_s": float(hdrs.get("x-caedral-audio-seconds", 0)),
                    "rtf": hdrs.get("x-caedral-rtf"),
                    "bytes": len(payload),
                    "riff_ok": payload[:4] == b"RIFF"
                               and payload[8:12] == b"WAVE"}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "submit": t0, "done": time.time(),
                "e2e_s": round(time.time() - t0, 3), "audio_s": 0.0,
                "rtf": None, "bytes": 0, "riff_ok": False}
    except Exception as exc:  # noqa: BLE001 — record honestly
        return {"status": f"ERR:{type(exc).__name__}", "submit": t0,
                "done": time.time(), "e2e_s": round(time.time() - t0, 3),
                "audio_s": 0.0, "rtf": None, "bytes": 0, "riff_ok": False}


def get_metrics(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--probe", required=True,
                    help="probe-results.json with summary.gen_p50_s")
    ap.add_argument("--label", required=True)
    ap.add_argument("--body", default="canonical-input.txt")
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--queue-capacity", type=int, default=8)
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--skip-bursts", action="store_true")
    ap.add_argument("--secret", default="")
    ap.add_argument("--out", default="artifacts/fire-test")
    args = ap.parse_args()

    text = Path(args.body).read_text(encoding="utf-8").strip()
    BODY = {"model": "caedral-voice-1", "voice": "caedral-f1",
            "input": text, "speed": 1.0, "response_format": "wav"}

    probe = json.loads(Path(args.probe).read_text())
    gen_p50 = probe["summary"]["gen_p50_s"]
    theory_rpm = 60.0 / gen_p50

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "load-test.csv"
    new_csv = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    wr = csv.writer(csv_f)
    if new_csv:
        wr.writerow(["label", "phase", "req_id", "submit_epoch",
                     "done_epoch", "e2e_s", "status", "audio_s", "rtf",
                     "bytes", "riff_ok"])
    summaries: dict = {"theory_rpm": round(theory_rpm, 4),
                       "gen_p50_s_source": gen_p50}

    while True:
        m = get_metrics(args.url)
        if m and m.get("ready"):
            break
        print("[WAIT] service not ready yet...", flush=True)
        time.sleep(3)

    # ---------------- A. RPM ladder ------------------------------------------
    if not args.skip_ladder:
        for frac in (0.25, 0.50, 0.75, 1.00, 1.25):
            rpm = theory_rpm * frac
            interval = 60.0 / rpm
            n_req = max(2, int(args.window * rpm) + 1)
            lvl = f"ladder-{frac:.2f}x"
            print(f"\n=== {args.label} {lvl}: target {rpm:.3f} RPM "
                  f"(interval {interval:.1f}s, ~{n_req} reqs, "
                  f"window {args.window:.0f}s) ===", flush=True)

            rows: list[dict] = []
            lock = threading.Lock()
            stop_flag = threading.Event()

            def submit_one(rid: int) -> None:
                if stop_flag.is_set():
                    return
                r = post_speech(args.url, BODY, secret=args.secret)
                r["rid"] = rid
                with lock:
                    rows.append(r)
                    wr.writerow([args.label, lvl, rid, f"{r['submit']:.3f}",
                                 f"{r['done']:.3f}", r["e2e_s"], r["status"],
                                 r["audio_s"], r["rtf"], r["bytes"],
                                 r["riff_ok"]])
                    csv_f.flush()

            metrics_log: list[tuple] = []
            threads_all: list[threading.Thread] = []
            t_start = time.time()
            next_t = t_start
            rid = 0
            stalled_since = None
            while time.time() - t_start < args.window:
                now = time.time()
                if now >= next_t and rid < n_req:
                    rid += 1
                    th = threading.Thread(target=submit_one, args=(rid,),
                                          daemon=True)
                    th.start()
                    threads_all.append(th)
                    next_t += interval
                m = get_metrics(args.url)
                if m:
                    metrics_log.append((round(time.time() - t_start, 1),
                                        m.get("queue_depth"),
                                        m.get("rss_mb"),
                                        m.get("completed")))
                    qd = m.get("queue_depth") or 0
                    recent = [r for r in rows
                              if r["done"] > time.time() - 120]
                    if qd >= args.queue_capacity and len(recent) == 0 \
                            and now - t_start > interval * 2:
                        if stalled_since is None:
                            stalled_since = now
                        elif now - stalled_since > 120:
                            print(f"[OVERLOAD] {lvl}: queue saturated, "
                                  "no completions in 120 s -> early stop",
                                  flush=True)
                            break
                    else:
                        stalled_since = None
                time.sleep(min(2.0, max(0.2, interval / 4)))

            stop_flag.set()
            for th in threads_all:
                th.join(timeout=args.window + 300)
            dur = time.time() - t_start
            ok_rows = [r for r in rows
                       if isinstance(r["status"], int) and r["status"] == 200]
            e2e = sorted(r["e2e_s"] for r in ok_rows)
            svc = sorted(r["audio_s"] / float(r["rtf"]) for r in ok_rows
                         if r["rtf"] and float(r["rtf"]) > 0)
            summaries[lvl] = {
                "target_rpm": round(rpm, 4),
                "submitted": rid,
                "completed_200": len(ok_rows),
                "failed": len(rows) - len(ok_rows),
                "statuses": {},
                "wall_window_s": round(dur, 1),
                "achieved_rpm": round(len(ok_rows) / (dur / 60.0), 4),
                "e2e_p50_s": statistics.median(e2e) if e2e else None,
                "e2e_max_s": e2e[-1] if e2e else None,
                "svc_p50_s": statistics.median(svc) if svc else None,
                "all_riff_ok": all(r["riff_ok"] for r in ok_rows),
                "overloaded": bool(stalled_since),
                "metrics_tail": metrics_log[-5:],
            }
            for r in rows:
                key = str(r["status"])
                summaries[lvl]["statuses"][key] = \
                    summaries[lvl]["statuses"].get(key, 0) + 1
            print(f"[LEVEL DONE] {lvl}: "
                  + json.dumps(summaries[lvl], default=str), flush=True)
            if summaries[lvl]["overloaded"]:
                summaries["overload_point"] = (
                    f"{lvl} (target {rpm:.3f} RPM): queue saturation without "
                    "completions; higher rates NOT tested")
                break

    # ---------------- B. concurrency bursts ----------------------------------
    if not args.skip_bursts:
        for conc in (2, 4):
            label = f"concurrency-{conc}"
            print(f"\n=== {args.label} {label}: one burst of {conc} "
                  "simultaneous requests ===", flush=True)
            rows2: list[dict] = []
            lock2 = threading.Lock()

            def burst():
                r = post_speech(args.url, BODY, secret=args.secret)
                with lock2:
                    rows2.append(r)
                    wr.writerow([args.label, label, "burst",
                                 f"{r['submit']:.3f}", f"{r['done']:.3f}",
                                 r["e2e_s"], r["status"], r["audio_s"],
                                 r["rtf"], r["bytes"], r["riff_ok"]])
                    csv_f.flush()

            time.sleep(5)
            t0 = time.time()
            ts = [threading.Thread(target=burst) for _ in range(conc)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            ok_rows = [r for r in rows2
                       if isinstance(r["status"], int) and r["status"] == 200]
            e2e = sorted(r["e2e_s"] for r in ok_rows)
            summaries[label] = {
                "burst_size": conc, "completed_200": len(ok_rows),
                "failed": len(rows2) - len(ok_rows),
                "e2e_each_s": [round(x, 2) for x in e2e],
                "burst_makespan_s": round(time.time() - t0, 2),
                "parallel_speedup_x":
                    round(sum(e2e) / max(time.time() - t0, 1e-9), 3)
                    if len(e2e) == conc else None,
                "all_riff_ok": all(r["riff_ok"] for r in ok_rows),
            }
            print(f"[BURST DONE] {label}: "
                  + json.dumps(summaries[label]), flush=True)

    csv_f.close()
    out_json = out_dir / f"load-summary-{args.label}.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
