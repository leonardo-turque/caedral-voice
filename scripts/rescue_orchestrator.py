"""LOCAL EVENT-DRIVEN RESCUE ORCHESTRATOR — detached from any Claude session.

Watches experiments/rescue-prosody/checkpoints/ for step_2000/step_3000 events,
renders A/B vs FAILED_BASELINE automatically, classifies, emits verdicts,
updates artifacts/rescue-state/orchestrator-state.json atomically.

Stdlib-only core; evaluation subprocesses reuse repo venv.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO / "experiments" / "rescue-prosody" / "checkpoints"
STATE_DIR = REPO / "artifacts" / "rescue-state"
STATE_FILE = STATE_DIR / "orchestrator-state.json"
LOG_FILE = STATE_DIR / "orchestrator.log"
AB_BASE = REPO / "artifacts" / "pre-arch-rescue"
WORKER_PID_FILE = REPO / "experiments" / "CV-G1-E0013-regressor" / "rescue_worker.pid"
TARGETS = [2000, 3000]
PROCESSED_SEED = [1000]
POLL_S = 10
STABLE_CHECKS = 3          # consecutive polls with identical size => complete
HANG_CPU_THRESHOLD = 1.0   # seconds of CPU delta considered minimum
HANG_WINDOW_S = 240
HANG_NO_PROGRESS_S = 900


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with io.open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "started_at_utc": datetime.utcnow().isoformat() + "Z",
        "processed": list(PROCESSED_SEED),
        "verdicts": {},
        "worker_pid": None,
        "phase": "WATCHING",
        "hang_evidence": [],
        "challenger_launched": None,
    }


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(STATE_FILE, state)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_complete(path: Path, last_size: int | None) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    return size > 1e6 and size == last_size


def find_worker_pid() -> int | None:
    pid_file = WORKER_PID_FILE
    _ = pid_file  # launcher may record it; fall back to scan
    import psutil

    best = None
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if ("train_regressor" in cl and "python" in (p.info["name"] or "").lower()
                    and "--exp-id rescue-prosody" in cl):
                rss = p.info["memory_info"].rss if p.info["memory_info"] else 0
                if best is None or rss > best[1]:
                    best = (p.info["pid"], rss)
        except Exception:
            continue
    return best[0] if best else None


def run_ab_evaluation(step: int, ckpt_path: Path) -> dict:
    out_dir = AB_BASE / f"checkpoint-{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{step}] rendering A/B via subprocess...")
    r = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(REPO / "scripts" / "render_rescue_ab.py"),
         f"checkpoints/step_{step:06d}.pt"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    (out_dir / "render-log.txt").write_text(r.stdout[-4000:] + "\n---ERR---\n" + r.stderr[-4000:],
                                            encoding="utf-8")
    if r.returncode != 0:
        return {"error": r.stderr[-500:]}

    sys.path.insert(0, str(REPO / "scripts"))
    import classify_preview as cp  # noqa: E402

    refs = {"evalA": SENTENCES_A, "evalB": SENTENCES_B}
    metrics_rows = []
    wer_list, dmos_list, classes = [], [], []
    for wav in sorted(out_dir.glob("*.wav")):
        parts = wav.stem.split("-")           # baseline|rescue, spk, sentence
        who, spk, sname = parts[0], parts[1], parts[2]
        ref = refs.get(sname, "")
        m = cp.classify_file(str(wav), ref)
        row = {"file": wav.name, "who": who, "speaker": spk, "sentence": sname,
               "class": m.get("class"), "min_wer": m.get("min_wer"),
               "dnsmos": m.get("dnsmos_ovrl"), "audio_seconds": m.get("audio_seconds")}
        metrics_rows.append(row)
        if who == "rescue":
            classes.append(m.get("class"))
            if m.get("min_wer") is not None:
                wer_list.append(m["min_wer"])
            if m.get("dnsmos_ovrl") is not None:
                dmos_list.append(m["dnsmos_ovrl"])

    rescue_rows = [r for r in metrics_rows if r["who"] == "rescue"]
    baseline_rows = [r for r in metrics_rows if r["who"] == "baseline"]
    catastrophic = (
        all(r["class"] in ("BABBLE", "NOISE") for r in rescue_rows)
        or (wer_list and max(wer_list) > 0.9)
        or any(r.get("audio_seconds", 1) < 0.4 for r in rescue_rows)
    )
    promising = (wer_list and sum(wer_list) / len(wer_list) <= 0.35
                 and not catastrophic)

    if catastrophic:
        verdict = "CATASTROPHIC"
    elif promising:
        verdict = "PROMISING"
    else:
        verdict = "AMBIGUOUS"

    result = {"step": step, "checkpoint_sha256": sha256(ckpt_path),
              "rows": metrics_rows, "baseline_rows": baseline_rows,
              "verdict": verdict,
              "rescue_avg_wer": round(sum(wer_list) / len(wer_list), 4) if wer_list else None,
              "rescue_dnsmos_max": round(max(dmos_list), 3) if dmos_list else None}
    atomic_write_json(out_dir / "metrics.json", result)
    verdict_doc = {
        "step": step, "verdict": verdict,
        "rule": ("CATASTROPHIC=all-babble/silence/gross-WER; PROMISING=avgWER<=0.35; "
                 "else AMBIGUOUS->continue-to-3000"),
    }
    atomic_write_json(out_dir / "verdict.json", verdict_doc)
    md = ["# Checkpoint-{0} A/B".format(step), "",
          "| file | class | WER | DNSMOS |", "|---|---|---|---|"]
    for row in metrics_rows:
        md.append("| {file} | {class} | {min_wer} | {dnsmos} |".format(**row))
    md += ["", f"VERDICT: {verdict}", ""]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")
    return result


SENTENCES_A = "Hello. This is the first generation of Caedral Voice."
SENTENCES_B = "Artificial intelligence should sound natural, expressive, and human."


def maybe_launch_challenger(state: dict, reason: str) -> None:
    """CHALLENGER-A: 3-speaker highest-quality subset, pitch+energy ON, 6k steps."""
    log(f"launching CHALLENGER-A ({reason})")
    subset_src = REPO / "data" / "manifests" / "rescue-subset.jsonl"
    if not subset_src.exists():
        log("CHALLENGER-A manifest missing; cannot auto-launch")
        state["challenger_launched"] = {"status": "BLOCKED", "reason": reason,
                                        "at": datetime.utcnow().isoformat() + "Z"}
        return
    py = str(REPO / ".venv" / "Scripts" / "python.exe")
    cmd = ("-X utf8 experiments\\CV-G1-E0013-regressor\\train_regressor.py "
           "--manifests rescue-subset,teacher-kokoro "
           "--durations data\\manifests\\all-durations.jsonl "
           "--mel-cache-dir data\\cache\\mel-v1 --steps 6000 --threads 6 "
           "--train-pitch --train-energy --exp-id challenger-a")
    logs = REPO / "artifacts" / "rescue-state" / "challenger-a.log"
    p = subprocess.Popen(
        ["cmd.exe", "/c", f'"{py}" {cmd} > "{logs}" 2>&1'],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    state["challenger_launched"] = {"status": "LAUNCHED", "pid": p.pid,
                                    "reason": reason,
                                    "at": datetime.utcnow().isoformat() + "Z"}
    log(f"CHALLENGER-A launched PID {p.pid}")


def kill_worker(pid: int, why: str) -> None:
    try:
        proc = psutil_process(pid)
        if proc is not None:
            proc.kill()
            log(f"killed worker PID {pid}: {why}")
    except Exception as exc:
        log(f"kill worker {pid} failed: {exc}")


def psutil_process(pid: int):
    try:
        import psutil
        return psutil.Process(pid)
    except Exception:
        return None


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    save_state(state)
    log("orchestrator started; processed seed = "
        f"{state['processed']}; targets = {TARGETS}")

    last_sizes: dict[int, int | None] = {}
    stable_counts: dict[int, int] = {}
    hang_check_every = HANG_WINDOW_S // POLL_S
    tick = 0

    while True:
        tick += 1
        pending = [t for t in TARGETS
                   if t not in state["processed"]
                   and t not in state.get("verdicts", {})]
        if pending:
            t = min(pending)
            ckpt = CKPT_DIR / f"step_{t:06d}.pt"
            last_size = last_sizes.get(t)
            size_now = ckpt.stat().st_size if ckpt.exists() else None
            if ckpt.exists():
                stable = (last_size == size_now)
                stable_counts[t] = stable_counts.get(t, 0) + 1 if stable else 0
                last_sizes[t] = size_now
                if stable and stable_counts[t] >= STABLE_CHECKS:
                    log(f"[{t}] checkpoint complete "
                        f"({size_now/1e6:.1f} MB); evaluating...")
                    state["phase"] = f"EVALUATING-{t}"
                    save_state(state)
                    try:
                        result = run_ab_evaluation(t, ckpt)
                        state["verdicts"][str(t)] = {
                            "verdict": result.get("verdict", "ERROR"),
                            "avg_wer": result.get("rescue_avg_wer"),
                            "dnsmos_max": result.get("rescue_dnsmos_max"),
                        }
                        log(f"[{t}] VERDICT = {result.get('verdict')}")
                    except Exception as exc:
                        state["verdicts"][str(t)] = {"verdict": "EVAL_ERROR",
                                                     "error": str(exc)[:300]}
                        log(f"[{t}] EVAL ERROR: {exc}")
                    state["processed"].append(t)
                    if state["verdicts"].get(str(t), {}).get("verdict") == "CATASTROPHIC":
                        pid = find_worker_pid()
                        if pid:
                            kill_worker(pid, "CATASTROPHIC verdict at "
                                        f"step {t}")
                        state["phase"] = "KILLED-CATASTROPHIC"
                        maybe_launch_challenger(state, f"probe killed at {t}")
                    elif t == TARGETS[-1]:
                        state["phase"] = "PROBE-FINISHED"
                    else:
                        state["phase"] = "WATCHING"
                    save_state(state)
                    last_sizes.pop(t, None)
        # worker health / hang detection
        if tick % hang_check_every == 0:
            pid = find_worker_pid()
            state["worker_pid"] = pid
            if pid is None:
                if not pending:
                    state["phase"] = state.get("phase", "WATCHING")
                log("no training worker process found (may have finished)")
            else:
                proc = psutil_process(pid)
                cpu_delta = 0.0
                if proc is not None:
                    t0 = proc.cpu_times()
                    time.sleep(2)
                    t1 = psutil_process(pid).cpu_times()
                    cpu_delta = (t1.user - t0.user) + (t1.system - t0.system)
                metrics_file = (REPO / "experiments" / "rescue-prosody"
                                / "metrics.jsonl")
                mtime_age = time.time() - metrics_file.stat().st_mtime \
                    if metrics_file.exists() else 1e9
                if (cpu_delta < HANG_CPU_THRESHOLD
                        and mtime_age > HANG_NO_PROGRESS_S / 3):
                    evidence = {"at": datetime.utcnow().isoformat() + "Z",
                                "cpu_delta_2s": cpu_delta,
                                "metrics_age_s": round(mtime_age)}
                    state.setdefault("hang_evidence", []).append(evidence)
                    log(f"HANG SUSPECTED: {evidence}")
        save_state(state)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
