"""CV-G1-E0003 — two-process resume-proof for the Phase I training system.

Phases (each a SEPARATE OS process; the resume phase uses nothing but the
checkpoint file on disk -- no shared memory with part 1):

    scratch  fit 30 steps            -> experiments/CV-G1-E0003-training-system/
    resume   resume(last.pt), fit to 60 (same exp dir; metrics appended)
    fresh    uninterrupted 60 steps  -> experiments/CV-G1-E0003-training-system-fresh/

orchestrate runs all three as subprocesses, captures their logs under logs/,
then verifies: split-run weights at step 60 == fresh-run weights at step 60
(bitwise), optimizer/scheduler/rng states identical, per-step loss and lr
sequences identical across the join, checkpoint rotation correct.

Usage: python run_experiment.py [--phase orchestrate|scratch|resume|fresh]
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from caedral_voice.training import (  # noqa: E402
    Trainer, build_scheduler, load_config, read_jsonl, set_determinism,
)

EXP_ID = "CV-G1-E0003-training-system"
EXP_ID_FRESH = "CV-G1-E0003-training-system-fresh"
SEED = 20260822

CFG = {
    "experiment_root": "experiments",
    "run_id": None,
    "seed": SEED,
    "max_steps": 60,
    "warmup_steps": 6,
    "grad_clip": 1.0,
    "loss_fn": "mse",
    "log_every": 1,
    "val_every_steps": 15,
    "val_at_end": True,
    "checkpoint_every_steps": 15,
    "keep_step_checkpoints": 2,
    "best_key": "val_loss",
    "best_mode": "min",
}


def build(seed: int = SEED):
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_determinism(seed)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 16, generator=g)
    w = torch.randn(16, 16, generator=g) * 0.4
    y = x @ w.T + 0.02 * torch.randn(128, 16, generator=g)
    xval = torch.randn(32, 16, generator=g)
    yval = xval @ w.T
    train_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(xval, yval), batch_size=16, shuffle=False)
    model = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = build_scheduler(optimizer, CFG)
    return model, optimizer, scheduler, train_loader, val_loader


def run_phase(phase: str) -> dict:
    t0 = time.perf_counter()
    if phase == "scratch":
        model, opt, sched, tl, vl = build()
        summary = Trainer().fit(model, opt, sched, tl, vl,
                                {**CFG, "exp_id": EXP_ID, "max_steps": 30,
                                 "run_id": "split-part1"})
    elif phase == "resume":
        model, opt, sched, tl, vl = build()  # fresh process, fresh objects
        trainer = Trainer()
        info = trainer.resume(REPO / "experiments" / EXP_ID / "checkpoints"
                              / "last.pt")
        print(f"[resume] restored from disk: {info}")
        summary = trainer.fit(model, opt, sched, tl, vl,
                              {**CFG, "exp_id": EXP_ID, "max_steps": 60,
                               "run_id": "split-part2"})
    elif phase == "fresh":
        model, opt, sched, tl, vl = build()
        summary = Trainer().fit(model, opt, sched, tl, vl,
                                {**CFG, "exp_id": EXP_ID_FRESH,
                                 "run_id": "continuous"})
    else:
        raise SystemExit(f"unknown phase {phase}")
    summary["phase_wall_s"] = round(time.perf_counter() - t0, 3)
    print(f"[{phase}] {json.dumps(summary)}")
    return summary


def sha_state(obj):
    if isinstance(obj, torch.Tensor):
        return hashlib.sha256(obj.detach().numpy().tobytes()).hexdigest()
    if isinstance(obj, dict):
        return {str(k): sha_state(v) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [sha_state(v) for v in obj]
    return obj


def verify() -> dict:
    ck_main = torch.load(REPO / "experiments" / EXP_ID / "checkpoints"
                         / "last.pt", map_location="cpu", weights_only=False)
    ck_fresh = torch.load(REPO / "experiments" / EXP_ID_FRESH / "checkpoints"
                          / "last.pt", map_location="cpu", weights_only=False)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checks["both_at_step_60"] = (
        ck_main["trainer"]["global_step"] == ck_fresh["trainer"]["global_step"]
        == 60)

    max_diff = 0.0
    tensors_equal = True
    for k in ck_main["model"]:
        a, b = ck_main["model"][k], ck_fresh["model"][k]
        if not torch.equal(a, b):
            tensors_equal = False
            max_diff = max(max_diff, float((a - b).abs().max()))
    checks["model_weights_bitwise_identical"] = tensors_equal
    details["model_max_abs_diff"] = max_diff
    checks["optimizer_state_identical"] = (
        sha_state(ck_main["optimizer"]) == sha_state(ck_fresh["optimizer"]))
    checks["scheduler_position_identical"] = (
        ck_main["scheduler"]["last_epoch"] == ck_fresh["scheduler"]["last_epoch"]
        and ck_main["scheduler"]["_last_lr"] == ck_fresh["scheduler"]["_last_lr"])
    checks["rng_states_identical"] = bool(
        ck_main["rng"]["torch"].equal(ck_fresh["rng"]["torch"]))

    rows_split = read_jsonl(REPO / "experiments" / EXP_ID / "metrics.jsonl")
    rows_fresh = read_jsonl(REPO / "experiments" / EXP_ID_FRESH
                            / "metrics.jsonl")
    tr_split = [(r["step"], r["loss"], r["lr"])
                for r in rows_split if r["type"] == "train"]
    tr_fresh = [(r["step"], r["loss"], r["lr"])
                for r in rows_fresh if r["type"] == "train"]
    checks["train_rows_complete_1_to_60_no_gaps"] = (
        [s for s, _, _ in tr_split] == list(range(1, 61)))
    checks["loss_trajectory_identical_across_join"] = (
        [l for _, l, _ in tr_split] == [l for _, l, _ in tr_fresh])
    checks["lr_trajectory_identical_across_join"] = (
        [r for _, _, r in tr_split] == [r for _, _, r in tr_fresh])

    ckpt_dir = REPO / "experiments" / EXP_ID / "checkpoints"
    step_files = sorted(p.name for p in ckpt_dir.glob("step_*.pt"))
    checks["rotation_keeps_last_two_step_ckpts"] = (
        step_files == ["step_000000045.pt", "step_000000060.pt"])
    details["step_checkpoint_files"] = step_files

    val_rows = [(r["step"], r.get("val_loss"))
                for r in rows_split if r["type"] == "val"]
    best = json.loads(json.dumps({
        "best_value": ck_main["trainer"]["best_value"],
        "best_step": ck_main["trainer"]["best_step"],
        "val_steps": [s for s, _ in val_rows]}))
    checks["best_tracked"] = ck_main["trainer"]["best_step"] is not None
    details["best"] = best

    nan_events = sum(1 for r in rows_split if r.get("event") in
                     ("nan_loss_skipped", "nonfinite_grad_skipped"))
    details["nan_events_total"] = nan_events

    passed = all(checks.values())
    proof = {
        "experiment": "CV-G1-E0003",
        "claim": ("30-step run + separate-process resume from last.pt "
                  "reaches bitwise-identical state to an uninterrupted "
                  "60-step run"),
        "passed": passed,
        "checks": checks,
        "details": details,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPO / "experiments" / EXP_ID / "results"
    out.mkdir(exist_ok=True)
    (out / "resume-proof.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8")
    return proof


def orchestrate() -> int:
    exp_dir = REPO / "experiments" / EXP_ID
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    cfg_manifest = {
        "exp_id": EXP_ID,
        "cfg": {**CFG, "run_id": ["split-part1", "split-part2",
                                  "continuous (fresh reference)"]},
        "task": {"kind": "synthetic linear regression + tanh MLP",
                 "n_train": 128, "n_val": 32, "dim": 16, "batch": 16,
                 "batches_per_epoch": 8, "optimizer": "Adam lr=1e-2"},
        "seed": SEED,
        "versions": {"python": platform.python_version(),
                     "machine": platform.platform()},
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [
                            REPO / "caedral_voice" / "training" / "seed.py",
                            REPO / "caedral_voice" / "training" / "trainer.py",
                            Path(__file__)]},
    }
    (exp_dir / "config.json").write_text(json.dumps(cfg_manifest, indent=2),
                                         encoding="utf-8")

    t_all = time.perf_counter()
    ok = True
    for i, phase in enumerate(("scratch", "resume", "fresh"), start=1):
        cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
               "--phase", phase]
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        log = log_dir / f"{i}-{phase}.log"
        log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr
                                      if proc.stderr else ""), encoding="utf-8")
        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[orchestrate] phase {phase}: {status} -> {log.name}")
        ok = ok and proc.returncode == 0
        if not ok:
            break

    proof = verify() if ok else {"passed": False,
                                 "error": "a phase failed; see logs/"}
    wall = round(time.perf_counter() - t_all, 2)
    summary = {"passed": proof.get("passed", False),
               "total_wall_s": wall, "proof": proof}
    (exp_dir / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    verdict = "PROVEN" if summary["passed"] else "NOT PROVEN"
    print(f"[orchestrate] resume-proof: {verdict} ({wall}s total)")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    phase = "orchestrate"
    if "--phase" in sys.argv:
        phase = sys.argv[sys.argv.index("--phase") + 1]
    if phase == "orchestrate":
        raise SystemExit(orchestrate())
    run_phase(phase)
