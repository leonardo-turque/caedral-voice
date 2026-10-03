"""Tests for the Phase I training infrastructure (caedral_voice.training).

Proves, on synthetic regression tasks only:
- determinism: same seed twice => bitwise-identical loss trajectories;
- resume-exactness: 30 steps + resume-to-60 == fresh 60-step run, exactly;
- nan-guard: a NaN batch skips the optimizer step, logs an event, never crashes;
- best-checkpoint protection: best.pt untouched (mtime + content) while val worsens;
- checkpoint rotation keeps the newest N step checkpoints.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.training import (  # noqa: E402
    Trainer,
    build_scheduler,
    load_config,
    read_jsonl,
    set_determinism,
)

SEED = 1234
DIM_IN, DIM_OUT, N, BATCH = 8, 8, 64, 16  # 4 optimizer steps per epoch


@pytest.fixture(autouse=True)
def _allow_unsafe_checkpoints(monkeypatch):
    # every checkpoint in this module is written and read by the test itself
    # (trusted), so the 2026-10-03 audit gate on full-pickle resume payloads
    # is explicitly enabled here.
    monkeypatch.setenv("CV_ALLOW_UNSAFE_CHECKPOINT", "1")


def make_task(seed: int = SEED):
    """Fixed synthetic regression task: y = x @ W^T + b + tiny noise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, DIM_IN, generator=g)
    w = torch.randn(DIM_OUT, DIM_IN, generator=g) * 0.5
    y = x @ w.T + 0.01 * torch.randn(N, DIM_OUT, generator=g)
    return x, y


def make_objects(seed: int = SEED, batch: int = BATCH):
    set_determinism(seed)
    x, y = make_task(seed)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)
    model = nn.Sequential(
        nn.Linear(DIM_IN, 32), nn.Tanh(), nn.Linear(32, DIM_OUT))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    cfg = {"max_steps": 60, "warmup_steps": 5, "grad_clip": 1.0}
    scheduler = build_scheduler(optimizer, cfg)
    return model, optimizer, scheduler, train_loader, val_loader


def base_cfg(tmp_path: Path, exp_id: str, **kw) -> dict:
    cfg = {
        "exp_id": exp_id,
        "experiment_root": str(tmp_path / "experiments"),
        "run_id": "test",
        "seed": SEED,
        "max_steps": 60,
        "warmup_steps": 5,
        "grad_clip": 1.0,
        "log_every": 1,
        "val_every_steps": 10,
        "checkpoint_every_steps": 20,
        "keep_step_checkpoints": 2,
        "best_key": "val_loss",
        "best_mode": "min",
    }
    cfg.update(kw)
    return cfg


def state_digest(obj) -> dict:
    """Recursive structural digest; leaf torch tensors hashed by raw bytes."""
    if isinstance(obj, torch.Tensor):
        return hashlib.sha256(obj.detach().numpy().tobytes()).hexdigest()
    if isinstance(obj, dict):
        return {str(k): state_digest(v) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [state_digest(v) for v in obj]
    return obj


def load_ckpt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_same_seed_twice_identical_loss_trajectories(tmp_path):
    def run(exp_id: str) -> list[float]:
        model, opt, sched, tl, vl = make_objects(SEED)
        Trainer().fit(model, opt, sched, tl, vl, base_cfg(tmp_path, exp_id))
        rows = read_jsonl(tmp_path / "experiments" / exp_id / "metrics.jsonl")
        return [(r["step"], r["loss"]) for r in rows if r["type"] == "train"]

    traj_a = run("det-a")
    traj_b = run("det-b")
    assert len(traj_a) == 60
    assert traj_a == traj_b  # bitwise-identical (step, loss) trajectory


def test_seed_info_dict(tmp_path):
    info = set_determinism(42)
    assert info["seed"] == 42
    assert info["deterministic_algorithms"] is True
    assert info["num_threads"] >= 1
    assert "torch" in info["versions"]
    assert torch.initial_seed() == 42


# --------------------------------------------------------------------------- #
# resume-exactness
# --------------------------------------------------------------------------- #

def test_resume_split_30_plus_30_equals_fresh_60(tmp_path):
    cfg = base_cfg(tmp_path, "resume-split", checkpoint_every_steps=30,
                   val_every_steps=15)

    # part 1: fresh 30-step run (its own Trainer, like its own process)
    model1, opt1, sched1, tl1, vl1 = make_objects(SEED)
    Trainer().fit(model1, opt1, sched1, tl1, vl1, {**cfg, "max_steps": 30})
    last = tmp_path / "experiments" / "resume-split" / "checkpoints" / "last.pt"
    assert load_ckpt(last)["trainer"]["global_step"] == 30

    # part 2: NEW Trainer (fresh process equivalent) resumes and runs to 60
    model2, opt2, sched2, tl2, vl2 = make_objects(SEED)
    t2 = Trainer()
    info = t2.resume(last)
    assert info["global_step"] == 30
    t2.fit(model2, opt2, sched2, tl2, vl2, {**cfg, "max_steps": 60,
                                            "run_id": "test-part2"})

    # reference: uninterrupted 60-step run
    model3, opt3, sched3, tl3, vl3 = make_objects(SEED)
    Trainer().fit(model3, opt3, sched3, tl3, vl3,
                  {**cfg, "exp_id": "resume-fresh", "max_steps": 60})

    ck_split = load_ckpt(last.parent / "last.pt")
    ck_fresh = load_ckpt(last.parent.parent.parent / "resume-fresh"
                         / "checkpoints" / "last.pt")
    assert ck_split["trainer"]["global_step"] == 60
    assert ck_fresh["trainer"]["global_step"] == 60

    # model weights: bitwise identical
    assert state_digest(ck_split["model"]) == state_digest(ck_fresh["model"])
    for k in ck_split["model"]:
        assert torch.equal(ck_split["model"][k], ck_fresh["model"][k]), k
    # optimizer state: identical too
    assert state_digest(ck_split["optimizer"]["state"]) == \
        state_digest(ck_fresh["optimizer"]["state"])
    # scheduler position + current lr identical
    assert ck_split["scheduler"]["last_epoch"] == ck_fresh["scheduler"]["last_epoch"]
    assert ck_split["scheduler"]["_last_lr"] == ck_fresh["scheduler"]["_last_lr"]
    assert ck_split["rng"]["torch"].equal(ck_fresh["rng"]["torch"])


def test_step_checkpoint_rotation_keeps_last_two(tmp_path):
    model, opt, sched, tl, vl = make_objects(SEED)
    Trainer().fit(model, opt, sched, tl, vl,
                  base_cfg(tmp_path, "rotation", checkpoint_every_steps=15,
                           max_steps=60))
    ckpt_dir = tmp_path / "experiments" / "rotation" / "checkpoints"
    step_files = sorted(p.name for p in ckpt_dir.glob("step_*.pt"))
    assert step_files == ["step_000000045.pt", "step_000000060.pt"]  # %09d layout
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    # no torn temp files left behind
    assert not list(ckpt_dir.glob("*.tmp-*"))


# --------------------------------------------------------------------------- #
# nan guard
# --------------------------------------------------------------------------- #

def test_nan_guard_skips_step_logs_event_keeps_weights(tmp_path):
    calls = {"n": 0}
    real_mse = torch.nn.functional.mse_loss

    def poisoned_loss(pred, target):
        calls["n"] += 1
        if calls["n"] == 3:  # exactly one poisoned training step
            return torch.tensor(float("nan"), requires_grad=True)
        return real_mse(pred, target)

    snapshots: list[tuple[int, bool, dict]] = []

    def on_step_end(trainer, skipped):
        snapshots.append((trainer.global_step, skipped,
                          state_digest(trainer._model.state_dict())))

    model, opt, sched, tl, vl = make_objects(SEED)
    summary = Trainer().fit(model, opt, sched, tl, vl, base_cfg(
        tmp_path, "nan-guard", max_steps=12, log_every=1,
        val_every_steps=0, val_at_end=False,
        loss_fn=poisoned_loss, hooks={"on_step_end": on_step_end}))

    # training continued to the full 12 steps despite the NaN attempt
    assert summary["global_step"] == 12
    assert summary["nan_events"] == 1

    # event logged in memory and in metrics.jsonl
    exp_dir = tmp_path / "experiments" / "nan-guard"
    rows = read_jsonl(exp_dir / "metrics.jsonl")
    nan_rows = [r for r in rows if r.get("event") == "nan_loss_skipped"]
    assert len(nan_rows) == 1
    assert nan_rows[0]["attempt_step"] == 3

    # weights were untouched by the skipped attempt: the digest after the
    # poisoned attempt equals the digest after the previous completed step
    idx = next(i for i, (_, skipped, _) in enumerate(snapshots) if skipped)
    assert snapshots[idx][2] == snapshots[idx - 1][2]
    # ...and only that one pair of consecutive attempts is identical
    equal_pairs = sum(
        1 for i in range(1, len(snapshots))
        if snapshots[i][2] == snapshots[i - 1][2])
    assert equal_pairs == 1


# --------------------------------------------------------------------------- #
# best-checkpoint protection
# --------------------------------------------------------------------------- #

def test_best_checkpoint_untouched_while_val_worsens(tmp_path):
    # Validation targets are ANTI-correlated with the training mapping, so as
    # train loss falls, val loss rises monotonically -> best is the FIRST val.
    set_determinism(SEED)
    x, _ = make_task(SEED)
    g = torch.Generator().manual_seed(SEED + 1)
    w = torch.randn(DIM_OUT, DIM_IN, generator=g) * 0.5
    y_train = x @ w.T
    y_val = -(x @ w.T)
    train_loader = DataLoader(TensorDataset(x, y_train), batch_size=BATCH,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y_val), batch_size=BATCH,
                            shuffle=False)
    model = nn.Sequential(nn.Linear(DIM_IN, 32), nn.Tanh(),
                          nn.Linear(32, DIM_OUT))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = build_scheduler(optimizer, {"max_steps": 40, "warmup_steps": 4})

    cfg = base_cfg(tmp_path, "best-protect", max_steps=40,
                   val_every_steps=5, checkpoint_every_steps=40)
    ckpt_dir = tmp_path / "experiments" / "best-protect" / "checkpoints"
    best = ckpt_dir / "best.pt"

    trainer = Trainer()
    trainer.fit(model, optimizer, scheduler, train_loader, val_loader, cfg)

    rows = read_jsonl(tmp_path / "experiments" / "best-protect" / "metrics.jsonl")
    vals = [(r["step"], r["val_loss"]) for r in rows if r["type"] == "val"]
    assert len(vals) >= 4
    assert vals[0][0] == 5
    # val genuinely worsens after the first measurement
    assert vals[0][1] < min(v for s, v in vals[1:])
    # best.pt was written at the first val and never again
    best_rows = [r for r in rows if r.get("event") == "best_updated"]
    assert [r["step"] for r in best_rows] == [5]

    stat = os.stat(best)
    content = hashlib.sha256(best.read_bytes()).hexdigest()

    # keep training (more worsening vals) in the SAME directory; best.pt must
    # not be rewritten -- neither its content nor its mtime
    trainer2 = Trainer()
    trainer2.resume(best.parent / "last.pt")
    trainer2.fit(model, optimizer, scheduler, train_loader, val_loader,
                 {**cfg, "max_steps": 80, "run_id": "test-part2"})

    stat2 = os.stat(best)
    content2 = hashlib.sha256(best.read_bytes()).hexdigest()
    assert stat.st_mtime_ns == stat2.st_mtime_ns
    assert content == content2
    assert load_ckpt(best)["trainer"]["global_step"] == 5


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #

def test_config_loads_from_json_with_defaults(tmp_path):
    cfg = load_config(REPO / "configs" / "training" / "smoke.json")
    assert cfg["max_steps"] == 60
    assert cfg["warmup_steps"] == 5
    assert cfg["best_key"] == "val_loss"
    assert cfg["keep_step_checkpoints"] == 2  # default carried through
    overrides = load_config(REPO / "configs" / "training" / "smoke.json",
                            {"max_steps": 10})
    assert overrides["max_steps"] == 10
    assert overrides["warmup_steps"] == 5


def test_scheduler_warmup_then_cosine(tmp_path):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=1.0)
    sched = build_scheduler(opt, {"max_steps": 10, "warmup_steps": 4})
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.25)      # warmup ramp 1/4
    assert lrs[3] == pytest.approx(1.0)       # end of warmup
    assert lrs[4] < 1.0                       # cosine decay begins
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)  # fully decayed
