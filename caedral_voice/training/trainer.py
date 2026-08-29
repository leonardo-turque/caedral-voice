"""Reproducible, crash-safe CPU-first training loop (Phase I infrastructure).

Contracts
---------
* **Reproducible** -- a fresh ``fit()`` re-seeds the global RNGs from
  ``cfg["seed"]`` (``reseed_on_fit``), so a loss trajectory depends only on
  (cfg, data, code), not on RNG consumed earlier while building the
  model/dataset.
* **Crash-safe resume** -- checkpoints store model/optimizer/scheduler state,
  the global step, the number of batches already consumed in the current
  epoch, and the global RNG state captured immediately after the last completed
  optimizer step. Data order is pinned to ``(cfg['seed'], epoch_index)`` by
  rebuilding the train DataLoader each epoch with an explicit per-epoch
  ``torch.Generator`` (DataLoader iterator creation otherwise consumes a draw
  from the GLOBAL torch RNG every time -- including for ``shuffle=False``
  loaders -- which silently desynchronizes resumed runs). On resume, the val/
  checkpoint actions that belong to the resume step are REPLAYED exactly as an
  uninterrupted run executed them, keeping even dropout streams aligned.
* **Atomic writes** -- every checkpoint is written to ``<name>.tmp-<pid>`` in
  the target directory, fsynced, then moved into place with ``os.replace``;
  a reader never observes a torn file and a crash never corrupts the previous
  checkpoint.
* **Never crash on bad numerics** -- NaN/Inf loss or gradients SKIP the
  optimizer step (weights untouched for that attempt), emit an event row to
  metrics.jsonl, and training continues with the next batch.

Checkpoint layout::

    <experiment_root>/<exp_id>/checkpoints/{last.pt, best.pt, step_%09d.pt}
    <experiment_root>/<exp_id>/metrics.jsonl

Rotating step checkpoints keep the newest ``keep_step_checkpoints`` files;
``best.pt`` is rewritten only when the tracked validation metric improves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .seed import get_rng_state, set_determinism, set_rng_state

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

DEFAULT_CFG: dict = {
    "exp_id": "unnamed",
    "experiment_root": "experiments",
    "run_id": None,                # tag written onto every metrics row
    "seed": 0,
    "reseed_on_fit": True,         # fresh runs re-seed global RNGs at fit() entry
    "num_threads": 1,              # torch intra-op threads pinned at reseed;
                                   # None leaves threading untouched
    "deterministic_algorithms": True,
    "max_steps": 1000,
    "grad_clip": 1.0,              # max grad L2 norm; None disables clipping
    "warmup_steps": 0,             # linear warmup steps (build_scheduler)
    "min_lr_ratio": 0.0,           # cosine floor as fraction of peak lr
    "loss_fn": "mse",              # "mse" | "l1" | callable(pred, target)
    "val_loss_fn": None,           # None -> mirror string spec; callable ok
    "log_every": 10,               # train-metric row cadence (steps); 0 disables
    "val_every_steps": 0,          # 0 disables step-cadence validation
    "val_every_epochs": 0,         # 0 disables epoch-cadence validation
    "val_at_end": True,
    "val_max_batches": 0,          # 0 = whole val_loader
    "checkpoint_every_steps": 0,   # 0 = only at end of fit
    "checkpoint_every_epochs": 0,
    "keep_step_checkpoints": 2,    # rotation window for step_%09d.pt
    "best_key": "val_loss",        # key inside validate()'s metric dict
    "best_mode": "min",            # "min" | "max"
}


# --------------------------------------------------------------------------- #
# config / small utilities
# --------------------------------------------------------------------------- #

def load_config(path: os.PathLike | str, overrides: dict | None = None) -> dict:
    """Load a training cfg JSON file merged over DEFAULT_CFG (+ overrides)."""
    with open(path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    cfg = dict(DEFAULT_CFG)
    cfg.update(loaded)
    if overrides:
        cfg.update(overrides)
    return cfg


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    """Read a JSONL log back as a list of dicts (skips blank lines)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_loss(spec: Any) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if callable(spec):
        return spec
    if spec == "mse":
        return F.mse_loss
    if spec == "l1":
        return F.l1_loss
    raise ValueError(f"unknown loss spec: {spec!r}")


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept {'x','y'} dicts or (x, y[, ...]) sequences."""
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("inputs"))
        y = batch.get("y", batch.get("targets", batch.get("labels")))
        if x is None or y is None:
            raise ValueError("dict batches need 'x' and 'y' keys")
        return x, y
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"cannot unpack batch of type {type(batch)!r}")


def _derive_epoch_seed(base_seed: int, epoch: int) -> int:
    """Stable per-epoch generator seed (process-independent, unlike hash())."""
    digest = hashlib.sha256(f"{int(base_seed)}:{int(epoch)}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _clone_loader_with_generator(loader, generator):
    """Rebuild a standard ``torch.utils.data.DataLoader`` with an explicit
    ``generator`` so its shuffling draws from THAT generator instead of the
    global torch RNG (DataLoader iterator creation otherwise consumes a
    global ``_base_seed`` draw every time, which breaks exact resume).

    Non-DataLoader iterables and exotic loaders are returned unchanged.
    """
    if not isinstance(loader, torch.utils.data.DataLoader):
        return loader
    from torch.utils.data import BatchSampler, RandomSampler

    kwargs = dict(
        dataset=loader.dataset,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=getattr(loader, "multiprocessing_context", None),
        generator=generator,
    )
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and not isinstance(batch_sampler, BatchSampler):
        # fully custom batch sampler: reuse as-is (cannot inject a generator)
        kwargs["batch_sampler"] = batch_sampler
    elif isinstance(loader.sampler, RandomSampler):
        kwargs["shuffle"] = True  # rebuilt RandomSampler binds our generator
    else:
        kwargs["sampler"] = loader.sampler
    if "batch_sampler" not in kwargs:
        kwargs["batch_size"] = loader.batch_size
        kwargs["drop_last"] = loader.drop_last
    try:
        return torch.utils.data.DataLoader(**kwargs)
    except TypeError:
        logger.warning("could not clone DataLoader with explicit generator; "
                       "using it as-is (resume order may not be exact)")
        return loader


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    """Linear warmup over ``cfg['warmup_steps']`` updates, then cosine decay to
    ``min_lr_ratio`` of peak over the remaining ``max_steps - warmup_steps``.

    Follows the HF convention: the lambda argument t counts completed
    scheduler.step() calls, so update k uses factor lambda(k-1); with
    warmup > 0 update 1 sees factor ~0, identical to
    get_linear_schedule_with_warmup semantics.
    """
    warmup = int(cfg.get("warmup_steps", 0))
    total = int(cfg["max_steps"])
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return float(current_step + 1) / float(max(1, warmup))
        decay_steps = max(1, total - warmup)
        progress = min(1.0, (current_step - warmup + 1) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class Trainer:
    """Plain-dict-configured training loop with exact, mid-epoch-capable resume."""

    def __init__(self) -> None:
        # loop position
        self.global_step = 0
        self.epoch = 0                       # completed epochs
        self.batches_consumed_in_epoch = 0
        self.max_steps = 0
        # best-tracking
        self.best_value: float | None = None
        self.best_step: int | None = None
        # dedupe val/checkpoint triggers when resuming exactly on a boundary
        self.last_val_step: int | None = None
        self.last_ckpt_step: int | None = None
        # reproducibility state
        self._last_rng: dict | None = None   # rng right after last completed step
        self._skip_batches = 0               # already-consumed batches to replay-skip
        self._pending_resume: dict | None = None
        # fit()-scoped objects/state
        self._iter = None
        self._model = None
        self._optimizer_ref = None
        self._scheduler_ref = None
        self._val_loader_ref = None
        self._val_loss_ref = F.mse_loss
        # opt-in CUDA mixed precision (cfg["amp"]; default OFF -> the FP32
        # code path below is bit-identical to pre-colab-r3 behavior)
        self._amp = False
        self._scaler = None
        self._window: list[float] = []
        self._last_loss = float("nan")
        self._last_grad_norm: float | None = None
        self._t_fit_start = 0.0
        # bookkeeping
        self.wall_accum_s = 0.0
        self.cfg: dict = {}
        self.out_dir: Path | None = None
        self.checkpoint_dir: Path | None = None
        self.events: deque = deque(maxlen=10_000)
        self._metrics_fh = None
        self.nan_events = 0

    # ------------------------------------------------------------- public --

    def fit(self, model, optimizer, scheduler,
            train_loader: Iterable, val_loader: Iterable | None,
            cfg: dict) -> dict:
        """Train until ``cfg['max_steps']`` global optimizer steps are reached.

        If :meth:`resume` was called beforehand, restores the checkpointed
        state into these objects first and continues from there. Returns a
        summary dict. ``cfg`` is a plain dict, loadable from
        ``configs/training/*.json`` via :func:`load_config`; callable entries
        (e.g. ``cfg['hooks']``, callable losses) are allowed but never come
        from JSON.
        """
        merged = dict(DEFAULT_CFG)
        merged.update(cfg or {})
        self.cfg = merged
        cfg = self.cfg
        self.max_steps = int(cfg["max_steps"])

        # opt-in CUDA mixed precision: cfg key "amp" (default False). When off,
        # _scaler stays None and _train_attempt takes the exact legacy path.
        self._amp = bool(cfg.get("amp", False)) and torch.cuda.is_available()
        if cfg.get("amp") and not self._amp:
            logger.warning("cfg['amp']=True requested but CUDA unavailable; "
                           "training in FP32")
        self._scaler = (torch.amp.GradScaler("cuda", enabled=True)
                        if self._amp else None)

        exp_id = str(cfg["exp_id"]).replace(os.sep, "-").strip(".") or "unnamed"
        self.out_dir = Path(cfg["experiment_root"]) / exp_id
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._model = model
        self._optimizer_ref = optimizer
        self._scheduler_ref = scheduler
        self._window = []
        run_id = cfg.get("run_id") or f"fit-{_now_iso()}"

        train_loss = _resolve_loss(cfg["loss_fn"])
        val_spec = cfg.get("val_loss_fn")
        if val_spec is not None:
            val_loss = _resolve_loss(val_spec)
        elif callable(cfg["loss_fn"]):
            val_loss = F.mse_loss  # never leak an instrumented train loss into val
        else:
            val_loss = train_loss
        self._val_loader_ref = val_loader
        self._val_loss_ref = val_loss

        hooks = cfg.get("hooks") or {}
        resumed = False

        self._metrics_fh = open(self.out_dir / "metrics.jsonl", "a",
                                encoding="utf-8", newline="\n")
        try:
            if self._pending_resume is not None:
                self._apply_resume()
                resumed = True
                # re-execute the val/checkpoint actions belonging to the
                # resume step exactly as an uninterrupted run did (keeps the
                # global RNG stream and all side effects aligned)
                self._replay_boundary_triggers(val_loader, run_id, hooks)
            elif cfg.get("reseed_on_fit", True) and cfg.get("seed") is not None:
                nt = cfg.get("num_threads")
                info = set_determinism(
                    int(cfg["seed"]),
                    deterministic_algorithms=bool(
                        cfg.get("deterministic_algorithms", True)),
                    num_threads=(int(nt) if nt else None))
                self._emit({"type": "event", "event": "seed",
                            "seed": info["seed"],
                            "num_threads": info["num_threads"]})

            self._t_fit_start = time.perf_counter()
            self._emit({"type": "event", "event": "fit_start", "run_id": run_id,
                        "resumed": resumed, "global_step": self.global_step,
                        "max_steps": self.max_steps})

            while self.global_step < self.max_steps:
                if self._iter is None:
                    self._open_epoch(train_loader)
                # replay-skip batches consumed before a mid-epoch interruption
                while self._skip_batches > 0 and self._iter is not None:
                    try:
                        next(self._iter)
                    except StopIteration:
                        self._close_epoch(model, optimizer, scheduler,
                                          val_loader, val_loss, run_id, hooks)
                        break
                    self._skip_batches -= 1
                    self.batches_consumed_in_epoch += 1
                if self._iter is None:
                    continue
                try:
                    batch = next(self._iter)
                except StopIteration:
                    self._close_epoch(model, optimizer, scheduler,
                                      val_loader, val_loss, run_id, hooks)
                    continue
                self.batches_consumed_in_epoch += 1

                skipped = self._train_attempt(batch, train_loss, hooks)
                if not skipped:
                    self._window.append(self._last_loss)
                self._after_attempt(run_id, hooks)

                if self.global_step >= self.max_steps:
                    break
        finally:
            if self._t_fit_start:
                self.wall_accum_s += time.perf_counter() - self._t_fit_start
                self._t_fit_start = 0.0
            self._finalize(val_loader, val_loss, run_id, hooks)

        return {
            "global_step": self.global_step,
            "epochs_completed": self.epoch,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "nan_events": self.nan_events,
            "wall_s": round(self.wall_accum_s, 3),
            "out_dir": str(self.out_dir),
        }

    def resume(self, path: os.PathLike | str) -> dict:
        """Load a checkpoint from disk; its state is applied to the objects
        passed to the next :meth:`fit` call. Nothing but file contents is used."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format_version "
                             f"{payload.get('format_version')!r} in {path}")
        self._pending_resume = payload
        meta = payload["trainer"]
        return {
            "resumed_from": str(path),
            "global_step": int(meta["global_step"]),
            "epoch": int(meta["epoch"]),
            "batches_consumed_in_epoch": int(meta["batches_consumed_in_epoch"]),
            "best_value": meta["best_value"],
            "best_step": meta["best_step"],
        }

    def validate(self, model, val_loader, loss_fn) -> dict:
        """Validation hook: eval-mode, no-grad pass; returns its metric dict."""
        was_training = model.training
        model.eval()
        totals, count = 0.0, 0
        cap = int(self.cfg.get("val_max_batches", 0) or 0)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if cap and i >= cap:
                    break
                x, y = _unpack_batch(batch)
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                totals += float(loss_fn(out, y))
                count += 1
        model.train(was_training)
        if count == 0:
            return {"val_loss": float("nan")}
        return {"val_loss": totals / count}

    def save_checkpoint(self, model, optimizer, scheduler,
                        name: str | int, val_metrics: dict | None = None) -> Path:
        """Atomically write ``last.pt`` / ``best.pt`` / ``step_%09d.pt``."""
        fname = f"step_{int(name):09d}.pt" if isinstance(name, int) else f"{name}.pt"
        payload = {
            "format_version": FORMAT_VERSION,
            "saved_at_utc": _now_iso(),
            "exp_id": self.cfg.get("exp_id"),
            "run_id": self.cfg.get("run_id"),
            "cfg": {k: v for k, v in self.cfg.items()
                    if k != "hooks" and _jsonable(v)},
            "trainer": {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "batches_consumed_in_epoch": self.batches_consumed_in_epoch,
                "best_value": self.best_value,
                "best_step": self.best_step,
                "last_val_step": self.last_val_step,
                "last_ckpt_step": self.last_ckpt_step,
                "wall_accum_s": self.wall_accum_s,
            },
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # rng captured right after the last completed optimizer step, so a
            # checkpoint written after a validation pass restores the exact
            # pre-aux-actions stream; data order is independent of this state
            "rng": self._last_rng if self._last_rng is not None else get_rng_state(),
            "val_metrics": val_metrics or {},
        }
        dest = self.checkpoint_dir / fname
        tmp = self.checkpoint_dir / f"{fname}.tmp-{os.getpid()}"
        try:
            with open(tmp, "wb") as fh:
                torch.save(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)  # atomic on POSIX and Windows
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    logger.warning("could not clean up temp file %s", tmp)
        if isinstance(name, int):
            self._rotate_step_checkpoints(int(self.cfg["keep_step_checkpoints"]))
        return dest

    # ------------------------------------------------------------ internal --

    def _open_epoch(self, train_loader) -> None:
        # data order depends only on (cfg seed, epoch index): rebuild the
        # loader with a dedicated generator instead of touching the global RNG
        gen = torch.Generator()
        gen.manual_seed(_derive_epoch_seed(int(self.cfg.get("seed") or 0),
                                           self.epoch))
        self._iter = iter(_clone_loader_with_generator(train_loader, gen))

    def _close_epoch(self, model, optimizer, scheduler, val_loader, val_loss,
                     run_id, hooks) -> None:
        self._iter = None
        self.batches_consumed_in_epoch = 0
        hook = hooks.get("on_epoch_end")
        if hook:
            hook(self)
        self.epoch += 1
        every = int(self.cfg.get("val_every_epochs") or 0)
        if (every and self.epoch % every == 0
                and self.global_step != self.last_val_step):
            self._do_validate(val_loader, val_loss, run_id, hooks)
        ck_every = int(self.cfg.get("checkpoint_every_epochs") or 0)
        if ck_every and self.epoch % ck_every == 0:
            self._save_rotating(run_id)

    def _train_attempt(self, batch, train_loss, hooks) -> bool:
        """One forward/backward/step attempt. Returns True if the optimizer
        step was SKIPPED due to non-finite numerics (weights untouched)."""
        model = self._model
        optimizer = self._optimizer_ref
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = _unpack_batch(batch)
        scaler = self._scaler
        if scaler is not None:
            # AMP (CUDA only; enabled solely via cfg["amp"]): fp16 autocast
            # around forward+loss. Input tensors keep their fp32 storage, and
            # the regressor's masked-loss reductions promote back to fp32 at
            # every boundary, so the loss stays an fp32 scalar.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = train_loss(out, y)
        else:
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = train_loss(out, y)
        loss_val = float(loss.detach())

        if not math.isfinite(loss_val):
            self.nan_events += 1
            self._emit({"type": "event", "event": "nan_loss_skipped",
                        "attempt_step": self.global_step + 1,
                        "epoch": self.epoch, "loss": loss_val})
            self._hook_step_end(hooks, skipped=True)
            return True

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if scaler is not None:
            scaler.unscale_(optimizer)   # in-place -> the check below sees unscaled grads
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            self.nan_events += 1
            self._emit({"type": "event", "event": "nonfinite_grad_skipped",
                        "attempt_step": self.global_step + 1,
                        "epoch": self.epoch})
            self._hook_step_end(hooks, skipped=True)
            if scaler is not None:
                scaler.update()  # close the scale iteration after unscale_
            return True

        clip = self.cfg.get("grad_clip")
        grad_norm: float | None = None
        if clip is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                max_norm=float(clip)))

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        self._scheduler_ref.step()
        self.global_step += 1
        self._last_loss = loss_val
        self._last_grad_norm = grad_norm
        # snapshot RNG immediately after the completed step
        self._last_rng = get_rng_state()
        self._hook_step_end(hooks, skipped=False)
        return False

    def _hook_step_end(self, hooks, skipped: bool) -> None:
        hook = hooks.get("on_step_end")
        if hook:
            hook(self, skipped)

    def _after_attempt(self, run_id, hooks) -> None:
        step = self.global_step
        log_every = int(self.cfg.get("log_every") or 0)
        if log_every and step % log_every == 0:
            avg = sum(self._window) / len(self._window) if self._window else None
            self._emit({"type": "train", "run_id": run_id, "step": step,
                        "epoch": self.epoch, "loss": self._last_loss,
                        "loss_window_avg": avg,
                        "lr": self._current_lr(),
                        "grad_norm": self._last_grad_norm,
                        "wall_s": round(time.perf_counter() - self._t_fit_start
                                        + self.wall_accum_s, 4)})
            self._window.clear()

        if (int(self.cfg.get("val_every_steps") or 0)
                and step % int(self.cfg["val_every_steps"]) == 0
                and step != self.last_val_step):
            self._do_validate(self._val_loader_ref, self._val_loss_ref,
                              run_id, hooks)

        if (int(self.cfg.get("checkpoint_every_steps") or 0)
                and step % int(self.cfg["checkpoint_every_steps"]) == 0
                and step != self.last_ckpt_step):
            self._save_rotating(run_id)

    def _do_validate(self, val_loader, val_loss, run_id, hooks) -> None:
        if val_loader is None:
            return
        metrics = self.validate(self._model, val_loader, val_loss)
        self.last_val_step = self.global_step
        best_improved = False
        value = metrics.get(self.cfg["best_key"])
        if value is not None and math.isfinite(float(value)):
            best_improved = self._is_better(float(value))
            if best_improved:
                self.best_value = float(value)
                self.best_step = self.global_step
        elif value is not None:
            self._emit({"type": "event", "event": "nonfinite_val_metric",
                        "step": self.global_step})
        row = {"type": "val", "run_id": run_id, "step": self.global_step,
               "epoch": self.epoch,
               "wall_s": round(time.perf_counter() - self._t_fit_start
                               + self.wall_accum_s, 4)}
        row.update(metrics)  # keys are already val_-prefixed by convention
        row["best_updated"] = bool(best_improved)
        self._emit(row)
        if best_improved:
            self.save_checkpoint(self._model, self._optimizer_ref,
                                 self._scheduler_ref, "best", val_metrics=metrics)
            self._emit({"type": "event", "event": "best_updated",
                        "step": self.global_step,
                        "best_key": self.cfg["best_key"],
                        "best_value": self.best_value})
        hook = hooks.get("on_validate")
        if hook:
            hook(self, metrics)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if str(self.cfg["best_mode"]).lower() == "max":
            return value > self.best_value
        return value < self.best_value

    def _save_rotating(self, run_id) -> None:
        self.last_ckpt_step = self.global_step
        path = self.save_checkpoint(self._model, self._optimizer_ref,
                                    self._scheduler_ref, int(self.global_step))
        self._link_last()
        self._emit({"type": "event", "event": "checkpoint_saved",
                    "step": self.global_step, "path": path.name})

    def _link_last(self) -> None:
        self.save_checkpoint(self._model, self._optimizer_ref,
                             self._scheduler_ref, "last")

    def _rotate_step_checkpoints(self, keep: int) -> None:
        keep = max(1, int(keep))
        ckpts = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in ckpts[:-keep]:
            try:
                old.unlink()
            except OSError:
                logger.warning("could not remove old checkpoint %s", old)

    def _apply_resume(self) -> None:
        payload = self._pending_resume
        self._pending_resume = None
        self._model.load_state_dict(payload["model"])
        self._optimizer_ref.load_state_dict(payload["optimizer"])
        self._scheduler_ref.load_state_dict(payload["scheduler"])
        # LambdaLR.load_state_dict does NOT touch param_group lrs; without this
        # a resumed run would silently train at the fresh-init (warmup-step-0)
        # lr instead of the schedule position at the checkpoint.
        for pg, lr in zip(self._optimizer_ref.param_groups,
                          self._scheduler_ref.get_last_lr()):
            pg["lr"] = lr
        meta = payload["trainer"]
        self.global_step = int(meta["global_step"])
        self.epoch = int(meta["epoch"])
        self.batches_consumed_in_epoch = int(meta["batches_consumed_in_epoch"])
        self.best_value = meta["best_value"]
        self.best_step = meta["best_step"]
        self.last_val_step = meta.get("last_val_step")
        self.last_ckpt_step = meta.get("last_ckpt_step")
        self.wall_accum_s = float(meta.get("wall_accum_s", 0.0))
        set_rng_state(payload["rng"])            # continue post-step stream
        self._last_rng = payload["rng"]
        self._skip_batches = self.batches_consumed_in_epoch
        self._emit({"type": "event", "event": "resume",
                    "global_step": self.global_step, "epoch": self.epoch,
                    "batches_consumed_in_epoch": self.batches_consumed_in_epoch,
                    "best_value": self.best_value})

    def _replay_boundary_triggers(self, val_loader, run_id, hooks) -> None:
        """Re-run the step-cadence val/checkpoint actions for the resume step.

        An uninterrupted run executes these as part of processing attempt N;
        a resumed run never re-processes attempt N, so without this replay its
        global-RNG stream (and best/last bookkeeping) drifts from the
        uninterrupted timeline.
        """
        step = self.global_step
        if (val_loader is not None and int(self.cfg.get("val_every_steps") or 0)
                and step % int(self.cfg["val_every_steps"]) == 0):
            self.last_val_step = None
            self._do_validate(val_loader, self._val_loss_ref, run_id, hooks)
        if (int(self.cfg.get("checkpoint_every_steps") or 0)
                and step % int(self.cfg["checkpoint_every_steps"]) == 0):
            self.last_ckpt_step = None
            self._save_rotating(run_id)

    def log_row(self, row: dict) -> None:
        """Public sink for experiment-specific metric rows (goes through the
        same dedupe/flush path as built-in rows into metrics.jsonl)."""
        self._emit(row)

    def _current_lr(self) -> float:
        try:
            return float(self._scheduler_ref.get_last_lr()[0])
        except Exception:
            return float(self._optimizer_ref.param_groups[0]["lr"])

    def _emit(self, row: dict) -> None:
        row.setdefault("ts", _now_iso())
        self.events.append(row)
        if self._metrics_fh is not None and not self._metrics_fh.closed:
            self._metrics_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._metrics_fh.flush()
        if row.get("type") == "event":
            logger.info("[%s] %s",
                        row.get("step", row.get("attempt_step")), row["event"])

    def _finalize(self, val_loader, val_loss, run_id, hooks) -> None:
        try:
            if (bool(self.cfg.get("val_at_end", True)) and val_loader is not None
                    and self.global_step > 0
                    and self.global_step != self.last_val_step):
                self._do_validate(val_loader, val_loss, run_id, hooks)
            if self.global_step > 0:
                if self.global_step != self.last_ckpt_step:
                    self._save_rotating(run_id)
                else:
                    self._link_last()
            self._emit({"type": "event", "event": "fit_end", "run_id": run_id,
                        "global_step": self.global_step})
        finally:
            if self._metrics_fh is not None and not self._metrics_fh.closed:
                self._metrics_fh.close()


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False
