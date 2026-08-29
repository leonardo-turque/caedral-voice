"""CV-G1-E0013 — supervised-duration mel-regressor training launcher (B-arm).

Wires the full pipeline end-to-end so the real run is ONE command once the
ALIGN-01 durations land:

    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0013-regressor/train_regressor.py \
        --manifests librittsr-clean100,teacher-kokoro \
        --durations data/manifests/librittsr-clean100.durations.jsonl \
        --steps <N> --threads 4 --exp-id CV-G1-E0013-regressor

Until ALIGN-01 lands, --synthetic-durations runs the identical pipeline on
uniform per-phoneme duration splits (smoke/validation only; documented).

Pipeline per step:
    manifests (+optional teacher manifest, phonemes resolved via source_id)
      -> PhonemeVocab (union chars, specials <pad>=0 |=1 ||=2)
      -> speaker slots (sorted prefixes; teachers tNN / tvNN after reals;
         table sized max(40, corpus))
      -> RegressorDataset (24 kHz log-Mel [80,T], memoized; GT durations or
         synthetic uniform splits)
      -> stratified-by-speaker val split (--val-frac)
      -> LengthBucketBatchSampler (<=16 utts AND <=2000 est frames)
      -> CaedralRegressor.losses_for_training
         (masked log-mel L1 + spectral convergence + supervised duration MSE;
          pitch/energy predictors frozen-off in v0 -- see regressor.py)
      -> Trainer (E0003: exact resume, NaN guard, warmup+cosine, atomic ckpts)

Optimizer: AdamW(lr peak 1e-4, betas (0.9, 0.98), weight_decay 1e-6);
schedule: linear warmup then cosine to min_lr_ratio of peak.
Resume: without --fresh, an existing checkpoints/last.pt resumes exactly
(E0003 guarantees bitwise continuation); data order pinned to (seed, epoch).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from caedral_voice.model.regressor import (  # noqa: E402
    CaedralRegressor,
    CaedralRegressorConfig,
)
from caedral_voice.training.mel_dataset import (  # noqa: E402
    LengthBucketBatchSampler,
    PhonemeVocab,
)
from caedral_voice.training.mel_cache import MelDiskCache  # noqa: E402
from caedral_voice.training.regressor_dataset import (  # noqa: E402
    RegressorDataset,
    build_speaker_index,
    collate_regressor,
    load_jsonl,
    stratified_split,
)
from caedral_voice.training.seed import set_determinism  # noqa: E402
from caedral_voice.training.trainer import Trainer, build_scheduler  # noqa: E402

DEFAULT_EXP_ID = "CV-G1-E0013-regressor"
DEFAULT_DURATIONS = "data/manifests/librittsr-clean100.durations.jsonl"


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
def _to_device(obj, device):
    """Recursively move tensors to ``device`` (dicts/lists/tuples traversed;
    non-tensors returned unchanged). CPU->CPU .to() returns the same objects,
    so the default local path is untouched."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def load_manifest_pair(name: str, manifests_dir: Path
                       ) -> tuple[list[dict], list[dict]]:
    """(audio rows, phoneme rows) for one slice name. The phonemes file is
    OPTIONAL (teacher manifests resolve phonemes through source_id into the
    text-source corpus's map instead)."""
    apath = manifests_dir / f"{name}.jsonl"
    ppath = manifests_dir / f"{name}.phonemes.jsonl"
    if not apath.is_file():
        available = sorted(q.name for q in manifests_dir.glob("*.jsonl"))
        raise FileNotFoundError(
            f"manifest missing: {apath}\navailable: {available}")
    phone_rows = load_jsonl(ppath) if ppath.is_file() else []
    return load_jsonl(apath), phone_rows


class RegressorTrainModule(torch.nn.Module):
    """Adapts CaedralRegressor.losses_for_training(**x) to the Trainer
    interface ``out = model(x); loss = loss_fn(out, y)`` and stashes per-batch
    diagnostics for metric logging (mirrors E0007's PilotTrainModule)."""

    def __init__(self, model: CaedralRegressor, sample_rate: int, hop: int):
        super().__init__()
        self.model = model
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.last_diag: dict = {}
        self.val_buffer: list[tuple[float, float]] = []
        self._param_device: torch.device | None = None

    def forward(self, inputs: dict) -> dict:
        if self._param_device is None:
            self._param_device = next(self.model.parameters()).device
        inputs = _to_device(inputs, self._param_device)
        out = self.model.losses_for_training(
            mel=inputs["mel"],
            phoneme_ids=inputs["phoneme_ids"],
            durations=inputs["duration_targets"],
            speaker_id=inputs["speaker_ids"],
            style_id=0,
            mel_lengths=inputs["mel_lengths"],
            phoneme_lengths=inputs["phoneme_lengths"],
        )
        xl = int(inputs["phoneme_lengths"].sum())
        yl = int(inputs["mel_lengths"].sum())
        seconds = yl * self.hop / self.sample_rate
        self.last_diag = {
            "mel_loss": float(out["mel_loss"]),
            "spec_loss": float(out["spec_loss"]),
            "dur_loss": float(out["dur_loss"]),
            "n_duration_adjusted": int(out["n_duration_adjusted"]),
            "phones_per_sec": (xl / seconds) if seconds > 0 else float("nan"),
            "n_phones": xl,
            "n_frames": yl,
        }
        if not self.training:
            self.val_buffer.append((self.last_diag["mel_loss"],
                                    self.last_diag["dur_loss"]))
        return out


# --------------------------------------------------------------------------- #
# script
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifests", type=str, default="librittsr-clean100",
                   help="comma-separated dataset names under data/manifests/ "
                        "(e.g. librittsr-clean100,teacher-kokoro)")
    p.add_argument("--config", type=str, default=None,
                   help="optional model config JSON overriding the built-in "
                        "~15M defaults (resolved against cwd then repo root)")
    p.add_argument("--durations", type=str, default=DEFAULT_DURATIONS,
                   help="ALIGN-01 durations jsonl {id,durations,total_frames}; "
                        "ids must match manifest ids")
    p.add_argument("--synthetic-durations", action="store_true",
                   help="skip the durations file: split decoded T_mel "
                        "uniformly across phonemes (smoke mode only)")
    p.add_argument("--mel-cache-dir", type=str, default=None,
                   help="E0018 opt-in: persist log-Mels under this directory "
                        "(default layout data/cache/mel-v1) so later launches "
                        "skip the ~25-35 min mel preprocessing; omitted = "
                        "exactly the pre-E0018 in-RAM-only behavior")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="cosine floor as fraction of peak lr")
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--max-utts", type=int, default=16)
    p.add_argument("--bucket", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--spec-loss-weight", type=float, default=-1.0,
                   help="<0 keeps the config default")
    p.add_argument("--dur-loss-weight", type=float, default=-1.0,
                   help="<0 keeps the config default")
    p.add_argument("--train-pitch", action="store_true",
                   help="unfreeze the pitch predictor (requires GT F0 tracks;"
                        " OFF by default in v0 -- see model/regressor.py)")
    p.add_argument("--train-energy", action="store_true",
                   help="train the energy predictor on mel-derived dB tracks"
                        " (implemented; OFF by default in v0)")
    p.add_argument("--exp-id", type=str, default=DEFAULT_EXP_ID,
                   help="subdir under experiments/ ('/' nests inside the "
                        "default experiment dir)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoints/last.pt")
    p.add_argument("--resume-from", type=str, default=None,
                   help="explicit checkpoint path (default: auto last.pt)")
    p.add_argument("--device", type=str, default=None,
                   help="torch device override (default: cuda when available, "
                        "else cpu -- pre-colab-r3 behavior on this machine "
                        "since it has no CUDA device)")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader worker processes (-1 = auto: 2 with cuda "
                        "and pinned memory, else 0 = legacy behavior)")
    p.add_argument("--amp", action="store_true",
                   help="CUDA mixed precision (fp16 autocast + GradScaler); "
                        "ignored on CPU; enable only after the bootstrap AMP "
                        "sanity gate passes")
    return p.parse_args(argv)


def resolve_config_path(spec: str) -> Path:
    cand = [Path(spec), REPO / spec]
    for c in cand:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"--config not found: {spec}")


def main(argv=None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    t0 = time.perf_counter()

    set_determinism(args.seed, deterministic_algorithms=True,
                    num_threads=args.threads)
    print(f"[regressor] threads={torch.get_num_threads()} "
          f"torch={torch.__version__} python={platform.python_version()}",
          flush=True)

    # ---- device (colab-r3: opt-in; unchanged on this CPU-only machine) ----#
    if args.device:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[model] WARNING --device cuda requested but CUDA is "
                  "unavailable; falling back to cpu", flush=True)
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = bool(args.amp) and device.type == "cuda"
    if args.amp and not amp_on:
        print("[model] --amp ignored (no CUDA device)", flush=True)

    # ---- durations source ------------------------------------------------ #
    if args.synthetic_durations:
        durations_rows = None
        durations_desc = "synthetic-uniform (pre-ALIGN smoke mode)"
    else:
        dpath = Path(args.durations)
        if not dpath.is_absolute():
            dpath = REPO / dpath
        if not dpath.is_file():
            raise FileNotFoundError(
                f"durations file missing: {dpath}\n"
                "ALIGN-01 has not landed yet; for a smoke run add "
                "--synthetic-durations")
        durations_rows = load_jsonl(dpath)
        durations_desc = f"{dpath} ({len(durations_rows)} rows)"
    print(f"[data] durations: {durations_desc}", flush=True)

    # ---- manifests --------------------------------------------------------#
    manifests_dir = REPO / "data" / "manifests"
    names = [s.strip() for s in args.manifests.split(",") if s.strip()]
    audio_rows: list[dict] = []
    phone_rows: list[dict] = []
    for name in names:
        a, ph = load_manifest_pair(name, manifests_dir)
        hours = sum(r["duration_s"] for r in a) / 3600
        n_teach = sum(1 for r in a if "teacher_voice" in r)
        print(f"[data] {name}: {len(a)} utt ({hours:.2f} h"
              f"{f', {n_teach} teacher' if n_teach else ''})", flush=True)
        audio_rows += a
        phone_rows += ph
    if not phone_rows:
        raise ValueError(
            "no phoneme rows loaded; teacher-only manifests must be paired "
            "with their text-source corpus (e.g. "
            "--manifests librittsr-clean100,teacher-kokoro)")

    speaker_index = build_speaker_index(audio_rows)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone_rows])
    print(f"[data] speakers={len(speaker_index)} vocab_size={vocab.size}",
          flush=True)

    mel_disk_cache = None
    if args.mel_cache_dir:
        mel_disk_cache = MelDiskCache(args.mel_cache_dir,
                                      sample_rate=24000, n_mels=80, hop=256,
                                      n_fft=1024)
        print(f"[data] mel disk cache ON -> {mel_disk_cache.root} "
              f"(v{mel_disk_cache.version})", flush=True)
    dataset = RegressorDataset(
        audio_rows, phone_rows, durations_rows, speaker_index,
        sample_rate=24000, n_mels=80, hop=256, vocab=vocab, repo_root=REPO,
        synthetic_uniform_durations=args.synthetic_durations,
        disk_cache=mel_disk_cache,
    )
    if dataset.skipped:
        preview = "; ".join(f"{i}: {why}" for i, why in dataset.skipped[:5])
        print(f"[data] SKIPPED {len(dataset.skipped)} rows ({preview} ...)",
              flush=True)
    total_h = sum(r["duration_s"] for r in dataset.rows) / 3600
    print(f"[data] usable utterances={len(dataset)} audio={total_h:.3f} h",
          flush=True)

    train_idx, val_idx = stratified_split(dataset.rows, args.val_frac)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} "
          f"(stratified by speaker, stride {max(1, round(1/args.val_frac))})",
          flush=True)

    train_sampler = LengthBucketBatchSampler(
        train_idx, dataset.est_frames, max_frames=args.max_frames,
        max_utts=args.max_utts, bucket_size=args.bucket,
        shuffle=True, base_seed=args.seed)
    val_sampler = LengthBucketBatchSampler(
        val_idx, dataset.est_frames, max_frames=args.max_frames,
        max_utts=args.max_utts, bucket_size=args.bucket,
        shuffle=False, base_seed=args.seed)
    loader_workers = (args.num_workers if args.num_workers >= 0
                      else (2 if device.type == "cuda" else 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(dataset, batch_sampler=train_sampler,
                              collate_fn=collate_regressor,
                              num_workers=loader_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler,
                            collate_fn=collate_regressor,
                            num_workers=loader_workers,
                            pin_memory=pin_memory)

    # ---- model ------------------------------------------------------------#
    cfg_path = resolve_config_path(args.config) if args.config else None
    model_cfg = (CaedralRegressorConfig.from_json(cfg_path) if cfg_path
                 else CaedralRegressorConfig())
    if model_cfg.vocab_size < vocab.size:
        raise ValueError(
            f"model vocab_size {model_cfg.vocab_size} < corpus vocab "
            f"{vocab.size}; pass --config or shrink the corpus")
    if model_cfg.n_speakers < len(speaker_index):
        print(f"[model] expanding speaker table {model_cfg.n_speakers} -> "
              f"{len(speaker_index)} (corpus exceeds the 40-slot floor)",
              flush=True)
    model_cfg.n_speakers = max(model_cfg.n_speakers, len(speaker_index))
    if args.train_pitch:
        model_cfg.train_pitch = True
    if args.train_energy:
        model_cfg.train_energy = True
    if args.spec_loss_weight >= 0:
        model_cfg.spec_loss_weight = float(args.spec_loss_weight)
    if args.dur_loss_weight >= 0:
        model_cfg.dur_loss_weight = float(args.dur_loss_weight)

    model = CaedralRegressor(model_cfg)
    module = RegressorTrainModule(model, sample_rate=model_cfg.sample_rate,
                                  hop=model_cfg.hop_length)
    module.to(device)
    print(f"[model] {model_cfg.name} params={module.model.count_params():,} "
          f"(trainable {module.model.count_params(trainable_only=True):,}) "
          f"pitch={'on' if model.use_pitch else 'off'} "
          f"energy={'on' if model.use_energy else 'off'}", flush=True)
    print(f"[model] device={device} amp={amp_on} "
          f"dataloader_workers={loader_workers} "
          f"pin_memory={pin_memory}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    sched_cfg = {"warmup_steps": args.warmup,
                 "max_steps": args.steps,
                 "min_lr_ratio": args.min_lr_ratio}
    scheduler = build_scheduler(optimizer, sched_cfg)

    trainer = Trainer()
    exp_root = REPO / "experiments"
    exp_dir = exp_root / args.exp_id
    ckpt_dir_hint = exp_dir / "checkpoints"

    def on_step_end(t: Trainer, skipped: bool) -> None:
        if skipped:
            return
        d = module.last_diag
        if t.global_step % max(1, args.log_every) == 0:
            t.log_row({"type": "regressor_train", "step": t.global_step,
                       "mel_loss": d.get("mel_loss"),
                       "spec_loss": d.get("spec_loss"),
                       "dur_loss": d.get("dur_loss"),
                       "n_duration_adjusted": d.get("n_duration_adjusted"),
                       "phones_per_sec": d.get("phones_per_sec"),
                       "n_phones": d.get("n_phones"),
                       "n_frames": d.get("n_frames")})

    def on_epoch_end(t: Trainer) -> None:
        # fires inside _close_epoch BEFORE t.epoch increments: pin the NEXT
        # epoch's data order so resume reproduces batches exactly (E0003)
        train_sampler.set_epoch(t.epoch + 1)

    def on_validate(t: Trainer, metrics: dict) -> None:
        buf = module.val_buffer
        module.val_buffer = []
        if buf:
            nm = sum(m for m, _ in buf) / len(buf)
            nd = sum(d for _, d in buf) / len(buf)
            t.log_row({"type": "regressor_val", "step": t.global_step,
                       "mel_l1_mean": nm, "dur_loss_mean": nd,
                       "batches": len(buf)})

    cfg = {
        "exp_id": args.exp_id,
        "experiment_root": str(exp_root),
        "run_id": f"regressor-{time.strftime('%Y%m%d-%H%M%S')}",
        "seed": args.seed,
        "num_threads": args.threads,
        "reseed_on_fit": True,
        "deterministic_algorithms": True,
        "max_steps": args.steps,
        "warmup_steps": args.warmup,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_clip": args.grad_clip,
        "amp": amp_on,
        "loss_fn": lambda out, _y: out["loss"],
        "val_loss_fn": lambda out, _y: out["mel_loss"],
        "log_every": args.log_every,
        "val_every_steps": args.val_every,
        "val_at_end": True,
        "checkpoint_every_steps": args.ckpt_every,
        "keep_step_checkpoints": 2,
        "best_key": "val_loss",
        "best_mode": "min",
        "hooks": {"on_step_end": on_step_end,
                  "on_epoch_end": on_epoch_end,
                  "on_validate": on_validate},
    }

    # ---- resume handling --------------------------------------------------#
    resume_info = None
    explicit = Path(args.resume_from) if args.resume_from else None
    auto = ckpt_dir_hint / "last.pt"
    if not args.fresh:
        target = explicit if explicit is not None else (
            auto if auto.is_file() else None)
        if target is not None and Path(target).is_file():
            resume_info = trainer.resume(target)
            train_sampler.set_epoch(resume_info["epoch"])
            print(f"[resume] {target} -> step "
                  f"{resume_info['global_step']} epoch "
                  f"{resume_info['epoch']} "
                  f"(batches {resume_info['batches_consumed_in_epoch']})",
                  flush=True)

    # ---- side-car artifacts -----------------------------------------------#
    ckpt_dir_hint.mkdir(parents=True, exist_ok=True)
    vocab.to_json(ckpt_dir_hint / "phoneme_vocab.json")
    (ckpt_dir_hint / "speaker_index.json").write_text(
        json.dumps(speaker_index, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "exp_id": args.exp_id,
        "run_id": cfg["run_id"],
        "resumed_from": (resume_info or {}).get("resumed_from"),
        "manifests": names,
        "durations_source": durations_desc,
        "synthetic_durations": bool(args.synthetic_durations),
        "usable_utterances": len(dataset),
        "train_utterances": len(train_idx),
        "val_utterances": len(val_idx),
        "skipped_rows": len(dataset.skipped),
        "total_audio_h": round(total_h, 4),
        "vocab_size": vocab.size,
        "speakers": len(speaker_index),
        "params_total": module.model.count_params(),
        "params_trainable": module.model.count_params(trainable_only=True),
        "param_breakdown": module.model.param_breakdown(),
        "model_config": json.loads(json.dumps(
            {k: getattr(model_cfg, k) for k in model_cfg.__dataclass_fields__})),
        "optimizer": {"kind": "AdamW", "lr": args.lr, "betas": [0.9, 0.98],
                      "weight_decay": args.weight_decay},
        "schedule": {"warmup_steps": args.warmup,
                     "min_lr_ratio": args.min_lr_ratio},
        "batching": {"max_utts": args.max_utts, "max_frames": args.max_frames,
                     "bucket_size": args.bucket},
        "args": {k: v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "machine": platform.platform(),
                     "device": str(device),
                     "amp": amp_on},
    }
    (exp_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    summary = trainer.fit(module, optimizer, scheduler, train_loader,
                          val_loader, cfg)
    summary["wall_total_s"] = round(time.perf_counter() - t0, 2)
    if mel_disk_cache is not None:
        summary["mel_cache_stats"] = dict(mel_disk_cache.stats)
        print(f"[melcache] final stats: {summary['mel_cache_stats']}",
              flush=True)
    print("[done] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        try:
            sys.stderr.flush()
        except Exception:
            pass
        raise SystemExit(1)
