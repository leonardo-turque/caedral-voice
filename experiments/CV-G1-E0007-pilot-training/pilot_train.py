"""CV-G1-E0007 — SMOKE-scale pilot training launcher (Phase J).

Wires the full pipeline end-to-end so the multi-hour run is one command:

    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0007-pilot-training/pilot_train.py \
        --manifests librittsr-smoke-10spk --config ../../configs/tiny.json \
        --steps 15000 --threads 4

(one command, cwd-independent, safe to launch detached via Start-Process /
nohup; all paths resolve from this file's location, stdout is UTF-8, no
console interaction, atomic checkpoints make hard kills safe.)

Pipeline per step:
    manifests -> PhonemeVocab (union chars, specials <pad>=0 |=1 ‖=2)
              -> speaker_index (sorted id prefixes)
              -> PilotDataset (resample->24 kHz log-Mel [80,T], memoized)
              -> stratified-by-speaker val split (--val-frac)
              -> LengthBucketBatchSampler (<=8 utts AND <=1600 mel frames)
              -> CAEDRAL-NAR tiny(use_flow=True) losses_for_training (E0006)
              -> Trainer (E0003: exact resume, NaN guard, warmup+cosine)

Optimizer: AdamW(lr 1e-4 peak, betas (0.9, 0.98), weight_decay 1e-6).
The beta pair follows the Glow-TTS/VITS lineage (stable under flow NLL);
weight decay is kept tiny because a CPU-only <=2 h-data pilot gains nothing
from real regularization and we do not want it fighting early loss descent.
Schedule: 400 linear-warmup steps then cosine to min_lr_ratio=0.1 (= 1e-5).
style_id = 0 for every row (G1-SMOKE has no style labels; documented
limitation -- row 0 is a learned style embedding, NOT the None-sentinel
neutral row of ConditioningProvider).

Resume: without --fresh, an existing checkpoints/last.pt is resumed
automatically (exactness guaranteed by E0003); data order is pinned to
(seed, epoch) via the sampler hooks below.
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

from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig  # noqa: E402
from caedral_voice.training.mel_dataset import (  # noqa: E402
    LengthBucketBatchSampler,
    PhonemeVocab,
    PilotDataset,
)
from caedral_voice.training.seed import set_determinism  # noqa: E402
from caedral_voice.training.trainer import Trainer, build_scheduler  # noqa: E402

DEFAULT_EXP_ID = "CV-G1-E0007-pilot-training"


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
def load_manifest_pair(name: str, manifests_dir: Path) -> tuple[list[dict], list[dict]]:
    """(audio rows, phoneme rows) for one dataset slice name."""
    apath = manifests_dir / f"{name}.jsonl"
    ppath = manifests_dir / f"{name}.phonemes.jsonl"
    for p in (apath, ppath):
        if not p.is_file():
            available = sorted(q.name for q in manifests_dir.glob("*.jsonl"))
            raise FileNotFoundError(
                f"manifest missing: {p}\navailable: {available}")
    def rd(p):
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    return rd(apath), rd(ppath)


def build_speaker_index(rows: list[dict]) -> dict[str, int]:
    prefixes = sorted({r["id"].split("_")[0] for r in rows})
    return {p: i for i, p in enumerate(prefixes)}


def stratified_split(dataset: PilotDataset, val_frac: float
                     ) -> tuple[list[int], list[int]]:
    """Deterministic per-speaker stride split: rows sorted by id inside each
    speaker, every round(1/val_frac)-th goes to val."""
    by_spk: dict[int, list[int]] = {}
    for i, row in enumerate(dataset.rows):
        by_spk.setdefault(row["speaker_id"], []).append(i)
    k = max(1, round(1.0 / float(val_frac))) if val_frac > 0 else 0
    train_idx: list[int] = []
    val_idx: list[int] = []
    for spk in sorted(by_spk):
        idxs = sorted(by_spk[spk],
                      key=lambda i: (dataset.rows[i]["id"], i))
        for pos, i in enumerate(idxs):
            (val_idx if (k and pos % k == 0) else train_idx).append(i)
    assert not (set(train_idx) & set(val_idx))
    return sorted(train_idx), sorted(val_idx)


def collate_pilot(batch: list[dict]) -> dict:
    """Pad right with vocab pad id 0 / mel 0.0; lengths mark valid extents."""
    tx = max(len(it["phoneme_ids"]) for it in batch)
    ty = max(it["T_mel"] for it in batch)
    b = len(batch)
    ids = torch.zeros(b, tx, dtype=torch.long)
    mel = torch.zeros(b, ty, batch[0]["mel"].shape[0], dtype=torch.float32)
    xlen = torch.zeros(b, dtype=torch.long)
    ylen = torch.zeros(b, dtype=torch.long)
    spk = torch.zeros(b, dtype=torch.long)
    for j, it in enumerate(batch):
        n_x = len(it["phoneme_ids"])
        n_y = it["T_mel"]
        ids[j, :n_x] = torch.tensor(it["phoneme_ids"], dtype=torch.long)
        mel[j, :n_y, :] = it["mel"].t()                # [80, T] -> [T, 80]
        xlen[j], ylen[j], spk[j] = n_x, n_y, it["speaker_id"]
    return {
        "x": {
            "mel": mel,
            "phoneme_ids": ids,
            "speaker_ids": spk,
            "mel_lengths": ylen,
            "phoneme_lengths": xlen,
        },
        "y": mel,  # unpacked by Trainer; unused by the pilot loss callables
    }


class PilotTrainModule(torch.nn.Module):
    """Adapts CaedralNAR.losses_for_training(**x) to the Trainer interface
    ``out = model(x); loss = loss_fn(out, y)`` and stashes per-batch
    diagnostics (MAS durations -> phones/s rate) for metric logging."""

    def __init__(self, model: CaedralNAR, sample_rate: int, hop: int):
        super().__init__()
        self.model = model
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.last_diag: dict = {}
        self.val_buffer: list[tuple[float, float]] = []

    def forward(self, inputs: dict) -> dict:
        out = self.model.losses_for_training(
            mel=inputs["mel"],
            phoneme_ids=inputs["phoneme_ids"],
            speaker_id=inputs["speaker_ids"],
            style_id=0,
            mel_lengths=inputs["mel_lengths"],
            phoneme_lengths=inputs["phoneme_lengths"],
        )
        xl = int(inputs["phoneme_lengths"].sum())
        yl = int(inputs["mel_lengths"].sum())
        seconds = yl * self.hop / self.sample_rate
        self.last_diag = {
            "flow_loss": float(out["flow_loss"]),
            "dur_loss": float(out["duration_loss"]),
            "phones_per_sec": (xl / seconds) if seconds > 0 else float("nan"),
            "n_phones": xl,
            "n_frames": yl,
        }
        if not self.training:
            self.val_buffer.append((self.last_diag["flow_loss"],
                                    self.last_diag["dur_loss"]))
        return out


# --------------------------------------------------------------------------- #
# script
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifests", type=str, default="librittsr-smoke-10spk",
                   help="comma-separated dataset names under data/manifests/")
    p.add_argument("--config", type=str,
                   default="../../configs/tiny.json",
                   help="model config JSON (resolved against cwd, experiment "
                        "dir, then repo root)")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="cosine floor as fraction of peak lr")
    p.add_argument("--max-frames", type=int, default=1600)
    p.add_argument("--max-utts", type=int, default=8)
    p.add_argument("--bucket", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--exp-id", type=str, default=DEFAULT_EXP_ID,
                   help="subdir under experiments/ ('/' nests inside the "
                        "default experiment dir)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoints/last.pt")
    p.add_argument("--resume-from", type=str, default=None,
                   help="explicit checkpoint path (default: auto last.pt)")
    return p.parse_args(argv)


def resolve_config_path(spec: str) -> Path:
    cand = [Path(spec), REPO / "experiments" / DEFAULT_EXP_ID / spec,
            REPO / spec]
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
    print(f"[pilot] threads={torch.get_num_threads()} "
          f"torch={torch.__version__} python={platform.python_version()}",
          flush=True)

    manifests_dir = REPO / "data" / "manifests"
    names = [s.strip() for s in args.manifests.split(",") if s.strip()]
    audio_rows: list[dict] = []
    phone_rows: list[dict] = []
    for name in names:
        a, ph = load_manifest_pair(name, manifests_dir)
        print(f"[data] {name}: {len(a)} utt "
              f"({sum(r['duration_s'] for r in a)/3600:.2f} h)", flush=True)
        audio_rows += a
        phone_rows += ph

    speaker_index = build_speaker_index(audio_rows)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone_rows])
    print(f"[data] speakers={len(speaker_index)} vocab_size={vocab.size}",
          flush=True)

    dataset = PilotDataset(
        audio_rows, phone_rows, speaker_index,
        sample_rate=24000, n_mels=80, hop=256, vocab=vocab, repo_root=REPO,
    )
    if dataset.skipped:
        preview = "; ".join(f"{i}: {why}" for i, why in dataset.skipped[:5])
        print(f"[data] SKIPPED {len(dataset.skipped)} rows ({preview} ...)",
              flush=True)
    print(f"[data] usable utterances={len(dataset)}", flush=True)
    total_h = sum(r["duration_s"] for r in dataset.rows) / 3600
    print(f"[data] usable audio={total_h:.3f} h", flush=True)

    train_idx, val_idx = stratified_split(dataset, args.val_frac)
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
    train_loader = DataLoader(dataset, batch_sampler=train_sampler,
                              collate_fn=collate_pilot, num_workers=0)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler,
                            collate_fn=collate_pilot, num_workers=0)

    cfg_path = resolve_config_path(args.config)
    model_cfg = CaedralNARConfig.from_json(cfg_path)
    model_cfg.use_flow = True
    if model_cfg.vocab_size < vocab.size:
        raise ValueError(
            f"model vocab_size {model_cfg.vocab_size} < corpus vocab "
            f"{vocab.size}; regenerate configs or shrink the corpus")
    if model_cfg.n_speakers < len(speaker_index):
        raise ValueError(
            f"model n_speakers {model_cfg.n_speakers} < "
            f"{len(speaker_index)} speakers")
    model_cfg.n_speakers = max(model_cfg.n_speakers, len(speaker_index))
    model = CaedralNAR(model_cfg)
    module = PilotTrainModule(model, sample_rate=model_cfg.sample_rate,
                              hop=model_cfg.hop_length)
    print(f"[model] {cfg_path.name}(use_flow=True) params="
          f"{module.model.count_params():,}", flush=True)

    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98),
                                  weight_decay=args.weight_decay)
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
            t.log_row({"type": "pilot_train", "step": t.global_step,
                       "flow_loss": d.get("flow_loss"),
                       "dur_loss": d.get("dur_loss"),
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
            nf = sum(f for f, _ in buf) / len(buf)
            nd = sum(d for _, d in buf) / len(buf)
            t.log_row({"type": "pilot_val", "step": t.global_step,
                       "flow_loss_mean": nf, "dur_loss_mean": nd,
                       "batches": len(buf)})

    cfg = {
        "exp_id": args.exp_id,
        "experiment_root": str(exp_root),
        "run_id": f"pilot-{time.strftime('%Y%m%d-%H%M%S')}",
        "seed": args.seed,
        "num_threads": args.threads,
        "reseed_on_fit": True,
        "deterministic_algorithms": True,
        "max_steps": args.steps,
        "warmup_steps": args.warmup,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_clip": args.grad_clip,
        "loss_fn": lambda out, _y: out["loss"],
        "val_loss_fn": lambda out, _y: out["flow_loss"],
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

    # ---- resume handling ------------------------------------------------ #
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

    # ---- side-car artifacts --------------------------------------------- #
    ckpt_dir_hint.mkdir(parents=True, exist_ok=True)
    vocab.to_json(ckpt_dir_hint / "phoneme_vocab.json")
    (ckpt_dir_hint / "speaker_index.json").write_text(
        json.dumps(speaker_index, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "exp_id": args.exp_id,
        "run_id": cfg["run_id"],
        "resumed_from": (resume_info or {}).get("resumed_from"),
        "manifests": names,
        "usable_utterances": len(dataset),
        "train_utterances": len(train_idx),
        "val_utterances": len(val_idx),
        "skipped_rows": len(dataset.skipped),
        "total_audio_h": round(total_h, 4),
        "vocab_size": vocab.size,
        "speakers": len(speaker_index),
        "model_config": json.loads(json.dumps(
            {k: getattr(model_cfg, k) for k in model_cfg.__dataclass_fields__})),
        "optimizer": {"kind": "AdamW", "lr": args.lr,
                      "betas": [0.9, 0.98],
                      "weight_decay": args.weight_decay},
        "schedule": {"warmup_steps": args.warmup,
                     "min_lr_ratio": args.min_lr_ratio},
        "batching": {"max_utts": args.max_utts,
                     "max_frames": args.max_frames,
                     "bucket_size": args.bucket},
        "args": {k: v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "machine": platform.platform()},
    }
    (exp_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    summary = trainer.fit(module, optimizer, scheduler, train_loader,
                          val_loader, cfg)
    summary["wall_total_s"] = round(time.perf_counter() - t0, 2)
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
