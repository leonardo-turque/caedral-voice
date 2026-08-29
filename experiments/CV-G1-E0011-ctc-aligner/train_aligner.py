"""CV-G1-E0011 — supervised CTC aligner training (founder pivot after DIAG-02).

Trains ``caedral_voice/model/ctc_aligner.py`` (log-mel -> conv /2 time ->
conv-transformer -> CTC over PhonemeVocab + appended blank) on
librittsr-clean100 so that Viterbi forced alignment replaces MAS durations
for the FastSpeech2-style mel-regressor stage.

    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0011-ctc-aligner/train_aligner.py --threads 6

cwd-independent; safe detached (Start-Process): UTF-8 stdout, no console
interaction, atomic checkpoints, ``metrics.jsonl`` + ``run_status.json``
(written in a ``finally`` so crashes always land there), auto-resume from
``checkpoints/last.pt`` unless ``--fresh``.

Spec pins: AdamW peak lr 3e-4 (linear warmup then cosine to 10%), <=16 utts
AND <=1600 estimated mel frames per length-bucketed batch, 6 threads, 15
epochs target with early stop when val token error rate plateaus
(no improvement >= ``--min-delta`` for ``--patience`` evals). Best checkpoint
selected by val token-ER (tie-break: lower val loss-per-token).

Mel pipeline is byte-identical to the established pilot stack (mel_dataset.
PilotDataset): resample to 24 kHz, 80-band power=1 mel, hop 256, natural log,
floor 1e-5. Mels are memoized as float16 (~2.3 GB for 44 h) to fit RAM;
collate casts back to float32.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from caedral_voice.model.ctc_aligner import (  # noqa: E402
    TIME_SUBSAMPLE,
    CTCAligner,
    CTCAlignerConfig,
    min_ctc_frames,
)
from caedral_voice.training.mel_dataset import (  # noqa: E402
    LengthBucketBatchSampler,
    PhonemeVocab,
    PilotDataset,
)
from caedral_voice.training.seed import set_determinism  # noqa: E402

DEFAULT_EXP_ID = "CV-G1-E0011-ctc-aligner"
EXP_DIR = REPO / "experiments" / DEFAULT_EXP_ID
MEL_KW = {"sample_rate": 24000, "n_mels": 80, "hop": 256, "n_fft": 1024}
#: extra frames required beyond the minimal CTC path (symbols + blanks
#: between adjacent-different ids). 0 = pure feasibility on conservative
#: frame estimates; ctc_loss(zero_infinity=True) is the final guard.
FEASIBILITY_MARGIN = 0


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
def load_manifest_pair(name: str, manifests_dir: Path):
    """(audio rows, phoneme rows) for one dataset slice name."""
    apath = manifests_dir / f"{name}.jsonl"
    ppath = manifests_dir / f"{name}.phonemes.jsonl"
    for p in (apath, ppath):
        if not p.is_file():
            raise FileNotFoundError(f"manifest missing: {p}")

    def rd(p):
        return [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rd(apath), rd(ppath)


def build_speaker_index(rows: list[dict]) -> dict[str, int]:
    prefixes = sorted({r["id"].split("_")[0] for r in rows})
    return {p: i for i, p in enumerate(prefixes)}


class AlignerDataset(PilotDataset):
    """PilotDataset whose mel cache downcasts to float16 after first use."""

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        t = self._mel_cache.get(index)
        if t is not None and t.dtype != torch.float16:
            self._mel_cache[index] = t.to(torch.float16)
        return item


def collate_aligner(batch: list[dict]) -> dict:
    """Right-pad phoneme ids with 0 and mels with 0.0; carry lengths."""
    tx = max(len(it["phoneme_ids"]) for it in batch)
    ty = max(it["T_mel"] for it in batch)
    b = len(batch)
    n_mels = batch[0]["mel"].shape[0]
    ids = torch.zeros(b, tx, dtype=torch.long)
    mel = torch.zeros(b, ty, n_mels, dtype=torch.float32)
    xlen = torch.zeros(b, dtype=torch.long)
    ylen = torch.zeros(b, dtype=torch.long)
    for j, it in enumerate(batch):
        nx, ny = len(it["phoneme_ids"]), it["T_mel"]
        ids[j, :nx] = torch.tensor(it["phoneme_ids"], dtype=torch.long)
        mel[j, :ny, :] = it["mel"].t().float()     # [80, T] -> [T, 80]
        xlen[j], ylen[j] = nx, ny
    return {"phoneme_ids": ids, "phoneme_lengths": xlen,
            "mel": mel, "mel_lengths": ylen}


def ctc_feasible_indices(dataset: PilotDataset) -> list[int]:
    """Rows whose subsampled frame count can cover the MINIMAL CTC path
    (symbols + blanks between adjacent-different ids; see
    ``ctc_aligner.min_ctc_frames``)."""
    out = []
    for i in range(len(dataset)):
        t_ds = -(-dataset.est_frames(i) // TIME_SUBSAMPLE)   # ceil
        if t_ds >= min_ctc_frames(dataset.rows[i]["ids"]) + FEASIBILITY_MARGIN:
            out.append(i)
    return out


def stratified_split(dataset: PilotDataset, indices: list[int], val_frac: float
                     ) -> tuple[list[int], list[int]]:
    """Per-speaker stride split (rows sorted by id within speaker)."""
    by_spk: dict[int, list[int]] = {}
    for i in indices:
        by_spk.setdefault(dataset.rows[i]["speaker_id"], []).append(i)
    k = max(1, round(1.0 / float(val_frac))) if val_frac > 0 else 0
    train_idx: list[int] = []
    val_idx: list[int] = []
    for spk in sorted(by_spk):
        idxs = sorted(by_spk[spk], key=lambda i: (dataset.rows[i]["id"], i))
        for pos, i in enumerate(idxs):
            (val_idx if (k and pos % k == 0) else train_idx).append(i)
    assert not (set(train_idx) & set(val_idx))
    return sorted(train_idx), sorted(val_idx)


# --------------------------------------------------------------------------- #
# metrics helpers
# --------------------------------------------------------------------------- #
def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor,
                  blank_id: int) -> list[list[int]]:
    """CTC greedy: argmax -> collapse repeats -> drop blanks."""
    pred = log_probs.argmax(dim=-1)                     # [B, T']
    outs = []
    for row, ln in zip(pred.tolist(), lengths.tolist()):
        collapsed: list[int] = []
        prev: int | None = None
        for cid in row[:ln]:
            if cid != prev:
                collapsed.append(cid)
            prev = cid
        outs.append([c for c in collapsed if c != blank_id])
    return outs


def levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ia, ca in enumerate(a, 1):
        cur = [ia]
        for ib, cb in enumerate(b, 1):
            cur.append(min(prev[ib] + 1, cur[-1] + 1,
                           prev[ib - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# script
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifests", type=str, default="librittsr-clean100")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-frames", type=int, default=1600)
    p.add_argument("--max-utts", type=int, default=16)
    p.add_argument("--bucket", type=int, default=32)
    p.add_argument("--patience", type=int, default=3,
                   help="early stop after this many val evals without "
                        "token-ER improving by --min-delta")
    p.add_argument("--min-delta", type=float, default=0.003)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit-train", type=int, default=0,
                   help="smoke: cap train utterances (deterministic subset)")
    p.add_argument("--limit-val", type=int, default=0, help="smoke: cap val")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints/last.pt")
    p.add_argument("--exp-id", type=str, default=DEFAULT_EXP_ID,
                   help="subdir under experiments/ (smoke runs use a "
                        "scratch id)")
    return p.parse_args(argv)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_model(vocab_size: int) -> CTCAligner:
    cfg = CTCAlignerConfig(vocab_size=vocab_size)
    model = CTCAligner(cfg)
    return model


class CosineSchedule:
    """Linear warmup -> cosine decay to min_lr_ratio * peak (per step)."""

    def __init__(self, optimizer, warmup: int, total_steps: int,
                 min_ratio: float):
        self.opt = optimizer
        self.warmup = int(warmup)
        self.total = max(int(total_steps), warmup + 1)
        self.min_ratio = float(min_ratio)
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        n = self.step_num
        if n <= self.warmup:
            scale = n / max(1, self.warmup)
        else:
            prog = min(1.0, (n - self.warmup) / (self.total - self.warmup))
            scale = self.min_ratio + (1 - self.min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * prog))
        for g in self.opt.param_groups:
            g["lr"] = g["initial_lr"] * scale

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, d):
        self.step_num = int(d["step_num"])
        self.step()   # re-apply lr at restored step


def evaluate(model, loader, blank_id: int) -> dict:
    """Val pass: sum-CTC-loss per target token + corpus-level token ER."""
    model.eval()
    tot_loss = tot_tok = tot_edits = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            logp, ol = model(batch["mel"], batch["mel_lengths"])
            loss_sum = F.ctc_loss(
                logp.transpose(0, 1), batch["phoneme_ids"], ol,
                batch["phoneme_lengths"], blank=blank_id,
                reduction="sum", zero_infinity=True).item()
            tot_loss += loss_sum
            ntok = int(batch["phoneme_lengths"].sum())
            tot_tok += ntok
            hyps = greedy_decode(logp, ol, blank_id)
            refs = [row[:ln].tolist() for row, ln in zip(
                batch["phoneme_ids"], batch["phoneme_lengths"])]
            tot_edits += sum(levenshtein(h, r) for h, r in zip(hyps, refs))
    dt = time.perf_counter() - t0
    return {"loss_per_token": tot_loss / max(tot_tok, 1),
            "token_er": tot_edits / max(tot_tok, 1),
            "n_utts": len(loader.dataset) if hasattr(loader, "dataset") else -1,
            "seconds": round(dt, 2)}


def write_status(exp_dir: Path, **kw) -> None:
    tmp = exp_dir / "run_status.json.tmp"
    tmp.write_text(json.dumps(kw, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, exp_dir / "run_status.json")


class MetricsLog:
    def __init__(self, path: Path):
        self.fh = path.open("a", encoding="utf-8")

    def row(self, **kw) -> None:
        kw["t"] = round(time.time(), 3)
        self.fh.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    t_start = time.perf_counter()
    exp_dir = REPO / "experiments" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLog(exp_dir / "metrics.jsonl")
    try:
        return _run(args, metrics, exp_dir, t_start)
    except SystemExit:
        raise
    except Exception as exc:
        metrics.row(type="fatal", error=repr(exc))
        write_status(exp_dir, phase="error", rc=1, error=repr(exc),
                     wall_s=round(time.perf_counter() - t_start, 1))
        raise
    finally:
        metrics.close()


def _run(args, metrics: MetricsLog, exp_dir: Path, t_start: float) -> int:
    ckpt_dir = exp_dir / "checkpoints"
    seed_info = set_determinism(args.seed, deterministic_algorithms=True,
                                num_threads=args.threads)
    print(f"[aligner] threads={torch.get_num_threads()} "
          f"torch={torch.__version__} python={platform.python_version()}",
          flush=True)
    metrics.row(type="start", args=vars(args), seed=seed_info["seed"])

    # ---- data ---------------------------------------------------------- #
    manifests_dir = REPO / "data" / "manifests"
    names = [s.strip() for s in args.manifests.split(",") if s.strip()]
    audio_rows: list[dict] = []
    phone_rows: list[dict] = []
    for name in names:
        a, ph = load_manifest_pair(name, manifests_dir)
        print(f"[data] {name}: {len(a)} utt "
              f"({sum(r['duration_s'] for r in a) / 3600:.2f} h)", flush=True)
        audio_rows += a
        phone_rows += ph

    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone_rows])
    speaker_index = build_speaker_index(audio_rows)
    vocab.to_json(ckpt_dir / "phoneme_vocab.json")
    blank_id = vocab.size                      # blank appended after vocab
    print(f"[data] vocab={vocab.size} classes={vocab.size + 1} "
          f"(blank id {blank_id})", flush=True)

    dataset = AlignerDataset(audio_rows, phone_rows, speaker_index,
                             vocab=vocab, repo_root=REPO, **MEL_KW)
    print(f"[data] usable={len(dataset)} skipped={len(dataset.skipped)}",
          flush=True)

    feasible = ctc_feasible_indices(dataset)
    dropped = len(dataset) - len(feasible)
    print(f"[data] CTC-feasible={len(feasible)} "
          f"(dropped {dropped}: subsampled frames < minimal CTC path)",
          flush=True)

    if args.limit_train or args.limit_val:      # smoke mode
        by_id = sorted(feasible, key=lambda i: dataset.rows[i]["id"])
        feasible = by_id[: args.limit_train + args.limit_val]
    train_idx, val_idx = stratified_split(dataset, feasible, args.val_frac)
    print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

    samplers = {}
    loaders = {}
    for tag, idxs, shuffle in (("train", train_idx, True),
                               ("val", val_idx, False)):
        s = LengthBucketBatchSampler(
            idxs, dataset.est_frames, max_frames=args.max_frames,
            max_utts=args.max_utts, bucket_size=args.bucket,
            shuffle=shuffle, base_seed=args.seed)
        samplers[tag] = s
        loaders[tag] = DataLoader(dataset, batch_sampler=s,
                                  collate_fn=collate_aligner, num_workers=0)

    # ---- model / optim -------------------------------------------------- #
    model = build_model(vocab.size)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 5_000_000, f"param budget blown: {n_params}"
    print(f"[model] CTCAligner params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98),
                                  weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])
    steps_per_epoch = len(samplers["train"])
    total_steps = steps_per_epoch * args.epochs
    sched = CosineSchedule(optimizer, args.warmup, total_steps,
                           args.min_lr_ratio)
    print(f"[sched] steps/epoch={steps_per_epoch} total={total_steps}",
          flush=True)

    # ---- resume --------------------------------------------------------- #
    start_epoch, global_step = 0, 0
    best = {"token_er": float("inf"), "loss_per_token": float("inf")}
    patience_left = args.patience
    last_path = ckpt_dir / "last.pt"
    if not args.fresh and last_path.is_file():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best = ck.get("best", best)
        patience_left = ck.get("patience_left", patience_left)
        if "rng" in ck:
            torch.set_rng_state(ck["rng"])
        samplers["train"].set_epoch(start_epoch)
        print(f"[resume] {last_path.name} -> epoch {start_epoch} "
              f"step {global_step}", flush=True)

    manifest = {
        "exp_id": DEFAULT_EXP_ID,
        "run_id": f"aligner-{time.strftime('%Y%m%d-%H%M%S')}",
        "manifests": names,
        "usable_utterances": len(dataset),
        "ctc_feasible": len(feasible),
        "infeasible_dropped": dropped,
        "skipped_rows": len(dataset.skipped),
        "train_utterances": len(train_idx),
        "val_utterances": len(val_idx),
        "vocab_size": vocab.size,
        "num_classes_with_blank": vocab.size + 1,
        "blank_id": blank_id,
        "model_params": n_params,
        "mel": dict(MEL_KW),
        "time_subsample": TIME_SUBSAMPLE,
        "args": vars(args),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "machine": platform.platform()},
    }
    (exp_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # ---- train ----------------------------------------------------------- #
    write_status(exp_dir, phase="running", epoch=start_epoch,
                 note="starting" if start_epoch == 0 else "resumed")
    stop = False
    epochs_done = 0
    for epoch in range(start_epoch, args.epochs):
        samplers["train"].set_epoch(epoch)
        model.train()
        ep_loss = ep_tok = 0.0
        n_batches = 0
        t_ep = time.perf_counter()
        run_loss = run_tok = 0.0
        for bi, batch in enumerate(loaders["train"]):
            logp, ol = model(batch["mel"], batch["mel_lengths"])
            loss = F.ctc_loss(logp.transpose(0, 1), batch["phoneme_ids"],
                              ol, batch["phoneme_lengths"],
                              blank=blank_id, reduction="mean",
                              zero_infinity=True)
            if not torch.isfinite(loss):
                metrics.row(type="nan_skip", epoch=epoch, step=global_step)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sched.step()
            global_step += 1
            ntok = int(batch["phoneme_lengths"].sum())
            lv = loss.item()
            ep_loss += lv * ntok
            ep_tok += ntok
            run_loss += lv * ntok
            run_tok += ntok
            n_batches += 1
            if global_step % args.log_every == 0:
                metrics.row(type="train", epoch=epoch, step=global_step,
                            loss_per_token=run_loss / max(run_tok, 1), lr=
                            optimizer.param_groups[0]["lr"])
                run_loss = run_tok = 0.0

        train_lpt = ep_loss / max(ep_tok, 1)
        val = evaluate(model, loaders["val"], blank_id)
        ep_wall = time.perf_counter() - t_ep
        metrics.row(type="val", epoch=epoch, step=global_step,
                    loss_per_token=val["loss_per_token"],
                    token_er=val["token_er"], seconds=val["seconds"])
        print(f"[epoch {epoch}] train_lpt={train_lpt:.4f} "
              f"val_lpt={val['loss_per_token']:.4f} "
              f"token_er={val['token_er']:.4f} "
              f"wall={ep_wall:.0f}s step={global_step}", flush=True)

        prev_best_er = best["token_er"]
        better_key = (val["token_er"], val["loss_per_token"]) < (
            best["token_er"], best["loss_per_token"])
        if better_key:
            best = {"token_er": val["token_er"],
                    "loss_per_token": val["loss_per_token"]}
        # plateau bookkeeping BEFORE writing checkpoints (uses previous best)
        if val["token_er"] <= prev_best_er - args.min_delta:
            patience_left = args.patience
        else:
            patience_left -= 1
        payload = {
            "format": "ctc-aligner-ckpt-v1",
            "model": model.state_dict(),
            "config": model.config.to_dict(),
            "vocab_symbols": list(vocab.symbols),
            "mel": dict(MEL_KW),
            "time_subsample": TIME_SUBSAMPLE,
            "blank_id": blank_id,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": torch.get_rng_state(),
            "best": best,
            "patience_left": patience_left,
            "val_metrics": {k: v for k, v in val.items()},
            "train_loss_per_token": train_lpt,
            "model_params": n_params,
        }
        _atomic_torch_save(payload, last_path)
        if better_key:
            _atomic_torch_save(payload, ckpt_dir / "best.pt")
            print(f"[ckpt] best.pt updated "
                  f"(token_er={best['token_er']:.4f})", flush=True)
        epochs_done = epoch + 1
        metrics.row(type="epoch", epoch=epoch, step=global_step,
                    train_loss_per_token=train_lpt,
                    val_loss_per_token=val["loss_per_token"],
                    token_er=val["token_er"], patience_left=patience_left,
                    epoch_wall_s=round(ep_wall, 1))
        write_status(exp_dir, phase="running", epoch=epoch + 1,
                     best_token_er=best["token_er"],
                     best_val_loss_per_token=best["loss_per_token"],
                     patience_left=patience_left)
        if patience_left <= 0:
            print(f"[early-stop] token-ER plateaued "
                  f"(best {best['token_er']:.4f})", flush=True)
            stop = True

    summary = {
        "phase": "done", "rc": 0,
        "epochs_completed": epochs_done,
        "best_token_er": best["token_er"],
        "best_val_loss_per_token": best["loss_per_token"],
        "global_step": global_step,
        "stopped_early": stop,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    write_status(exp_dir, **summary)
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
