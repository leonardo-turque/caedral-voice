"""CV-G1-E0011 — validation gates for the trained CTC aligner.

Runs the four mission gates against ``checkpoints/best.pt`` and writes
``validation_results.json`` (imported into conclusion.md):

* G1 loss curve:   val CTC loss/token decreases steadily across epochs
                   (parsed from metrics.jsonl).
* G2 rate sanity:  forced-align 50 held-out utterances -> phones_per_sec
                   within [8, 25] for >= 90 % of them.
* G3 monotonicity: Viterbi state paths strictly non-decreasing; durations
                   positive and summing exactly to total_frames.
* G4 silence map:  the 5 longest low-energy spans among those utterances are
                   covered by pause tokens (| / U+2016), not content phones.

Usage:
    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0011-ctc-aligner/validate_aligner.py \
        [--checkpoint .../best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from caedral_voice.model.ctc_aligner import (  # noqa: E402
    TIME_SUBSAMPLE,
    CTCAligner,
    CTCAlignerConfig,
    expand_ds_durations_to_mel,
    viterbi_force_align,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_aligner import (  # noqa: E402
    MEL_KW,
    AlignerDataset,
    build_speaker_index,
    ctc_feasible_indices,
    load_manifest_pair,
    stratified_split,
)

EXP_DIR = REPO / "experiments" / "CV-G1-E0011-ctc-aligner"
N_HELDOUT = 50
PPS_RANGE = (8.0, 25.0)
PPS_MIN_FRAC = 0.90
SILENCE_DROP_NATS = 5.0     # frame mean log-mel below loudest frame - 5.0
SILENCE_MIN_FRAMES = 12     # ~128 ms at 93.75 fps


def gate1_loss_curve(metrics_path: Path) -> dict:
    if not metrics_path.is_file():
        return {"pass": False, "note": f"missing {metrics_path.name}",
                "curve": [], "final_token_er": None}
    rows = [json.loads(l) for l in
            metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    vals = [r for r in rows if r.get("type") == "val"]
    curve = [{"epoch": r["epoch"], "loss_per_token": round(r["loss_per_token"], 5),
              "token_er": round(r["token_er"], 5)} for r in vals]
    if len(curve) < 2:
        return {"pass": False, "note": "fewer than 2 val evals",
                "curve": curve, "final_token_er":
                curve[-1]["token_er"] if curve else None}
    losses = [c["loss_per_token"] for c in curve]
    first, last = losses[0], losses[-1]
    best = min(losses)
    ok = (last <= first * 0.70) and (last <= best * 1.20)
    return {"pass": bool(ok), "curve": curve,
            "first": first, "last": last, "best": best,
            "drop_frac": round(1 - last / first, 4),
            "final_token_er": curve[-1]["token_er"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str,
                   default=str(EXP_DIR / "checkpoints" / "best.pt"))
    p.add_argument("--exp-dir", type=str, default=str(EXP_DIR),
                   help="where metrics.jsonl / validation_results.json live")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--n", type=int, default=N_HELDOUT)
    args = p.parse_args(argv)
    exp_dir = Path(args.exp_dir)

    torch.set_num_threads(6)

    ck = torch.load(Path(args.checkpoint), map_location="cpu",
                    weights_only=False)
    cfg = CTCAlignerConfig.from_dict(ck["config"])
    model = CTCAligner(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    blank_id = cfg.vocab_size
    print(f"[model] epoch={ck.get('epoch')} "
          f"best_token_er={ck.get('best', {}).get('token_er')}", flush=True)

    audio_rows, phone_rows = load_manifest_pair(
        "librittsr-clean100", REPO / "data" / "manifests")
    ph_by_id = {r["id"]: r for r in phone_rows}
    from caedral_voice.training.mel_dataset import PhonemeVocab
    vocab = PhonemeVocab(list(ck["vocab_symbols"]))
    dataset = AlignerDataset(audio_rows, phone_rows,
                             build_speaker_index(audio_rows),
                             repo_root=REPO, **MEL_KW)
    feasible = ctc_feasible_indices(dataset)
    _, val_idx = stratified_split(dataset, feasible, args.val_frac)
    held = sorted(val_idx, key=lambda i: dataset.rows[i]["id"])[: args.n]
    print(f"[heldout] {len(held)} val utterances", flush=True)

    hop, sr = MEL_KW["hop"], MEL_KW["sample_rate"]
    results = []
    silence_regions = []          # (len_frames, rid, start, end)
    g2_in = g2_total = 0
    g3_violations = 0

    for pos, idx in enumerate(held):
        row = dataset.rows[idx]
        rid = row["id"]
        ids = list(row["ids"])
        mel = dataset._load_mel(idx)                     # [80, T] float32
        T = int(mel.shape[1])
        with torch.no_grad():
            logp, ol = model(mel.t().unsqueeze(0),
                             torch.tensor([T], dtype=torch.long))
        lp = logp[0, : int(ol[0])]
        res = viterbi_force_align(lp, ids, blank_id)
        durs_mel = expand_ds_durations_to_mel(res["durations_ds"], T)

        # --- G3 monotonicity ------------------------------------------- #
        states = res["states"]
        monotone = bool(np.all(np.diff(states) >= 0))
        sums_ok = (int(sum(durs_mel)) == T) and min(durs_mel) >= 1
        starts_ok = all(b > a for a, b in zip(res["starts_ds"],
                                              res["starts_ds"][1:]))
        if not (monotone and sums_ok and starts_ok):
            g3_violations += 1

        # --- G2 phones-per-second -------------------------------------- #
        seconds = T * hop / sr
        n_phones = ph_by_id[rid]["n_phones"]
        pps = n_phones / seconds
        g2_total += 1
        g2_in += PPS_RANGE[0] <= pps <= PPS_RANGE[1]

        # --- G4 silence regions ---------------------------------------- #
        energy = mel.mean(dim=0).numpy()
        quiet = energy < (energy.max() - SILENCE_DROP_NATS)
        run = 0
        for t in range(T):
            run = run + 1 if quiet[t] else 0
            if run == SILENCE_MIN_FRAMES:
                end = t + 1
                start = end - SILENCE_MIN_FRAMES
                while end < T and quiet[end]:
                    end += 1
                silence_regions.append((end - start, rid, start, end))
                run = 0

        results.append({
            "id": rid, "total_frames": T, "n_ids": len(ids),
            "n_phones": n_phones, "seconds": round(seconds, 3),
            "phones_per_sec": round(pps, 2),
            "in_range": PPS_RANGE[0] <= pps <= PPS_RANGE[1],
            "monotone": monotone, "sums_ok": sums_ok,
            "min_dur_frames": int(min(durs_mel)),
            "max_dur_frames": int(max(durs_mel)),
            "viterbi_score": round(float(res["score"]), 1),
            "starts_ds": res["starts_ds"],
        })

    # covering token for each silence region (mel-frame spans)
    def cover_symbol(idx_pos: int, center_frame: int) -> tuple[str, int]:
        ids = list(dataset.rows[held[idx_pos]]["ids"])
        T = results[idx_pos]["total_frames"]
        starts_mel = [s * TIME_SUBSAMPLE for s in
                      results[idx_pos]["starts_ds"]]
        edges = sorted(set(starts_mel[1:] + [T]))
        tok = len(ids) - 1
        for i, e in enumerate(edges):
            if center_frame < e:
                tok = min(i, len(ids) - 1)
                break
        return vocab.id_to_sym[ids[tok]], tok

    silence_regions.sort(key=lambda r: (-r[0], r[1]))
    top5 = []
    for length, rid, start, end in silence_regions[:5]:
        k = next(i for i, r in enumerate(results) if r["id"] == rid)
        sym, tok = cover_symbol(k, (start + end) // 2)
        top5.append({"rid": rid, "frames": length,
                     "ms": round(length * hop / sr * 1000), "start": start,
                     "end": end, "covered_symbol": sym,
                     "token_index": tok,
                     "is_pause": sym in ("|", "‖")})
    g4_pass = sum(r["is_pause"] for r in top5) >= 4

    out = {
        "checkpoint": args.checkpoint,
        "g1_loss_curve": gate1_loss_curve(exp_dir / "metrics.jsonl"),
        "g2_rate": {"n": g2_total, "in_range": g2_in,
                    "frac": round(g2_in / max(g2_total, 1), 4),
                    "pass": bool(g2_in / max(g2_total, 1) >= PPS_MIN_FRAC)},
        "g3_monotonicity": {"n": len(results), "violations": g3_violations,
                            "pass": g3_violations == 0},
        "g4_silences": {"top5": top5, "pauses_covered":
                        sum(r["is_pause"] for r in top5),
                        "pass": bool(g4_pass)},
        "per_utt": results,
    }
    out_path = exp_dir / "validation_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    pps_vals = [r["phones_per_sec"] for r in results]
    print("\n=== CV-G1-E0011 validation ===", flush=True)
    c1 = out["g1_loss_curve"]
    if "curve" in c1 and len(c1["curve"]) >= 2:
        g1_detail = (f"({len(c1['curve'])} evals, {c1['first']:.4f} -> "
                     f"{c1['last']:.4f} loss/token, drop "
                     f"{c1['drop_frac']*100:.1f}%)")
    else:
        g1_detail = f"({c1.get('note', 'insufficient evals')})"
    print(f"G1 loss curve: {'PASS' if c1['pass'] else 'FAIL'} {g1_detail}, "
          f"final ER {c1.get('final_token_er')}", flush=True)
    print(f"G2 rate sanity: {'PASS' if out['g2_rate']['pass'] else 'FAIL'} "
          f"{g2_in}/{g2_total} in [{PPS_RANGE[0]:.0f},{PPS_RANGE[1]:.0f}] "
          f"(min {min(pps_vals):.1f} max {max(pps_vals):.1f})", flush=True)
    print(f"G3 monotonicity: {'PASS' if out['g3_monotonicity']['pass'] else 'FAIL'} "
          f"({g3_violations} violations / {len(results)})", flush=True)
    for r in top5:
        print(f"  silence {r['ms']}ms @{r['rid']} -> '{r['covered_symbol']}' "
              f"{'OK' if r['is_pause'] else 'CONTENT-PHONE'}", flush=True)
    print(f"G4 silences->pauses: {'PASS' if g4_pass else 'FAIL'} "
          f"({out['g4_silences']['pauses_covered']}/5)", flush=True)
    print(f"[done] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
