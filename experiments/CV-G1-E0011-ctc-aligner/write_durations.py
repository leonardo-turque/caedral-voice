"""CV-G1-E0011 — write supervised durations for EVERY kept utterance.

Forced-aligns the whole corpus with the trained CTC aligner and writes
``data/manifests/<name>.durations.jsonl`` with one row per KEPT manifest
utterance::

    {"id": ..., "durations": [int per phoneme-id, mel frames],
     "total_frames": int, "aligned": bool}

* ``durations`` are in FULL-RATE mel frames (24 kHz / hop 256 -> 93.75 fps),
  matching what the FastSpeech2-style regressor consumes next; boundaries
  live on the /4 subsample grid, the last phone absorbs the remainder so
  ``sum(durations) == total_frames == T_mel`` exactly.
* ``aligned=false`` rows (subsampled frames cannot cover the MINIMAL CTC
  path -- symbols + blanks between adjacent-different ids; ~0.7 % of this
  corpus at /2 subsampling) get deterministic UNIFORM splits; downstream
  can filter on the flag.
* Output written atomically (tmp + os.replace); safe detached like
  train_aligner.py. A sidecar ``<out>.stats.json`` carries counts and a
  phones-per-second distribution for the conclusion gates.

Usage:
    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0011-ctc-aligner/write_durations.py \
        [--checkpoint .../best.pt] [--out data/manifests/....durations.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torchaudio  # noqa: E402

from caedral_voice.model.ctc_aligner import (  # noqa: E402
    TIME_SUBSAMPLE,
    CTCAligner,
    CTCAlignerConfig,
    expand_ds_durations_to_mel,
    min_ctc_frames,
    viterbi_force_align,
)
from caedral_voice.training.mel_dataset import PhonemeVocab  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_aligner import MEL_KW, load_manifest_pair  # noqa: E402


_MEL_FN = {}


def load_mel(path: Path, sample_rate: int, n_mels: int, hop: int,
             n_fft: int) -> torch.Tensor:
    """wav -> mono -> resample -> log-Mel [n_mels, T]; identical math to
    mel_dataset.PilotDataset._load_mel (power=1, natural log, floor 1e-5)."""
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    fn = _MEL_FN.get("fn")
    if fn is None:
        fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, power=1.0)
        _MEL_FN["fn"] = fn
    return torch.log(fn(wav).clamp_min(1e-5))


def uniform_durations(n_frames: int, n_tokens: int) -> list[int]:
    base, rem = divmod(int(n_frames), int(n_tokens))
    return [base + 1 if i < rem else base for i in range(n_tokens)]


def self_check_mel_equality(audio_rows, phone_rows, n_rows: int) -> float:
    """Standalone loader must equal PilotDataset's cached mel exactly."""
    from caedral_voice.training.mel_dataset import PilotDataset

    from train_aligner import build_speaker_index
    ds = PilotDataset(audio_rows[:n_rows], phone_rows[:n_rows],
                      build_speaker_index(audio_rows[:n_rows]),
                      repo_root=REPO, **MEL_KW)
    worst = 0.0
    for i in range(len(ds)):
        rel = ds.rows[i]["audio_path"]
        mine = load_mel(REPO / rel, **MEL_KW)
        theirs = ds._load_mel(i)
        assert mine.shape == theirs.shape, (rel, mine.shape, theirs.shape)
        worst = max(worst, float((mine - theirs).abs().max()))
    return worst


def parse_args(argv=None):
    exp_dir = REPO / "experiments" / "CV-G1-E0011-ctc-aligner"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str,
                   default=str(exp_dir / "checkpoints" / "best.pt"))
    p.add_argument("--out", type=str,
                   default=str(REPO / "data" / "manifests"
                               / "librittsr-clean100.durations.jsonl"))
    p.add_argument("--manifests", type=str, default="librittsr-clean100")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="smoke cap")
    p.add_argument("--self-check", type=int, default=25,
                   help="rows used to verify standalone mel == PilotDataset "
                        "(0 disables)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()

    # ---- checkpoint ------------------------------------------------------ #
    ckpt_path = Path(args.checkpoint)
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CTCAlignerConfig.from_dict(ck["config"])
    model = CTCAligner(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    blank_id = cfg.vocab_size
    print(f"[model] {ckpt_path.name}: params="
          f"{sum(q.numel() for q in model.parameters()):,} "
          f"epoch={ck.get('epoch')} "
          f"best_token_er={ck.get('best', {}).get('token_er')}", flush=True)

    # ---- data ------------------------------------------------------------ #
    audio_rows, phone_rows = load_manifest_pair(
        args.manifests, REPO / "data" / "manifests")
    if args.limit:
        audio_rows = audio_rows[: args.limit]
    ph_by_id = {r["id"]: r for r in phone_rows}
    vocab = PhonemeVocab(list(ck["vocab_symbols"]))
    assert vocab.size == cfg.vocab_size, "vocab/checkpoint mismatch"

    if args.self_check:
        worst = self_check_mel_equality(audio_rows, phone_rows,
                                        args.self_check)
        print(f"[self-check] max |standalone - PilotDataset| mel diff over "
              f"<= {args.self_check} rows = {worst:.2e}", flush=True)
        assert worst < 1e-4, "standalone mel diverges from PilotDataset"

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_aligned = n_uniform = n_empty = 0
    pps = []
    rows_out = 0
    fh = tmp_path.open("w", encoding="utf-8")
    try:
        for row in audio_rows:
            rid = row["id"]
            ph = ph_by_id.get(rid)
            if ph is None:
                raise KeyError(f"no phoneme row for kept utterance {rid}")
            ids = vocab.encode(ph["phoneme_string"])
            L = len(ids)
            mel = load_mel(REPO / row["audio_path"], **MEL_KW)
            T = int(mel.shape[1])

            durations: list[int] | None = None
            aligned = False
            if L > 0:
                t_ds = -(-T // TIME_SUBSAMPLE)             # ceil(T / 2)
                if t_ds >= min_ctc_frames(ids):
                    with torch.no_grad():
                        logp, ol = model(mel.t().unsqueeze(0),
                                         torch.tensor([T], dtype=torch.long))
                    lp1 = logp[0, : int(ol[0])]
                    try:
                        res = viterbi_force_align(lp1, ids, blank_id)
                        durations = expand_ds_durations_to_mel(
                            res["durations_ds"], T)
                        aligned = True
                    except ValueError:
                        durations = None        # no valid path -> fallback
            if durations is None:
                if L == 0:
                    durations = []
                    n_empty += 1
                else:
                    durations = uniform_durations(T, L)
                n_uniform += 1
            else:
                n_aligned += 1

            seconds = T * MEL_KW["hop"] / MEL_KW["sample_rate"]
            if seconds > 0 and L > 0:
                pps.append(ph.get("n_phones", L) / seconds)

            fh.write(json.dumps(
                {"id": rid, "durations": [int(d) for d in durations],
                 "total_frames": T, "aligned": aligned},
                ensure_ascii=False) + "\n")
            rows_out += 1
            if rows_out % 2000 == 0:
                el = time.perf_counter() - t0
                rate = rows_out / el
                print(f"[write] {rows_out}/{len(audio_rows)} utts "
                      f"(aligned {n_aligned}, uniform {n_uniform}) "
                      f"{rate:.0f}/s eta {(len(audio_rows)-rows_out)/rate/60:.1f} min",
                      flush=True)
    finally:
        fh.close()

    stats = {
        "checkpoint": str(ckpt_path),
        "rows_written": rows_out,
        "aligned": n_aligned,
        "uniform_fallback": n_uniform,
        "empty_phonemes": n_empty,
        "phones_per_sec": (
            {"p01": round(statistics.quantiles(pps, n=100)[0], 2),
             "p50": round(statistics.median(pps), 2),
             "p99": round(statistics.quantiles(pps, n=100)[98], 2),
             "min": round(min(pps), 2), "max": round(max(pps), 2)}
            if pps else None),
        "in_8_25_pct": round(100.0 * sum(8 <= x <= 25 for x in pps)
                             / max(len(pps), 1), 2) if pps else None,
        "wall_s": round(time.perf_counter() - t0, 1),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__},
    }
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    tmp_stats = stats_path.with_suffix(".json.tmp")
    tmp_stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_stats, stats_path)
    os.replace(tmp_path, out_path)                  # atomic publish
    print(f"[done] {json.dumps(stats)}", flush=True)
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
