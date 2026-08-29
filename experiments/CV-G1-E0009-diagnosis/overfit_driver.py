"""CV-G1-E0009 part 2 — mandatory overfit gate (DIAG-02).

Question: CAN this pipeline memorize? If 16 utterances x 2000 steps cannot
drive flow_loss to <15% of its step-200 value and synthesize recognizable
clips, no amount of steps on the full corpus will fix the babble found by the
15k pilot -- the failure is structural (model-side, since E0009 part 1 showed
the vocoder chain clean on ground-truth audio).

Design (mirrors experiments/CV-G1-E0007-pilot-training/pilot_train.py wiring;
that script is imported, not edited):

    subset (2 spk x 8 utt, ~21 s) -> PhonemeVocab built over the FULL corpus
    -> PilotDataset -> LengthBucketBatchSampler(max_utts=3, max_frames=600)
    -> CaedralNAR tiny(use_flow=True).losses_for_training -> Trainer

Optimizer AdamW(lr 2e-4, betas (0.9, 0.98), wd 1e-6); schedule: 50 linear
warmup then CONSTANT (min_lr_ratio=1.0) per DIAG-02 spec ("constant-ish").
Threads pinned to 4 to coexist with the running 15k pilot (PID 50376).

Gates (results.json + conclusion.md):
  1. flow_loss(step 2000) < 15% of flow_loss(step 200)
  2. inferred phones_per_sec on TRAIN utterances within [8, 25]
  3. decode 4 memorized utterances (model.infer -> mel 80->100 interp ->
     pretrained Vocos) -> overfit/audio/
  4. ASR (faster-whisper small.en int8) -> WER vs transcripts;
     PASS if best-WER < 0.35 on >= 3/4 clips

Suspect checklist if FAIL: mel normalization consistency train vs infer,
expand_tokens mask/padding at inference, flow inverse noise_scale,
vocab coverage (skipped symbols logged).

Run:
    .venv/Scripts/python.exe -X utf8 \
        experiments/CV-G1-E0009-diagnosis/overfit_driver.py

Stages can be skipped for re-analysis: --skip-train resumes analysis from
overfit/checkpoints/last.pt; --skip-asr omits stage 4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "CV-G1-E0007-pilot-training"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
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

from pilot_train import (  # noqa: E402  (pilot wiring reused verbatim)
    PilotTrainModule,
    build_speaker_index,
    collate_pilot,
    load_manifest_pair,
)

EXP_DIR = REPO / "experiments" / "CV-G1-E0009-diagnosis"
OUT_DIR = EXP_DIR / "overfit"
AUDIO_DIR = OUT_DIR / "audio"

MANIFEST = "librittsr-smoke-10spk"

# 2 speakers x 8 utterances (~21 s total): speaker 1081 contributes its whole
# set (7 recipe fragments + 1 real sentence), speaker 1034 its 8 shortest
# lines. Short clips keep the 2000-step budget inside the coexistence window.
SUBSET_IDS = [
    # speaker 1081 (speaker_id 4 in the full-corpus index)
    "1081_128618_000027_000000",  # BAKED PERCH                    0.92 s
    "1081_128618_000029_000000",  # PERCH SALAD                    0.96 s
    "1081_128618_000007_000000",  # BROILED PERCH                  1.04 s
    "1081_128618_000003_000000",  # FRIED PERCH-two                1.20 s
    "1081_128618_000023_000000",  # PERCH A l a SICILY             1.27 s
    "1081_128618_000005_000000",  # FRIED PERCH-three              1.40 s
    "1081_128618_000013_000000",  # PERCH A L'ALLEMANDE            1.60 s
    "1081_128618_000024_000003",  # Cook together four tablespoonfuls ... 4.00 s
    # speaker 1034 (speaker_id 1)
    "1034_121119_000001_000001",  # The Law.                       0.64 s
    "1034_121119_000019_000001",  # Your name?"                    0.69 s
    "1034_121119_000053_000000",  # "Well, I own it."              0.95 s
    "1034_121119_000064_000008",  # Have they loved me?            1.02 s
    "1034_121119_000070_000000",  # "The weakness of a murderer!"  1.28 s
    "1034_121119_000064_000009",  # Have they spared me?           1.38 s
    "1034_121119_000018_000000",  # "Oh, this is too much!"        1.69 s
    "1034_121119_000071_000000",  # "His dishonor reflects upon us." 1.78 s
]

# 4 memorized utterances for gates 3+4 (both speakers, varied content/length)
DECODE_IDS = [
    "1081_128618_000027_000000",  # BAKED PERCH                     0.92 s
    "1081_128618_000013_000000",  # PERCH A L'ALLEMANDE             1.60 s
    "1034_121119_000053_000000",  # "Well, I own it."               0.95 s
    "1034_121119_000071_000000",  # "His dishonor reflects upon us." 1.78 s
]

NOISE_SCALES = [0.667, 0.333, 0.1]   # latent temperature sweep at inference
PPS_RANGE = (8.0, 25.0)
FL_RATIO_THRESHOLD = 0.15
WER_THRESHOLD = 0.35


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_words(text: str) -> list[str]:
    """Lowercase; punctuation except apostrophes becomes a separator."""
    out = []
    for ch in text.lower():
        out.append(ch if (ch.isalnum() or ch in "' ") else " ")
    return "".join(out).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0.0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def mel_to_vocos(mel_t: torch.Tensor) -> np.ndarray:
    """Our ln-Mel (T, 80) -> vocos 100-band grid -> waveform (numpy)."""
    mel = mel_t.detach().cpu().numpy()
    xs100 = np.linspace(0, 79, 100)
    mel100 = np.stack(
        [np.interp(xs100, np.arange(80), frame) for frame in mel],
        axis=0).T.astype(np.float32)                      # (100, T)
    global _VOCOS
    if _VOCOS is None:
        from vocos import Vocos
        _VOCOS = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos = _VOCOS
    with torch.no_grad():
        wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
    return np.clip(wav, -1.0, 1.0)


_VOCOS = None


def load_audio16k(path: Path) -> tuple[np.ndarray, int]:
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.numpy(), sr


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-utts", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--config", type=str, default=str(REPO / "configs/tiny.json"))
    p.add_argument("--skip-train", action="store_true",
                   help="load overfit/checkpoints/last.pt instead of fitting")
    p.add_argument("--skip-asr", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    t_start = time.perf_counter()
    print(f"[overfit] threads={args.threads} steps={args.steps} lr={args.lr} "
          f"warmup={args.warmup} seed={args.seed}", flush=True)

    set_determinism(args.seed, deterministic_algorithms=True,
                    num_threads=args.threads)

    # ------------------------------------------------------------------ #
    # data: 16-utterance subset, FULL-corpus vocab (pilot parity)
    # ------------------------------------------------------------------ #
    audio_rows_all, phone_rows_all = load_manifest_pair(
        MANIFEST, REPO / "data" / "manifests")
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone_rows_all])
    speaker_index = build_speaker_index(audio_rows_all)

    by_id_a = {r["id"]: r for r in audio_rows_all}
    by_id_p = {r["id"]: r for r in phone_rows_all}
    audio_rows = [by_id_a[i] for i in SUBSET_IDS]
    phone_rows = [by_id_p[i] for i in SUBSET_IDS]

    dataset = PilotDataset(
        audio_rows, phone_rows, speaker_index,
        sample_rate=24000, n_mels=80, hop=256, vocab=vocab, repo_root=REPO,
    )
    if len(dataset) != len(SUBSET_IDS):
        reasons = "; ".join(f"{i}: {w}" for i, w in dataset.skipped)
        raise RuntimeError(f"subset rows dropped ({len(dataset.skipped)}): "
                           f"{reasons}")
    total_s = sum(r["duration_s"] for r in dataset.rows)

    # ---- suspect D: vocab coverage (skipped symbols) ------------------ #
    skipped_symbols: dict[str, int] = {}
    for row in phone_rows:
        for tok in row["phoneme_string"].split():
            if tok in vocab.sym_to_id:
                continue
            for ch in tok:
                if ch not in vocab.sym_to_id:
                    skipped_symbols[ch] = skipped_symbols.get(ch, 0) + 1
    print(f"[data] usable={len(dataset)} total={total_s:.2f}s "
          f"vocab={vocab.size} skipped_symbols={skipped_symbols or '{}'}",
          flush=True)

    sampler = LengthBucketBatchSampler(
        list(range(len(dataset))), dataset.est_frames,
        max_frames=args.max_frames, max_utts=args.max_utts,
        bucket_size=32, shuffle=True, base_seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=collate_pilot, num_workers=0)
    n_batches = len(sampler._plan())
    print(f"[data] batches/epoch={n_batches} "
          f"(max_utts={args.max_utts}, max_frames={args.max_frames})",
          flush=True)

    # ------------------------------------------------------------------ #
    # model + optimizer + trainer (pilot wiring)
    # ------------------------------------------------------------------ #
    model_cfg = CaedralNARConfig.from_json(Path(args.config))
    model_cfg.use_flow = True
    model_cfg.n_speakers = max(model_cfg.n_speakers, len(speaker_index))
    model = CaedralNAR(model_cfg)
    module = PilotTrainModule(model, sample_rate=model_cfg.sample_rate,
                              hop=model_cfg.hop_length)
    print(f"[model] tiny(use_flow=True) params="
          f"{module.model.count_params():,}", flush=True)

    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=1e-6)
    scheduler = build_scheduler(optimizer, {
        "warmup_steps": args.warmup, "max_steps": args.steps,
        "min_lr_ratio": 1.0})          # constant lr after warmup

    captured: dict[int, dict] = {}
    window200: list[float] = []
    window2000: list[float] = []
    traj: list[tuple[int, float]] = []

    def on_step_end(t: Trainer, skipped: bool) -> None:
        if skipped:
            t.log_row({"type": "event", "event": "nan_skip",
                       "step": t.global_step})
            return
        d = module.last_diag
        s = t.global_step
        if s == 200:
            captured[200] = dict(d)
            window200.append(d["flow_loss"])
        elif 191 <= s <= 210:
            window200.append(d["flow_loss"])
        elif 1981 <= s <= 2000:
            window2000.append(d["flow_loss"])
        if s in (200, 2000):
            captured[s] = dict(d)
        if s % 10 == 0:
            traj.append((s, d["flow_loss"]))
        if s % 100 == 0:
            rate = s / (time.perf_counter() - t._t_fit_start)
            eta = (args.steps - s) / max(rate, 1e-9) / 60.0
            print(f"[step {s}] flow={d['flow_loss']:.2f} "
                  f"dur={d['dur_loss']:.3f} pps={d['phones_per_sec']:.1f} "
                  f"| {rate:.2f} it/s ETA {eta:.1f} min", flush=True)

    def on_epoch_end(t: Trainer) -> None:
        sampler.set_epoch(t.epoch + 1)

    trainer = Trainer()
    ckpt_dir = OUT_DIR / "checkpoints"
    if args.skip_train:
        info = trainer.resume(ckpt_dir / "last.pt")
        print(f"[resume] step {info['global_step']}", flush=True)

    t_fit0 = time.perf_counter()
    summary = trainer.fit(
        module, optimizer, scheduler, loader, None,
        {
            "exp_id": "overfit",
            "experiment_root": str(EXP_DIR),
            "run_id": f"overfit-{time.strftime('%Y%m%d-%H%M%S')}",
            "seed": args.seed,
            "num_threads": args.threads,
            "reseed_on_fit": True,
            "deterministic_algorithms": True,
            "max_steps": args.steps,
            "warmup_steps": args.warmup,
            "min_lr_ratio": 1.0,
            "grad_clip": 1.0,
            "loss_fn": lambda out, _y: out["loss"],
            "log_every": 10,
            "val_every_steps": 0,
            "val_at_end": False,
            "checkpoint_every_steps": 500,
            "keep_step_checkpoints": 1,
            "best_key": "val_loss",
            "best_mode": "min",
            "hooks": {"on_step_end": on_step_end,
                      "on_epoch_end": on_epoch_end},
        })
    fit_wall = time.perf_counter() - t_fit0
    print(f"[done] {json.dumps(summary)} fit_wall={fit_wall:.1f}s", flush=True)

    fl = {k: captured.get(k, {}).get("flow_loss") for k in (200, 2000)}
    if fl[200] is None or fl[2000] is None:
        print(f"[warn] gate-1 captures missing ({sorted(captured)}); "
              f"run used fewer than 210 steps?", flush=True)
        ratio = None
        gate1_pass = False
    else:
        ratio = fl[2000] / fl[200]
        gate1_pass = bool(ratio < FL_RATIO_THRESHOLD)

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    model.eval()
    sr_model = model_cfg.sample_rate
    hop = model_cfg.hop_length
    id_by_row = {r["id"]: r for r in dataset.rows}

    def infer_one(ids_list: list[int], spk: int, noise_scale: float,
                  durations: torch.Tensor | None = None) -> dict:
        with torch.no_grad():
            return model.infer(
                torch.tensor([ids_list], dtype=torch.long),
                speaker_id=spk, style_id=0,
                noise_scale=noise_scale, durations=durations)

    # ---- gate 2: inferred phones_per_sec on all TRAIN utterances ------ #
    pps_rows = []
    for row in dataset.rows:
        out = infer_one(row["ids"], row["speaker_id"], NOISE_SCALES[0])
        d = out["durations"][0]
        seconds = int(d.sum()) * hop / sr_model
        pps = len(row["ids"]) / seconds if seconds > 0 else float("nan")
        pps_rows.append({"id": row["id"], "n_phones": len(row["ids"]),
                         "pred_frames": int(d.sum()),
                         "gt_frames_est": dataset.est_frames(
                             dataset.rows.index(row)),
                         "audio_s_pred": round(seconds, 3),
                         "gt_duration_s": row["duration_s"],
                         "phones_per_sec": round(pps, 2)})
    pps_vals = [r["phones_per_sec"] for r in pps_rows]
    pps_mean = float(np.mean(pps_vals))
    gate2_pass = bool(PPS_RANGE[0] <= pps_mean <= PPS_RANGE[1])

    # ---- suspects A/B probes ------------------------------------------ #
    suspect_probes: dict = {}
    probe_row = next(r for r in dataset.rows if r["id"] == DECODE_IDS[0])
    gt_mel = dataset._load_mel(dataset.rows.index(probe_row))  # (80, T) GT
    inf = infer_one(probe_row["ids"], probe_row["speaker_id"], 0.667)
    inf_mel = inf["mel"][0]                                    # (T, 80)
    n_valid = int(inf["n_mel_frames"])

    def stats(x: torch.Tensor) -> dict:
        xf = x.float()
        return {"mean": round(float(xf.mean()), 3),
                "std": round(float(xf.std()), 3),
                "min": round(float(xf.min()), 3),
                "max": round(float(xf.max()), 3)}

    suspect_probes["mel_norm_train_vs_infer"] = {
        "gt_logmel_stats": stats(gt_mel.t()),
        "inferred_mel_stats_ns0667": stats(inf_mel[:n_valid]),
        "note": ("same transform family expected: power=1 mel, natural log, "
                 "floor 1e-5; large offset/scale gap here would indicate a "
                 "train/infer space mismatch")}
    # suspect B: expand_tokens mask/padding behavior at inference (B=1)
    from caedral_voice.model.variance import expand_tokens
    d_pred = inf["durations"]
    h = model.phoneme_embed(
        torch.tensor([probe_row["ids"]], dtype=torch.long))
    with torch.no_grad():
        h_enc = model.encoder(h)
    h_exp, mask = expand_tokens(h_enc, d_pred)
    suspect_probes["expand_tokens_mask_padding"] = {
        "batch": 1,
        "sum_durations": int(d_pred.sum()),
        "mask_true_frames": int(mask.sum()),
        "expanded_T": int(h_exp.shape[1]),
        "n_mel_frames_reported": n_valid,
        "consistent": bool(int(mask.sum()) == int(d_pred.sum())
                           and int(h_exp.shape[1]) == int(d_pred.sum())),
        "pad_frames_in_batch": 0,
        "wav_len_equals_T_times_hop":
            bool(int(inf["waveform"].shape[1]) ==
                 int(h_exp.shape[1]) * hop)}

    # ---- gates 3+4: decode 4 memorized utterances --------------------- #
    from faster_whisper import WhisperModel
    whisper = None
    if not args.skip_asr:
        whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

    def asr(path_or_arr):
        segs, _info = whisper.transcribe(
            path_or_arr, beam_size=5, condition_on_previous_text=False)
        return " ".join(s.text for s in segs).strip()

    decode_rows = []
    for cid in DECODE_IDS:
        row = id_by_row[cid]
        stem = cid.replace("_", "-")
        variants: dict[str, dict] = {}

        for ns in NOISE_SCALES:
            out = infer_one(row["ids"], row["speaker_id"], ns)
            n = int(out["n_mel_frames"])
            mel = out["mel"][0, :n]                            # (T, 80)
            wav = mel_to_vocos(mel)
            fname = f"{stem}-ns{str(ns).replace('.', '')}.wav"
            sf.write(AUDIO_DIR / fname, wav, sr_model)
            variants[f"ns{ns}"] = {"file": fname,
                                   "mel_frames": n,
                                   "audio_s": round(len(wav) / sr_model, 3)}

        # teacher-forced durations (MAS on GT mel) -> prior heads -> inverse:
        # separates duration-predictor failure from prior/flow failure
        with torch.no_grad():
            losses = model.losses_for_training(
                mel=gt_mel.t()[None],
                phoneme_ids=torch.tensor([row["ids"]], dtype=torch.long),
                speaker_id=torch.tensor([row["speaker_id"]]),
                style_id=0,
                mel_lengths=torch.tensor([gt_mel.shape[1]]),
                phoneme_lengths=torch.tensor([len(row["ids"])]))
        d_mas = losses["mas_durations"]
        for ns in (NOISE_SCALES[0], NOISE_SCALES[-1]):
            out = infer_one(row["ids"], row["speaker_id"], ns,
                            durations=d_mas)
            n = int(out["n_mel_frames"])
            wav = mel_to_vocos(out["mel"][0, :n])
            fname = f"{stem}-tfd-ns{str(ns).replace('.', '')}.wav"
            sf.write(AUDIO_DIR / fname, wav, sr_model)
            variants[f"tfd_ns{ns}"] = {"file": fname,
                                       "mel_frames": n,
                                       "audio_s": round(len(wav) / sr_model, 3),
                                       "used": "MAS durations from GT mel"}

        entry = {"id": cid, "transcript": row["phoneme_string"],
                 "text": by_id_a[cid]["transcript"],
                 "speaker_prefix": row["speaker_prefix"],
                 "variants": variants}
        if whisper is not None:
            ref = entry["text"]
            wers = {}
            hyps = {}
            for vtag, v in variants.items():
                wav16, _sr = load_audio16k(AUDIO_DIR / v["file"])
                hyp = asr(wav16)
                hyps[vtag] = hyp
                wers[vtag] = round(wer(ref, hyp), 3)
            gt16, _sr = load_audio16k(REPO / row["audio_path"])
            gt_hyp = asr(gt16)
            hyps["GT_AUDIO"] = gt_hyp
            wers["GT_AUDIO"] = round(wer(ref, gt_hyp), 3)
            best_v = min((v for v in wers if v != "GT_AUDIO"),
                         key=lambda v: wers[v])
            entry.update({
                "hypotheses": hyps, "wer": wers,
                "best_variant": best_v, "best_wer": wers[best_v],
                "wer_gt_audio_ceiling": wers["GT_AUDIO"],
                "pass_clip": bool(wers[best_v] < WER_THRESHOLD)})
        decode_rows.append(entry)
        best_txt = (f"best={entry.get('best_wer')} ({entry.get('best_variant')})"
                    if whisper is not None else "")
        print(f"[decode] {cid}: {best_txt}", flush=True)

    n_clips_pass = sum(1 for e in decode_rows if e.get("pass_clip"))
    gate4_pass = bool(n_clips_pass >= 3)

    # ------------------------------------------------------------------ #
    # verdict + artifacts
    # ------------------------------------------------------------------ #
    gates = {
        "gate1_flow_ratio": {
            "flow_loss_step200": (round(fl[200], 4)
                                  if fl[200] is not None else None),
            "flow_loss_step2000": (round(fl[2000], 4)
                                   if fl[2000] is not None else None),
            "window_mean_191_210": (round(float(np.mean(window200)), 4)
                                    if window200 else None),
            "window_mean_1981_2000": (round(float(np.mean(window2000)), 4)
                                      if window2000 else None),
            "ratio": (round(ratio, 4) if ratio is not None else None),
            "threshold": FL_RATIO_THRESHOLD, "pass": gate1_pass},
        "gate2_phones_per_sec": {
            "mean": round(pps_mean, 2),
            "min": round(min(pps_vals), 2), "max": round(max(pps_vals), 2),
            "range": list(PPS_RANGE), "pass": gate2_pass,
            "per_utterance": pps_rows},
        "gate3_decode": {
            "dir": str(AUDIO_DIR.relative_to(REPO)),
            "clips": [e["id"] for e in decode_rows]},
        "gate4_asr": {
            "model": "faster-whisper small.en int8",
            "wer_threshold": WER_THRESHOLD,
            "n_clips_pass": n_clips_pass, "required": 3,
            "pass": gate4_pass},
    }
    verdict = "PASS" if all((gate1_pass, gate2_pass, gate4_pass)) else "FAIL"

    results = {
        "experiment": "CV-G1-E0009-diagnosis part 2 (DIAG-02 overfit gate)",
        "date": _now(),
        "subset": {"n_utts": len(dataset), "speakers": ["1081", "1034"],
                   "total_audio_s": round(total_s, 2),
                   "ids": SUBSET_IDS},
        "train": {"steps": summary["global_step"],
                  "optimizer": "AdamW lr 2e-4 betas (0.9,0.98) wd 1e-6",
                  "schedule": f"warmup {args.warmup} then constant",
                  "batching": f"max_utts={args.max_utts} "
                              f"max_frames={args.max_frames}",
                  "epochs_completed": summary["epochs_completed"],
                  "nan_events": summary["nan_events"],
                  "wall_s": round(fit_wall, 1),
                  "it_s": round(summary["global_step"] / fit_wall, 3)},
        "flow_trajectory_every10": traj,
        "gates": gates,
        "suspect_probes": suspect_probes,
        "decode_asr_detail": decode_rows,
        "verdict": verdict,
        "coexistence_note": "ran alongside 15k pilot PID 50376 (4+4 threads)",
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"[out] {OUT_DIR / 'results.json'}", flush=True)
    print(f"[verdict] {verdict}", flush=True)

    # stash for conclusion.md writer
    (OUT_DIR / "_summary_for_conclusion.json").write_text(json.dumps({
        "verdict": verdict,
        "gates": gates,
        "train": results["train"],
        "suspect_probes": suspect_probes,
        "decode_best": [{k: e[k] for k in
                         ("id", "text", "best_variant", "best_wer",
                          "wer_gt_audio_ceiling", "wer", "hypotheses")
                         if k in e} for e in decode_rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
