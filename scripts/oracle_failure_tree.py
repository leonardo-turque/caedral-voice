"""RESCUE ORACLE TREE — quantify WHY current output is robotic/unintelligible.

Compares TEACHER reference vs FAILED_BASELINE (r2-best) renders on real data.
No training. Light CPU. Evidence for the failure tree.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.text import TextFrontend  # noqa: E402
sys.path.insert(0, str(REPO))

OUT = REPO / "artifacts" / "pre-arch-rescue" / "oracles"
OUT.mkdir(parents=True, exist_ok=True)


def f0_stats(wav: np.ndarray, sr: int) -> dict:
    y = wav.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return {"voiced_frames": int(len(v)), "f0_std_semitones": None,
                "f0_range_semitones": None}
    import numpy as _np
    st = 12 * _np.log2(v / _np.median(v))
    return {"voiced_frames": int(len(v)),
            "f0_std_semitones": round(float(_np.std(st)), 3),
            "f0_range_semitones": round(float(12 * _np.log2(v.max() / v.min())), 3),
            "f0_median_hz": round(float(_np.median(v)), 1)}


def mel_flatness(mel: np.ndarray) -> dict:
    """Over-smoothing proxy: per-frame spectral variance of mel bands + HF ratio."""
    var_per_frame = mel.var(axis=0)
    hf = mel[int(mel.shape[0] * 0.6):].mean()
    lf = mel[: int(mel.shape[0] * 0.3)].mean()
    return {"mel_var_median": round(float(np.median(var_per_frame)), 4),
            "hf_over_lf_ratio": round(float(hf / max(lf, 1e-6)), 4)}


def main() -> None:
    torch.set_num_threads(3)

    # --- pick 3 real utterances from clean100 val range (diverse speakers) ---
    manifest = [json.loads(l) for l in
                io.open(REPO / "data/manifests/librittsr-clean100.jsonl", encoding="utf-8")]
    durs = {json.loads(l)["id"]: json.loads(l)["durations"]
            for l in io.open(REPO / "data/manifests/librittsr-clean100.durations.jsonl",
                             encoding="utf-8")}
    picked = []
    seen_spk = set()
    for row in manifest:
        spk = row["id"].split("_")[0]
        if spk in ("78", "2416", "8838") and spk not in seen_spk and row["id"] in durs:
            picked.append(row)
            seen_spk.add(spk)
        if len(picked) == 3:
            break

    # --- load FAILED_BASELINE model (r2 best) ---
    from caedral_voice.model.regressor import CaedralRegressor, CaedralRegressorConfig
    ck = torch.load(REPO / "experiments/r2/checkpoints/best.pt", map_location="cpu",
                    weights_only=False)
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    cfg["n_speakers"] = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    model = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    vocab = json.loads(io.open(REPO / "experiments/r2/checkpoints/phoneme_vocab.json",
                               encoding="utf-8").read())
    sym2id = {s: i for i, s in enumerate(vocab["symbols"])}
    spk_index = json.loads(io.open(REPO / "experiments/r2/checkpoints/speaker_index.json",
                                   encoding="utf-8").read())
    tf = TextFrontend("lexicon+g2pen")

    report = {"items": []}
    for row in picked:
        spk = row["id"].split("_")[0]
        sid = spk_index.get(spk)
        wav_path = REPO / "data" / "processed" / "librittsr-clean100" / f"{row['id']}.wav"
        wav_gt, sr = sf.read(str(wav_path), dtype="float32")
        gt_f0 = f0_stats(wav_gt, sr)

        # GT-duration render (oracle 3C): use CTC durations, not predicted
        ids = [sym2id[t] for t in tf.process(row["transcript"]).phoneme_tokens
               if t in sym2id][: len(durs[row["id"]])]
        dd = durs[row["id"]][: len(ids)]
        with torch.no_grad():
            out_gt = model(torch.tensor([ids]), speaker_id=sid, style_id=0,
                           durations=[dd])
            out_pd = model(torch.tensor([ids]), speaker_id=sid, style_id=0)
        mel_gt = out_gt["mel"].squeeze(0)[: out_gt["n_mel_frames"]].numpy()
        mel_pd = out_pd["mel"].squeeze(0)[: out_pd["n_mel_frames"]].numpy()

        flat_gt = mel_flatness(mel_gt.T if mel_gt.shape[0] != 80 else mel_gt)
        flat_pd = mel_flatness(mel_pd.T if mel_pd.shape[0] != 80 else mel_pd)

        item = {
            "id": row["id"], "speaker": spk,
            "gt_f0": gt_f0,
            "gt_dur_render_mel": flat_gt,
            "pred_dur_render_mel": flat_pd,
        }
        report["items"].append(item)
        print(json.dumps(item))

    json.dump(report, io.open(OUT / "oracle-failure-tree.json", "w", encoding="utf-8"),
              indent=2)
    print(f"written -> {OUT / 'oracle-failure-tree.json'}")


if __name__ == "__main__":
    main()
