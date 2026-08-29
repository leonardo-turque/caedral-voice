"""Post-mortem probes round 2 (E0009 part-2 FAIL):

P4  mel fidelity under teacher-forced MAS durations, ns=0.1 (near-deterministic
    prior-mean synthesis): per-frame L1 between inferred mel and the GT mel
    they are aligned to. Small => the MODEL reproduces content and any failure
    is downstream; large => prior sampling does not reconstruct content.
P5  vocoder chain on THESE clips: GT log-mel -> 80->100 interp -> Vocos ->
    whisper small.en. Isolates the preview-chain contribution per clip
    (E0009 part 1 used different utterances).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "CV-G1-E0007-pilot-training"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig  # noqa: E402
from caedral_voice.training.mel_dataset import PhonemeVocab, PilotDataset  # noqa: E402
from caedral_voice.training.seed import set_determinism  # noqa: E402
from pilot_train import build_speaker_index, load_manifest_pair  # noqa: E402

OUT = REPO / "experiments" / "CV-G1-E0009-diagnosis" / "overfit"
AUDIO = OUT / "audio"
CLIPS = [
    "1081_128618_000027_000000",
    "1081_128618_000013_000000",
    "1034_121119_000053_000000",
    "1034_121119_000071_000000",
]


def normalize_words(text):
    return "".join(c if (c.isalnum() or c in "' ") else " "
                   for c in text.lower()).split()


def wer(ref, hyp):
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    set_determinism(1234, deterministic_algorithms=True, num_threads=4)
    audio_rows_all, phone_rows_all = load_manifest_pair(
        "librittsr-smoke-10spk", REPO / "data" / "manifests")
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone_rows_all])
    spk_index = build_speaker_index(audio_rows_all)
    by_a = {r["id"]: r for r in audio_rows_all}
    by_p = {r["id"]: r for r in phone_rows_all}
    ids_sub = json.loads(
        (OUT / "results.json").read_text(encoding="utf-8"))["subset"]["ids"]
    ds = PilotDataset([by_a[i] for i in ids_sub],
                      [by_p[i] for i in ids_sub], spk_index,
                      sample_rate=24000, n_mels=80, hop=256, vocab=vocab,
                      repo_root=REPO)

    cfg = CaedralNARConfig.from_json(REPO / "configs/tiny.json")
    cfg.use_flow = True
    ck = torch.load(OUT / "checkpoints" / "last.pt", map_location="cpu",
                    weights_only=False)
    model = CaedralNAR(cfg)
    model.load_state_dict({k.replace("model.", "", 1): v
                           for k, v in ck["model"].items()
                           if k.startswith("model.")})
    model.eval()

    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    from faster_whisper import WhisperModel
    wh = WhisperModel("small.en", device="cpu", compute_type="int8")

    def asr(arr16k):
        segs, _ = wh.transcribe(arr16k, beam_size=5,
                                condition_on_previous_text=False)
        return " ".join(s.text for s in segs).strip()

    rows_out = []
    for cid in CLIPS:
        row = next(r for r in ds.rows if r["id"] == cid)
        idx = ds.rows.index(row)
        gt_mel = ds._load_mel(idx)                        # (80, T)
        ids_t = torch.tensor([row["ids"]], dtype=torch.long)
        spk = torch.tensor([row["speaker_id"]])
        with torch.no_grad():
            losses = model.losses_for_training(
                mel=gt_mel.t()[None], phoneme_ids=ids_t, speaker_id=spk,
                style_id=0,
                mel_lengths=torch.tensor([gt_mel.shape[1]]),
                phoneme_lengths=torch.tensor([len(row["ids"])]))
            d_mas = losses["mas_durations"]
            inf = model.infer(ids_t, speaker_id=int(spk[0]), style_id=0,
                              noise_scale=0.1, durations=d_mas)
        T_gt = gt_mel.shape[1]
        n = min(int(inf["n_mel_frames"]), T_gt)
        rec = inf["mel"][0, :n].t()                       # (80, n)
        ref = gt_mel[:, :n]
        l1 = float((rec - ref).abs().mean())
        corr = float(np.corrcoef(rec.numpy().ravel(),
                                 ref.numpy().ravel())[0, 1])
        # spectral-detail proxy: fraction of frames whose max channel is high
        frac_hi_rec = float((rec.max(dim=0).values > 1.0).float().mean())
        frac_hi_ref = float((ref.max(dim=0).values > 1.0).float().mean())

        # P5: GT mel -> vocos -> whisper (chain ceiling for this clip)
        xs100 = np.linspace(0, 79, 100)
        mel100 = np.stack(
            [np.interp(xs100, np.arange(80), f) for f in gt_mel.t().numpy()],
            axis=0).T.astype(np.float32)
        with torch.no_grad():
            wav_chain = vocos.decode(
                torch.from_numpy(mel100)[None]).squeeze(0).numpy()
        sf.write(AUDIO / f"{cid.replace('_', '-')}-GTmel-chain.wav",
                 np.clip(wav_chain, -1, 1), 24000)
        import torchaudio
        w16 = torchaudio.functional.resample(
            torch.from_numpy(wav_chain), 24000, 16000).numpy()
        hyp_chain = asr(w16)

        rows_out.append({
            "id": cid, "text": by_a[cid]["transcript"],
            "P4_tfd_ns01_mel_L1": round(l1, 3),
            "P4_mel_corr": round(corr, 3),
            "frac_frames_maxch_gt1": {"gt": round(frac_hi_ref, 3),
                                      "inferred": round(frac_hi_rec, 3)},
            "T_gt": T_gt, "T_inferred": int(inf["n_mel_frames"]),
            "P5_chain_hyp": hyp_chain,
            "P5_chain_wer": round(wer(by_a[cid]["transcript"], hyp_chain), 3),
        })
        print(rows_out[-1], flush=True)

    (OUT / "postmortem_probes2.json").write_text(
        json.dumps(rows_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
