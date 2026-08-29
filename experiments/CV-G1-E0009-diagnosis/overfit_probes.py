"""Post-mortem probes for the E0009 part-2 overfit FAIL (run after
overfit_driver.py --skip-train analysis landed). Quantifies:

P1  prior-head text sensitivity: |m_p(true ids) - m_p(shuffled ids)| vs the
    within-sequence spread of m_p. Near-zero => the text prior barely depends
    on the text => low flow NLL cannot encode content.
P2  sigma collapse: mean/std of exp(logs_p) (prior scales) and of the flow's
    own per-frame standardized residual |z_data - m_exp|/sigma_exp. A tiny
    sigma with small residual means the NLL was won by variance shrinkage,
    not by structure.
P3  duration pattern: predicted vs MAS durations for one clip (correlation,
    first values) - checks whether timing carries any phoneme-specific info.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "CV-G1-E0007-pilot-training"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig  # noqa: E402
from caedral_voice.training.mel_dataset import PhonemeVocab, PilotDataset  # noqa: E402
from caedral_voice.training.seed import set_determinism  # noqa: E402
from pilot_train import build_speaker_index, load_manifest_pair  # noqa: E402

OUT = REPO / "experiments" / "CV-G1-E0009-diagnosis" / "overfit"
CLIP = "1034_121119_000071_000000"  # longest decode clip


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
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval()

    row = next(r for r in ds.rows if r["id"] == CLIP)
    gt_mel = ds._load_mel(ds.rows.index(row))            # (80, T)
    ids = torch.tensor([row["ids"]], dtype=torch.long)
    spk = torch.tensor([row["speaker_id"]])

    with torch.no_grad():
        h = model.encoder(model.phoneme_embed(ids))
        s = model.conditioning(spk, 0)
        m_p, logs_p = model.flow_prior_proj(h).chunk(2, dim=-1)

        # P1: shuffle the phoneme order -> how much does the prior move?
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(ids.shape[1], generator=g)
        h_shuf = model.encoder(model.phoneme_embed(ids[:, perm]))
        m_shuf, logs_shuf = model.flow_prior_proj(h_shuf).chunk(2, dim=-1)

        # P2: data latent vs prior under MAS alignment
        losses = model.losses_for_training(
            mel=gt_mel.t()[None], phoneme_ids=ids, speaker_id=spk, style_id=0,
            mel_lengths=torch.tensor([gt_mel.shape[1]]),
            phoneme_lengths=torch.tensor([ids.shape[1]]))
        d_mas = losses["mas_durations"]
        z, _ = model.flow.forward(gt_mel[None], cond=None, vec=s)
        z = z.transpose(1, 2)                            # (1, T_y, 80)

    from caedral_voice.model.variance import expand_tokens
    m_e, _ = expand_tokens(m_p, d_mas)
    logs_e, _ = expand_tokens(logs_p, d_mas)
    resid = ((z - m_e) / torch.exp(logs_e))
    T_valid = int(d_mas.sum())
    diff_true_shuf = (m_p - m_shuf).abs().mean()
    spread_within = m_p - m_p.mean(dim=1, keepdim=True)

    out = {
        "clip": CLIP,
        "P1_prior_text_sensitivity": {
            "mean_abs_diff_mp_true_vs_shuffled":
                round(float(diff_true_shuf), 4),
            "within_seq_std_of_mp": round(float(spread_within.std()), 4),
            "ratio": round(float(diff_true_shuf / spread_within.std()), 4),
            "reading": ("<0.1 => prior nearly text-independent"),
        },
        "P2_sigma_and_residual": {
            "prior_scale_exp_logs_p_mean":
                round(float(torch.exp(logs_p).mean()), 5),
            "prior_scale_exp_logs_p_std":
                round(float(torch.exp(logs_p).std()), 5),
            "standardized_residual_mean_abs":
                round(float(resid[0, :T_valid].abs().mean()), 4),
            "data_z_std_across_frames":
                round(float(z[0, :T_valid].std()), 4),
            "mp_mean_abs":
                round(float(m_p.abs().mean()), 4),
        },
        "P3_durations": {
            "pred_first20": model.infer(
                ids, speaker_id=int(spk[0]), style_id=0
            )["durations"][0][:20].tolist(),
            "mas_first20": d_mas[0][:20].tolist(),
            "corr_pred_vs_mas": round(float(np.corrcoef(
                model.infer(ids, speaker_id=int(spk[0]), style_id=0
                            )["durations"][0].numpy(),
                d_mas[0].numpy())[0, 1]), 3),
        },
    }
    (OUT / "postmortem_probes.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
