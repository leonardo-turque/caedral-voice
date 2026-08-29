"""Render A/B pairs: FAILED_BASELINE (r2-best) vs RESCUE checkpoint, same text/speaker.
Usage: python scripts/render_rescue_ab.py <rescue_ckpt_relative_to_rescue-prosody>
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch

from caedral_voice.model.regressor import CaedralRegressor, CaedralRegressorConfig
from caedral_voice.text import TextFrontend

OUT = REPO / "artifacts" / "rescue-comparison"
SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}
SPEAKERS = ["78", "2416", "6415"]


def load_reg(ckpt: Path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    cfg["n_speakers"] = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    m = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def render(model, sym2id, sid, text, vocos, path):
    ids = [sym2id[t] for t in TextFrontend("lexicon+g2pen").process(text).phoneme_tokens
           if t in sym2id]
    ids = [sym2id.get("|", 1)] + ids
    with torch.no_grad():
        out = model(torch.tensor([ids]), speaker_id=sid, style_id=0, speed=1.05)
    mel = out["mel"].squeeze(0)[: out["n_mel_frames"]].numpy()
    xs100 = np.linspace(0, 79, 100)
    mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                      axis=1).astype(np.float32)
    with torch.no_grad():
        wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
    sf.write(path, wav, 24000)


def main() -> None:
    ck_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/last.pt"
    run_dir = REPO / "experiments" / "rescue-prosody"
    ckpt_path = run_dir / ck_arg if not Path(ck_arg).is_absolute() else Path(ck_arg)
    torch.set_num_threads(4)

    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    base_ckpt = REPO / "experiments/r2/checkpoints/best.pt"
    model_b = load_reg(base_ckpt)
    vocab_b = json.loads(io.open(REPO / "experiments/r2/checkpoints/phoneme_vocab.json",
                                 encoding="utf-8").read())
    spk_b = json.loads(io.open(REPO / "experiments/r2/checkpoints/speaker_index.json",
                               encoding="utf-8").read())

    model_r = load_reg(ckpt_path)
    vocab_r = json.loads(io.open(run_dir / "checkpoints/phoneme_vocab.json",
                                 encoding="utf-8").read())
    spk_r = json.loads(io.open(run_dir / "checkpoints/speaker_index.json",
                               encoding="utf-8").read())
    ck_r = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step_r = ck_r.get("trainer", {}).get("global_step")

    ab = OUT / f"step-{step_r}"
    ab.mkdir(parents=True, exist_ok=True)
    for spk in SPEAKERS:
        if spk not in spk_b or spk not in spk_r:
            continue
        for sname, text in SENTENCES.items():
            render(model_b, {s: i for i, s in enumerate(
                json.loads(io.open(REPO / "experiments/r2/checkpoints/phoneme_vocab.json",
                                   encoding="utf-8").read())["symbols"])},
                spk_b[spk], text, vocos, ab / f"baseline-{spk}-{sname}.wav")
            render(model_r, {s: i for i, s in enumerate(vocab_r["symbols"])},
                   spk_r[spk], text, vocos, ab / f"rescue-{spk}-{sname}.wav")
            print(f"A/B rendered: {spk} {sname}")
    print(f"done -> {ab}")


if __name__ == "__main__":
    main()
