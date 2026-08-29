"""Render frozen rescue sentences from a rescue-prosody checkpoint + classify."""
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

RUN = REPO / "experiments" / "rescue-prosody"
OUT = REPO / "artifacts" / "pre-arch-rescue" / "renders"
SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}
SPEAKERS = ["78", "2416", "6415"]


def main() -> None:
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/last.pt"
    ckpt_path = RUN / ckpt_arg if not Path(ckpt_arg).is_absolute() else Path(ckpt_arg)
    torch.set_num_threads(4)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from caedral_voice.model.regressor import CaedralRegressorConfig
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    cfg["n_speakers"] = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    model = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    vocab = json.loads(io.open(RUN / "checkpoints" / "phoneme_vocab.json",
                               encoding="utf-8").read())
    sym2id = {s: i for i, s in enumerate(vocab["symbols"])}
    spk_index = json.loads(io.open(RUN / "checkpoints" / "speaker_index.json",
                                   encoding="utf-8").read())
    tf = TextFrontend("lexicon+g2pen")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    sys.path.insert(0, str(REPO / "scripts"))
    import classify_preview as cp

    out_dir = REPO / "artifacts" / "pre-arch-rescue" / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = ck.get("trainer", {}).get("global_step")

    rows = []
    for spk in SPEAKERS:
        if spk not in spk_index:
            continue
        sid = spk_index[spk]
        for sname, text in SENTENCES.items():
            ids = [sym2id[t] for t in tf.process(text).phoneme_tokens if t in sym2id]
            ids = [sym2id.get("|", 1)] + ids  # leading pause (harmless here)
            with torch.no_grad():
                out = model(torch.tensor([ids]), speaker_id=sid, style_id=0,
                            speed=1.05)
            mel = out["mel"].squeeze(0)[: out["n_mel_frames"]].numpy()
            xs100 = np.linspace(0, 79, 100)
            mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                              axis=1).astype(np.float32)
            with torch.no_grad():
                wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
            f = out_dir / f"rescue-s{step}-spk{spk}-{sname}.wav"
            sf.write(f, wav, 24000)
            r = cp.classify_file(str(f), text)
            rows.append({"file": f.name, "step": step, "speaker": spk,
                         "sentence": sname, "class": r.get("class"),
                         "min_wer": r.get("min_wer"), "dnsmos": r.get("dnsmos_ovrl")})
            print(rows[-1])
    json.dump(rows, io.open(out_dir / f"render-s{step}.json", "w", encoding="utf-8"),
              indent=2)


if __name__ == "__main__":
    main()
