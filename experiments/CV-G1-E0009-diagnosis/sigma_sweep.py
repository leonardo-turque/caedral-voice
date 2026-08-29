"""R4 free-win check: inference-time sigma sweep on the live pilot checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import io
import json
import numpy as np
import soundfile as sf
import torch

from caedral_voice.model.caedral_nar import CaedralNAR
from caedral_voice.text import TextFrontend

OUT = REPO / "experiments" / "CV-G1-E0009-diagnosis" / "sigma_sweep"
OUT.mkdir(parents=True, exist_ok=True)

run_dir = REPO / "experiments" / "pilot-tiny-15k"
ck_path = sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
from caedral_voice.model.caedral_nar import CaedralNARConfig
known = set(CaedralNARConfig.__dataclass_fields__)
base = json.loads(io.open(REPO / "configs" / "tiny.json", encoding="utf-8").read())
base.update({k: v for k, v in ck["cfg"].items() if k in known})
base["use_flow"] = True
model = CaedralNAR(base)
model.load_state_dict({k.replace("model.", "", 1): v for k, v in ck["model"].items()
                       if k.startswith("model.")}, strict=False)
model.eval()
vocab = json.loads(io.open(run_dir / "checkpoints" / "phoneme_vocab.json", encoding="utf-8").read())
sym2id = {s: i for i, s in enumerate(vocab["symbols"])}
tf = TextFrontend("lexicon+g2pen")
text = "Hello. This is the first generation of Caedral Voice."
ids = [sym2id[t] for t in tf.process(text).phoneme_tokens if t in sym2id]

from vocos import Vocos
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

with torch.no_grad():
    for sigma in (1.0, 0.667, 0.5, 0.333, 0.2):
        out = model.infer(torch.tensor([ids]), speaker_id=0, style_id=0,
                          noise_scale=sigma)
        mel = out["mel"].squeeze(0).numpy()
        xs100 = np.linspace(0, 79, 100)
        mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                          axis=1).astype(np.float32)
        wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
        f = OUT / f"sigma-{sigma}.wav"
        sf.write(f, wav, 24000)
        print(f"sigma={sigma}: {len(wav)/24000:.2f}s -> {f.name}")
print("done; classify next")
