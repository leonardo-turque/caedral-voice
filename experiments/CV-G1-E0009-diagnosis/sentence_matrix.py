"""Multi-sentence x multi-speaker intelligibility matrix at low sigma."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch

from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig
from caedral_voice.text import TextFrontend

OUT = REPO / "artifacts" / "latest-preview" / "voices"
OUT.mkdir(parents=True, exist_ok=True)

run_dir = REPO / "experiments" / "pilot-tiny-15k"
ck_path = sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
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
spk_index = json.loads(io.open(run_dir / "checkpoints" / "speaker_index.json", encoding="utf-8").read())
tf = TextFrontend("lexicon+g2pen")

SENTENCES = {
    "sentA": "Hello. This is the first generation of Caedral Voice.",
    "sentB": "Artificial intelligence should sound natural, expressive, and human.",
}
SIGMA = 0.25

from vocos import Vocos
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
sys.path.insert(0, str(REPO / "scripts"))
import classify_preview as cp


def synth(text: str, spk: int):
    ids = [sym2id[t] for t in tf.process(text).phoneme_tokens if t in sym2id]
    with torch.no_grad():
        out = model.infer(torch.tensor([ids]), speaker_id=spk, style_id=0,
                          noise_scale=SIGMA)
    mel = out["mel"].squeeze(0).numpy()
    xs100 = np.linspace(0, 79, 100)
    mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                      axis=1).astype(np.float32)
    with torch.no_grad():
        return vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()


results = []
for spk_name, spk_id in sorted(spk_index.items(), key=lambda kv: kv[1])[:10]:
    for sname, text in SENTENCES.items():
        wav = synth(text, spk_id)
        f = OUT / f"probe-spk{spk_id}-{sname}.wav"
        sf.write(f, wav, 24000)
        r = cp.classify_file(str(f), text)
        results.append({"speaker": spk_name, "sentence": sname,
                        "class": r.get("class"), "wer": r.get("min_wer"),
                        "ovrl": r.get("dnsmos_ovrl")})
        print(f"spk{spk_id}({spk_name}) {sname}: {r.get('class')} WER={r.get('min_wer')}")

intel = [r for r in results if r["class"] == "INTELLIGIBLE"]
print(f"\nINTELLIGIBLE: {len(intel)}/{len(results)} | "
      f"speakers with sentA AND sentB intelligent: "
      f"{len({r['speaker'] for r in intel})}")
json.dump({"checkpoint": str(ck_path), "sigma": SIGMA, "results": results},
          io.open(REPO / "experiments" / "CV-G1-E0009-diagnosis" / "sigma_matrix.json", "w"),
          indent=2)
