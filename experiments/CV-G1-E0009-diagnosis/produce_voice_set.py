"""Produce the official artifacts/latest-preview/voices/ set (founder criteria)."""
from __future__ import annotations

import io
import json
import shutil
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

EVAL_A = "Hello. This is the first generation of Caedral Voice."
EVAL_B = "Artificial intelligence should sound natural, expressive, and human."
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


rows = []
for i, (spk_name, spk_id) in enumerate(sorted(spk_index.items(), key=lambda kv: kv[1]), 1):
    wav_a = synth(EVAL_A, spk_id)
    wav_b = synth(EVAL_B, spk_id)
    fa = OUT / f"voice-{i:02d}-evalA.wav"
    fb = OUT / f"voice-{i:02d}-evalB.wav"
    sf.write(fa, wav_a, 24000)
    sf.write(fb, wav_b, 24000)
    ra = cp.classify_file(str(fa), EVAL_A)
    rb = cp.classify_file(str(fb), EVAL_B)
    rows.append({"voice": f"voice-{i:02d}", "libritts_speaker": spk_name,
                 "speaker_index": spk_id,
                 "evalA_class": ra.get("class"), "evalA_wer": ra.get("min_wer"),
                 "evalB_class": rb.get("class"), "evalB_wer": rb.get("min_wer"),
                 "intelligible": (ra.get("class") == "INTELLIGIBLE"
                                  and rb.get("class") == "INTELLIGIBLE")})
    print(rows[-1])

def _w(x):
    return x if x is not None else 1.0


intel_rows = [r for r in rows if r["intelligible"]]
intel_rows.sort(key=lambda r: _w(r["evalA_wer"]) + _w(r["evalB_wer"]))
for rank, r in enumerate(intel_rows[:3], 1):
    shutil.copy(OUT / f"{r['voice']}-evalA.wav", OUT / f"best-voice-{rank:02d}.wav")
    shutil.copy(OUT / f"{r['voice']}-evalB.wav", OUT / f"best-voice-{rank:02d}-evalB.wav")
    print(f"best-{rank}: {r['voice']} ({r['libritts_speaker']}) "
          f"WER {r['evalA_wer']}/{r['evalB_wer']}")

provenance = {
    "checkpoint": str(ck_path),
    "sigma": SIGMA,
    "criteria": "INTELLIGIBLE = both evalA and evalB classified INTELLIGIBLE by scripts/classify_preview.py",
    "note": ("10 trained speakers available in this pilot run; the full 15-voice slate "
             "lands with the clean100 (44h/247spk) retrain per founder plan."),
    "results": rows,
}
io.open(OUT / "provenance.json", "w", encoding="utf-8").write(
    json.dumps(provenance, indent=2))
print(f"\n{len(intel_rows)} intelligible voices; wrote {OUT}")
