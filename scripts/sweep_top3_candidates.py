"""PRE-ARCH TOP-3 candidate sweep: checkpoint x speaker x speed x sentence.

Renders candidates sequentially (low footprint, protects r3 worker),
classifies each immediately, writes JSONL results for ranking.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch

from caedral_voice.model.regressor import CaedralRegressor, CaedralRegressorConfig
from caedral_voice.text import TextFrontend

OUT = REPO / "artifacts" / "pre-arch-top3" / "candidates"
OUT.mkdir(parents=True, exist_ok=True)
SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}
SPEAKERS = ["2416", "78", "6147", "696", "2836", "6415"]
SPEEDS = [0.95, 1.0, 1.05]

sys.path.insert(0, str(REPO / "scripts"))
import classify_preview as cp


def load_regressor(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    spk_rows = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    cfg["n_speakers"] = spk_rows
    model = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()
    step = ck.get("trainer", {}).get("global_step") or ck.get("global_step")
    return model, cfg, step


def main() -> None:
    torch.set_num_threads(4)
    tf = TextFrontend("lexicon+g2pen")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    # Freeze current r3 last.pt snapshot as a candidate source (worker keeps running).
    r3_snap = REPO / "experiments" / "r3" / "checkpoints"
    src_last = r3_snap / "last.pt"
    frozen = OUT / "r3-frozen-last.pt"
    if src_last.exists():
        shutil.copy2(src_last, frozen)

    candidates = []
    if (REPO / "experiments/r2/checkpoints/best.pt").exists():
        candidates.append(("r2-best-15k", REPO / "experiments/r2/checkpoints/best.pt"))
    if frozen.exists():
        candidates.append(("r3-snapshot", frozen))

    spk_index_r2 = json.loads(io.open(REPO / "experiments/r2/checkpoints/speaker_index.json",
                                      encoding="utf-8").read())
    spk_index_r3 = json.loads(io.open(REPO / "experiments/r3/checkpoints/speaker_index.json",
                                      encoding="utf-8").read())

    results_path = OUT / "sweep-results.jsonl"
    with io.open(results_path, "w", encoding="utf-8") as resfile:
        for label, ckpt_path in candidates:
            model, cfg, step = load_regressor(ckpt_path)
            spk_idx = spk_index_r2 if label.startswith("r2") else spk_index_r3
            sym2id = None
            vocab_p = (REPO / "experiments/r2/checkpoints/phoneme_vocab.json"
                       if label.startswith("r2") else
                       REPO / "experiments/r3/checkpoints/phoneme_vocab.json")
            sym2id = {s: i for i, s in enumerate(
                json.loads(io.open(vocab_p, encoding="utf-8").read())["symbols"])}
            for spk in SPEAKERS:
                if spk not in spk_idx:
                    continue
                sid = spk_idx[spk]
                for sname, text in SENTENCES.items():
                    for speed in SPEEDS:
                        res = tf.process(text)
                        ids = [sym2id[t] for t in res.phoneme_tokens if t in sym2id]
                        if not ids:
                            continue
                        with torch.no_grad():
                            out = model(torch.tensor([ids]), speaker_id=sid, style_id=0,
                                        speed=speed)
                        mel = out["mel"].squeeze(0)[: out["n_mel_frames"]].numpy()
                        xs100 = np.linspace(0, 79, 100)
                        mel100 = np.stack(
                            [np.interp(xs100, np.arange(80), fr) for fr in mel],
                            axis=1).astype(np.float32)
                        with torch.no_grad():
                            wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
                        tag = f"{label}-spk{spk}-{sname}-spd{speed}"
                        f = OUT / f"cand-{tag}.wav"
                        sf.write(f, wav, 24000)
                        r = cp.classify_file(str(f), text)
                        row = {"tag": tag, "checkpoint": label, "step": step,
                               "speaker": spk, "speaker_index": sid, "sentence": sname,
                               "speed": speed, "class": r.get("class"),
                               "min_wer": r.get("min_wer"), "dnsmos": r.get("dnsmos_ovrl"),
                               "audio_seconds": round(len(wav) / 24000, 3)}
                        resfile.write(json.dumps(row) + "\n")
                        resfile.flush()
                        print(row)
    print(f"sweep complete -> {results_path}")


if __name__ == "__main__":
    main()
