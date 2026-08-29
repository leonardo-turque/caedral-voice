"""TOP-3 iteration 2: onset-mitigation variants + replacement speaker sweep.

Hypotheses for the "Artificial"-onset defect (7/7 renders):
  H-a: no left pause context -> model rushes the first word.
       Mitigation: prepend '|' pause token to the phoneme stream.
  H-b: duration predictor under-allocates early frames.
       Mitigation: scale predicted durations with floor>=2 for the first K tokens.
Also sweep replacement third-voice candidates (males 7505/8838/7067, F 6415-evalA).
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

OUT = Path("artifacts/pre-arch-top3/candidates-v2")
OUT.mkdir(parents=True, exist_ok=True)
SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}
sys.path.insert(0, str(REPO / "scripts"))
import classify_preview as cp


def load_regressor(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    cfg["n_speakers"] = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    model = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ck.get("trainer", {}).get("global_step")


def mel_to_wav(model_out, vocos):
    mel = model_out["mel"].squeeze(0)[: model_out["n_mel_frames"]].numpy()
    xs100 = np.linspace(0, 79, 100)
    mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                      axis=1).astype(np.float32)
    with torch.no_grad():
        return vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()


def main() -> None:
    torch.set_num_threads(4)
    tf = TextFrontend("lexicon+g2pen")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    results_path = OUT / "iter2-results.jsonl"
    resfile = io.open(results_path, "w", encoding="utf-8")

    def render_and_score(model, sym2id, sid, text, sname, tag, spk):
        res = tf.process(text)
        toks = res.phoneme_tokens
        ids = [sym2id[t] for t in toks if t in sym2id]
        with torch.no_grad():
            out = model(torch.tensor([ids]), speaker_id=sid, style_id=0)
        wav = mel_to_wav(out, vocos)
        f = OUT / f"{tag}.wav"
        sf.write(f, wav, 24000)
        r = cp.classify_file(str(f), text)
        row = {"tag": tag, "speaker": spk, "sentence": sname,
               "class": r.get("class"), "min_wer": r.get("min_wer"),
               "dnsmos": r.get("dnsmos_ovrl"),
               "audio_seconds": round(len(wav) / 24000, 3)}
        resfile.write(json.dumps(row) + "\n"); resfile.flush()
        print(row)
        return row

    # --- Part 1: onset mitigation on the best current voice (spk78, r2-best) ---
    model_r2, step_r2 = load_regressor(REPO / "experiments/r2/checkpoints/best.pt")
    vocab_r2 = json.loads(io.open(REPO / "experiments/r2/checkpoints/phoneme_vocab.json",
                                  encoding="utf-8").read())
    sym2id = {s: i for i, s in enumerate(vocab_r2["symbols"])}
    PAUSE = sym2id.get("|")
    text_b = SENTENCES["evalB"]

    base_ids = [sym2id[t] for t in tf.process(text_b).phoneme_tokens if t in sym2id]
    sid78 = json.loads(io.open(REPO / "experiments/r2/checkpoints/speaker_index.json",
                               encoding="utf-8").read())["78"]
    rows = []
    # baseline (no mitigation)
    with torch.no_grad():
        out = model_r2(torch.tensor([base_ids]), speaker_id=sid78, style_id=0)
    rows.append(("v2-baseline", out))
    # H-a: leading pause token(s)
    for npause in (1, 2):
        ids = ([PAUSE] * npause) + base_ids
        with torch.no_grad():
            out = model_r2(torch.tensor([ids]), speaker_id=sid78, style_id=0)
        rows.append((f"v2-leadpause{npause}", out))
    # H-b: duration floor 2 for first 4 tokens (inference-time override)
    with torch.no_grad():
        out = model_r2(torch.tensor([base_ids]), speaker_id=sid78, style_id=0)
        d = out["durations"].clone()
        d[:, :4] = torch.clamp(d[:, :4], min=2)
        out2 = model_r2(torch.tensor([base_ids]), speaker_id=sid78, style_id=0,
                        durations=d)
    rows.append(("v2-durfloor", out2))

    for tag, out in rows:
        wav = mel_to_wav(out, vocos)
        f = OUT / f"mit-{tag}.wav"
        sf.write(f, wav, 24000)
        r = cp.classify_file(str(f), text_b)
        print({"tag": tag, "class": r.get("class"), "wer": r.get("min_wer"),
               "dnsmos": r.get("dnsmos_ovrl")})

    # --- Part 2: replacement third-voice candidates on r3 snapshot ---
    model_r3, step_r3 = load_regressor(Path("experiments/r3/checkpoints/last.pt"))
    vocab_r3 = json.loads(io.open(REPO / "experiments/r3/checkpoints/phoneme_vocab.json",
                                  encoding="utf-8").read())
    sym3 = {s: i for i, s in enumerate(vocab_r3["symbols"])}
    spk3 = json.loads(io.open(REPO / "experiments/r3/checkpoints/speaker_index.json",
                              encoding="utf-8").read())
    for spk in ("7505", "8838", "7067"):
        if spk not in spk3:
            print(f"spk{spk}: not in table"); continue
        sid = spk3[spk]
        for sname, text in SENTENCES.items():
            ids = [sym3[t] for t in tf.process(text).phoneme_tokens if t in sym3]
            ids = [sym3["|"]] + ids  # apply leading-pause mitigation by default
            with torch.no_grad():
                out = model_r3(torch.tensor([ids]), speaker_id=sid, style_id=0)
            wav = mel_to_wav(out, vocos)
            f = OUT / f"cand3-spk{spk}-{sname}.wav"
            sf.write(f, wav, 24000)
            r = cp.classify_file(str(f), text)
            row = {"tag": f"cand3-spk{spk}-{sname}", "speaker": spk,
                   "sentence": sname, "class": r.get("class"),
                   "min_wer": r.get("min_wer"), "dnsmos": r.get("dnsmos_ovrl")}
            resfile.write(json.dumps(row) + "\n"); resfile.flush()
            print(row)
    resfile.close()
    print(f"done -> {results_path}")


if __name__ == "__main__":
    main()
