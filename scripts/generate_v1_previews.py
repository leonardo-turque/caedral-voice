"""Generate V1 candidate previews from the trained CaedralRegressor (E0013 r2).

Pipeline: OUR frontend -> OUR trained regressor (supervised durations) ->
pretrained Vocos decode (MIT). Outputs to artifacts/caedral-voice-v1/voices/
with per-file classification + provenance.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch

from caedral_voice.model.regressor import CaedralRegressor
from caedral_voice.text import TextFrontend

OUT = REPO / "artifacts" / "caedral-voice-v1" / "voices"
RUN = REPO / "experiments" / "r2"
CKPT = RUN / "checkpoints" / "best.pt"

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "evalB": "Artificial intelligence should sound natural, expressive, and human.",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    from caedral_voice.model.regressor import CaedralRegressorConfig
    known = set(CaedralRegressorConfig.__dataclass_fields__)
    # The checkpoint cfg is authoritative for regressor-specific fields;
    # tiny.json keys that regressor doesn't define (head_dim etc.) are dropped.
    cfg = {k: v for k, v in ck["cfg"].items() if k in known}
    # speaker table was runtime-expanded (40 -> 257); infer true size from weights
    spk_rows = ck["model"]["model.conditioning.speaker_table.weight"].shape[0]
    cfg["n_speakers"] = spk_rows
    model = CaedralRegressor(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    vocab = json.loads(io.open(RUN / "checkpoints" / "phoneme_vocab.json", encoding="utf-8").read())
    sym2id = {s: i for i, s in enumerate(vocab["symbols"])}
    spk_index = json.loads(io.open(RUN / "checkpoints" / "speaker_index.json", encoding="utf-8").read())

    # 15-voice slate from docs/data/VOICE-SLATE-V1.md
    SLATE = ["6209", "6415", "696", "2416", "6147", "1088", "3879", "2836",
             "78", "7447", "8419", "7067", "8838", "7505", "7402"]
    missing = [s for s in SLATE if s not in spk_index]
    if missing:
        print(f"WARNING: slate speakers not in training table: {missing}")

    tf = TextFrontend("lexicon+g2pen")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    sys.path.insert(0, str(REPO / "scripts"))
    import classify_preview as cp

    t_start = time.perf_counter()
    results = []
    for voice_no, spk in enumerate(SLATE, 1):
        if spk not in spk_index:
            continue
        sid = spk_index[spk]
        for sname, text in SENTENCES.items():
            res = tf.process(text)
            ids = [sym2id[t] for t in res.phoneme_tokens if t in sym2id]
            skipped = len(res.phoneme_tokens) - len(ids)
            with torch.no_grad():
                out = model(torch.tensor([ids]), speaker_id=sid, style_id=0)
            mel = out["mel"].squeeze(0)[: out["n_mel_frames"]].numpy()
            xs100 = np.linspace(0, 79, 100)
            mel100 = np.stack([np.interp(xs100, np.arange(80), fr) for fr in mel],
                              axis=1).astype(np.float32)
            with torch.no_grad():
                wav = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
            fname = f"voice-{voice_no:02d}-{sname}.wav"
            sf.write(OUT / fname, wav, 24000)
            r = cp.classify_file(str(OUT / fname), text)
            results.append({
                "file": fname, "libritts_speaker": spk, "speaker_index": sid,
                "sentence": sname, "class": r.get("class"),
                "min_wer": r.get("min_wer"), "dnsmos_ovrl": r.get("dnsmos_ovrl"),
                "audio_seconds": round(len(wav) / 24000, 3),
                "skipped_symbols": skipped,
            })
            print(f"v{voice_no:02d} ({spk}) {sname}: {r.get('class')} WER={r.get('min_wer')} DNSMOS={r.get('dnsmos_ovrl')}")

    intel = [r for r in results if r["class"] == "INTELLIGIBLE"
             and r["sentence"] == "evalA"]
    by_voice = {}
    for r in intel:
        by_voice.setdefault(r["file"].split("-")[1], []).append(r)

    provenance = {
        "checkpoint": str(CKPT),
        "step": ck.get("trainer", {}).get("global_step"),
        "val_mel_l1": 0.8647432296943292,
        "architecture": "CaedralRegressor medium 15.94M (supervised durations via CTC aligner E0011)",
        "decoder": "Vocos charactr/mel-24khz (MIT) via 80->100 band interpolation",
        "frontend": "lexicon+g2pen shippable ladder",
        "training_data": "LibriTTS-R clean100 44.11h/30917utts/257spk (CC-BY-4.0)",
        "git_commit": git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    io.open(OUT.parent / "provenance" / "voices-provenance.json", "w", encoding="utf-8").write(
        json.dumps(provenance, indent=2))
    n_int = len({r["libritts_speaker"] for r in intel})
    print(f"\nDONE: {len(results)} clips | intelligible voices: {n_int}/15 | "
          f"wall {time.perf_counter()-t_start:.0f}s")


if __name__ == "__main__":
    main()
