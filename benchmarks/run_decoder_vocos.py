"""Phase G decoder measurement: Vocos (MIT, charactr weights) on CPU.

Measures mel->audio decode RTF for the leading iSTFT-head vocoder candidate
on this machine, using mels derived from the persisted baseline samples
(24 kHz). Decode-only timing excludes feature extraction; reported separately.

Output: benchmarks/results/decoders/vocos-<stamp>.jsonl
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

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from benchmarks.harness.measure import measure, rtf  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402

AUDIO_ROOT = REPO / "artifacts" / "audio" / "baselines" / "kokoro"
RESULTS = REPO / "benchmarks" / "results" / "decoders"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.set_num_threads(ap_threads)

    from vocos import Vocos

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS / f"vocos-{stamp}.jsonl"

    print("loading vocos charactr/mel-24khz ...")
    t0 = time.perf_counter()
    model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    load_s = time.perf_counter() - t0

    header = {
        "decoder": "vocos-mel-24khz (charactr)",
        "license": "MIT (code+weights)",
        "machine": capture(),
        "git_commit": git_commit(),
        "torch_num_threads": ap_threads,
        "load_seconds": round(load_s, 3),
    }
    (RESULTS / f"vocos-{stamp}.meta.json").write_text(
        json.dumps(header, indent=2), encoding="utf-8")

    wavs = sorted(AUDIO_ROOT.rglob("af_heart/*.wav"))[:3]
    print(f"decoding {len(wavs)} clips @ {ap_threads} threads")

    with io.open(out_path, "w", encoding="utf-8") as out:
        # warmup
        wav_path = wavs[0]
        audio, sr = sf.read(wav_path, dtype="float32")
        feats = model.feature_extractor(torch.from_numpy(audio)[None])
        measure(lambda: model.decode(feats))
        for wav_path in wavs + sorted(AUDIO_ROOT.rglob("para_001.wav"))[:1]:
            audio, sr = sf.read(wav_path, dtype="float32")
            feats = model.feature_extractor(torch.from_numpy(audio)[None])
            result, perf = measure(lambda f=feats: model.decode(f))
            audio_out = result.squeeze().detach().numpy()
            audio_s = len(audio_out) / sr
            row = {
                "clip": wav_path.relative_to(REPO).as_posix(),
                "mel_frames": int(feats.shape[-1]),
                "audio_seconds": round(audio_s, 3),
                "decode_rtf": round(rtf(perf.wall_seconds, audio_s), 5),
                **{f"perf_{k}": v for k, v in perf.to_dict().items()},
            }
            out.write(json.dumps(row) + "\n")
            print(f"{wav_path.name}: {audio_s:.2f}s audio, decode RTF {row['decode_rtf']}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
