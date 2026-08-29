"""Persist fixed baseline samples for quality scoring (eval stack runs later).

Writes artifacts/audio/baselines/<baseline>/<voice>/<item_id>.wav for the same
corpus subset used in timing benchmarks. These files are gitignored; their
existence + hashes are recorded in a manifest that IS committed.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "artifacts" / "audio" / "baselines"
CORPUS = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
ITEMS = ["conv_001", "num_001", "acr_005", "het_002", "para_001"]

# espeak via loader for KittenTTS
import espeakng_loader  # noqa: E402
import phonemizer.backend.espeak.wrapper as _wrapper  # noqa: E402

_wrapper.EspeakWrapper.set_library(espeakng_loader.get_library_path())
_wrapper.EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

import soundfile as sf  # noqa: E402


def load_items():
    data = json.loads(io.open(CORPUS, encoding="utf-8").read())
    return {
        item["id"]: item["text"]
        for cat in data["categories"] for item in cat["items"]
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_wav(path: Path, samples, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def main() -> None:
    texts = load_items()
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "files": {}}

    # --- Kokoro ---
    from kokoro_onnx import Kokoro

    t0 = time.perf_counter()
    kokoro = Kokoro(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
                    str(REPO / "models/kokoro/voices-v1.0.bin"))
    print(f"kokoro loaded {time.perf_counter()-t0:.1f}s")
    for voice in ("af_heart", "am_michael"):
        for iid in ITEMS:
            samples, sr = kokoro.create(texts[iid], voice=voice, lang="en-us")
            rel = f"kokoro/{voice}/{iid}.wav"
            p = OUT / rel
            save_wav(p, samples, sr)
            manifest["files"][rel] = sha256(p)
    del kokoro

    # --- KittenTTS ---
    from kittentts import KittenTTS

    kitten = KittenTTS()
    for voice in ("expr-voice-2-m", "expr-voice-3-f"):
        for iid in ITEMS:
            audio = kitten.generate(texts[iid], voice=voice)
            rel = f"kittentts/{voice}/{iid}.wav"
            p = OUT / rel
            save_wav(p, audio, 24000)
            manifest["files"][rel] = sha256(p)

    # --- Piper (isolated subprocess) ---
    exe = REPO / "models/piper/piper/piper.exe"
    models = {"lessac": REPO / "models/piper/en_US-lessac-medium.onnx",
              "ryan": REPO / "models/piper/en_US-ryan-medium.onnx"}
    for voice, model in models.items():
        for iid in ITEMS:
            out_wav = OUT / f"piper/{voice}/{iid}.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [str(exe), "--model", str(model), "--output_file", str(out_wav)],
                input=texts[iid].encode("utf-8"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            rel = f"piper/{voice}/{iid}.wav"
            manifest["files"][rel] = sha256(OUT / rel)

    mpath = REPO / "benchmarks" / "results" / "baseline-samples-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest['files'])} wavs; manifest -> {mpath}")


if __name__ == "__main__":
    main()
