"""E0009 layer-1 diagnosis: where does intelligibility die in the preview chain?

Chain under test (ground-truth audio, NO model involved):
  A) GT wav -> Vocos NATIVE feature extractor -> decode      (upper bound)
  B) GT wav -> OUR 80-band ln-mel -> naive band-interp -> Vocos  (preview path)
If B is unintelligible while A is clean, the mel ADAPTER is a primary cause
of bad previews and must be fixed before judging the model at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch
import torchaudio

OUT = REPO / "experiments" / "CV-G1-E0009-diagnosis" / "audio"
OUT.mkdir(parents=True, exist_ok=True)


def our_log_mel(audio: np.ndarray, sr: int, n_mels: int = 80,
                n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    t = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0)
    m = t(torch.from_numpy(audio.astype(np.float32))[None])
    return torch.log(m.clamp_min(1e-5)).squeeze(0)  # (80, T)


def main() -> None:
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    src = sorted((REPO / "data" / "processed" / "librittsr-smoke-10spk").rglob("*.wav"))[:3]
    assert src, "no processed librittsr wavs found"
    for wav_path in src:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        stem = Path(wav_path).stem[:24]

        # A) native roundtrip
        feats_native = vocos.feature_extractor(torch.from_numpy(audio)[None])
        with torch.no_grad():
            wav_a = vocos.decode(feats_native).squeeze(0).numpy()
        sf.write(OUT / f"A-native-{stem}.wav", wav_a, sr)

        # B) our-mel + naive interpolation
        mel80 = our_log_mel(audio, sr).numpy()          # (80, T)
        assert mel80.shape[0] == 80, f"unexpected mel layout {mel80.shape}"
        xs100 = np.linspace(0, 79, 100)
        mel100 = np.stack([np.interp(xs100, np.arange(80), mel80[:, t])
                           for t in range(mel80.shape[1])], axis=1).astype(np.float32)
        assert mel100.shape[0] == 100, f"adapter layout wrong {mel100.shape}"
        with torch.no_grad():
            wav_b = vocos.decode(torch.from_numpy(mel100)[None]).squeeze(0).numpy()
        sf.write(OUT / f"B-adapter-{stem}.wav", wav_b, sr)
        print(f"{stem}: native {len(wav_a)/sr:.1f}s | adapter {len(wav_b)/sr:.1f}s")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
