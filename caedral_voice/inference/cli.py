"""Command-line entrypoint for the Caedral inference engine.

Usage:
  python -m caedral_voice.inference --voice voice-03 --text "..." [--out out.wav]

or equivalently:

  python -m caedral_voice.inference.cli --voice 103 --text "..."

Run-dir resolution order (first hit wins):
  1. --run-dir argument
  2. $CAEDRAL_RUN_DIR environment variable
  3. experiments/pilot-tiny-15k (canonical pilot run)
  4. newest experiments/*/checkpoints/last.pt by modification time

After synthesis the CLI prints wall time, audio duration, RTF and peak RSS.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caedral_voice.inference",
        description="Synthesize speech from text with a trained CAEDRAL-NAR "
                    "checkpoint (CPU, Vocos decode).",
    )
    p.add_argument("--voice", required=True,
                   help="voice speaker name (e.g. 103) or index (e.g. 0)")
    p.add_argument("--text", required=True, help='text to synthesize ("..." )')
    p.add_argument("--out", default=None,
                   help="output .wav path (default: artifacts/audio/"
                        "caedral-<voice>-<timestamp>.wav)")
    p.add_argument("--run-dir", default=None,
                   help="training run dir containing checkpoints/ "
                        "(default: see module docstring resolution order)")
    p.add_argument("--checkpoint", default=None,
                   help="explicit checkpoint .pt (default: <run-dir>/"
                        "checkpoints/last.pt > best.pt > newest step_*.pt)")
    p.add_argument("--config", default=None,
                   help="base model config JSON (default: configs/tiny.json; "
                        "checkpoint cfg overrides it)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="rate multiplier on predicted durations (default 1.0)")
    p.add_argument("--pitch", "--pitch-shift", dest="pitch_shift", type=float,
                   default=0.0, help="F0 shift in semitones (default 0)")
    p.add_argument("--energy", "--energy-gain", dest="energy_gain", type=float,
                   default=0.0, help="energy-track gain in dB (default 0)")
    p.add_argument("--noise-scale", type=float, default=None,
                   help="flow latent temperature (default: config value)")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads (default 4)")
    return p


def resolve_run_dir(explicit: str | None) -> Path:
    """--run-dir > $CAEDRAL_RUN_DIR > pilot-tiny-15k > newest last.pt."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("CAEDRAL_RUN_DIR")
    if env:
        return Path(env)
    canonical = REPO_ROOT / "experiments" / "pilot-tiny-15k"
    if (canonical / "checkpoints" / "phoneme_vocab.json").is_file():
        return canonical
    candidates = sorted(
        (REPO_ROOT / "experiments").glob("*/checkpoints/last.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "no run dir found: pass --run-dir or set $CAEDRAL_RUN_DIR "
            f"(searched {REPO_ROOT / 'experiments'}/*/checkpoints/last.pt)"
        )
    return candidates[-1].parents[1]


def _rss_bytes():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # pragma: no cover - psutil optional for reporting
        return None


def synth_with_peak_rss(engine, **kwargs):
    """engine.synth(**kwargs) alongside a 5 ms RSS sampler thread.

    Returns (audio, peak_rss_bytes_or_None). Sampler peaks are approximate
    under GIL pressure; good enough for a CLI report line.
    """
    baseline = _rss_bytes()
    stop = threading.Event()
    peak = [baseline or 0]

    def _sample():
        while not stop.is_set():
            rss = _rss_bytes()
            if rss is None:
                break
            if rss > peak[0]:
                peak[0] = rss
            stop.wait(0.005)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        audio = engine.synth(**kwargs)
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    reported = peak[0] if baseline is not None else None
    return audio, reported


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Local imports keep --help fast and deps lazy.
    import numpy as np
    import soundfile as sf

    from caedral_voice.inference.engine import CaedralEngine

    run_dir = resolve_run_dir(args.run_dir)
    engine = CaedralEngine.from_run(
        run_dir,
        args.checkpoint,
        args.config,
        threads=args.threads,
    )

    voice_index = engine.resolve_voice(args.voice)
    voice_name = next(
        (v["speaker_name"] for v in engine.voices if v["index"] == voice_index),
        str(voice_index),
    )

    audio, peak_rss = synth_with_peak_rss(
        engine,
        text=args.text,
        voice_index=voice_index,
        speed=args.speed,
        pitch_shift=args.pitch_shift,
        energy_gain=args.energy_gain,
        noise_scale=args.noise_scale,
    )
    stats = engine.last_stats
    assert stats is not None  # synth just ran

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_voice = "".join(c if c.isalnum() else "-" for c in voice_name)
        out_path = (REPO_ROOT / "artifacts" / "audio" /
                    f"caedral-{safe_voice}-{stamp}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, engine.sample_rate)

    rtf = stats.rtf
    peak_mb = (peak_rss / (1024 * 1024)) if peak_rss else float("nan")
    print(f"voice          : {voice_name} (index {voice_index})")
    print(f"text           : {args.text!r} -> {stats.n_phonemes} phonemes")
    print(f"checkpoint     : {engine.checkpoint_path}")
    print(f"audio          : {stats.audio_seconds:.2f} s @ {engine.sample_rate} Hz")
    print(f"wall_seconds   : {stats.wall_seconds:.3f}")
    print(f"RTF            : {rtf:.4f} ({'faster' if rtf < 1 else 'slower'} than real time)")
    print(f"peak RSS       : {peak_mb:.1f} MB (sampled during synthesis)")
    print(f"output         : {out_path}")
    if np.asarray(audio).dtype != np.float32:  # defensive; engine casts already
        logger.warning("unexpected audio dtype %s", audio.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
