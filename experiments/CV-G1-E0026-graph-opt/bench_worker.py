"""CV-G1-E0026-graph-opt: RTF benchmark worker (fresh subprocess per config).

Two modes:
  --engine MODEL THREADS
      Loads ONE model via CaedralVoiceKokoroEngine at a pinned thread count,
      warms up (pays NLTK g2p + ORT graph init), discards one cold synth,
      renders the frozen sentence 10 times (voice=af_sky), reports walls.
      Methodology identical to CV-G1-E0024 quant_bench.py.

  --raw MODEL THREADS BUCKET_LEN TOKENS_JSON EXPECTED_SAMPLES
      Static-shape bucket benchmark: pads the real token window to
      BUCKET_LEN with pad token 0, runs the fused graph directly, trims the
      output tail to EXPECTED_SAMPLES (the unpadded useful audio). Reports
      walls including pad+trim overhead.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SENTENCE = "Hello. This is the first generation of Caedral Voice."
VOICE = "af_sky"
REPS = 10


def bench_engine(model_path: str, threads: int) -> dict:
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
    voices = str(REPO / "models/kokoro/voices-v1.0.bin")

    t0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(model_path, voices, threads=threads)
    load_s = time.perf_counter() - t0
    g2p_warmup_s = engine.warmup()

    tc = time.perf_counter()
    cold_audio = engine.synth(SENTENCE, VOICE)
    cold_wall = time.perf_counter() - tc

    walls, audio = [], None
    sr = engine.sample_rate
    for _ in range(REPS):
        t = time.perf_counter()
        audio = engine.synth(SENTENCE, VOICE)
        walls.append(round(time.perf_counter() - t, 4))
    phonemes, _ = engine.phonemize(SENTENCE)
    n_tok = len(engine._kokoro.tokenizer.tokenize(phonemes))
    med = statistics.median(walls)
    return {
        "mode": "engine", "model": model_path, "threads": threads,
        "load_s": round(load_s, 3),
        "g2p_warmup_s": round(g2p_warmup_s, 3),
        "cold": {"wall_s": round(cold_wall, 4),
                 "rtf": round(cold_wall / (len(cold_audio) / sr), 4)},
        "warm": {"reps": REPS, "walls_s": walls,
                 "median_wall_s": round(med, 4),
                 "audio_s": round(len(audio) / sr, 4),
                 "rtf_median": round(med / (len(audio) / sr), 4)},
        "n_tokens_unpadded": n_tok,
        "phonemes": phonemes,
    }


def bench_raw(model_path: str, threads: int, bucket_len: int,
              tokens_json: str, expected_samples: int) -> dict:
    import numpy as np
    from pathlib import Path
    import onnxruntime as ort

    repo = Path(__file__).resolve().parents[2]
    voices = np.load(str(repo / "models/kokoro/voices-v1.0.bin"))
    tokens = json.loads(tokens_json)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0

    # same style-row rule as kokoro_onnx: index by unpadded token count;
    # voices[voice][i] is already [1, 256] - no extra axis
    style = np.asarray(voices[VOICE][max(0, min(len(tokens), 510) - 1)],
                       dtype=np.float32)
    speed = np.array([1.0], dtype=np.float32)
    padded = ([0] + list(tokens) + [0])
    padded = padded + [0] * (bucket_len - len(padded))
    assert len(padded) == bucket_len, (len(padded), bucket_len)
    tok_arr = np.array([padded], dtype=np.int64)

    def run():
        out = sess.run(["audio"], {"tokens": tok_arr, "style": style,
                                   "speed": speed})[0]
        return np.asarray(out).reshape(-1)[:expected_samples]

    tc = time.perf_counter()
    cold = run()
    cold_wall = time.perf_counter() - tc

    walls, full_lens = [], []
    for _ in range(REPS):
        t = time.perf_counter()
        out = sess.run(["audio"], {"tokens": tok_arr, "style": style,
                                   "speed": speed})[0]
        full_lens.append(int(out.size))
        trimmed = np.asarray(out).reshape(-1)[:expected_samples]
        walls.append(round(time.perf_counter() - t, 4))

    med = statistics.median(walls)
    return {
        "mode": "raw-bucket", "model": model_path, "threads": threads,
        "bucket_len": bucket_len, "load_s": round(load_s, 3),
        "expected_samples": int(expected_samples),
        "produced_samples_median": statistics.median(full_lens),
        "cold": {"wall_s": round(cold_wall, 4),
                 "rtf": round(cold_wall / (expected_samples / 24000), 4)},
        "warm": {"reps": REPS, "walls_s": walls,
                 "median_wall_s": round(med, 4),
                 "audio_useful_s": round(expected_samples / 24000, 4),
                 "rtf_median": round(med / (expected_samples / 24000), 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", nargs=2, metavar=("MODEL", "THREADS"))
    ap.add_argument("--raw", nargs=5,
                    metavar=("MODEL", "THREADS", "BUCKET_LEN",
                             "TOKENS_JSON", "EXPECTED_SAMPLES"))
    args = ap.parse_args()

    if args.engine:
        result = bench_engine(args.engine[0], int(args.engine[1]))
    elif args.raw:
        result = bench_raw(args.raw[0], int(args.raw[1]), int(args.raw[2]),
                           args.raw[3], int(args.raw[4]))
    else:
        raise SystemExit("nothing to do")
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
