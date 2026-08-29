"""CV-G1-E0026-graph-opt step 3: numerical equivalence verification.

Loads two sessions in one process, feeds IDENTICAL inputs (real frontend
tokens for the frozen sentence at natural length, plus synthetic token sets
at other lengths; af_sky style vectors replicated from voices-v1.0.bin;
speed=1.0), compares the audio outputs elementwise.

Reports max/mean abs diff, bitwise-equal fraction, Pearson correlation and
verifies |diff| < tol (default 1e-4, audio lives in [-1, 1]).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import onnxruntime as ort

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
VOICES_BIN = REPO / "models/kokoro/voices-v1.0.bin"


def make_session(path: str, threads: int = 4) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def style_for(voice: str, n_tokens: int) -> np.ndarray:
    voices = np.load(str(VOICES_BIN))
    arr = voices[voice]           # [510, 1, 256]
    return arr[min(n_tokens, len(arr)) - 1].astype(np.float32)


def run_case(sess: ort.InferenceSession, tokens: list[int],
             voice: str = "af_sky") -> tuple[np.ndarray, dict]:
    tok = np.array([[0, *tokens, 0]], dtype=np.int64)
    style = style_for(voice, len(tok[0]))
    speed = np.array([1.0], dtype=np.float32)
    out = sess.run(["audio"], {"tokens": tok, "style": style, "speed": speed})
    return np.asarray(out[0]).reshape(-1), {
        "n_tokens_padded": int(tok.shape[1]), "audio_len": int(out[0].size)}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    d = np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64))
    corr = float(np.corrcoef(a[:n].astype(np.float64),
                             b[:n].astype(np.float64))[0, 1])
    return {
        "len_a": int(len(a)), "len_b": int(len(b)),
        "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
        "bitwise_equal_frac": float(np.mean(a[:n] == b[:n])),
        "pearson_r": corr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--pad-to", type=int, default=0,
                    help="pad token windows with 0 up to this length "
                         "(for static-shape bucket comparisons)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(26)
    # realistic: real phoneme tokens of the frozen sentence via the engine
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
        eng = CaedralVoiceKokoroEngine(args.a, str(VOICES_BIN),
                                       threads=args.threads)
        phonemes, _ = eng.phonemize(
            "Hello. This is the first generation of Caedral Voice.")
        real_tokens = eng._kokoro.tokenizer.tokenize(phonemes)[: args.pad_to or None]
        del eng
        import gc
        gc.collect()
        cases = [("frozen-sentence-real-tokens", list(real_tokens))]
    except Exception as exc:                      # noqa: BLE001
        print(f"[warn] frontend unavailable ({exc}); using synthetic only")
        cases = []
    vocab_hi = 100   # tokenizer vocab is small; keep ids in safe range
    cases += [
        ("synthetic-len30", rng.integers(1, vocab_hi, size=30).tolist()),
        ("synthetic-len80", rng.integers(1, vocab_hi, size=80).tolist()),
        ("synthetic-len150", rng.integers(1, vocab_hi, size=150).tolist()),
    ]

    sa = make_session(args.a, args.threads)
    sb = make_session(args.b, args.threads)

    results = {}
    worst = 0.0
    for name, toks in cases:
        if args.pad_to:
            # window fed to the graph is [0, *toks, 0]; static buckets
            # require exactly pad_to positions -> keep pad_to-2 real tokens
            target = max(1, args.pad_to - 2)
            toks = toks[:target]
            toks = toks + [0] * (target - len(toks))
        ra, info_a = run_case(sa, toks)
        rb, info_b = run_case(sb, toks)
        c = compare(ra, rb)
        c.update({"input": info_a})
        results[name] = c
        worst = max(worst, c["max_abs_diff"])
        print(f"[{name}] len {c['len_a']} vs {c['len_b']} "
              f"max|d|={c['max_abs_diff']:.3e} mean|d|={c['mean_abs_diff']:.3e} "
              f"bitwise={c['bitwise_equal_frac']:.4f} r={c['pearson_r']:.9f}",
              flush=True)

    ok = worst < 1e-4
    print("EQUIV_JSON:" + json.dumps({
        "label": args.label, "a": args.a, "b": args.b,
        "tolerance": 1e-4, "worst_max_abs_diff": worst, "pass": bool(ok),
        "cases": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
