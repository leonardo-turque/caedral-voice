"""E0028 follow-up: (A) rigorous ABAB wall-clock timing, (B) OpenVINO
compatibility probe of the split model (E0025's crash node is pruned here).
Appends findings to results.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from native_istft_render import (  # noqa: E402
    MODEL, VOICES, SplitRenderer, load_inverse_basis, native_istft,
)

TEXT = "Hello. This is the first generation of Caedral Voice."
THREADS = 4


def median(xs):
    return statistics.median(xs)


def main():
    import onnxruntime as rt
    import kokoro_onnx

    res_path = EXP / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = THREADS
    sess_full = rt.InferenceSession(str(MODEL), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    k = kokoro_onnx.Kokoro.from_session(sess_full, str(VOICES))
    rend = SplitRenderer(EXP / "kokoro-v1.0-noistft.onnx", VOICES,
                         threads=THREADS)
    basis = load_inverse_basis()

    phonemes = k.tokenizer.phonemize(TEXT)
    tokens = k.tokenizer.tokenize(phonemes)
    style = k.get_voice_style("af_heart")
    style_row = np.asarray(style[min(len(tokens), len(style)) - 1],
                           dtype=np.float32)
    feed = {
        k._tokens_input: np.array([[0, *tokens, 0]],
                                  dtype=k._input_dtypes[k._tokens_input]),
        "style": style_row.reshape(1, -1),
        "speed": np.array([1.0], dtype=np.float32),
    }

    # ---- A: ABAB interleaved timing, 3 independent rounds ---------------
    for _ in range(2):  # global warmup both paths
        sess_full.run(["audio"], feed)
        spec = rend.infer_raw(tokens, style_row, 1.0)
        native_istft(spec, basis)

    rounds = []
    for r in range(3):
        tf, ts = [], []
        for _ in range(4):  # A B A B ...
            t0 = time.perf_counter(); sess_full.run(["audio"], feed)
            tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            s = rend.infer_raw(tokens, style_row, 1.0)
            tn = time.perf_counter() - t0
            t0 = time.perf_counter(); native_istft(s, basis)
            tn += time.perf_counter() - t0
            ts.append(tn)
        rounds.append({
            "round": r + 1,
            "full_median_s": round(median(tf), 4),
            "split_native_median_s": round(median(ts), 4),
            "ratio_split_over_full":
                round(median(ts) / median(tf), 4),
        })
        print(rounds[-1])

    med_full = median([x["full_median_s"] for x in rounds])
    med_split = median([x["split_native_median_s"] for x in rounds])
    abab = {
        "method": "3 interleaved ABAB rounds x4 reps each after shared "
                  "warmup; same process, alternating order per rep",
        "rounds": rounds,
        "grand_median_full_s": round(med_full, 4),
        "grand_median_split_native_s": round(med_split, 4),
        "grand_ratio_split_over_full": round(med_split / med_full, 4),
        "verdict_speed": "KILL" if med_split >= med_full else "KEEP",
    }
    results["timing"]["abab_rigorous"] = abab
    print(f"ABAB grand: full {med_full:.4f}s split+native {med_split:.4f}s "
          f"ratio {med_split/med_full:.4f}")

    # ---- B: OpenVINO probe on the SPLIT model ---------------------------
    ov_result = {"skipped": False}
    try:
        import openvino as ov
        core = ov.Core()
        t0 = time.perf_counter()
        ov_model = ov.convert_model(str(EXP / "kokoro-v1.0-noistft.onnx"))
        t_conv = time.perf_counter() - t0
        t0 = time.perf_counter()
        compiled = core.compile_model(ov_model, "CPU",
                                      config={"PERFORMANCE_HINT": "LATENCY",
                                              "INFERENCE_NUM_THREADS": THREADS})
        t_comp = time.perf_counter() - t0
        out_name = next(iter(compiled.outputs))
        t0 = time.perf_counter()
        spec_ov = np.asarray(compiled({k._tokens_input: feed[k._tokens_input],
                                       "style": feed["style"],
                                       "speed": feed["speed"]})[out_name])
        t_first = time.perf_counter() - t0
        audio_ov = native_istft(spec_ov, basis)
        audio_ref = np.asarray(sess_full.run(["audio"], feed)[0]).ravel()
        d = np.abs(audio_ov.astype(np.float64) -
                   audio_ref.astype(np.float64)).max()
        # warm reps
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            compiled({k._tokens_input: feed[k._tokens_input],
                      "style": feed["style"], "speed": feed["speed"]})
            times.append(time.perf_counter() - t0)
        ov_result = {
            "skipped": False,
            "convert_s": round(t_conv, 2),
            "compile_s": round(t_comp, 2),
            "first_infer_s": round(t_first, 3),
            "warm_median_s": round(median(times), 4),
            "spec_shape": list(spec_ov.shape),
            "end_to_end_maxdiff_vs_ort_full": float(d),
            "status": "PASS" if d < 1e-3 else "CHECK",
            "note": "E0025 crashed inside istft/stft/Expand; that subgraph "
                    "is absent from the split model",
        }
        print("OpenVINO:", ov_result)
    except Exception as e:  # noqa: BLE001
        ov_result = {"skipped": False, "status": "FAIL", "error": repr(e)}
        print("OpenVINO probe failed:", e)
    results["openvino_split_probe"] = ov_result

    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("updated", res_path)


if __name__ == "__main__":
    main()
