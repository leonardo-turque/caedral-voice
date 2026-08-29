"""CLOSE-GAPS-01 GAP 2: OpenVINO on the noistft (generator-split) model.

E0028 proved the noistft ONNX runs under ORT and its E0028 OV probe showed
large spec divergence; this closes the gap with the focused gate:
    ORT noistft vs OV noistft on the SAME model file, same frozen feed.
Gates: corr(flattened outputs) < 0.99 -> KILL (quality risk);
       equivalent AND RTF improves >20% -> KEEP for benchmark.
RTF denominator: audio-equivalent seconds = frames * 5 / 24000 (E0028: T=18961
frames <-> 94800 samples @24 kHz).
"""
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NOISTFT = REPO / "experiments/CV-G1-E0028-istft/kokoro-v1.0-noistft.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"
OUT = Path(__file__).resolve().parent

SENTENCE = "Hello. This is the first generation of Caedral Voice."
VOICE = "af_sky"
THREADS = 4
WARMUP = 2
REPS = 6

CORR_GATE = 0.99
RTF_GAIN_GATE = 20.0  # percent


def _emit(tag: str, payload: dict) -> None:
    print(tag + json.dumps(payload, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- child modes
def cmd_convert() -> None:
    import openvino as ov
    t0 = time.perf_counter()
    model = ov.convert_model(str(NOISTFT))
    xml = OUT / "noistft.xml"
    ov.save_model(model, str(xml), compress_to_fp16=False)
    _emit("CONVERT_JSON:", {
        "ok": True,
        "convert_s": round(time.perf_counter() - t0, 2),
        "ir_mb": round((xml.stat().st_size +
                        xml.with_suffix(".bin").stat().st_size) / 2**20, 2)})


def cmd_worker(backend: str) -> dict:
    """Render frozen sentence through noistft model, report walls + output."""
    import numpy as np
    t0 = time.perf_counter()
    if backend == "ort":
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        sess = rt.InferenceSession(str(NOISTFT), sess_options=opts,
                                   providers=["CPUExecutionProvider"])

        def infer(feeds):
            return np.asarray(sess.run(None, feeds)[0])
    elif backend == "ov":
        import openvino as ov
        core = ov.Core()
        cm = core.compile_model(str(OUT / "noistft.xml"), "CPU",
                                {"INFERENCE_NUM_THREADS": str(THREADS)})
        req = cm.create_infer_request()

        def infer(feeds):
            res = req.infer(feeds)
            key = next(iter(res.keys()))
            return np.asarray(res[key])
    else:
        raise ValueError(backend)
    load_s = time.perf_counter() - t0

    if backend == "ort":
        # build the frozen feed once (real kokoro_onnx tokenizer), share it
        import kokoro_onnx
        k = kokoro_onnx.Kokoro.from_session(sess, str(VOICES))
        phonemes = k.tokenizer.phonemize(SENTENCE)
        tokens = k.tokenizer.tokenize(phonemes)
        style = k.get_voice_style(VOICE)
        row = int(min(len(tokens), len(style)) - 1)
        feeds = {
            "tokens": np.array([[0, *tokens, 0]], dtype=np.int64),
            "style": np.asarray(style[row], dtype=np.float32).reshape(1, -1),
            "speed": np.array([1.0], dtype=np.float32),
        }
        np.savez(OUT / "feed.npz", **feeds)
    else:
        feeds = {kk: v for kk, v in np.load(OUT / "feed.npz").items()}

    cold = infer(feeds)                     # discarded cold call
    walls, out = [], None
    for _ in range(WARMUP):
        infer(feeds)
    for _ in range(REPS):
        t = time.perf_counter()
        out = infer(feeds)
        walls.append(round(time.perf_counter() - t, 4))

    audio_eq_s = out.shape[-1] * 5 / 24000
    np.save(OUT / f"spec_{backend}.npy",
            out.astype(np.float32))
    return {"backend": backend, "load_s": round(load_s, 2),
            "out_shape": list(out.shape), "absmax": float(np.abs(out).max()),
            "audio_eq_s": round(audio_eq_s, 3),
            "walls_s": walls,
            "median_wall_s": round(statistics.median(walls), 4),
            "rtf_median": round(statistics.median(walls) / audio_eq_s, 4)}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        cmd_convert()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        _emit("WORKER_JSON:", cmd_worker(sys.argv[2]))
        return

    import onnxruntime
    import openvino as ov

    results: dict = {"experiment": "CLOSE-GAPS-01/gap2",
                     "date": "2026-08-23",
                     "model": NOISTFT.name,
                     "sentence": SENTENCE, "voice": VOICE,
                     "threads": THREADS,
                     "machine": {"platform": platform.platform(),
                                 "cpus": __import__("os").cpu_count()},
                     "versions": {"onnxruntime": onnxruntime.__version__,
                                  "openvino": ov.__version__}}

    # step 1: convert in isolated subprocess (fresh process per experiment)
    if not (OUT / "noistft.xml").exists():
        p = subprocess.run([sys.executable, str(Path(__file__)),
                            "--convert"], capture_output=True, text=True,
                           timeout=420)
        line = next(l for l in p.stdout.splitlines()
                    if l.startswith("CONVERT_JSON:"))
        conv = json.loads(line[len("CONVERT_JSON:"):])
        if not conv.get("ok"):
            results.update(verdict="SKIP",
                           verdict_reason=f"convert failed: {conv}")
            _write(results)
            print(json.dumps(results)[:400])
            return
        results["conversion"] = conv

    # steps 2-4: render through both runtimes in fresh processes
    workers = {}
    for b in ("ort", "ov"):
        p = subprocess.run([sys.executable, str(Path(__file__)),
                            "--worker", b], capture_output=True, text=True,
                           timeout=600)
        line = next((l for l in reversed(p.stdout.splitlines())
                     if l.startswith("WORKER_JSON:")), None)
        if line is None:
            raise SystemExit(f"worker {b} failed rc={p.returncode}:\n"
                             f"{p.stdout[-800:]}\n{p.stderr[-1500:]}")
        w = json.loads(line[len("WORKER_JSON:"):])
        workers[b] = w

    spec_ort = np.load(OUT / "spec_ort.npy")
    spec_ov = np.load(OUT / "spec_ov.npy")

    shape_ok = spec_ort.shape == spec_ov.shape
    if not shape_ok:
        results.update(verdict="KILL",
                       verdict_reason=(f"shape mismatch ort "
                                       f"{list(spec_ort.shape)} vs ov "
                                       f"{list(spec_ov.shape)}"))
        _write(results)
        print(json.dumps({k: results[k] for k in (
            "verdict", "verdict_reason")}))
        return

    diff = np.abs(spec_ort - spec_ov)
    absmax = float(np.abs(spec_ort).max())
    denom = float(np.linalg.norm(spec_ort - spec_ort.mean())) or 1.0
    corr = float(np.corrcoef(spec_ort.ravel(), spec_ov.ravel())[0, 1])
    snr = float(10 * np.log10(np.sum(spec_ort.astype(np.float64) ** 2) /
                              (np.sum(diff.astype(np.float64) ** 2) + 1e-30)))

    rtf_gain = 100 * (workers["ort"]["rtf_median"] /
                      workers["ov"]["rtf_median"] - 1)

    results["comparison"] = {
        "shape": list(spec_ort.shape),
        "absmax_ort": round(absmax, 4),
        "maxabsdiff": round(float(diff.max()), 5),
        "rel_to_absmax": round(float(diff.max()) / absmax, 5),
        "corr": round(corr, 5),
        "snr_db": round(snr, 1),
    }
    results["rtf"] = {
        "ort_noistft_rtf": workers["ort"]["rtf_median"],
        "ov_noistft_rtf": workers["ov"]["rtf_median"],
        "median_wall_s": {b: workers[b]["median_wall_s"] for b in workers},
        "improvement_pct": round(rtf_gain, 1),
    }
    fails = []
    if corr < CORR_GATE:
        fails.append(f"corr {corr:.4f} < {CORR_GATE} (quality risk)")
    elif rtf_gain <= RTF_GAIN_GATE:
        fails.append(f"RTF improvement {rtf_gain:.1f}% <= {RTF_GAIN_GATE}%")
    results.update(verdict=("KILL" if fails else "KEEP"),
                   verdict_reason="; ".join(fails) if fails else
                   f"equivalent (corr {corr:.4f}) AND RTF +{rtf_gain:.1f}%")
    _write(results)
    slim = {k: v for k, v in results.items() if k != "date"}
    slim.pop("machine", None)
    print(json.dumps(slim, indent=1)[:1600])


def _write(results: dict) -> None:
    results.pop("spec", None)
    (OUT / "gap2_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
