"""CV-G1-E0026-graph-opt parent orchestrator: interleaved A/B RTF benchmark.

Protocol (mirrors E0024 thermal-drift cancellation):
  - fresh subprocess per (model, threads) config via bench_worker.py
  - two rounds per thread tier in OPPOSITE order
  - per config: pooled median wall over 2x10 reps -> RTF vs own audio seconds
Configs: {fp32-baseline, kokoro-opt(onnxsim), kokoro-conservative} x {4,6}
          + shape buckets {L100,L300,L600} @6thr (raw padded-session mode)
Writes results.json in this directory.
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

FP32 = REPO / "models/kokoro/kokoro-v1.0.onnx"
OPT = REPO / "artifacts/optimized/kokoro-opt.onnx"
CONS = REPO / "artifacts/optimized/kokoro-conservative.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"
SENTENCE = "Hello. This is the first generation of Caedral Voice."

WORKER = HERE / "bench_worker.py"


def run_child(args: list[str], timeout: int = 900) -> dict:
    proc = subprocess.run(
        [sys.executable, str(WORKER), *args],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        raise SystemExit(f"worker {args} failed:\n{proc.stdout[-1500:]}\n"
                         f"{proc.stderr[-2500:]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"no RESULT_JSON from worker {args}")


def main() -> None:
    import onnxruntime

    t_start = time.perf_counter()
    results: dict = {
        "experiment": "CV-G1-E0026-graph-opt",
        "task_id": "GRAPH-OPT-01",
        "date": time.strftime("%Y-%m-%d"),
        "frozen_sentence": SENTENCE,
        "voice": "af_sky",
        "method": {
            "session": "CPUExecutionProvider, ENABLE_ALL (ORT default "
                       "graph optimization), inter_op=1",
            "rtf": "engine.synth() wall / audio seconds; median of pooled "
                   "2x10 warm reps (1 cold discarded per subprocess); two "
                   "rounds in opposite order per thread tier to cancel "
                   "thermal drift; fresh subprocess per config",
            "buckets": "raw session mode: real tokens zero-padded to bucket "
                       "length, output tail trimmed to unpadded useful "
                       "samples; RTF uses useful audio seconds",
            "baseline_reference": "RTF ~0.46 @6thr ENABLE_ALL measured in a "
                                  "prior session (task brief)",
        },
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpus": __import__("os").cpu_count()},
        "versions": {"python": platform.python_version(),
                     "onnxruntime": onnxruntime.__version__,
                     "onnx": __import__("onnx").__version__},
        "configs": {},
    }

    # ---------- engine-mode tiers ----------
    tiers = {
        4: [("fp32", FP32), ("opt", OPT), ("conservative", CONS)],
        6: [("fp32", FP32), ("opt", OPT), ("conservative", CONS)],
    }
    for threads, models in tiers.items():
        orders = [models, list(reversed(models))]
        acc: dict[str, dict] = {}
        for rnd, order in enumerate(orders, start=1):
            for name, path in order:
                key = f"{name}@t{threads}"
                print(f"[parent] round{rnd} {key} ...", flush=True)
                r = run_child(["--engine", str(path), str(threads)])
                a = acc.setdefault(key, {"walls": [], "rounds": {}, "meta": r})
                a["walls"].extend(r["warm"]["walls_s"])
                a["rounds"][f"r{rnd}"] = r["warm"]["median_wall_s"]
                print(f"[parent]   {key}: med {r['warm']['median_wall_s']}s "
                      f"rtf {r['warm']['rtf_median']}", flush=True)
        for key, a in acc.items():
            med = statistics.median(a["walls"])
            audio_s = a["meta"]["warm"]["audio_s"]
            results["configs"][key] = {
                "model": a["meta"]["model"],
                "reps_total": len(a["walls"]),
                "median_wall_s": round(med, 4),
                "audio_s": round(audio_s, 4),
                "rtf_median": round(med / audio_s, 4),
                "round_medians_s": a["rounds"],
                "cold_rtf": a["meta"]["cold"]["rtf"],
                "n_tokens_unpadded": a["meta"].get("n_tokens_unpadded"),
            }

    # ---------- bucket mode @6 ----------
    # Raw-session block with consistent denominators: every config renders
    # through sess.run directly; useful audio = RAW unpadded output length
    # of the exact-semantics dynamic model on the natural token window.
    import numpy as np
    import onnxruntime as ort

    eng_probe = run_child(["--engine", str(FP32), "6"])
    phonemes = eng_probe["phonemes"]
    del eng_probe

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 6
    opts.inter_op_num_threads = 1
    probe_sess = ort.InferenceSession(str(CONS), sess_options=opts,
                                      providers=["CPUExecutionProvider"])
    voices = np.load(str(VOICES))

    # session-less tokenizer: kokoro_onnx.Tokenizer() defaults match the
    # wrapper's (DEFAULT_VOCAB + espeak-ng), so ids are identical
    from kokoro_onnx.tokenizer import Tokenizer
    token_ids = [int(t) for t in Tokenizer().tokenize(phonemes)]

    style = np.asarray(
        voices["af_sky"][max(0, min(len(token_ids), 510) - 1)],
        dtype=np.float32)
    nat_window = np.array([[0, *token_ids, 0]], dtype=np.int64)
    speed = np.array([1.0], dtype=np.float32)
    raw_out = probe_sess.run(["audio"], {"tokens": nat_window, "style": style,
                                         "speed": speed})[0]
    expected_samples = int(np.asarray(raw_out).reshape(-1).size)
    del probe_sess
    tokens_json = json.dumps(token_ids)
    results["bucket_setup"] = {
        "n_tokens_unpadded": len(token_ids),
        "natural_window_len": len(token_ids) + 2,
        "expected_raw_samples": expected_samples,
        "expected_audio_s": round(expected_samples / 24000, 4),
        "style_row_index": max(0, min(len(token_ids), 510) - 1),
        "note": "raw-block RTFs use raw output length (no wrapper trim); "
                "not directly comparable to engine-mode rows",
    }

    raw_cfgs = [
        ("dynamic-natural@t6", str(CONS), len(token_ids) + 2),
        ("static-L100@t6",
         str(REPO / "artifacts/optimized/kokoro-static-L100.onnx"), 100),
        ("static-L300@t6",
         str(REPO / "artifacts/optimized/kokoro-static-L300.onnx"), 300),
        ("static-L512@t6",
         str(REPO / "artifacts/optimized/kokoro-static-L512.onnx"), 512),
    ]
    accb: dict[str, dict] = {}
    orders = [raw_cfgs, list(reversed(raw_cfgs))]
    for rnd, order in enumerate(orders, start=1):
        for key, path, L in order:
            print(f"[parent] round{rnd} {key} ...", flush=True)
            r = run_child(["--raw", path, "6", str(L), tokens_json,
                           str(expected_samples)], timeout=1800)
            a = accb.setdefault(key, {"walls": [], "rounds": {}, "meta": r})
            a["walls"].extend(r["warm"]["walls_s"])
            a["rounds"][f"r{rnd}"] = r["warm"]["median_wall_s"]
            print(f"[parent]   {key}: med {r['warm']['median_wall_s']}s "
                  f"rtf {r['warm']['rtf_median']}", flush=True)
    for key, a in accb.items():
        med = statistics.median(a["walls"])
        results["configs"][key] = {
            "model": a["meta"]["model"],
            "reps_total": len(a["walls"]),
            "median_wall_s": round(med, 4),
            "audio_useful_s": round(a["meta"]["warm"]["audio_useful_s"], 4),
            "produced_samples_median":
                a["meta"]["produced_samples_median"],
            "rtf_median": round(med / a["meta"]["warm"]["audio_useful_s"], 4),
            "round_medians_s": a["rounds"],
            "cold_rtf": a["meta"]["cold"]["rtf"],
        }

    # ---------- deltas ----------
    def rtf_of(key):
        c = results["configs"].get(key)
        return c["rtf_median"] if c else None

    deltas = {}
    for thr in (4, 6):
        base = rtf_of(f"fp32@t{thr}")
        for name in ("opt", "conservative"):
            k = f"{name}@t{thr}"
            v = rtf_of(k)
            if base and v:
                deltas[k] = {
                    "baseline_rtf": base, "variant_rtf": v,
                    "speedup_pct": round(100 * (base / v - 1), 2),
                    "rtf_delta": round(v - base, 4)}
    for L in (100, 300, 512):
        dyn = rtf_of("dynamic-natural@t6")
        stc = rtf_of(f"static-L{L}@t6")
        if dyn and stc:
            deltas[f"static-L{L}-vs-dynamic-natural"] = {
                "dynamic_rtf": dyn, "static_rtf": stc,
                "speedup_pct": round(100 * (dyn / stc - 1), 2)}

    target_gate, target_stretch = 0.40, 0.20
    best_key = None
    best_rtf = None
    for key, c in results["configs"].items():
        if c.get("rtf_median") and (
                best_rtf is None or c["rtf_median"] < best_rtf):
            best_key, best_rtf = key, c["rtf_median"]

    results["deltas"] = deltas
    results["targets"] = {
        "gate_rtf_max": target_gate, "stretch_rtf_max": target_stretch,
        "best_config": best_key, "best_rtf": best_rtf,
        "gate_met": bool(best_rtf is not None and best_rtf <= target_gate),
        "stretch_met": bool(best_rtf is not None
                            and best_rtf <= target_stretch),
    }
    results["wall_total_min"] = round((time.perf_counter() - t_start) / 60, 1)

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({"configs": {k: c["rtf_median"] for k, c in
                                  results["configs"].items()},
                      "deltas": deltas, "targets": results["targets"]},
                     indent=2))


if __name__ == "__main__":
    main()
