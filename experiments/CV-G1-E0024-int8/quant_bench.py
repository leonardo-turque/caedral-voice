"""CV-G1-E0024-int8: INT8 dynamic quantization A/B for the Kokoro ONNX model.

Pipeline (all measurement in isolated subprocesses, mirroring E0021):
    1. --quantize : onnxruntime.quantization.quantize_dynamic(fp32 -> int8,
                    weight_type=QuantType.QInt8). Parent enforces the 15 min
                    SKIP gate via subprocess timeout.
    2. --worker   : fresh process loads ONE model (fp32 or int8) into an ORT
                    session (4 intra-op threads, CPUExecutionProvider, graph
                    opt ENABLE_ALL), renders the frozen sentence with
                    voice=af_sky 10 times (plus 1 discarded cold call), saves
                    the last render as WAV, reports per-rep walls.
    3. --score    : fresh process scores both WAVs with the repo-standard
                    dual-ASR minWER (faster-whisper small.en/base.en int8)
                    and speechmos DNSMOS ovrl (via classify_preview helpers).
    4. parent     : orchestrates 2 rounds in opposite order (fp32->int8 then
                    int8->fp32) to cancel thermal drift, aggregates medians,
                    applies KEEP/KILL gates, writes results.json +
                    conclusion.md.

Gates (from task):
    SKIP : quantize crashes or exceeds 15 min
    KILL : int8 WER > 0.30 or int8 DNSMOS < 2.80
    KEEP : otherwise (quality holds); RTF speedup decides usefulness
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

MODEL_FP32 = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES_BIN = REPO / "models/kokoro/voices-v1.0.bin"
OUT_DIR = Path(__file__).resolve().parent
INT8_PATH = OUT_DIR / "kokoro-v1.0.int8.onnx"
WAV_FP32 = OUT_DIR / "evalA_fp32.wav"
WAV_INT8 = OUT_DIR / "evalA_int8.wav"

SENTENCE = "Hello. This is the first generation of Caedral Voice."
VOICE = "af_sky"
REPS = 10
THREADS = 4
QUANT_TIMEOUT_S = 900          # task rule: >15 min quantize -> SKIP
WORKER_TIMEOUT_S = 600
SCORE_TIMEOUT_S = 1200

WER_MAX = 0.30                 # KILL gate
DNSMOS_MIN = 2.80              # KILL gate


def _emit(tag: str, payload: dict) -> None:
    print(tag + json.dumps(payload, ensure_ascii=False), flush=True)


def _extract_json_line(stdout: str, tag: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(tag):
            return json.loads(line[len(tag):])
    raise RuntimeError(f"no {tag} line in child stdout:\n{stdout[-2000:]}")


# --------------------------------------------------------------------------
# child modes
# --------------------------------------------------------------------------
def cmd_quantize() -> None:
    t0 = time.perf_counter()
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(MODEL_FP32), str(INT8_PATH),
                         weight_type=QuantType.QInt8)
        ok, err = True, None
    except Exception as exc:                       # noqa: BLE001
        ok, err = False, f"{type(exc).__name__}: {exc}"
    _emit("QUANT_JSON:", {
        "ok": ok, "error": err, "wall_s": round(time.perf_counter() - t0, 2),
        "output": str(INT8_PATH),
        "out_size_mb": round(INT8_PATH.stat().st_size / 2**20, 2)
        if ok and INT8_PATH.exists() else None,
    })


def cmd_worker(model_path: str, wav_out: str) -> dict:
    import numpy as np
    import soundfile as sf
    from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine

    t0 = time.perf_counter()
    engine = CaedralVoiceKokoroEngine(model_path, VOICES_BIN, threads=THREADS)
    load_s = time.perf_counter() - t0

    g2p_warmup_s = engine.warmup()

    # cold call: first real synth in this session (graph init cost), excluded
    # from warm stats but reported separately
    tc = time.perf_counter()
    cold_audio = engine.synth(SENTENCE, VOICE)
    cold_wall = time.perf_counter() - tc

    walls = []
    audio = None
    sr = engine.sample_rate
    for _ in range(REPS):
        t = time.perf_counter()
        audio = engine.synth(SENTENCE, VOICE)
        walls.append(round(time.perf_counter() - t, 4))

    audio_s = len(audio) / sr
    sf.write(wav_out, np.asarray(audio, dtype=np.float32), sr,
             format="WAV", subtype="FLOAT")

    return {
        "model": model_path,
        "load_s": round(load_s, 3),
        "g2p_warmup_s": round(g2p_warmup_s, 3),
        "cold": {"wall_s": round(cold_wall, 4),
                 "rtf": round(cold_wall / (len(cold_audio) / sr), 4)},
        "warm": {
            "reps": REPS,
            "walls_s": walls,
            "median_wall_s": round(statistics.median(walls), 4),
            "audio_s": round(audio_s, 4),
            "rtf_median": round(statistics.median(walls) / audio_s, 4),
        },
        "wav": wav_out,
    }


def cmd_score(wav_paths: list[str]) -> list[dict]:
    import numpy as np
    import soundfile as sf
    from classify_preview import classify_file  # repo-standard scorer

    out = []
    for p in wav_paths:
        try:
            r = classify_file(Path(p), SENTENCE)
        except ValueError:
            # headroom clamp (same fix as score_v1_candidates.py)
            data, sr = sf.read(p, dtype="float32")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(np.abs(data).max()) if data.size else 0.0
            scale = 0.90 / peak if peak > 0.9 else 1.0
            scaled = str(p) + ".scaled.wav"
            sf.write(scaled, data * scale, sr)
            r = classify_file(Path(scaled), SENTENCE)
            r["headroom_scaled"] = True
        out.append({"file": Path(p).name, **{
            k: r[k] for k in (
                "class", "min_wer", "wer_small_en", "wer_base_en",
                "dnsmos_ovrl", "hypothesis", "rep_fraction",
                "spectral_flatness")}})
    return out


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------
def run_child(args: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return proc.returncode, proc.stdout + "\n---STDERR---\n" + proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("MODEL_PATH", "WAV_OUT"))
    ap.add_argument("--score", nargs="+", metavar="WAV")
    args = ap.parse_args()

    if args.quantize:
        cmd_quantize()
        return
    if args.worker:
        _emit("RESULT_JSON:", cmd_worker(*args.worker))
        return
    if args.score:
        _emit("SCORE_JSON:", cmd_score(args.score))
        return

    # ---------------- parent ----------------
    import onnxruntime

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_mb = MODEL_FP32.stat().st_size / 2**20
    results: dict = {
        "experiment": "CV-G1-E0024-int8",
        "task_id": "QUANT-01",
        "date": time.strftime("%Y-%m-%d"),
        "frozen_sentence": SENTENCE,
        "voice": VOICE,
        "method": {
            "quantization": "onnxruntime.quantization.quantize_dynamic, "
                            "weight_type=QInt8, dynamic (activations "
                            "quantized at runtime)",
            "session": f"{THREADS} intra-op threads, inter_op=1, CPU EP, "
                       "graph opt ENABLE_ALL, fresh subprocess per model",
            "rtf": "engine.synth() wall / audio seconds, median of 10 warm "
                   "reps after 1 discarded cold call; 2 rounds in opposite "
                   "order to cancel thermal drift",
            "quality": "dual-ASR minWER (faster-whisper small.en/base.en "
                       "int8) + speechmos DNSMOS ovrl @16k (repo-standard "
                       "classify_preview scorers)",
        },
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpus": os.cpu_count()},
        "versions": {
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
        },
    }

    # ---- step 1: quantize ----
    int8_ok = INT8_PATH.exists()
    quant_info: dict = {"preexisting": bool(int8_ok)}
    if not int8_ok:
        print("[parent] quantizing fp32 -> int8 ...", flush=True)
        try:
            rc, out = run_child(["--quantize"], QUANT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            quant_info = {"ok": False,
                          "error": f"exceeded {QUANT_TIMEOUT_S}s SKIP gate"}
        else:
            qj = _extract_json_line(out, "QUANT_JSON:") if rc == 0 else \
                {"ok": False, "error": out[-1500:]}
            quant_info = {**qj}
    if not quant_info.get("ok", int8_ok):
        results["verdict"] = "SKIP"
        results["verdict_reason"] = (
            f"quantization failed/slow: {quant_info.get('error')}")
        results["quantization"] = quant_info
        (OUT_DIR / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    int8_mb = INT8_PATH.stat().st_size / 2**20
    results["size"] = {
        "fp32_mb": round(fp32_mb, 2), "int8_mb": round(int8_mb, 2),
        "reduction_pct": round(100 * (1 - int8_mb / fp32_mb), 1),
        "ratio": round(fp32_mb / int8_mb, 2),
    }
    results["quantization"] = quant_info

    # ---- steps 2-5: RTF benchmark, two opposite-order rounds ----
    models = {"fp32": str(MODEL_FP32), "int8": str(INT8_PATH)}
    rounds_spec = [("round1", ["fp32", "int8"]), ("round2", ["int8", "fp32"])]
    per_model: dict[str, dict] = {m: {"walls": [], "cold": [], "rounds": {}}
                                  for m in models}
    for rnd, order in rounds_spec:
        for m in order:
            wav = WAV_INT8 if m == "int8" else WAV_FP32
            print(f"[parent] [{rnd}] benchmarking {m} ({models[m]}) ...",
                  flush=True)
            rc, out = run_child(["--worker", models[m], str(wav)],
                                WORKER_TIMEOUT_S)
            if rc != 0:
                raise SystemExit(f"worker {m} failed:\n{out[-2500:]}")
            r = _extract_json_line(out, "RESULT_JSON:")
            per_model[m]["walls"].extend(r["warm"]["walls_s"])
            per_model[m]["cold"].append(r["cold"])
            per_model[m]["rounds"][rnd] = r["warm"]
            print(f"[parent]   {m} {rnd}: median "
                  f"{r['warm']['median_wall_s']}s rtf "
                  f"{r['warm']['rtf_median']}", flush=True)

    def agg(m: str) -> dict:
        pm = per_model[m]
        walls = pm["walls"]
        med = statistics.median(walls)
        audio_s = pm["rounds"]["round2"]["audio_s"]
        return {
            "reps_total": len(walls),
            "median_wall_s": round(med, 4),
            "audio_s": round(audio_s, 4),
            "rtf_median": round(med / audio_s, 4),
            "min_wall_s": round(min(walls), 4),
            "max_wall_s": round(max(walls), 4),
            "round_medians_s": {
                rnd: round(statistics.median(pm["rounds"][rnd]["walls_s"]), 4)
                for rnd in pm["rounds"]},
            "cold_rtf": round(statistics.median(
                [c["rtf"] for c in pm["cold"]]), 4),
            "walls_all_s": walls,
        }

    fp32_r, int8_r = agg("fp32"), agg("int8")
    speedup = fp32_r["rtf_median"] / int8_r["rtf_median"]
    results["rtf"] = {
        "fp32": fp32_r, "int8": int8_r,
        "speedup_ratio": round(speedup, 3),
        "speedup_pct": round(100 * (speedup - 1), 1),
        "note": "speedup>1 means int8 is FASTER (lower RTF)",
    }

    # ---- step 6: quality (fresh scorer subprocess over saved WAVs) ----
    print("[parent] scoring quality (whisper x2 + DNSMOS) ...", flush=True)
    rc, out = run_child(["--score", str(WAV_FP32), str(WAV_INT8)],
                        SCORE_TIMEOUT_S)
    if rc != 0:
        raise SystemExit(f"scorer failed:\n{out[-2500:]}")
    scores = _extract_json_line(out, "SCORE_JSON:")
    q = {row.pop("file").replace("evalA_", "").replace(".wav", ""): row
         for row in scores}
    results["quality"] = q

    # ---- step 7: gates + verdict ----
    i8 = q.get("int8", {})
    wer = i8.get("min_wer")
    mos = i8.get("dnsmos_ovrl")
    fails = []
    if wer is not None and wer > WER_MAX:
        fails.append(f"WER {wer} > {WER_MAX}")
    if mos is not None and mos < DNSMOS_MIN:
        fails.append(f"DNSMOS {mos} < {DNSMOS_MIN}")
    if fails:
        results["verdict"] = "KILL"
        results["verdict_reason"] = "; ".join(fails)
    elif speedup < 0.98:
        # quality gate passed, but int8 slower than fp32 is not deployable
        results["verdict"] = "KILL"
        results["verdict_reason"] = (
            f"perf regression: RTF {fp32_r['rtf_median']} -> "
            f"{int8_r['rtf_median']} ({results['rtf']['speedup_pct']:+.1f}%); "
            f"quality gates passed (WER {wer}, DNSMOS {mos}) but int8 is not "
            "deployable for latency")
    else:
        results["verdict"] = "KEEP"
        results["verdict_reason"] = (
            f"quality holds (WER {wer}, DNSMOS {mos}); RTF "
            f"{fp32_r['rtf_median']}->{int8_r['rtf_median']} "
            f"({results['rtf']['speedup_pct']:+.1f}%), size "
            f"{results['size']['fp32_mb']}MB->{results['size']['int8_mb']}MB")
        if speedup <= 1.02:
            results["verdict_reason"] += (
                " NOTE: no material speed gain - int8 useful mainly for "
                "size/memory, not latency")

    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: results[k] for k in
                      ("size", "rtf", "quality", "verdict")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
