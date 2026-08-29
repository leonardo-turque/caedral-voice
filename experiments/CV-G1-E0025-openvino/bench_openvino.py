"""CV-G1-E0025-openvino: OpenVINO IR vs ONNX Runtime CPU A/B for Kokoro v1.0.

Pipeline (all measurement in isolated subprocesses, mirroring E0024):
    1. --convert : openvino.tools.ovc converts models/kokoro/kokoro-v1.0.onnx
                    -> kokoro-v10.xml/.bin (fp32 preserved,
                    compress_to_fp16=False). Parent enforces SKIP gates.
    2. --worker  : fresh process drives the SAME kokoro_onnx.Kokoro wrapper
                    through two interchangeable session objects -
                    ORT InferenceSession vs an OpenVINOSessionShim around
                    ov.CompiledModel("CPU", INFERENCE_NUM_THREADS=4) - renders
                    the frozen sentence with voice=af_sky 10 times (plus 1
                    discarded cold call), saves the last render as WAV,
                    reports per-rep walls. Neither backend passes vocab_config:
                    this export carries no embedded vocab, so both fall back
                    to kokoro_onnx DEFAULT_VOCAB (same as production engine).
    3. --score   : fresh process scores both WAVs with the repo-standard
                    dual-ASR minWER (faster-whisper small.en/base.en int8)
                    and speechmos DNSMOS ovrl (classify_preview helpers).
    4. parent    : orchestrates 2 rounds in opposite order (ort->ov then
                    ov->ort) to cancel thermal drift, aggregates medians,
                    applies KEEP/KILL gates, writes results.json +
                    conclusion.md.

Gates (from task OPENVINO-01):
    SKIP : install fails or >10 min / conversion fails
    KILL : OV RTF worse than ORT, or WER not within +/-0.10 of ORT
    KEEP : otherwise
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

MODEL_ONNX = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES_BIN = REPO / "models/kokoro/voices-v1.0.bin"
OUT_DIR = Path(__file__).resolve().parent
IR_XML = OUT_DIR / "kokoro-v1.0.xml"
IR_BIN = OUT_DIR / "kokoro-v1.0.bin"
WAV_ORT = OUT_DIR / "evalA_ort.wav"
WAV_OV = OUT_DIR / "evalA_ov.wav"

SENTENCE = "Hello. This is the first generation of Caedral Voice."
WARMUP_TEXT = "Hello caedral."
VOICE = "af_sky"
REPS = 10
THREADS = 4

CONVERT_TIMEOUT_S = 600        # task rule: conversion fail -> SKIP
WORKER_TIMEOUT_S = 900
SCORE_TIMEOUT_S = 1200

WER_GAP_MAX = 0.10             # KILL gate: quality comparable to ORT


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
def cmd_convert() -> None:
    t0 = time.perf_counter()
    try:
        import openvino as ov
        model = ov.convert_model(str(MODEL_ONNX))
        ov.save_model(model, str(IR_XML), compress_to_fp16=False)
        ok, err = True, None
    except Exception as exc:                       # noqa: BLE001
        ok, err = False, f"{type(exc).__name__}: {exc}"
    _emit("CONVERT_JSON:", {
        "ok": ok, "error": err,
        "wall_s": round(time.perf_counter() - t0, 2),
        "xml": str(IR_XML),
        "ir_mb": round((IR_XML.stat().st_size + IR_BIN.stat().st_size) / 2**20, 2)
        if ok and IR_XML.exists() else None,
    })


class OpenVINOSessionShim:
    """Drop-in stand-in for ort.InferenceSession over an ov.CompiledModel.

    Implements exactly the session surface kokoro_onnx touches:
      run(None, feeds)          -> list of outputs in graph output order
      get_inputs()/get_outputs()-> objects with .name (+ .type for inputs)
      _model_path               -> attribute read by Kokoro.from_session
    """

    _TYPE_MAP = {
        "f32": "tensor(float)", "f16": "tensor(float16)",
        "f64": "tensor(double)", "i32": "tensor(int32)",
        "i64": "tensor(int64)",
    }

    def __init__(self, ir_xml: str | Path, threads: int) -> None:
        import openvino as ov
        core = ov.Core()
        self._cm = core.compile_model(
            str(ir_xml), "CPU",
            {"INFERENCE_NUM_THREADS": str(int(threads))})
        self._ir = self._cm.create_infer_request()
        self._model_path = str(ir_xml)   # Kokoro.from_session reads this

    class _IO:
        __slots__ = ("name", "type")

        def __init__(self, name: str, otype: str) -> None:
            self.name = name
            self.type = otype

    def _io(self, ports) -> list["OpenVINOSessionShim._IO"]:
        out = []
        for p in ports:
            et = str(p.element_type).split(".")[-1].rstrip(">").strip(">' ")
            out.append(self._IO(p.get_any_name(), self._TYPE_MAP.get(et,
                        "tensor(float)")))
        return out

    def get_inputs(self):
        return self._io(self._cm.inputs)

    def get_outputs(self):
        return self._io(self._cm.outputs)

    def get_modelmeta(self):
        shim = self

        class _Meta:
            custom_metadata_map: dict = {}

        return _Meta()

    def run(self, _output_names, feeds: dict):
        res = self._ir.infer(feeds)
        by_name = {}
        for k, v in res.items():
            key = k if isinstance(k, str) else k.get_any_name()
            by_name[key] = v
        return [by_name[o.get_any_name()] for o in self._cm.outputs]


def _build_phonemizer():
    """Mirror CaedralVoiceKokoroEngine.phonemize without an ORT dependency."""
    from caedral_voice.text import TextFrontend
    frontend = TextFrontend("lexicon+g2pen")

    def phonemize(kokoro, text: str) -> tuple[str, list[str]]:
        from caedral_voice.text.kokoro_map import (
            caedral_tokens_to_kokoro, kokoro_vocab_set)
        vocab = kokoro_vocab_set(kokoro.tokenizer.vocab)
        result = frontend.process(text)
        word_phonemes, skipped_all = [], []
        for w in result.words:
            mapped, skipped = caedral_tokens_to_kokoro(list(w.selected), vocab)
            word_phonemes.append(mapped)
            skipped_all.extend(skipped)
        phonemes = " ".join(word_phonemes)
        if text.rstrip().endswith("."):
            phonemes += "."
        return phonemes, skipped_all

    return frontend, phonemize


def cmd_worker(backend: str, wav_out: str) -> dict:
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro

    t0 = time.perf_counter()
    if backend == "ort":
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        session = rt.InferenceSession(str(MODEL_ONNX), sess_options=opts,
                                      providers=["CPUExecutionProvider"])
        kokoro = Kokoro.from_session(session, str(VOICES_BIN))
    elif backend == "ov":
        shim = OpenVINOSessionShim(IR_XML, THREADS)
        kokoro = Kokoro.from_session(shim, str(VOICES_BIN))
    else:
        raise ValueError(f"unknown backend {backend}")
    load_s = time.perf_counter() - t0

    # pay NLTK g2p construction outside all timings
    frontend, phonemize = _build_phonemizer()
    tg = time.perf_counter()
    phonemize(kokoro, WARMUP_TEXT)
    g2p_warmup_s = time.perf_counter() - tg

    phonemes, skipped = phonemize(kokoro, SENTENCE)

    def synth() -> np.ndarray:
        audio, _ = kokoro.create(phonemes, voice=VOICE, speed=1.0,
                                 lang="en-us", is_phonemes=True)
        return np.ascontiguousarray(audio, dtype=np.float32)

    # cold call: graph init cost in this session, excluded from warm stats
    tc = time.perf_counter()
    cold_audio = synth()
    cold_wall = time.perf_counter() - tc

    walls, audio = [], None
    sr = kokoro.config.sample_rate
    for _ in range(REPS):
        t = time.perf_counter()
        audio = synth()
        walls.append(round(time.perf_counter() - t, 4))

    audio_s = len(audio) / sr
    sf.write(wav_out, audio, sr, format="WAV", subtype="FLOAT")

    return {
        "backend": backend,
        "model": str(IR_XML) if backend == "ov" else str(MODEL_ONNX),
        "load_s": round(load_s, 3),
        "g2p_warmup_s": round(g2p_warmup_s, 3),
        "phoneme_len": len(phonemes),
        "skipped_symbols": skipped[:8],
        "cold": {"wall_s": round(cold_wall, 4),
                 "rtf": round(cold_wall / (len(cold_audio) / sr), 4)},
        "warm": {
            "reps": REPS,
            "threads": THREADS,
            "walls_s": walls,
            "median_wall_s": round(statistics.median(walls), 4),
            "audio_s": round(audio_s, 4),
            "samples": int(len(audio)),
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
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("BACKEND", "WAV_OUT"))
    ap.add_argument("--score", nargs="+", metavar="WAV")
    args = ap.parse_args()

    if args.convert:
        cmd_convert()
        return
    if args.worker:
        _emit("RESULT_JSON:", cmd_worker(*args.worker))
        return
    if args.score:
        _emit("SCORE_JSON:", cmd_score(args.score))
        return

    import onnxruntime
    import openvino as ov

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "experiment": "CV-G1-E0025-openvino",
        "task_id": "OPENVINO-01",
        "date": time.strftime("%Y-%m-%d"),
        "frozen_sentence": SENTENCE,
        "voice": VOICE,
        "method": {
            "conversion": "openvino.tools.ovc (ov.convert_model + "
                          "save_model, compress_to_fp16=False -> true fp32 "
                          "IR, no accuracy loss from compression)",
            "runtime": "ov.Core().compile_model(IR, 'CPU', "
                       "{INFERENCE_NUM_THREADS: 4}) wrapped in a session "
                       "shim so kokoro_onnx.Kokoro drives BOTH backends "
                       "identically (same tokenizer/batching/trim code); "
                       "ORT side: intra_op_num_threads=4, inter_op=1, CPU EP",
            "rtf": "kokoro.create() wall / audio seconds, median of 10 warm "
                   "reps after 1 discarded cold call; 2 rounds in opposite "
                   "order (ort->ov, ov->ort) in fresh subprocesses to cancel "
                   "thermal drift",
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
            "openvino": ov.__version__,
        },
    }

    # ---- step 0a: convert ONNX -> IR (SKIP gate) ----
    if not IR_XML.exists():
        print("[parent] converting ONNX -> OpenVINO IR ...", flush=True)
        try:
            rc, out = run_child(["--convert"], CONVERT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            results.update(verdict="SKIP", verdict_reason=(
                f"conversion exceeded {CONVERT_TIMEOUT_S}s"))
            _write(results)
            print(json.dumps(results, indent=2))
            return
        cj = _extract_json_line(out, "CONVERT_JSON:") if rc == 0 else \
            {"ok": False, "error": out[-1500:]}
        results["conversion"] = cj
        if not cj.get("ok"):
            results.update(verdict="SKIP",
                           verdict_reason=f"conversion failed: {cj.get('error')}")
            _write(results)
            print(json.dumps(results, indent=2))
            return
    else:
        results["conversion"] = {
            "ok": True, "preexisting": True,
            "ir_mb": round((IR_XML.stat().st_size + IR_BIN.stat().st_size)
                           / 2**20, 2)}
    results["size"] = {
        "onnx_mb": round(MODEL_ONNX.stat().st_size / 2**20, 2),
        "ir_mb": round((IR_XML.stat().st_size + IR_BIN.stat().st_size) / 2**20,
                       2)}

    # ---- steps 1-2: RTF benchmark, two opposite-order rounds ----
    backends = ["ort", "ov"]
    rounds_spec = [("round1", ["ort", "ov"]), ("round2", ["ov", "ort"])]
    per_backend: dict[str, dict] = {b: {"walls": [], "cold": [],
                                        "audio_s": [], "rounds": {}}
                                    for b in backends}
    for rnd, order in rounds_spec:
        for b in order:
            wav = WAV_OV if b == "ov" else WAV_ORT
            print(f"[parent] [{rnd}] benchmarking {b} ...", flush=True)
            rc, out = run_child(["--worker", b, str(wav)], WORKER_TIMEOUT_S)
            if rc != 0:
                raise SystemExit(f"worker {b} failed:\n{out[-2500:]}")
            r = _extract_json_line(out, "RESULT_JSON:")
            pb = per_backend[b]
            pb["walls"].extend(r["warm"]["walls_s"])
            pb["cold"].append(r["cold"])
            pb["audio_s"].append(r["warm"]["audio_s"])
            pb["rounds"][rnd] = r["warm"]
            pb["load_s"] = r["load_s"]
            pb["phoneme_len"] = r["phoneme_len"]
            print(f"[parent]   {b} {rnd}: median "
                  f"{r['warm']['median_wall_s']}s rtf "
                  f"{r['warm']['rtf_median']}", flush=True)

    def agg(b: str) -> dict:
        pb = per_backend[b]
        walls = pb["walls"]
        med = statistics.median(walls)
        audio_s = statistics.median(pb["audio_s"])
        return {
            "reps_total": len(walls),
            "median_wall_s": round(med, 4),
            "audio_s": round(audio_s, 4),
            "samples_median": per_backend[b]["rounds"]["round2"]["samples"],
            "rtf_median": round(med / audio_s, 4),
            "min_wall_s": round(min(walls), 4),
            "max_wall_s": round(max(walls), 4),
            "round_medians_s": {
                rnd: round(statistics.median(pb["rounds"][rnd]["walls_s"]), 4)
                for rnd in pb["rounds"]},
            "cold_rtf": round(statistics.median(
                [c["rtf"] for c in pb["cold"]]), 4),
            "load_s": pb["load_s"],
            "walls_all_s": walls,
        }

    ort_r, ov_r = agg("ort"), agg("ov")
    speedup = ort_r["rtf_median"] / ov_r["rtf_median"]
    results["rtf"] = {
        "ort_fp32": ort_r, "openvino_fp32": ov_r,
        "speedup_ratio": round(speedup, 3),
        "speedup_pct": round(100 * (speedup - 1), 1),
        "note": "speedup>1 means OpenVINO is FASTER (lower RTF); task "
                "baseline band ORT RTF 0.62-0.82",
    }
    results["frontend_equivalence"] = {
        "ort_phoneme_len": per_backend["ort"]["phoneme_len"],
        "ov_phoneme_len": per_backend["ov"]["phoneme_len"],
        "identical_frontend": (per_backend["ort"]["phoneme_len"]
                               == per_backend["ov"]["phoneme_len"]),
    }

    # ---- step 3: quality (fresh scorer subprocess over saved WAVs) ----
    print("[parent] scoring quality (whisper x2 + DNSMOS) ...", flush=True)
    rc, out = run_child(["--score", str(WAV_ORT), str(WAV_OV)],
                        SCORE_TIMEOUT_S)
    if rc != 0:
        raise SystemExit(f"scorer failed:\n{out[-2500:]}")
    scores = _extract_json_line(out, "SCORE_JSON:")
    q = {row.pop("file").replace("evalA_", "").replace(".wav", ""): row
         for row in scores}
    results["quality"] = q

    # ---- step 4: gates + verdict ----
    ort_q, ov_q = q.get("ort", {}), q.get("ov", {})
    wer_o, wer_v = ort_q.get("min_wer"), ov_q.get("min_wer")
    mos_o, mos_v = ort_q.get("dnsmos_ovrl"), ov_q.get("dnsmos_ovrl")
    fails = []
    if ov_r["rtf_median"] > ort_r["rtf_median"]:
        fails.append(
            f"OpenVINO RTF {ov_r['rtf_median']} WORSE than ORT "
            f"{ort_r['rtf_median']} -> KILL per task rule")
    if wer_o is not None and wer_v is not None and \
            abs(wer_v - wer_o) > WER_GAP_MAX:
        fails.append(
            f"WER gap |{wer_v} - {wer_o}| > {WER_GAP_MAX}")
    if fails:
        results["verdict"] = "KILL"
        results["verdict_reason"] = "; ".join(fails)
    else:
        results["verdict"] = "KEEP"
        results["verdict_reason"] = (
            f"quality comparable (WER ort {wer_o} / ov {wer_v}, DNSMOS ort "
            f"{mos_o} / ov {mos_v}); RTF ort {ort_r['rtf_median']} -> ov "
            f"{ov_r['rtf_median']} ({results['rtf']['speedup_pct']:+.1f}%); "
            f"load ort {ort_r['load_s']}s / ov {ov_r['load_s']}s")

    _write(results)
    print(json.dumps({k: results[k] for k in (
        "size", "rtf", "quality", "frontend_equivalence", "verdict",
        "verdict_reason")}, indent=2, ensure_ascii=False))


def _write(results: dict) -> None:
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
