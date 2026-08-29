"""CV-G1-E0028 main driver: iSTFT split equivalence + timing.

Produces results.json. Original model untouched; all artifacts land in this
experiment directory. Light CPU: threads=4, short sentence set.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from native_istft_render import (  # noqa: E402
    CUT_TENSOR, LIBROSA_ALPHA, MODEL, VOICES,
    SplitRenderer, build_split_model, load_inverse_basis,
    librosa_istft, native_istft,
)

SENTENCES = {
    "evalA": "Hello. This is the first generation of Caedral Voice.",
    "avx": "The CPU supports AVX2 instructions but not AVX-512.",
    "numbers": "The invoice totals $1,247.83, due within 30 days.",
    "natural": "Artificial intelligence should sound natural, expressive, and human.",
}
VOICE = "af_heart"
THREADS = 4
WARMUP = 2
REPS = 8


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def cmp_audio(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.size == b.size, f"length mismatch {a.size} vs {b.size}"
    d = np.abs(a - b)
    denom = max(float(np.abs(b).max()), 1e-12)
    snr = float(10 * np.log10((b ** 2).sum() /
                              max(float(((a - b) ** 2).sum()), 1e-24)))
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"len": int(a.size), "maxabsdiff": float(d.max()),
            "rel_to_absmax": float(d.max() / denom),
            "corr": round(corr, 9), "snr_db": round(snr, 1)}


def make_feed(k, text):
    phonemes = k.tokenizer.phonemize(text)
    tokens = k.tokenizer.tokenize(phonemes)
    style = k.get_voice_style(VOICE)
    style_row = np.asarray(style[min(len(tokens), len(style)) - 1],
                           dtype=np.float32)
    feed = {
        k._tokens_input: np.array([[0, *tokens, 0]],
                                  dtype=k._input_dtypes[k._tokens_input]),
        "style": style_row.reshape(1, -1),
        "speed": np.array([k._speed_value(1.0)], dtype=np.float32),
    }
    return tokens, style_row, feed


def profile_full_graph(runs=5):
    """ORT kernel-time profile of the fused graph; isolate STFT/iSTFT cost."""
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.enable_profiling = True
    opts.profile_file_prefix = str(EXP / "ort-profile")
    sess = rt.InferenceSession(str(MODEL), sess_options=opts,
                               providers=["CPUExecutionProvider"])
    import kokoro_onnx
    k = kokoro_onnx.Kokoro.from_session(sess, str(VOICES))
    tokens, style_row, feed = make_feed(k, SENTENCES["evalA"])
    for _ in range(WARMUP):
        sess.run(None, feed)
    for _ in range(runs):
        sess.run(None, feed)
    pf = Path(sess.end_profiling())

    events = []
    for line in pf.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and '"cat"' in line:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("cat") in ("Kernel", "Node") and ev.get("dur", 0) > 0:
                events.append(ev)

    groups = {
        "istft_tail_nodes": "/istft/stft/",
        "forward_STFT_node": "/decoder/decoder/generator/STFT",
    }
    per_call = {}
    matched_names = {}
    for gname, needle in groups.items():
        tot_us = sum(ev["dur"] for ev in events if needle in ev.get("name", ""))
        names = sorted({ev["name"] for ev in events if needle in ev.get("name", "")})
        matched_names[gname] = names
        per_call[gname + "_ms_per_call"] = round(tot_us / 1000 / runs, 3)

    # normalization chain that is provably a no-op post-trim
    noop_nodes = ["Greater", "NonZero", "ScatterND", "Where", "Expand",
                  "Div", "Equal"]
    noop_us = sum(ev["dur"] for ev in events
                  if "/istft/stft/" in ev.get("name", "")
                  and any(n in ev["name"] for n in noop_nodes))
    per_call["istft_noop_norm_chain_ms_per_call"] = round(noop_us / 1000 / runs, 3)

    total_us = sum(ev["dur"] for ev in events)
    per_call["total_kernel_ms_per_call"] = round(total_us / 1000 / runs, 1)
    per_call["matched_node_names"] = matched_names
    return per_call


def main():
    t_start = time.time()
    results = {
        "experiment": "CV-G1-E0028-istft-split",
        "date": time.strftime("%Y-%m-%d"),
        "task_id": "ISTFT-SPLIT-01",
        "model": str(MODEL.relative_to(REPO)),
        "cut_tensor": CUT_TENSOR,
        "threads": THREADS,
        "versions": {},
        "graph_findings": {},
        "equivalence": {},
        "timing": {},
        "ort_kernel_profile": None,
    }
    import onnx, scipy, librosa, kokoro_onnx
    results["versions"] = {
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "onnxruntime": rt.__version__,
        "scipy": scipy.__version__,
        "librosa": librosa.__version__,
        "kokoro_onnx": kokoro_onnx.__version__
        if hasattr(kokoro_onnx, "__version__") else "n/a",
        "numpy": np.__version__,
    }

    # ---- graph findings -------------------------------------------------
    m = onnx.load(str(MODEL), load_external_data=False)
    stft_ops = [(n.name, n.op_type, n.domain) for n in m.graph.node
                if "stft" in n.name.lower()]
    results["graph_findings"] = {
        "opset": [o.version for o in m.opset_import],
        "outputs_original": [o.name for o in m.graph.output],
        "forward_stft_node": dict(
            name="/decoder/decoder/generator/STFT", op_type="STFT",
            domain="", note="kept in-graph; feeds source-envelope branch"),
        "istft_implementation":
            "no ISTFT op: subgraph of Cos/Sin/Mul/Concat -> ConvTranspose "
            "(inverse_basis 22x1x20, stride=5, kernel=20) -> window_sum "
            "edge-normalization (Greater/NonZero/Gather/Div/ScatterND) -> "
            "Mul(4.0) -> Slice[10:-10]",
        "n_fft": 20, "hop": 5, "sample_rate": 24000,
        "window_constant_is_initializer": True,
        "stft_related_nodes": len(stft_ops),
    }

    # ---- build split model ----------------------------------------------
    split_path = EXP / "kokoro-v1.0-noistft.onnx"
    t0 = time.perf_counter()
    if not split_path.exists():
        build_split_model(split_path)
    results["timing"]["split_model_build_s"] = round(time.perf_counter() - t0, 1)

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = THREADS
    sess_full = rt.InferenceSession(str(MODEL), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    k = kokoro_onnx.Kokoro.from_session(sess_full, str(VOICES))
    rend = SplitRenderer(split_path, VOICES, threads=THREADS)
    basis = load_inverse_basis()

    # ---- equivalence over sentence set ----------------------------------
    wav_dir = EXP / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    eq = {}
    for name, text in SENTENCES.items():
        tokens, style_row, feed = make_feed(k, text)
        audio_full = np.asarray(sess_full.run(["audio"], feed)[0]).ravel()
        spec = rend.infer_raw(tokens, style_row, 1.0)
        audio_nat = native_istft(spec, basis)
        audio_lib = librosa_istft(spec)
        entry = {
            "native_exact": cmp_audio(audio_nat, audio_full),
            "librosa_x1.5": cmp_audio(audio_lib, audio_full),
        }
        sf.write(wav_dir / f"{name}.full.wav", audio_full, 24000)
        sf.write(wav_dir / f"{name}.native.wav", audio_nat, 24000)
        eq[name] = entry
        print(f"[eq] {name}: native maxdiff="
              f"{entry['native_exact']['maxabsdiff']:.3e} "
              f"corr={entry['native_exact']['corr']} | librosa "
              f"{entry['librosa_x1.5']['maxabsdiff']:.3e}")
    results["equivalence"]["waveform"] = eq

    # ---- timing benchmark (evalA) ---------------------------------------
    tokens, style_row, feed = make_feed(k, SENTENCES["evalA"])
    for _ in range(WARMUP):
        sess_full.run(["audio"], feed)
        rend.infer_raw(tokens, style_row, 1.0)
    t_full, t_split = [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); sess_full.run(["audio"], feed)
        t_full.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); spec = rend.infer_raw(tokens, style_row, 1.0)
        t_split.append(time.perf_counter() - t0)
        native_istft(spec, basis)  # keep allocator warm, untimed here
    t_nat, t_lib = [], []
    for _ in range(30):
        t0 = time.perf_counter(); native_istft(spec, basis)
        t_nat.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); librosa_istft(spec)
        t_lib.append(time.perf_counter() - t0)

    med_full = median(t_full)
    med_split = median(t_split)
    med_nat = median(t_nat)
    med_lib = median(t_lib)
    results["timing"].update({
        "sentence": SENTENCES["evalA"],
        "audio_s": round(eq["evalA"]["native_exact"]["len"] / 24000, 3),
        "warmup": WARMUP, "reps": REPS,
        "full_graph_median_s": round(med_full, 4),
        "split_graph_median_s": round(med_split, 4),
        "native_istft_median_ms": round(med_nat * 1000, 3),
        "librosa_istft_median_ms": round(med_lib * 1000, 3),
        "split_plus_native_median_s": round(med_split + med_nat, 4),
        "split_plus_librosa_median_s": round(med_split + med_lib, 4),
        "speedup_vs_full": round(med_full / (med_split + med_nat), 4),
        "in_graph_istft_implied_ms":
            round((med_full - med_split) * 1000, 2),
    })

    # ---- ORT kernel profile ---------------------------------------------
    print("[profile] profiling fused graph kernels ...")
    prof = profile_full_graph()
    results["ort_kernel_profile"] = prof

    # ---- whisper perceptual check ----------------------------------------
    print("[whisper] loading faster-whisper base.en int8 ...")
    try:
        from faster_whisper import WhisperModel
        import jiwer
        wm = WhisperModel("base.en", device="cpu", compute_type="int8")

        def tr(p):
            segs, _ = wm.transcribe(str(p), beam_size=1)
            return " ".join(s.text for s in segs).strip()

        wsp = {}
        for name, text in SENTENCES.items():
            hyp_f = tr(wav_dir / f"{name}.full.wav")
            hyp_n = tr(wav_dir / f"{name}.native.wav")
            wer_f = jiwer.wer(text.lower(), hyp_f.lower())
            wer_n = jiwer.wer(text.lower(), hyp_n.lower())
            wsp[name] = {
                "ref": text,
                "hyp_full": hyp_f, "wer_full": round(float(wer_f), 4),
                "hyp_native": hyp_n, "wer_native": round(float(wer_n), 4),
                "transcripts_identical": hyp_f == hyp_n,
                "delta_wer": round(float(wer_n) - float(wer_f), 4),
            }
            print(f"    {name}: full='{hyp_f[:48]}' wer={wer_f:.3f} | "
                  f"native='{hyp_n[:48]}' wer={wer_n:.3f}")
        results["equivalence"]["whisper"] = wsp
    except Exception as e:  # noqa: BLE001
        results["equivalence"]["whisper"] = {"error": repr(e)}
        print("whisper check failed:", e)

    results["runtime_s"] = round(time.time() - t_start, 1)
    out = EXP / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
