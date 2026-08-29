"""CV-G1-E0017: ONNX export of the CaedralRegressor inference graph.

Exports the supervised-duration mel regressor (caedral_voice/model/regressor.py,
E0013) to three STATIC opset-17 ONNX graphs -- one per phoneme-length bucket
T_x in {50, 200, 800} -- using RANDOM-INIT weights, so a trained checkpoint
drops in later with zero changes (state_dict loads into the wrapper unchanged;
the wrapper only re-exposes the model's own modules).

Design decisions (per RUNTIME-02 plan docs/architecture/ONNX-EXPORT-PLAN-G1.md
and the E0017 mission):

* Single graph per bucket covering phoneme_ids + durations + speaker/style ->
  mel (the durations-override inference path; the duration PREDICTOR head is
  kept as an auxiliary output so the runtime can close the loop without a
  second graph).
* MAS / flow / losses_for_training are never touched (plan hazard H1): the
  wrapper calls only _encode / conditioning / duration_predictor / _decode.
* No randomness in-graph (H5), no python scalars (H6): speaker_id and style_id
  are int64 tensor inputs; max_duration is baked as an export-time constant.
* expand_tokens (python loop + .tolist(), untraceable) is replaced IN-GRAPH by
  a loop-free B=1 construction over primitive ops proven on ORT 1.29 CPU:
      ends = cumsum(d)                     # (T_x,)
      pos  = arange(total)                 # dynamic Range op
      src[p] = sum(pos >= ends)            # searchsorted-right == source phone
      h_exp = Gather(h0, src, axis=0)      # index_select along T_x
  This is the H3 *fallback* path (in-graph expansion); the plan's primary
  host-side option was not needed because the fallback traced and ran exactly.
  Memory note: the comparison matrix is O(T_x * T_y) int64 transiently.
* The mel output axis is DYNAMIC (T_y = sum of clamped durations). Static-bucket
  tracing bakes only T_x; RoPE tables export as Range/Cos/Sin and attention via
  SDPA decomposition stays shape-generic (both probed before this experiment).
* B=1 contract: batch dim fixed at 1 (matches RUNTIME-02 target); with exact
  expansion the forward()'s mask multiply is identity at B=1 and is omitted.

Parity protocol: identical byte-level inputs to torch eager (fp32 CPU eval) and
ORT CPUExecutionProvider; gate = mel max-abs-diff <= 3e-4 per bucket (plan
Gate-B level for the deep decoder stack).

Usage: .venv/Scripts/python.exe experiments/CV-G1-E0017-onnx-export/export_regressor.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from caedral_voice.model.regressor import (  # noqa: E402
    CaedralRegressor,
    CaedralRegressorConfig,
)

BUCKETS_ALL = (50, 200, 800)
# per bucket: list of (case_name, valid_phoneme_extent, duration_hi_exclusive)
CASES = {
    50: [("full", 50, 20), ("partial", 35, 20)],
    200: [("full", 200, 12), ("partial", 120, 20)],
    800: [("full", 800, 5), ("partial", 450, 12)],
}
MEL_GATE = 3e-4          # max-abs-diff gate (mel)
LOGD_GATE = 1e-4         # informational gate (log_duration)
WARMUP = 2
TIMED_ITERS = 8


class RegressorOnnxWrapper(nn.Module):
    """Trace-friendly module wrapping the CaedralRegressor inference path.

    Contract (B=1):
        phoneme_ids : (1, T_b) int64   right-zero-padded token ids
        speaker_id  : (1,)     int64
        style_id    : (1,)     int64   neutral style = config.n_styles (last row)
        durations   : (1, T_b) int64   frames/phone, pad slots = 0
    Returns:
        mel         : (1, T_y, n_mels) f32, T_y = sum(clamped durations)
        log_duration: (1, T_b) f32 (auxiliary; predictor output pre-expansion)

    Duration handling mirrors CaedralRegressor.forward verbatim: entries >0
    clamp to [1, max_duration]; zeros stay 0 and emit no frames. Expansion is
    the loop-free gather construction described at module level (replaces the
    untraceable expand_tokens python loop; same repeat_interleave ordering).
    """

    def __init__(self, model: CaedralRegressor):
        super().__init__()
        self.model = model                      # submodule -> exported weights
        self.max_duration = int(model.config.max_duration)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_id: torch.Tensor,
        style_id: torch.Tensor,
        durations: torch.Tensor,
    ):
        m = self.model
        # ---- encode + conditioning (no masks, no scalars) ------------------
        h = m._encode(phoneme_ids)              # (1, T_b, D)
        s = m.conditioning(speaker_id, style_id)  # (1, C_cond)
        log_d = m.duration_predictor(h, s)      # (1, T_b) aux output

        # ---- normalize durations exactly like CaedralRegressor.forward ----
        d = torch.where(durations > 0, durations.clamp_min(1), durations)
        d = d.clamp_max(self.max_duration)      # (1, T_b) int64

        # ---- in-graph length regulator (B=1) --------------------------------
        d1 = d.squeeze(0)                       # (T_b,)
        ends = torch.cumsum(d1, dim=0)          # (T_b,)
        total = ends[-1]                        # 0-dim int64 tensor (dynamic)
        pos = torch.arange(total)               # (T_y,) dynamic Range
        cmp = (pos[:, None] >= ends[None, :]).to(torch.int64)
        src = cmp.sum(dim=1)                    # (T_y,) source phoneme index
        h_exp = torch.index_select(h.squeeze(0), 0, src).unsqueeze(0)
        # (1, T_y, D) -- same frame order as repeat_interleave

        # ---- decoder (pitch/energy projections excluded: v0 frozen-off) ----
        mel = m._decode(h_exp, s)               # (1, T_y, n_mels)
        return mel, log_d


def make_inputs(rng: np.random.Generator, cfg: CaedralRegressorConfig,
                bucket: int, extent: int, dur_hi: int):
    """Byte-level fixture shared by torch reference and ORT."""
    ids = rng.integers(1, cfg.vocab_size, size=(1, bucket)).astype(np.int64)
    ids[0, extent:] = 0
    dur = rng.integers(1, dur_hi, size=(1, bucket)).astype(np.int64)
    dur[0, extent:] = 0
    return ids, dur


def run_torch(wrapper: RegressorOnnxWrapper, ids, dur, spk, sty):
    with torch.inference_mode():
        mel, logd = wrapper(
            torch.from_numpy(ids),
            torch.tensor([spk], dtype=torch.int64),
            torch.tensor([sty], dtype=torch.int64),
            torch.from_numpy(dur),
        )
    return mel.numpy(force=True), logd.numpy(force=True)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "p99_abs": float(np.percentile(d, 99))}


def measure(fn, warmup: int = WARMUP, iters: int = TIMED_ITERS) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "min_ms": ts[0],
            "mean_ms": sum(ts) / len(ts)}


def graph_info(path: Path) -> dict:
    import onnx
    g = onnx.load(str(path))
    ops: dict[str, int] = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    init_bytes = sum(np.prod(list(i.dims or [1])) *
                     (4 if i.data_type in (1, 11) else 1) for i in g.graph.initializer)
    opsets = {o.domain or "ai.onnx": o.version for o in g.opset_import}
    return {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 3),
            "ir_version": g.ir_version, "opsets": opsets,
            "n_nodes": len(g.graph.node),
            "n_initializers": len(g.graph.initializer),
            "initializer_mb": round(init_bytes / 1e6, 2),
            "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS_ALL))
    ap.add_argument("--iters", type=int, default=TIMED_ITERS)
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "experiments" / "CV-G1-E0017-onnx-export"
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=str(REPO), check=True).stdout.strip()
    except Exception:
        git_rev = "unknown"

    meta = {
        "experiment": "CV-G1-E0017-onnx-export",
        "task": "ONNX-REG-01",
        "timestamp_utc": stamp,
        "git_rev": git_rev,
        "torch": torch.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "torch_num_threads": None,
        "ort_intra_op_num_threads": 4,
        "opset_version": 17,
        "weights": "random-init seed 1234 (checkpoint drop-in contract)",
        "gate_mel_max_abs": MEL_GATE,
    }
    try:
        sys.path.insert(0, str(REPO / "benchmarks"))
        from harness.system_info import capture
        meta["system"] = capture()
    except Exception as exc:  # pragma: no cover
        meta["system_capture_error"] = repr(exc)

    torch.manual_seed(1234)
    model = CaedralRegressor(CaedralRegressorConfig()).eval()
    wrapper = RegressorOnnxWrapper(model).eval()
    assert not wrapper.model.use_pitch and not wrapper.model.use_energy
    print(f"model params {model.count_params()/1e6:.2f}M "
          f"(pitch={wrapper.model.use_pitch} energy={wrapper.model.use_energy})")

    rows: list[dict] = []
    graphs: list[dict] = []
    all_pass = True

    for bucket in args.buckets:
        cfg = wrapper.model.config
        rng = np.random.default_rng(1234 + bucket)
        cases = CASES[bucket]
        # fixtures: case 0 doubles as the static-trace example
        fixtures = {}
        for name, extent, hi in cases:
            ids, dur = make_inputs(rng, cfg, bucket, extent, hi)
            # fixture tuple order everywhere below: (ids, dur, speaker, style)
            fixtures[name] = (
                ids, dur,
                7 if name == "full" else 3,               # speaker id
                cfg.n_styles if name == "full" else 3,    # style (neutral=last)
            )

        ex_ids, ex_dur, ex_spk, ex_sty = fixtures[cases[0][0]]
        onnx_path = out_dir / f"regressor-t{bucket}.onnx"
        t0 = time.perf_counter()
        torch.onnx.export(
            wrapper,
            (torch.from_numpy(ex_ids),
             torch.tensor([ex_spk], dtype=torch.int64),
             torch.tensor([ex_sty], dtype=torch.int64),
             torch.from_numpy(ex_dur)),
            str(onnx_path),
            input_names=["phoneme_ids", "speaker_id", "style_id", "durations"],
            output_names=["mel", "log_duration"],
            dynamic_axes={"mel": {1: "t_mel"}},
            opset_version=17,
            do_constant_folding=True,
        )
        export_s = time.perf_counter() - t0
        gi = graph_info(onnx_path)
        gi["export_seconds"] = round(export_s, 1)
        graphs.append(gi)
        print(f"[t{bucket}] exported {onnx_path.name} in {export_s:.1f}s "
              f"({gi['size_mb']} MB, {gi['n_nodes']} nodes)")

        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(onnx_path), so,
                                    providers=["CPUExecutionProvider"])

        def ort_run(fx):
            ids, dur, spk, sty = fx
            return sess.run(None, {
                "phoneme_ids": ids,
                "speaker_id": np.array([spk], dtype=np.int64),
                "style_id": np.array([sty], dtype=np.int64),
                "durations": dur,
            })

        bucket_gate_pass = True
        for name, fx in fixtures.items():
            ref_mel, ref_logd = run_torch(wrapper, *fx)
            out_mel, out_logd = ort_run(fx)
            shape_ok = out_mel.shape == ref_mel.shape
            ms = diff_stats(ref_mel, out_mel)
            md = diff_stats(ref_logd, out_logd)
            row_pass = bool(shape_ok and ms["max_abs"] <= MEL_GATE)
            bucket_gate_pass &= row_pass
            ty = int(ref_mel.shape[1])
            audio_s = round(ty * cfg.hop_length / cfg.sample_rate, 3)
            rows.append({
                "kind": "parity", "bucket": bucket, "case": name,
                "t_mel_frames": ty, "audio_seconds": audio_s,
                "shape_match": shape_ok, "mel": ms, "log_duration": md,
                "gate_mel_max_abs": MEL_GATE, "pass": row_pass,
                "stamp": stamp,
            })
            print(f"[t{bucket}/{name}] Ty={ty} ({audio_s}s audio) "
                  f"mel max-abs {ms['max_abs']:.3e} mean {ms['mean_abs']:.3e} "
                  f"p99 {ms['p99_abs']:.3e} logd {md['max_abs']:.3e} "
                  f"-> {'PASS' if row_pass else 'FAIL'}")

        # ---- cross-check: wrapper(torch) vs CaedralRegressor.forward -------
        for name, fx in fixtures.items():
            ids, dur, spk, sty = fx
            w_mel, _ = run_torch(wrapper, *fx)
            with torch.inference_mode():
                fwd = wrapper.model.forward(
                    torch.from_numpy(ids), speaker_id=spk,
                    style_id=None if sty == cfg.n_styles else sty,
                    speed=1.0, durations=torch.from_numpy(dur))
            m_f = diff_stats(w_mel, fwd["mel"].numpy(force=True))
            agree = int(fwd["n_mel_frames"]) == w_mel.shape[1]
            rows.append({"kind": "wrapper-vs-forward", "bucket": bucket,
                         "case": name, "mel": m_f, "frames_agree": agree,
                         "stamp": stamp})
            if m_f["max_abs"] > 1e-6 or not agree:
                all_pass = False
            print(f"[t{bucket}/{name}] wrapper vs model.forward: "
                  f"max-abs {m_f['max_abs']:.3e}, frames_agree={agree}")

        if not bucket_gate_pass:
            all_pass = False

        if not args.skip_latency:
            ids, dur, spk, sty = fixtures[cases[0][0]]
            torch.set_num_threads(4)
            meta["torch_num_threads"] = torch.get_num_threads()

            def torch_fn():
                run_torch(wrapper, ids, dur, spk, sty)

            lt = measure(torch_fn, iters=args.iters)

            def ort_fn():
                ort_run(fixtures[cases[0][0]])

            lo = measure(ort_fn, iters=args.iters)
            ty = None
            with torch.inference_mode():
                ty = int(run_torch(wrapper, ids, dur, spk, sty)[0].shape[1])
            audio_s = ty * cfg.hop_length / cfg.sample_rate
            rows.append({
                "kind": "latency", "bucket": bucket, "case": cases[0][0],
                "t_mel_frames": ty, "audio_seconds": round(audio_s, 3),
                "torch_eager": lt, "ort_cpu_4thr": lo,
                "speedup_ort_vs_torch": round(lt["median_ms"] / lo["median_ms"], 2),
                "rtf_torch": round(lt["median_ms"] / 1000.0 / audio_s, 4),
                "rtf_ort": round(lo["median_ms"] / 1000.0 / audio_s, 4),
                "stamp": stamp,
            })
            print(f"[t{bucket}] latency full-case: torch median "
                  f"{lt['median_ms']:.1f} ms | ORT {lo['median_ms']:.1f} ms "
                  f"| speedup x{lt['median_ms']/lo['median_ms']:.2f}")

    jsonl = res_dir / f"parity-{stamp}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta["all_gates_pass"] = all_pass
    meta["graphs"] = graphs
    meta_path = res_dir / f"parity-{stamp}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {jsonl}")
    print(f"wrote {meta_path}")
    print("ALL GATES PASS" if all_pass else "GATE FAILURE(S) PRESENT")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
