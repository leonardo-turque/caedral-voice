"""Merge graph stats + equivalence evidence into results.json and add
per-technique verdicts. Run AFTER run_bench.py."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

# ---- graph stats ----
stages = json.loads((HERE / "opt_stages.json").read_text(encoding="utf-8"))
b, a = stages["baseline"], stages["CD_manual_cleanup"]
cons_counts = {"nodes": 2464, "initializers": 546}
results["graph_stats"] = {
    "baseline": {"nodes": b["nodes"], "initializers": b["initializers"],
                 "file_mb": round(
                     (REPO / "models/kokoro/kokoro-v1.0.onnx")
                     .stat().st_size / 2**20, 2)},
    "kokoro-opt-onnxsim-v2": {
        "nodes": 2544, "initializers": 553,
        "op_deltas_vs_baseline": {
            "MatMul->Gemm": "+76/-76", "Reshape": "+128", "Concat": "+44",
            "Slice": "+39", "Shape": "+20", "Div": "-12", "Sqrt": "-24",
            "Unsqueeze": "-10", "Gather": "-9", "Cast": "-12",
            "ConstantOfShape": "-5"},
        "note": "onnxsim canonicalization GROWS the graph; transpose-fusion "
                "passes also break ORT load (Gemm_11 shape unification) and "
                "were disabled in v2"},
    "kokoro-conservative": {
        **cons_counts,
        "removed": "8 duplicate initializers (52 B)",
        "dead_nodes_removed": 0,
        "unused_initializers_removed": 0},
    "buckets": json.loads(
        (HERE / "bucket_stats.json").read_text(encoding="utf-8")),
    "structural_census": {
        "transpose_after_transpose_pairs": 0,
        "cast_after_cast_pairs": 0,
        "identity_nodes": 0,
        "dropout_nodes": 0,
        "unused_initializers": 0,
        "duplicate_initializers": 8,
        "unused_output_tensors": 12,
        "note": "graph is already free of naive redundant patterns; "
                "Sin x51 are Snake-activation terms (Gemm-fed, not "
                "foldable); position_embeddings cap window at 512"},
}

# ---- equivalence evidence ----
def equiv(path):
    for line in reversed(Path(path).read_text(encoding="utf-8")
                         .splitlines()):
        if line.startswith("EQUIV_JSON:"):
            d = json.loads(line[len("EQUIV_JSON:"):])
            return {k: d[k] for k in ("label", "tolerance",
                                      "worst_max_abs_diff", "pass")}
    return None

results["equivalence"] = {}
for log, key in [
    ("equiv_cons.log", "conservative-vs-baseline"),
    ("equiv_opt.log", "onnxsim-v2-vs-baseline"),
    ("equiv_L100.log", "static-L100-vs-conservative-dynamic"),
    ("equiv_L300.log", "static-L300-vs-conservative-dynamic"),
    ("equiv_L512.log", "static-L512-vs-conservative-dynamic"),
]:
    p = HERE / log
    if p.exists():
        results["equivalence"][key] = equiv(p)

# ---- verdicts ----
def rtf(key):
    c = results["configs"].get(key)
    return c["rtf_median"] if c else None

fp32_4, fp32_6 = rtf("fp32@t4"), rtf("fp32@t6")
cons_4, cons_6 = rtf("conservative@t4"), rtf("conservative@t6")
opt_4, opt_6 = rtf("opt@t4"), rtf("opt@t6")


def dec(var, base):
    if var is None or base is None:
        return "N/A"
    pct = 100 * (base / var - 1)
    if pct < 0:
        return f"KILL: slower ({pct:+.1f}%): RTF {base} -> {var}"
    return (f"{'KEEP' if pct >= 3 else 'KILL (no material gain)'}: "
            f"RTF {base} -> {var} ({pct:+.1f}%)")


results["verdicts"] = {
    "manual_zero_loss_cleanup (dedup+prune, kokoro-conservative.onnx)": {
        "quality": "BITWISE IDENTICAL to baseline (max|d|=0 on 4 input "
                   "cases incl real frontend tokens)",
        "rtf": {"t4": cons_4, "t6": cons_6},
        "decision_t4": dec(cons_4, fp32_4),
        "decision_t6_raw_pooled": dec(cons_6, fp32_6),
        "caveat_t6": "fp32@t6 round2 (0.7282) is a thermal-spike outlier; "
                     "clean-round pairing r1: cons 0.6154 vs fp32 0.6257 "
                     "(+1.7%); r2 pairing confounded by the spike. True "
                     "effect ~0-2%, not a deployable speed claim.",
        "overall": "KEEP as hygiene only (bitwise-exact, 8 dup initializers "
                   "removed); NO RTF gain beyond thermal noise",
    },
    "onnxsim_canonicalization (kokoro-opt.onnx)": {
        "quality": "FAILS equivalence gate: max|d| up to 0.104 "
                   "(r>=0.9964) - chaotic amplification through "
                   "transformer+vocoder; also needed transpose-fusion "
                   "passes disabled to even load (Gemm_11 shape error)",
        "rtf": {"t4": opt_4, "t6": opt_6},
        "decision": "KILL (violates zero-loss rule; its +10.7% @t6 vs the "
                    "spiked fp32 baseline sits in the same thermal-noise "
                    "band as conservative - no reliable gain)",
    },
    "shape_specialization_buckets (static L100/L300/L512)": {
        "quality": "FAILS strict equivalence vs dynamic natural render "
                   "(max|d| up to 0.43 from ORT plan changes under static "
                   "shape declarations; graphs are pure dim edits of an "
                   "exact base)",
        "rtf_vs_dynamic_natural": {
            "L100": results["deltas"]["static-L100-vs-dynamic-natural"],
            "L300": results["deltas"]["static-L300-vs-dynamic-natural"],
            "L512": results["deltas"]["static-L512-vs-dynamic-natural"],
        },
        "decision": "KILL decisively: padding compute swamps any planning "
                    "gain (-34% / -72% / -81%). Architectural cap found: "
                    "position_embeddings [512,128] -> no variant can exceed "
                    "a 512-token window",
    },
}

(HERE / "results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("finalized:", HERE / "results.json")
