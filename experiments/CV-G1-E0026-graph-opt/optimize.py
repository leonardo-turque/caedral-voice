"""CV-G1-E0026-graph-opt step 2: zero-loss graph optimization pipeline.

Stages (each stage reports node/init counts so gains are attributable):
    A. onnxsim.simplify (constant folding + redundant op elimination,
       numeric check disabled - equivalence verified separately against ORT)
    B. onnxoptimizer cleanup passes (nop eliminations, idempotent pairs)
    C. manual initializer dedup (hash of dtype+dims+raw bytes)
    D. dead-subgraph pruning + unused-initializer removal
Saves artifacts/optimized/kokoro-opt.onnx.
"""
from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
OUT_DIR = REPO / "artifacts/optimized"
OUT_MODEL = OUT_DIR / "kokoro-opt.onnx"
STATS = Path(__file__).resolve().parent / "opt_stages.json"


def counts(g) -> dict:
    return {
        "nodes": len(g.node),
        "initializers": len(g.initializer),
        "op_counts": dict(collections.Counter(n.op_type for n in g.node)),
        "bytes": sum(len(t.raw_data) for t in g.initializer),
    }


def dedup_initializers(g) -> int:
    """Point duplicate initializers at one tensor; returns removed count."""
    by_key = {}
    dups = set()
    for t in g.initializer:
        key = (t.data_type, tuple(t.dims), hashlib.blake2b(t.raw_data, digest_size=16).hexdigest())
        if key in by_key:
            dups.add(t.name)
            # rewrite every consumer to use the canonical name
        else:
            by_key[key] = t.name
    if not dups:
        return 0
    remap = {}
    seen = {}
    for t in list(g.initializer):
        key = (t.data_type, tuple(t.dims), hashlib.blake2b(t.raw_data, digest_size=16).hexdigest())
        if key in seen:
            remap[t.name] = seen[key]
        else:
            seen[key] = t.name
    for n in g.node:
        for i, name in enumerate(n.input):
            if name in remap:
                n.input[i] = remap[name]
    keep = [t for t in g.initializer if t.name not in set(remap)]
    del g.initializer[:]
    g.initializer.extend(keep)
    return len(remap)


def prune_dead(g) -> int:
    """Remove nodes not reachable from graph outputs; then unused inits."""
    nodes = list(g.node)
    prod = {}
    for n in nodes:
        for o in n.output:
            if o:
                prod[o] = n
    # reverse BFS over tensors from graph outputs
    needed_nodes = set()
    stack = [o.name for o in g.output]
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is None or n.name in needed_nodes:
            continue
        needed_nodes.add(n.name)
        stack.extend(i for i in n.input if i)
    before = len(nodes)
    keep = [n for n in nodes if n.name in needed_nodes]
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input if i}
    used |= {o.name for o in g.output}
    inits = [t for t in g.initializer if t.name in used]
    removed_inits = len(g.initializer) - len(inits)
    del g.initializer[:]
    g.initializer.extend(inits)
    return before - len(keep), removed_inits


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = {}
    t0 = time.perf_counter()
    m = onnx.load(str(MODEL))
    stages["baseline"] = counts(m.graph)

    # ---- A: onnxsim ----
    import onnxsim
    ta = time.perf_counter()
    m_sim, check_ok = onnxsim.simplify(m, check_n=0)
    stages["A_onnxsim"] = {
        **counts(m_sim.graph),
        "wall_s": round(time.perf_counter() - ta, 1),
        "internal_check_ok": bool(check_ok),
    }
    print("stage A done:", stages["A_onnxsim"]["nodes"], "nodes", flush=True)

    # ---- B: onnxoptimizer passes ----
    import onnxoptimizer
    tb = time.perf_counter()
    passes = [
        "eliminate_nop_cast", "eliminate_nop_dropout", "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
        "eliminate_nop_transpose", "eliminate_consecutive_idempotent_ops",
        "eliminate_consecutive_transpose", "eliminate_deadend",
        "eliminate_identity", "eliminate_unused_initializer",
        "fuse_consecutive_transposes", "extract_constant_to_initializer",
    ]
    available = set(onnxoptimizer.get_available_passes())
    used = [p_ for p_ in passes if p_ in available]
    try:
        m_opt = onnxoptimizer.optimize(m_sim, used)
    except Exception as exc:
        stages["B_onnxoptimizer"] = {"error": f"{type(exc).__name__}: {exc}",
                                     "passes_attempted": used}
        m_opt = m_sim
        used = []
    else:
        stages["B_onnxoptimizer"] = {
            **counts(m_opt.graph),
            "wall_s": round(time.perf_counter() - tb, 1),
            "passes_applied": used,
        }
    print("stage B done:", stages["B_onnxoptimizer"].get("nodes"), flush=True)

    # ---- C: manual dedup + D: dead prune (idempotent belt-and-suspenders) --
    tc = time.perf_counter()
    n_dedup = dedup_initializers(m_opt.graph)
    pruned_nodes, pruned_inits = prune_dead(m_opt.graph)
    stages["CD_manual_cleanup"] = {
        **counts(m_opt.graph),
        "wall_s": round(time.perf_counter() - tc, 1),
        "duplicates_removed": n_dedup,
        "dead_nodes_removed": pruned_nodes,
        "unused_inits_removed": pruned_inits,
    }

    # clear doc_string/sparseness cruft? keep metadata - harmless.

    onnx.save(m_opt, str(OUT_MODEL))
    stages["final_file_mb"] = round(OUT_MODEL.stat().st_size / 2**20, 2)
    stages["total_wall_s"] = round(time.perf_counter() - t0, 1)
    stages["out_model"] = str(OUT_MODEL.relative_to(REPO))
    STATS.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    b, f = stages["baseline"], stages["CD_manual_cleanup"]
    print(json.dumps({
        "nodes": [b["nodes"], f["nodes"]],
        "initializers": [b["initializers"], f["initializers"]],
        "file_mb": [round(MODEL.stat().st_size / 2**20, 1), stages["final_file_mb"]],
        "wall_s": stages["total_wall_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
