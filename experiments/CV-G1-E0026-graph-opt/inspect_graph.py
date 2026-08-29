"""CV-G1-E0026-graph-opt step 1: inspect kokoro-v1.0.onnx for redundant patterns.

Counts operators, finds redundant patterns (Transpose->Transpose, Cast->Cast,
Reshape->Reshape, Identity chains, duplicate constants, unused initializers,
dead-end nodes) and reports what a zero-loss cleanup can plausibly remove.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
OUT = Path(__file__).resolve().parent / "graph_inspect.json"


def main() -> None:
    t0 = time.perf_counter()
    m = onnx.load(str(MODEL))
    g = m.graph
    nodes = list(g.node)
    op_counts = collections.Counter(n.op_type for n in nodes)

    prod = {}
    for n in nodes:
        for o in n.output:
            if o:
                prod[o] = n
    consumed = set()
    for n in nodes:
        consumed.update(n.input)

    graph_out_names = {o.name for o in g.output}

    # ---------- pattern census ----------
    def attr_i(node, name):
        for kv in node.attribute:
            if kv.name == name:
                return kv.i
        return None

    def perm_of(node):
        p = attr_i(node, "perm")
        if p is None:
            return None
        return tuple(p)

    def compose(a: tuple, b: tuple) -> tuple:
        # apply transpose-perm a then perm b
        r = [None] * len(a)
        for i in range(len(a)):
            r[b[i]] = a[i]
        return tuple(r)

    patterns = {
        "transpose_after_transpose": [],
        "cast_after_cast": [],
        "identity_chain": [],
        "identity_after_identity": [],
        "reshape_after_reshape": [],
        "unsqueeze_after_unsqueeze": [],
        "slice_of_initializer": 0,
        "shape_gather_unsqueeze_chains": 0,
    }
    for n in nodes:
        src = prod.get(n.input[0]) if n.input else None
        if src is None:
            continue
        ot, st = n.op_type, src.op_type
        if ot == "Transpose" and st == "Transpose":
            p1, p2 = perm_of(src), perm_of(n)
            patterns["transpose_after_transpose"].append(
                {"src": src.name, "dst": n.name,
                 "compose_to_identity": (
                     compose(p1, p2) == tuple(range(len(p1)))
                     if p1 is not None and p2 is not None else None)})
        elif ot == "Cast" and st == "Cast":
            to1, to2 = attr_i(src, "to"), attr_i(n, "to")
            patterns["cast_after_cast"].append(
                {"src": src.name, "dst": n.name, "same_to": to1 == to2})
        elif ot == "Identity":
            patterns["identity_chain"].append(
                {"node": n.name, "src_type": st})
            if st == "Identity":
                patterns["identity_after_identity"].append(
                    {"src": src.name, "dst": n.name})

    # Reshape->Reshape where second reshape's shape tensor comes from an
    # initializer equal to the first output's shape -> pure no-op
    init_shapes = {}
    for t in g.initializer:
        if t.data_type == onnx.TensorProto.INT64 and len(t.dims) <= 2:
            try:
                arr = numpy_helper.to_array(t)
                init_shapes[t.name] = arr.reshape(-1).tolist()
            except Exception:
                pass

    for n in nodes:
        if n.op_type != "Reshape" or not n.input or len(n.input) < 2:
            continue
        src = prod.get(n.input[0])
        if src is None or src.op_type != "Reshape":
            continue
        shp = init_shapes.get(n.input[1])
        if shp is None:
            continue
        # first reshape's target shape:
        shp1 = init_shapes.get(src.input[1]) if len(src.input) > 1 else None
        patterns["reshape_after_reshape"].append(
            {"src": src.name, "dst": n.name,
             "dst_shape": shp, "src_shape": shp1})

    # ---------- unused initializers ----------
    used_inits = {name for name in consumed if name in
                  {t.name for t in g.initializer}}
    all_init_names = [t.name for t in g.initializer]
    unused_inits = [n_ for n_ in all_init_names if n_ not in consumed]

    # ---------- duplicate initializers (by raw bytes + dtype + dims) ----------
    seen = {}
    dups = []
    for t in g.initializer:
        key = (t.data_type, tuple(t.dims),
               hashlib_blake2b(t.raw_data))
        if key in seen:
            dups.append({"dup": t.name, "of": seen[key],
                         "bytes": len(t.raw_data)})
        else:
            seen[key] = t.name

    # ---------- dead-end nodes (outputs never consumed, not graph outputs) ----
    dead_nodes = [n_.name for n_ in nodes
                  if all(o in consumed or o in graph_out_names
                         or not o for o in n_.output)]

    result = {
        "model_mb": round(MODEL.stat().st_size / 2**20, 1),
        "load_s": round(time.perf_counter() - t0, 1),
        "nodes": len(nodes),
        "initializers": len(all_init_names),
        "op_counts": dict(op_counts.most_common()),
        "patterns": {
            k: (len(v) if isinstance(v, list) else v)
            for k, v in patterns.items()},
        "pattern_detail_samples": {
            "transpose_after_transpose":
                patterns["transpose_after_transpose"][:10],
            "cast_after_cast": patterns["cast_after_cast"][:10],
        },
        "unused_initializers": unused_inits[:30],
        "n_unused_initializers": len(unused_inits),
        "duplicate_initializers": dups[:30],
        "n_duplicate_initializers": len(dups),
        "dup_wasted_bytes": sum(d["bytes"] for d in dups),
        "dead_end_nodes": dead_nodes[:40],
        "n_dead_end_nodes": len(dead_nodes),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    brief = {k: result[k] for k in (
        "nodes", "initializers", "n_unused_initializers",
        "n_duplicate_initializers", "dup_wasted_bytes", "n_dead_end_nodes")}
    brief.update({k: v for k, v in patterns.items()
                  if isinstance(v, int) or (isinstance(v, list) and v)})
    print(json.dumps(brief, indent=2))


def hashlib_blake2b(raw: bytes) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(raw)
    return h.hexdigest()


if __name__ == "__main__":
    main()
