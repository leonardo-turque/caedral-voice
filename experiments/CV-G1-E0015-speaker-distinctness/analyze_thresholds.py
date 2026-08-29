"""CV-G1-E0015 follow-up: verification-threshold calibration + confusion checks.

Loads embeddings.json (no model reload needed) and computes, on the slate:
  A) utterance-vs-anchor verification scores (the exact geometry of the future
     synth-vs-real gate: one clip vs the 6-utt real anchor):
       genuine = utt vs OWN speaker anchor; impostor = utt vs OTHER anchors.
     -> EER operating point, score at FAR=1%/0.1%, genuine p05/p50.
  B) 15-way leave-one-out nearest-centroid identification accuracy
     (overall + per worst pair) — does the encoder ever confuse two slate ids?
Writes thresholds.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

data = json.loads((HERE / "embeddings.json").read_text())
slate = data["slate"]
ids = list(slate.keys())

utts, labels, anchors = [], [], []
for k, s in enumerate(ids):
    e = np.asarray(slate[s]["embeddings"], dtype=np.float64)
    utts.append(e)
    labels.append(np.full(len(e), k))
    anchors.append(np.asarray(slate[s]["mean_embedding"]))
X = np.concatenate(utts)                      # (90, 256) unit norm
y = np.concatenate(labels)
A = np.stack(anchors)                         # (15, 256) unit norm

S_utt_anchor = X @ A.T                        # (90, 15) cosine (unit vectors)
own = S_utt_anchor[np.arange(len(y)), y]
mask = np.ones_like(S_utt_anchor, dtype=bool)
mask[np.arange(len(y)), y] = False
other = S_utt_anchor[mask]

# --- A) verification calibration ---
genuine, impostor = own, other
lo, hi = 0.0, 1.0
grid = np.linspace(lo, hi, 2001)


def far_frr(t):
    far = float((impostor >= t).mean())   # impostor accepted
    frr = float((genuine < t).mean())     # genuine rejected
    return far, frr


fars = np.asarray([far_frr(t)[0] for t in grid])
frrs = np.asarray([far_frr(t)[1] for t in grid])
eer_t = float(grid[np.argmin(np.abs(fars - frrs))])
eer = far_frr(eer_t)[0]

t_far1 = float(grid[np.where(fars <= 0.01)[0][0]])    # highest t with FAR<=1% ... first from low
# choose the LARGEST threshold whose FAR<=target (search from the right)
idx1 = np.where(fars <= 0.01)[0]
idx01 = np.where(fars <= 0.001)[0]
t_far1 = float(grid[idx1[0]]) if idx1.size else None   # smallest t achieving FAR<=target
far_at1 = far_frr(t_far1)[1] if t_far1 else None
t_far01 = float(grid[idx01[0]]) if idx01.size else None
far_at01 = far_frr(t_far01)[1] if t_far01 else None

op_table = {}
for t in (0.80, 0.82, 0.84, 0.85, 0.86):
    far, frr = far_frr(t)
    op_table[f"{t:.2f}"] = {"FAR": round(far, 4), "FRR": round(frr, 4)}

# --- B) LOO nearest-centroid identification ---
correct = 0
confusions = {}
for i in range(len(y)):
    own_spk = y[i]
    # centroid excluding this utterance
    cents = []
    for k in range(len(ids)):
        pts = X[(y == k) & (np.arange(len(y)) != i)]
        c = pts.mean(axis=0)
        cents.append(c / np.linalg.norm(c))
    cents = np.stack(cents)
    pred = int(np.argmax(X[i] @ cents.T))
    correct += int(pred == own_spk)
    if pred != own_spk:
        key = tuple(sorted((ids[own_spk], ids[pred])))
        confusions[key] = confusions.get(key, 0) + 1

pair_acc = {}
for a, b in [("696", "2416"), ("6209", "7505"), ("6209", "78"), ("78", "8419"),
             ("696", "6147"), ("7505", "7402")]:
    ka, kb = ids.index(a), ids.index(b)
    sub = X[(y == ka) | (y == kb)]
    sub_y = y[(y == ka) | (y == kb)]
    ca = sub[sub_y == ka].mean(axis=0); ca /= np.linalg.norm(ca)
    cb = sub[sub_y == kb].mean(axis=0); cb /= np.linalg.norm(cb)
    sa, sb = sub @ ca, sub @ cb
    acc = float(((sa > sb) == (sub_y == ka)).mean())
    pair_acc[f"{a}|{b}"] = round(acc, 4)

out = {
    "protocol": "utterance embedding vs 6-utt real anchor cosine (gate geometry)",
    "n_genuine_pairs": int(genuine.size),
    "n_impostor_pairs": int(impostor.size),
    "genuine": {"min": round(float(genuine.min()), 4),
                "p05": round(float(np.quantile(genuine, 0.05)), 4),
                "mean": round(float(genuine.mean()), 4)},
    "impostor": {"max": round(float(impostor.max()), 4),
                 "p999": round(float(np.quantile(impostor, 0.999)), 4),
                 "p99": round(float(np.quantile(impostor, 0.99)), 4),
                 "mean": round(float(impostor.mean()), 4)},
    "eer_threshold": round(eer_t, 3),
    "eer_rate": round(eer, 4),
    "operating_points": op_table,
    "threshold_at_far_1pct": None if t_far1 is None else {
        "t": round(t_far1, 3), "FAR_at_t": 0.01,
        "FRR_at_t": round(far_at1, 4)},
    "threshold_at_far_0pct_observed": {
        "t": round(float(impostor.max()) + 0.001, 3),
        "note": "above max observed impostor; FRR there:",
        "frr": round(float((genuine < impostor.max() + 0.001).mean()), 4)},
    "loo_nearest_centroid_identification_accuracy": round(correct / len(y), 4),
    "loo_confusions": {f"{a}|{b}": n for (a, b), n in sorted(confusions.items(), key=lambda kv: -kv[1])},
    "worst_pair_binary_centroid_accuracy": pair_acc,
}
(HERE / "thresholds.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
