"""CV-G1-E0015-speaker-distinctness — voice-identity separation evidence for the V1
slate, measured independently of our TTS model with Resemblyzer's GE2E speaker
encoder (d-vector, 256-d, L2-normalized).

Method
------
1. One VoiceEncoder load (CPU), reused for everything, unloaded at exit.
2. Slate: the 15 VOICE-SLATE-V1 speakers; 6 real utterances each from
   data/processed/librittsr-clean100 (flat `<spk>_*.wav` naming; deterministic
   selection: sorted list, evenly spaced indices, endpoints included).
3. Pilot: the 10 trained speaker indices from CV-G1-E0007 speaker_index.json;
   6 real utterances each from the same processed corpus -> per-speaker mean
   "real anchor" embedding for the future synth-vs-real V1 identity gate.
4. Metrics: intra-speaker mean pairwise cosine, inter-speaker cosine matrix on
   mean embeddings, utterance-level genuine/impostor distributions, flagged
   pairs > 0.80, sklearn silhouette score (cosine) on utterance embeddings.
5. RAM: model unloaded after use; peak process RSS recorded.

Outputs (this directory): embeddings.json, similarity_matrix.json,
run_manifest.json; conclusion.md is written by hand afterwards from the numbers.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed" / "librittsr-clean100"
OUT_DIR = Path(__file__).resolve().parent

SLATE = [
    # (rank, libritts_speaker) from docs/data/VOICE-SLATE-V1.md
    ("01", "6209"), ("02", "6415"), ("03", "696"), ("04", "2416"),
    ("05", "6147"), ("06", "1088"), ("07", "3879"), ("08", "2836"),
    ("09", "78"), ("10", "7447"), ("11", "8419"), ("12", "7067"),
    ("13", "8838"), ("14", "7505"), ("15", "7402"),
]
PILOT = ["103", "1034", "1040", "1069", "1081", "1088",
         "1098", "1116", "118", "1183"]  # from smoke-a/checkpoints/speaker_index.json
N_UTTS = 6
MAX_SECONDS = 10.0  # cap per clip post-VAD (VOICE-SLATE-15 convention)


def select_files(spk: str, n: int) -> list[Path]:
    """Sorted file list, evenly spaced indices, endpoints included."""
    files = sorted(PROCESSED.glob(f"{spk}_*.wav"))
    if len(files) < n:
        raise RuntimeError(f"speaker {spk}: only {len(files)} files")
    idx = np.round(np.linspace(0, len(files) - 1, n)).astype(int)
    return [files[i] for i in dict.fromkeys(idx.tolist())]  # dedupe, keep order


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    import psutil
    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / 1e6

    from resemblyzer import VoiceEncoder, preprocess_wav

    t0 = time.time()
    encoder = VoiceEncoder("cpu")  # single model load for the whole experiment
    log(f"VoiceEncoder loaded ({time.time()-t0:.1f} s)")

    def embed_group(group: list[str]) -> dict:
        out = {}
        for spk in group:
            rec = {"files": [], "embeddings": []}
            for f in select_files(spk, N_UTTS):
                wav = preprocess_wav(str(f))
                if MAX_SECONDS and len(wav) > 16000 * MAX_SECONDS:
                    wav = wav[: int(16000 * MAX_SECONDS)]
                emb = encoder.embed_utterance(wav)
                rec["files"].append(f.name)
                rec["embeddings"].append([round(float(x), 6) for x in emb])
            embs = np.asarray(rec["embeddings"], dtype=np.float64)
            mean = embs.mean(axis=0)
            rec["mean_embedding"] = [round(float(x), 6) for x in mean / np.linalg.norm(mean)]
            out[spk] = rec
            log(f"  embedded spk {spk}: {len(rec['files'])} utts")
        return out

    t1 = time.time()
    slate = embed_group([s for _, s in SLATE])
    pilot = embed_group(PILOT)
    embed_seconds = time.time() - t1

    def cos(a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---------- slate metrics ----------
    ids = [s for _, s in SLATE]
    means = np.asarray([slate[s]["mean_embedding"] for s in ids])
    utt = np.concatenate([np.asarray(slate[s]["embeddings"]) for s in ids], axis=0)
    labels = np.repeat(np.arange(len(ids)), N_UTTS)

    inter = np.asarray([[cos(means[i], means[j]) for j in range(len(ids))]
                        for i in range(len(ids))])
    off_diag = inter[~np.eye(len(ids), dtype=bool)]

    intra_mean = {}
    for k, s in enumerate(ids):
        e = np.asarray(slate[s]["embeddings"])
        g = np.asarray([[cos(e[i], e[j]) for j in range(len(e))] for i in range(len(e))])
        intra_mean[s] = float(g[np.triu_indices_from(g, 1)].mean())

    # utterance-level genuine / impostor distributions
    sim_all = utt @ utt.T  # unit-norm embeddings
    lab_mat = labels[:, None] == labels[None, :]
    genuine = sim_all[np.triu(lab_mat, 1)]
    impostor = sim_all[np.triu(~lab_mat, 1)]

    iu = np.triu_indices(len(ids), 1)
    pair_sims = [(ids[i], ids[j], float(inter[i, j])) for i, j in zip(*iu)]
    worst_pairs = sorted(pair_sims, key=lambda p: -p[2])[:5]
    flagged = [{"a": a, "b": b, "sim": round(v, 4)} for a, b, v in pair_sims if v > 0.80]

    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(utt, labels, metric="cosine"))

    # per-speaker separation margin: intra mean minus max sim of mean emb to others
    margins = {s: round(intra_mean[s] - max(inter[k][j] for j in range(len(ids)) if j != k), 4)
               for k, s in enumerate(ids)}

    # ---------- pilot metrics ----------
    p_ids = PILOT
    p_means = np.asarray([pilot[s]["mean_embedding"] for s in p_ids])
    p_inter = np.asarray([[cos(p_means[i], p_means[j]) for j in range(len(p_ids))]
                          for i in range(len(p_ids))])
    piu = np.triu_indices(len(p_ids), 1)
    p_pairs = [(p_ids[i], p_ids[j], float(p_inter[i, j])) for i, j in zip(*piu)]
    p_worst = sorted(p_pairs, key=lambda p: -p[2])[:5]
    p_flagged = [{"a": a, "b": b, "sim": round(v, 4)} for a, b, v in p_pairs if v > 0.80]
    p_intra = {}
    for s in p_ids:
        e = np.asarray(pilot[s]["embeddings"])
        g = np.asarray([[cos(e[i], e[j]) for j in range(len(e))] for i in range(len(e))])
        p_intra[s] = float(g[np.triu_indices_from(g, 1)].mean())
    p_utt = np.concatenate([np.asarray(pilot[s]["embeddings"]) for s in p_ids], axis=0)
    p_labels = np.repeat(np.arange(len(p_ids)), N_UTTS)
    p_sil = float(silhouette_score(p_utt, p_labels, metric="cosine"))

    # consistency check: 1088 is in both groups (identical file sets by rule)
    x1088 = cos(np.asarray(slate["1088"]["mean_embedding"]),
                np.asarray(pilot["1088"]["mean_embedding"]))

    # ---------- write outputs ----------
    n_emb = len(ids) * N_UTTS + len(p_ids) * N_UTTS
    (OUT_DIR / "embeddings.json").write_text(json.dumps({
        "experiment": "CV-G1-E0015-speaker-distinctness",
        "model": {"package": "resemblyzer", "version": "0.1.4",
                  "license": "Apache-2.0 (verified from repo LICENSE, not MIT)",
                  "encoder": "GE2E LSTM d-vector, 256-d, L2-normalized",
                  "device": "cpu"},
        "audio_source": str(PROCESSED.relative_to(REPO)),
        "selection_rule": f"sorted <spk>_*.wav, evenly spaced {N_UTTS}, endpoints included; "
                          f"resemblyzer.preprocess_wav (16 kHz + VAD trim), capped {MAX_SECONDS:.0f} s",
        "slate": {s: slate[s] for _, s in SLATE},
        "pilot_real_anchor": {s: {"files": pilot[s]["files"],
                                  "mean_embedding": pilot[s]["mean_embedding"],
                                  "n_reference_utts": len(pilot[s]["files"]),
                                  "role": "real-audio identity anchor for synth-vs-real V1 gate"}
                              for s in p_ids},
    }, indent=1))

    (OUT_DIR / "similarity_matrix.json").write_text(json.dumps({
        "slate_order_rank_speaker": [[r, s] for r, s in SLATE],
        "inter_speaker_cosine_matrix_mean_embeddings": {
            "order": ids, "matrix": [[round(float(x), 4) for x in row] for row in inter]},
        "intra_speaker_mean_pairwise_cosine": {s: round(v, 4) for s, v in intra_mean.items()},
        "per_speaker_margin_intra_minus_max_inter": margins,
        "inter_stats": {"min": round(float(off_diag.min()), 4),
                        "max": round(float(off_diag.max()), 4),
                        "mean": round(float(off_diag.mean()), 4)},
        "worst_pairs_top5_by_mean_sim": [
            {"a": a, "b": b, "sim": round(v, 4)} for a, b, v in worst_pairs],
        "flagged_pairs_gt_0_80": flagged,
        "utterance_level": {
            "genuine_min": round(float(genuine.min()), 4),
            "genuine_mean": round(float(genuine.mean()), 4),
            "impostor_max": round(float(impostor.max()), 4),
            "impostor_p999": round(float(np.quantile(impostor, 0.999)), 4),
            "impostor_mean": round(float(impostor.mean()), 4),
            "impostor_n_pairs": int(impostor.size)},
        "silhouette_score_cosine_utterance_level": round(sil, 4),
        "pilot_group": {
            "order": p_ids,
            "inter_matrix": [[round(float(x), 4) for x in row] for row in p_inter],
            "intra_mean": {s: round(v, 4) for s, v in p_intra.items()},
            "worst_pairs_top5": [{"a": a, "b": b, "sim": round(v, 4)} for a, b, v in p_worst],
            "flagged_pairs_gt_0_80": p_flagged,
            "silhouette_cosine": round(p_sil, 4)},
        "cross_check_speaker_1088_slate_vs_pilot_mean_cosine": round(x1088, 6),
    }, indent=1))

    rss_peak_mb = proc.memory_info().rss / 1e6
    del encoder
    gc.collect()
    rss_after_mb = psutil.Process().memory_info().rss / 1e6

    (OUT_DIR / "run_manifest.json").write_text(json.dumps({
        "experiment": "CV-G1-E0015-speaker-distinctness",
        "date": time.strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "torch": __import__("torch").__version__,
        "sklearn": __import__("sklearn").__version__,
        "resemblyzer": "0.1.4 (installed --no-deps; VAD via webrtcvad-wheels 2.0.14)",
        "n_embeddings": n_emb,
        "embed_seconds": round(embed_seconds, 1),
        "wall_seconds_total": round(time.time() - t0, 1),
        "rss_mb": {"before_load": round(rss_before_mb, 1),
                   "peak_with_model": round(rss_peak_mb, 1),
                   "after_unload": round(rss_after_mb, 1)},
        "commit_policy": "nothing committed (per task)",
    }, indent=1))

    log(f"done: {n_emb} utt embeddings in {embed_seconds:.1f}s; "
        f"silhouette={sil:.4f}; max inter={off_diag.max():.4f}; RSS peak {rss_peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
