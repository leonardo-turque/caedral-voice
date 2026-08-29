"""VOICE-SLATE-15: measure per-speaker features for eligible clean100 speakers.

Inputs : experiments/DATA-FULL-01/speaker_stats.json
         data/raw/librittsr-full/<spk>/*.wav (24 kHz)
Output : experiments/VOICE-SLATE-15/features.json
No model loading. Deterministic sampling of utts.
"""
import json
import glob
import os
import time

import numpy as np
import librosa

ROOT = "C:/Users/leona/Downloads/caedral-voice"
STATS = f"{ROOT}/experiments/DATA-FULL-01/speaker_stats.json"
RAW = f"{ROOT}/data/raw/librittsr-full"
OUT = f"{ROOT}/experiments/VOICE-SLATE-15/features.json"

MIN_HOURS = 18 / 60  # eligibility threshold from task spec
N_SAMPLE = 12        # sample at most this many utts per speaker
N_PYIN = 8           # run pyin on up to this many of the sampled utts
TRIM_S = 10.0        # cap per-utt analyzed duration
HOP = 320            # 13.3 ms @ 24 kHz


def even_sample(items, k):
    """Deterministically pick k evenly spaced items (endpoints included)."""
    n = len(items)
    if n <= k:
        return list(items)
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def analyze():
    stats = json.load(open(STATS))["speakers"]
    elig = {k: v for k, v in stats.items() if v["hours"] >= MIN_HOURS}
    print(f"eligible speakers (>= {MIN_HOURS*60:.0f} min): {len(elig)}", flush=True)

    out = {}
    t_start = time.time()
    for si, (spk, meta) in enumerate(sorted(elig.items(), key=lambda x: -x[1]["hours"])):
        wavs = sorted(glob.glob(f"{RAW}/{spk}/*.wav"))
        sampled = even_sample(wavs, N_SAMPLE)
        pyin_files = sampled[:N_PYIN]

        f0_all, cent_all, rms_db_all = [], [], []
        onsets, total_dur = 0, 0.0
        n_used, n_frames, n_voiced = 0, 0, 0
        for p in pyin_files:
            try:
                y, sr = librosa.load(p, sr=None, mono=True, duration=TRIM_S)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL {p}: {e}", flush=True)
                continue
            dur = len(y) / sr
            if dur < 0.5:
                continue
            total_dur += dur
            n_used += 1

            # F0 (voiced only)
            f0, vflag, _ = librosa.pyin(
                y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=HOP
            )
            n_frames += len(f0)
            vmask = vflag & ~np.isnan(f0) if vflag is not None else np.zeros(len(f0), bool)
            n_voiced += int(vmask.sum())
            if vmask.any():
                f0_all.append(f0[vmask].astype(np.float64))

            # spectral centroid (brightness proxy), pooled frames
            cent = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=1024, hop_length=HOP
            )[0]
            cent_all.append(cent.astype(np.float64))

            # energy stability: std of RMS dB over frames within 40 dB of peak
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
            db = 20 * np.log10(rms + 1e-10)
            keep = db > (db.max() - 40.0)
            if keep.sum() >= 5:
                rms_db_all.append(db[keep])

            # speaking-rate proxy: onset density (energy-envelope events/sec)
            on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="frames")
            onsets += len(on)

        rec = {
            "hours": meta["hours"],
            "n_utts_total": meta["n_utts"],
            "n_utts_sampled": len(sampled),
            "n_utts_analyzed": n_used,
            "analyzed_s": round(total_dur, 1),
        }
        if f0_all:
            f0v = np.concatenate(f0_all)
            rec["f0_median_hz"] = round(float(np.median(f0v)), 1)
            q25, q75 = np.percentile(f0v, [25, 75])
            rec["f0_iqr_hz"] = round(float(q75 - q25), 1)
            rec["voiced_frac"] = round(n_voiced / max(1, n_frames), 3)
        if cent_all:
            cv = np.concatenate(cent_all)
            rec["centroid_median_hz"] = round(float(np.median(cv)), 0)
        if rms_db_all:
            rec["rms_db_std"] = round(float(np.std(np.concatenate(rms_db_all))), 2)
        if total_dur > 0:
            rec["onset_rate_per_s"] = round(onsets / total_dur, 2)

        out[spk] = rec
        print(
            f"[{si+1}/{len(elig)}] {spk}: "
            + ", ".join(f"{k}={v}" for k, v in rec.items() if k != 'voiced_frac'),
            flush=True,
        )

    with open(OUT, "w") as fh:
        json.dump({"task": "VOICE-SLATE-15", "features": out}, fh, indent=2)
    print(f"DONE in {(time.time()-t_start)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    analyze()
