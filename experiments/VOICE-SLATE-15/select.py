"""VOICE-SLATE-15 selection: 7F + 7M + 1 narrator from measured features.

Spread strategy: farthest-point sampling on log(F0) inside each gender
(range too narrow for uniform hard 15% chaining), 15% enforced as a large
penalty, cross-gender proximity as a smaller penalty. Ties broken by data
volume (hours) + energy stability + centroid distance from nearest pick.
"""
import json

import numpy as np

ROOT = "C:/Users/leona/Downloads/caedral-voice"
FEAT = f"{ROOT}/experiments/VOICE-SLATE-15/features.json"
PILOT = {"103", "1034", "1040", "1069", "1081", "1088", "1098", "1116", "118", "1183"}
F0_SPLIT = 155.0        # Hz, feminine/masculine split (measured median F0)
MIN_SEP_FRAC = 0.15     # 15% neighbor penalty threshold from task spec
N_PER_GENDER = 7


def load():
    feats = json.load(open(FEAT))["features"]
    return {k: v for k, v in feats.items() if "f0_median_hz" in v}


def main():
    feats = load()
    narrator_id = max(feats, key=lambda s: feats[s]["hours"])
    nf = feats[narrator_id]
    pool = {s: v for s, v in feats.items() if s != narrator_id}
    fem = {s: v for s, v in pool.items() if v["f0_median_hz"] >= F0_SPLIT}
    mas = {s: v for s, v in pool.items() if v["f0_median_hz"] < F0_SPLIT}
    print(f"narrator={narrator_id} ({nf['hours']:.3f} h, F0={nf['f0_median_hz']:.0f}) "
          f"| fem={len(fem)} masc={len(mas)}")

    hours = np.array([v["hours"] for v in pool.values()])
    stab = np.array([v.get("rms_db_std", 10.0) for v in pool.values()])
    zh = (hours - hours.mean()) / hours.std()
    zs = (stab.mean() - stab) / (stab.std() + 1e-9)
    score = {s: float(zh[i] + zs[i]) for i, s in enumerate(pool)}

    # occupied F0 anchors outside this gender's own picks (narrator + other gender)
    def spread_pick(grp, other_f0):
        ids = sorted(grp, key=lambda s: grp[s]["f0_median_hz"])
        lf = {s: float(np.log(grp[s]["f0_median_hz"])) for s in ids}

        def mind(s, chosen):
            return min((abs(lf[s] - lf[c]) for c in chosen), default=9e9)

        chosen = [min(ids, key=lambda s: lf[s])]
        while len(chosen) < N_PER_GENDER:
            dmax = max(mind(s, chosen) for s in ids if s not in chosen)
            cands = [s for s in ids if s not in chosen
                     and mind(s, chosen) >= dmax - 0.01]
            def key(s):
                pen = 0.0
                if mind(s, chosen) < np.log(1 + MIN_SEP_FRAC):
                    pen -= 10.0                       # in-gender <15% neighbor
                for of in other_f0:
                    if abs(grp[s]["f0_median_hz"] - of) / of < MIN_SEP_FRAC:
                        pen -= 1.0                    # cross-gender proximity
                cent = grp[s].get("centroid_median_hz", 1000.0)
                cd = min((abs(cent - grp[c].get("centroid_median_hz", 1000.0))
                          for c in chosen), default=500.0)
                return score[s] + pen + min(cd, 400.0) / 4000.0
            chosen.append(max(cands, key=key))
        return chosen

    m_other = [nf["f0_median_hz"]] + [v["f0_median_hz"] for v in fem.values()]
    f_other = [v["f0_median_hz"] for v in mas.values()] + [nf["f0_median_hz"]]
    m_ids = spread_pick(mas, m_other)
    f_ids = spread_pick(fem, f_other)

    rows = []
    rows.append({"spk": narrator_id, "role": "narrator"})
    for s in f_ids:
        rows.append({"spk": s, "role": "F"})
    for s in m_ids:
        rows.append({"spk": s, "role": "M"})
    json.dump(rows, open(f"{ROOT}/experiments/VOICE-SLATE-15/picks.json", "w"),
              indent=2)

    for r in sorted(rows, key=lambda x: (x["role"], feats[x["spk"]]["f0_median_hz"])):
        f = feats[r["spk"]]
        fast = " FAST-PATH(pilot)" if r["spk"] in PILOT else ""
        print(f"{r['role']:>8} {r['spk']} F0={f['f0_median_hz']:.0f}"
              f"(IQR {f['f0_iqr_hz']:.0f}) cent={f.get('centroid_median_hz', 0):.0f} "
              f"h={f['hours']:.3f} onset={f.get('onset_rate_per_s', 0):.1f}"
              f" rms_std={f.get('rms_db_std', 0):.1f}{fast}")

    # measured separation check within each gender
    for gname, gids in (("F", f_ids), ("M", m_ids)):
        f0s = sorted(feats[s]["f0_median_hz"] for s in gids)
        gaps = [round(100 * (b - a) / a, 1) for a, b in zip(f0s, f0s[1:])]
        print(f"[{gname}] adjacent-gap %: {gaps}")


if __name__ == "__main__":
    main()
