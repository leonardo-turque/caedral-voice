"""ANCHOR-QEX aggregation: merge generation + classify + prosody into results.json.

Gate validation (mission step 4): do the TEACHER anchors pass the gates
PROSODY-EVAL proposes? Per program honesty rules ("a gate our teachers cannot
pass is not a gate"), any teacher failure is a loud FLAG requiring
recalibration before Tier C can gate qex_002..006.

Validated gates:
  G1 P1 direction table (§2): rise REQUIRED on qex_001/qex_003 (B>=1.15),
     FORBIDDEN on qex_002/qex_006 (B<1.15), NA on exclamations.
  G2 Tier C band X=0.20 (§3): |B - median(B_anchor,item)| <= 0.20 per item;
     reported both as distance-from-median (the formal rule) and as raw
     spread max-min (how §3 derived X from anchor spread).
  G3 P3 upper-bound pause rule (§2/§4): P150<=E+2 AND no internal silence
     >700 ms AND (L>=1 => some pause >=200 ms).
"""
from __future__ import annotations

import json
import subprocess
import statistics
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent

RISE_THRESHOLD = 1.15
X_BAND = 0.20

DIRECTION = {  # PROSODY-EVAL §2 P1 expectations
    "qex_001": "required",
    "qex_002": "forbidden",   # wh-question: flat/fall expected
    "qex_003": "required",    # tag question
    "qex_004": "na",          # exclamation -> P2/P3
    "qex_005": "na",          # exclamation -> P2/P3
    "qex_006": "forbidden",   # wh-question
}


def main() -> None:
    gen_k = json.loads((EXP / "gen-meta-kokoro.json").read_text(encoding="utf-8"))
    gen_p = json.loads((EXP / "gen-meta-piper.json").read_text(encoding="utf-8"))
    scores = json.loads((EXP / "scores.json").read_text(encoding="utf-8"))
    pros = json.loads((EXP / "prosody.json").read_text(encoding="utf-8"))

    scores_by_stem = {Path(r["file"]).stem: r for r in scores}
    pros_by_stem = {Path(r["file"]).stem: r for r in pros["rows"]}
    gen_rows = gen_k["rows"] + gen_p["rows"]

    clips = []
    for stem, pro in pros_by_stem.items():
        sc = scores_by_stem.get(stem)
        clips.append({
            "stem": stem,
            "teacher": pro["teacher"],
            "voice": pro["voice"],
            "item_id": pro["item_id"],
            "origin": pro["origin"],
            "audio_seconds": pro["audio_seconds"],
            "min_wer": None if sc is None else sc["min_wer"],
            "dnsmos_ovrl": None if sc is None else sc["dnsmos_ovrl"],
            "class": None if sc is None else sc["class"],
            "chars_per_sec": None if sc is None else sc["chars_per_sec"],
            "rate_plausible": None if sc is None else sc["rate_plausible"],
            "median_f0_hz": pro.get("median_f0_hz"),
            "B_rise_flag": pro.get("B"),
            "C_slope_st_per_s": pro.get("C_slope_st_per_s"),
            "voiced_fraction": pro.get("voiced_fraction"),
            "E_expected_pauses": pro["expected_internal_boundaries_E"],
            "L_long_marks": pro["long_marks_L"],
            "P150": pro["P150"],
            "P350": pro["P350"],
            "max_internal_silence_ms": pro["max_internal_silence_ms"],
            "p3_rule_pass": pro["p3_rule_pass"],
        })

    items = sorted({c["item_id"] for c in clips})

    # ---- G1: P1 direction table vs teachers -------------------------------
    g1_items = []
    for it in items:
        expect = DIRECTION[it]
        row_clips = [c for c in clips if c["item_id"] == it]
        checks = []
        for c in row_clips:
            if expect == "na" or c["B_rise_flag"] is None:
                ok = None
            elif expect == "required":
                ok = c["B_rise_flag"] >= RISE_THRESHOLD
            else:  # forbidden
                ok = c["B_rise_flag"] < RISE_THRESHOLD
            checks.append({"voice": f"{c['teacher']}/{c['voice']}",
                           "B": c["B_rise_flag"], "direction_ok": ok})
        measurable = [ch for ch in checks if ch["direction_ok"] is not None]
        n_pass = sum(1 for ch in measurable if ch["direction_ok"])
        g1_items.append({
            "item_id": it,
            "expectation": expect,
            "checks": checks,
            "n_pass": n_pass,
            "n_measurable": len(measurable),
            "all_teachers_pass": None if expect == "na"
                                 else (n_pass == len(measurable) and len(measurable) > 0),
        })
    g1_verdict_items = [g for g in g1_items if g["expectation"] != "na"]
    g1_total_fail = sum(1 for g in g1_verdict_items if not g["all_teachers_pass"])

    # ---- G2: Tier C band X=0.20 -------------------------------------------
    g2_items = []
    for it in items:
        bs = [(f"{c['teacher']}/{c['voice']}", c["B_rise_flag"])
              for c in clips if c["item_id"] == it and c["B_rise_flag"] is not None]
        vals = [b for _, b in bs]
        med = statistics.median(vals)
        deviations = {k: round(abs(b - med), 4) for k, b in bs}
        spread = round(max(vals) - min(vals), 4)
        band_ok_median = all(d <= X_BAND for d in deviations.values())
        # how §3 derived X: tolerance must cover anchor spread with margin
        spread_within_x = spread <= X_BAND
        g2_items.append({
            "item_id": it,
            "B_values": dict(bs),
            "anchor_median_B": round(med, 4),
            "abs_deviation_from_median": deviations,
            "max_abs_deviation_from_median": round(max(deviations.values()), 4),
            "band_ok_vs_median_rule": bool(band_ok_median),
            "spread_max_minus_min": spread,
            "spread_within_X_as_derived": bool(spread_within_x),
        })

    # ---- G3: P3 upper-bound rule ------------------------------------------
    p3_new = [c for c in clips if c["origin"] == "CV-G1-E0016"]
    g3 = {
        "rule": "P150 <= E+2 AND no internal silence > 700 ms AND (L>=1 => some pause >= 200 ms)",
        "clips_total": len(p3_new),
        "clips_pass": sum(1 for c in p3_new if c["p3_rule_pass"]),
        "failures": [
            {"clip": c["stem"], "item": c["item_id"], "voice": c["voice"],
             "L": c["L_long_marks"], "P150": c["P150"], "E": c["E_expected_pauses"],
             "max_internal_silence_ms": c["max_internal_silence_ms"]}
            for c in p3_new if not c["p3_rule_pass"]],
        "legacy_qex001_all_pass": all(c["p3_rule_pass"] for c in clips
                                      if c["origin"] != "CV-G1-E0016"),
    }

    # ---- summary stats ------------------------------------------------------
    def agg(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else {
            "median": round(statistics.median(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}

    by_teacher: dict[str, list[dict]] = {}
    for c in clips:
        if c["origin"] != "CV-G1-E0016":
            continue
        by_teacher.setdefault(f"{c['teacher']}/{c['voice']}", []).append(c)
    teacher_summary = {
        k: {"n": len(v), "dnsmos_ovrl": agg([x["dnsmos_ovrl"] for x in v]),
            "min_wer": agg([x["min_wer"] for x in v])}
        for k, v in sorted(by_teacher.items())}

    flags = []
    for g in g1_verdict_items:
        if not g["all_teachers_pass"]:
            fails = [f"{ch['voice']} B={ch['B']}" for ch in g["checks"]
                     if ch["direction_ok"] is False]
            flags.append(
                f"P1-DIRECTION {g['item_id']} ({g['expectation']} rise): teachers FAIL — "
                f"{g['n_pass']}/{g['n_measurable']} comply [{'; '.join(fails)}]. "
                f"Absolute B>=/{'<' if g['expectation'] == 'forbidden' else '>='}1.15 "
                f"needs recalibration before gating this item.")
    for g in g2_items:
        if not g["band_ok_vs_median_rule"]:
            over = {k: d for k, d in g["abs_deviation_from_median"].items()
                    if d > X_BAND}
            flags.append(
                f"TIER-C-BAND {g['item_id']}: anchor {over} exceeds |dB|<={X_BAND} "
                f"vs anchor median — a teacher falls outside its own proposed band.")
        if not g["spread_within_X_as_derived"]:
            flags.append(
                f"TIER-C-SPREAD {g['item_id']}: anchor spread "
                f"{g['spread_max_minus_min']} > X={X_BAND} (X was derived as max "
                f"anchor spread + margin; new items exceed the calibration basis).")
    if g3["failures"]:
        det = "; ".join(f"{f['clip']} L={f['L']} maxSil={f['max_internal_silence_ms']}ms"
                        for f in g3["failures"])
        flags.append(
            f"P3-LONG-PAUSE CLAUSE: {len(g3['failures'])}/{g3['clips_total']} teacher "
            f"clips fail the '(L>=1 => pause >=200 ms)' requirement on qex_005 "
            f"[{det}] — teachers do not realize a >=200 ms gap after '‖' before an "
            f"exclamatory continuation; clause needs recalibration (or expressive-"
            f"context exemption) before gating.")

    results = {
        "experiment": "CV-G1-E0016-qex-anchors",
        "task": "ANCHOR-QEX",
        "purpose": ("Render + score missing teacher anchors qex_002..qex_006 so "
                    "PROSODY-EVAL Tier C can gate the full qex set; validate that "
                    "teachers pass their own proposed gates."),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": pros["git_commit"],
        "protocols": ["docs/benchmarks/PROSODY-EVAL.md",
                      "benchmarks/run_natgate.py",
                      "scripts/classify_preview.py"],
        "corpus_version": "1.0",
        "generation": {
            "kokoro_meta_git_commit": gen_k["git_commit"],
            "piper_license": gen_p["license"],
            "rows": gen_rows,
        },
        "scoring_signals": ("scripts/classify_preview.py dual-ASR minWER + DNSMOS "
                            "ovrl (speechmos) + rate plausibility; separate process "
                            "per RAM policy"),
        "prosody_detector_source": pros["detector_source"],
        "calibration_reproducibility_check": {
            "note": ("probe implementation re-measured the 3 legacy qex_001 natgate "
                     "anchors; matches PROSODY-EVAL §7 to 3 decimals (1.169/1.122/"
                     "1.256) -> identical code path confirmed"),
            "measured_B_qex001": {"af_heart": 1.1688, "am_michael": 1.1225,
                                  "lessac": 1.2563},
        },
        "clips": clips,
        "teacher_summary_per_voice": teacher_summary,
        "gate_validation": {
            "G1_p1_direction_table": {
                "description": ("rise REQUIRED qex_001+qex_003, FORBIDDEN qex_002+"
                                "qex_006 (B>=1.15 / <1.15); NA exclamations"),
                "items": g1_items,
                "verdict": ("FLAG — teachers fail their own direction table on "
                            f"{g1_total_fail} of {len(g1_verdict_items)} gated items"
                            if g1_total_fail else "PASS"),
                "teacher_compliance_overall": "{} of {} voice-item checks".format(
                    sum(g["n_pass"] for g in g1_verdict_items),
                    sum(g["n_measurable"] for g in g1_verdict_items)),
            },
            "G2_tier_c_band": {
                "X": X_BAND,
                "formal_rule": "|B_ours - median(B_anchor,item)| <= 0.20 (rise-required items)",
                "items": g2_items,
                "verdict": ("FLAG — anchors themselves violate the proposed band/"
                            "spread basis on qex_003 (+qex_006 under the spread "
                            "derivation)" if any(
                                not g["band_ok_vs_median_rule"] or
                                not g["spread_within_X_as_derived"]
                                for g in g2_items) else "PASS"),
            },
            "G3_p3_pause_rule": g3,
            "flags": flags,
        },
    }
    (EXP / "results.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")

    print("=== gate validation ===")
    print(f"G1 direction: {results['gate_validation']['G1_p1_direction_table']['verdict']}")
    print(f"G1 overall compliance: "
          f"{results['gate_validation']['G1_p1_direction_table']['teacher_compliance_overall']}")
    print(f"G2 tier-c band: {results['gate_validation']['G2_tier_c_band']['verdict']}")
    print(f"G3 p3 rule: {g3['clips_pass']}/{g3['clips_total']} new clips pass; "
          f"legacy qex_001 all pass = {g3['legacy_qex001_all_pass']}")
    print(f"\n{len(flags)} FLAGS:")
    for f in flags:
        print(f"  !! {f}\n")
    print("wrote results.json")


if __name__ == "__main__":
    main()
