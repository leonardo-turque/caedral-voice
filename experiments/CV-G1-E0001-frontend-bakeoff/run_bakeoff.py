"""CV-G1-E0001 — text frontend bake-off runner (Phase E).

Runs the frozen eval corpus through the Caedral candidate lexicon frontend
and the dev-only espeak-ng reference, producing results/*.jsonl + summary.json:

  a. char->phoneme-token length ratio (corpus-wide, per category)
  b. CMUdict OOV rate + top-30 OOV words
  c. batch-1 latency: normalize / g2p / total (ms)
  d. heteronym torture (het_001..het_010): flagging recall vs known targets
  e. reference comparison vs espeak_ref on 20 deterministic samples
     (word-stress placement agreement; method documented inline)
  f. TN accuracy hand-gold checks (num_001..num_010)

Local CPU only; no TTS models. Deterministic sampling everywhere.
"""
from __future__ import annotations

import io
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CORPUS_PATH = REPO / "benchmarks" / "corpus" / "eval-corpus-v1.0.json"
OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "results"

HETERONYM_TARGETS = {
    "het_001": "bow", "het_002": "wind", "het_003": "tear",
    "het_004": "close", "het_005": "lead", "het_006": "object",
    "het_007": "produce", "het_008": "dove", "het_009": "wound",
    "het_010": "desert",
}

# Hand-authored gold verbalization substrings (mission f): each check must
# appear verbatim in the normalized text. Authored by hand from the frozen
# corpus texts before running the engine.
TN_GOLD = {
    "num_001": ["one thousand two hundred forty-seven dollars",
                "eighty-three cents", "thirty days"],
    "num_002": ["forty-eight thousand two hundred thirteen people",
                "twelve point five percent"],
    "num_003": ["February twenty-ninth", "nineteen ninety-six"],
    "num_004": ["five five five oh one four two", "nine thirty A M",
                "five o'clock P M"],
    "num_005": ["three hundred fifty grams", "two eggs",
                "one hundred eighty milliliters"],
    "num_006": ["nineteen forty-seven", "fourteen percent",
                "the third quarter of twenty twenty-six", "two point one percent"],
    "num_007": ["two thousand four hundred euros", "one hundred fifty euros"],
    "num_008": ["two hours", "zero minutes", "thirty-five seconds"],
    "num_009": ["plus four four two oh seven nine four six oh nine five eight"],
    "num_010": ["Version two point eleven point three", "forty-seven bugs",
                "sixteen thousand three hundred eighty-four simultaneous connections"],
}

REF_SAMPLE_N = 20
_VOWEL_CHARS = set("aeiouyɑæɒʌɔəɛɜɝɪʊ")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_corpus():
    data = json.loads(io.open(CORPUS_PATH, encoding="utf-8").read())
    items = []
    for cat in data["categories"]:
        for item in cat["items"]:
            items.append({
                "id": item["id"], "text": item["text"],
                "category": cat["id"],
                "expected": item.get("expected"),
            })
    return data["version"], items


def jdump(path: Path, rows: list[dict]) -> None:
    with io.open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def stressed_nuclei(ipa_word: str) -> tuple[str, ...]:
    """Primary-stress vowel nuclei of an IPA word chunk (experiment e method).

    Scan left to right; after each primary stress mark ˈ take the next vowel
    character as the stressed nucleus. Returns a sorted unique tuple so
    onset-cluster differences in stress-mark placement don't create false
    mismatches; only WHICH vowel carries stress matters.
    """
    nuclei: list[str] = []
    pending_primary = False
    for ch in ipa_word:
        if ch == "ˈ":
            pending_primary = True
            continue
        if ch in _VOWEL_CHARS:
            if pending_primary:
                nuclei.append(ch)
                pending_primary = False
    return tuple(sorted(set(nuclei)))


def espeak_words(ipa_text: str) -> list[str]:
    """Split espeak IPA output into word chunks, dropping residual punctuation."""
    cleaned = re.sub(r"[^\wˈˌːɑæɒʌɔəɛɜɝɪʊʃʒθðŋɹ]+", " ", ipa_text)
    return [w for w in cleaned.split() if any(c.isalnum() or c in _VOWEL_CHARS for c in w)]


def main() -> None:
    from benchmarks.harness.system_info import capture
    from caedral_voice.text import TextFrontend, get_lexicon, tokenize
    from caedral_voice.text.normalize import (
        _ascii_fold,
        normalize,
        normalize_detailed,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    corpus_version, items = load_corpus()

    # Force lexicon load BEFORE any timing.
    lexicon = get_lexicon()
    frontend = TextFrontend("lexicon")
    ref_frontend = TextFrontend("espeak_ref")  # dev-only GPL path, lazy-wired

    config = {
        "experiment_id": "CV-G1-E0001",
        "title": "Text frontend bake-off (length ratio, OOV, latency, "
                 "heteronyms, espeak reference agreement, TN accuracy)",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "machine": capture(),
        "corpus_id": "caedral-eval-corpus",
        "corpus_version": corpus_version,
        "candidate_backend": "caedral_voice.text TextFrontend('lexicon') "
                             "(CMUdict 0.7b -> own ARPAbet-to-IPA map -> morphology "
                             "-> letter rules; rule-based own TN)",
        "reference_backend": "TextFrontend('espeak_ref') — DEV ONLY GPLv3 "
                             "(phonemizer + espeakng_loader), never shippable",
        "lexicon_path": str(Path(lexicon.path)),
        "lexicon_entries": len(lexicon),
        "heteronym_targets": HETERONYM_TARGETS,
        "tn_gold_checks": {k: len(v) for k, v in TN_GOLD.items()},
        "ref_sample_n": REF_SAMPLE_N,
        "ref_sampling": f"deterministic even spacing over sorted item ids, n={REF_SAMPLE_N}",
        "stress_agreement_method": (
            "Both outputs fed OUR normalized text. Per aligned word pair "
            "(positional, up to min word count), compare the set of vowels "
            "immediately following a primary stress mark ˈ; a pair agrees when "
            "the stressed-nuclei sets are equal. Onset-cluster placement "
            "differences of ˈ itself are thus ignored; WHICH vowel carries "
            "primary stress must match."
        ),
        "phoneme_token_definition": (
            "One token per IPA base symbol incl. stress marks ˈ ˌ; length mark "
            "ː merges into its vowel; pause marks | ‖ counted separately."
        ),
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Warmup (excluded from timing): one short item through both paths.
    # ------------------------------------------------------------------
    frontend.process("Warmup sentence number twelve.")
    ref_frontend.process("Warmup sentence.")

    # ------------------------------------------------------------------
    # a+b+c: full corpus pass
    # ------------------------------------------------------------------
    ratio_rows: list[dict] = []
    oov_rows: list[dict] = []
    latency_rows: list[dict] = []
    oov_counter: dict[str, int] = {}
    total_word_tokens = 0
    total_raw_misses = 0
    total_fallback_tokens = 0

    for item in items:
        text = item["text"]

        t0 = time.perf_counter_ns()
        norm = normalize_detailed(text)
        t1 = time.perf_counter_ns()
        result = frontend.process(text)
        t2 = time.perf_counter_ns()

        norm_ms = (t1 - t0) / 1e6
        total_ms = (t2 - t0) / 1e6
        g2p_ms = total_ms - norm_ms
        latency_rows.append({
            "item_id": item["id"], "category": item["category"],
            "chars": len(text),
            "normalize_ms": round(norm_ms, 4),
            "g2p_ms": round(g2p_ms, 4),
            "total_ms": round(total_ms, 4),
        })

        # --- OOV accounting (word tokens after normalization) ---
        tokens = [t for t in tokenize(norm.text)]
        words = [t for t in tokens if t[0].isalpha()]
        raw_miss, fallback_words = [], []
        for w in words:
            hit = lexicon.contains(w)
            if not hit:
                folded = _ascii_fold(w).lower()
                hit = folded != w.lower() and lexicon.contains(folded)
            if not hit:
                raw_miss.append(w.lower())
                oov_counter[w.lower()] = oov_counter.get(w.lower(), 0) + 1
        wp_by_token = [wp for wp in result.words]
        n_fb = sum(1 for wp in wp_by_token if wp.source == "rule_fallback")
        total_word_tokens += len(words)
        total_raw_misses += len(raw_miss)
        total_fallback_tokens += n_fb
        oov_rows.append({
            "item_id": item["id"], "category": item["category"],
            "word_tokens": len(words),
            "raw_lexicon_misses": len(raw_miss),
            "missed_words": raw_miss,
            "rule_fallback_tokens_in_stream": n_fb,
        })

        # --- length ratio ---
        if not text.strip():
            ratio_rows.append({
                "item_id": item["id"], "category": item["category"],
                "chars": len(text), "skipped": "empty_input",
                "expected": item["expected"],
            })
            continue
        try:
            pcount = result.phoneme_count
            pcount_incl_pauses = len(result.phoneme_tokens)
        except Exception as exc:  # graceful skip contract
            ratio_rows.append({
                "item_id": item["id"], "category": item["category"],
                "chars": len(text), "skipped": f"error:{type(exc).__name__}",
                "expected": item["expected"],
            })
            continue
        ratio_rows.append({
            "item_id": item["id"], "category": item["category"],
            "chars": len(text), "normalized_chars": len(norm.text),
            "phoneme_tokens_excl_pauses": pcount,
            "phoneme_tokens_incl_pauses": pcount_incl_pauses,
            "ratio_excl_pauses": round(pcount / max(len(text), 1), 4),
            "ratio_incl_pauses": round(pcount_incl_pauses / max(len(text), 1), 4),
        })

    jdump(RESULTS / "length_ratio.jsonl", ratio_rows)
    jdump(RESULTS / "oov_per_item.jsonl", oov_rows)
    jdump(RESULTS / "latency.jsonl", latency_rows)

    top30 = sorted(oov_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    (RESULTS / "oov_top30.json").write_text(json.dumps({
        "total_word_tokens_after_normalization": total_word_tokens,
        "raw_lexicon_misses": total_raw_misses,
        "raw_miss_rate": round(total_raw_misses / max(total_word_tokens, 1), 4),
        "rule_fallback_tokens_in_phoneme_stream": total_fallback_tokens,
        "rule_fallback_rate_vs_raw_misses": round(
            total_fallback_tokens / max(total_raw_misses, 1), 4),
        "top30_oov": [{"word": w, "count": c} for w, c in top30],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # d: heteronym torture
    # ------------------------------------------------------------------
    het_rows: list[dict] = []
    tgt_total = tgt_flagged = 0
    non_target_flags: dict[str, int] = {}
    for item in items:
        if item["category"] != "heteronyms":
            continue
        target = HETERONYM_TARGETS[item["id"]]
        result = frontend.process(item["text"])
        occurrences = [wp for wp in result.words if wp.token.lower().strip("'’") == target]
        flagged = [wp for wp in occurrences if wp.ambiguous]
        others = [wp for wp in result.words
                  if wp.ambiguous and wp.token.lower().strip("'’") != target]
        for wp in others:
            non_target_flags[wp.token.lower()] = non_target_flags.get(wp.token.lower(), 0) + 1
        tgt_total += len(occurrences)
        tgt_flagged += len(flagged)
        het_rows.append({
            "item_id": item["id"],
            "target_heteronym": target,
            "text": item["text"],
            "target_occurrences": len(occurrences),
            "target_flagged_ambiguous": len(flagged),
            "target_options": occurrences[0].options if occurrences else [],
            "selected_first_sense": occurrences[0].selected if occurrences else None,
            "fully_flagged": len(occurrences) == len(flagged),
            "non_target_words_marked_ambiguous": [
                {"token": wp.token, "options": wp.options} for wp in others
            ],
        })
    jdump(RESULTS / "heteronyms.jsonl", het_rows)

    # ------------------------------------------------------------------
    # e: reference comparison (20 deterministic samples)
    # ------------------------------------------------------------------
    ordered = sorted(items, key=lambda it: it["id"])
    n = len(ordered)
    sample_ids = [ordered[round(k * (n - 1) / (REF_SAMPLE_N - 1))]["id"]
                  for k in range(REF_SAMPLE_N)]
    by_id = {it["id"]: it for it in items}
    ref_rows: list[dict] = []
    agree_words = aligned_pairs = 0
    agree_ref_marked = ref_marked_pairs = 0
    for sid in sample_ids:
        item = by_id[sid]
        ours = frontend.process(item["text"])
        our_words = [wp.selected for wp in ours.words]
        try:
            ref_ipa = ref_frontend.phonemize(item["text"])
        except Exception as exc:
            ref_rows.append({"item_id": sid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ref_w = espeak_words(ref_ipa)
        pairs = list(zip([re.sub(r"[^\wˈˌːɑæɒʌɔəɛɜɝɪʊʃʒθðŋɹ]", "", w) for w in our_words], ref_w))
        item_agree = item_agree_marked = 0
        for ow, rw in pairs:
            a, b = stressed_nuclei(ow), stressed_nuclei(rw)
            ok = (a == b) and bool(a)
            item_agree += ok
            if b:  # espeak actually marked a primary stress on this word
                ref_marked_pairs += 1
                item_agree_marked += ok
        agree_words += item_agree
        agree_ref_marked += item_agree_marked
        aligned_pairs += len(pairs)
        ref_rows.append({
            "item_id": sid, "category": item["category"],
            "our_word_count": len(our_words), "ref_word_count": len(ref_w),
            "aligned_pairs": len(pairs),
            "stress_agree": item_agree,
            "stress_agreement_rate": round(item_agree / max(len(pairs), 1), 4),
            "ref_marked_pairs_in_item": sum(
                1 for ow, rw in pairs if stressed_nuclei(rw)),
            "agree_when_ref_marked": item_agree_marked,
            "ours_ipa": " ".join(our_words)[:400],
            "ref_ipa": ref_ipa[:400],
        })
    jdump(RESULTS / "ref_comparison.jsonl", ref_rows)

    # ------------------------------------------------------------------
    # f: TN hand-gold accuracy
    # ------------------------------------------------------------------
    tn_rows: list[dict] = []
    checks_passed = checks_total = 0
    for nid, golds in TN_GOLD.items():
        item = by_id[nid]
        out = normalize(item["text"])
        checks = [{"gold": g, "present": g in out} for g in golds]
        passed = sum(c["present"] for c in checks)
        checks_passed += passed
        checks_total += len(checks)
        tn_rows.append({
            "item_id": nid,
            "source": item["text"],
            "normalized": out,
            "checks_passed": passed,
            "checks_total": len(checks),
            "mismatches": [c["gold"] for c in checks if not c["present"]],
        })
    jdump(RESULTS / "tn_accuracy.jsonl", tn_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    valid_ratios = [r["ratio_excl_pauses"] for r in ratio_rows if "ratio_excl_pauses" in r]
    valid_ratios_ip = [r["ratio_incl_pauses"] for r in ratio_rows if "ratio_incl_pauses" in r]
    by_cat: dict[str, list[float]] = {}
    for r in ratio_rows:
        if "ratio_excl_pauses" in r:
            by_cat.setdefault(r["category"], []).append(r["ratio_excl_pauses"])
    skipped = [r for r in ratio_rows if "skipped" in r]

    def med(field):
        vals = [r[field] for r in latency_rows]
        return {
            "median_ms": round(statistics.median(vals), 4),
            "mean_ms": round(statistics.fmean(vals), 4),
            "min_ms": round(min(vals), 4),
            "max_ms": round(max(vals), 4),
        }

    summary = {
        "experiment_id": "CV-G1-E0001",
        "items_total": len(items),
        "items_skipped_gracefully": [
            {"item_id": r["item_id"], "reason": r["skipped"]} for r in skipped],
        "a_length_ratio": {
            "definition": config["phoneme_token_definition"],
            "mean_ratio_chars_to_phonemes_excl_pauses": round(statistics.mean(valid_ratios), 4),
            "mean_ratio_chars_to_phonemes_incl_pauses": round(statistics.mean(valid_ratios_ip), 4),
            "per_category_mean_ratio_excl_pauses": {
                k: round(statistics.mean(v), 4) for k, v in sorted(by_cat.items())
            },
        },
        "b_oov": {
            "word_tokens_after_normalization": total_word_tokens,
            "raw_lexicon_misses": total_raw_misses,
            "raw_miss_rate": round(total_raw_misses / max(total_word_tokens, 1), 4),
            "rule_fallback_tokens": total_fallback_tokens,
            "top30_file": "results/oov_top30.json",
        },
        "c_latency_batch1_ms": {
            "normalize": med("normalize_ms"),
            "g2p": med("g2p_ms"),
            "total": med("total_ms"),
        },
        "d_heteronyms": {
            "targets_total_occurrences": tgt_total,
            "targets_flagged": tgt_flagged,
            "flag_recall": round(tgt_flagged / max(tgt_total, 1), 4),
            "non_target_flags": non_target_flags,
        },
        "e_reference_comparison": {
            "sampled_items": sample_ids,
            "aligned_word_pairs": aligned_pairs,
            "stress_matching_pairs": agree_words,
            "agreement_rate_strict": round(agree_words / max(aligned_pairs, 1), 4),
            "ref_marked_pairs": ref_marked_pairs,
            "agree_when_ref_marks_stress": agree_ref_marked,
            "agreement_rate_ref_marked_only": round(
                agree_ref_marked / max(ref_marked_pairs, 1), 4),
            "method": config["stress_agreement_method"],
            "caveat": "Strict rate penalizes CMUdict citation stress on "
                      "function words that espeak leaves unmarked in context; "
                      "the ref_marked_only rate isolates content-word stress "
                      "placement agreement.",
        },
        "f_tn_accuracy": {
            "gold_checks_total": checks_total,
            "gold_checks_passed": checks_passed,
            "accuracy": round(checks_passed / max(checks_total, 1), 4),
            "detail_file": "results/tn_accuracy.jsonl",
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
