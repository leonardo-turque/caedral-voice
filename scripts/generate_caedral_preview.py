"""Generate Caedral preview WAVs from a training checkpoint.

FIRST-CAEDRAL-VOICE track (founder directive): earliest decodable checkpoint
-> artifacts/latest-preview/caedral-preview-earliest.wav, then refreshed as
caedral-preview-current-best.wav on measurable improvement.

Path: OUR frontend -> OUR checkpoint -> mel -> PRETRAINED Vocos decode.
The model's own iSTFT head is an untrained stub at pilot stage, so previews
use the MIT-licensed Vocos decoder; this dependency is documented in the
generated README and docs/data/DECODER-LICENSES.md. Mel adapter: our 80-band
ln-mel is linearly interpolated to Vocos's 100-band grid — an approximation,
labeled as such.

Usage:
  python scripts/generate_caedral_preview.py --run-dir <exp>/pilot-tiny-15k \
      --checkpoint checkpoints/step_000001000.pt --label earliest

After generation each wav is automatically classified by
scripts/classify_preview.py (EVAL-CLASSIFY-01: dual-ASR minWER + repetition +
rate + spectral flatness + DNSMOS) using --reference-texts
(default: benchmarks/preview-sentences.json, sections matched from filenames).
Labels land per-file in preview-provenance.json; NATURAL is a CANDIDATE label
that still requires human confirmation before any quality claim.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.harness.measure import measure, rtf  # noqa: E402
from benchmarks.harness.system_info import capture  # noqa: E402
from caedral_voice.model.caedral_nar import CaedralNAR  # noqa: E402
from caedral_voice.text import TextFrontend  # noqa: E402

SENTENCES = json.loads(io.open(
    REPO / "benchmarks" / "preview-sentences.json", encoding="utf-8").read())
OUT = REPO / "artifacts" / "latest-preview"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def text_to_ids(text: str, symbols: list[str], specials: dict) -> tuple[list[int], list[str]]:
    res = TextFrontend("lexicon+g2pen").process(text)
    sym2id = {s: i for i, s in enumerate(symbols)}
    ids: list[int] = []
    skipped = []
    for tok in res.phoneme_tokens:
        if tok in sym2id:
            ids.append(sym2id[tok])
        elif tok == "|":
            ids.append(int(specials.get("|", 1)))
        else:
            skipped.append(tok)
    return ids, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--speakers", default="0,1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", required=True,
                    help="earliest | current-best | other tag")
    ap.add_argument("--reference-texts", default=None,
                    help="JSON mapping sections/filenames/stems -> reference "
                         "text for automatic classification "
                         "(default: benchmarks/preview-sentences.json)")
    ap.add_argument("--skip-classification", action="store_true",
                    help="skip the automatic perceptual classification pass")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from caedral_voice.model.caedral_nar import CaedralNAR, CaedralNARConfig
    known = set(CaedralNARConfig.__dataclass_fields__)
    # The run's config file is authoritative; checkpoint cfg may be a partial
    # wrapper dict. Start from the file, overlay ck values, force use_flow.
    base = json.loads(io.open(REPO / "configs" / "tiny.json", encoding="utf-8").read())
    base.update({k: v for k, v in ck["cfg"].items() if k in known})
    base["use_flow"] = True
    cfg = base
    model = CaedralNAR(cfg)
    state = {k.replace("model.", "", 1): v for k, v in ck["model"].items()
             if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    vocab = json.loads(io.open(run_dir / "checkpoints" / "phoneme_vocab.json",
                               encoding="utf-8").read())
    specials = vocab.get("specials", {})
    symbols = vocab["symbols"]
    spk_index = json.loads(io.open(run_dir / "checkpoints" / "speaker_index.json",
                                   encoding="utf-8").read())

    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    step = ck.get("trainer", {}).get("global_step", None)
    if step is None:
        step = ck.get("global_step", "unknown")

    rows = []
    for spk_name, spk_id in sorted(spk_index.items(), key=lambda kv: kv[1]):
        if str(spk_id) not in args.speakers.split(","):
            continue
        for section in ("neutral", "conversational", "expressive"):
            text = SENTENCES[section][0]
            ids, skipped = text_to_ids(text, symbols, specials)
            with torch.no_grad():
                result, perf = measure(
                    lambda i=ids, s=spk_id: model.infer(
                        torch.tensor([i]), speaker_id=s, style_id=0))
            mel = result["mel"].squeeze(0).numpy()          # (T, 80)
            audio_s_est = mel.shape[0] * cfg["hop_length"] / cfg["sample_rate"]
            # 80-band ln-mel -> vocos 100-band grid (approximation, documented)
            xs100 = np.linspace(0, 79, 100)
            mel100 = np.stack(
                [np.interp(xs100, np.arange(80), frame) for frame in mel],
                axis=0).T.astype(np.float32)                # (100, T)
            mel100_t = torch.from_numpy(mel100).unsqueeze(0)   # (1, 100, T)
            with torch.no_grad():
                wav = vocos.decode(mel100_t).squeeze(0).numpy()
            fname = f"caedral-preview-{args.label}-spk{spk_id}-{section}.wav"
            sf.write(out_dir / fname, wav, cfg["sample_rate"])
            rows.append({
                "file": fname, "speaker": spk_name, "speaker_id": spk_id,
                "section": section, "chars": len(text),
                "skipped_tokens": sorted(set(skipped))[:10],
                "audio_seconds": round(len(wav) / cfg["sample_rate"], 3),
                "rtf_full_infer": round(rtf(perf.wall_seconds, max(audio_s_est, 0.1)), 4),
            })
            print(f"[spk{spk_id}] {section}: {rows[-1]['audio_seconds']}s "
                  f"RTF {rows[-1]['rtf_full_infer']} skipped={len(skipped)}")

    # --- automatic perceptual classification (EVAL-CLASSIFY-01) -------------
    # Every preview gets an audit label from MEASURED audio signals before any
    # quality claim is allowed; see scripts/classify_preview.py for thresholds.
    classification_block = None
    if not args.skip_classification:
        sys.path.insert(0, str(REPO / "scripts"))
        import classify_preview as cp
        ref_path = Path(args.reference_texts) if args.reference_texts \
            else REPO / "benchmarks" / "preview-sentences.json"
        refs = cp.load_reference_map(ref_path) if ref_path.exists() else {}
        if not refs:
            print(f"WARNING: no reference texts at {ref_path}; labels are "
                  "capped below INTELLIGIBLE")
        counts: dict[str, int] = {}
        for row in rows:
            ref_text = cp.resolve_reference(refs, Path(row["file"]))
            try:
                res = cp.classify_file(out_dir / row["file"], ref_text)
                print(cp.format_line(res))
            except Exception as e:  # never kill generation over labeling
                res = {"class": "ERROR", "error": repr(e)}
                print(f"[ERROR] {row['file']}: {e}")
            row["classification"] = res
            counts[res["class"]] = counts.get(res["class"], 0) + 1
        classification_block = {
            "tool": "scripts/classify_preview.py (EVAL-CLASSIFY-01)",
            "note": ("AUTOMATIC pre-audit labels from MEASURED signals only; "
                     "NATURAL is a candidate label and still requires human "
                     "confirmation before any quality claim."),
            "reference_texts": str(ref_path),
            "counts": counts,
            "thresholds": cp.THRESHOLDS,
        }

    readme = {
        "label": args.label,
        "warning": ("EARLY TRAINING CHECKPOINT - NOT QUALITY REPRESENTATIVE"
                    if args.label == "earliest"
                    else "PROTOTYPE - NOT PRODUCTION QUALITY"),
        "checkpoint": str(ckpt_path), "step": step,
        "architecture": f"CAEDRAL-NAR tiny use_flow params~8.7M (E0006 MAS+flow)",
        "decoder": "pretrained Vocos charactr/mel-24khz via 80->100 mel-band interpolation (approximation)",
        "frontend": "lexicon+g2pen shippable ladder (E0001/E0004)",
        "sample_rate": cfg["sample_rate"],
        "git_commit": git_commit(),
        "machine": capture(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "classification": classification_block,
        "known_limitations": [
            "trained on 1.66 h / 10 speakers only",
            "style table has no style-labeled data (style_id=0 neutral)",
            "mel adapter approximation vs vocos native features",
            "expected quality: babble-to-rudimentary at early steps",
        ],
    }
    (out_dir / "preview-provenance.json").write_text(
        json.dumps(readme, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} previews + preview-provenance.json -> {out_dir}")


if __name__ == "__main__":
    main()
