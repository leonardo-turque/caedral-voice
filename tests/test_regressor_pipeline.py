"""Tests for the CV-G1-E0013 regressor pipeline (real mels, SYNTHETIC durations).

Does NOT require the ALIGN-01 durations file: every dataset exercise uses
``synthetic_uniform_durations=True`` (uniform splits of decoded T_mel) or
locally fabricated durations rows. Covers:
- speaker-slot mapping incl. tNN teacher ids and tvNN fallback;
- RegressorDataset.__getitem__ shapes + synthetic-duration contracts
  (sum == T_mel, one entry per token, all >= 1);
- fabricated-durations validation (length mismatch -> skip, pass-through);
- collate shapes/padding/masks;
- LengthBucketBatchSampler E0013 caps (<=16 utts, <=2000 frames) on DECODED
  mels, exact partition, per-epoch determinism;
- fit_durations_to_frames rounding guard;
- default-config parameter target (~15M) and
- one full training chain (collate -> CaedralRegressor -> AdamW) finite,
  loss-reducing, grads flowing into encoder/decoder/duration predictor while
  frozen-off pitch/energy predictors receive none.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.model.regressor import (  # noqa: E402
    CaedralRegressor,
    CaedralRegressorConfig,
    fit_durations_to_frames,
)
from caedral_voice.training.mel_dataset import (  # noqa: E402
    LengthBucketBatchSampler,
    MEL_FLOOR,
    PhonemeVocab,
)
from caedral_voice.training.regressor_dataset import (  # noqa: E402
    RegressorDataset,
    build_speaker_index,
    collate_regressor,
    load_jsonl,
    stratified_split,
    uniform_durations,
)

MANIFEST = REPO / "data" / "manifests" / "librittsr-smoke-10spk.jsonl"
PHONEMES = REPO / "data" / "manifests" / "librittsr-smoke-10spk.phonemes.jsonl"

N_SUBSET = 24


def _shortest_subset(n: int = N_SUBSET):
    audio = load_jsonl(MANIFEST)
    by_id = {r["id"]: r for r in load_jsonl(PHONEMES)}
    rows = sorted(audio, key=lambda r: r["duration_s"])[:n]
    phones = [by_id[r["id"]] for r in rows]
    return rows, phones


@pytest.fixture(scope="module")
def subset():
    return _shortest_subset()


@pytest.fixture(scope="module")
def dataset(subset):
    rows, phones = subset
    spk = build_speaker_index(rows)
    vocab = PhonemeVocab.build([p["phoneme_string"] for p in phones])
    return RegressorDataset(
        rows, phones, None, spk, vocab=vocab, repo_root=REPO,
        synthetic_uniform_durations=True)


# --------------------------------------------------------------------------- #
# speaker slots
# --------------------------------------------------------------------------- #
def test_teacher_slots_tnn_and_tvnn_fallback(subset):
    rows, _ = subset
    base = build_speaker_index(rows)
    real_prefixes = {r["id"].split("_")[0] for r in rows}
    # tNN-shaped teacher id passes through as its own slot
    teach = [{"id": f"t07_{rows[0]['id']}", **{k: v for k, v in rows[0].items()
                                               if k not in ("id",)}}]
    idx = build_speaker_index(rows + teach)
    assert idx["t07"] == max(idx.values())          # teachers sort last
    assert set(real_prefixes) <= set(idx)
    # non-tNN teacher_voice fallback -> deterministic tvNN slot
    named = [{"id": f"kokoro_{rows[0]['id']}", "teacher_voice": "zf_xiaobei",
              **{k: v for k, v in rows[0].items()
                 if k not in ("id", "audio_path")}},
             {"id": f"kokoro_{rows[1]['id']}", "teacher_voice": "af_heart",
              **{k: v for k, v in rows[1].items()
                 if k not in ("id", "audio_path")}}]
    idx2 = build_speaker_index(rows + named)
    assert idx2["tv00"] < idx2["tv01"]              # sorted voice names
    slot_of = {"af_heart": "tv00", "zf_xiaobei": "tv01"}   # alphabetical
    assert all(idx2[slot_of[v]] > max(base.values())
               for v in slot_of)


def test_uniform_durations_contract():
    assert uniform_durations(100, 7) == [15, 15, 14, 14, 14, 14, 14]
    d = uniform_durations(50, 50)
    assert d == [1] * 50
    assert sum(uniform_durations(97, 12)) == 97


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def test_dataset_items_synthetic_uniform(dataset):
    assert len(dataset) == N_SUBSET
    for i in range(len(dataset)):
        item = dataset[i]
        ids = item["phoneme_ids"]
        durs = item["duration_targets"]
        assert item["mel"].shape[0] == 80
        t = int(item["mel"].shape[1])
        assert item["T_mel"] == t
        assert len(ids) == len(durs)
        assert sum(durs) == t                       # split matches decoded mel
        assert min(durs) >= 1
        assert torch.isfinite(item["mel"]).all()
        assert float(item["mel"].min()) >= math.log(MEL_FLOOR) - 1e-4


def test_dataset_fabricated_durations_pass_through_and_validate(subset):
    rows, phones = subset
    spk = build_speaker_index(rows)
    vocab = PhonemeVocab.build([p["phoneme_string"] for p in phones])
    pick = rows[:3]
    ph_by_id = {r["id"]: r for r in phones}
    ph_pick = [ph_by_id[r["id"]] for r in pick]

    def dur_row(r, mult=2):
        n = len(vocab.encode(ph_by_id[r["id"]]["phoneme_string"]))
        return {"id": r["id"], "durations": [mult] * n,
                "total_frames": mult * n}

    dur_rows = [dur_row(r) for r in pick]
    ds = RegressorDataset(pick, ph_pick, dur_rows, spk, vocab=vocab,
                          repo_root=REPO)
    assert len(ds) == 3
    item = ds[0]
    assert item["duration_targets"] == dur_rows[0]["durations"]  # pass-through
    # sampler estimate uses the aligner total when available (+ slack)
    assert ds.est_frames(0) == dur_rows[0]["total_frames"] + 4

    # a durations row with the WRONG token count drops just that utterance
    bad = dict(dur_rows[0])
    bad["durations"] = dur_rows[0]["durations"][:-1]
    ds2 = RegressorDataset(pick, ph_pick, [bad] + dur_rows[1:], spk,
                           vocab=vocab, repo_root=REPO)
    assert len(ds2) == 2
    assert any("duration length" in why for _, why in ds2.skipped)


def test_missing_durations_file_rows_skipped_when_not_synthetic(subset):
    rows, phones = subset
    spk = build_speaker_index(rows[:4])
    vocab = PhonemeVocab.build([p["phoneme_string"] for p in phones[:4]])
    ds = RegressorDataset(rows[:4], phones[:4], None, spk, vocab=vocab,
                          repo_root=REPO, synthetic_uniform_durations=False)
    assert len(ds) == 0
    assert all(why == "no durations row" for _, why in ds.skipped)


def test_collate_shapes_and_masks(dataset):
    items = [dataset[i] for i in range(4)]
    batch = collate_regressor(items)
    x, y = batch["x"], batch["y"]
    b = len(items)
    tx = max(len(it["phoneme_ids"]) for it in items)
    ty = max(it["T_mel"] for it in items)
    assert x["phoneme_ids"].shape == (b, tx)
    assert x["duration_targets"].shape == (b, tx)
    assert x["mel"].shape == (b, ty, 80) == y.shape
    assert x["phoneme_lengths"].tolist() == [
        len(it["phoneme_ids"]) for it in items]
    assert x["mel_lengths"].tolist() == [it["T_mel"] for it in items]
    assert x["speaker_ids"].tolist() == [it["speaker_id"] for it in items]
    for j, it in enumerate(items):                  # durations sum == mel len
        nx = len(it["phoneme_ids"])
        assert int(x["duration_targets"][j, :nx].sum()) == it["T_mel"]
        assert bool(x["duration_targets"][j, nx:].eq(0).all())
    assert bool(x["phoneme_ids"][0, tx:].eq(0).all()) or \
        x["phoneme_lengths"][0] == tx


def test_sampler_caps_partition_and_epoch_determinism(dataset):
    def actual(i):
        return dataset[i]["T_mel"]

    smp = LengthBucketBatchSampler(list(range(len(dataset))), dataset.est_frames,
                                   max_frames=2000, max_utts=16,
                                   bucket_size=32, base_seed=1234)
    plans = []
    for epoch in (0, 0, 1):
        smp.set_epoch(epoch)
        batches = list(iter(smp))
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(dataset)))     # exact partition
        for b in batches:
            assert 1 <= len(b) <= 16                          # E0013 utt cap
            assert sum(actual(i) for i in b) <= 2000          # HARD cap, real
        plans.append([[dataset.rows[i]["id"] for i in b] for b in batches])
    assert plans[0] == plans[1]                                # same epoch same
    assert any(a != b for a, b in zip(plans[0], plans[2]))     # epochs differ


def test_stratified_split_disjoint_covers(dataset):
    tr, va = stratified_split(dataset.rows, 0.25)
    assert not set(tr) & set(va)
    assert sorted(tr + va) == list(range(len(dataset)))


# --------------------------------------------------------------------------- #
# duration fitting guard
# --------------------------------------------------------------------------- #
def test_fit_durations_surplus_deficit_and_pads():
    d = torch.tensor([[10, 10, 10], [5, 5, 5]], dtype=torch.long)
    xl = torch.tensor([3, 3])
    yl = torch.tensor([32, 12])
    fitted, moved = fit_durations_to_frames(d, yl, phoneme_lengths=xl)
    assert fitted[0].tolist() == [10, 10, 12]       # LAST absorbs remainder
    assert fitted[1].tolist() == [5, 5, 2]          # deficit trimmed at tail
    assert moved == 5                               # |+2| + |-3|
    assert int(fitted[0].sum()) == 32 and int(fitted[1].sum()) == 12
    # padded slots stay 0 and are never touched; deficit trimmed from tail
    # down to the 1-frame floor
    d2 = torch.tensor([[4, 4, 4, 0, 0]], dtype=torch.long)
    fitted2, _ = fit_durations_to_frames(
        d2, torch.tensor([9]), phoneme_lengths=torch.tensor([3]))
    assert fitted2[0].tolist() == [4, 4, 1, 0, 0]
    assert int(fitted2[0].sum()) == 9


# --------------------------------------------------------------------------- #
# model-level contracts
# --------------------------------------------------------------------------- #
SMALL_CFG = dict(
    name="regressor-test", d_model=128, n_heads=4, n_kv_heads=4,
    n_layers_enc=1, n_layers_dec=2, decoder_conv_layers=1,
    decoder_conv_kernel=3, variance_hidden=64, variance_layers=2,
)


def test_default_config_hits_15m_target():
    cfg = CaedralRegressorConfig()
    model = CaedralRegressor(cfg)
    n = model.count_params()
    assert 13_500_000 <= n <= 16_500_000, n         # ~15M between tiny/lite
    # v0 freezing contract
    assert not model.use_pitch and not model.use_energy
    assert all(not p.requires_grad for p in model.pitch_predictor.parameters())
    assert all(not p.requires_grad for p in model.energy_predictor.parameters())


def test_training_chain_finite_grads_reduce_loss(dataset):
    info_spk = max(r["speaker_id"] for r in dataset.rows) + 1
    vocab_size = dataset.vocab.size
    cfg = CaedralRegressorConfig(
        vocab_size=max(200, vocab_size), n_speakers=max(8, info_spk),
        seed=1234, **SMALL_CFG)
    model = CaedralRegressor(cfg)
    torch.manual_seed(1234)

    # smallest utterances keep this test fast; batch of 4 fits caps easily
    order = sorted(range(len(dataset)), key=lambda i: dataset[i]["T_mel"])[:4]
    batch = collate_regressor([dataset[i] for i in order])
    x = batch["x"]

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-6)
    losses, specs = [], []
    model.train()
    for _ in range(5):
        out = model.losses_for_training(
            mel=x["mel"], phoneme_ids=x["phoneme_ids"],
            durations=x["duration_targets"], speaker_id=x["speaker_ids"],
            style_id=0, mel_lengths=x["mel_lengths"],
            phoneme_lengths=x["phoneme_lengths"])
        assert math.isfinite(float(out["loss"]))
        assert math.isfinite(float(out["mel_loss"]))
        assert math.isfinite(float(out["spec_loss"]))
        assert math.isfinite(float(out["dur_loss"]))
        assert out["n_duration_adjusted"] == 0      # synthetic splits fit
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()

        grads = {n: p.grad for n, p in model.named_parameters()}
        for probe in ("encoder.blocks.0.attn.q_proj.weight",
                      "decoder.blocks.0.attn.q_proj.weight",
                      "decoder.films.0.proj.weight",
                      "duration_predictor.convs.0.weight",
                      "conv_front.0.weight",
                      "mel_proj.weight"):
            g = grads[probe]
            assert g is not None and torch.isfinite(g).all(), probe
        assert next(iter(model.pitch_predictor.parameters())).grad is None
        assert all(p.grad is None
                   for p in model.energy_predictor.parameters())
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0)
        assert math.isfinite(float(gn))
        opt.step()
        losses.append(float(out["loss"]))
        specs.append(float(out["mel_loss"]))

    assert losses[-1] < losses[0]                   # reduces over 5 steps
    # inference path: override durations -> mel with matching frame budget
    model.eval()
    with torch.no_grad():
        inf = model(x["phoneme_ids"][:1], speaker_id=0,
                    durations=x["duration_targets"][:1, :x["phoneme_lengths"][0]])
    assert inf["mel"].shape[0] == 1 and inf["mel"].shape[2] == cfg.n_mels
    assert int(inf["n_mel_frames"]) == \
        int(x["duration_targets"][0, :x["phoneme_lengths"][0]].sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
