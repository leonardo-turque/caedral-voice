"""Tests for the CV-G1-E0007 pilot training pipeline (real smoke data).

Covers, on the real librittsr-smoke-10spk manifest:
- PhonemeVocab determinism (build order, specials, JSON round-trip);
- PilotDataset.__getitem__ shapes on real manifest rows (+mel floor, cache);
- LengthBucketBatchSampler hard bounds on DECODED mels (<=8 utts,
  <=1600 frames), exact partition, per-epoch determinism;
- stratified-by-speaker split;
- multi-manifest union (second manifest synthesized into a temp tree);
- one full training step chain (collate -> tiny(use_flow=True) ->
  losses_for_training -> AdamW step) runs finite and reduces flow_loss.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caedral_voice.model.caedral_nar import CaedralNARConfig  # noqa: E402
from caedral_voice.training.mel_dataset import (  # noqa: E402
    LengthBucketBatchSampler,
    MEL_FLOOR,
    PhonemeVocab,
    PilotDataset,
)
from caedral_voice.training.seed import set_determinism  # noqa: E402

sys.path.insert(0, str(REPO / "experiments" / "CV-G1-E0007-pilot-training"))
import pilot_train  # noqa: E402

MANIFEST = REPO / "data" / "manifests" / "librittsr-smoke-10spk.jsonl"
PHONEMES = REPO / "data" / "manifests" / "librittsr-smoke-10spk.phonemes.jsonl"
TINY_CONFIG = REPO / "configs" / "tiny.json"


def _rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


@pytest.fixture(scope="module")
def corpus():
    return _rd(MANIFEST), _rd(PHONEMES)


@pytest.fixture(scope="module")
def full_dataset(corpus):
    audio, phone = corpus
    spk = pilot_train.build_speaker_index(audio)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in phone])
    return PilotDataset(audio, phone, spk, vocab=vocab, repo_root=REPO)


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
def test_vocab_determinism_and_specials(corpus, tmp_path):
    audio, phone = corpus
    strings = [r["phoneme_string"] for r in phone]

    v1 = PhonemeVocab.build(strings)
    v2 = PhonemeVocab.build(strings)
    shuffled = list(strings)
    random.Random(99).shuffle(shuffled)
    v3 = PhonemeVocab.build(shuffled)

    assert v1.symbols == v2.symbols == v3.symbols
    assert v1.sym_to_id == v2.sym_to_id == v3.sym_to_id

    # specials pinned exactly
    assert v1.sym_to_id["<pad>"] == 0
    assert v1.sym_to_id["|"] == 1
    assert v1.sym_to_id["‖"] == 2
    rest = [s for s in v1.symbols[3:]]
    assert rest == sorted(rest)                       # deterministic sorted
    assert " " not in v1.symbols                      # separator never a symbol

    p = tmp_path / "vocab.json"
    v1.to_json(p)
    v4 = PhonemeVocab.from_json(p)
    assert v4 == v1 and v4.encode(strings[0]) == v1.encode(strings[0])

    # word boundary encodes to its special id inside a real string
    sample = next(s for s in strings if "|" in s)
    ids = v1.encode(sample)
    assert 1 in ids and all(isinstance(i, int) for i in ids)
    assert max(ids) < v1.size


def test_vocab_roundtrip_and_unknown_char(corpus, tmp_path):
    _, phone = corpus
    v = PhonemeVocab.build([r["phoneme_string"] for r in phone])
    with pytest.raises(KeyError):
        v.encode("€")                                 # char absent from corpus


# --------------------------------------------------------------------------- #
# dataset items on real rows
# --------------------------------------------------------------------------- #
def test_dataset_getitem_shapes_on_real_rows(corpus):
    audio, phone = corpus
    pick = [audio[0], audio[len(audio) // 2], audio[-1]]
    by_id = {r["id"]: r for r in phone}
    ph = [by_id[r["id"]] for r in pick]
    spk = pilot_train.build_speaker_index(pick)
    ds = PilotDataset(pick, ph, spk, repo_root=REPO)
    assert len(ds) == 3
    for i in range(3):
        item = ds[i]
        assert set(item) >= {"phoneme_ids", "mel", "T_mel", "speaker_id"}
        assert isinstance(item["phoneme_ids"], list)
        assert all(isinstance(x, int) for x in item["phoneme_ids"])
        t = item["mel"].shape[1]
        assert item["mel"].shape == (80, t)
        assert item["T_mel"] == t
        assert item["speaker_id"] == spk[pick[i]["id"].split("_")[0]]
        assert item["mel"].dtype == torch.float32
        assert torch.isfinite(item["mel"]).all()
        assert float(item["mel"].min()) >= math.log(MEL_FLOOR) - 1e-4
        # MAS needs >= 1 frame per phone
        assert t >= len(item["phoneme_ids"])


def test_dataset_memoizes_mel(full_dataset):
    a = full_dataset[0]["mel"]
    b = full_dataset[0]["mel"]
    assert a is b                                     # cached, not recomputed


def test_dataset_preflight_skips_missing_audio(corpus):
    audio, phone = corpus
    broken = [dict(audio[0]), dict(audio[1])]
    broken[0]["audio_path"] = "data/processed/nowhere/missing.wav"
    broken[0]["id"] = "999_missing_000000_000000"
    spk = pilot_train.build_speaker_index(broken)
    spk["999"] = len(spk)
    by_id = {r["id"]: r for r in phone}
    ph = [by_id.get(r["id"], {"id": r["id"],
                              "phoneme_string": "m æ θ",
                              "n_phones": 3}) for r in broken]
    ds = PilotDataset(broken, ph, spk, repo_root=REPO)
    assert len(ds) == 1
    assert any("missing audio" in why for _, why in ds.skipped)


# --------------------------------------------------------------------------- #
# batch sampler bounds on decoded mels
# --------------------------------------------------------------------------- #
def test_batch_sampler_bounds_partition_and_epoch_determinism(corpus):
    audio, phone = corpus
    # shortest ~100 utterances keeps decode time bounded
    rows = sorted(audio, key=lambda r: r["duration_s"])[:100]
    by_id = {r["id"]: r for r in phone}
    ph = [by_id[r["id"]] for r in rows]
    spk = pilot_train.build_speaker_index(rows)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in ph])
    ds = PilotDataset(rows, ph, spk, vocab=vocab, repo_root=REPO)

    def actual_frames(i):
        return ds[i]["T_mel"]

    smp = LengthBucketBatchSampler(list(range(len(ds))), ds.est_frames,
                                   max_frames=1600, max_utts=8,
                                   bucket_size=32, base_seed=1234)
    seen_plans = []
    for epoch in (0, 0, 1):
        smp.set_epoch(epoch)
        batches = list(iter(smp))
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(ds)))    # exact partition
        assert len(batches) == len(smp)
        for b in batches:
            assert 1 <= len(b) <= 8                    # utterance bound
            tot = sum(actual_frames(i) for i in b)
            assert tot <= 1600                          # HARD bound on real mels
        seen_plans.append([[ds.rows[i]["id"] for i in b] for b in batches])
    assert seen_plans[0] == seen_plans[1]              # same epoch => same plan
    assert any(a != b for a, b in zip(seen_plans[0], seen_plans[2]))


def test_val_sampler_fixed_plan(corpus):
    audio, phone = corpus
    rows = sorted(audio, key=lambda r: r["duration_s"])[:60]
    by_id = {r["id"]: r for r in phone}
    ph = [by_id[r["id"]] for r in rows]
    spk = pilot_train.build_speaker_index(rows)
    ds = PilotDataset(rows, ph, spk, repo_root=REPO)
    smp = LengthBucketBatchSampler(list(range(len(ds))), ds.est_frames,
                                   shuffle=False, base_seed=7)
    p1 = [list(b) for b in iter(smp)]
    smp.set_epoch(5)
    p2 = [list(b) for b in iter(smp)]
    assert p1 == p2                                    # val plan never drifts
    frames = [sum(ds.est_frames(i) for i in b) for b in p1]
    assert all(f <= 1600 for f in frames)
    assert all(len(b) <= 8 for b in p1)
    # fixed plan is globally length-sorted (ascending batches)
    means = [sum(ds.est_frames(i) for i in b) / len(b) for b in p1]
    assert means == sorted(means)


# --------------------------------------------------------------------------- #
# split + collate + multi-manifest union
# --------------------------------------------------------------------------- #
def test_stratified_split_by_speaker(corpus):
    audio, _ = corpus
    spk = pilot_train.build_speaker_index(audio)
    class _FakeDs:
        rows = [{"id": r["id"], "speaker_id": spk[r["id"].split("_")[0]],
                 "duration_s": r["duration_s"]} for r in audio]
    tr, va = pilot_train.stratified_split(_FakeDs(), 0.1)
    assert not set(tr) & set(va)
    assert len(tr) + len(va) == len(audio)
    # every speaker contributes to val (each has >= 8 utterances)
    val_spk = {_FakeDs().rows[i]["speaker_id"] for i in va}
    assert val_spk == set(range(len(spk)))
    counts = {}
    for i in va:
        s = _FakeDs().rows[i]["speaker_id"]
        counts[s] = counts.get(s, 0) + 1
    for s, c in counts.items():
        n = sum(1 for r in _FakeDs().rows
                if r["speaker_id"] == s)
        assert c == (n + 9) // 10                      # stride picks pos%10==0


def test_collate_shapes(full_dataset):
    items = [full_dataset[i] for i in (0, 1, 2, 3)]
    batch = pilot_train.collate_pilot(items)
    x, y = batch["x"], batch["y"]
    b = len(items)
    tx = max(len(it["phoneme_ids"]) for it in items)
    ty = max(it["T_mel"] for it in items)
    assert x["phoneme_ids"].shape == (b, tx)
    assert x["mel"].shape == (b, ty, 80)
    assert y.shape == x["mel"].shape
    assert x["phoneme_lengths"].tolist() == [
        len(it["phoneme_ids"]) for it in items]
    assert x["mel_lengths"].tolist() == [it["T_mel"] for it in items]
    assert x["phoneme_ids"][0, tx:].eq(0).all() or \
        x["phoneme_lengths"][0] == tx
    assert x["speaker_ids"].tolist() == [it["speaker_id"] for it in items]


def test_multi_manifest_union(tmp_path, corpus):
    """Second manifest accepted even though ljspeech-smoke-400 does not exist
    yet: synthesize a temp slice pointing at copies of real wavs."""
    audio, phone = corpus
    mdir = tmp_path / "data" / "manifests"
    proc = tmp_path / "data" / "processed" / "fake-lj-slice"
    mdir.mkdir(parents=True)
    proc.mkdir(parents=True)

    src_rows = sorted(corpus[0], key=lambda r: r["duration_s"])[:2]
    fake_audio, fake_phone = [], []
    by_id = {r["id"]: r for r in phone}
    for j, row in enumerate(src_rows):
        wav_name = f"LJ00{j+1}-0001.wav"
        shutil.copyfile(REPO / row["audio_path"], proc / wav_name)
        rid = f"LJ00{j+1}_0001"  # underscore prefix = speaker per the index rule
        # absolute path: one dataset spans both roots (repo data + temp slice)
        fake_audio.append({"id": rid,
                           "audio_path": str(proc / wav_name),
                           "transcript": "x",
                           "duration_s": row["duration_s"],
                           "sample_rate": 22050})
        src_ph = by_id[row["id"]]
        fake_phone.append({"id": rid,
                           "transcript_norm": "x",
                           "phoneme_string": src_ph["phoneme_string"],
                           "n_phones": src_ph["n_phones"]})
    (mdir / "fake-lj-slice.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fake_audio) + "\n", encoding="utf-8")
    (mdir / "fake-lj-slice.phonemes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fake_phone) + "\n", encoding="utf-8")

    a1, p1 = pilot_train.load_manifest_pair("librittsr-smoke-10spk",
                                            REPO / "data" / "manifests")
    a2, p2 = pilot_train.load_manifest_pair("fake-lj-slice", mdir)
    all_a, all_p = a1 + a2, p1 + p2
    spk = pilot_train.build_speaker_index(all_a)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in all_p])
    ds = PilotDataset(all_a, all_p, spk, vocab=vocab, repo_root=REPO)
    assert len(ds) == len(a1) + 2
    lj_ids = [i for i, r in enumerate(ds.rows) if r["id"].startswith("LJ")]
    assert len(lj_ids) == 2
    # 'LJ001'/'LJ002' sort after every LibriTTS numeric prefix -> ids 10, 11
    assert sorted(p for p in spk if p.startswith("LJ")) == ["LJ001", "LJ002"]
    assert {ds.speaker_id(i) for i in lj_ids} == {10, 11}
    assert max(spk.values()) == 11
    item = ds[lj_ids[0]]
    assert item["mel"].shape[0] == 80 and item["T_mel"] > 0
    with pytest.raises(FileNotFoundError, match="ljspeech-smoke-400"):
        pilot_train.load_manifest_pair("ljspeech-smoke-400",
                                       REPO / "data" / "manifests")


# --------------------------------------------------------------------------- #
# one full training step chain on real data
# --------------------------------------------------------------------------- #
def test_training_step_chain_finite_and_reduces_loss(corpus):
    audio, phone = corpus
    rows = sorted(audio, key=lambda r: r["duration_s"])[:16]  # short utts
    by_id = {r["id"]: r for r in phone}
    ph = [by_id[r["id"]] for r in rows]
    spk = pilot_train.build_speaker_index(rows)
    vocab = PhonemeVocab.build([r["phoneme_string"] for r in ph])
    ds = PilotDataset(rows, ph, spk, vocab=vocab, repo_root=REPO)

    info = set_determinism(1234, num_threads=4)
    assert info["num_threads"] == 4
    try:
        cfg = CaedralNARConfig.from_json(TINY_CONFIG)
        cfg.use_flow = True
        cfg.vocab_size = max(cfg.vocab_size, vocab.size)
        model = pilot_train.CaedralNAR(cfg)
        module = pilot_train.PilotTrainModule(model, cfg.sample_rate,
                                              cfg.hop_length)
        opt = torch.optim.AdamW(module.parameters(), lr=1e-4,
                                betas=(0.9, 0.98), weight_decay=1e-6)
        sched = pilot_train.build_scheduler(opt, {"warmup_steps": 400,
                                                  "max_steps": 15000,
                                                  "min_lr_ratio": 0.1})
        batch = pilot_train.collate_pilot([ds[i] for i in range(4)])
        losses, flows, durs = [], [], []
        module.train()
        for _ in range(6):
            out = module(batch["x"])
            loss = out["loss"]
            assert math.isfinite(float(loss))
            assert math.isfinite(float(out["flow_loss"]))
            assert math.isfinite(float(out["duration_loss"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            assert math.isfinite(float(gnorm))
            opt.step()
            sched.step()
            losses.append(float(loss))
            flows.append(float(out["flow_loss"]))
            durs.append(float(out["duration_loss"]))
        diag = module.last_diag
        assert math.isfinite(diag["phones_per_sec"]) and \
            diag["phones_per_sec"] > 0
        # deterministic seeded run: measured once, holds forever
        assert sum(flows[-2:]) < sum(flows[:2])       # flow loss reduced
        print(f"\n[chain] flow {flows[0]:.3f} -> {flows[-1]:.3f} | "
              f"dur {durs[0]:.3f} -> {durs[-1]:.3f} | "
              f"pps {diag['phones_per_sec']:.1f}", file=sys.stderr)
    finally:
        torch.set_num_threads(max(1, torch.get_num_threads()))
