"""Supervised-duration regressor data pipeline (experiment CV-G1-E0013).

Extends the E0007 pilot pipeline (mel_dataset.py) with duration targets:

    manifest row (+phoneme row, +durations row) ->
        {phoneme_ids, duration_targets [n], mel [80, T], T_mel, speaker_id}

Sources per utterance:
* audio + speaker prefix  : ``<name>.jsonl`` (clean100 rows; teacher rows carry
  a ``teacher_voice`` field and ids prefixed ``tNN_``);
* phonemes                : ``<name>.phonemes.jsonl`` -- OR, for teacher rows
  (which share the clean100 text), resolved through their ``source_id`` into
  the clean100 phoneme map;
* durations               : ALIGN-01 ``<name>.durations.jsonl`` rows
  ``{id, durations:[int], total_frames}`` on the SAME char-level id sequence
  produced by ``PhonemeVocab.encode`` (the CTC aligner emits one duration per
  vocab token). Rows without a durations entry are dropped (reason recorded)
  unless ``synthetic_uniform_durations=True`` -- the pre-ALIGN smoke mode that
  splits decoded T_mel uniformly across phonemes (first ``T % n`` phonemes get
  the +1), guaranteeing sum(durations) == T_mel exactly.

Speaker slots: prefixes sorted lexicographically (numeric LibriTTS voices
first, then teacher slots t00..tNN, then any tvNN fallbacks). The model's
speaker table is sized max(40, corpus) per the E0013 sizing directive.

Batching reuses mel_dataset.LengthBucketBatchSampler with the E0013 caps
(<=16 utterances AND <=2000 estimated frames per batch); estimates use exact
aligner totals when available.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .mel_dataset import MEL_FLOOR, FRAME_EST_SLACK, PhonemeVocab

_TNN = re.compile(r"^t\d+$")


# --------------------------------------------------------------------------- #
# speaker slots
# --------------------------------------------------------------------------- #
def build_speaker_index(rows: list[dict]) -> dict[str, int]:
    """Deterministic prefix -> slot map over real + teacher rows.

    Teacher manifests encode the voice slot directly in the id (t00_...) which
    passes through unchanged. For future manifests where a teacher row carries
    ``teacher_voice`` but a non-tNN-shaped id prefix, deterministic tvNN slots
    are assigned by sorted unique voice name instead.
    """
    raw: list[str] = []
    voices: set[str] = set()
    for r in rows:
        p = str(r["id"]).split("_")[0]
        if _TNN.match(p):
            raw.append(p)
        elif "teacher_voice" in r:
            voices.add(str(r["teacher_voice"]))
        else:
            raw.append(p)
    tv_slots = {v: f"tv{i:02d}" for i, v in enumerate(sorted(voices))}
    prefixes = sorted(set(raw) | set(tv_slots.values()))
    return {p: i for i, p in enumerate(prefixes)}


def stratified_split(rows: list[dict], val_frac: float
                     ) -> tuple[list[int], list[int]]:
    """Deterministic per-speaker stride split (same rule as E0007 pilot):
    rows ordered by id inside each speaker, every round(1/val_frac)-th to val."""
    by_spk: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_spk.setdefault(int(row["speaker_id"]), []).append(i)
    k = max(1, round(1.0 / float(val_frac))) if val_frac > 0 else 0
    train_idx: list[int] = []
    val_idx: list[int] = []
    for spk in sorted(by_spk):
        idxs = sorted(by_spk[spk],
                      key=lambda i: (rows[i]["id"], i))
        for pos, i in enumerate(idxs):
            (val_idx if (k and pos % k == 0) else train_idx).append(i)
    assert not (set(train_idx) & set(val_idx))
    return sorted(train_idx), sorted(val_idx)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8")
            .splitlines() if l.strip()]


def uniform_durations(total_frames: int, n_tokens: int) -> list[int]:
    """Split total_frames across n_tokens; first (total % n) get the +1."""
    base, rem = divmod(int(total_frames), int(n_tokens))
    return [base + (1 if k < rem else 0) for k in range(n_tokens)]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


class RegressorDataset(Dataset):
    """Manifest+phonemes+durations -> tensors for CaedralRegressor training."""

    def __init__(
        self,
        manifest_rows: list[dict],
        phone_rows: list[dict],
        durations_rows: list[dict] | None,
        speaker_index: dict[str, int],
        sample_rate: int = 24000,
        n_mels: int = 80,
        hop: int = 256,
        n_fft: int = 1024,
        vocab: PhonemeVocab | None = None,
        repo_root: str | Path | None = None,
        synthetic_uniform_durations: bool = False,
        disk_cache=None,               # E0018: optional MelDiskCache (opt-in)
    ):
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.disk_cache = disk_cache
        self.synthetic_uniform = bool(synthetic_uniform_durations)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.vocab = vocab if vocab is not None else PhonemeVocab.build(
            [p["phoneme_string"] for p in phone_rows])

        ph_by_id: dict[str, dict] = {}
        ph_by_source: dict[str, dict] = {}
        for r in phone_rows:
            ph_by_id[r["id"]] = r
            src = r.get("source_id")
            if src is not None:
                ph_by_source.setdefault(src, r)

        dur_by_id: dict[str, dict] = {}
        for r in durations_rows or []:
            dur_by_id[r["id"]] = r

        self.speaker_missing: set[str] = set()
        self.skipped: list[tuple[str, str]] = []  # (id, reason)

        # tvNN fallback table mirrors build_speaker_index's second pass so a
        # dataset always agrees with the index built from the same rows
        self._tv_slot_by_voice: dict[str, str] = {}
        deferred = [r for r in manifest_rows
                    if not _TNN.match(str(r["id"]).split("_")[0])
                    and "teacher_voice" in r]
        for i, v in enumerate(sorted({str(r["teacher_voice"]) for r in deferred})):
            self._tv_slot_by_voice[v] = f"tv{i:02d}"

        kept_rows: list[dict] = []
        for r in manifest_rows:
            rid = r["id"]
            ph = ph_by_id.get(rid)
            if ph is None and "source_id" in r:
                ph = ph_by_source.get(r["source_id"])
            if ph is None:
                self.skipped.append((rid, "no phoneme row"))
                continue
            path = _resolve(self.repo_root, r["audio_path"])
            if not path.is_file():
                self.skipped.append((rid, f"missing audio: {path.name}"))
                continue
            try:
                ids = self.vocab.encode(ph["phoneme_string"])
            except KeyError as e:
                self.skipped.append((rid, f"vocab miss: {e}"))
                continue
            if not ids:
                self.skipped.append((rid, "empty phoneme string"))
                continue
            prefix = rid.split("_")[0]
            if not _TNN.match(prefix) and "teacher_voice" in r:
                prefix = self._tv_slot_by_voice[str(r["teacher_voice"])]
            spk_id = speaker_index.get(prefix)
            if spk_id is None:
                self.speaker_missing.add(prefix)
                self.skipped.append(
                    (rid, f"unknown speaker prefix {prefix!r}"))
                continue

            durs: list[int] | None = None
            total_frames: int | None = None
            if not self.synthetic_uniform:
                dr = dur_by_id.get(rid)
                if dr is None:
                    self.skipped.append((rid, "no durations row"))
                    continue
                durs = [int(d) for d in dr["durations"]]
                if len(durs) != len(ids):
                    self.skipped.append(
                        (rid, f"duration length {len(durs)} != "
                              f"phones {len(ids)}"))
                    continue
                total_frames = int(dr.get("total_frames", sum(durs)))

            est = (
                int(r["duration_s"] * self.sample_rate / self.hop) + 1
                + FRAME_EST_SLACK
                if total_frames is None else int(total_frames) + FRAME_EST_SLACK
            )
            if est < len(ids):
                self.skipped.append(
                    (rid, f"infeasible: est_frames {est} < phones {len(ids)}"))
                continue
            kept_rows.append({
                "id": rid,
                "audio_path": r["audio_path"],
                "duration_s": float(r["duration_s"]),
                "phoneme_string": ph["phoneme_string"],
                "ids": ids,
                "speaker_prefix": prefix,
                "speaker_id": int(spk_id),
                "durations": durs,             # None => lazy synthetic split
                "total_frames": total_frames,
                "est_frames": est,
            })
        self.rows = kept_rows
        self._mel_cache: dict[int, torch.Tensor] = {}

    #: rows flagged unknown-speaker during build
    speaker_missing: set[str]

    def __len__(self) -> int:
        return len(self.rows)

    def duration_s(self, index: int) -> float:
        return self.rows[index]["duration_s"]

    def est_frames(self, index: int) -> int:
        """Conservative mel-frame estimate for the batch sampler (exact aligner
        totals when available, duration-based otherwise)."""
        return self.rows[index]["est_frames"]

    def speaker_id(self, index: int) -> int:
        return self.rows[index]["speaker_id"]

    def _load_mel(self, index: int) -> torch.Tensor:
        """wav -> target rate -> log-Mel (magnitude, natural log, floor 1e-5),
        shape (n_mels, T); memoized (same recipe as mel_dataset.PilotDataset)."""
        cached = self._mel_cache.get(index)
        if cached is not None:
            return cached
        import torchaudio

        row = self.rows[index]
        path = _resolve(self.repo_root, row["audio_path"])
        if self.disk_cache is not None:            # E0018: opt-in disk cache
            arr = self.disk_cache.get_or_compute(path, self.sample_rate)
            mel = torch.from_numpy(arr)            # zero-copy view; [80, T]
            self._mel_cache[index] = mel
            return mel
        wav, sr = torchaudio.load(str(path))       # (C, T) float32
        wav = wav.mean(dim=0)                      # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        mel_fn = getattr(self, "_mel_fn", None)
        if mel_fn is None:
            mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop, n_mels=self.n_mels, power=1.0,
            )
            self._mel_fn = mel_fn
        mel = mel_fn(wav)                          # (n_mels, T) magnitude
        mel = torch.log(mel.clamp_min(MEL_FLOOR))
        self._mel_cache[index] = mel
        return mel

    def __getitem__(self, index: int) -> dict:
        mel = self._load_mel(index)                # (n_mels, T)
        row = self.rows[index]
        t_mel = int(mel.shape[1])
        n = len(row["ids"])
        if row["durations"] is not None:
            durs = list(row["durations"])
        else:
            # pre-ALIGN smoke mode: uniform split of the decoded frame count
            durs = uniform_durations(t_mel, n)
        if t_mel < n or sum(durs) < n:
            # pathological short-audio guard: pad mel by repeating the last
            # frame so every phoneme keeps >= 1 frame (should not occur on
            # real corpora; keeps the run alive if it does)
            need = max(n - t_mel, 1)
            mel = torch.cat([mel, mel[:, -1:].expand(-1, need)], dim=1)
            t_mel = int(mel.shape[1])
            durs = uniform_durations(t_mel, n)
        return {
            "index": int(index),
            "id": row["id"],
            "phoneme_ids": list(row["ids"]),
            "duration_targets": durs,
            "mel": mel,
            "T_mel": t_mel,
            "speaker_id": row["speaker_id"],
        }


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def collate_regressor(batch: list[dict]) -> dict:
    """Pad right with vocab pad id 0 / duration 0 / mel 0.0; lengths mark valid
    extents. Returns {'x': ..., 'y': mel} for the E0003 Trainer unpacker (y is
    carried for interface compatibility; losses consume x only)."""
    tx = max(len(it["phoneme_ids"]) for it in batch)
    ty = max(it["T_mel"] for it in batch)
    b = len(batch)
    n_mels = batch[0]["mel"].shape[0]
    ids = torch.zeros(b, tx, dtype=torch.long)
    durs = torch.zeros(b, tx, dtype=torch.long)
    mel = torch.zeros(b, ty, n_mels, dtype=torch.float32)
    xlen = torch.zeros(b, dtype=torch.long)
    ylen = torch.zeros(b, dtype=torch.long)
    spk = torch.zeros(b, dtype=torch.long)
    for j, it in enumerate(batch):
        n_x = len(it["phoneme_ids"])
        n_y = it["T_mel"]
        ids[j, :n_x] = torch.tensor(it["phoneme_ids"], dtype=torch.long)
        durs[j, :n_x] = torch.tensor(it["duration_targets"], dtype=torch.long)
        mel[j, :n_y, :] = it["mel"].t()            # [80, T] -> [T, 80]
        xlen[j], ylen[j], spk[j] = n_x, n_y, it["speaker_id"]
    return {
        "x": {
            "mel": mel,
            "phoneme_ids": ids,
            "duration_targets": durs,
            "speaker_ids": spk,
            "mel_lengths": ylen,
            "phoneme_lengths": xlen,
        },
        "y": mel,
    }
