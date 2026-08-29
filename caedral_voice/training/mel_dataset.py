"""Pilot training data pipeline (Phase J, experiment CV-G1-E0007).

Turns processed-wav manifests (.jsonl) + phoneme manifests (*.phonemes.jsonl)
into tensors for CAEDRAL-NAR flow training (MAS alignment):

    wav (repo-relative path) -> resample to target sr -> log-Mel [80, T]
    phoneme_string (space-separated IPA tokens) -> char-vocab ids
    speaker prefix ("103_1241_..." -> "103") -> speaker-table id

Design decisions (documented for the pilot conclusion):

* **Resample-at-load**: processed slices live at their pipeline target rate
  (LibriTTS-R slice: 22050 Hz); the acoustic front (tiny config / istft_head /
  planned Vocos vocoder) is defined at 24 kHz / hop 256. The dataset resamples
  once per item and memoizes the log-Mel in RAM (~44 MB for the 1.66 h slice),
  so epoch 2+ pays zero audio-decode cost. num_workers=0 keeps the cache in
  one process (also the Windows-detached-friendly choice).
* **Char vocabulary** per spec: specials ``<pad>=0``, ``|=1``, ``‖=2`` then
  every remaining character of the union of ``phoneme_string`` values, sorted,
  ids assigned in order. Space-separated tokens longer than one char (the five
  length-marked vowels iː uː ɑː ɔː ɝː in the LibriTTS-R slice) encode as their
  char sequence. Whitespace is a separator, never a symbol.
* **Conservative frame estimates** for the batch sampler: the sampler runs
  before any audio is decoded, so it bounds batches by
  ``floor(duration_s * sr / hop) + SLACK``; the slack covers resampler-length
  rounding and center-padding, keeping the "total mel frames per batch"
  invariant strict on *actual* mels (unit-tested).
* **MAS feasibility preflight**: MAS (mas.py) raises if an item has fewer mel
  frames than phonemes. The dataset drops such rows at build time (with the
  reason recorded) instead of letting one pathological row crash a multi-hour
  run hours in.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from torch.utils.data import Sampler

MEL_FLOOR = 1e-5
FRAME_EST_SLACK = 4  # frames added per utterance for sampler bounds


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
class PhonemeVocab:
    """Deterministic char vocab: specials + sorted union of string chars.

    ``<pad>``=0 (batch padding), ``|``=1 (word boundary, occurs inside
    phoneme strings), ``‖``=2 (utterance boundary, reserved -- G1-SMOKE rows
    carry no explicit silence tokens). All other symbols follow in sorted
    order starting at 3.
    """

    VERSION = 1
    SPECIALS = {"<pad>": 0, "|": 1, "‖": 2}  # ‖ == ‖

    def __init__(self, symbols: list[str]):
        self.symbols = list(symbols)
        self.id_to_sym = {i: s for i, s in enumerate(self.symbols)}
        self.sym_to_id = {s: i for i, s in enumerate(self.symbols)}
        for sym, sid in self.SPECIALS.items():
            got = self.sym_to_id.get(sym)
            if got != sid:
                raise ValueError(
                    f"vocab malformed: special {sym!r} must have id "
                    f"{sid}, found {got}")

    @classmethod
    def build(cls, phoneme_strings: list[str]) -> "PhonemeVocab":
        """Union of chars across strings (+specials), sorted deterministically."""
        chars: set[str] = set()
        for s in phoneme_strings:
            chars.update(s)
        chars.discard(" ")          # whitespace is a separator, not a symbol
        ordered = cls.SPECIALS_ORDER()
        ordered += sorted(c for c in chars if c not in cls.SPECIALS)
        return cls(ordered)

    @classmethod
    def SPECIALS_ORDER(cls) -> list[str]:
        pairs = sorted(cls.SPECIALS.items(), key=lambda kv: kv[1])
        return [sym for sym, _ in pairs]

    def encode(self, phoneme_string: str) -> list[int]:
        """Space-separated tokens -> ids. Single-char tokens (and the word
        separator ``|``) map directly; longer tokens (iː, t͡ʃ, ...)
        fall back to per-char ids (every char is guaranteed present when the
        vocab was built from the same corpus)."""
        out: list[int] = []
        for tok in phoneme_string.split():
            if tok in self.sym_to_id:
                out.append(self.sym_to_id[tok])
                continue
            for ch in tok:
                sid = self.sym_to_id.get(ch)
                if sid is None:
                    raise KeyError(
                        f"character {ch!r} (token {tok!r}) not in vocab")
                out.append(sid)
        return out

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Ids -> space-joined symbols (diagnostic use; multi-char tokens are
        not re-joined)."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(self.id_to_sym[int(i)] for i in ids)

    @property
    def size(self) -> int:
        return len(self.symbols)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"version": self.VERSION,
                        "specials": dict(sorted(self.SPECIALS.items(),
                                                key=lambda kv: kv[1])),
                        "symbols": self.symbols},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PhonemeVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("version", -1)) != cls.VERSION:
            raise ValueError(f"unsupported vocab version {data.get('version')}")
        return cls(list(data["symbols"]))

    def __eq__(self, other) -> bool:
        return isinstance(other, PhonemeVocab) and \
            self.symbols == other.symbols

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


class PilotDataset(Dataset):
    """Manifest rows -> {phoneme_ids, mel [n_mels, T], T_mel, speaker_id}.

    Args mirror the pilot spec; ``vocab``/``repo_root`` are extension points
    so train and val datasets share one vocab and one path root (tests point
    ``repo_root`` at a temp tree for the multi-manifest case).
    """

    def __init__(
        self,
        manifest_rows: list[dict],
        phoneme_rows: list[dict],
        speaker_index: dict[str, int],
        sample_rate: int = 24000,
        n_mels: int = 80,
        hop: int = 256,
        n_fft: int = 1024,
        vocab: PhonemeVocab | None = None,
        repo_root: str | Path | None = None,
        disk_cache=None,               # E0018: optional MelDiskCache (opt-in)
    ):
        import torchaudio

        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.disk_cache = disk_cache
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.vocab = vocab if vocab is not None else PhonemeVocab.build(
            [p["phoneme_string"] for p in phoneme_rows])

        ph_by_id: dict[str, dict] = {}
        for r in phoneme_rows:
            ph_by_id[r["id"]] = r

        self.speaker_missing: set[str] = set()
        self.skipped: list[tuple[str, str]] = []  # (id, reason)
        kept_rows: list[dict] = []
        for r in manifest_rows:
            rid = r["id"]
            ph = ph_by_id.get(rid)
            if ph is None:
                self.skipped.append((rid, "no phoneme row"))
                continue
            path = _resolve(self.repo_root, r["audio_path"])
            if not path.is_file():
                self.skipped.append((rid, f"missing audio: {path.name}"))
                continue
            ids = self.vocab.encode(ph["phoneme_string"])
            if not ids:
                self.skipped.append((rid, "empty phoneme string"))
                continue
            prefix = rid.split("_")[0]
            if prefix not in speaker_index:
                self.speaker_missing.add(prefix)
                self.skipped.append((rid, f"unknown speaker prefix {prefix!r}"))
                continue
            est = int(r["duration_s"] * self.sample_rate / self.hop) + 1
            if est < len(ids):
                self.skipped.append(
                    (rid, f"MAS-infeasible: est_frames {est} < phones {len(ids)}"))
                continue
            kept_rows.append({
                "id": rid,
                "audio_path": r["audio_path"],
                "duration_s": float(r["duration_s"]),
                "phoneme_string": ph["phoneme_string"],
                "ids": ids,
                "speaker_prefix": prefix,
                "speaker_id": int(speaker_index[prefix]),
            })
        self.rows = kept_rows
        self._mel_cache: dict[int, torch.Tensor] = {}

    #: rows flagged unknown-speaker during build (populated before skips list)
    speaker_missing: set[str]

    def __len__(self) -> int:
        return len(self.rows)

    def duration_s(self, index: int) -> float:
        return self.rows[index]["duration_s"]

    def est_frames(self, index: int) -> int:
        """Conservative mel-frame estimate used for batch-size bounding."""
        return int(self.rows[index]["duration_s"] * self.sample_rate
                   / self.hop) + FRAME_EST_SLACK

    def speaker_id(self, index: int) -> int:
        return self.rows[index]["speaker_id"]

    def _load_mel(self, index: int) -> torch.Tensor:
        """wav -> 24 kHz -> log-Mel (power=1, natural log, floor 1e-5),
        shape (n_mels, T); memoized."""
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
        mel = mel_fn(wav)                          # (n_mels, T) power=1
        mel = torch.log(mel.clamp_min(MEL_FLOOR))
        self._mel_cache[index] = mel
        return mel

    def __getitem__(self, index: int) -> dict:
        mel = self._load_mel(index)                # (n_mels, T)
        row = self.rows[index]
        return {
            "index": int(index),
            "phoneme_ids": list(row["ids"]),
            "mel": mel,
            "T_mel": int(mel.shape[1]),
            "speaker_id": row["speaker_id"],
        }


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def _epoch_seed(base_seed: int, epoch: int) -> int:
    digest = hashlib.sha256(
        f"mel-bucket:{int(base_seed)}:{int(epoch)}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle -> length-bucket -> sort-within-bucket -> capped batches.

    Per epoch (data order pinned to ``(base_seed, epoch)``, matching the
    trainer's per-epoch generator discipline):

    1. global shuffle of the active indices with a dedicated PRNG;
    2. cut into consecutive buckets of ``bucket_size``;
    3. sort each bucket by estimated frame length;
    4. greedily pack batches of at most ``max_utts`` utterances AND at most
       ``max_frames`` total estimated frames (slack-covered, so the bound
       holds on decoded mels too);
    5. shuffle the resulting batch order.

    With ``shuffle=False`` the plan is a fixed length-sorted packing (used for
    val so val loss stays comparable across evals). ``set_epoch`` is driven by
    the training script's hooks so resume reproduces order exactly (see
    pilot_train.py); a fresh sampler starts at epoch 0.
    """

    def __init__(
        self,
        indices: list[int],
        est_frames_fn,                     # index -> estimated mel frames
        max_frames: int = 1600,
        max_utts: int = 8,
        bucket_size: int = 32,
        shuffle: bool = True,
        base_seed: int = 0,
    ):
        self.indices = list(indices)
        self.est_frames_fn = est_frames_fn
        self.max_frames = int(max_frames)
        self.max_utts = int(max_utts)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self._last_plan: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._last_plan = None

    def _plan(self) -> list[list[int]]:
        if self._last_plan is not None:
            return self._last_plan
        rng = random.Random(_epoch_seed(self.base_seed, self.epoch))
        order = list(self.indices)
        if self.shuffle:
            rng.shuffle(order)
        else:
            order.sort(key=lambda i: (self.est_frames_fn(i), i))
        batches: list[list[int]] = []
        for b0 in range(0, len(order), self.bucket_size):
            bucket = sorted(order[b0:b0 + self.bucket_size],
                            key=lambda i: (self.est_frames_fn(i), i))
            cur: list[int] = []
            cur_f = 0
            for i in bucket:
                f = self.est_frames_fn(i)
                if cur and (
                    len(cur) >= self.max_utts
                    or cur_f + f > self.max_frames
                    # single oversized utterance still forms its own batch
                ):
                    batches.append(cur)
                    cur, cur_f = [], 0
                cur.append(i)
                cur_f += f
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        self._last_plan = batches
        return batches

    def __iter__(self):
        yield from self._plan()

    def __len__(self) -> int:
        return len(self._plan())
