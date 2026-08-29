"""Monotonic Alignment Search (Phase F+, CV-G1-E0006).

Hard-alignment search from Glow-TTS (Kim et al. 2020, section 3.2): given a
per-candidate score matrix ``loglik[b, i, j]`` (log-likelihood of mel frame
``j`` under phoneme ``i``'s distribution), find the monotonic path that

    covers every frame exactly once,
    assigns every phoneme at least one frame,
    advances the phoneme index by 0 or 1 per frame step,

maximizing the summed log-likelihood. The optimal durations are the run
lengths of the winning path. This removes any external aligner from pilot
training (CV-G1-E0005 decision).

Implementation notes
--------------------
The textbook recursion (equivalent to the b_top/b_bot diagonal recursion of
the reference Glow-TTS implementation) is

    V[0][j] = sum_{t<=j} C[0][t]
    V[i][j] = C[i][j] + max(V[i-1][j-1], V[i][j-1])

Unrolling the in-row dependency gives a prefix-sum form that vectorizes each
phoneme row over batch/columns as one cumulative sum + one running-max scan:

    V[i][j] = Pc_i[j-i+1] + max_{k in [i..j]} (V[i-1][k-1] - Pc_i[k-i])

with ``Pc_i[m] = sum(C[i, i : i+m])``, O(T_x * T_y) total in two numpy vector
ops per row plus an O(T_x) backtrack. Correctness-first discipline: the
literal nested-loop DP is kept as ``maximum_path_reference`` and tests assert
both return identical optima.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


_NEG = -1e18  # finite sentinel: keeps cumsum/subtraction arithmetic NaN-free


def _item_scores(s: np.ndarray, x_len: int, y_len: int) -> np.ndarray:
    """Copy one score matrix restricted to its valid rectangle.

    Padding uses a huge finite negative (not -inf): the fast DP subtracts
    prefix sums, and +-inf mixing would poison the running-max scan with
    NaN. A path touching >=1 padded cell scores <= -1e18 while every
    feasible all-valid path exists (y_len >= x_len validated upstream), so
    padding can never win.
    """
    out = s.astype(np.float64, copy=True)
    out[np.isnan(out)] = _NEG
    out[:, y_len:] = _NEG
    out[x_len:, :] = _NEG
    return out


def maximum_path(
    loglik: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    y_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast MAS. Returns (durations (B, T_x) int64, best_scores (B,) float64).

    ``loglik``: (B, T_x, T_y) values to MAXIMIZE along the monotonic path.
    Lengths default to the full extent of the matrix. Raises ValueError if
    any item has fewer mel frames than phonemes.
    """
    s_full = _to_numpy(loglik)
    squeeze = s_full.ndim == 2  # unbatched convenience
    if squeeze:
        s_full = s_full[None]
    b, tx, ty = s_full.shape
    if x_lengths is None:
        xl = np.full(b, tx, dtype=np.int64)
    else:
        xl = _to_numpy(x_lengths).astype(np.int64).reshape(-1)
    if y_lengths is None:
        yl = np.full(b, ty, dtype=np.int64)
    else:
        yl = _to_numpy(y_lengths).astype(np.int64).reshape(-1)
    if (yl < xl).any():
        raise ValueError(
            "MAS needs at least one mel frame per phoneme "
            f"(got y_lengths={yl.tolist()} vs x_lengths={xl.tolist()})"
        )
    durations = np.zeros((b, tx), dtype=np.int64)
    scores = np.zeros(b, dtype=np.float64)
    for item in range(b):
        si = _item_scores(s_full[item], int(xl[item]), int(yl[item]))
        d_i, scores[item] = _dp_single(si, int(xl[item]), int(yl[item]))
        durations[item, : xl[item]] = d_i  # rows beyond x_len stay 0
    if squeeze:
        durations = durations[:1]
        scores = scores[:1]
    return torch.from_numpy(durations.copy()), torch.from_numpy(scores.copy())


def _dp_single(s: np.ndarray, x_len: int, y_len: int):
    """Row-scan DP on one (T_x, T_y) pre-masked score matrix."""
    neg = _NEG
    ty = s.shape[1]

    def forbidden_from(i: int) -> int:
        # Row i may use column j only if enough frames remain for the
        # x_len - 1 - i later phonemes (>=1 frame each):
        # allowed j <= y_len - x_len + i  =>  first forbidden = y_len-x_len+i+1
        return max(y_len - x_len + i + 1, 0)

    kmat = np.zeros((x_len, ty), dtype=np.int64)

    # ---- row 0: forced start at column 0 -------------------------------- #
    v_prev = np.cumsum(s[0])
    v_prev[forbidden_from(0):] = neg
    kmat[0, :] = 0

    for i in range(1, x_len):
        seg = s[i, i:]                       # masked tails carry the sentinel
        lm = ty - i
        pc = np.concatenate((np.zeros(1), np.cumsum(seg)))  # pc[m]=sum(seg[:m])
        kk = np.arange(lm)                   # candidate k = i + kk
        cols = i + kk - 1                    # ABSOLUTE V_prev column (k - 1)
        vv = v_prev[np.clip(cols, 0, ty - 1)]
        cand = vv - pc[:-1]                  # A[kk]
        running = np.maximum.accumulate(cand)         # max over kk' <= kk
        ar = np.arange(lm)
        pick = np.maximum.accumulate(np.where(cand >= running, ar, -1))

        v_row = np.full(ty, neg)
        # column j = i + m - 1 for m in [1..lm]: V = pc[m] + running[m-1]
        v_row[i:] = pc[1:] + running
        k_row = np.zeros(ty, dtype=np.int64)
        k_row[i:] = pick + i
        stop = forbidden_from(i)
        if stop < ty:
            v_row[stop:] = neg
        v_prev = v_row
        kmat[i, :] = k_row

    corner = v_prev[y_len - 1]
    if not np.isfinite(corner) or corner <= -1e17:
        raise ValueError("no feasible monotonic alignment (degenerate mask?)")

    # ---- backtrack: read durations off the chosen run-start points ------ #
    durs = np.zeros(x_len, dtype=np.int64)
    j = y_len - 1
    for i in range(x_len - 1, 0, -1):
        k = int(kmat[i, j])
        durs[i] = j - k + 1
        j = k - 1
    durs[0] = j + 1
    return durs, float(corner)


def maximum_path_reference(
    loglik: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    y_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Textbook nested-loop DP + backtracking (validation path, small inputs).

    Same contract as :func:`maximum_path`; exists to prove the fast row-scan
    implementation returns the exact same optimum.
    """
    s_full = _to_numpy(loglik)
    squeeze = s_full.ndim == 2  # unbatched convenience
    if squeeze:
        s_full = s_full[None]
    b, tx, ty = s_full.shape
    xl = (
        np.full(b, tx, dtype=np.int64) if x_lengths is None
        else _to_numpy(x_lengths).astype(np.int64).reshape(-1)
    )
    yl = (
        np.full(b, ty, dtype=np.int64) if y_lengths is None
        else _to_numpy(y_lengths).astype(np.int64).reshape(-1)
    )
    if (yl < xl).any():
        raise ValueError("MAS needs at least one mel frame per phoneme")
    durations = np.zeros((b, tx), dtype=np.int64)
    scores = np.zeros(b, dtype=np.float64)
    neg = -np.inf
    for item in range(b):
        s = _item_scores(s_full[item], int(xl[item]), int(yl[item]))
        X, Y = int(xl[item]), int(yl[item])
        v = np.full((X, Y), neg)
        take_left = np.zeros((X, Y), dtype=bool)  # True: V[i][j] came from left
        v[0, 0] = s[0, 0]
        for j in range(1, Y):
            v[0, j] = v[0, j - 1] + s[0, j]
        for i in range(1, X):
            for j in range(i, Y):
                left = v[i, j - 1]
                diag = v[i - 1, j - 1]
                if diag >= left:
                    v[i, j] = s[i, j] + diag
                else:
                    v[i, j] = s[i, j] + left
                    take_left[i, j] = True
        if not np.isfinite(v[X - 1, Y - 1]):
            raise ValueError("no feasible monotonic alignment")
        scores[item] = v[X - 1, Y - 1]
        i, j = X - 1, Y - 1
        while i > 0:
            start = j
            while start > 0 and take_left[i, start]:
                start -= 1
            durations[item, i] = j - start + 1
            j = start - 1
            i -= 1
        durations[item, 0] = j + 1
    if squeeze:
        durations = durations[:1]
        scores = scores[:1]
    return torch.from_numpy(durations.copy()), torch.from_numpy(scores.copy())
