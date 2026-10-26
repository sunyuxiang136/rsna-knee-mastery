"""Percentile-rank fusion with failure handling.

Mirrors "Stage 3" of ``reference/notebook_pipeline.py``:

* each model is ranked **only over its own non-failed studies**;
* the Raptor half is ``(1 - a224) * rank(384) + a224 * rank(224)``; a study one
  of the two failed on takes the other's rank;
* the outer blend is ``w * raptor_half + (1 - w) * ours``; a study one side
  failed on takes the other side's rank; if both failed the score is 0.5.
"""

from __future__ import annotations

import numpy as np

from .constants import A224, FAILURE_SENTINEL, W_RAPTOR, W_RAPTOR_PER_FINDING


def pct_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in ``(0, 1]``; ties get their average rank.

    Equivalent to ``pandas.Series.rank(method="average", pct=True)``.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v.copy()
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    avg_rank = start + (counts + 1) / 2.0
    return avg_rank[inverse] / n


def pct_rank_on(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Percentile rank among rows where ``ok`` is true; other rows are NaN.

    Excluding failed rows keeps a sentinel/mean-filled value from shifting the
    ranks of healthy studies.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    out = np.full(len(v), np.nan)
    if ok.any():
        out[ok] = pct_rank(v[ok])
    return out


def failed_rows(raw: np.ndarray) -> np.ndarray:
    """Rows left at the 0.5 sentinel in every column (``[n, findings]`` input)."""
    return (np.asarray(raw) == FAILURE_SENTINEL).all(axis=1)


def fuse_finding(
    raptor384: np.ndarray,
    raptor224: np.ndarray,
    ours: np.ndarray,
    fail_384: np.ndarray,
    fail_224: np.ndarray,
    fail_ours: np.ndarray,
    w_raptor: float = W_RAPTOR,
    a224: float = A224,
) -> np.ndarray:
    """Fuse one finding's scores from three models into a single ranking score."""
    if not 0.0 < a224 < 1.0:
        raise ValueError("a224 must be in (0, 1)")
    if not 0.0 <= w_raptor <= 1.0:
        raise ValueError("w_raptor must be in [0, 1]")

    r384 = pct_rank_on(raptor384, ~np.asarray(fail_384, dtype=bool))
    r224 = pct_rank_on(raptor224, ~np.asarray(fail_224, dtype=bool))
    fail_384 = np.asarray(fail_384, dtype=bool)
    fail_224 = np.asarray(fail_224, dtype=bool)
    fail_ours = np.asarray(fail_ours, dtype=bool)

    raptor_half = np.where(
        fail_384, r224, np.where(fail_224, r384, (1.0 - a224) * r384 + a224 * r224)
    )
    ro = pct_rank_on(ours, ~fail_ours)

    blend = w_raptor * raptor_half + (1.0 - w_raptor) * ro
    rap_fail = fail_384 & fail_224
    blend[rap_fail] = ro[rap_fail]
    blend[fail_ours] = raptor_half[fail_ours]
    blend[rap_fail & fail_ours] = FAILURE_SENTINEL
    return blend


def fuse_all(
    raptor384: np.ndarray,
    raptor224: np.ndarray,
    ours: np.ndarray,
    findings: tuple[str, ...],
    per_finding_weights: dict[str, float] | None = None,
    a224: float = A224,
) -> np.ndarray:
    """Fuse ``[n, findings]`` score matrices from three models."""
    weights = W_RAPTOR_PER_FINDING if per_finding_weights is None else per_finding_weights
    raptor384, raptor224, ours = (np.asarray(a, dtype=np.float64) for a in (raptor384, raptor224, ours))
    if not (raptor384.shape == raptor224.shape == ours.shape):
        raise ValueError("score matrices must have identical shapes")
    if raptor384.shape[1] != len(findings):
        raise ValueError("number of columns must match number of findings")
    f384, f224, fours = failed_rows(raptor384), failed_rows(raptor224), failed_rows(ours)
    out = np.zeros_like(raptor384)
    for j, name in enumerate(findings):
        out[:, j] = fuse_finding(
            raptor384[:, j], raptor224[:, j], ours[:, j], f384, f224, fours,
            w_raptor=weights.get(name, W_RAPTOR), a224=a224,
        )
    return out
