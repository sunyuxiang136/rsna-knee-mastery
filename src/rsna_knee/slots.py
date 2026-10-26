"""Fixed-slot slice sampling."""

from __future__ import annotations

import numpy as np

from .constants import SLICE_RANGE, SLOTS


def total_slices() -> int:
    """Total number of slices across all slots (96)."""
    return sum(n for _, _, n in SLOTS)


def pick_indices(
    n_slices: int,
    k: int,
    lo_frac: float = SLICE_RANGE[0],
    hi_frac: float = SLICE_RANGE[1],
) -> list[int]:
    """Pick ``k`` evenly spaced slice indices from a stack of ``n_slices``.

    Sampling covers ``lo_frac``-``hi_frac`` of the stack rather than only the
    middle, because the outer slices hold the collateral ligaments and the
    lateral meniscus. Indices may repeat when ``k > n_slices``.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if k == 1:
        return [0]
    lo = int(n_slices * lo_frac)
    hi = max(int(n_slices * hi_frac) - 1, lo)
    idx = np.linspace(lo, hi, k).round().astype(int)
    return [int(min(i, n_slices - 1)) for i in idx]
