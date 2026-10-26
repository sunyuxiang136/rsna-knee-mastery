"""Per-finding attention pooling (NumPy re-implementation for teaching/tests).

The production model does the same thing in PyTorch with an ``einsum`` over a
``[windows, findings]`` attention matrix (see ``RaptorClassifier`` in
``reference/raptor224_run.py``).
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def class_attention(features: np.ndarray, att_weights: np.ndarray) -> np.ndarray:
    """Attention over windows, one column per finding.

    Args:
        features: ``[K, F]`` window features.
        att_weights: ``[F, N]`` attention projection, one column per finding.

    Returns:
        ``[K, N]`` weights; every column sums to 1.
    """
    return softmax(features @ att_weights, axis=0)


def class_attention_probs(
    features: np.ndarray,
    att_weights: np.ndarray,
    cls_weights: np.ndarray,
    cls_bias: np.ndarray,
) -> np.ndarray:
    """Per-finding probabilities from per-finding attention pooling.

    ``cls_weights`` is ``[N, F]`` and ``cls_bias`` is ``[N]``.
    """
    att = class_attention(features, att_weights)
    pooled = np.einsum("kn,kf->nf", att, features)
    logits = (pooled * cls_weights).sum(-1) + cls_bias
    return 1.0 / (1.0 + np.exp(-logits))
