"""rsna_knee: tested, dependency-light building blocks of a public RSNA knee MRI pipeline."""

from .attention import class_attention, class_attention_probs, softmax
from .constants import FINDINGS, SLOTS
from .fusion import fuse_all, fuse_finding, pct_rank, pct_rank_on
from .geometry import center_crop_box, crop_pixels
from .slots import pick_indices, total_slices

__version__ = "0.1.0"

__all__ = [
    "FINDINGS",
    "SLOTS",
    "center_crop_box",
    "class_attention",
    "class_attention_probs",
    "crop_pixels",
    "fuse_all",
    "fuse_finding",
    "pct_rank",
    "pct_rank_on",
    "pick_indices",
    "softmax",
    "total_slices",
]
