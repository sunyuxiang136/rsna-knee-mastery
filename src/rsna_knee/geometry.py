"""Crop in millimetres, not pixels."""

from __future__ import annotations

from .constants import CROP_MM


def crop_pixels(pixel_spacing_mm: float, crop_mm: float = CROP_MM) -> int:
    """Number of pixels spanning ``crop_mm`` at the given in-plane spacing.

    >>> crop_pixels(0.3)
    467
    >>> crop_pixels(0.5)
    280
    """
    if pixel_spacing_mm <= 0:
        raise ValueError("pixel_spacing_mm must be positive")
    return int(round(crop_mm / max(pixel_spacing_mm, 1e-3)))


def center_crop_box(
    height: int,
    width: int,
    pixel_spacing_mm: float,
    crop_mm: float = CROP_MM,
) -> tuple[int, int, int]:
    """Return ``(y0, x0, size)`` of a square, centred, ``crop_mm``-wide box.

    The box is clamped to the image so small images are never padded.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    size = min(crop_pixels(pixel_spacing_mm, crop_mm), height, width)
    return (height - size) // 2, (width - size) // 2, size
