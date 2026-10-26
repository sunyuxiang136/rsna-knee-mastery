# API Reference (`rsna_knee`)

This page documents the public modules and functions provided by the `rsna_knee` package.

All functions are written in pure Python and NumPy with full PEP 484 type annotations and docstrings.

---

## Table of Contents
- [Constants (`rsna_knee.constants`)](#constants-rsna_kneeconstants)
- [Geometry (`rsna_knee.geometry`)](#geometry-rsna_kneegeometry)
- [Slots & Sampling (`rsna_knee.slots`)](#slots--sampling-rsna_kneeslots)
- [Attention Pooling (`rsna_knee.attention`)](#attention-pooling-rsna_kneeattention)
- [Rank Fusion (`rsna_knee.fusion`)](#rank-fusion-rsna_kneefusion)

---

## Constants (`rsna_knee.constants`)

Mirrors configuration parameters from the upstream competition pipeline.

### `FINDINGS`
`tuple[str, ...]`: The twelve target findings in submission column order:
`("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")`.

### `CROP_MM`
`float`: Default physical field of view in millimetres (`140.0`).

### `SLOTS`
`tuple[tuple[str, int, int], ...]`: Tuple of `(plane, fluid_preference, n_slices)`.
Total slices across all 5 slots equal `96`.

### `SLICE_RANGE`
`tuple[float, float]`: Slicing stack bounds `(0.02, 0.98)` to protect collateral ligaments.

### `A224`
`float`: Inner Raptor blend share allocated to the 224-px CoAtNet model (`0.35`).

### `W_RAPTOR`
`float`: Default weight assigned to the Raptor half in the outer blend (`0.5`).

### `W_RAPTOR_PER_FINDING`
`dict[str, float]`: Finding-specific outer blend weights:
- `"Medial OA": 0.5`
- `"Lateral OA": 0.2`
- `"PF OA": 0.6`
- `"Synovitis": 0.8`
- `"Baker's": 0.35`

### `FAILURE_SENTINEL`
`float`: Probability value filled for failed predictions (`0.5`).

---

## Geometry (`rsna_knee.geometry`)

Physical coordinate scaling and centered bounding box computation.

### `crop_pixels(pixel_spacing_mm: float, crop_mm: float = CROP_MM) -> int`
Computes the number of pixels spanning `crop_mm` given the in-plane scanner pixel spacing.
- **Parameters**:
  - `pixel_spacing_mm`: DICOM pixel spacing in mm/pixel (must be `> 0`).
  - `crop_mm`: Desired physical width/height in mm (default: `140.0`).
- **Returns**: Integer pixel count `round(crop_mm / pixel_spacing_mm)`.
- **Raises**: `ValueError` if `pixel_spacing_mm <= 0`.

### `center_crop_box(height: int, width: int, pixel_spacing_mm: float, crop_mm: float = CROP_MM) -> tuple[int, int, int]`
Computes the `(y0, x0, size)` bounding box centered in the given 2D image matrix.
- **Parameters**:
  - `height`: Image matrix height in pixels.
  - `width`: Image matrix width in pixels.
  - `pixel_spacing_mm`: DICOM pixel spacing.
  - `crop_mm`: Desired physical crop size.
- **Returns**: `(y0, x0, size)` where `size` is clamped to image boundaries.

---

## Slots & Sampling (`rsna_knee.slots`)

Series slice sampling and slot bookkeeping.

### `total_slices() -> int`
Returns the total number of slices across all configured slots (`96`).

### `pick_indices(n_slices: int, k: int, lo_frac: float = 0.02, hi_frac: float = 0.98) -> list[int]`
Selects `k` evenly spaced slice indices from a stack of `n_slices` covering `lo_frac` to `hi_frac`.
- **Parameters**:
  - `n_slices`: Number of available slices in the series (`> 0`).
  - `k`: Number of target slices to select (`> 0`).
  - `lo_frac`: Lower fraction of the stack (default: `0.02`).
  - `hi_frac`: Upper fraction of the stack (default: `0.98`).
- **Returns**: List of sorted integer slice indices of length `k`.

---

## Attention Pooling (`rsna_knee.attention`)

Multi-instance learning attention mechanisms in NumPy.

### `softmax(x: np.ndarray, axis: int = -1) -> np.ndarray`
Numerically stable softmax normalization over the specified axis.

### `class_attention(features: np.ndarray, att_weights: np.ndarray) -> np.ndarray`
Calculates independent attention weights for each target disease across windows.
- **Parameters**:
  - `features`: `[K, F]` array of window feature vectors.
  - `att_weights`: `[F, N]` attention projection matrix for `N` findings.
- **Returns**: `[K, N]` normalized attention matrix where every column sums to 1.0.

### `class_attention_probs(features: np.ndarray, att_weights: np.ndarray, cls_weights: np.ndarray, cls_bias: np.ndarray) -> np.ndarray`
Computes final classification probabilities for all `N` findings via attention-weighted feature pooling.
- **Returns**: 1D array of shape `(N,)` with probabilities in `(0, 1)`.

---

## Rank Fusion (`rsna_knee.fusion`)

Stage 3 percentile-rank ensembling and failure quarantine.

### `pct_rank(values: np.ndarray) -> np.ndarray`
Computes empirical percentile ranks in `(0, 1]` with average rank assigned to ties.
Matches `pandas.Series(values).rank(method='average', pct=True)`.

### `pct_rank_on(values: np.ndarray, ok: np.ndarray) -> np.ndarray`
Computes percentile ranks strictly among non-failed rows indicated by boolean mask `ok`.
Rows where `ok` is `False` remain `np.nan`.

### `failed_rows(raw: np.ndarray) -> np.ndarray`
Identifies rows where all column probabilities equal `0.5` (the failure sentinel).

### `fuse_finding(raptor384, raptor224, ours, fail_384, fail_224, fail_ours, w_raptor=0.5, a224=0.35) -> np.ndarray`
Fuses 1D predictions for a single finding across three models with hierarchical fallback.

### `fuse_all(raptor384, raptor224, ours, findings, per_finding_weights=None, a224=0.35) -> np.ndarray`
Executes multi-model rank fusion for full `[N_studies, 12]` prediction matrices.
- **Parameters**:
  - `raptor384`: `[N, 12]` probabilities from the 384px model / S6 replacement.
  - `raptor224`: `[N, 12]` probabilities from the 224px CoAtNet model.
  - `ours`: `[N, 12]` probabilities from the multi-fold R6 model.
  - `findings`: Tuple of 12 finding names matching columns.
  - `per_finding_weights`: Optional dictionary overriding default Raptor weights.
  - `a224`: Raptor internal blend fraction for the 224-px model.
- **Returns**: `[N, 12]` fused ranking scores in `[0, 1]`.
