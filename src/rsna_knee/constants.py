"""Constants mirrored from the reference pipeline in ``reference/``.

Every value here is taken from the public Kaggle notebook code; the source
location is noted next to it so it can be audited.
"""

from __future__ import annotations

#: The twelve findings scored by the competition (column order of submission).
FINDINGS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

#: Physical crop window in millimetres (``CROP_MM`` in raptor224_run.py).
CROP_MM: float = 140.0

#: ``(plane, fluid_sensitive_preference, n_slices)``; preference 1 = prefer a
#: fluid-sensitive series, 0 = prefer a non-fluid one, -1 = no preference.
#: Mirrors ``SLOTS`` in raptor224_run.py (96 slices in total).
SLOTS: tuple[tuple[str, int, int], ...] = (
    ("Sagittal", 1, 26),
    ("Sagittal", 0, 22),
    ("Coronal", 1, 18),
    ("Coronal", 0, 12),
    ("Axial", -1, 18),
)

#: Fraction of a series' slice stack that is sampled (2 %-98 %), see
#: ``pick_indices`` in the reference code. The outer slices matter for the
#: collateral ligaments and the lateral meniscus.
SLICE_RANGE: tuple[float, float] = (0.02, 0.98)

#: Share of the "Raptor half" given to the 224-px model (``A224``).
A224: float = 0.35

#: Default Raptor weight in the outer blend (``W_RAPTOR``); ours = 1 - it.
W_RAPTOR: float = 0.5

#: Per-finding Raptor weight overrides (``W_RAPTOR_C`` in notebook_pipeline.py).
W_RAPTOR_PER_FINDING: dict[str, float] = {
    "Medial OA": 0.5,
    "Lateral OA": 0.2,
    "PF OA": 0.6,
    "Synovitis": 0.8,
    "Baker's": 0.35,
}

#: Value a failed study is left at by the reference inference scripts.
FAILURE_SENTINEL: float = 0.5
