import numpy as np
import pytest

from rsna_knee import (
    FINDINGS,
    SLOTS,
    center_crop_box,
    class_attention,
    class_attention_probs,
    crop_pixels,
    fuse_all,
    fuse_finding,
    pct_rank,
    pct_rank_on,
    pick_indices,
    softmax,
    total_slices,
)


# ---- constants -------------------------------------------------------------
def test_twelve_findings_and_96_slices():
    assert len(FINDINGS) == 12
    assert len(set(FINDINGS)) == 12
    assert total_slices() == 96
    assert len(SLOTS) == 5


# ---- geometry --------------------------------------------------------------
@pytest.mark.parametrize("spacing,expected", [(0.3, 467), (0.5, 280)])
def test_crop_pixels(spacing, expected):
    assert crop_pixels(spacing) == expected


def test_same_physical_size_different_pixels():
    # 140 mm must come out regardless of scanner resolution
    for spacing in (0.3, 0.4, 0.5):
        assert abs(crop_pixels(spacing) * spacing - 140.0) < spacing


def test_crop_pixels_rejects_bad_spacing():
    with pytest.raises(ValueError):
        crop_pixels(0)


def test_center_crop_box_is_centered_and_clamped():
    y0, x0, size = center_crop_box(512, 512, 0.5)
    assert size == 280 and y0 == x0 == 116
    y0, x0, size = center_crop_box(200, 300, 0.3)  # window larger than image
    assert size == 200 and y0 == 0 and x0 == 50


# ---- slots -----------------------------------------------------------------
def test_pick_indices_covers_outer_slices():
    idx = pick_indices(100, 10)
    assert len(idx) == 10
    assert idx == sorted(idx)
    assert idx[0] == 2 and idx[-1] == 97  # 2 %-98 %, not just the middle


def test_pick_indices_bounds_when_oversampling():
    idx = pick_indices(5, 26)
    assert len(idx) == 26
    assert min(idx) >= 0 and max(idx) <= 4


def test_pick_indices_validation():
    with pytest.raises(ValueError):
        pick_indices(0, 3)
    with pytest.raises(ValueError):
        pick_indices(10, 0)


# ---- attention -------------------------------------------------------------
def test_softmax_sums_to_one_and_is_stable():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isclose(p.sum(), 1.0) and np.isfinite(p).all()


def test_class_attention_columns_sum_to_one_and_differ():
    rng = np.random.default_rng(0)
    feats, w = rng.normal(size=(4, 6)), rng.normal(size=(6, 3))
    att = class_attention(feats, w)
    assert att.shape == (4, 3)
    assert np.allclose(att.sum(axis=0), 1.0)
    assert not np.allclose(att[:, 0], att[:, 1])


def test_class_attention_probs_in_unit_interval():
    rng = np.random.default_rng(1)
    p = class_attention_probs(
        rng.normal(size=(4, 6)), rng.normal(size=(6, 3)), rng.normal(size=(3, 6)), np.zeros(3)
    )
    assert p.shape == (3,) and ((p > 0) & (p < 1)).all()


# ---- fusion ----------------------------------------------------------------
def test_pct_rank_matches_expected_and_ties():
    assert np.allclose(pct_rank([10, 30, 20]), [1 / 3, 1.0, 2 / 3])
    assert np.allclose(pct_rank([1, 1, 3]), [1.5 / 3, 1.5 / 3, 1.0])  # average ties


def test_pct_rank_is_scale_free():
    a = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert np.allclose(pct_rank(a), pct_rank(a * 0.1 + 0.4))


def test_pct_rank_on_ignores_failed_rows():
    v = np.array([0.9, 0.5, 0.1, 0.4])
    ok = np.array([True, False, True, True])
    r = pct_rank_on(v, ok)
    assert np.isnan(r[1])
    assert np.allclose(r[[0, 2, 3]], pct_rank(v[[0, 2, 3]]))


def test_fuse_finding_plain_blend():
    n = np.array([0.1, 0.2, 0.3, 0.4])
    none = np.zeros(4, dtype=bool)
    out = fuse_finding(n, n, n, none, none, none, w_raptor=0.5)
    assert np.allclose(out, pct_rank(n))  # identical models -> identical ranking


def test_fuse_finding_failure_fallbacks():
    n = np.array([0.1, 0.2, 0.3, 0.4])
    none = np.zeros(4, dtype=bool)
    f = np.array([False, False, True, False])
    out = fuse_finding(n, n, n, f, f, none)  # Raptor half missing -> ours alone
    assert np.allclose(out[2], pct_rank_on(n, ~none)[2])
    out = fuse_finding(n, n, n, none, none, f)  # ours missing -> Raptor half alone
    assert np.isclose(out[2], pct_rank_on(n, ~none)[2])
    out = fuse_finding(n, n, n, f, f, f)  # everything failed -> sentinel
    assert out[2] == 0.5


def test_fuse_finding_single_raptor_failure_uses_other_rank():
    r384 = np.array([0.1, 0.2, 0.3, 0.4])
    r224 = np.array([0.4, 0.3, 0.2, 0.1])
    ours = np.array([0.1, 0.2, 0.3, 0.4])
    none = np.zeros(4, dtype=bool)
    f384 = np.array([True, False, False, False])
    out = fuse_finding(r384, r224, ours, f384, none, none, w_raptor=1.0)
    assert np.isclose(out[0], pct_rank(r224)[0])


def test_fuse_finding_validation():
    n, none = np.zeros(3), np.zeros(3, dtype=bool)
    with pytest.raises(ValueError):
        fuse_finding(n, n, n, none, none, none, a224=1.0)
    with pytest.raises(ValueError):
        fuse_finding(n, n, n, none, none, none, w_raptor=1.5)


def test_fuse_all_shape_range_and_per_finding_weights():
    rng = np.random.default_rng(7)
    n = 30
    a, b, c = (rng.uniform(0.01, 0.99, size=(n, 12)) for _ in range(3))
    out = fuse_all(a, b, c, FINDINGS)
    assert out.shape == (n, 12)
    assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
    # Synovitis weights Raptor 0.8, so it should track Raptor more than "ours"
    j = FINDINGS.index("Synovitis")
    raptor = 0.65 * pct_rank(a[:, j]) + 0.35 * pct_rank(b[:, j])
    corr_r = np.corrcoef(out[:, j], raptor)[0, 1]
    corr_o = np.corrcoef(out[:, j], pct_rank(c[:, j]))[0, 1]
    assert corr_r > corr_o


def test_fuse_all_detects_sentinel_failures():
    rng = np.random.default_rng(3)
    a, b, c = (rng.uniform(0.01, 0.99, size=(10, 12)) for _ in range(3))
    a[4, :] = 0.5  # study 4 failed in Raptor-384
    out = fuse_all(a, b, c, FINDINGS)
    assert np.isfinite(out).all()


def test_fuse_all_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fuse_all(np.zeros((3, 12)), np.zeros((3, 12)), np.zeros((4, 12)), FINDINGS)
