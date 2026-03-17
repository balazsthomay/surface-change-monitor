"""Tests for the change label generation module.

Tests cover:
  - Binary change detection (only imperviousness increases count)
  - Configurable threshold for minimum detectable increase
  - Patch extraction with stride filtering, including NaN-heavy patch rejection
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import rioxarray  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_density_da(
    values: np.ndarray,
    *,
    west: float = 297000.0,
    south: float = 6690000.0,
    pixel_size: float = 10.0,
) -> xr.DataArray:
    """Wrap a 2-D numpy array in a georeferenced xr.DataArray.

    The array is placed in EPSG:32632 (UTM 32N) for convenience.
    """
    from rasterio.transform import from_origin
    import rioxarray  # noqa: F401 – needed for .rio accessor side-effects

    height, width = values.shape
    transform = from_origin(west, south + height * pixel_size, pixel_size, pixel_size)

    da = xr.DataArray(
        values.astype(np.float32),
        dims=["y", "x"],
        coords={
            "y": [south + height * pixel_size - (i + 0.5) * pixel_size for i in range(height)],
            "x": [west + (j + 0.5) * pixel_size for j in range(width)],
        },
    )
    da = da.rio.write_crs("EPSG:32632")
    da = da.rio.write_transform(transform)
    return da


def _make_composite_da(
    values: np.ndarray,
    *,
    n_bands: int = 4,
    west: float = 297000.0,
    south: float = 6690000.0,
    pixel_size: float = 10.0,
) -> xr.DataArray:
    """Wrap a (H, W) numpy array into a (bands, H, W) composite DataArray."""
    height, width = values.shape
    band_data = np.stack([values] * n_bands, axis=0).astype(np.float32)
    from rasterio.transform import from_origin

    transform = from_origin(west, south + height * pixel_size, pixel_size, pixel_size)

    da = xr.DataArray(
        band_data,
        dims=["band", "y", "x"],
        coords={
            "band": [f"B{i:02d}" for i in range(n_bands)],
            "y": [south + height * pixel_size - (i + 0.5) * pixel_size for i in range(height)],
            "x": [west + (j + 0.5) * pixel_size for j in range(width)],
        },
    )
    da = da.rio.write_crs("EPSG:32632")
    da = da.rio.write_transform(transform)
    return da


# ---------------------------------------------------------------------------
# Tests: generate_change_labels
# ---------------------------------------------------------------------------


class TestBinaryChange:
    """test_binary_change: 0->50 = change; 80->80 = no change; 50->20 = no change."""

    def test_increase_produces_change(self):
        """density 0→50 (delta=50): should produce label=1."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((4, 4), 0.0))
        t2 = _make_density_da(np.full((4, 4), 50.0))

        labels = generate_change_labels(t1, t2, threshold=10.0)

        assert labels.values.dtype == np.uint8
        assert np.all(labels.values == 1), (
            f"Expected all 1s for 0→50 change, got {labels.values}"
        )

    def test_no_change_produces_zero(self):
        """density 80→80 (delta=0): should produce label=0."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((4, 4), 80.0))
        t2 = _make_density_da(np.full((4, 4), 80.0))

        labels = generate_change_labels(t1, t2, threshold=10.0)

        assert np.all(labels.values == 0), (
            f"Expected all 0s for 80→80 (no change), got {labels.values}"
        )

    def test_decrease_produces_no_change(self):
        """density 50→20 (delta=-30): should produce label=0 (only increases count)."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((4, 4), 50.0))
        t2 = _make_density_da(np.full((4, 4), 20.0))

        labels = generate_change_labels(t1, t2, threshold=10.0)

        assert np.all(labels.values == 0), (
            f"Expected all 0s for 50→20 (decrease, not counted), got {labels.values}"
        )

    def test_mixed_pixel_map(self):
        """Mixed densities produce the correct per-pixel binary label."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1_data = np.array([[0.0, 50.0], [80.0, 30.0]], dtype=np.float32)
        t2_data = np.array([[50.0, 20.0], [80.0, 45.0]], dtype=np.float32)
        # Deltas:        [[50,  -30],   [0,    15]]
        # With threshold=10: [[1, 0], [0, 1]]
        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        t1 = _make_density_da(t1_data)
        t2 = _make_density_da(t2_data)

        labels = generate_change_labels(t1, t2, threshold=10.0)

        np.testing.assert_array_equal(labels.values, expected)

    def test_output_is_xarray_dataarray(self):
        """generate_change_labels returns an xr.DataArray."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.zeros((4, 4)))
        t2 = _make_density_da(np.full((4, 4), 20.0))

        labels = generate_change_labels(t1, t2)

        assert isinstance(labels, xr.DataArray)

    def test_output_dtype_is_uint8(self):
        """Output array must have uint8 dtype."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.zeros((4, 4)))
        t2 = _make_density_da(np.full((4, 4), 20.0))

        labels = generate_change_labels(t1, t2)

        assert labels.values.dtype == np.uint8


class TestChangeThreshold:
    """test_change_threshold: increases < threshold are ignored."""

    def test_exactly_at_threshold_is_change(self):
        """Delta == threshold (e.g. 10.0) should be labelled as change (>=)."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((2, 2), 30.0))
        t2 = _make_density_da(np.full((2, 2), 40.0))  # delta = 10 == threshold

        labels = generate_change_labels(t1, t2, threshold=10.0)

        assert np.all(labels.values == 1)

    def test_below_threshold_is_no_change(self):
        """Delta < threshold (e.g. 9.9) should be labelled as no change."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((2, 2), 30.0))
        t2 = _make_density_da(np.full((2, 2), 39.9))  # delta = 9.9 < 10

        labels = generate_change_labels(t1, t2, threshold=10.0)

        assert np.all(labels.values == 0)

    def test_custom_threshold_5(self):
        """With threshold=5.0, delta=7 counts as change."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((2, 2), 0.0))
        t2 = _make_density_da(np.full((2, 2), 7.0))  # delta = 7 >= 5

        labels = generate_change_labels(t1, t2, threshold=5.0)

        assert np.all(labels.values == 1)

    def test_custom_threshold_20(self):
        """With threshold=20.0, delta=15 should not count as change."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((2, 2), 10.0))
        t2 = _make_density_da(np.full((2, 2), 25.0))  # delta = 15 < 20

        labels = generate_change_labels(t1, t2, threshold=20.0)

        assert np.all(labels.values == 0)

    def test_default_threshold_is_10(self):
        """Default threshold should be 10.0."""
        from surface_change_monitor.labels.change import generate_change_labels

        t1 = _make_density_da(np.full((2, 2), 0.0))
        # delta = 10 exactly at default threshold → change
        t2 = _make_density_da(np.full((2, 2), 10.0))

        labels = generate_change_labels(t1, t2)  # no threshold arg

        assert np.all(labels.values == 1)


# ---------------------------------------------------------------------------
# Tests: extract_patches
# ---------------------------------------------------------------------------


class TestExtractPatches:
    """test_extract_patches: 256×256 patches with stride, filters >50% NaN patches."""

    def _make_inputs(
        self,
        height: int = 512,
        width: int = 512,
        n_bands: int = 4,
        nan_fraction: float = 0.0,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Return (composite_t1, composite_t2, labels) of given size."""
        rng = np.random.default_rng(99)
        base = rng.random((height, width)).astype(np.float32)
        if nan_fraction > 0:
            mask = rng.random((height, width)) < nan_fraction
            base[mask] = np.nan

        c1 = _make_composite_da(base, n_bands=n_bands)
        c2 = _make_composite_da(base * 1.1, n_bands=n_bands)
        lbl = _make_density_da(np.where(np.isnan(base), np.nan, (base > 0.5).astype(np.float32)))
        return c1, c2, lbl.astype(np.uint8)

    def test_returns_list_of_dicts(self):
        """extract_patches must return a list of dicts."""
        from surface_change_monitor.labels.change import extract_patches

        c1, c2, lbl = self._make_inputs()
        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        assert isinstance(patches, list)
        assert len(patches) > 0
        assert isinstance(patches[0], dict)

    def test_patch_dict_keys(self):
        """Each patch dict must contain 't1', 't2', 'label' keys."""
        from surface_change_monitor.labels.change import extract_patches

        c1, c2, lbl = self._make_inputs()
        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        for patch in patches:
            assert "t1" in patch, f"Missing 't1' key in patch: {patch.keys()}"
            assert "t2" in patch, f"Missing 't2' key in patch: {patch.keys()}"
            assert "label" in patch, f"Missing 'label' key in patch: {patch.keys()}"

    def test_patch_shapes(self):
        """t1 and t2 are (C, 256, 256), label is (256, 256)."""
        from surface_change_monitor.labels.change import extract_patches

        n_bands = 4
        c1, c2, lbl = self._make_inputs(n_bands=n_bands)
        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        for patch in patches:
            assert patch["t1"].shape == (n_bands, 256, 256), (
                f"t1 shape {patch['t1'].shape} != ({n_bands}, 256, 256)"
            )
            assert patch["t2"].shape == (n_bands, 256, 256), (
                f"t2 shape {patch['t2'].shape} != ({n_bands}, 256, 256)"
            )
            assert patch["label"].shape == (256, 256), (
                f"label shape {patch['label'].shape} != (256, 256)"
            )

    def test_patches_are_numpy_arrays(self):
        """t1, t2 and label must be numpy arrays (not xarray)."""
        from surface_change_monitor.labels.change import extract_patches

        c1, c2, lbl = self._make_inputs()
        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        for patch in patches:
            assert isinstance(patch["t1"], np.ndarray)
            assert isinstance(patch["t2"], np.ndarray)
            assert isinstance(patch["label"], np.ndarray)

    def test_stride_controls_number_of_patches(self):
        """Smaller stride yields more patches than larger stride."""
        from surface_change_monitor.labels.change import extract_patches

        c1, c2, lbl = self._make_inputs(height=512, width=512)

        patches_stride128 = extract_patches(c1, c2, lbl, patch_size=256, stride=128)
        patches_stride256 = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        # stride=128 → 3×3 = 9 windows; stride=256 → 2×2 = 4 windows (no NaN filtering here)
        assert len(patches_stride128) >= len(patches_stride256), (
            f"Expected more patches with stride=128 ({len(patches_stride128)}) "
            f"than stride=256 ({len(patches_stride256)})"
        )

    def test_filters_high_nan_patches(self):
        """Patches with >50% NaN pixels in t1 are excluded from output."""
        from surface_change_monitor.labels.change import extract_patches

        # Create a 512×512 composite where the right half (x >= 256) is all NaN
        height, width, n_bands = 512, 512, 4
        base_left = np.ones((height, width // 2), dtype=np.float32) * 100.0
        base_right = np.full((height, width // 2), np.nan, dtype=np.float32)
        base = np.concatenate([base_left, base_right], axis=1)

        c1 = _make_composite_da(base, n_bands=n_bands)
        c2 = _make_composite_da(base, n_bands=n_bands)
        lbl_data = np.where(np.isnan(base), np.nan, np.zeros_like(base)).astype(np.float32)
        lbl = _make_density_da(lbl_data)

        # With stride=256, patch_size=256 → 4 possible patches (2×2 grid)
        # Patches covering the right half have >50% NaN → filtered out
        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        # Only patches from the left half (2 patches: top-left, bottom-left) survive
        assert len(patches) == 2, (
            f"Expected 2 patches (left-half only), got {len(patches)}"
        )

    def test_keeps_low_nan_patches(self):
        """Patches with <=50% NaN pixels are retained."""
        from surface_change_monitor.labels.change import extract_patches

        # All-clear 512×512 data
        c1, c2, lbl = self._make_inputs(height=512, width=512, nan_fraction=0.0)

        patches = extract_patches(c1, c2, lbl, patch_size=256, stride=256)

        # 2×2 grid → 4 patches, none filtered
        assert len(patches) == 4, f"Expected 4 clean patches, got {len(patches)}"

    def test_custom_patch_size(self):
        """patch_size parameter controls output array dimensions."""
        from surface_change_monitor.labels.change import extract_patches

        c1, c2, lbl = self._make_inputs(height=512, width=512, n_bands=6)
        patches = extract_patches(c1, c2, lbl, patch_size=128, stride=128)

        assert len(patches) > 0
        for patch in patches:
            assert patch["t1"].shape == (6, 128, 128)
            assert patch["label"].shape == (128, 128)
