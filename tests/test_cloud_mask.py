"""Tests for cloud masking module using Sentinel-2 SCL band."""

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers to build synthetic xarray DataArrays
# ---------------------------------------------------------------------------


def _make_scl(values: list[list[int]], resolution: float = 20.0) -> xr.DataArray:
    """Create a minimal SCL DataArray from a 2-D list of integer values."""
    data = np.array(values, dtype=np.uint8)
    height, width = data.shape
    x_coords = np.arange(width) * resolution
    y_coords = np.arange(height) * -resolution  # north-up convention
    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"x": x_coords, "y": y_coords},
        attrs={"resolution": resolution},
    )


def _make_band(values: list[list[float]], resolution: float = 10.0) -> xr.DataArray:
    """Create a minimal band DataArray from a 2-D list of float values."""
    data = np.array(values, dtype=np.float32)
    height, width = data.shape
    x_coords = np.arange(width) * resolution
    y_coords = np.arange(height) * -resolution
    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"x": x_coords, "y": y_coords},
    )


# ---------------------------------------------------------------------------
# 5.1  Failing tests
# ---------------------------------------------------------------------------


class TestCreateCloudMask:
    """Tests for create_cloud_mask."""

    def test_create_cloud_mask_from_scl(self):
        """Synthetic SCL array produces the expected binary mask (True = clear)."""
        from surface_change_monitor.cloud_mask import create_cloud_mask

        # Row 0: all clear SCL values (4=vegetation, 5=soil, 6=water, 11=snow)
        # Row 1: all masked SCL values (3=shadow, 7=unclassified, 8=med cloud, 9=high cloud, 10=cirrus)
        scl = _make_scl([[4, 5, 6, 11], [3, 7, 8, 9]])
        mask = create_cloud_mask(scl)

        # All of row 0 should be True (clear)
        assert bool(mask[0, 0]), "SCL=4 (vegetation) must be clear"
        assert bool(mask[0, 1]), "SCL=5 (bare soil) must be clear"
        assert bool(mask[0, 2]), "SCL=6 (water) must be clear"
        assert bool(mask[0, 3]), "SCL=11 (snow) must be clear"

        # All of row 1 should be False (masked/cloudy)
        assert not bool(mask[1, 0]), "SCL=3 (cloud shadow) must be masked"
        assert not bool(mask[1, 1]), "SCL=7 (unclassified) must be masked"
        assert not bool(mask[1, 2]), "SCL=8 (medium cloud) must be masked"
        assert not bool(mask[1, 3]), "SCL=9 (high cloud) must be masked"

    def test_scl_values_masked(self):
        """Exactly values 3, 7, 8, 9, 10 are masked; all others are clear."""
        from surface_change_monitor.cloud_mask import create_cloud_mask

        mask_values = {3, 7, 8, 9, 10}

        for v in range(0, 12):
            scl = _make_scl([[v]])
            mask = create_cloud_mask(scl)
            pixel = bool(mask[0, 0])
            if v in mask_values:
                assert not pixel, f"SCL={v} should be masked (False) but was True"
            else:
                assert pixel, f"SCL={v} should be clear (True) but was False"

    def test_create_cloud_mask_returns_bool_dtype(self):
        """Result dtype must be boolean."""
        from surface_change_monitor.cloud_mask import create_cloud_mask

        scl = _make_scl([[4, 9], [5, 10]])
        mask = create_cloud_mask(scl)
        assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"

    def test_create_cloud_mask_preserves_coords(self):
        """Output mask must share coordinates with the input SCL array."""
        from surface_change_monitor.cloud_mask import create_cloud_mask

        scl = _make_scl([[4, 5], [9, 8]])
        mask = create_cloud_mask(scl)
        xr.testing.assert_equal(mask.coords["x"], scl.coords["x"])
        xr.testing.assert_equal(mask.coords["y"], scl.coords["y"])


class TestApplyCloudMask:
    """Tests for apply_cloud_mask."""

    def test_apply_cloud_mask(self):
        """Masked pixels become NaN; clear pixels retain their values."""
        from surface_change_monitor.cloud_mask import apply_cloud_mask

        band = _make_band([[1.0, 2.0], [3.0, 4.0]])
        # True = clear, False = masked
        mask = xr.DataArray(
            np.array([[True, False], [False, True]], dtype=bool),
            dims=["y", "x"],
            coords={"x": band.coords["x"], "y": band.coords["y"]},
        )
        result = apply_cloud_mask(band, mask)

        assert float(result[0, 0]) == pytest.approx(1.0), "Clear pixel must keep its value"
        assert float(result[1, 1]) == pytest.approx(4.0), "Clear pixel must keep its value"
        assert np.isnan(float(result[0, 1])), "Masked pixel must become NaN"
        assert np.isnan(float(result[1, 0])), "Masked pixel must become NaN"

    def test_apply_cloud_mask_all_clear(self):
        """When all pixels are clear, output equals input."""
        from surface_change_monitor.cloud_mask import apply_cloud_mask

        band = _make_band([[5.0, 6.0], [7.0, 8.0]])
        mask = xr.DataArray(
            np.ones((2, 2), dtype=bool),
            dims=["y", "x"],
            coords={"x": band.coords["x"], "y": band.coords["y"]},
        )
        result = apply_cloud_mask(band, mask)
        xr.testing.assert_allclose(result, band)

    def test_apply_cloud_mask_all_masked(self):
        """When all pixels are masked, output is all NaN."""
        from surface_change_monitor.cloud_mask import apply_cloud_mask

        band = _make_band([[5.0, 6.0], [7.0, 8.0]])
        mask = xr.DataArray(
            np.zeros((2, 2), dtype=bool),
            dims=["y", "x"],
            coords={"x": band.coords["x"], "y": band.coords["y"]},
        )
        result = apply_cloud_mask(band, mask)
        assert bool(np.all(np.isnan(result.values))), "All pixels should be NaN"

    def test_apply_cloud_mask_preserves_attrs(self):
        """Band attributes must be preserved through masking."""
        from surface_change_monitor.cloud_mask import apply_cloud_mask

        band = _make_band([[1.0, 2.0], [3.0, 4.0]])
        band.attrs["units"] = "reflectance"
        mask = xr.DataArray(
            np.ones((2, 2), dtype=bool),
            dims=["y", "x"],
            coords={"x": band.coords["x"], "y": band.coords["y"]},
        )
        result = apply_cloud_mask(band, mask)
        assert result.attrs.get("units") == "reflectance"


class TestCloudFreeFraction:
    """Tests for cloud_free_fraction."""

    def test_cloud_free_fraction(self):
        """Correct fraction is computed from a boolean mask."""
        from surface_change_monitor.cloud_mask import cloud_free_fraction

        mask = xr.DataArray(
            np.array([[True, True], [False, False]], dtype=bool),
            dims=["y", "x"],
        )
        fraction = cloud_free_fraction(mask)
        assert fraction == pytest.approx(0.5)

    def test_cloud_free_fraction_all_clear(self):
        """Returns 1.0 when all pixels are clear."""
        from surface_change_monitor.cloud_mask import cloud_free_fraction

        mask = xr.DataArray(np.ones((4, 4), dtype=bool), dims=["y", "x"])
        assert cloud_free_fraction(mask) == pytest.approx(1.0)

    def test_cloud_free_fraction_all_cloudy(self):
        """Returns 0.0 when all pixels are masked."""
        from surface_change_monitor.cloud_mask import cloud_free_fraction

        mask = xr.DataArray(np.zeros((4, 4), dtype=bool), dims=["y", "x"])
        assert cloud_free_fraction(mask) == pytest.approx(0.0)

    def test_cloud_free_fraction_returns_float(self):
        """Return type must be float."""
        from surface_change_monitor.cloud_mask import cloud_free_fraction

        mask = xr.DataArray(np.array([[True, False]], dtype=bool), dims=["y", "x"])
        result = cloud_free_fraction(mask)
        assert isinstance(result, float)

    def test_cloud_free_fraction_partial(self):
        """Three-out-of-four clear pixels -> 0.75."""
        from surface_change_monitor.cloud_mask import cloud_free_fraction

        mask = xr.DataArray(
            np.array([[True, True], [True, False]], dtype=bool),
            dims=["y", "x"],
        )
        assert cloud_free_fraction(mask) == pytest.approx(0.75)


class TestCloudMaskResampling:
    """Tests for 20m SCL upsampling to a 10m grid (nearest-neighbour)."""

    def test_cloud_mask_resampling(self):
        """20m SCL correctly upsampled to 10m grid via nearest-neighbour."""
        from surface_change_monitor.cloud_mask import create_cloud_mask, resample_mask_to_band

        # 2×2 SCL at 20m resolution
        # pixel (0,0)=4 clear, (0,1)=9 cloudy, (1,0)=5 clear, (1,1)=8 cloudy
        scl = _make_scl([[4, 9], [5, 8]], resolution=20.0)
        mask_20m = create_cloud_mask(scl)

        # 4×4 band at 10m resolution covering the same spatial extent
        band = _make_band(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]],
            resolution=10.0,
        )

        mask_10m = resample_mask_to_band(mask_20m, band)

        assert mask_10m.shape == band.shape, (
            f"Resampled mask shape {mask_10m.shape} must match band shape {band.shape}"
        )

    def test_resample_mask_preserves_clear_flag(self):
        """A fully clear 20m mask resampled to 10m remains fully clear."""
        from surface_change_monitor.cloud_mask import create_cloud_mask, resample_mask_to_band

        scl = _make_scl([[4, 5], [6, 4]], resolution=20.0)
        mask_20m = create_cloud_mask(scl)

        band = _make_band(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]],
            resolution=10.0,
        )
        mask_10m = resample_mask_to_band(mask_20m, band)

        assert bool(mask_10m.values.all()), "All pixels should remain clear after resampling"

    def test_resample_mask_preserves_cloudy_flag(self):
        """A fully cloudy 20m mask resampled to 10m remains fully cloudy."""
        from surface_change_monitor.cloud_mask import create_cloud_mask, resample_mask_to_band

        scl = _make_scl([[8, 9], [10, 3]], resolution=20.0)
        mask_20m = create_cloud_mask(scl)

        band = _make_band(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]],
            resolution=10.0,
        )
        mask_10m = resample_mask_to_band(mask_20m, band)

        assert not bool(mask_10m.values.any()), "All pixels should remain cloudy after resampling"
