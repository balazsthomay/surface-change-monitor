"""Tests for spectral indices module.

Tests cover:
  - NDVI formula: (B08 - B04) / (B08 + B04)
  - NDBI formula: (B11 - B08) / (B11 + B08)
  - NDWI formula: (B03 - B08) / (B03 + B08)
  - Zero-denominator handling: result = 0.0 (not NaN/inf)
  - NaN input propagation: NaN in -> NaN out
  - Index value range: all values in [-1, 1]
  - add_indices_to_composite: appends indices as bands 7–9
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from surface_change_monitor.config import AOI
from surface_change_monitor.composite import MonthlyComposite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataarray(values: np.ndarray, name: str = "band") -> xr.DataArray:
    """Wrap a 2-D numpy array in an xr.DataArray with x/y dims."""
    h, w = values.shape
    return xr.DataArray(
        values.astype(np.float32),
        dims=["y", "x"],
        coords={"y": np.arange(h, dtype=np.float32), "x": np.arange(w, dtype=np.float32)},
    )


def _make_composite(band_data: dict[str, np.ndarray]) -> MonthlyComposite:
    """Build a minimal MonthlyComposite from a dict of band_name -> 2-D array.

    All arrays must have the same shape. Band dimension coordinates are the
    dict keys in insertion order.
    """
    band_names = list(band_data.keys())
    h, w = next(iter(band_data.values())).shape

    slices = []
    for band_name in band_names:
        arr = band_data[band_name].astype(np.float32)
        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            coords={
                "y": np.arange(h, dtype=np.float32),
                "x": np.arange(w, dtype=np.float32),
            },
        )
        slices.append(da)

    data = xr.concat(slices, dim="band")
    data = data.assign_coords(band=band_names)

    aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)
    clear_obs = xr.DataArray(
        np.full((h, w), 3, dtype=np.int32),
        dims=["y", "x"],
        coords={"y": np.arange(h, dtype=np.float32), "x": np.arange(w, dtype=np.float32)},
    )
    return MonthlyComposite(
        data=data,
        year_month="2024-06",
        n_scenes=3,
        clear_obs_count=clear_obs,
        reliable=True,
        aoi=aoi,
    )


# ---------------------------------------------------------------------------
# 7.1a  test_ndvi
# ---------------------------------------------------------------------------


class TestNdvi:
    def test_ndvi_known_values(self):
        """NDVI = (B08 - B04) / (B08 + B04) with known input -> known output."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.array([[0.8, 0.6]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.2, 0.4]], dtype=np.float32))

        result = ndvi(b08, b04)

        expected = np.array([[(0.8 - 0.2) / (0.8 + 0.2), (0.6 - 0.4) / (0.6 + 0.4)]])
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndvi_uniform_vegetation(self):
        """High NIR, low red -> positive NDVI close to 1."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.full((2, 2), 0.9, dtype=np.float32))
        b04 = _make_dataarray(np.full((2, 2), 0.1, dtype=np.float32))

        result = ndvi(b08, b04)

        expected = (0.9 - 0.1) / (0.9 + 0.1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndvi_returns_dataarray(self):
        """ndvi() returns an xr.DataArray."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.array([[0.5]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.3]], dtype=np.float32))

        result = ndvi(b08, b04)

        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# 7.1b  test_ndbi
# ---------------------------------------------------------------------------


class TestNdbi:
    def test_ndbi_known_values(self):
        """NDBI = (B11 - B08) / (B11 + B08) with known input -> known output."""
        from surface_change_monitor.indices import ndbi

        b11 = _make_dataarray(np.array([[0.7, 0.3]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.3, 0.7]], dtype=np.float32))

        result = ndbi(b11, b08)

        expected = np.array([[(0.7 - 0.3) / (0.7 + 0.3), (0.3 - 0.7) / (0.3 + 0.7)]])
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndbi_built_up_area(self):
        """High SWIR, low NIR -> positive NDBI (built-up / impervious)."""
        from surface_change_monitor.indices import ndbi

        b11 = _make_dataarray(np.full((2, 2), 0.8, dtype=np.float32))
        b08 = _make_dataarray(np.full((2, 2), 0.2, dtype=np.float32))

        result = ndbi(b11, b08)

        expected = (0.8 - 0.2) / (0.8 + 0.2)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndbi_returns_dataarray(self):
        """ndbi() returns an xr.DataArray."""
        from surface_change_monitor.indices import ndbi

        b11 = _make_dataarray(np.array([[0.5]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.3]], dtype=np.float32))

        result = ndbi(b11, b08)

        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# 7.1c  test_ndwi
# ---------------------------------------------------------------------------


class TestNdwi:
    def test_ndwi_known_values(self):
        """NDWI = (B03 - B08) / (B03 + B08) with known input -> known output."""
        from surface_change_monitor.indices import ndwi

        b03 = _make_dataarray(np.array([[0.6, 0.2]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.2, 0.6]], dtype=np.float32))

        result = ndwi(b03, b08)

        expected = np.array([[(0.6 - 0.2) / (0.6 + 0.2), (0.2 - 0.6) / (0.2 + 0.6)]])
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndwi_water_body(self):
        """High green, low NIR -> positive NDWI (open water)."""
        from surface_change_monitor.indices import ndwi

        b03 = _make_dataarray(np.full((2, 2), 0.7, dtype=np.float32))
        b08 = _make_dataarray(np.full((2, 2), 0.1, dtype=np.float32))

        result = ndwi(b03, b08)

        expected = (0.7 - 0.1) / (0.7 + 0.1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_ndwi_returns_dataarray(self):
        """ndwi() returns an xr.DataArray."""
        from surface_change_monitor.indices import ndwi

        b03 = _make_dataarray(np.array([[0.4]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.3]], dtype=np.float32))

        result = ndwi(b03, b08)

        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# 7.1d  test_zero_denominator
# ---------------------------------------------------------------------------


class TestZeroDenominator:
    def test_ndvi_zero_denominator_returns_zero(self):
        """When B08 + B04 == 0, NDVI must return 0.0, not NaN or inf."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.array([[0.0, 0.5]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.0, 0.3]], dtype=np.float32))

        result = ndvi(b08, b04)

        assert result.values[0, 0] == pytest.approx(0.0), (
            f"Zero denominator should give 0.0, got {result.values[0, 0]}"
        )
        assert np.isfinite(result.values[0, 0]), "Result must not be NaN or inf"

    def test_ndbi_zero_denominator_returns_zero(self):
        """When B11 + B08 == 0, NDBI must return 0.0."""
        from surface_change_monitor.indices import ndbi

        b11 = _make_dataarray(np.array([[0.0]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.0]], dtype=np.float32))

        result = ndbi(b11, b08)

        assert result.values[0, 0] == pytest.approx(0.0)
        assert np.isfinite(result.values[0, 0])

    def test_ndwi_zero_denominator_returns_zero(self):
        """When B03 + B08 == 0, NDWI must return 0.0."""
        from surface_change_monitor.indices import ndwi

        b03 = _make_dataarray(np.array([[0.0]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.0]], dtype=np.float32))

        result = ndwi(b03, b08)

        assert result.values[0, 0] == pytest.approx(0.0)
        assert np.isfinite(result.values[0, 0])

    def test_mixed_zero_and_nonzero_denominators(self):
        """Zero-denom pixels get 0.0; non-zero pixels get the correct value."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.array([[0.0, 0.8]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.0, 0.2]], dtype=np.float32))

        result = ndvi(b08, b04)

        assert result.values[0, 0] == pytest.approx(0.0)
        assert result.values[0, 1] == pytest.approx((0.8 - 0.2) / (0.8 + 0.2), rel=1e-5)


# ---------------------------------------------------------------------------
# 7.1e  test_nan_input
# ---------------------------------------------------------------------------


class TestNanInput:
    def test_ndvi_nan_propagates(self):
        """NaN in either input band -> NaN in output."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.array([[np.nan, 0.5]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.2, 0.3]], dtype=np.float32))

        result = ndvi(b08, b04)

        assert np.isnan(result.values[0, 0]), "NaN input should produce NaN output"
        assert np.isfinite(result.values[0, 1]), "Valid input should produce finite output"

    def test_ndbi_nan_propagates(self):
        """NaN in either input -> NaN in NDBI output."""
        from surface_change_monitor.indices import ndbi

        b11 = _make_dataarray(np.array([[np.nan]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[0.4]], dtype=np.float32))

        result = ndbi(b11, b08)

        assert np.isnan(result.values[0, 0])

    def test_ndwi_nan_propagates(self):
        """NaN in either input -> NaN in NDWI output."""
        from surface_change_monitor.indices import ndwi

        b03 = _make_dataarray(np.array([[0.5]], dtype=np.float32))
        b08 = _make_dataarray(np.array([[np.nan]], dtype=np.float32))

        result = ndwi(b03, b08)

        assert np.isnan(result.values[0, 0])

    def test_all_nan_input(self):
        """Fully NaN arrays produce fully NaN outputs."""
        from surface_change_monitor.indices import ndvi

        b08 = _make_dataarray(np.full((2, 2), np.nan, dtype=np.float32))
        b04 = _make_dataarray(np.full((2, 2), np.nan, dtype=np.float32))

        result = ndvi(b08, b04)

        assert np.all(np.isnan(result.values))


# ---------------------------------------------------------------------------
# 7.1f  test_index_range
# ---------------------------------------------------------------------------


class TestIndexRange:
    def test_ndvi_range(self):
        """NDVI values lie within [-1, 1] for random reflectance-like inputs."""
        from surface_change_monitor.indices import ndvi

        rng = np.random.default_rng(42)
        vals = rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32)
        b08 = _make_dataarray(vals)
        b04 = _make_dataarray(rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32))

        result = ndvi(b08, b04)
        finite_vals = result.values[np.isfinite(result.values)]

        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0), (
            "NDVI must lie within [-1, 1]"
        )

    def test_ndbi_range(self):
        """NDBI values lie within [-1, 1]."""
        from surface_change_monitor.indices import ndbi

        rng = np.random.default_rng(43)
        b11 = _make_dataarray(rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32))
        b08 = _make_dataarray(rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32))

        result = ndbi(b11, b08)
        finite_vals = result.values[np.isfinite(result.values)]

        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0)

    def test_ndwi_range(self):
        """NDWI values lie within [-1, 1]."""
        from surface_change_monitor.indices import ndwi

        rng = np.random.default_rng(44)
        b03 = _make_dataarray(rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32))
        b08 = _make_dataarray(rng.uniform(0.0, 1.0, (50, 50)).astype(np.float32))

        result = ndwi(b03, b08)
        finite_vals = result.values[np.isfinite(result.values)]

        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0)

    def test_ndvi_boundary_values(self):
        """Edge cases: NDVI of pure NIR (B04=0) and pure red (B08=0)."""
        from surface_change_monitor.indices import ndvi

        # B08=1, B04=0 -> NDVI = 1.0
        b08 = _make_dataarray(np.array([[1.0]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[0.0]], dtype=np.float32))
        assert ndvi(b08, b04).values[0, 0] == pytest.approx(1.0)

        # B08=0, B04=1 -> NDVI = -1.0
        b08 = _make_dataarray(np.array([[0.0]], dtype=np.float32))
        b04 = _make_dataarray(np.array([[1.0]], dtype=np.float32))
        assert ndvi(b08, b04).values[0, 0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 7.1g  test_add_indices_to_composite
# ---------------------------------------------------------------------------


class TestAddIndicesToComposite:
    def _make_6band_composite(self) -> MonthlyComposite:
        """Build a composite with the 6 canonical Sentinel-2 spectral bands."""
        shape = (4, 4)
        rng = np.random.default_rng(0)
        band_data = {
            "B02": rng.uniform(0.01, 0.3, shape).astype(np.float32),
            "B03": rng.uniform(0.01, 0.3, shape).astype(np.float32),
            "B04": rng.uniform(0.01, 0.3, shape).astype(np.float32),
            "B08": rng.uniform(0.1, 0.9, shape).astype(np.float32),
            "B11": rng.uniform(0.01, 0.5, shape).astype(np.float32),
            "B12": rng.uniform(0.01, 0.4, shape).astype(np.float32),
        }
        return _make_composite(band_data)

    def test_output_has_9_bands(self):
        """add_indices_to_composite expands 6-band composite to 9 bands."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        assert result.data.shape[0] == 9, (
            f"Expected 9 bands, got {result.data.shape[0]}"
        )

    def test_indices_appended_as_bands_7_to_9(self):
        """Bands 7-9 (0-indexed 6-8) are NDVI, NDBI, NDWI."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        band_coords = list(result.data.coords["band"].values)
        assert band_coords[6] == "NDVI", f"Band 7 should be NDVI, got {band_coords[6]}"
        assert band_coords[7] == "NDBI", f"Band 8 should be NDBI, got {band_coords[7]}"
        assert band_coords[8] == "NDWI", f"Band 9 should be NDWI, got {band_coords[8]}"

    def test_original_bands_preserved(self):
        """Original 6 spectral bands are unchanged in the output."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        original_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
        for band in original_bands:
            original = composite.data.sel(band=band).values
            updated = result.data.sel(band=band).values
            np.testing.assert_array_equal(original, updated, err_msg=f"Band {band} was modified")

    def test_ndvi_values_correct(self):
        """NDVI band in composite matches independently computed values."""
        from surface_change_monitor.indices import add_indices_to_composite, ndvi

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        b08 = composite.data.sel(band="B08")
        b04 = composite.data.sel(band="B04")
        expected_ndvi = ndvi(b08, b04).values

        actual_ndvi = result.data.sel(band="NDVI").values
        np.testing.assert_allclose(actual_ndvi, expected_ndvi, rtol=1e-5)

    def test_ndbi_values_correct(self):
        """NDBI band in composite matches independently computed values."""
        from surface_change_monitor.indices import add_indices_to_composite, ndbi

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        b11 = composite.data.sel(band="B11")
        b08 = composite.data.sel(band="B08")
        expected_ndbi = ndbi(b11, b08).values

        actual_ndbi = result.data.sel(band="NDBI").values
        np.testing.assert_allclose(actual_ndbi, expected_ndbi, rtol=1e-5)

    def test_ndwi_values_correct(self):
        """NDWI band in composite matches independently computed values."""
        from surface_change_monitor.indices import add_indices_to_composite, ndwi

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        b03 = composite.data.sel(band="B03")
        b08 = composite.data.sel(band="B08")
        expected_ndwi = ndwi(b03, b08).values

        actual_ndwi = result.data.sel(band="NDWI").values
        np.testing.assert_allclose(actual_ndwi, expected_ndwi, rtol=1e-5)

    def test_returns_new_composite_instance(self):
        """add_indices_to_composite returns a new MonthlyComposite, not mutating in place."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        assert result is not composite
        assert composite.data.shape[0] == 6, "Original composite must not be mutated"

    def test_metadata_preserved(self):
        """Metadata (year_month, n_scenes, reliable, aoi) is preserved."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        assert result.year_month == composite.year_month
        assert result.n_scenes == composite.n_scenes
        assert result.reliable == composite.reliable
        assert result.aoi is composite.aoi

    def test_clear_obs_count_preserved(self):
        """clear_obs_count DataArray is preserved unchanged."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        np.testing.assert_array_equal(
            result.clear_obs_count.values,
            composite.clear_obs_count.values,
        )

    def test_spatial_dims_unchanged(self):
        """Spatial dimensions (H, W) are unchanged after adding indices."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        assert result.data.shape[1:] == composite.data.shape[1:], (
            "Spatial dimensions must be preserved"
        )

    def test_index_bands_are_float32(self):
        """Appended index bands have float32 dtype."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        for idx_band in ("NDVI", "NDBI", "NDWI"):
            band_data = result.data.sel(band=idx_band)
            assert band_data.dtype == np.float32, (
                f"Expected float32 for {idx_band}, got {band_data.dtype}"
            )

    def test_index_values_in_range(self):
        """All appended index bands have finite values in [-1, 1]."""
        from surface_change_monitor.indices import add_indices_to_composite

        composite = self._make_6band_composite()
        result = add_indices_to_composite(composite)

        for idx_band in ("NDVI", "NDBI", "NDWI"):
            vals = result.data.sel(band=idx_band).values
            finite = vals[np.isfinite(vals)]
            assert np.all(finite >= -1.0) and np.all(finite <= 1.0), (
                f"{idx_band} values out of [-1, 1] range"
            )
