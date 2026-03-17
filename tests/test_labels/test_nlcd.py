"""Tests for the NLCD impervious tile loading and processing module.

Tests cover:
  - Loading a mock GeoTIFF in EPSG:5070 (Albers Equal Area) with values 0–100
  - Reprojecting from EPSG:5070 to target UTM CRS (EPSG:32615 for Houston)
  - Resampling from 30m to 10m using bilinear interpolation
  - Annual change pairs: two years produce meaningful change labels
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401 – registers .rio accessor
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from surface_change_monitor.config import AOI, HOUSTON_AOI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_nlcd_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    crs_epsg: int = 5070,
    nodata: int = 127,
) -> Path:
    """Write a uint8 GeoTIFF that mimics an NLCD fractional impervious tile.

    NLCD uses EPSG:5070 (NAD83 / Conus Albers), 30m resolution, values 0-100
    with 127 as nodata.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    count, height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


# Houston AOI in EPSG:5070 (NAD83 / Conus Albers Equal Area Conic).
# Computed via pyproj:
#   Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
#   transform(-95.45, 29.70) → W~53203, S~736249
#   transform(-95.30, 29.80) → E~67631, N~747435
# (with margin so the mock tile comfortably contains the AOI)
_HOUSTON_5070_WEST = 51000.0
_HOUSTON_5070_SOUTH = 734000.0
_HOUSTON_5070_EAST = 70000.0
_HOUSTON_5070_NORTH = 750000.0

HOUSTON_UTM_AOI = AOI("houston", (-95.45, 29.70, -95.30, 29.80), 32615)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nlcd_tile_path(tmp_path: Path) -> Path:
    """A 32×32 NLCD tile in EPSG:5070 at 30m with imperviousness values 0–100."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 101, (32, 32), dtype=np.uint8)
    tile_path = tmp_path / "NLCD_2019_Impervious.tif"
    _write_nlcd_geotiff(
        tile_path,
        data,
        west=_HOUSTON_5070_WEST,
        south=_HOUSTON_5070_SOUTH,
        east=_HOUSTON_5070_EAST,
        north=_HOUSTON_5070_NORTH,
    )
    return tile_path


@pytest.fixture()
def nlcd_tile_with_nodata(tmp_path: Path) -> Path:
    """A 16×16 NLCD tile with some nodata (127) values."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 101, (16, 16), dtype=np.uint8)
    # Set some pixels as nodata (127)
    data[0, 0] = 127
    data[7, 7] = 127
    data[15, 15] = 127
    tile_path = tmp_path / "NLCD_2020_Impervious_nodata.tif"
    _write_nlcd_geotiff(
        tile_path,
        data,
        west=_HOUSTON_5070_WEST,
        south=_HOUSTON_5070_SOUTH,
        east=_HOUSTON_5070_EAST,
        north=_HOUSTON_5070_NORTH,
    )
    return tile_path


@pytest.fixture()
def nlcd_year_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two NLCD tiles (2019, 2020) with known change pattern for testing."""
    rng = np.random.default_rng(7)
    # t1: low imperviousness
    data_t1 = np.full((16, 16), 10, dtype=np.uint8)
    # t2: higher imperviousness in one quadrant (top-left 8x8)
    data_t2 = np.full((16, 16), 10, dtype=np.uint8)
    data_t2[:8, :8] = 50  # 40pp increase in top-left quadrant
    tile_2019 = tmp_path / "NLCD_2019_Impervious.tif"
    tile_2020 = tmp_path / "NLCD_2020_Impervious.tif"
    _write_nlcd_geotiff(
        tile_2019,
        data_t1,
        west=_HOUSTON_5070_WEST,
        south=_HOUSTON_5070_SOUTH,
        east=_HOUSTON_5070_EAST,
        north=_HOUSTON_5070_NORTH,
    )
    _write_nlcd_geotiff(
        tile_2020,
        data_t2,
        west=_HOUSTON_5070_WEST,
        south=_HOUSTON_5070_SOUTH,
        east=_HOUSTON_5070_EAST,
        north=_HOUSTON_5070_NORTH,
    )
    return tile_2019, tile_2020


# ---------------------------------------------------------------------------
# Tests: test_load_nlcd_tile
# ---------------------------------------------------------------------------


class TestLoadNLCDTile:
    """test_load_nlcd_tile: Verify CRS (Albers Equal Area), values 0–100."""

    def test_native_crs_is_epsg_5070(self, nlcd_tile_path: Path):
        """The raw NLCD tile must be in EPSG:5070 (Albers Equal Area)."""
        da = xr.open_dataarray(nlcd_tile_path, engine="rasterio")
        assert da.rio.crs is not None
        assert da.rio.crs.to_epsg() == 5070

    def test_values_in_range_0_to_100(self, nlcd_tile_path: Path):
        """Valid NLCD imperviousness values must be in [0, 100]."""
        da = xr.open_dataarray(nlcd_tile_path, engine="rasterio")
        values = da.values.astype(np.int32)
        valid = values[values != 127]  # exclude nodata
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_nodata_value_is_127(self, nlcd_tile_path: Path):
        """NLCD encoded_nodata value should be 127 (rioxarray decodes it to NaN)."""
        da = xr.open_dataarray(nlcd_tile_path, engine="rasterio")
        # rioxarray auto-converts nodata=127 to NaN on open (float32 output);
        # the original fill value is stored in encoded_nodata.
        assert da.rio.encoded_nodata == 127

    def test_load_returns_dataarray(self, nlcd_tile_path: Path, tmp_path: Path):
        """load_nlcd_impervious returns an xr.DataArray."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert isinstance(result, xr.DataArray)

    def test_load_has_crs(self, nlcd_tile_path: Path):
        """load_nlcd_impervious result has a CRS attached."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert result.rio.crs is not None

    def test_load_values_non_negative(self, nlcd_tile_path: Path):
        """After loading, non-NaN values should be >= 0."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        arr = result.values
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            assert valid.min() >= 0.0

    def test_load_values_at_most_100(self, nlcd_tile_path: Path):
        """After loading, non-NaN values should be <= 100."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        arr = result.values
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            assert valid.max() <= 100.0

    def test_nodata_pixels_become_nan(self, nlcd_tile_with_nodata: Path):
        """Pixels with value 127 (nodata) must become NaN in the output."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_with_nodata, HOUSTON_UTM_AOI)
        arr = result.values
        # Result must contain at least some NaN values (from the nodata pixels)
        assert np.any(np.isnan(arr)), "Expected NaN pixels from nodata=127 but found none"

    def test_load_result_is_float32(self, nlcd_tile_path: Path):
        """load_nlcd_impervious result should be float32."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: test_reproject_to_utm
# ---------------------------------------------------------------------------


class TestReprojectToUTM:
    """test_reproject_to_utm: To EPSG:32615 (Houston UTM)."""

    def test_output_crs_matches_aoi_epsg(self, nlcd_tile_path: Path):
        """After load_nlcd_impervious, CRS must match the AOI's EPSG (32615)."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert result.rio.crs.to_epsg() == HOUSTON_UTM_AOI.epsg

    def test_output_crs_is_not_native_5070(self, nlcd_tile_path: Path):
        """Source is EPSG:5070; output must be different (UTM, not Albers)."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert result.rio.crs.to_epsg() != 5070

    def test_reproject_to_houston_utm_32615(self, nlcd_tile_path: Path):
        """Explicitly verify reprojection to EPSG:32615 for Houston UTM zone 15N."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert result.rio.crs.to_epsg() == 32615

    def test_result_has_spatial_dimensions(self, nlcd_tile_path: Path):
        """Result must have x and y dimensions."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        assert "x" in result.dims or "longitude" in result.dims
        assert "y" in result.dims or "latitude" in result.dims


# ---------------------------------------------------------------------------
# Tests: test_resample_30m_to_10m
# ---------------------------------------------------------------------------


class TestResample30mTo10m:
    """test_resample_30m_to_10m: Bilinear resampling from 30m to 10m."""

    def test_output_resolution_is_10m(self, nlcd_tile_path: Path):
        """Output resolution must be approximately 10m after resampling."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        result = load_nlcd_impervious(nlcd_tile_path, HOUSTON_UTM_AOI)
        res = result.rio.resolution()
        # Resolution is (x_res, y_res); y_res is typically negative
        assert abs(res[0]) == pytest.approx(10.0, abs=1.0), (
            f"Expected ~10m x-resolution, got {abs(res[0]):.1f}m"
        )
        assert abs(res[1]) == pytest.approx(10.0, abs=1.0), (
            f"Expected ~10m y-resolution, got {abs(res[1]):.1f}m"
        )

    def test_output_has_more_pixels_than_30m_input(self, tmp_path: Path):
        """10m output should have ~9x more pixels than a 30m input of same area."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        rng = np.random.default_rng(99)
        data = rng.integers(0, 101, (10, 10), dtype=np.uint8)
        tile_path = tmp_path / "NLCD_30m.tif"
        _write_nlcd_geotiff(
            tile_path,
            data,
            west=_HOUSTON_5070_WEST,
            south=_HOUSTON_5070_SOUTH,
            east=_HOUSTON_5070_EAST,
            north=_HOUSTON_5070_NORTH,
        )

        result = load_nlcd_impervious(tile_path, HOUSTON_UTM_AOI)
        input_pixels = 10 * 10
        output_pixels = result.sizes.get("x", 0) * result.sizes.get("y", 0)
        assert output_pixels > input_pixels, (
            f"10m output ({output_pixels} px) should have more pixels than "
            f"30m input ({input_pixels} px)"
        )

    def test_bilinear_resampling_produces_intermediate_values(self, tmp_path: Path):
        """Bilinear resampling creates interpolated values between input pixel values."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious

        # Create a 2x2 tile with distinct corner values to force interpolation
        data = np.array([[0, 100], [0, 100]], dtype=np.uint8)
        tile_path = tmp_path / "NLCD_bilinear_test.tif"
        _write_nlcd_geotiff(
            tile_path,
            data,
            west=_HOUSTON_5070_WEST,
            south=_HOUSTON_5070_SOUTH,
            east=_HOUSTON_5070_EAST,
            north=_HOUSTON_5070_NORTH,
        )

        result = load_nlcd_impervious(tile_path, HOUSTON_UTM_AOI)
        arr = result.values
        valid = arr[~np.isnan(arr)]
        if len(valid) > 4:  # More pixels than input → resampling occurred
            # Bilinear interpolation: some pixels should have values between 0 and 100
            has_intermediate = np.any((valid > 0) & (valid < 100))
            assert has_intermediate, (
                "Expected bilinear interpolation to produce intermediate values"
            )


# ---------------------------------------------------------------------------
# Tests: test_annual_change_pairs
# ---------------------------------------------------------------------------


class TestAnnualChangePairs:
    """test_annual_change_pairs: 2019 vs 2020 density -> change labels."""

    def test_change_labels_produced_from_two_nlcd_years(self, nlcd_year_pair: tuple[Path, Path]):
        """Loading two years and calling generate_change_labels produces valid labels."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious
        from surface_change_monitor.labels.change import generate_change_labels

        tile_2019, tile_2020 = nlcd_year_pair
        density_2019 = load_nlcd_impervious(tile_2019, HOUSTON_UTM_AOI)
        density_2020 = load_nlcd_impervious(tile_2020, HOUSTON_UTM_AOI)

        labels = generate_change_labels(density_2019, density_2020, threshold=10.0)

        assert isinstance(labels, xr.DataArray)
        assert labels.dtype == np.uint8

    def test_changed_region_detected(self, nlcd_year_pair: tuple[Path, Path]):
        """The region with 40pp increase should be detected as changed."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious
        from surface_change_monitor.labels.change import generate_change_labels

        tile_2019, tile_2020 = nlcd_year_pair
        density_2019 = load_nlcd_impervious(tile_2019, HOUSTON_UTM_AOI)
        density_2020 = load_nlcd_impervious(tile_2020, HOUSTON_UTM_AOI)

        labels = generate_change_labels(density_2019, density_2020, threshold=10.0)

        valid = labels.values[~np.isnan(labels.values.astype(float))]
        # With 40pp increase in part of the tile, some pixels must be labelled change
        assert np.any(valid == 1), (
            "Expected at least some changed pixels (1) in labels, but found none"
        )

    def test_unchanged_region_is_zero(self, nlcd_year_pair: tuple[Path, Path]):
        """The region with 0pp change should remain 0 in the labels."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious
        from surface_change_monitor.labels.change import generate_change_labels

        tile_2019, tile_2020 = nlcd_year_pair
        density_2019 = load_nlcd_impervious(tile_2019, HOUSTON_UTM_AOI)
        density_2020 = load_nlcd_impervious(tile_2020, HOUSTON_UTM_AOI)

        labels = generate_change_labels(density_2019, density_2020, threshold=10.0)

        valid = labels.values[~np.isnan(labels.values.astype(float))]
        # Must have some no-change pixels (the bottom-right quadrant had no increase)
        assert np.any(valid == 0), (
            "Expected some unchanged pixels (0) in labels, but found none"
        )

    def test_labels_spatial_dimensions_match(self, nlcd_year_pair: tuple[Path, Path]):
        """Change labels should have same spatial dimensions as input density maps."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious
        from surface_change_monitor.labels.change import generate_change_labels

        tile_2019, tile_2020 = nlcd_year_pair
        density_2019 = load_nlcd_impervious(tile_2019, HOUSTON_UTM_AOI)
        density_2020 = load_nlcd_impervious(tile_2020, HOUSTON_UTM_AOI)

        labels = generate_change_labels(density_2019, density_2020, threshold=10.0)

        # Labels should have the same shape as the densities
        assert labels.shape == density_2019.shape, (
            f"Labels shape {labels.shape} != density shape {density_2019.shape}"
        )

    def test_labels_binary_values_only(self, nlcd_year_pair: tuple[Path, Path]):
        """Change labels must only contain 0 or 1 values (binary mask)."""
        from surface_change_monitor.labels.nlcd import load_nlcd_impervious
        from surface_change_monitor.labels.change import generate_change_labels

        tile_2019, tile_2020 = nlcd_year_pair
        density_2019 = load_nlcd_impervious(tile_2019, HOUSTON_UTM_AOI)
        density_2020 = load_nlcd_impervious(tile_2020, HOUSTON_UTM_AOI)

        labels = generate_change_labels(density_2019, density_2020, threshold=10.0)

        unique_values = np.unique(labels.values)
        assert set(unique_values).issubset({0, 1}), (
            f"Labels contain non-binary values: {unique_values}"
        )
