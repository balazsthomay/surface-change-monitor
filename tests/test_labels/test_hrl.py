"""Tests for the HRL (High Resolution Layer) tile loading and processing module.

Tests cover:
  - Loading a mock GeoTIFF in EPSG:3035 with values 0–100
  - Reprojecting from EPSG:3035 to target UTM CRS
  - Clipping to AOI bounds
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401 – registers .rio accessor
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from surface_change_monitor.config import AOI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hrl_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    crs_epsg: int = 3035,
    nodata: float | None = 255,
) -> Path:
    """Write a uint8 GeoTIFF that mimics an HRL imperviousness tile."""
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


# Bergen AOI expressed in EPSG:3035 projected coordinates.
# Computed via pyproj: Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
# transform(5.20, 60.30) → W~4055088, S~4142283
# transform(5.50, 60.50) → E~4073173, N~4163370
# (with margin so the mock tile comfortably contains the AOI)
_BERGEN_3035_WEST = 4055000.0
_BERGEN_3035_SOUTH = 4142000.0
_BERGEN_3035_EAST = 4074000.0
_BERGEN_3035_NORTH = 4164000.0

BERGEN_UTM_AOI = AOI("bergen", (5.27, 60.35, 5.40, 60.44), 32632)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hrl_tile_path(tmp_path: Path) -> Path:
    """A 32×32 HRL tile in EPSG:3035 with imperviousness values 0–100."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 101, (32, 32), dtype=np.uint8)
    tile_path = tmp_path / "IMD_2018.tif"
    _write_hrl_geotiff(
        tile_path,
        data,
        west=_BERGEN_3035_WEST,
        south=_BERGEN_3035_SOUTH,
        east=_BERGEN_3035_EAST,
        north=_BERGEN_3035_NORTH,
    )
    return tile_path


@pytest.fixture()
def small_aoi_3035(tmp_path: Path) -> tuple[Path, AOI]:
    """A 16×16 HRL tile plus an AOI that covers half of it in x and all in y."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 101, (16, 16), dtype=np.uint8)
    tile_path = tmp_path / "IMD_clip_test.tif"
    _write_hrl_geotiff(
        tile_path,
        data,
        west=_BERGEN_3035_WEST,
        south=_BERGEN_3035_SOUTH,
        east=_BERGEN_3035_EAST,
        north=_BERGEN_3035_NORTH,
    )
    # AOI slightly smaller than the tile to confirm clipping happens
    aoi = AOI("bergen_small", (5.27, 60.35, 5.40, 60.44), 32632)
    return tile_path, aoi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadHRLTile:
    """test_load_hrl_tile: GeoTIFF in EPSG:3035 with values 0–100."""

    def test_crs_is_epsg_3035(self, hrl_tile_path: Path):
        """The raw HRL tile must be in EPSG:3035 (verified before any reprojection)."""
        da = xr.open_dataarray(hrl_tile_path, engine="rasterio")
        assert da.rio.crs is not None
        assert da.rio.crs.to_epsg() == 3035

    def test_values_in_range_0_to_100(self, hrl_tile_path: Path):
        """Imperviousness density values must be in [0, 100] (uint8 0–100 valid range)."""
        da = xr.open_dataarray(hrl_tile_path, engine="rasterio")
        values = da.values.astype(np.int32)
        # Exclude nodata (255)
        valid = values[values != 255]
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_load_hrl_density_returns_dataarray(self, hrl_tile_path: Path):
        """load_hrl_density returns an xr.DataArray."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)
        assert isinstance(result, xr.DataArray)

    def test_load_hrl_density_has_crs(self, hrl_tile_path: Path):
        """load_hrl_density result has a CRS attached."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)
        assert result.rio.crs is not None

    def test_load_hrl_density_values_non_negative(self, hrl_tile_path: Path):
        """After loading, non-NaN values should be >= 0."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)
        arr = result.values
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            assert valid.min() >= 0.0


class TestReprojectToUTM:
    """test_reproject_to_utm: Reprojection from EPSG:3035 to target UTM."""

    def test_output_crs_matches_aoi_epsg(self, hrl_tile_path: Path):
        """After load_hrl_density, the CRS must match the AOI's EPSG."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)
        assert result.rio.crs.to_epsg() == BERGEN_UTM_AOI.epsg

    def test_input_crs_differs_from_output_crs(self, hrl_tile_path: Path):
        """Source is 3035; output must be the UTM CRS (32632), not 3035."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)
        assert result.rio.crs.to_epsg() != 3035

    def test_reprojection_to_different_utm_zones(self, tmp_path: Path):
        """Works for Houston AOI in UTM zone 15N (EPSG:32615)."""
        from surface_change_monitor.config import HOUSTON_AOI
        from surface_change_monitor.labels.hrl import load_hrl_density

        rng = np.random.default_rng(42)
        data = rng.integers(0, 101, (16, 16), dtype=np.uint8)
        # Houston in EPSG:3035: W~-2457869, N~7152931, E~-2430508, S~7116748
        # Note: LAEA south > north for North America (Y axis inverted outside EU).
        # from_bounds expects (west, south, east, north) where south < north.
        tile_path = tmp_path / "IMD_houston.tif"
        _write_hrl_geotiff(
            tile_path,
            data,
            west=-2457869.0,
            south=7116748.0,  # smaller y
            east=-2430508.0,
            north=7152931.0,  # larger y
        )
        result = load_hrl_density(tile_path, HOUSTON_AOI)
        assert result.rio.crs.to_epsg() == 32615


class TestClipToAOI:
    """test_clip_to_aoi: Verify clipping to AOI bounds."""

    def test_clipped_result_is_smaller_or_equal_to_original(self, hrl_tile_path: Path):
        """Clipping can only reduce (or keep equal) the spatial extent."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        raw = xr.open_dataarray(hrl_tile_path, engine="rasterio").squeeze()
        result = load_hrl_density(hrl_tile_path, BERGEN_UTM_AOI)

        # Pixel count in the result may be different (due to reprojection + clip),
        # but the bounds should not exceed the AOI.
        result_bounds = result.rio.bounds()
        # AOI bbox is in WGS84; convert to UTM to compare properly
        from pyproj import Transformer

        transformer = Transformer.from_crs(4326, BERGEN_UTM_AOI.epsg, always_xy=True)
        west, south = transformer.transform(BERGEN_UTM_AOI.bbox[0], BERGEN_UTM_AOI.bbox[1])
        east, north = transformer.transform(BERGEN_UTM_AOI.bbox[2], BERGEN_UTM_AOI.bbox[3])

        # Allow 2-pixel tolerance to account for reprojection pixel snapping.
        # Mock tiles have coarse pixels (~600–700 m), so tolerance must be >= 1 pixel.
        tolerance = 1500.0
        assert result_bounds[0] >= west - tolerance, (
            f"Left bound {result_bounds[0]:.0f} exceeds AOI west {west:.0f}"
        )
        assert result_bounds[1] >= south - tolerance, (
            f"Bottom bound {result_bounds[1]:.0f} exceeds AOI south {south:.0f}"
        )
        assert result_bounds[2] <= east + tolerance, (
            f"Right bound {result_bounds[2]:.0f} exceeds AOI east {east:.0f}"
        )
        assert result_bounds[3] <= north + tolerance, (
            f"Top bound {result_bounds[3]:.0f} exceeds AOI north {north:.0f}"
        )

    def test_clip_reduces_spatial_extent_when_tile_is_larger(self, tmp_path: Path):
        """When the HRL tile extends well beyond the AOI, clipping reduces the extent."""
        from surface_change_monitor.labels.hrl import load_hrl_density

        # Very large tile: extends 50 km in each direction beyond the Bergen AOI
        rng = np.random.default_rng(7)
        data = rng.integers(0, 101, (64, 64), dtype=np.uint8)
        big_tile = tmp_path / "IMD_big.tif"
        _write_hrl_geotiff(
            big_tile,
            data,
            west=_BERGEN_3035_WEST - 50000.0,
            south=_BERGEN_3035_SOUTH - 50000.0,
            east=_BERGEN_3035_EAST + 50000.0,
            north=_BERGEN_3035_NORTH + 50000.0,
        )

        result = load_hrl_density(big_tile, BERGEN_UTM_AOI)
        result_bounds = result.rio.bounds()

        from pyproj import Transformer

        transformer = Transformer.from_crs(4326, BERGEN_UTM_AOI.epsg, always_xy=True)
        west, south = transformer.transform(BERGEN_UTM_AOI.bbox[0], BERGEN_UTM_AOI.bbox[1])
        east, north = transformer.transform(BERGEN_UTM_AOI.bbox[2], BERGEN_UTM_AOI.bbox[3])

        # The clipped result must not extend more than 2 pixels beyond the AOI.
        # Mock tiles have coarse resolution (~600–700 m/pixel), so use 1500 m tolerance.
        tolerance = 1500.0
        assert result_bounds[0] >= west - tolerance
        assert result_bounds[3] <= north + tolerance
