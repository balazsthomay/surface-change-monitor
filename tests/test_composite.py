"""Tests for the monthly compositing module.

Tests cover:
  - Median compositing with known values
  - NaN exclusion from cloud masking
  - All-NaN pixel handling
  - Reliability flagging (< 3 clear scenes)
  - Per-pixel clear observation count
  - Georeference preservation
  - Scene grouping by month
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from surface_change_monitor.config import AOI, BERGEN_AOI


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_scene_metadata(scene_id: str, dt: datetime, cloud_cover: float = 5.0):
    """Create a minimal SceneMetadata for grouping tests."""
    from surface_change_monitor.stac import SceneMetadata

    return SceneMetadata(
        scene_id=scene_id,
        datetime=dt,
        cloud_cover=cloud_cover,
        product_id=f"{scene_id}.SAFE",
        tile_id="T32VNM",
        geometry={"type": "Polygon", "coordinates": []},
        assets={},
    )


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    crs_epsg: int = 32632,
    west: float = 297000.0,
    south: float = 6690000.0,
    east: float = 297200.0,
    north: float = 6690200.0,
) -> Path:
    """Write a single-band or multi-band uint16 GeoTIFF to *path*."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # add band dim
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
    ) as dst:
        dst.write(data)
    return path


def _make_band_paths(
    tmp_path: Path,
    scene_idx: int,
    bands: dict[str, np.ndarray],
    crs_epsg: int = 32632,
    spatial_extent: tuple[float, float, float, float] = (297000.0, 6690000.0, 297200.0, 6690200.0),
) -> dict[str, Path]:
    """Write each band array to a separate GeoTIFF and return a name->Path dict."""
    scene_dir = tmp_path / f"scene_{scene_idx:02d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = spatial_extent
    paths: dict[str, Path] = {}
    for band_name, arr in bands.items():
        p = scene_dir / f"{band_name}.tif"
        _write_geotiff(p, arr, crs_epsg=crs_epsg, west=west, south=south, east=east, north=north)
        paths[band_name] = p
    return paths


def _all_clear_scl(shape: tuple[int, int]) -> np.ndarray:
    """Return an SCL array where all pixels are clear (vegetation=4)."""
    return np.full(shape, 4, dtype=np.uint8)


def _all_cloudy_scl(shape: tuple[int, int]) -> np.ndarray:
    """Return an SCL array where all pixels are cloud (high prob=9)."""
    return np.full(shape, 9, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 6.1a  test_group_scenes_by_month
# ---------------------------------------------------------------------------


class TestGroupScenesByMonth:
    def test_group_scenes_by_month_basic(self):
        """Scenes in different months are assigned to separate groups."""
        from surface_change_monitor.composite import group_scenes_by_month

        scenes = [
            _make_scene_metadata("s1", datetime(2024, 6, 1, tzinfo=timezone.utc)),
            _make_scene_metadata("s2", datetime(2024, 6, 15, tzinfo=timezone.utc)),
            _make_scene_metadata("s3", datetime(2024, 7, 1, tzinfo=timezone.utc)),
        ]
        groups = group_scenes_by_month(scenes)

        assert "2024-06" in groups
        assert "2024-07" in groups
        assert len(groups["2024-06"]) == 2
        assert len(groups["2024-07"]) == 1

    def test_group_scenes_by_month_single_month(self):
        """All scenes in same month produce exactly one group."""
        from surface_change_monitor.composite import group_scenes_by_month

        scenes = [
            _make_scene_metadata("a", datetime(2024, 1, 5, tzinfo=timezone.utc)),
            _make_scene_metadata("b", datetime(2024, 1, 20, tzinfo=timezone.utc)),
        ]
        groups = group_scenes_by_month(scenes)

        assert list(groups.keys()) == ["2024-01"]
        assert len(groups["2024-01"]) == 2

    def test_group_scenes_by_month_empty_list(self):
        """Empty scene list produces empty dict."""
        from surface_change_monitor.composite import group_scenes_by_month

        assert group_scenes_by_month([]) == {}

    def test_group_scenes_by_month_multi_year(self):
        """Scenes spanning year boundary are grouped correctly by YYYY-MM."""
        from surface_change_monitor.composite import group_scenes_by_month

        scenes = [
            _make_scene_metadata("x", datetime(2023, 12, 25, tzinfo=timezone.utc)),
            _make_scene_metadata("y", datetime(2024, 1, 3, tzinfo=timezone.utc)),
        ]
        groups = group_scenes_by_month(scenes)

        assert "2023-12" in groups
        assert "2024-01" in groups


# ---------------------------------------------------------------------------
# 6.1b  test_monthly_median_basic
# ---------------------------------------------------------------------------


class TestMonthlyMedianBasic:
    def test_monthly_median_basic(self, tmp_path):
        """Three scenes with known values produce the expected median composite."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (4, 4)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_values = [100, 200, 300]  # expected median = 200
        band_paths_list = []
        for i, val in enumerate(band_values):
            arr = np.full(shape, val, dtype=np.uint16)
            paths = _make_band_paths(
                tmp_path,
                i,
                {"B04": arr, "SCL": _all_clear_scl(shape)},
            )
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        # The median of [100, 200, 300] = 200 everywhere
        assert composite.year_month == "2024-06"
        medians = composite.data.values  # (bands, H, W) or (H, W)
        assert np.allclose(medians, 200.0, atol=1e-3), (
            f"Expected median=200 everywhere, got {medians}"
        )


# ---------------------------------------------------------------------------
# 6.1c  test_ignores_nan
# ---------------------------------------------------------------------------


class TestIgnoresNan:
    def test_ignores_nan(self, tmp_path):
        """Cloud-masked (NaN) values are excluded; median uses only clear pixels."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        # Scene 0: band=100, all clear
        # Scene 1: band=200, all clear
        # Scene 2: band=999, all cloudy -> should be excluded
        scenarios = [
            (100, _all_clear_scl(shape)),
            (200, _all_clear_scl(shape)),
            (999, _all_cloudy_scl(shape)),
        ]
        band_paths_list = []
        for i, (val, scl) in enumerate(scenarios):
            arr = np.full(shape, val, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": scl})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-07")

        # median of [100, 200] = 150 (nan-safe)
        medians = composite.data.values
        assert np.allclose(medians, 150.0, atol=1e-3), (
            f"Expected median=150 after ignoring NaN, got {medians}"
        )


# ---------------------------------------------------------------------------
# 6.1d  test_all_nan_pixel
# ---------------------------------------------------------------------------


class TestAllNanPixel:
    def test_all_nan_pixel(self, tmp_path):
        """A pixel that is cloudy in all scenes becomes NaN in the composite."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        # All 3 scenes are fully cloudy
        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 500, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_cloudy_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-08")

        medians = composite.data.values
        assert np.all(np.isnan(medians)), (
            f"Expected all-NaN composite when all scenes are cloudy, got {medians}"
        )


# ---------------------------------------------------------------------------
# 6.1e  test_flag_unreliable_month
# ---------------------------------------------------------------------------


class TestFlagUnreliableMonth:
    def test_flag_unreliable_month_few_scenes(self, tmp_path):
        """Fewer than 3 scenes -> reliable=False."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(2):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-09")

        assert composite.reliable is False

    def test_flag_reliable_month_three_scenes(self, tmp_path):
        """Exactly 3 scenes -> reliable=True."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-10")

        assert composite.reliable is True

    def test_flag_reliable_month_more_than_three(self, tmp_path):
        """More than 3 scenes -> reliable=True."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(5):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-11")

        assert composite.reliable is True


# ---------------------------------------------------------------------------
# 6.1f  test_count_clear_obs_per_pixel
# ---------------------------------------------------------------------------


class TestCountClearObsPerPixel:
    def test_count_clear_obs_per_pixel(self, tmp_path):
        """clear_obs_count accurately reflects per-pixel clear observation count."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        # Create scenes with mixed clear/cloudy pixels
        # Use a 2x2 grid where:
        #   - All 3 scenes contribute clear obs at pixel (0,0) and (0,1)
        #   - Only 1 scene contributes at (1,0) and (1,1) (other 2 are cloudy)
        scl_clear = np.array([[4, 4], [4, 4]], dtype=np.uint8)   # all clear
        scl_partial = np.array([[4, 4], [9, 9]], dtype=np.uint8)  # top row clear, bottom cloudy

        scenarios = [
            (scl_clear,),    # scene 0: all clear
            (scl_partial,),  # scene 1: top row clear
            (scl_partial,),  # scene 2: top row clear
        ]
        band_paths_list = []
        for i, (scl,) in enumerate(scenarios):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": scl})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-12")

        counts = composite.clear_obs_count.values
        assert counts[0, 0] == 3, f"Top-left pixel should have 3 clear obs, got {counts[0,0]}"
        assert counts[0, 1] == 3, f"Top-right pixel should have 3 clear obs, got {counts[0,1]}"
        assert counts[1, 0] == 1, f"Bottom-left pixel should have 1 clear obs, got {counts[1,0]}"
        assert counts[1, 1] == 1, f"Bottom-right pixel should have 1 clear obs, got {counts[1,1]}"


# ---------------------------------------------------------------------------
# 6.1g  test_preserves_georeference
# ---------------------------------------------------------------------------


class TestPreservesGeoreference:
    def test_preserves_georeference(self, tmp_path):
        """CRS, transform and bounds survive compositing unchanged."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (4, 4)
        west, south, east, north = 297000.0, 6690000.0, 297040.0, 6690040.0
        crs_epsg = 32632
        aoi = AOI("test_geo", (5.27, 60.35, 5.30, 60.37), crs_epsg)

        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(
                tmp_path,
                i,
                {"B04": arr, "SCL": _all_clear_scl(shape)},
                crs_epsg=crs_epsg,
                spatial_extent=(west, south, east, north),
            )
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        # CRS must be present and correct
        crs = composite.data.rio.crs
        assert crs is not None, "Composite must have a CRS"
        assert crs.to_epsg() == crs_epsg, f"Expected EPSG:{crs_epsg}, got {crs.to_epsg()}"

        # Bounds should roughly match (within a pixel width)
        bounds = composite.data.rio.bounds()
        assert abs(bounds[0] - west) < 20, f"West bound mismatch: {bounds[0]} vs {west}"
        assert abs(bounds[3] - north) < 20, f"North bound mismatch: {bounds[3]} vs {north}"


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------


class TestMonthlyCompositeStructure:
    def test_composite_data_shape_single_band(self, tmp_path):
        """Composite data array has shape (1, H, W) for a single band."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (3, 3)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        # shape: (n_spectral_bands, H, W)
        assert composite.data.ndim == 3, "Composite data must be 3D (bands, H, W)"
        assert composite.data.shape[1] == shape[0]
        assert composite.data.shape[2] == shape[1]

    def test_composite_data_dtype_float32(self, tmp_path):
        """Composite data is stored as float32."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        assert composite.data.dtype == np.float32, (
            f"Expected float32, got {composite.data.dtype}"
        )

    def test_composite_n_scenes_matches_input(self, tmp_path):
        """n_scenes reflects the number of band_paths dicts passed in."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(4):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        assert composite.n_scenes == 4

    def test_composite_stores_aoi(self, tmp_path):
        """The returned MonthlyComposite carries the AOI passed in."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("bergen", (5.27, 60.35, 5.30, 60.37), 32632)

        band_paths_list = []
        for i in range(3):
            arr = np.full(shape, 100, dtype=np.uint16)
            paths = _make_band_paths(tmp_path, i, {"B04": arr, "SCL": _all_clear_scl(shape)})
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        assert composite.aoi is aoi

    def test_composite_multi_band(self, tmp_path):
        """Multiple spectral bands (B02, B03, B04) are all composited."""
        from surface_change_monitor.composite import create_monthly_composite

        shape = (2, 2)
        aoi = AOI("test", (5.27, 60.35, 5.30, 60.37), 32632)

        band_names = ["B02", "B03", "B04"]
        band_paths_list = []
        for i in range(3):
            bands: dict[str, np.ndarray] = {
                b: np.full(shape, (i + 1) * 100, dtype=np.uint16) for b in band_names
            }
            bands["SCL"] = _all_clear_scl(shape)
            paths = _make_band_paths(tmp_path, i, bands)
            band_paths_list.append(paths)

        composite = create_monthly_composite(band_paths_list, aoi, "2024-06")

        # 3 spectral bands -> first dim = 3
        assert composite.data.shape[0] == 3, (
            f"Expected 3 spectral bands in composite, got {composite.data.shape[0]}"
        )
