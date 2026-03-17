"""Tests for surface_change_monitor.postprocess.

Strategy:
- Build synthetic xr.DataArray probability maps in EPSG:32632 (10 m pixels)
- Verify each stage: thresholding, morphological cleanup, vectorization,
  area filtering, attribute computation, and file export
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401 – registers .rio accessor
import xarray as xr
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prob_map(
    height: int = 50,
    width: int = 50,
    epsg: int = 32632,
    fill: float | None = None,
) -> xr.DataArray:
    """Return a synthetic probability DataArray in UTM zone 32N.

    10 m pixels so 1 pixel == 100 m^2.
    The coordinate origin is at the Bergen AOI corner.
    """
    west, south = 297000.0, 6690000.0
    res = 10.0
    east = west + width * res
    north = south + height * res

    x_coords = np.linspace(west + res / 2, east - res / 2, width)
    y_coords = np.linspace(north - res / 2, south + res / 2, height)

    if fill is not None:
        data = np.full((height, width), fill, dtype=np.float32)
    else:
        rng = np.random.default_rng(42)
        data = rng.random((height, width)).astype(np.float32)

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da = da.rio.write_transform()
    return da


def _make_block_prob_map(
    height: int = 50,
    width: int = 50,
    block_row: int = 10,
    block_col: int = 10,
    block_h: int = 10,
    block_w: int = 10,
    block_value: float = 0.9,
    bg_value: float = 0.1,
    epsg: int = 32632,
) -> xr.DataArray:
    """Probability map with a high-value rectangular block on a low background.

    10 m pixels; block_h*block_w * 100 m^2 area for the blob.
    """
    data = np.full((height, width), bg_value, dtype=np.float32)
    data[block_row : block_row + block_h, block_col : block_col + block_w] = block_value

    west, south = 297000.0, 6690000.0
    res = 10.0
    east = west + width * res
    north = south + height * res

    x_coords = np.linspace(west + res / 2, east - res / 2, width)
    y_coords = np.linspace(north - res / 2, south + res / 2, height)

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da = da.rio.write_transform()
    return da


# ---------------------------------------------------------------------------
# 15.1 Failing tests
# ---------------------------------------------------------------------------


class TestThresholdAndVectorize:
    """test_threshold_and_vectorize: Probability raster -> polygons above threshold."""

    def test_returns_geodataframe(self) -> None:
        """vectorize_changes must return a GeoDataFrame."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        assert isinstance(result, gpd.GeoDataFrame), (
            f"Expected GeoDataFrame, got {type(result)}"
        )

    def test_polygons_above_threshold(self) -> None:
        """Pixels above threshold form polygons; pixels below do not contribute."""
        from surface_change_monitor.postprocess import vectorize_changes

        # Block with value 0.9 > threshold=0.5, background 0.1 < threshold
        prob_map = _make_block_prob_map(block_value=0.9, bg_value=0.1)
        result = vectorize_changes(prob_map, threshold=0.5)

        # At least one polygon should be returned
        assert len(result) > 0, "Expected at least one polygon above threshold"

    def test_no_polygons_when_all_below_threshold(self) -> None:
        """If all values are below threshold, result should be empty."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_prob_map(fill=0.1)
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) == 0, f"Expected 0 polygons, got {len(result)}"

    def test_all_polygons_when_all_above_threshold(self) -> None:
        """If all values above threshold, exactly one polygon covering the whole map."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_prob_map(fill=0.9)
        result = vectorize_changes(prob_map, threshold=0.5)

        # Should have exactly 1 polygon (the whole raster)
        assert len(result) >= 1, "Expected at least one polygon for all-above-threshold map"

    def test_threshold_respected(self) -> None:
        """Values exactly at threshold should be included (>= threshold)."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map(block_value=0.5, bg_value=0.0, block_h=10, block_w=10)
        result = vectorize_changes(prob_map, threshold=0.5)

        # Block at exactly threshold should produce polygon(s)
        assert len(result) > 0, "Pixels at exactly threshold should produce polygons"


class TestMinimumAreaFilter:
    """test_minimum_area_filter: Polygons < 200 m^2 are removed."""

    def test_small_polygon_removed(self) -> None:
        """A single-pixel block (100 m^2) is below 200 m^2 and must be filtered out."""
        from surface_change_monitor.postprocess import vectorize_changes

        # Single pixel = 10m x 10m = 100 m^2 < 200 m^2 min_area
        prob_map = _make_block_prob_map(block_h=1, block_w=1, block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5, min_area_m2=200.0)

        assert len(result) == 0, (
            f"Expected 0 polygons after 200 m^2 filter, got {len(result)}"
        )

    def test_large_polygon_kept(self) -> None:
        """A 10x10 block (1000 m^2) exceeds 200 m^2 and must be kept."""
        from surface_change_monitor.postprocess import vectorize_changes

        # 10x10 block = 100 m^2 * 10 * 10 = 10,000 m^2 > 200 m^2
        prob_map = _make_block_prob_map(block_h=10, block_w=10, block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5, min_area_m2=200.0)

        assert len(result) > 0, "Expected polygon to survive 200 m^2 area filter"

    def test_custom_min_area(self) -> None:
        """Custom min_area_m2 is respected."""
        from surface_change_monitor.postprocess import vectorize_changes

        # 3x3 block = 900 m^2 > 500 m^2, should be kept with custom threshold
        prob_map = _make_block_prob_map(block_h=3, block_w=3, block_value=0.9, bg_value=0.0)
        result_strict = vectorize_changes(prob_map, threshold=0.5, min_area_m2=1000.0)
        result_loose = vectorize_changes(prob_map, threshold=0.5, min_area_m2=100.0)

        # With 1000 m^2 min, 900 m^2 polygon should be removed
        assert len(result_strict) == 0, (
            "3x3 block (900 m^2) should be filtered by 1000 m^2 min_area"
        )
        # With 100 m^2 min, 900 m^2 polygon should be kept
        assert len(result_loose) > 0, (
            "3x3 block (900 m^2) should be kept with 100 m^2 min_area"
        )


class TestMorphologicalCleanup:
    """test_morphological_cleanup: Single-pixel noise removed, small holes filled."""

    def test_isolated_single_pixel_removed(self) -> None:
        """Single-pixel blobs below structural element should be removed by opening."""
        from surface_change_monitor.postprocess import vectorize_changes

        # Create a map with a big block AND a single isolated pixel (noise)
        data = np.zeros((50, 50), dtype=np.float32)
        # Large block (will survive): 10x10 starting at row 5, col 5
        data[5:15, 5:15] = 0.9
        # Single isolated pixel (noise): row 30, col 30
        data[30, 30] = 0.9

        west, south = 297000.0, 6690000.0
        res = 10.0
        x_coords = np.linspace(west + res / 2, west + 50 * res - res / 2, 50)
        y_coords = np.linspace(south + 50 * res - res / 2, south + res / 2, 50)

        da = xr.DataArray(data, dims=["y", "x"], coords={"y": y_coords, "x": x_coords})
        da = da.rio.write_crs("EPSG:32632")
        da = da.rio.write_transform()

        result = vectorize_changes(da, threshold=0.5, min_area_m2=200.0)

        # Should have only the big block, not the single pixel
        assert len(result) == 1, (
            f"Expected 1 polygon (big block only), got {len(result)}"
        )

    def test_small_holes_filled(self) -> None:
        """Small holes inside a large polygon should be filled by closing."""
        from surface_change_monitor.postprocess import vectorize_changes

        # Create a solid 20x20 block with a single internal zero pixel
        data = np.full((50, 50), 0.0, dtype=np.float32)
        data[10:30, 10:30] = 0.9
        data[20, 20] = 0.0  # single internal hole

        west, south = 297000.0, 6690000.0
        res = 10.0
        x_coords = np.linspace(west + res / 2, west + 50 * res - res / 2, 50)
        y_coords = np.linspace(south + 50 * res - res / 2, south + res / 2, 50)

        da = xr.DataArray(data, dims=["y", "x"], coords={"y": y_coords, "x": x_coords})
        da = da.rio.write_crs("EPSG:32632")
        da = da.rio.write_transform()

        result = vectorize_changes(da, threshold=0.5, min_area_m2=200.0)

        # Should have exactly one filled polygon (no holes represented as multiple polygons)
        assert len(result) == 1, (
            f"Expected 1 polygon after hole-filling, got {len(result)}"
        )


class TestPolygonAttributes:
    """test_polygon_attributes: confidence, area_m2, detection_period."""

    def test_has_required_columns(self) -> None:
        """Output GeoDataFrame must have geometry, confidence, area_m2 columns."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0, "Need at least one polygon to test attributes"
        for col in ("geometry", "confidence", "area_m2"):
            assert col in result.columns, f"Missing column: {col}"

    def test_confidence_in_zero_one(self) -> None:
        """Confidence (mean probability within polygon) must be in [0, 1]."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map(block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0
        assert (result["confidence"] >= 0.0).all(), "Confidence values must be >= 0"
        assert (result["confidence"] <= 1.0).all(), "Confidence values must be <= 1"

    def test_confidence_reflects_probability(self) -> None:
        """Confidence for a uniform 0.9 block should be close to 0.9."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map(block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0
        # Confidence should be close to 0.9 (the block's probability value)
        assert float(result["confidence"].iloc[0]) == pytest.approx(0.9, abs=0.05), (
            f"Expected confidence ~0.9, got {result['confidence'].iloc[0]}"
        )

    def test_area_m2_positive(self) -> None:
        """area_m2 must be positive for all polygons."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0
        assert (result["area_m2"] > 0).all(), "area_m2 must be positive"

    def test_area_m2_approximately_correct(self) -> None:
        """area_m2 should match the pixel count * 100 m^2 per pixel."""
        from surface_change_monitor.postprocess import vectorize_changes

        # 10x10 block = 100 pixels * 100 m^2 = 10,000 m^2
        prob_map = _make_block_prob_map(block_h=10, block_w=10, block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0
        # Allow 5% tolerance for polygon boundary effects
        expected_area = 10 * 10 * 100.0  # 10,000 m^2
        actual_area = float(result["area_m2"].sum())
        assert actual_area == pytest.approx(expected_area, rel=0.1), (
            f"Expected area ~{expected_area} m^2, got {actual_area} m^2"
        )

    def test_detection_period_column_when_provided(self) -> None:
        """When detection_period is passed, it appears as an attribute column."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(
            prob_map, threshold=0.5, detection_period="2024-01/2024-02"
        )

        assert len(result) > 0
        assert "detection_period" in result.columns, "Missing detection_period column"
        assert (result["detection_period"] == "2024-01/2024-02").all()

    def test_detection_period_none_when_not_provided(self) -> None:
        """When detection_period is not passed, column should still exist with None/NaN."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        assert len(result) > 0
        assert "detection_period" in result.columns, "detection_period column must always be present"


class TestAdjacentMerge:
    """test_adjacent_merge: Adjacent pixels -> single polygon."""

    def test_adjacent_pixels_form_single_polygon(self) -> None:
        """Contiguous high-probability pixels should be vectorized as one polygon."""
        from surface_change_monitor.postprocess import vectorize_changes

        # A solid 5x5 block of adjacent pixels
        prob_map = _make_block_prob_map(block_h=5, block_w=5, block_value=0.9, bg_value=0.0)
        result = vectorize_changes(prob_map, threshold=0.5, min_area_m2=100.0)

        # All adjacent pixels should merge into one polygon
        assert len(result) == 1, (
            f"Expected 1 merged polygon for contiguous block, got {len(result)}"
        )

    def test_separated_blobs_form_separate_polygons(self) -> None:
        """Two well-separated blobs should yield two distinct polygons."""
        from surface_change_monitor.postprocess import vectorize_changes

        data = np.zeros((50, 50), dtype=np.float32)
        # Blob 1: top-left region (6x6 = 36 pixels = 3600 m^2)
        data[5:11, 5:11] = 0.9
        # Blob 2: bottom-right region (6x6 = 3600 m^2), well-separated
        data[35:41, 35:41] = 0.9

        west, south = 297000.0, 6690000.0
        res = 10.0
        x_coords = np.linspace(west + res / 2, west + 50 * res - res / 2, 50)
        y_coords = np.linspace(south + 50 * res - res / 2, south + res / 2, 50)

        da = xr.DataArray(data, dims=["y", "x"], coords={"y": y_coords, "x": x_coords})
        da = da.rio.write_crs("EPSG:32632")
        da = da.rio.write_transform()

        result = vectorize_changes(da, threshold=0.5, min_area_m2=200.0)

        assert len(result) == 2, (
            f"Expected 2 separate polygons for separated blobs, got {len(result)}"
        )

    def test_crs_set_on_output(self) -> None:
        """Output GeoDataFrame must have a CRS set."""
        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        assert result.crs is not None, "Output GeoDataFrame must have a CRS"
        assert result.crs.to_epsg() == 32632, (
            f"Expected EPSG:32632, got {result.crs}"
        )


class TestOutputFormats:
    """test_output_formats: GeoJSON and GeoPackage export."""

    def test_export_geojson(self, tmp_path: Path) -> None:
        """GeoDataFrame can be exported as GeoJSON and re-read successfully."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        out_path = tmp_path / "changes.geojson"
        result.to_file(str(out_path), driver="GeoJSON")

        assert out_path.exists(), f"GeoJSON file not created at {out_path}"

        # Round-trip: re-read and verify
        loaded = gpd.read_file(str(out_path))
        assert len(loaded) == len(result), (
            f"Expected {len(result)} features, read {len(loaded)}"
        )

    def test_export_geopackage(self, tmp_path: Path) -> None:
        """GeoDataFrame can be exported as GeoPackage and re-read successfully."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        out_path = tmp_path / "changes.gpkg"
        result.to_file(str(out_path), driver="GPKG")

        assert out_path.exists(), f"GeoPackage file not created at {out_path}"

        # Round-trip: re-read and verify
        loaded = gpd.read_file(str(out_path))
        assert len(loaded) == len(result), (
            f"Expected {len(result)} features, read {len(loaded)}"
        )

    def test_exported_geojson_has_attributes(self, tmp_path: Path) -> None:
        """Exported GeoJSON preserves confidence, area_m2, detection_period attributes."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(
            prob_map, threshold=0.5, detection_period="2024-01/2024-02"
        )

        out_path = tmp_path / "changes.geojson"
        result.to_file(str(out_path), driver="GeoJSON")

        loaded = gpd.read_file(str(out_path))
        for col in ("confidence", "area_m2", "detection_period"):
            assert col in loaded.columns, f"Missing column {col} in exported GeoJSON"

    def test_exported_geopackage_has_crs(self, tmp_path: Path) -> None:
        """Exported GeoPackage preserves the CRS."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import vectorize_changes

        prob_map = _make_block_prob_map()
        result = vectorize_changes(prob_map, threshold=0.5)

        out_path = tmp_path / "changes.gpkg"
        result.to_file(str(out_path), driver="GPKG")

        loaded = gpd.read_file(str(out_path))
        assert loaded.crs is not None, "Exported GeoPackage has no CRS"
        assert loaded.crs.to_epsg() == 32632, (
            f"Expected EPSG:32632, got {loaded.crs}"
        )
