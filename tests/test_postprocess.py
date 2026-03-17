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


# ---------------------------------------------------------------------------
# Helpers for Task 16 tests
# ---------------------------------------------------------------------------


def _make_change_gdf(
    polygons: list,
    epsg: int = 32632,
) -> "gpd.GeoDataFrame":
    """Build a minimal change GeoDataFrame with required columns."""
    import geopandas as gpd

    records = [
        {"geometry": p, "confidence": 0.9, "area_m2": p.area, "detection_period": None}
        for p in polygons
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    return gdf.set_crs(epsg=epsg)


def _make_buildings_gdf(
    polygons: list,
    epsg: int = 32632,
) -> "gpd.GeoDataFrame":
    """Build a minimal buildings GeoDataFrame."""
    import geopandas as gpd

    records = [{"geometry": p} for p in polygons]
    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    return gdf.set_crs(epsg=epsg)


# UTM 32N coordinates within Bergen AOI area (metres)
# Origin at (297000, 6690000); 10 m pixel grid

# A 50x50 m building footprint at (297100, 6690100) – (297150, 6690150)
_BUILDING_POLY = _make_buildings_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297100, 6690100, 297150, 6690150)
])

# Change polygon that is fully inside the building footprint
_CHANGE_INSIDE_BUILDING = _make_change_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297110, 6690110, 297140, 6690140)
])

# Change polygon with no overlap with any building
_CHANGE_NO_BUILDING = _make_change_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297300, 6690300, 297350, 6690350)
])

# Change polygon that partially overlaps a building (40% overlap)
# Building: 297100-297150 x 6690100-6690150 (50x50=2500 m²)
# Change polygon: 297120-297200 x 6690100-6690150 (80x50=4000 m²)
# Intersection: 297120-297150 x 6690100-6690150 (30x50=1500 m²)
# overlap_fraction = 1500/4000 = 0.375 -> above 0.3 threshold -> new_building
_BUILDING_PARTIAL = _make_buildings_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297100, 6690100, 297150, 6690150)
])
_CHANGE_PARTIAL_HIGH = _make_change_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297120, 6690100, 297200, 6690150)
])
# Change polygon: 297150-297300 x 6690100-6690150 (150x50=7500 m²)
# Intersection: 0 (no overlap since building ends at 297150 and change starts at 297150)
# Actually set up a 5% overlap case -> new_paving
# Change polygon: 297145-297300 x 6690100-6690150 (155x50=7750 m²)
# Intersection: 297145-297150 x 6690100-6690150 (5x50=250 m²)
# overlap_fraction = 250/7750 = 0.032 -> below 0.3 threshold -> new_paving
_CHANGE_PARTIAL_LOW = _make_change_gdf([
    __import__("shapely.geometry", fromlist=["box"]).box(297145, 6690100, 297300, 6690150)
])


# ---------------------------------------------------------------------------
# 16.1 Failing tests: change type classification
# ---------------------------------------------------------------------------


class TestClassifyBuildingOverlap:
    """test_classify_building_overlap: Polygon overlapping building -> 'new_building'."""

    def test_full_overlap_classified_as_new_building(self) -> None:
        """Change polygon fully inside a building footprint -> 'new_building'."""
        from surface_change_monitor.postprocess import classify_change_type

        result = classify_change_type(_CHANGE_INSIDE_BUILDING, _BUILDING_POLY)

        assert "change_type" in result.columns, "Missing 'change_type' column"
        assert result["change_type"].iloc[0] == "new_building", (
            f"Expected 'new_building', got '{result['change_type'].iloc[0]}'"
        )

    def test_high_partial_overlap_classified_as_new_building(self) -> None:
        """Change polygon with >= 0.3 overlap fraction -> 'new_building'."""
        from surface_change_monitor.postprocess import classify_change_type

        result = classify_change_type(_CHANGE_PARTIAL_HIGH, _BUILDING_PARTIAL)

        assert result["change_type"].iloc[0] == "new_building", (
            f"Expected 'new_building' for 37.5% overlap, got '{result['change_type'].iloc[0]}'"
        )

    def test_original_gdf_unchanged(self) -> None:
        """classify_change_type must not mutate the input GeoDataFrame."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import classify_change_type

        changes_copy = _CHANGE_INSIDE_BUILDING.copy()
        classify_change_type(_CHANGE_INSIDE_BUILDING, _BUILDING_POLY)

        # Original should not have change_type column added
        assert "change_type" not in _CHANGE_INSIDE_BUILDING.columns, (
            "classify_change_type must not mutate the input GeoDataFrame"
        )


class TestClassifyNoOverlap:
    """test_classify_no_overlap: No building overlap -> 'new_paving'."""

    def test_no_overlap_classified_as_new_paving(self) -> None:
        """Change polygon far from any building -> 'new_paving'."""
        from surface_change_monitor.postprocess import classify_change_type

        result = classify_change_type(_CHANGE_NO_BUILDING, _BUILDING_POLY)

        assert "change_type" in result.columns, "Missing 'change_type' column"
        assert result["change_type"].iloc[0] == "new_paving", (
            f"Expected 'new_paving', got '{result['change_type'].iloc[0]}'"
        )

    def test_empty_buildings_all_new_paving(self) -> None:
        """When buildings GeoDataFrame is empty, all changes are 'new_paving'."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import classify_change_type

        empty_buildings = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
        empty_buildings = empty_buildings.set_crs(epsg=32632)

        result = classify_change_type(_CHANGE_INSIDE_BUILDING, empty_buildings)

        assert (result["change_type"] == "new_paving").all(), (
            "All polygons should be 'new_paving' when no buildings available"
        )

    def test_low_partial_overlap_classified_as_new_paving(self) -> None:
        """Change polygon with < 0.3 overlap fraction -> 'new_paving'."""
        from surface_change_monitor.postprocess import classify_change_type

        result = classify_change_type(_CHANGE_PARTIAL_LOW, _BUILDING_PARTIAL)

        assert result["change_type"].iloc[0] == "new_paving", (
            f"Expected 'new_paving' for ~3.2% overlap, got '{result['change_type'].iloc[0]}'"
        )


class TestPartialOverlap:
    """test_partial_overlap: Classification based on overlap fraction and threshold."""

    def test_custom_threshold_respected(self) -> None:
        """Overlap at 37.5% is above 0.3 but below 0.5 -> result changes with threshold."""
        from surface_change_monitor.postprocess import classify_change_type

        result_low_thresh = classify_change_type(
            _CHANGE_PARTIAL_HIGH, _BUILDING_PARTIAL, overlap_threshold=0.3
        )
        result_high_thresh = classify_change_type(
            _CHANGE_PARTIAL_HIGH, _BUILDING_PARTIAL, overlap_threshold=0.5
        )

        assert result_low_thresh["change_type"].iloc[0] == "new_building", (
            "37.5% overlap should be 'new_building' at 0.3 threshold"
        )
        assert result_high_thresh["change_type"].iloc[0] == "new_paving", (
            "37.5% overlap should be 'new_paving' at 0.5 threshold"
        )

    def test_exact_threshold_boundary(self) -> None:
        """Overlap exactly at threshold should be classified as 'new_building' (>=)."""
        from shapely.geometry import box

        from surface_change_monitor.postprocess import classify_change_type

        # Change polygon: 297100-297200 x 6690100-6690130 (100x30=3000 m²)
        # Building:       297100-297200 x 6690100-6690109 (100x9=900 m²)
        # overlap_fraction = 900/3000 = 0.3 -> exactly at threshold -> new_building
        change = _make_change_gdf([box(297100, 6690100, 297200, 6690130)])
        building = _make_buildings_gdf([box(297100, 6690100, 297200, 6690109)])

        result = classify_change_type(change, building, overlap_threshold=0.3)

        assert result["change_type"].iloc[0] == "new_building", (
            "Overlap at exactly 0.3 threshold should be 'new_building'"
        )

    def test_multiple_polygons_classified_independently(self) -> None:
        """Multiple change polygons are each classified independently."""
        import geopandas as gpd
        from shapely.geometry import box

        from surface_change_monitor.postprocess import classify_change_type

        # Two change polygons: one inside building, one outside
        changes = _make_change_gdf([
            box(297110, 6690110, 297140, 6690140),   # inside building
            box(297300, 6690300, 297350, 6690350),   # no building overlap
        ])
        buildings = _make_buildings_gdf([
            box(297100, 6690100, 297150, 6690150)
        ])

        result = classify_change_type(changes, buildings)

        assert len(result) == 2
        assert result["change_type"].iloc[0] == "new_building"
        assert result["change_type"].iloc[1] == "new_paving"

    def test_returns_geodataframe(self) -> None:
        """classify_change_type must return a GeoDataFrame."""
        import geopandas as gpd

        from surface_change_monitor.postprocess import classify_change_type

        result = classify_change_type(_CHANGE_INSIDE_BUILDING, _BUILDING_POLY)

        assert isinstance(result, gpd.GeoDataFrame), (
            f"Expected GeoDataFrame, got {type(result)}"
        )

    def test_change_type_column_values(self) -> None:
        """change_type column must only contain valid values."""
        import geopandas as gpd
        from shapely.geometry import box

        from surface_change_monitor.postprocess import classify_change_type

        changes = _make_change_gdf([
            box(297110, 6690110, 297140, 6690140),
            box(297300, 6690300, 297350, 6690350),
        ])
        buildings = _make_buildings_gdf([box(297100, 6690100, 297150, 6690150)])

        result = classify_change_type(changes, buildings)

        valid_types = {"new_building", "new_paving", "other"}
        unexpected = set(result["change_type"].unique()) - valid_types
        assert not unexpected, f"Unexpected change_type values: {unexpected}"


class TestLoadBuildingFootprints:
    """test_load_building_footprints: Load from GeoParquet for AOI."""

    def test_returns_geodataframe(self, tmp_path: Path) -> None:
        """load_building_footprints returns a GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import box

        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.postprocess import load_building_footprints

        # Build a minimal GeoParquet with some buildings inside and outside AOI
        # Bergen AOI in WGS84: (5.27, 60.35, 5.40, 60.44) lon/lat
        buildings = gpd.GeoDataFrame(
            {"geometry": [box(5.28, 60.36, 5.30, 60.38)]},
            geometry="geometry",
        ).set_crs(epsg=4326)

        parquet_path = tmp_path / "buildings.parquet"
        buildings.to_parquet(str(parquet_path))

        result = load_building_footprints(parquet_path, BERGEN_AOI)

        assert isinstance(result, gpd.GeoDataFrame), (
            f"Expected GeoDataFrame, got {type(result)}"
        )

    def test_clips_to_aoi_bbox(self, tmp_path: Path) -> None:
        """Buildings outside the AOI bounding box are excluded."""
        import geopandas as gpd
        from shapely.geometry import box

        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.postprocess import load_building_footprints

        west, south, east, north = BERGEN_AOI.bbox  # WGS84 degrees
        inside = box(west + 0.01, south + 0.01, west + 0.05, south + 0.04)
        outside = box(west - 1.0, south - 1.0, west - 0.5, south - 0.5)

        buildings = gpd.GeoDataFrame(
            {"geometry": [inside, outside]},
            geometry="geometry",
        ).set_crs(epsg=4326)

        parquet_path = tmp_path / "buildings.parquet"
        buildings.to_parquet(str(parquet_path))

        result = load_building_footprints(parquet_path, BERGEN_AOI)

        # Only the building inside the AOI should be returned
        assert len(result) == 1, (
            f"Expected 1 building inside AOI, got {len(result)}"
        )

    def test_has_geometry_column(self, tmp_path: Path) -> None:
        """Returned GeoDataFrame has a geometry column."""
        import geopandas as gpd
        from shapely.geometry import box

        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.postprocess import load_building_footprints

        west, south, east, north = BERGEN_AOI.bbox
        buildings = gpd.GeoDataFrame(
            {"geometry": [box(west + 0.01, south + 0.01, west + 0.05, south + 0.04)]},
            geometry="geometry",
        ).set_crs(epsg=4326)

        parquet_path = tmp_path / "buildings.parquet"
        buildings.to_parquet(str(parquet_path))

        result = load_building_footprints(parquet_path, BERGEN_AOI)

        assert result.geometry is not None, "GeoDataFrame must have geometry column"

    def test_empty_parquet_returns_empty_gdf(self, tmp_path: Path) -> None:
        """Empty GeoParquet file returns an empty GeoDataFrame."""
        import geopandas as gpd

        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.postprocess import load_building_footprints

        empty_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry").set_crs(epsg=4326)

        parquet_path = tmp_path / "empty_buildings.parquet"
        empty_gdf.to_parquet(str(parquet_path))

        result = load_building_footprints(parquet_path, BERGEN_AOI)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0, "Expected empty GeoDataFrame from empty parquet"

    def test_file_not_found_raises(self) -> None:
        """FileNotFoundError is raised when the parquet file does not exist."""
        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.postprocess import load_building_footprints

        with pytest.raises(FileNotFoundError):
            load_building_footprints(Path("/nonexistent/path/buildings.parquet"), BERGEN_AOI)
