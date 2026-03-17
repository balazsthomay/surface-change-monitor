"""Post-processing: convert probability raster to clean vector polygons.

Takes the probability map produced by :func:`surface_change_monitor.model.predict.predict_change`
and converts it into a GeoDataFrame of imperviousness-change polygons ready for
ingestion into the 7Analytics flood-model update pipeline.

Pipeline
--------
1. Threshold probability map to binary (>= threshold -> 1)
2. Morphological opening  (removes single-pixel noise)
3. Morphological closing  (fills small holes inside blobs)
4. Vectorize connected regions with ``rasterio.features.shapes``
5. Filter polygons by minimum area
6. Compute attributes: confidence (mean probability), area_m2, detection_period

Typical usage
-------------
>>> from pathlib import Path
>>> import rioxarray
>>> from surface_change_monitor.postprocess import vectorize_changes
>>> prob_map = rioxarray.open_rasterio("change_prob.tif").squeeze()
>>> gdf = vectorize_changes(prob_map, threshold=0.5, detection_period="2024-01/2024-02")
>>> gdf.to_file("changes.geojson", driver="GeoJSON")
>>> gdf.to_file("changes.gpkg", driver="GPKG")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio.features
import xarray as xr
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import shape

from surface_change_monitor.config import AOI


def vectorize_changes(
    probability_map: xr.DataArray,
    threshold: float = 0.5,
    min_area_m2: float = 200.0,
    detection_period: str | None = None,
) -> gpd.GeoDataFrame:
    """Convert a probability raster to clean vector change polygons.

    Parameters
    ----------
    probability_map:
        Float DataArray of shape ``(H, W)`` with values in ``[0, 1]``.
        Must carry a CRS (via rioxarray) and a valid affine transform.
    threshold:
        Probability cutoff.  Pixels with probability >= threshold are
        considered "changed".  Default 0.5.
    min_area_m2:
        Minimum polygon area in square metres.  Polygons smaller than this
        are discarded.  Default 200 m^2.
    detection_period:
        Optional ISO-8601 interval string (e.g. ``"2024-01/2024-02"``) that
        is added as a ``detection_period`` attribute column.  If *None*, the
        column is still created and populated with ``None``.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: geometry, confidence, area_m2, detection_period.
        The GeoDataFrame CRS matches the input *probability_map*.
    """
    prob_values = probability_map.values  # (H, W) float32/float64

    # ------------------------------------------------------------------
    # 1. Threshold -> binary mask
    # ------------------------------------------------------------------
    binary: np.ndarray = (prob_values >= threshold).astype(np.uint8)

    # ------------------------------------------------------------------
    # 2. Morphological opening: removes isolated single-pixel noise
    #    Uses a 3x3 cross-shaped structuring element so diagonal
    #    single pixels are also killed without over-eroding real blobs.
    # ------------------------------------------------------------------
    struct = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=bool,
    )
    binary = binary_opening(binary, structure=struct).astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. Morphological closing: fills small internal holes
    # ------------------------------------------------------------------
    binary = binary_closing(binary, structure=struct).astype(np.uint8)

    # ------------------------------------------------------------------
    # 4. Vectorize with rasterio.features.shapes
    # ------------------------------------------------------------------
    transform = probability_map.rio.transform()
    crs = probability_map.rio.crs

    shapes_gen = rasterio.features.shapes(
        binary,
        mask=binary,          # only emit shapes for value=1 pixels
        transform=transform,
    )

    # ------------------------------------------------------------------
    # 5. Build GeoDataFrame and filter by area
    # ------------------------------------------------------------------
    records: list[dict] = []
    for geom_dict, _pixel_val in shapes_gen:
        geom = shape(geom_dict)
        area = geom.area  # CRS units^2 (metres^2 for projected CRS)
        if area < min_area_m2:
            continue

        # Compute confidence: mean probability of all pixels inside polygon
        # Use rasterio.features.geometry_mask for pixel membership
        poly_mask = rasterio.features.geometry_mask(
            [geom_dict],
            out_shape=prob_values.shape,
            transform=transform,
            invert=True,  # True inside polygon
        )
        confidence = float(prob_values[poly_mask].mean()) if poly_mask.any() else float(threshold)

        records.append(
            {
                "geometry": geom,
                "confidence": confidence,
                "area_m2": area,
                "detection_period": detection_period,
            }
        )

    if not records:
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "confidence", "area_m2", "detection_period"],
            geometry="geometry",
        )
    else:
        gdf = gpd.GeoDataFrame(records, geometry="geometry")

    if crs is not None:
        gdf = gdf.set_crs(crs)

    return gdf


def load_building_footprints(footprints_path: Path, aoi: AOI) -> gpd.GeoDataFrame:
    """Load Microsoft Building Footprints from a local GeoParquet file, clipped to an AOI.

    Parameters
    ----------
    footprints_path:
        Path to a GeoParquet file containing building polygon geometries.
        The file must have a geometry column and a CRS.
    aoi:
        Area of interest used to spatially filter the footprints.
        The ``bbox`` is expected as ``(west, south, east, north)`` in WGS84 degrees.

    Returns
    -------
    gpd.GeoDataFrame
        Building polygons clipped to the AOI bounding box, in the file's original CRS.

    Raises
    ------
    FileNotFoundError
        If *footprints_path* does not exist.
    """
    if not footprints_path.exists():
        raise FileNotFoundError(f"Building footprints file not found: {footprints_path}")

    gdf = gpd.read_parquet(str(footprints_path))

    if gdf.empty:
        return gdf

    west, south, east, north = aoi.bbox
    clipped = gdf.cx[west:east, south:north]

    return clipped.reset_index(drop=True)


def classify_change_type(
    changes: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    overlap_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Classify each change polygon as 'new_building', 'new_paving', or 'other'.

    For each change polygon the function computes the fraction of the polygon's
    area that overlaps with any building footprint:

        overlap_fraction = total_intersection_area / change_polygon_area

    If ``overlap_fraction >= overlap_threshold`` the change is labelled
    ``"new_building"``; otherwise ``"new_paving"``.  The ``"other"`` label is
    reserved for edge cases (e.g. zero-area polygons) but is not expected in
    normal operation.

    Parameters
    ----------
    changes:
        GeoDataFrame of change polygons (output of :func:`vectorize_changes`).
        Must carry a valid CRS.
    buildings:
        GeoDataFrame of building footprint polygons (e.g. from
        :func:`load_building_footprints`).  Must be in the same CRS as
        *changes*, or empty.
    overlap_threshold:
        Minimum overlap fraction to classify a polygon as ``"new_building"``.
        Default 0.3 (30 %).

    Returns
    -------
    gpd.GeoDataFrame
        A copy of *changes* with an added ``change_type`` string column.
        The input GeoDataFrame is never mutated.
    """
    result = changes.copy()

    if result.empty:
        result["change_type"] = []
        return result

    change_types: list[str] = []

    if buildings.empty:
        result["change_type"] = "new_paving"
        return result

    # Ensure buildings are in the same CRS as changes
    if buildings.crs is not None and result.crs is not None and buildings.crs != result.crs:
        buildings = buildings.to_crs(result.crs)

    # Build a union of all building geometries for efficient intersection queries
    # using a spatial index (sindex) on the buildings GeoDataFrame
    buildings_sindex = buildings.sindex

    for change_geom in result.geometry:
        if change_geom is None or change_geom.is_empty or change_geom.area == 0:
            change_types.append("other")
            continue

        # Query the spatial index for candidate buildings
        candidate_idx = list(buildings_sindex.query(change_geom, predicate="intersects"))

        if not candidate_idx:
            change_types.append("new_paving")
            continue

        candidate_buildings = buildings.iloc[candidate_idx]
        total_intersection_area = float(
            candidate_buildings.geometry.intersection(change_geom).area.sum()
        )
        overlap_fraction = total_intersection_area / change_geom.area

        if overlap_fraction >= overlap_threshold:
            change_types.append("new_building")
        else:
            change_types.append("new_paving")

    result["change_type"] = change_types
    return result
