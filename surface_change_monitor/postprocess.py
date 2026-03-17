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

import numpy as np
import geopandas as gpd
import rasterio.features
import xarray as xr
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import shape


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
