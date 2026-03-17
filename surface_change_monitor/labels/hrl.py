"""HRL (High Resolution Layer) imperviousness tile loading and processing.

Copernicus HRL Imperviousness provides per-pixel imperviousness density (0–100 %)
at 10 m resolution in EPSG:3035 (ETRS89-LAEA).  This module loads an HRL tile,
reprojects it to the AOI's native UTM CRS, and clips it to the AOI extent.

Typical epoch pairs used as change-label references: 2018 vs 2021.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 – registers .rio accessor on xr.DataArray
import xarray as xr
from pyproj import Transformer

from surface_change_monitor.config import AOI


def load_hrl_density(tile_path: Path, aoi: AOI) -> xr.DataArray:
    """Load an HRL imperviousness tile, reproject to AOI CRS, and clip to AOI.

    The HRL tile is expected to be a single-band GeoTIFF in EPSG:3035 with
    uint8 values in [0, 100] plus a nodata marker (commonly 255).  The function:

    1. Opens the tile with rioxarray.
    2. Replaces the nodata marker with NaN.
    3. Reprojects the full tile to the AOI's UTM EPSG.
       (Clipping is done *after* reprojection because EPSG:3035 axis order can
       be non-monotonic for study areas far outside Europe, e.g. Houston.)
    4. Clips the reprojected raster to the AOI bounding box in the target CRS.
    5. Returns the result as a float32 DataArray with nodata pixels as NaN.

    Parameters
    ----------
    tile_path:
        Path to a single-band HRL GeoTIFF in EPSG:3035.
    aoi:
        Area of interest.  Its ``epsg`` attribute defines the target CRS and
        its ``bbox`` (west, south, east, north in WGS84) defines the clip
        extent.

    Returns
    -------
    xr.DataArray
        2-D float32 DataArray clipped to the AOI and in the AOI's UTM CRS.
        Nodata pixels are encoded as NaN.  The ``.rio.crs`` attribute
        is set to the AOI's EPSG.
    """
    da: xr.DataArray = xr.open_dataarray(tile_path, engine="rasterio")

    # Squeeze the band dimension if present (single-band HRL tile).
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)

    # Convert nodata values to NaN so arithmetic is clean.
    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata)

    target_crs_str = f"EPSG:{aoi.epsg}"

    # Reproject the full tile to the target UTM CRS.
    # We defer clipping until after reprojection because EPSG:3035 (LAEA) can
    # produce non-monotonic or inverted axes when transforming bounding boxes for
    # regions far from the projection centre (e.g. North America).
    da_reprojected: xr.DataArray = da.rio.reproject(target_crs_str)

    # Ensure float32; the reprojection may have changed dtype.
    da_reprojected = da_reprojected.astype(np.float32)

    # After reprojection rioxarray may re-introduce an integer nodata fill;
    # convert it back to NaN.
    reproj_nodata = da_reprojected.rio.nodata
    if reproj_nodata is not None and not np.isnan(float(reproj_nodata)):
        da_reprojected = da_reprojected.where(da_reprojected != reproj_nodata)

    # Clip to AOI bounding box in the target (UTM) CRS.
    transformer = Transformer.from_crs("EPSG:4326", target_crs_str, always_xy=True)
    west_utm, south_utm = transformer.transform(aoi.bbox[0], aoi.bbox[1])
    east_utm, north_utm = transformer.transform(aoi.bbox[2], aoi.bbox[3])

    da_clipped: xr.DataArray = da_reprojected.rio.clip_box(
        minx=west_utm,
        miny=south_utm,
        maxx=east_utm,
        maxy=north_utm,
    )

    return da_clipped
