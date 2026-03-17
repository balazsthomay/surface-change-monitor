"""NLCD (National Land Cover Database) fractional impervious surface loading.

NLCD provides annual fractional imperviousness at 30 m resolution for CONUS
in EPSG:5070 (NAD83 / Conus Albers Equal Area Conic).  This module loads an
NLCD imperviousness GeoTIFF (Collection 1.1, 1985–2024), clips it to an AOI,
reprojects it to the AOI's UTM CRS, and resamples from 30 m to 10 m using
bilinear interpolation to align with Sentinel-2 imagery.

Important note on resampling
-----------------------------
Downsampling from 30 m → 10 m does *not* create new information; it merely
increases pixel density to match the Sentinel-2 grid.  NLCD-derived labels
should therefore be down-weighted (weight=0.5) relative to HRL-derived labels
(weight=1.0) during model training to reflect this lower effective resolution.

Typical epoch pairs used as change-label references: 2019 vs 2020.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 – registers .rio accessor on xr.DataArray
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling

from surface_change_monitor.config import AOI

# NLCD nodata sentinel value.  The file may already have been decoded by
# rioxarray (nodata → NaN), but we guard against both cases.
_NLCD_NODATA = 127

# Target output resolution in metres after resampling.
_TARGET_RESOLUTION_M = 10.0


def load_nlcd_impervious(tile_path: Path, aoi: AOI) -> xr.DataArray:
    """Load NLCD fractional impervious for a single tile, clip to AOI, and resample.

    The function performs the following steps:

    1. Opens the GeoTIFF with rioxarray (EPSG:5070, 30 m, uint8 0–100, nodata=127).
    2. Replaces nodata (127) with NaN and casts to float32.
    3. Reprojects the full tile from EPSG:5070 to the AOI's UTM EPSG using
       bilinear resampling at the target 10 m resolution.
       (Clipping is deferred until after reprojection to avoid issues with
       EPSG:5070 bounding-box transformations crossing the projection edge.)
    4. Clips the result to the AOI bounding box in the UTM CRS.
    5. Returns the result as a float32 DataArray with nodata pixels as NaN.

    Parameters
    ----------
    tile_path:
        Path to a single-band NLCD impervious GeoTIFF in EPSG:5070.
        The file is expected to exist on disk — downloading is handled
        separately (NLCD data is downloaded manually from MRLC/AWS).
    aoi:
        Area of interest.  Its ``epsg`` attribute defines the target CRS and
        its ``bbox`` (west, south, east, north in WGS84) defines the clip
        extent.

    Returns
    -------
    xr.DataArray
        2-D float32 DataArray clipped to the AOI and in the AOI's UTM CRS
        at 10 m resolution.  Nodata pixels are encoded as NaN.
        The ``.rio.crs`` attribute is set to the AOI's EPSG.

    Notes
    -----
    * NLCD native CRS: EPSG:5070 (NAD83 / Conus Albers Equal Area Conic).
    * Native resolution: 30 m.
    * Valid value range: 0–100 (percent impervious).
    * Nodata value: 127.
    * The 30 m → 10 m resampling uses bilinear interpolation via
      ``Resampling.bilinear`` (rasterio), consistent with rioxarray's
      ``reproject`` API.
    """
    da: xr.DataArray = xr.open_dataarray(tile_path, engine="rasterio")

    # Squeeze the band dimension if present (single-band NLCD tile).
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)

    # Replace nodata values with NaN.
    # rioxarray may have already decoded nodata=127 → NaN when opening the file
    # (stored as encoded_nodata).  We handle both the raw-uint8 case and the
    # already-decoded float32 case.
    raw_nodata = da.rio.nodata
    encoded_nodata = da.rio.encoded_nodata

    if raw_nodata is not None and not (isinstance(raw_nodata, float) and np.isnan(raw_nodata)):
        da = da.where(da != raw_nodata)
    if encoded_nodata is not None and not (isinstance(encoded_nodata, float) and np.isnan(encoded_nodata)):
        da = da.where(da != encoded_nodata)

    # Cast to float32 for consistent downstream arithmetic.
    da = da.astype(np.float32)

    target_crs_str = f"EPSG:{aoi.epsg}"

    # Reproject + resample in one pass:
    #   - target CRS: AOI's UTM EPSG
    #   - target resolution: 10 m (x and y)
    #   - resampling method: bilinear (smooth interpolation, appropriate for
    #     continuous percentage values)
    da_reprojected: xr.DataArray = da.rio.reproject(
        target_crs_str,
        resolution=_TARGET_RESOLUTION_M,
        resampling=Resampling.bilinear,
    )

    # After reprojection rioxarray may re-introduce an integer nodata fill.
    reproj_nodata = da_reprojected.rio.nodata
    if reproj_nodata is not None and not (isinstance(reproj_nodata, float) and np.isnan(float(reproj_nodata))):
        da_reprojected = da_reprojected.where(da_reprojected != reproj_nodata)

    # Ensure float32 after reprojection.
    da_reprojected = da_reprojected.astype(np.float32)

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
