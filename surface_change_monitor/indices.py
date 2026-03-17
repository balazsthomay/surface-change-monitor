"""Spectral index computation for Sentinel-2 monthly composites.

Computes standard normalized-difference indices used to discriminate
impervious surfaces, vegetation, and water:

- NDVI  (Normalized Difference Vegetation Index) : (B08 - B04) / (B08 + B04)
- NDBI  (Normalized Difference Built-up Index)   : (B11 - B08) / (B11 + B08)
- NDWI  (Normalized Difference Water Index)      : (B03 - B08) / (B03 + B08)

All indices use :func:`_safe_normalized_difference` which maps zero-denominator
pixels to **0.0** (avoiding NaN / inf) while propagating NaN from missing data.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from surface_change_monitor.composite import MonthlyComposite


def _safe_normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Compute (a - b) / (a + b) with safe handling of zero denominators.

    Parameters
    ----------
    a, b:
        Input arrays of the same shape.  NaN values propagate naturally
        through the arithmetic; only the case where ``a + b == 0`` (and
        neither operand is NaN) is treated specially.

    Returns
    -------
    xr.DataArray
        Normalized difference with the same shape as the inputs, dtype
        float32.  Pixels where ``a + b == 0`` are set to **0.0**; pixels
        where either input is NaN remain NaN.
    """
    denom = a + b
    return xr.where(denom != 0, (a - b) / denom, 0.0).astype(np.float32)


def ndvi(b08: xr.DataArray, b04: xr.DataArray) -> xr.DataArray:
    """Normalized Difference Vegetation Index.

    NDVI = (B08 - B04) / (B08 + B04)

    High values indicate dense healthy vegetation; low / negative values
    indicate bare soil, built-up surfaces, or water.

    Parameters
    ----------
    b08:
        Near-infrared (NIR) band, Sentinel-2 Band 8 at 10 m.
    b04:
        Red band, Sentinel-2 Band 4 at 10 m.

    Returns
    -------
    xr.DataArray
        NDVI in [-1, 1], float32.  Zero-denominator pixels → 0.0.
    """
    return _safe_normalized_difference(b08, b04)


def ndbi(b11: xr.DataArray, b08: xr.DataArray) -> xr.DataArray:
    """Normalized Difference Built-up Index.

    NDBI = (B11 - B08) / (B11 + B08)

    Positive values highlight impervious / built-up areas where shortwave
    infrared reflectance exceeds NIR.

    Parameters
    ----------
    b11:
        Shortwave infrared (SWIR) band, Sentinel-2 Band 11 at 20 m.
    b08:
        Near-infrared (NIR) band, Sentinel-2 Band 8 at 10 m.

    Returns
    -------
    xr.DataArray
        NDBI in [-1, 1], float32.  Zero-denominator pixels → 0.0.
    """
    return _safe_normalized_difference(b11, b08)


def ndwi(b03: xr.DataArray, b08: xr.DataArray) -> xr.DataArray:
    """Normalized Difference Water Index.

    NDWI = (B03 - B08) / (B03 + B08)

    Positive values indicate open water; negative values indicate
    vegetation or built-up land.

    Parameters
    ----------
    b03:
        Green band, Sentinel-2 Band 3 at 10 m.
    b08:
        Near-infrared (NIR) band, Sentinel-2 Band 8 at 10 m.

    Returns
    -------
    xr.DataArray
        NDWI in [-1, 1], float32.  Zero-denominator pixels → 0.0.
    """
    return _safe_normalized_difference(b03, b08)


def add_indices_to_composite(composite: MonthlyComposite) -> MonthlyComposite:
    """Append NDVI, NDBI, and NDWI as new bands to a monthly composite.

    Extracts the required spectral bands by name from ``composite.data``,
    computes the three indices, and concatenates them along the ``band``
    dimension.  The original bands are left untouched.

    Final band order after expansion (9 channels total):
    ``B02, B03, B04, B08, B11, B12, NDVI, NDBI, NDWI``

    Parameters
    ----------
    composite:
        A :class:`~surface_change_monitor.composite.MonthlyComposite` whose
        ``data`` array carries at minimum the bands ``B03``, ``B04``,
        ``B08``, and ``B11`` as coordinates on the ``band`` dimension.

    Returns
    -------
    MonthlyComposite
        A new composite with the same metadata and spatial dimensions,
        but with three extra index bands appended.  The original composite
        is not modified.
    """
    b03 = composite.data.sel(band="B03").drop_vars("band")
    b04 = composite.data.sel(band="B04").drop_vars("band")
    b08 = composite.data.sel(band="B08").drop_vars("band")
    b11 = composite.data.sel(band="B11").drop_vars("band")

    ndvi_da = ndvi(b08, b04).expand_dims("band").assign_coords(band=["NDVI"])
    ndbi_da = ndbi(b11, b08).expand_dims("band").assign_coords(band=["NDBI"])
    ndwi_da = ndwi(b03, b08).expand_dims("band").assign_coords(band=["NDWI"])

    extended_data = xr.concat(
        [composite.data, ndvi_da, ndbi_da, ndwi_da],
        dim="band",
    ).astype(np.float32)

    return MonthlyComposite(
        data=extended_data,
        year_month=composite.year_month,
        n_scenes=composite.n_scenes,
        clear_obs_count=composite.clear_obs_count,
        reliable=composite.reliable,
        aoi=composite.aoi,
    )
