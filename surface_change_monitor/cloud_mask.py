"""Cloud masking utilities for Sentinel-2 L2A Scene Classification Layer (SCL).

The SCL band classifies every pixel into one of 12 categories:
  0  No data
  1  Saturated / defective
  2  Dark area pixels
  3  Cloud shadows          <- masked
  4  Vegetation
  5  Bare soils
  6  Water
  7  Unclassified           <- masked
  8  Cloud medium probability  <- masked
  9  Cloud high probability    <- masked
  10 Thin cirrus             <- masked
  11 Snow / Ice

Values in SCL_MASK_VALUES (config) are treated as cloud-contaminated.  All
other values are considered *clear*.  The binary mask follows the convention:
  True  = clear pixel  (use in analysis)
  False = cloudy pixel (exclude from analysis)
"""

import numpy as np
import xarray as xr

from surface_change_monitor.config import SCL_MASK_VALUES


def create_cloud_mask(scl: xr.DataArray) -> xr.DataArray:
    """Build a boolean clear-sky mask from an SCL DataArray.

    Parameters
    ----------
    scl:
        Integer DataArray containing Sentinel-2 SCL pixel classifications.

    Returns
    -------
    xr.DataArray
        Boolean DataArray with the same shape and coordinates as *scl*.
        ``True`` means the pixel is clear; ``False`` means it is cloud-
        contaminated or otherwise unreliable.
    """
    # Start with every pixel considered clear, then mark masked values False.
    is_cloudy = xr.zeros_like(scl, dtype=bool)
    for value in SCL_MASK_VALUES:
        is_cloudy = is_cloudy | (scl == value)

    mask: xr.DataArray = ~is_cloudy
    return mask


def apply_cloud_mask(band: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Replace cloud-contaminated pixels in *band* with NaN.

    Parameters
    ----------
    band:
        Floating-point spectral band DataArray.
    mask:
        Boolean clear-sky mask aligned with *band* (True = clear).

    Returns
    -------
    xr.DataArray
        A copy of *band* where pixels flagged as cloudy (``mask == False``)
        are set to ``NaN``.  Band attributes are preserved.
    """
    # Ensure the band is float so it can hold NaN values.
    result: xr.DataArray = band.where(mask)
    result.attrs = band.attrs.copy()
    return result


def cloud_free_fraction(mask: xr.DataArray) -> float:
    """Compute the fraction of clear (non-cloudy) pixels.

    Parameters
    ----------
    mask:
        Boolean DataArray where ``True`` represents a clear pixel.

    Returns
    -------
    float
        Value in [0.0, 1.0].  1.0 means fully cloud-free; 0.0 means
        entirely cloud-covered.
    """
    total: int = int(mask.size)
    if total == 0:
        return 0.0
    clear: int = int(np.count_nonzero(mask.values))
    return float(clear / total)


def resample_mask_to_band(mask: xr.DataArray, band: xr.DataArray) -> xr.DataArray:
    """Upsample a coarser-resolution mask to match the grid of *band*.

    The Sentinel-2 SCL band is delivered at 20 m resolution while most
    spectral bands are at 10 m.  This function reprojects the boolean mask
    onto the target band's coordinate grid using nearest-neighbour
    interpolation, which preserves the True/False semantics without
    introducing interpolated intermediate values.

    Parameters
    ----------
    mask:
        Boolean clear-sky mask, typically derived from a 20 m SCL array.
    band:
        Target band DataArray defining the destination coordinate grid.

    Returns
    -------
    xr.DataArray
        Boolean DataArray with the same spatial shape and coordinates as
        *band*.
    """
    # Nearest-neighbour resampling using index arithmetic — no scipy needed.
    # For each target coordinate, find the index of the nearest source coordinate.
    src_x = mask.coords["x"].values
    src_y = mask.coords["y"].values
    tgt_x = band.coords["x"].values
    tgt_y = band.coords["y"].values

    # searchsorted-based nearest-neighbour: pick the closest source index.
    def _nearest_indices(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Return the index in *src* nearest to each element of *tgt*."""
        sorted_src = np.sort(src)
        # Find insertion points in the sorted source array.
        idx = np.searchsorted(sorted_src, tgt, side="left")
        idx = np.clip(idx, 0, len(sorted_src) - 1)
        # Compare with the left neighbour and choose the closer one.
        left_idx = np.clip(idx - 1, 0, len(sorted_src) - 1)
        closer_left = np.abs(tgt - sorted_src[left_idx]) < np.abs(tgt - sorted_src[idx])
        idx = np.where(closer_left, left_idx, idx)
        # Map back to original (unsorted) source indices.
        sort_order = np.argsort(src)
        return sort_order[idx]

    xi = _nearest_indices(src_x, tgt_x)
    yi = _nearest_indices(src_y, tgt_y)

    # Index into the mask array: shape (len(tgt_y), len(tgt_x)).
    resampled_values = mask.values[np.ix_(yi, xi)]

    result = xr.DataArray(
        resampled_values,
        dims=["y", "x"],
        coords={"x": band.coords["x"], "y": band.coords["y"]},
    )
    return result
