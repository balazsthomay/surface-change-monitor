"""Monthly median compositing for Sentinel-2 scenes.

Takes a collection of downloaded scenes for a single month, applies the
Sentinel-2 Scene Classification Layer (SCL) cloud mask to each scene, then
stacks all cloud-free observations and computes a per-pixel nanmedian.

The result is a :class:`MonthlyComposite` that records:
- The median spectral values as a float32 DataArray of shape (bands, H, W)
- The number of clear observations per pixel
- Whether the month is *reliable* (>= 3 scenes contributed)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 – registers .rio accessor on xr.DataArray
import xarray as xr

from surface_change_monitor.cloud_mask import (
    apply_cloud_mask,
    create_cloud_mask,
    resample_mask_to_band,
)
from surface_change_monitor.config import AOI
from surface_change_monitor.stac import SceneMetadata

# Minimum number of scenes required for a composite to be considered reliable.
_MIN_RELIABLE_SCENES: int = 3


@dataclass
class MonthlyComposite:
    """Cloud-free monthly median composite derived from multiple Sentinel-2 scenes.

    Attributes:
        data: Median spectral values as a float32 DataArray of shape
              (bands, H, W).  Band names are preserved as the first coordinate.
        year_month: Month of the composite in ``"YYYY-MM"`` format.
        n_scenes: Number of input scenes that contributed to this composite.
        clear_obs_count: Per-pixel count of clear (non-NaN) observations,
                         DataArray of shape (H, W) with integer dtype.
        reliable: ``True`` when *n_scenes* >= 3, meaning the nanmedian is
                  statistically meaningful.
        aoi: The area of interest this composite was built for.
    """

    data: xr.DataArray
    year_month: str
    n_scenes: int
    clear_obs_count: xr.DataArray
    reliable: bool
    aoi: AOI


def group_scenes_by_month(scenes: list[SceneMetadata]) -> dict[str, list[SceneMetadata]]:
    """Partition a list of scenes by calendar month.

    Parameters
    ----------
    scenes:
        Scene metadata objects, each carrying a ``datetime`` attribute.

    Returns
    -------
    dict[str, list[SceneMetadata]]
        Keys are ``"YYYY-MM"`` strings; values are lists of scenes whose
        sensing datetime falls in that month.  An empty list returns ``{}``.
    """
    groups: dict[str, list[SceneMetadata]] = defaultdict(list)
    for scene in scenes:
        key = scene.datetime.strftime("%Y-%m")
        groups[key].append(scene)
    return dict(groups)


def create_monthly_composite(
    band_paths: list[dict[str, Path]],
    aoi: AOI,
    year_month: str,
) -> MonthlyComposite:
    """Build a cloud-free monthly median composite from per-scene GeoTIFF files.

    For each scene the function:
    1. Identifies spectral bands (all keys that are **not** ``"SCL"``).
    2. Loads the SCL band and derives a boolean clear-sky mask.
    3. Applies the mask to each spectral band (cloudy pixels → NaN).
    4. Upsamples the mask to 10 m if the spectral band has finer resolution
       than the SCL band.

    After processing all scenes the arrays are stacked along a ``"time"``
    dimension and a per-band ``nanmedian`` is computed to form the composite.

    Parameters
    ----------
    band_paths:
        One dict per scene mapping band name (e.g. ``"B04"``, ``"SCL"``) to
        the path of the corresponding single-band GeoTIFF.
    aoi:
        The geographic area of interest for which the composite is built.
        Stored on the returned object but not used for spatial subsetting here
        (clipping is assumed to have happened during download).
    year_month:
        The calendar month in ``"YYYY-MM"`` format.

    Returns
    -------
    MonthlyComposite
        The finished composite, including reliability flag and per-pixel
        clear-observation count.
    """
    n_scenes = len(band_paths)

    # Collect per-scene masked band stacks.
    # scene_arrays[band_name] = list of (H, W) float32 DataArrays
    scene_arrays: dict[str, list[xr.DataArray]] = defaultdict(list)

    for scene_band_paths in band_paths:
        spectral_bands = {k: v for k, v in scene_band_paths.items() if k != "SCL"}

        # Load SCL and create cloud mask (always at its native resolution)
        scl_path = scene_band_paths.get("SCL")
        scl: xr.DataArray | None = None
        cloud_mask: xr.DataArray | None = None
        if scl_path is not None:
            scl = _load_band(scl_path).squeeze()  # (H_scl, W_scl)
            cloud_mask = create_cloud_mask(scl.astype(np.uint8))

        # Load, mask, and collect each spectral band.
        for band_name, tif_path in spectral_bands.items():
            band = _load_band(tif_path).squeeze().astype(np.float32)  # (H, W)

            if cloud_mask is not None:
                # Upsample mask if SCL is coarser than the spectral band
                if cloud_mask.shape != band.shape:
                    effective_mask = resample_mask_to_band(cloud_mask, band)
                else:
                    effective_mask = cloud_mask
                band = apply_cloud_mask(band, effective_mask)

            scene_arrays[band_name].append(band)

    # Determine the canonical spatial reference from the first available band
    # of the first scene (needed to attach CRS to the output DataArray).
    first_scene = band_paths[0]
    first_spectral_key = next(k for k in first_scene if k != "SCL")
    reference_band = _load_band(first_scene[first_spectral_key]).squeeze()

    band_names = sorted(scene_arrays.keys())

    # Compute nanmedian and clear-obs count per band, then stack.
    median_slices: list[xr.DataArray] = []
    count_accum: xr.DataArray | None = None

    for band_name in band_names:
        arrays = scene_arrays[band_name]  # list[DataArray(H, W)]

        # Stack along a synthetic time dimension: shape (T, H, W)
        time_stack = np.stack([a.values for a in arrays], axis=0)

        median_vals = np.nanmedian(time_stack, axis=0).astype(np.float32)
        # Non-NaN count per pixel (same for every band given the same SCL mask)
        if count_accum is None:
            obs_count = np.sum(~np.isnan(time_stack), axis=0).astype(np.int32)

        median_da = xr.DataArray(
            median_vals,
            dims=["y", "x"],
            coords={"y": arrays[0].coords["y"], "x": arrays[0].coords["x"]},
        )
        median_slices.append(median_da)

    # Stack bands: shape (bands, H, W)
    composite_data = xr.concat(median_slices, dim="band")
    composite_data = composite_data.assign_coords(band=band_names)
    composite_data = composite_data.astype(np.float32)

    # Transfer the CRS from the reference band onto the composite
    if reference_band.rio.crs is not None:
        composite_data = composite_data.rio.write_crs(reference_band.rio.crs)

    # Build clear_obs_count DataArray (H, W)
    obs_da = xr.DataArray(
        obs_count,
        dims=["y", "x"],
        coords={"y": median_slices[0].coords["y"], "x": median_slices[0].coords["x"]},
    )

    return MonthlyComposite(
        data=composite_data,
        year_month=year_month,
        n_scenes=n_scenes,
        clear_obs_count=obs_da,
        reliable=n_scenes >= _MIN_RELIABLE_SCENES,
        aoi=aoi,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_band(path: Path) -> xr.DataArray:
    """Load a single-band GeoTIFF as an xarray DataArray using rioxarray.

    Parameters
    ----------
    path:
        Path to a GeoTIFF file.

    Returns
    -------
    xr.DataArray
        Array with ``x``, ``y`` (and ``band``) dimensions and a ``spatial_ref``
        coordinate set by rioxarray.
    """
    return xr.open_dataarray(path, engine="rasterio")
