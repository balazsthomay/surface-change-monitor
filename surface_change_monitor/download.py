"""Band download module for the surface change monitor pipeline.

Handles fetching individual Sentinel-2 bands from CDSE STAC asset HREFs using
OAuth2 Bearer authentication, clipping/reprojecting to an AOI, and resampling
20 m bands to 10 m resolution.

Public API:
    - :func:`download_band` – stream a single band file with retry logic.
    - :func:`clip_and_reproject` – clip to AOI, reproject to AOI CRS, resample.
    - :func:`download_scene_bands` – orchestrate per-scene band downloads.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
import rioxarray  # noqa: F401 – registers the .rio accessor on xarray objects
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from surface_change_monitor.config import AOI, BANDS_20M
from surface_change_monitor.stac import SceneMetadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB

# HTTP status codes that warrant an automatic retry.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Resampling algorithm names understood by rasterio/rioxarray.
_RESAMPLING_MAP: dict[str, Resampling] = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DownloadError(RuntimeError):
    """Raised when a band download fails after exhausting all retries."""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def download_band(
    href: str,
    dest: Path,
    token: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    """Download a single band file from *href* to *dest* with auth and retry.

    Streams the response body in chunks so that large (40+ MB) JP2 files do
    not fill RAM.  Retries on HTTP 429 (rate-limited) and 5xx server errors
    with exponential back-off.

    Args:
        href:        CDSE asset URL, e.g. from :attr:`SceneMetadata.assets`.
        dest:        Local file path where the band will be saved.
        token:       OAuth2 Bearer token (plain string, no ``"Bearer "`` prefix).
        max_retries: Maximum number of retry attempts before raising.
        chunk_size:  Byte size of each streamed chunk (default 1 MiB).

    Returns:
        *dest* — the path to the saved file.

    Raises:
        DownloadError: If the download fails after *max_retries* attempts.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}

    last_status: int = 0
    for attempt in range(max_retries + 1):
        response = requests.get(href, headers=headers, stream=True, timeout=120)
        last_status = response.status_code

        if response.status_code == 200:
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
            return dest

        if response.status_code in _RETRYABLE_STATUSES:
            if attempt < max_retries:
                # Honour Retry-After when present, otherwise exponential backoff.
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 2.0 ** attempt
                time.sleep(wait)
                continue
            break

        # Non-retryable error – surface immediately.
        response.raise_for_status()

    raise DownloadError(
        f"Failed to download {href!r} after {max_retries} retries "
        f"(last HTTP status: {last_status})."
    )


def clip_and_reproject(
    src: Path,
    dest: Path,
    *,
    aoi: AOI,
    target_resolution: int,
    resampling: str = "bilinear",
) -> Path:
    """Clip *src* raster to *aoi*, reproject to AOI CRS, resample to *target_resolution*.

    Opens *src* with rioxarray, reprojects to the AOI's UTM CRS at
    *target_resolution* metres/pixel, then clips to the AOI bounding box.
    Saves the result as a Cloud-optimised GeoTIFF at *dest*.

    Args:
        src:               Path to the source raster (JP2 or GeoTIFF).
        dest:              Destination path for the output GeoTIFF.
        aoi:               AOI whose EPSG CRS and bbox are used.
        target_resolution: Output pixel size in metres (e.g. 10 for 10 m).
        resampling:        Resampling algorithm name; one of ``"bilinear"``,
                           ``"nearest"``, ``"cubic"``, ``"lanczos"``.
                           Defaults to ``"bilinear"``.

    Returns:
        *dest* — the path to the saved GeoTIFF.

    Raises:
        ValueError: If *resampling* is not a known algorithm name.
    """
    if resampling not in _RESAMPLING_MAP:
        raise ValueError(
            f"Unknown resampling algorithm {resampling!r}. "
            f"Choose from: {sorted(_RESAMPLING_MAP)}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    target_crs = f"EPSG:{aoi.epsg}"
    resamp = _RESAMPLING_MAP[resampling]

    # Open without masked=True so that integer nodata values are preserved
    # as-is and not converted to NaN (which cannot be stored back to int dtypes).
    da: xr.DataArray = rioxarray.open_rasterio(src)

    # Reproject to target CRS and resolution in a single step.
    da_reproj: xr.DataArray = da.rio.reproject(
        target_crs,
        resolution=target_resolution,
        resampling=resamp,
    )

    # Transform the AOI bbox from WGS-84 to the target CRS so that clip_box
    # operates in projected coordinates (avoids NoDataInBounds when the raster
    # is already in UTM).
    west_geo, south_geo, east_geo, north_geo = aoi.bbox
    west_proj, south_proj, east_proj, north_proj = transform_bounds(
        "EPSG:4326",
        target_crs,
        west_geo,
        south_geo,
        east_geo,
        north_geo,
    )

    da_clipped: xr.DataArray = da_reproj.rio.clip_box(
        minx=west_proj,
        miny=south_proj,
        maxx=east_proj,
        maxy=north_proj,
    )

    da_clipped.rio.to_raster(dest, driver="GTiff")
    return dest


def download_scene_bands(
    scene: SceneMetadata,
    bands: list[str],
    aoi: AOI,
    token: str,
    raw_dir: Path,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[str, Path]:
    """Download, clip, and reproject all requested bands for one scene.

    For each band in *bands* that appears in ``scene.assets``:

    1. Downloads the raw JP2 to a temporary location inside *raw_dir*.
    2. Clips and reprojects it to *aoi* at 10 m resolution (upsampling 20 m
       bands with bilinear interpolation; nearest-neighbour for SCL).
    3. Removes the raw download.

    Bands absent from ``scene.assets`` are silently skipped.

    Args:
        scene:       Scene whose assets map is consulted for download URLs.
        bands:       Band names to download, e.g. ``["B02", "B08", "SCL"]``.
        aoi:         Area of interest for clipping and CRS selection.
        token:       OAuth2 Bearer token.
        raw_dir:     Root directory for raw downloads and processed outputs.
        max_retries: Retry budget passed through to :func:`download_band`.

    Returns:
        Mapping of band name to the path of the processed (clipped/reprojected)
        GeoTIFF.  Only bands that were successfully downloaded are included.
    """
    result: dict[str, Path] = {}
    scene_dir = raw_dir / scene.scene_id

    for band in bands:
        href = scene.assets.get(band)
        if href is None:
            continue

        # Determine resampling strategy: SCL uses nearest, spectral bands bilinear.
        is_20m = band in BANDS_20M
        resampling = "nearest" if band == "SCL" else "bilinear"

        # Raw download path (temporary JP2).
        raw_path = scene_dir / "raw" / f"{band}.jp2"
        # Final processed path.
        processed_path = scene_dir / f"{band}.tif"

        # Step 1: download.
        download_band(href, raw_path, token, max_retries=max_retries)

        # Step 2: clip, reproject, resample to 10 m.
        clip_and_reproject(
            raw_path,
            processed_path,
            aoi=aoi,
            target_resolution=10,
            resampling=resampling,
        )

        # Step 3: clean up the raw file to save disk space.
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass

        result[band] = processed_path

    return result
