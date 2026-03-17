"""Extract training patches from composites and HRL change labels.

For each city, loads the HRL 2018/2021 change labels and the corresponding
composites, aligns them, and extracts 256x256 patches.

Usage:
    uv run python scripts/extract_patches.py
"""

import logging
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

from surface_change_monitor.config import AOI, BERGEN_AOI
from surface_change_monitor.labels.change import extract_patches, generate_change_labels
from surface_change_monitor.labels.hrl import load_hrl_density

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# City configs: (aoi, primary_tile, epsg)
CITIES = {
    "bergen": {
        "aoi": BERGEN_AOI,
        "epsg": 32632,
    },
    "oslo": {
        "aoi": AOI("oslo", (10.65, 59.85, 10.85, 59.95), 32632),
        "epsg": 32632,
    },
    "amsterdam": {
        "aoi": AOI("amsterdam", (4.75, 52.30, 4.95, 52.42), 32631),
        "epsg": 32631,
    },
    "warsaw": {
        "aoi": AOI("warsaw", (20.85, 52.15, 21.10, 52.30), 32634),
        "epsg": 32634,
    },
    "dublin": {
        "aoi": AOI("dublin", (-6.40, 53.28, -6.15, 53.40), 32629),
        "epsg": 32629,
    },
}


def load_composite(path: Path) -> xr.DataArray:
    """Load a saved monthly composite GeoTIFF."""
    return rioxarray.open_rasterio(path)


def create_mean_composite(composite_dir: Path) -> xr.DataArray | None:
    """Create a mean composite from all monthly composites in a directory."""
    tifs = sorted(composite_dir.glob("*.tif"))
    if not tifs:
        return None

    arrays = [rioxarray.open_rasterio(t) for t in tifs]
    # Use the first as reference grid
    ref = arrays[0]

    # Align all to reference grid
    aligned = [ref]
    for arr in arrays[1:]:
        aligned.append(arr.rio.reproject_match(ref))

    # Stack and take mean
    stacked = xr.concat(aligned, dim="time")
    mean = stacked.mean(dim="time").astype(np.float32)
    # Copy CRS from reference
    mean.rio.write_crs(ref.rio.crs, inplace=True)
    mean.rio.write_transform(ref.rio.transform(), inplace=True)
    return mean


def extract_city_patches(
    city: str,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("data/patches"),
    patch_size: int = 256,
    stride: int = 128,
) -> int:
    """Extract patches for a city from composites and HRL labels."""
    config = CITIES[city]
    aoi = config["aoi"]

    hrl_dir = data_dir / "labels" / "hrl"
    hrl_2018_path = hrl_dir / f"{city}_imd_2018.tif"
    hrl_2021_path = hrl_dir / f"{city}_imd_2021.tif"

    if not hrl_2018_path.exists() or not hrl_2021_path.exists():
        log.warning(f"  HRL tiles missing for {city}, skipping")
        return 0

    # Load and align HRL data
    log.info(f"  Loading HRL tiles...")
    hrl_2018 = load_hrl_density(hrl_2018_path, aoi)
    hrl_2021 = load_hrl_density(hrl_2021_path, aoi)

    # Align to same grid
    hrl_2018_aligned = hrl_2018.rio.reproject_match(hrl_2021)

    # Generate change labels
    labels = generate_change_labels(hrl_2018_aligned, hrl_2021, threshold=10.0)
    change_pct = float(labels.sum()) / max(labels.size, 1) * 100
    log.info(f"  Change: {int(labels.sum())} pixels ({change_pct:.1f}%)")

    # Load composites for T1 (2018) and T2 (2021)
    composite_dir_2018 = data_dir / "composites" / city / "2018"
    composite_dir_2021 = data_dir / "composites" / city

    # Try to load composites; fall back to mean of available months
    comp_2018 = None
    comp_2021 = None

    # Check for 2018 composites
    tifs_2018 = sorted((data_dir / "composites" / city).glob("2018-*.tif"))
    tifs_2021 = sorted((data_dir / "composites" / city).glob("2021-*.tif"))

    if tifs_2018:
        log.info(f"  Loading {len(tifs_2018)} composites for 2018...")
        arrays = [rioxarray.open_rasterio(t) for t in tifs_2018]
        if len(arrays) > 1:
            stacked = xr.concat(arrays, dim="time")
            comp_2018 = stacked.mean(dim="time").astype(np.float32)
        else:
            comp_2018 = arrays[0].astype(np.float32)
        comp_2018.rio.write_crs(arrays[0].rio.crs, inplace=True)

    if tifs_2021:
        log.info(f"  Loading {len(tifs_2021)} composites for 2021...")
        arrays = [rioxarray.open_rasterio(t) for t in tifs_2021]
        if len(arrays) > 1:
            stacked = xr.concat(arrays, dim="time")
            comp_2021 = stacked.mean(dim="time").astype(np.float32)
        else:
            comp_2021 = arrays[0].astype(np.float32)
        comp_2021.rio.write_crs(arrays[0].rio.crs, inplace=True)

    if comp_2018 is None or comp_2021 is None:
        log.warning(f"  Missing composites for {city} (2018={len(tifs_2018)}, 2021={len(tifs_2021)}), skipping patches")
        return 0

    # Align labels to composite grid (composites cover less area than labels)
    log.info(f"  Aligning labels to composite grid...")
    # Use the 2021 composite as reference (typically better quality)
    ref = comp_2021
    labels_f = labels.astype(np.float32)
    labels_f.rio.write_crs(labels.rio.crs, inplace=True)
    labels_f.rio.write_nodata(np.nan, inplace=True)
    labels_f = labels_f.rio.reproject_match(ref.isel(band=0), resampling=0)
    # Convert back to uint8 after reprojection
    labels = xr.where(np.isnan(labels_f), 0, labels_f).astype(np.uint8)
    comp_2018_aligned = comp_2018.rio.reproject_match(ref, resampling=0)
    comp_2021_aligned = comp_2021

    # Extract patches
    log.info(f"  Extracting {patch_size}x{patch_size} patches (stride={stride})...")
    patches = extract_patches(
        comp_2018_aligned, comp_2021_aligned, labels,
        patch_size=patch_size, stride=stride,
    )

    # Save patches as .npz
    city_patch_dir = output_dir / city
    city_patch_dir.mkdir(parents=True, exist_ok=True)
    for i, patch in enumerate(patches):
        patch["city"] = city
        patch["source"] = "hrl"
        np.savez(
            city_patch_dir / f"patch_{i:04d}.npz",
            **patch,
        )

    log.info(f"  Saved {len(patches)} patches to {city_patch_dir}")
    return len(patches)


def main():
    total = 0
    for city in CITIES:
        log.info(f"Processing {city}...")
        n = extract_city_patches(city)
        total += n
    log.info(f"Total patches: {total}")


if __name__ == "__main__":
    main()
