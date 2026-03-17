"""Download Sentinel-2 imagery and create monthly composites for training.

Usage:
    uv run python scripts/acquire_data.py --city bergen --year 2021 --months 6 7 8
"""

import argparse
import logging
from pathlib import Path

from surface_change_monitor.auth import TokenManager
from surface_change_monitor.composite import create_monthly_composite, group_scenes_by_month
from surface_change_monitor.config import (
    BANDS_10M,
    BANDS_20M,
    BERGEN_AOI,
    HOUSTON_AOI,
    get_cdse_credentials,
)
from surface_change_monitor.download import download_scene_bands
from surface_change_monitor.indices import add_indices_to_composite
from surface_change_monitor.stac import search_scenes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AOIS = {"bergen": BERGEN_AOI, "houston": HOUSTON_AOI}

# Primary Sentinel-2 tiles for each AOI (reduces redundant downloads)
PRIMARY_TILES = {"bergen": "T32VLN", "houston": None}
MAX_SCENES_PER_MONTH = 5


def acquire_composites(
    city: str,
    year: int,
    months: list[int],
    max_cloud_cover: float = 40.0,
    data_dir: Path = Path("data"),
) -> list[Path]:
    """Download scenes and create monthly composites for a city/year."""
    aoi = AOIS[city]
    raw_dir = data_dir / "raw" / city
    composite_dir = data_dir / "composites" / city

    username, password = get_cdse_credentials()
    tm = TokenManager(username, password)

    saved_paths: list[Path] = []

    for month in months:
        year_month = f"{year}-{month:02d}"
        start_date = f"{year}-{month:02d}-01"
        # Handle end of month
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        log.info(f"Searching scenes for {city} {year_month}...")
        token = tm.get_token()
        scenes = search_scenes(aoi, start_date, end_date, max_cloud_cover)
        # Filter to primary tile if specified
        primary_tile = PRIMARY_TILES.get(city)
        if primary_tile:
            scenes = [s for s in scenes if s.tile_id == primary_tile]
        # Sort by cloud cover and limit
        scenes.sort(key=lambda s: s.cloud_cover)
        scenes = scenes[:MAX_SCENES_PER_MONTH]
        log.info(f"  Using {len(scenes)} scenes (tile={primary_tile or 'any'})")

        if len(scenes) < 1:
            log.warning(f"  No scenes for {year_month}, skipping")
            continue

        # Download bands for each scene
        all_band_paths: list[dict[str, Path]] = []
        bands = BANDS_10M + BANDS_20M

        for i, scene in enumerate(scenes):
            log.info(f"  Downloading scene {i + 1}/{len(scenes)}: {scene.scene_id} "
                     f"(cloud={scene.cloud_cover:.1f}%)")
            token = tm.get_token()  # Refresh before each scene
            band_paths = download_scene_bands(scene, bands, aoi, token, raw_dir)
            if band_paths:
                all_band_paths.append(band_paths)
            else:
                log.warning(f"  No bands downloaded for {scene.scene_id}")

        if not all_band_paths:
            log.warning(f"  No bands downloaded for any scene in {year_month}")
            continue

        # Create composite
        log.info(f"  Creating composite for {year_month} from {len(all_band_paths)} scenes...")
        composite = create_monthly_composite(all_band_paths, aoi, year_month)
        log.info(f"  Composite: reliable={composite.reliable}, n_scenes={composite.n_scenes}")

        # Add spectral indices
        composite = add_indices_to_composite(composite)

        # Save composite
        out_path = composite_dir / f"{year_month}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composite.data.rio.to_raster(out_path, driver="GTiff")
        log.info(f"  Saved composite to {out_path}")
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Acquire Sentinel-2 composites")
    parser.add_argument("--city", required=True, choices=list(AOIS.keys()))
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--months", required=True, type=int, nargs="+")
    parser.add_argument("--max-cloud", type=float, default=40.0)
    args = parser.parse_args()

    paths = acquire_composites(args.city, args.year, args.months, args.max_cloud)
    log.info(f"Done. Created {len(paths)} composites.")


if __name__ == "__main__":
    main()
