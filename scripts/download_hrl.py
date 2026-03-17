"""Download HRL Imperviousness Density tiles from ArcGIS REST ImageServer.

Downloads 2018 data via the EEA ImageServer (no auth required).
For each city, exports the AOI region at 10m resolution in EPSG:3035.

Usage:
    uv run python scripts/download_hrl.py
"""

import logging
from pathlib import Path

import requests
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ArcGIS REST ImageServer endpoints (no auth required)
HRL_2018_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/"
    "GioLandPublic/HRL_ImperviousnessDensity_2018/ImageServer/exportImage"
)

# Training cities with WGS84 bounding boxes (west, south, east, north)
CITIES = {
    "bergen": (5.27, 60.35, 5.40, 60.44),
    "oslo": (10.65, 59.85, 10.85, 59.95),
    "amsterdam": (4.75, 52.30, 4.95, 52.42),
    "warsaw": (20.85, 52.15, 21.10, 52.30),
    "dublin": (-6.40, 53.28, -6.15, 53.40),
}

# WGS84 -> EPSG:3035 transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def download_hrl_2018(city: str, bbox_wgs84: tuple, output_dir: Path) -> Path:
    """Download HRL 2018 imperviousness for a city AOI via ArcGIS REST."""
    west, south, east, north = bbox_wgs84

    # Transform bbox to EPSG:3035
    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    # Calculate pixel dimensions at 10m resolution
    width = int((xmax - xmin) / 10)
    height = int((ymax - ymin) / 10)

    # Clamp to server limits
    max_pixels = 4000
    if width > max_pixels or height > max_pixels:
        log.warning(f"  Clamping {width}x{height} to {max_pixels}x{max_pixels}")
        width = min(width, max_pixels)
        height = min(height, max_pixels)

    log.info(f"  Requesting {width}x{height} pixels for bbox {xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}")

    # Request export
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "3035",
        "imageSR": "3035",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "U8",
        "noData": "255",
        "interpolation": "+RSP_NearestNeighbor",
        "f": "json",
    }

    resp = requests.get(HRL_2018_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "href" not in data:
        log.error(f"  No href in response: {data}")
        raise RuntimeError(f"Export failed for {city}: {data}")

    tiff_url = data["href"]
    log.info(f"  Downloading from {tiff_url[:80]}...")

    tiff_resp = requests.get(tiff_url, timeout=120)
    tiff_resp.raise_for_status()

    output_path = output_dir / f"{city}_imd_2018.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tiff_resp.content)
    size_mb = len(tiff_resp.content) / (1024 * 1024)
    log.info(f"  Saved {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    output_dir = Path("data/labels/hrl")

    for city, bbox in CITIES.items():
        log.info(f"Downloading HRL 2018 for {city}...")
        try:
            download_hrl_2018(city, bbox, output_dir)
        except Exception as e:
            log.error(f"  Failed for {city}: {e}")


if __name__ == "__main__":
    main()
