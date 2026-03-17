"""Download HRL Imperviousness Density 2021 via WMS GetMap.

Usage:
    uv run python scripts/download_hrl_2021.py
"""

import logging
from pathlib import Path

import requests
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WMS_URL = "https://geoserver.geoville.com/geoserver/nvlcc/ows"
LAYER = "HRL_NVLCC_IMD_10m"

CITIES = {
    "bergen": (5.27, 60.35, 5.40, 60.44),
    "oslo": (10.65, 59.85, 10.85, 59.95),
    "amsterdam": (4.75, 52.30, 4.95, 52.42),
    "warsaw": (20.85, 52.15, 21.10, 52.30),
    "dublin": (-6.40, 53.28, -6.15, 53.40),
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def download_hrl_2021_wms(city: str, bbox_wgs84: tuple, output_dir: Path) -> Path:
    """Download HRL 2021 via WMS GetMap."""
    west, south, east, north = bbox_wgs84
    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    width = int((xmax - xmin) / 10)
    height = int((ymax - ymin) / 10)

    # WMS 1.3.0 with EPSG:3035: bbox is minY,minX,maxY,maxX (northing first for projected CRS)
    # Actually for EPSG:3035, axes are easting,northing so bbox = minE,minN,maxE,maxN
    bbox_str = f"{ymin},{xmin},{ymax},{xmax}"

    log.info(f"  Requesting {width}x{height} pixels via WMS")

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": LAYER,
        "crs": "EPSG:3035",
        "bbox": bbox_str,
        "width": str(width),
        "height": str(height),
        "format": "image/geotiff",
    }

    resp = requests.get(WMS_URL, params=params, timeout=120)
    resp.raise_for_status()

    if "tiff" not in resp.headers.get("content-type", ""):
        log.error(f"  Unexpected content: {resp.text[:200]}")
        raise RuntimeError(f"WMS did not return TIFF for {city}")

    output_path = output_dir / f"{city}_imd_2021.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    log.info(f"  Saved {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    output_dir = Path("data/labels/hrl")

    for city, bbox in CITIES.items():
        log.info(f"Downloading HRL 2021 for {city}...")
        try:
            download_hrl_2021_wms(city, bbox, output_dir)
        except Exception as e:
            log.error(f"  Failed for {city}: {e}")


if __name__ == "__main__":
    main()
