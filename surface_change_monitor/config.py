"""Configuration constants and helpers for the surface change monitor pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CDSE endpoints
# ---------------------------------------------------------------------------
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/"

# ---------------------------------------------------------------------------
# Sentinel-2 collection and band configuration
# ---------------------------------------------------------------------------
SENTINEL2_COLLECTION = "sentinel-2-l2a"

BANDS_10M: list[str] = ["B02", "B03", "B04", "B08"]
BANDS_20M: list[str] = ["B11", "B12", "SCL"]

# Scene Classification Layer values to mask out
# 3=cloud shadows, 7=unclassified, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus
SCL_MASK_VALUES: list[int] = [3, 7, 8, 9, 10]

# ---------------------------------------------------------------------------
# Area of Interest definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AOI:
    """Defines a geographic area of interest."""

    name: str
    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    epsg: int

    def to_geojson(self) -> dict:
        """Return a GeoJSON Polygon dict for this bounding box."""
        west, south, east, north = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],  # closed ring
                ]
            ],
        }


BERGEN_AOI = AOI("bergen", (5.27, 60.35, 5.40, 60.44), 32632)
HOUSTON_AOI = AOI("houston", (-95.45, 29.70, -95.30, 29.80), 32615)

# ---------------------------------------------------------------------------
# Filesystem paths (relative to project root, resolved at runtime)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
COMPOSITE_DIR = DATA_DIR / "composites"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Credentials helper
# ---------------------------------------------------------------------------


def get_cdse_credentials() -> tuple[str, str]:
    """Return (username, password) from environment variables.

    Reads CDSE_USERNAME and CDSE_PASSWORD. Raises ValueError if either is absent.
    """
    username = (os.environ.get("CDSE_USERNAME") or "").strip()
    password = (os.environ.get("CDSE_PASSWORD") or "").strip()

    if not username:
        raise ValueError(
            "CDSE_USERNAME environment variable is not set. "
            "Add it to your .env file or shell environment."
        )
    if not password:
        raise ValueError(
            "CDSE_PASSWORD environment variable is not set. "
            "Add it to your .env file or shell environment."
        )

    return username, password
