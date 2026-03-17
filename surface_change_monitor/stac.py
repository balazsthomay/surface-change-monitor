"""STAC search module for Sentinel-2 scene discovery on CDSE.

Provides :class:`SceneMetadata` and :func:`search_scenes` to query the
Copernicus Data Space Ecosystem STAC API for Sentinel-2 L2A scenes that
intersect a given area of interest within a date range and cloud cover limit.
"""

import re
from dataclasses import dataclass
from datetime import datetime

import pystac
import pystac_client

from surface_change_monitor.config import (
    AOI,
    CDSE_STAC_URL,
    SENTINEL2_COLLECTION,
)

# Regex that captures the tile ID segment from a Sentinel-2 scene ID.
# Example: S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000
#                                                        ^^^^^^
_TILE_ID_RE = re.compile(r"_(T[0-9]{2}[A-Z]{3})_")


@dataclass
class SceneMetadata:
    """Metadata for a single Sentinel-2 scene returned by STAC search.

    Attributes:
        scene_id:   The STAC item ID (Sentinel-2 granule name).
        datetime:   Sensing datetime in UTC.
        cloud_cover: Cloud cover percentage in [0, 100].
        product_id: The ``s2:product_uri`` property value (e.g. ``<scene>.SAFE``).
                    Used as an identifier for OData download.
        tile_id:    The Sentinel-2 Military Grid Reference System tile identifier,
                    e.g. ``"T32VNM"``.  Extracted from *scene_id*.
        geometry:   Scene footprint as a GeoJSON geometry dict.
        assets:     Mapping from band/asset name to its download HREF.
    """

    scene_id: str
    datetime: datetime
    cloud_cover: float
    product_id: str
    tile_id: str
    geometry: dict
    assets: dict[str, str]


def _extract_tile_id(scene_id: str) -> str:
    """Extract the Sentinel-2 tile ID from a granule scene ID.

    Args:
        scene_id: Sentinel-2 granule identifier, e.g.
            ``S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000``.

    Returns:
        The tile ID string, e.g. ``"T32VNM"``.  Returns an empty string if the
        pattern cannot be matched (graceful degradation).
    """
    match = _TILE_ID_RE.search(scene_id)
    return match.group(1) if match else ""


def _item_to_scene_metadata(item: pystac.Item) -> SceneMetadata:
    """Convert a :class:`pystac.Item` into a :class:`SceneMetadata` instance.

    Args:
        item: A pystac Item from the CDSE STAC catalogue.

    Returns:
        Populated :class:`SceneMetadata`.
    """
    cloud_cover: float = float(item.properties.get("eo:cloud_cover", 0.0))
    product_id: str = item.properties.get("s2:product_uri", item.id)
    assets: dict[str, str] = {
        name: asset.href for name, asset in item.assets.items()
    }

    return SceneMetadata(
        scene_id=item.id,
        datetime=item.datetime,
        cloud_cover=cloud_cover,
        product_id=product_id,
        tile_id=_extract_tile_id(item.id),
        geometry=item.geometry or {},
        assets=assets,
    )


def search_scenes(
    aoi: AOI,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 50.0,
) -> list[SceneMetadata]:
    """Search the CDSE STAC API for Sentinel-2 L2A scenes.

    Opens a :class:`pystac_client.Client` against the CDSE STAC endpoint,
    registers ITEM_SEARCH conformance (required by that endpoint), then
    searches the ``sentinel-2-l2a`` collection filtered by bounding box,
    date range, and cloud cover.

    Args:
        aoi:             Area of interest whose bounding box is used for spatial
                         filtering.
        start_date:      Start of the search window in ``YYYY-MM-DD`` format.
        end_date:        End of the search window in ``YYYY-MM-DD`` format.
        max_cloud_cover: Maximum allowed cloud cover percentage (inclusive).
                         Defaults to 50.0.

    Returns:
        A list of :class:`SceneMetadata` instances, one per matching STAC item.
        Returns an empty list when no scenes satisfy the filter criteria.
    """
    client = pystac_client.Client.open(CDSE_STAC_URL)
    # Required by the CDSE STAC endpoint to enable item-level search.
    client.add_conforms_to("ITEM_SEARCH")

    search = client.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=list(aoi.bbox),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
    )

    return [_item_to_scene_metadata(item) for item in search.items()]
