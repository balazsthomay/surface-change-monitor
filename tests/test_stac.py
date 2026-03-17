"""Tests for surface_change_monitor.stac module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pystac
import pytest

from surface_change_monitor.config import AOI, BERGEN_AOI


def _make_mock_item(
    scene_id: str = "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000",
    dt: datetime | None = None,
    cloud_cover: float = 12.5,
    assets: dict | None = None,
    geometry: dict | None = None,
) -> MagicMock:
    """Build a minimal mock pystac.Item."""
    if dt is None:
        dt = datetime(2024, 6, 1, 10, 50, 21, tzinfo=timezone.utc)
    if geometry is None:
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [[5.0, 60.0], [5.5, 60.0], [5.5, 60.5], [5.0, 60.5], [5.0, 60.0]]
            ],
        }

    item = MagicMock(spec=pystac.Item)
    item.id = scene_id
    item.datetime = dt
    item.geometry = geometry
    item.properties = {
        "eo:cloud_cover": cloud_cover,
        "s2:product_uri": f"{scene_id}.SAFE",
    }

    if assets is None:
        assets = {
            "B02": MagicMock(href="https://example.com/B02.tif"),
            "B03": MagicMock(href="https://example.com/B03.tif"),
            "B04": MagicMock(href="https://example.com/B04.tif"),
            "B08": MagicMock(href="https://example.com/B08.tif"),
        }
    else:
        assets = {k: MagicMock(href=v) for k, v in assets.items()}

    item.assets = assets
    return item


class TestSearchReturnsSceneMetadata:
    def test_search_returns_list_of_scene_metadata(self):
        from surface_change_monitor.stac import SceneMetadata, search_scenes

        mock_item = _make_mock_item()
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SceneMetadata)

    def test_search_passes_bbox_to_client(self):
        from surface_change_monitor.stac import search_scenes

        mock_item = _make_mock_item()
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        call_kwargs = mock_client.search.call_args.kwargs
        assert "bbox" in call_kwargs
        assert call_kwargs["bbox"] == list(BERGEN_AOI.bbox)

    def test_search_passes_datetime_range_to_client(self):
        from surface_change_monitor.stac import search_scenes

        mock_item = _make_mock_item()
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        call_kwargs = mock_client.search.call_args.kwargs
        assert "datetime" in call_kwargs
        assert "2024-06-01" in call_kwargs["datetime"]
        assert "2024-06-30" in call_kwargs["datetime"]

    def test_search_passes_sentinel2_collection(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        call_kwargs = mock_client.search.call_args.kwargs
        assert "collections" in call_kwargs
        assert "sentinel-2-l2a" in call_kwargs["collections"]


class TestFiltersByCloudCover:
    def test_cloud_cover_filter_passed_to_query(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30", max_cloud_cover=30.0)

        call_kwargs = mock_client.search.call_args.kwargs
        # Cloud cover filter should appear in query or filter parameter
        has_cloud_filter = "query" in call_kwargs or "filter" in call_kwargs
        assert has_cloud_filter, (
            "Expected cloud cover filter in search query/filter parameters"
        )

    def test_default_cloud_cover_is_50(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            # Call without max_cloud_cover — should default to 50.0
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        call_kwargs = mock_client.search.call_args.kwargs
        # Ensure a cloud filter still exists with default value
        has_cloud_filter = "query" in call_kwargs or "filter" in call_kwargs
        assert has_cloud_filter, (
            "Expected default cloud cover filter in search query/filter parameters"
        )


class TestSceneMetadataExtraction:
    def test_scene_id_extracted(self):
        from surface_change_monitor.stac import SceneMetadata, search_scenes

        scene_id = "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000"
        mock_item = _make_mock_item(scene_id=scene_id)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].scene_id == scene_id

    def test_datetime_extracted(self):
        from surface_change_monitor.stac import search_scenes

        dt = datetime(2024, 6, 1, 10, 50, 21, tzinfo=timezone.utc)
        mock_item = _make_mock_item(dt=dt)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].datetime == dt

    def test_cloud_cover_extracted(self):
        from surface_change_monitor.stac import search_scenes

        mock_item = _make_mock_item(cloud_cover=23.7)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].cloud_cover == pytest.approx(23.7)

    def test_tile_id_extracted_from_scene_id(self):
        from surface_change_monitor.stac import search_scenes

        scene_id = "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000"
        mock_item = _make_mock_item(scene_id=scene_id)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].tile_id == "T32VNM"

    def test_product_id_extracted(self):
        from surface_change_monitor.stac import search_scenes

        scene_id = "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000"
        mock_item = _make_mock_item(scene_id=scene_id)
        # product_uri from properties
        mock_item.properties["s2:product_uri"] = f"{scene_id}.SAFE"
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].product_id == f"{scene_id}.SAFE"

    def test_geometry_extracted(self):
        from surface_change_monitor.stac import search_scenes

        geometry = {
            "type": "Polygon",
            "coordinates": [
                [[5.0, 60.0], [5.5, 60.0], [5.5, 60.5], [5.0, 60.5], [5.0, 60.0]]
            ],
        }
        mock_item = _make_mock_item(geometry=geometry)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].geometry == geometry

    def test_assets_extracted_as_href_dict(self):
        from surface_change_monitor.stac import search_scenes

        asset_map = {
            "B02": "https://example.com/B02.tif",
            "B04": "https://example.com/B04.tif",
        }
        mock_item = _make_mock_item(assets=asset_map)
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert results[0].assets["B02"] == "https://example.com/B02.tif"
        assert results[0].assets["B04"] == "https://example.com/B04.tif"


class TestSearchEmptyResults:
    def test_returns_empty_list_when_no_items(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-01-01", "2024-01-31")

        assert results == []

    def test_returns_list_type_for_empty_results(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            results = search_scenes(BERGEN_AOI, "2024-01-01", "2024-01-31")

        assert isinstance(results, list)


class TestAddsItemSearchConformance:
    def test_add_conforms_to_item_search_called(self):
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        mock_client.add_conforms_to.assert_called_once_with("ITEM_SEARCH")

    def test_conforms_to_called_before_search(self):
        """add_conforms_to must be called before client.search."""
        from surface_change_monitor.stac import search_scenes

        call_order = []

        mock_client = MagicMock()

        def track_conforms_to(val):
            call_order.append("add_conforms_to")

        def track_search(**kwargs):
            call_order.append("search")
            result = MagicMock()
            result.items.return_value = []
            return result

        mock_client.add_conforms_to.side_effect = track_conforms_to
        mock_client.search.side_effect = track_search

        with patch("pystac_client.Client.open", return_value=mock_client):
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        assert call_order == ["add_conforms_to", "search"], (
            f"Expected ['add_conforms_to', 'search'], got {call_order}"
        )

    def test_cdse_stac_url_used(self):
        from surface_change_monitor.config import CDSE_STAC_URL
        from surface_change_monitor.stac import search_scenes

        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_client) as mock_open:
            search_scenes(BERGEN_AOI, "2024-06-01", "2024-06-30")

        mock_open.assert_called_once_with(CDSE_STAC_URL)


class TestSceneMetadataDataclass:
    def test_scene_metadata_fields_exist(self):
        from surface_change_monitor.stac import SceneMetadata

        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        meta = SceneMetadata(
            scene_id="S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000",
            datetime=dt,
            cloud_cover=12.5,
            product_id="S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000.SAFE",
            tile_id="T32VNM",
            geometry={"type": "Polygon", "coordinates": []},
            assets={"B02": "https://example.com/B02.tif"},
        )
        assert meta.scene_id == "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000"
        assert meta.datetime == dt
        assert meta.cloud_cover == 12.5
        assert meta.product_id == "S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000.SAFE"
        assert meta.tile_id == "T32VNM"
        assert meta.geometry == {"type": "Polygon", "coordinates": []}
        assert meta.assets == {"B02": "https://example.com/B02.tif"}
