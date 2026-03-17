"""Tests for surface_change_monitor.config module."""

import os
from pathlib import Path

import pytest


class TestConstants:
    def test_cdse_stac_url(self):
        from surface_change_monitor.config import CDSE_STAC_URL

        assert CDSE_STAC_URL == "https://stac.dataspace.copernicus.eu/v1/"

    def test_cdse_token_url(self):
        from surface_change_monitor.config import CDSE_TOKEN_URL

        assert CDSE_TOKEN_URL == (
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
            "/protocol/openid-connect/token"
        )

    def test_cdse_odata_url(self):
        from surface_change_monitor.config import CDSE_ODATA_URL

        assert CDSE_ODATA_URL == "https://catalogue.dataspace.copernicus.eu/odata/v1/"

    def test_sentinel2_collection(self):
        from surface_change_monitor.config import SENTINEL2_COLLECTION

        assert SENTINEL2_COLLECTION == "sentinel-2-l2a"

    def test_bands_10m(self):
        from surface_change_monitor.config import BANDS_10M

        assert BANDS_10M == ["B02", "B03", "B04", "B08"]

    def test_bands_20m(self):
        from surface_change_monitor.config import BANDS_20M

        assert BANDS_20M == ["B11", "B12", "SCL"]

    def test_scl_mask_values(self):
        from surface_change_monitor.config import SCL_MASK_VALUES

        assert SCL_MASK_VALUES == [3, 7, 8, 9, 10]


class TestAOIDataclass:
    def test_aoi_has_name_field(self):
        from surface_change_monitor.config import AOI

        aoi = AOI("test", (0.0, 0.0, 1.0, 1.0), 4326)
        assert aoi.name == "test"

    def test_aoi_has_bbox_field(self):
        from surface_change_monitor.config import AOI

        bbox = (5.27, 60.35, 5.40, 60.44)
        aoi = AOI("test", bbox, 32632)
        assert aoi.bbox == bbox

    def test_aoi_has_epsg_field(self):
        from surface_change_monitor.config import AOI

        aoi = AOI("test", (0.0, 0.0, 1.0, 1.0), 32632)
        assert aoi.epsg == 32632

    def test_aoi_to_geojson_returns_polygon(self):
        from surface_change_monitor.config import AOI

        aoi = AOI("test", (5.27, 60.35, 5.40, 60.44), 32632)
        geojson = aoi.to_geojson()
        assert geojson["type"] == "Polygon"

    def test_aoi_to_geojson_coordinates_form_closed_ring(self):
        from surface_change_monitor.config import AOI

        west, south, east, north = 5.27, 60.35, 5.40, 60.44
        aoi = AOI("test", (west, south, east, north), 32632)
        geojson = aoi.to_geojson()
        coords = geojson["coordinates"][0]
        # Closed ring: first == last
        assert coords[0] == coords[-1]
        # All four corners present
        assert [west, south] in coords
        assert [east, south] in coords
        assert [east, north] in coords
        assert [west, north] in coords

    def test_aoi_to_geojson_bbox_order(self):
        """bbox is (west, south, east, north) — verify corners map correctly."""
        from surface_change_monitor.config import AOI

        west, south, east, north = -95.45, 29.70, -95.30, 29.80
        aoi = AOI("test", (west, south, east, north), 32615)
        geojson = aoi.to_geojson()
        coords = geojson["coordinates"][0]
        flat = set(map(tuple, coords))
        assert (west, south) in flat
        assert (east, north) in flat


class TestPredefinedAOIs:
    def test_bergen_aoi_name(self):
        from surface_change_monitor.config import BERGEN_AOI

        assert BERGEN_AOI.name == "bergen"

    def test_bergen_aoi_bbox(self):
        from surface_change_monitor.config import BERGEN_AOI

        west, south, east, north = BERGEN_AOI.bbox
        assert west == pytest.approx(5.27)
        assert south == pytest.approx(60.35)
        assert east == pytest.approx(5.40)
        assert north == pytest.approx(60.44)

    def test_bergen_aoi_epsg(self):
        from surface_change_monitor.config import BERGEN_AOI

        assert BERGEN_AOI.epsg == 32632

    def test_houston_aoi_name(self):
        from surface_change_monitor.config import HOUSTON_AOI

        assert HOUSTON_AOI.name == "houston"

    def test_houston_aoi_bbox(self):
        from surface_change_monitor.config import HOUSTON_AOI

        west, south, east, north = HOUSTON_AOI.bbox
        assert west == pytest.approx(-95.45)
        assert south == pytest.approx(29.70)
        assert east == pytest.approx(-95.30)
        assert north == pytest.approx(29.80)

    def test_houston_aoi_epsg(self):
        from surface_change_monitor.config import HOUSTON_AOI

        assert HOUSTON_AOI.epsg == 32615


class TestDirectoryPaths:
    def test_data_dir_is_path(self):
        from surface_change_monitor.config import DATA_DIR

        assert isinstance(DATA_DIR, Path)
        assert DATA_DIR == Path("data")

    def test_raw_dir_under_data(self):
        from surface_change_monitor.config import DATA_DIR, RAW_DIR

        assert RAW_DIR == DATA_DIR / "raw"

    def test_composite_dir_under_data(self):
        from surface_change_monitor.config import COMPOSITE_DIR, DATA_DIR

        assert COMPOSITE_DIR == DATA_DIR / "composites"

    def test_labels_dir_under_data(self):
        from surface_change_monitor.config import DATA_DIR, LABELS_DIR

        assert LABELS_DIR == DATA_DIR / "labels"

    def test_models_dir_is_path(self):
        from surface_change_monitor.config import MODELS_DIR

        assert isinstance(MODELS_DIR, Path)
        assert MODELS_DIR == Path("models")

    def test_output_dir_is_path(self):
        from surface_change_monitor.config import OUTPUT_DIR

        assert isinstance(OUTPUT_DIR, Path)
        assert OUTPUT_DIR == Path("output")


class TestGetCdseCredentials:
    def test_returns_username_and_password(self, monkeypatch):
        monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
        monkeypatch.setenv("CDSE_PASSWORD", "secret")

        from surface_change_monitor import config

        # Reload to pick up monkeypatched env (dotenv may have already loaded)
        username, password = config.get_cdse_credentials()
        assert username == "user@example.com"
        assert password == "secret"

    def test_raises_if_username_missing(self, monkeypatch):
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.setenv("CDSE_PASSWORD", "secret")

        from surface_change_monitor import config

        with pytest.raises(ValueError, match="CDSE_USERNAME"):
            config.get_cdse_credentials()

    def test_raises_if_password_missing(self, monkeypatch):
        monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)

        from surface_change_monitor import config

        with pytest.raises(ValueError, match="CDSE_PASSWORD"):
            config.get_cdse_credentials()

    def test_raises_if_both_missing(self, monkeypatch):
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)

        from surface_change_monitor import config

        with pytest.raises(ValueError):
            config.get_cdse_credentials()
