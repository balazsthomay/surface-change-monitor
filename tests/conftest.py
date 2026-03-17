"""Shared pytest fixtures for the surface change monitor test suite."""

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """A temporary data directory with the standard sub-tree."""
    for subdir in ("raw", "composites", "labels"):
        (tmp_path / subdir).mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def sample_raster_10m(tmp_path: Path) -> Path:
    """A 64×64 single-band GeoTIFF at 10 m resolution in EPSG:32632."""
    raster_path = tmp_path / "sample_10m.tif"

    height, width = 64, 64
    # Roughly covers the Bergen AOI in UTM zone 32N
    west, south, east, north = 297000.0, 6690000.0, 297640.0, 6690640.0

    transform = from_bounds(west, south, east, north, width, height)
    data = np.random.default_rng(42).integers(0, 10000, (1, height, width), dtype=np.uint16)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(32632),
        transform=transform,
    ) as dst:
        dst.write(data)

    return raster_path


@pytest.fixture()
def bergen_aoi():
    """The predefined Bergen AOI instance."""
    from surface_change_monitor.config import BERGEN_AOI

    return BERGEN_AOI


@pytest.fixture()
def mock_env_credentials(monkeypatch):
    """Inject dummy CDSE credentials into the environment."""
    monkeypatch.setenv("CDSE_USERNAME", "test_user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "test_password_123")
    return {"username": "test_user@example.com", "password": "test_password_123"}
