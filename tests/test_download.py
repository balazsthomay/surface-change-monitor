"""Tests for surface_change_monitor.download module."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from surface_change_monitor.config import AOI, BERGEN_AOI, BANDS_10M, BANDS_20M
from surface_change_monitor.stac import SceneMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raster(
    path: Path,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    height: int,
    width: int,
    epsg: int = 32632,
    dtype=np.uint16,
    nodata: int | None = None,
    fill_value: int = 5000,
) -> Path:
    """Write a minimal single-band GeoTIFF and return the path."""
    transform = from_bounds(west, south, east, north, width, height)
    data = np.ones((1, height, width), dtype=dtype) * np.array(fill_value, dtype=dtype)
    kwargs = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform,
    )
    if nodata is not None:
        kwargs["nodata"] = nodata
    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data)
    return path


def _make_scene(assets: dict[str, str] | None = None) -> SceneMetadata:
    """Minimal SceneMetadata used across tests."""
    if assets is None:
        assets = {
            "B02": "https://cdse.example.com/B02.jp2",
            "B03": "https://cdse.example.com/B03.jp2",
            "B04": "https://cdse.example.com/B04.jp2",
            "B08": "https://cdse.example.com/B08.jp2",
            "B11": "https://cdse.example.com/B11.jp2",
            "B12": "https://cdse.example.com/B12.jp2",
            "SCL": "https://cdse.example.com/SCL.jp2",
        }
    return SceneMetadata(
        scene_id="S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000",
        datetime=datetime(2024, 6, 1, 10, 50, 21, tzinfo=timezone.utc),
        cloud_cover=5.0,
        product_id="S2A_MSIL2A_20240601T105021_N0510_R051_T32VNM_20240601T140000.SAFE",
        tile_id="T32VNM",
        geometry={},
        assets=assets,
    )


# ---------------------------------------------------------------------------
# Minimal JP2-shaped bytes (just enough to satisfy write; we mock rioxarray)
# ---------------------------------------------------------------------------

_FAKE_JP2_BYTES = b"\x00\x00\x00\x0c\x6a\x50\x20\x20" + b"\x00" * 64


# ---------------------------------------------------------------------------
# 4.1  test_download_band_saves_file
# ---------------------------------------------------------------------------


class TestDownloadBandSavesFile:
    def test_download_band_saves_file(self, tmp_path: Path) -> None:
        """Mock HTTP with bytes, verify file is written to disk."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "B02.jp2"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [_FAKE_JP2_BYTES]
        mock_response.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", return_value=mock_response):
            result = download_band(
                href="https://cdse.example.com/B02.jp2",
                dest=dest,
                token="tok123",
            )

        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_download_band_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "nested" / "dir" / "B02.jp2"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [_FAKE_JP2_BYTES]
        mock_response.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", return_value=mock_response):
            download_band(href="https://cdse.example.com/B02.jp2", dest=dest, token="tok")

        assert dest.exists()


# ---------------------------------------------------------------------------
# 4.2  test_download_includes_auth_header
# ---------------------------------------------------------------------------


class TestDownloadIncludesAuthHeader:
    def test_download_includes_bearer_token(self, tmp_path: Path) -> None:
        """The Authorization: Bearer <token> header must be sent with every request."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "B02.jp2"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [_FAKE_JP2_BYTES]
        mock_response.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", return_value=mock_response) as mock_get:
            download_band(
                href="https://cdse.example.com/B02.jp2",
                dest=dest,
                token="my_secret_token",
            )

        call_kwargs = mock_get.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer my_secret_token"

    def test_download_sends_stream_true(self, tmp_path: Path) -> None:
        """stream=True must be set so large files aren't buffered in memory."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "B02.jp2"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [_FAKE_JP2_BYTES]
        mock_response.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", return_value=mock_response) as mock_get:
            download_band(href="https://cdse.example.com/B02.jp2", dest=dest, token="tok")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs.get("stream") is True


# ---------------------------------------------------------------------------
# 4.3  test_download_retries_on_429
# ---------------------------------------------------------------------------


class TestDownloadRetriesOn429:
    def test_download_retries_on_429_then_succeeds(self, tmp_path: Path) -> None:
        """A 429 response is retried and the file is eventually written."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "B02.jp2"

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}

        success = MagicMock()
        success.status_code = 200
        success.iter_content.return_value = [_FAKE_JP2_BYTES]
        success.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", side_effect=[rate_limited, success]):
            with patch("surface_change_monitor.download.time.sleep"):  # speed test up
                result = download_band(
                    href="https://cdse.example.com/B02.jp2",
                    dest=dest,
                    token="tok",
                )

        assert result == dest
        assert dest.exists()

    def test_download_retries_on_5xx(self, tmp_path: Path) -> None:
        """A 5xx response is also retried."""
        from surface_change_monitor.download import download_band

        dest = tmp_path / "B02.jp2"

        server_error = MagicMock()
        server_error.status_code = 503

        success = MagicMock()
        success.status_code = 200
        success.iter_content.return_value = [_FAKE_JP2_BYTES]
        success.raise_for_status.return_value = None

        with patch("surface_change_monitor.download.requests.get", side_effect=[server_error, success]):
            with patch("surface_change_monitor.download.time.sleep"):
                result = download_band(
                    href="https://cdse.example.com/B02.jp2",
                    dest=dest,
                    token="tok",
                )

        assert dest.exists()

    def test_download_raises_after_max_retries(self, tmp_path: Path) -> None:
        """Exhausting all retries raises an exception."""
        from surface_change_monitor.download import download_band, DownloadError

        dest = tmp_path / "B02.jp2"
        always_429 = MagicMock()
        always_429.status_code = 429
        always_429.headers = {}

        with patch("surface_change_monitor.download.requests.get", return_value=always_429):
            with patch("surface_change_monitor.download.time.sleep"):
                with pytest.raises(DownloadError):
                    download_band(
                        href="https://cdse.example.com/B02.jp2",
                        dest=dest,
                        token="tok",
                        max_retries=3,
                    )


# ---------------------------------------------------------------------------
# 4.4  test_clip_to_aoi
# ---------------------------------------------------------------------------


class TestClipToAoi:
    def test_clip_to_aoi_produces_smaller_extent(self, tmp_path: Path) -> None:
        """A raster larger than the AOI is clipped down to AOI bounds."""
        from surface_change_monitor.download import clip_and_reproject

        # Raster covering a broad area (UTM zone 32N)
        src_path = tmp_path / "large.tif"
        _make_raster(
            src_path,
            west=290000.0,
            south=6680000.0,
            east=310000.0,
            north=6700000.0,
            height=200,
            width=200,
            epsg=32632,
        )

        # AOI is a small subset of that raster in geographic coords
        small_aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)

        dest = tmp_path / "clipped.tif"
        clip_and_reproject(src_path, dest, aoi=small_aoi, target_resolution=10)

        with rasterio.open(dest) as ds_src, rasterio.open(dest) as ds_dst:
            pass  # just check it opens cleanly

        with rasterio.open(src_path) as ds_src:
            src_bounds = ds_src.bounds
        with rasterio.open(dest) as ds_dst:
            dst_bounds = ds_dst.bounds

        # Output width/height in metres should be smaller than input
        src_width = src_bounds.right - src_bounds.left
        src_height = src_bounds.top - src_bounds.bottom
        dst_width = dst_bounds.right - dst_bounds.left
        dst_height = dst_bounds.top - dst_bounds.bottom

        assert dst_width < src_width
        assert dst_height < src_height

    def test_clip_to_aoi_reprojects_to_aoi_crs(self, tmp_path: Path) -> None:
        """Output raster CRS matches the AOI's EPSG."""
        from surface_change_monitor.download import clip_and_reproject

        src_path = tmp_path / "large.tif"
        _make_raster(
            src_path,
            west=290000.0,
            south=6680000.0,
            east=310000.0,
            north=6700000.0,
            height=200,
            width=200,
            epsg=32632,
        )

        aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)
        dest = tmp_path / "clipped.tif"
        clip_and_reproject(src_path, dest, aoi=aoi, target_resolution=10)

        with rasterio.open(dest) as ds:
            assert ds.crs.to_epsg() == 32632

    def test_clip_to_aoi_returns_dest_path(self, tmp_path: Path) -> None:
        """clip_and_reproject returns the dest path."""
        from surface_change_monitor.download import clip_and_reproject

        src_path = tmp_path / "large.tif"
        _make_raster(
            src_path,
            west=290000.0,
            south=6680000.0,
            east=310000.0,
            north=6700000.0,
            height=200,
            width=200,
            epsg=32632,
        )

        aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)
        dest = tmp_path / "clipped.tif"
        result = clip_and_reproject(src_path, dest, aoi=aoi, target_resolution=10)

        assert result == dest


# ---------------------------------------------------------------------------
# 4.5  test_resample_20m_to_10m
# ---------------------------------------------------------------------------


class TestResample20mTo10m:
    # The test AOI (5.32, 60.37, 5.37, 60.41) covers UTM 32N roughly:
    # easting 297100–300100, northing 6698100–6702700
    # Test rasters must overlap this area to avoid NoDataInBounds errors.

    def test_resample_20m_to_10m_doubles_pixel_count(self, tmp_path: Path) -> None:
        """Upsampling a 20 m raster to 10 m should roughly double the width/height."""
        from surface_change_monitor.download import clip_and_reproject

        # 20 m pixels covering the AOI region (200 px × 20 m = 4000 m per side)
        src_path = tmp_path / "band_20m.tif"
        _make_raster(
            src_path,
            west=295000.0,
            south=6696000.0,
            east=303000.0,
            north=6704000.0,
            height=400,
            width=400,
            epsg=32632,
        )

        aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)
        dest = tmp_path / "band_10m.tif"
        clip_and_reproject(src_path, dest, aoi=aoi, target_resolution=10)

        with rasterio.open(dest) as ds:
            res_x = abs(ds.transform.a)
            res_y = abs(ds.transform.e)

        assert abs(res_x - 10.0) < 1.0, f"Expected ~10 m x-resolution, got {res_x}"
        assert abs(res_y - 10.0) < 1.0, f"Expected ~10 m y-resolution, got {res_y}"

    def test_resample_uses_bilinear_for_spectral_bands(self, tmp_path: Path) -> None:
        """Bilinear resampling should preserve approximate mean value (no extreme artefacts)."""
        from surface_change_monitor.download import clip_and_reproject

        src_path = tmp_path / "band_20m.tif"
        _make_raster(
            src_path,
            west=295000.0,
            south=6696000.0,
            east=303000.0,
            north=6704000.0,
            height=400,
            width=400,
            epsg=32632,
        )

        aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)
        dest = tmp_path / "band_10m.tif"
        clip_and_reproject(src_path, dest, aoi=aoi, target_resolution=10, resampling="bilinear")

        with rasterio.open(dest) as ds:
            data = ds.read(1)
        # All values were 5000; bilinear resampling on constant array stays ~5000
        assert data.mean() == pytest.approx(5000.0, abs=500)

    def test_resample_nearest_for_scl(self, tmp_path: Path) -> None:
        """Nearest-neighbour resampling for SCL band preserves integer class labels."""
        from surface_change_monitor.download import clip_and_reproject

        src_path = tmp_path / "SCL_20m.tif"
        _make_raster(
            src_path,
            west=295000.0,
            south=6696000.0,
            east=303000.0,
            north=6704000.0,
            height=400,
            width=400,
            epsg=32632,
            dtype=np.uint8,
            fill_value=4,  # uint8-safe SCL class value
        )

        aoi = AOI("test", (5.32, 60.37, 5.37, 60.41), 32632)
        dest = tmp_path / "SCL_10m.tif"
        clip_and_reproject(src_path, dest, aoi=aoi, target_resolution=10, resampling="nearest")

        with rasterio.open(dest) as ds:
            data = ds.read(1)
        unique_vals = set(data.flatten().tolist())
        # Nearest-neighbour should keep only the original integer class value (4)
        # plus potentially 0 at boundaries; bilinear would produce many fractional values
        assert len(unique_vals) <= 4, f"Too many unique values after nearest resampling: {unique_vals}"


# ---------------------------------------------------------------------------
# 4.6  test_download_scene_bands
# ---------------------------------------------------------------------------


class TestDownloadSceneBands:
    def test_download_scene_bands_returns_dict_of_paths(self, tmp_path: Path) -> None:
        """download_scene_bands returns a dict mapping band name -> Path."""
        from surface_change_monitor.download import download_scene_bands

        scene = _make_scene()
        bands = ["B02", "B03"]

        # Stub out download_band to avoid HTTP
        def _fake_download(href: str, dest: Path, token: str, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_FAKE_JP2_BYTES)
            return dest

        # Stub out clip_and_reproject to copy src -> dest unchanged
        def _fake_clip(src: Path, dest: Path, *, aoi: AOI, target_resolution: int, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, dest)
            return dest

        with patch("surface_change_monitor.download.download_band", side_effect=_fake_download):
            with patch("surface_change_monitor.download.clip_and_reproject", side_effect=_fake_clip):
                result = download_scene_bands(
                    scene=scene,
                    bands=bands,
                    aoi=BERGEN_AOI,
                    token="tok",
                    raw_dir=tmp_path,
                )

        assert isinstance(result, dict)
        for band in bands:
            assert band in result
            assert isinstance(result[band], Path)

    def test_download_scene_bands_downloads_each_band(self, tmp_path: Path) -> None:
        """download_scene_bands calls download_band once per requested band."""
        from surface_change_monitor.download import download_scene_bands

        scene = _make_scene()
        bands = ["B02", "B04", "B08"]

        downloaded_hrefs: list[str] = []

        def _fake_download(href: str, dest: Path, token: str, **kwargs) -> Path:
            downloaded_hrefs.append(href)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_FAKE_JP2_BYTES)
            return dest

        def _fake_clip(src: Path, dest: Path, *, aoi: AOI, target_resolution: int, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, dest)
            return dest

        with patch("surface_change_monitor.download.download_band", side_effect=_fake_download):
            with patch("surface_change_monitor.download.clip_and_reproject", side_effect=_fake_clip):
                download_scene_bands(
                    scene=scene,
                    bands=bands,
                    aoi=BERGEN_AOI,
                    token="tok",
                    raw_dir=tmp_path,
                )

        assert len(downloaded_hrefs) == len(bands)
        for band in bands:
            assert scene.assets[band] in downloaded_hrefs

    def test_download_scene_bands_clips_every_band(self, tmp_path: Path) -> None:
        """clip_and_reproject is invoked once for every downloaded band."""
        from surface_change_monitor.download import download_scene_bands

        scene = _make_scene()
        bands = ["B02", "B11"]
        clip_calls: list[tuple[Path, Path]] = []

        def _fake_download(href: str, dest: Path, token: str, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_FAKE_JP2_BYTES)
            return dest

        def _fake_clip(src: Path, dest: Path, *, aoi: AOI, target_resolution: int, **kwargs) -> Path:
            clip_calls.append((src, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, dest)
            return dest

        with patch("surface_change_monitor.download.download_band", side_effect=_fake_download):
            with patch("surface_change_monitor.download.clip_and_reproject", side_effect=_fake_clip):
                download_scene_bands(
                    scene=scene,
                    bands=bands,
                    aoi=BERGEN_AOI,
                    token="tok",
                    raw_dir=tmp_path,
                )

        assert len(clip_calls) == len(bands)

    def test_download_scene_bands_uses_20m_resolution_for_20m_bands(self, tmp_path: Path) -> None:
        """B11, B12, SCL are passed with target_resolution=10 (upsampled to 10m)."""
        from surface_change_monitor.download import download_scene_bands

        scene = _make_scene()
        bands = ["B11", "SCL"]
        resolutions_seen: list[int] = []

        def _fake_download(href: str, dest: Path, token: str, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_FAKE_JP2_BYTES)
            return dest

        def _fake_clip(src: Path, dest: Path, *, aoi: AOI, target_resolution: int, **kwargs) -> Path:
            resolutions_seen.append(target_resolution)
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, dest)
            return dest

        with patch("surface_change_monitor.download.download_band", side_effect=_fake_download):
            with patch("surface_change_monitor.download.clip_and_reproject", side_effect=_fake_clip):
                download_scene_bands(
                    scene=scene,
                    bands=bands,
                    aoi=BERGEN_AOI,
                    token="tok",
                    raw_dir=tmp_path,
                )

        # Both 20m bands should be resampled to 10m
        assert all(r == 10 for r in resolutions_seen)

    def test_download_scene_bands_skips_missing_assets(self, tmp_path: Path) -> None:
        """If a band is not in scene.assets it is silently skipped."""
        from surface_change_monitor.download import download_scene_bands

        scene = _make_scene(assets={"B02": "https://example.com/B02.jp2"})  # only B02
        bands = ["B02", "B08"]  # B08 missing

        downloaded: list[str] = []

        def _fake_download(href: str, dest: Path, token: str, **kwargs) -> Path:
            downloaded.append(href)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_FAKE_JP2_BYTES)
            return dest

        def _fake_clip(src: Path, dest: Path, *, aoi: AOI, target_resolution: int, **kwargs) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, dest)
            return dest

        with patch("surface_change_monitor.download.download_band", side_effect=_fake_download):
            with patch("surface_change_monitor.download.clip_and_reproject", side_effect=_fake_clip):
                result = download_scene_bands(
                    scene=scene,
                    bands=bands,
                    aoi=BERGEN_AOI,
                    token="tok",
                    raw_dir=tmp_path,
                )

        assert "B02" in result
        assert "B08" not in result
