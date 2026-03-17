"""Tests for surface_change_monitor.model.predict.

Strategy:
- Mock the model forward pass to avoid real inference (keeps tests fast)
- Use tiny composites (small spatial dims) for the bulk of tests
- Verify the full sliding-window + Gaussian blending contract
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import torch
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_composite(
    height: int = 128,
    width: int = 128,
    bands: int = 9,
    epsg: int = 32632,
) -> "MonthlyComposite":
    """Return a synthetic MonthlyComposite for testing."""
    from surface_change_monitor.composite import MonthlyComposite
    from surface_change_monitor.config import BERGEN_AOI

    west, south = 297000.0, 6690000.0
    east = west + width * 10.0  # 10 m pixels
    north = south + height * 10.0

    # Build x/y coordinates matching rioxarray conventions (centre of pixel)
    res = 10.0
    x_coords = np.linspace(west + res / 2, east - res / 2, width)
    y_coords = np.linspace(north - res / 2, south + res / 2, height)

    band_names = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "NDBI"][:bands]
    data_np = np.random.default_rng(0).random((bands, height, width)).astype(np.float32)

    da = xr.DataArray(
        data_np,
        dims=["band", "y", "x"],
        coords={"band": band_names, "y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da = da.rio.write_transform()

    obs = xr.DataArray(
        np.ones((height, width), dtype=np.int32),
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )

    return MonthlyComposite(
        data=da,
        year_month="2024-01",
        n_scenes=5,
        clear_obs_count=obs,
        reliable=True,
        aoi=BERGEN_AOI,
    )


def _mock_task_returning(value: float = 0.0) -> MagicMock:
    """Return a mock BinaryChangeDetectionTask whose forward() returns a constant logit."""

    def _forward(x: torch.Tensor) -> torch.Tensor:
        # x is (1, 2, C, H, W) -> return (1, 1, H, W) constant logits
        b, _two, _c, h, w = x.shape
        return torch.full((b, 1, h, w), value, dtype=torch.float32)

    mock = MagicMock()
    mock.eval.return_value = mock
    mock.return_value = None  # __call__ side-effect set below
    mock.side_effect = None
    mock.__call__ = lambda self, x: _forward(x)
    mock.forward = _forward
    # Make the mock callable as a normal function
    mock.configure_mock(**{"__call__": _forward})
    mock.eval.return_value = mock
    return mock


# ---------------------------------------------------------------------------
# 12.1 Failing tests
# ---------------------------------------------------------------------------


class TestReturnsProbabilityMap:
    """test_returns_probability_map: Output shape (H, W), values in [0, 1]."""

    def test_output_shape_matches_composite(self, tmp_path: Path) -> None:
        """predict_change must return DataArray of shape (H, W) == composite spatial dims."""
        from surface_change_monitor.model.predict import predict_change

        height, width = 64, 64
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.__call__ = _fwd
            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        assert result.dims == ("y", "x"), f"Expected (y, x), got {result.dims}"
        assert result.shape == (height, width), (
            f"Expected ({height}, {width}), got {result.shape}"
        )

    def test_output_values_in_zero_one(self, tmp_path: Path) -> None:
        """All probability values must lie in [0, 1]."""
        from surface_change_monitor.model.predict import predict_change

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                # Return large positive logits -> sigmoid -> ~1.0
                return torch.full((b, 1, h, w), 5.0)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        vals = result.values
        assert float(vals.min()) >= 0.0, f"Min value {vals.min()} < 0"
        assert float(vals.max()) <= 1.0, f"Max value {vals.max()} > 1"

    def test_output_dtype_float32(self, tmp_path: Path) -> None:
        """Output DataArray must have float32 dtype."""
        from surface_change_monitor.model.predict import predict_change

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestTiledPrediction:
    """test_tiled_prediction: Large AOI -> tiled + stitched."""

    def test_large_aoi_tiled_correctly(self, tmp_path: Path) -> None:
        """A 200x200 composite with tile_size=64 should be processed in multiple tiles."""
        from surface_change_monitor.model.predict import predict_change

        height, width = 200, 200
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        call_count = 0

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                nonlocal call_count
                call_count += 1
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(
                ckpt, composite, composite, tile_size=64, overlap=16
            )

        # With tile_size=64, overlap=16, stride=48, 200px -> multiple tiles
        assert call_count > 1, f"Expected multiple tile calls, got {call_count}"
        assert result.shape == (height, width)

    def test_output_shape_preserved_for_non_divisible_size(self, tmp_path: Path) -> None:
        """Output shape must match input even when dims are not evenly divisible by stride."""
        from surface_change_monitor.model.predict import predict_change

        # 100x100 is not evenly divisible by stride=48 (tile=64, overlap=16)
        height, width = 100, 100
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(
                ckpt, composite, composite, tile_size=64, overlap=16
            )

        assert result.shape == (height, width), (
            f"Expected ({height}, {width}), got {result.shape}"
        )


class TestTileOverlapBlending:
    """test_tile_overlap_blending: Overlap regions blended with Gaussian weights."""

    def test_gaussian_blending_produces_smooth_values(self, tmp_path: Path) -> None:
        """Overlap regions must receive blended (not hard-cut) predictions."""
        from surface_change_monitor.model.predict import predict_change

        height, width = 128, 128
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        call_idx = 0

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                nonlocal call_idx
                b, _two, _c, h, w = x.shape
                # Alternating values: first tile high logit, rest low
                val = 5.0 if call_idx == 0 else -5.0
                call_idx += 1
                return torch.full((b, 1, h, w), val)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(
                ckpt, composite, composite, tile_size=64, overlap=32
            )

        # With blending, the result should have values strictly between 0 and 1
        # in the overlap region (not hard 0 or 1 for all pixels)
        vals = result.values.ravel()
        # There should be some intermediate values due to blending in overlap regions
        n_between = np.sum((vals > 0.01) & (vals < 0.99))
        assert n_between > 0, (
            "Expected some blended intermediate values in overlap region, "
            f"but all values were near 0 or 1. Unique values sample: {np.unique(vals[:20])}"
        )

    def test_uniform_logit_produces_uniform_probability(self, tmp_path: Path) -> None:
        """When all tiles return the same logit, blending should yield uniform probability."""
        from surface_change_monitor.model.predict import predict_change

        height, width = 128, 128
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)  # logit=0 -> prob=0.5

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(
                ckpt, composite, composite, tile_size=64, overlap=32
            )

        vals = result.values
        # All values should be close to sigmoid(0) = 0.5
        assert np.allclose(vals, 0.5, atol=1e-4), (
            f"Expected uniform 0.5, got range [{vals.min():.4f}, {vals.max():.4f}]"
        )


class TestPreservesGeoreference:
    """test_preserves_georeference: Same CRS/transform as input composites."""

    def test_output_has_same_crs(self, tmp_path: Path) -> None:
        """CRS of output DataArray must match input composite CRS."""
        from surface_change_monitor.model.predict import predict_change

        epsg = 32632
        composite = _make_composite(height=64, width=64, epsg=epsg)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        assert result.rio.crs is not None, "Output has no CRS"
        assert result.rio.crs.to_epsg() == epsg, (
            f"Expected EPSG:{epsg}, got {result.rio.crs}"
        )

    def test_output_has_same_spatial_coords(self, tmp_path: Path) -> None:
        """x and y coordinates of output must match the input composite."""
        from surface_change_monitor.model.predict import predict_change

        height, width = 64, 64
        composite = _make_composite(height=height, width=width)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        np.testing.assert_array_almost_equal(
            result.coords["x"].values,
            composite.data.coords["x"].values,
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            result.coords["y"].values,
            composite.data.coords["y"].values,
            decimal=3,
        )

    def test_output_has_same_transform(self, tmp_path: Path) -> None:
        """Affine transform of output must match the input composite transform."""
        from surface_change_monitor.model.predict import predict_change

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        input_transform = composite.data.rio.transform()
        output_transform = result.rio.transform()
        assert input_transform == pytest.approx(output_transform, abs=1e-3), (
            f"Transform mismatch: input={input_transform}, output={output_transform}"
        )


class TestSavesGeoTiff:
    """test_saves_geotiff: Output written to file with correct geospatial metadata."""

    def test_saves_geotiff_creates_file(self, tmp_path: Path) -> None:
        """save_prediction must create a GeoTIFF file at the given path."""
        from surface_change_monitor.model.predict import predict_change, save_prediction

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")
        out_path = tmp_path / "prediction.tif"

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        save_prediction(result, out_path)

        assert out_path.exists(), f"Expected GeoTIFF at {out_path}"

    def test_saved_geotiff_has_correct_crs(self, tmp_path: Path) -> None:
        """Saved GeoTIFF must have the same CRS as the input composite."""
        from surface_change_monitor.model.predict import predict_change, save_prediction

        epsg = 32632
        composite = _make_composite(height=64, width=64, epsg=epsg)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")
        out_path = tmp_path / "prediction.tif"

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        save_prediction(result, out_path)

        with rasterio.open(out_path) as src:
            saved_epsg = src.crs.to_epsg()

        assert saved_epsg == epsg, f"Expected EPSG:{epsg}, got {saved_epsg}"

    def test_saved_geotiff_single_band_float32(self, tmp_path: Path) -> None:
        """Saved GeoTIFF must be single-band float32."""
        from surface_change_monitor.model.predict import predict_change, save_prediction

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")
        out_path = tmp_path / "prediction.tif"

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        save_prediction(result, out_path)

        with rasterio.open(out_path) as src:
            assert src.count == 1, f"Expected 1 band, got {src.count}"
            assert src.dtypes[0] == "float32", f"Expected float32, got {src.dtypes[0]}"

    def test_saved_geotiff_values_match(self, tmp_path: Path) -> None:
        """Values read back from the GeoTIFF must match the in-memory prediction."""
        from surface_change_monitor.model.predict import predict_change, save_prediction

        composite = _make_composite(height=64, width=64)
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"fake")
        out_path = tmp_path / "prediction.tif"

        with patch(
            "surface_change_monitor.model.predict.BinaryChangeDetectionTask"
        ) as MockTask:
            mock_instance = MagicMock()
            mock_instance.eval.return_value = mock_instance

            def _fwd(x):
                b, _two, _c, h, w = x.shape
                return torch.zeros(b, 1, h, w)

            mock_instance.side_effect = _fwd
            MockTask.load_from_checkpoint.return_value = mock_instance

            result = predict_change(ckpt, composite, composite, tile_size=32, overlap=8)

        save_prediction(result, out_path)

        with rasterio.open(out_path) as src:
            saved_data = src.read(1)

        np.testing.assert_array_almost_equal(result.values, saved_data, decimal=5)
