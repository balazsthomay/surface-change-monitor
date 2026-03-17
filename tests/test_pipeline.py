"""Tests for surface_change_monitor.pipeline and run_pipeline CLI.

Strategy:
- Mock ALL external calls (auth, STAC search, download, composite, predict)
- test_pipeline_end_to_end_mock: full happy-path with mocked externals
- test_handles_no_scenes: empty STAC search -> PipelineError raised
- test_handles_unreliable_month: reliable=False composite -> warning logged, continues
- test_cli_argument_parsing: argparse --aoi, --start, --end, --model, --output
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_composite(year_month: str, reliable: bool = True) -> MagicMock:
    """Return a mock MonthlyComposite with synthetic data."""
    from surface_change_monitor.config import BERGEN_AOI
    from surface_change_monitor.composite import MonthlyComposite

    height, width = 32, 32
    west, south = 297000.0, 6690000.0
    res = 10.0

    x_coords = np.linspace(west + res / 2, west + width * res - res / 2, width)
    y_coords = np.linspace(south + height * res - res / 2, south + res / 2, height)

    bands = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDBI", "NDWI"]
    data = xr.DataArray(
        np.random.default_rng(42).random((len(bands), height, width), dtype=None).astype(
            np.float32
        ),
        dims=["band", "y", "x"],
        coords={"band": bands, "y": y_coords, "x": x_coords},
    )
    import rioxarray  # noqa: F401
    data = data.rio.write_crs("EPSG:32632")

    obs = xr.DataArray(
        np.ones((height, width), dtype=np.int32),
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )

    return MonthlyComposite(
        data=data,
        year_month=year_month,
        n_scenes=3 if reliable else 1,
        clear_obs_count=obs,
        reliable=reliable,
        aoi=BERGEN_AOI,
    )


def _make_prob_map() -> xr.DataArray:
    """Return a synthetic (32, 32) probability DataArray in EPSG:32632."""
    import rioxarray  # noqa: F401

    height, width = 32, 32
    west, south = 297000.0, 6690000.0
    res = 10.0
    x_coords = np.linspace(west + res / 2, west + width * res - res / 2, width)
    y_coords = np.linspace(south + height * res - res / 2, south + res / 2, height)

    data = np.zeros((height, width), dtype=np.float32)
    # Insert a 10x10 block so vectorize_changes produces at least one polygon
    data[5:15, 5:15] = 0.9

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:32632")
    da = da.rio.write_transform()
    return da


def _make_scene(year_month: str, day: int = 15) -> MagicMock:
    """Return a minimal mock SceneMetadata."""
    from datetime import datetime, timezone

    scene = MagicMock()
    scene.scene_id = f"S2A_MSIL2A_{year_month.replace('-', '')}T{day:02d}0000_fake"
    scene.datetime = datetime(int(year_month[:4]), int(year_month[5:]), day, tzinfo=timezone.utc)
    scene.assets = {"B02": "https://fake/B02.jp2", "B08": "https://fake/B08.jp2"}
    return scene


# ---------------------------------------------------------------------------
# 17.1 Failing tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEndMock:
    """test_pipeline_end_to_end_mock: All externals mocked, runs to GeoJSON output."""

    def test_produces_geojson_output(self, tmp_path: Path) -> None:
        """Pipeline produces a GeoJSON file in output_dir when mocked externals succeed."""
        import geopandas as gpd

        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "output"

        # Two months so we get one pair
        scenes_2021_01 = [_make_scene("2021-01")]
        scenes_2021_02 = [_make_scene("2021-02")]
        all_scenes = scenes_2021_01 + scenes_2021_02

        composite_01 = _make_composite("2021-01", reliable=True)
        composite_02 = _make_composite("2021-02", reliable=True)

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("user", "pass"),
            ),
            patch(
                "surface_change_monitor.pipeline.TokenManager"
            ) as mock_tm_cls,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=all_scenes,
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={"B02": tmp_path / "B02.tif"},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                side_effect=[composite_01, composite_02],
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
            patch(
                "surface_change_monitor.pipeline.predict_change",
                return_value=_make_prob_map(),
            ),
        ):
            mock_tm_cls.return_value.get_token.return_value = "fake-token"

            result_path = run_pipeline(
                aoi_name="bergen",
                start_date="2021-01",
                end_date="2021-02",
                model_path=model_path,
                output_dir=output_dir,
            )

        assert result_path.exists(), f"Expected output file at {result_path}"
        assert result_path.suffix in {".geojson", ".gpkg"}, (
            f"Expected GeoJSON or GeoPackage, got {result_path.suffix}"
        )

        gdf = gpd.read_file(str(result_path))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "geometry" in gdf.columns

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Pipeline creates output_dir if it does not exist."""
        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "new_subdir" / "output"

        scenes = [_make_scene("2021-01"), _make_scene("2021-02")]
        composite_01 = _make_composite("2021-01")
        composite_02 = _make_composite("2021-02")

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=scenes,
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                side_effect=[composite_01, composite_02],
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
            patch(
                "surface_change_monitor.pipeline.predict_change",
                return_value=_make_prob_map(),
            ),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            run_pipeline("bergen", "2021-01", "2021-02", model_path, output_dir)

        assert output_dir.exists(), "Pipeline must create output_dir"

    def test_result_has_detection_period(self, tmp_path: Path) -> None:
        """Each row in the output file has a detection_period attribute."""
        import geopandas as gpd

        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "output"

        scenes = [_make_scene("2021-01"), _make_scene("2021-02")]
        composite_01 = _make_composite("2021-01")
        composite_02 = _make_composite("2021-02")

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=scenes,
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                side_effect=[composite_01, composite_02],
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
            patch(
                "surface_change_monitor.pipeline.predict_change",
                return_value=_make_prob_map(),
            ),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            result_path = run_pipeline(
                "bergen", "2021-01", "2021-02", model_path, output_dir
            )

        gdf = gpd.read_file(str(result_path))
        if len(gdf) > 0:
            assert "detection_period" in gdf.columns, (
                "Output must contain detection_period column"
            )


class TestHandlesNoScenes:
    """test_handles_no_scenes: Empty STAC search yields graceful error."""

    def test_raises_when_no_scenes_found(self, tmp_path: Path) -> None:
        """PipelineError (or ValueError) raised when search returns empty list."""
        from surface_change_monitor.pipeline import PipelineError, run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "output"

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=[],
            ),
            pytest.raises(PipelineError),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            run_pipeline("bergen", "2021-01", "2021-12", model_path, output_dir)

    def test_error_message_mentions_no_scenes(self, tmp_path: Path) -> None:
        """Error message should mention that no scenes were found."""
        from surface_change_monitor.pipeline import PipelineError, run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=[],
            ),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            with pytest.raises(PipelineError, match="[Nn]o scenes"):
                run_pipeline(
                    "bergen", "2021-01", "2021-12", model_path, tmp_path / "out"
                )

    def test_raises_when_fewer_than_two_months(self, tmp_path: Path) -> None:
        """Only one month of data is not enough to form a change pair -> PipelineError."""
        from surface_change_monitor.pipeline import PipelineError, run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()

        composite_01 = _make_composite("2021-01")

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=[_make_scene("2021-01")],
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                return_value=composite_01,
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            with pytest.raises(PipelineError):
                run_pipeline(
                    "bergen", "2021-01", "2021-01", model_path, tmp_path / "out"
                )


class TestHandlesUnreliableMonth:
    """test_handles_unreliable_month: Unreliable composite logged but pipeline continues."""

    def test_unreliable_month_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When a composite has reliable=False, a WARNING is logged but processing continues."""
        import logging

        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "output"

        scenes = [_make_scene("2021-01"), _make_scene("2021-02")]
        # First composite is unreliable (only 1 scene)
        composite_01 = _make_composite("2021-01", reliable=False)
        composite_02 = _make_composite("2021-02", reliable=True)

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=scenes,
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                side_effect=[composite_01, composite_02],
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
            patch(
                "surface_change_monitor.pipeline.predict_change",
                return_value=_make_prob_map(),
            ),
            caplog.at_level(logging.WARNING, logger="surface_change_monitor.pipeline"),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            run_pipeline("bergen", "2021-01", "2021-02", model_path, output_dir)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("unreliable" in m.lower() or "2021-01" in m for m in warning_messages), (
            f"Expected a warning about unreliable month; got: {warning_messages}"
        )

    def test_unreliable_month_still_produces_output(self, tmp_path: Path) -> None:
        """Pipeline produces output even when one month is unreliable."""
        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        output_dir = tmp_path / "output"

        scenes = [_make_scene("2021-01"), _make_scene("2021-02")]
        composite_01 = _make_composite("2021-01", reliable=False)
        composite_02 = _make_composite("2021-02", reliable=True)

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=scenes,
            ),
            patch(
                "surface_change_monitor.pipeline.download_scene_bands",
                return_value={},
            ),
            patch(
                "surface_change_monitor.pipeline.create_monthly_composite",
                side_effect=[composite_01, composite_02],
            ),
            patch(
                "surface_change_monitor.pipeline.add_indices_to_composite",
                side_effect=lambda c: c,
            ),
            patch(
                "surface_change_monitor.pipeline.predict_change",
                return_value=_make_prob_map(),
            ),
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            result_path = run_pipeline(
                "bergen", "2021-01", "2021-02", model_path, output_dir
            )

        assert result_path.exists(), "Pipeline must still produce output for unreliable month"


def _load_run_pipeline_module():
    """Load run_pipeline.py as a module without executing main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_pipeline",
        "/Users/thomaybalazs/Projects/surface-change-monitor/run_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class TestCliArgumentParsing:
    """test_cli_argument_parsing: --aoi, --start, --end, --model, --output."""

    def test_required_args_parsed(self, tmp_path: Path) -> None:
        """All required CLI args are parsed into the correct types."""
        model_path = tmp_path / "model.ckpt"
        output_dir = tmp_path / "output"

        test_argv = [
            "--aoi", "bergen",
            "--start", "2021-01",
            "--end", "2021-12",
            "--model", str(model_path),
            "--output", str(output_dir),
        ]

        mod = _load_run_pipeline_module()
        parser = mod.build_parser()
        args = parser.parse_args(test_argv)

        assert args.aoi == "bergen"
        assert args.start == "2021-01"
        assert args.end == "2021-12"
        assert args.model == model_path
        assert args.output == output_dir

    def test_aoi_choices_valid(self, tmp_path: Path) -> None:
        """--aoi accepts 'bergen' and 'houston' but rejects unknown values."""
        mod = _load_run_pipeline_module()
        parser = mod.build_parser()

        # Valid choices must not raise
        for aoi in ["bergen", "houston"]:
            args = parser.parse_args([
                "--aoi", aoi,
                "--start", "2021-01",
                "--end", "2021-12",
                "--model", str(tmp_path / "m.ckpt"),
                "--output", str(tmp_path / "out"),
            ])
            assert args.aoi == aoi

    def test_aoi_rejects_invalid_value(self, tmp_path: Path) -> None:
        """--aoi rejects unknown study area names."""
        mod = _load_run_pipeline_module()
        parser = mod.build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([
                "--aoi", "oslo",  # not a valid choice
                "--start", "2021-01",
                "--end", "2021-12",
                "--model", str(tmp_path / "m.ckpt"),
                "--output", str(tmp_path / "out"),
            ])

    def test_model_is_path_type(self, tmp_path: Path) -> None:
        """--model arg is parsed as a Path object."""
        mod = _load_run_pipeline_module()
        parser = mod.build_parser()
        args = parser.parse_args([
            "--aoi", "houston",
            "--start", "2022-06",
            "--end", "2022-08",
            "--model", str(tmp_path / "best.ckpt"),
            "--output", str(tmp_path / "out"),
        ])

        assert isinstance(args.model, Path), f"Expected Path, got {type(args.model)}"
        assert isinstance(args.output, Path), f"Expected Path, got {type(args.output)}"

    def test_main_calls_run_pipeline(self, tmp_path: Path) -> None:
        """main() in run_pipeline.py calls surface_change_monitor.pipeline.run_pipeline."""
        model_path = tmp_path / "model.ckpt"
        output_dir = tmp_path / "output"

        test_argv = [
            "run_pipeline.py",
            "--aoi", "bergen",
            "--start", "2021-01",
            "--end", "2021-02",
            "--model", str(model_path),
            "--output", str(output_dir),
        ]

        with (
            patch.object(sys, "argv", test_argv),
            patch(
                "surface_change_monitor.pipeline.run_pipeline",
                return_value=output_dir / "changes.geojson",
            ) as mock_run,
        ):
            mod = _load_run_pipeline_module()
            mod.main()

        mock_run.assert_called_once_with(
            aoi_name="bergen",
            start_date="2021-01",
            end_date="2021-02",
            model_path=model_path,
            output_dir=output_dir,
        )


class TestPipelineAoiLookup:
    """Verify the pipeline correctly maps AOI name strings to AOI objects."""

    def test_unknown_aoi_raises(self, tmp_path: Path) -> None:
        """Passing an unknown aoi_name raises PipelineError or ValueError."""
        from surface_change_monitor.pipeline import PipelineError, run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()

        with pytest.raises((PipelineError, ValueError, KeyError)):
            run_pipeline(
                aoi_name="atlantis",  # unknown AOI
                start_date="2021-01",
                end_date="2021-02",
                model_path=model_path,
                output_dir=tmp_path / "out",
            )

    def test_bergen_aoi_used(self, tmp_path: Path) -> None:
        """'bergen' maps to BERGEN_AOI; search_scenes is called with the correct AOI."""
        from surface_change_monitor.config import BERGEN_AOI
        from surface_change_monitor.pipeline import run_pipeline

        model_path = tmp_path / "model.ckpt"
        model_path.touch()

        with (
            patch(
                "surface_change_monitor.pipeline.get_cdse_credentials",
                return_value=("u", "p"),
            ),
            patch("surface_change_monitor.pipeline.TokenManager") as mock_tm,
            patch(
                "surface_change_monitor.pipeline.search_scenes",
                return_value=[],
            ) as mock_search,
        ):
            mock_tm.return_value.get_token.return_value = "tok"
            try:
                run_pipeline("bergen", "2021-01", "2021-02", model_path, tmp_path / "out")
            except Exception:
                pass  # We only care that search_scenes was called with BERGEN_AOI

        mock_search.assert_called_once()
        call_args = mock_search.call_args
        # search_scenes may be called with positional or keyword args
        passed_aoi = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("aoi")
        )
        assert passed_aoi == BERGEN_AOI, (
            f"Expected BERGEN_AOI to be passed to search_scenes; got {passed_aoi}"
        )
