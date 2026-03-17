"""Pipeline orchestration for the impervious surface change detection system.

Ties together authentication, STAC search, band download, monthly compositing,
spectral index computation, change prediction, and post-processing into a single
end-to-end function that produces a GeoJSON/GeoPackage output file.

Typical usage
-------------
>>> from pathlib import Path
>>> from surface_change_monitor.pipeline import run_pipeline
>>> out = run_pipeline(
...     aoi_name="bergen",
...     start_date="2021-01",
...     end_date="2021-12",
...     model_path=Path("models/checkpoints/best.ckpt"),
...     output_dir=Path("output/bergen_2021"),
... )
>>> print(out)  # Path to the GeoJSON file
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from surface_change_monitor.auth import TokenManager
from surface_change_monitor.composite import create_monthly_composite, group_scenes_by_month
from surface_change_monitor.config import BERGEN_AOI, HOUSTON_AOI, AOI, get_cdse_credentials
from surface_change_monitor.download import download_scene_bands
from surface_change_monitor.indices import add_indices_to_composite
from surface_change_monitor.model.predict import predict_change
from surface_change_monitor.postprocess import vectorize_changes
from surface_change_monitor.stac import search_scenes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported study areas
# ---------------------------------------------------------------------------

_AOI_MAP: dict[str, AOI] = {
    "bergen": BERGEN_AOI,
    "houston": HOUSTON_AOI,
}

# Bands to download per scene
_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot proceed due to missing or insufficient data."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    aoi_name: str,
    start_date: str,
    end_date: str,
    model_path: Path,
    output_dir: Path,
    *,
    max_cloud_cover: float = 50.0,
    threshold: float = 0.5,
    min_area_m2: float = 200.0,
    tile_size: int = 256,
    overlap: int = 64,
) -> Path:
    """Run the full impervious surface change detection pipeline.

    Pipeline steps
    --------------
    1. Resolve AOI name to an :class:`~surface_change_monitor.config.AOI`.
    2. Authenticate with CDSE to obtain a Bearer token.
    3. Search for Sentinel-2 scenes covering the date range.
    4. Group scenes by calendar month.
    5. For each month: download bands -> build composite -> add spectral indices.
    6. For each consecutive composite pair (month N, month N+1):
       - Predict change probability map.
       - Vectorize changes to polygons.
    7. Concatenate all monthly change GeoDataFrames.
    8. Save output to GeoJSON (and a companion GeoPackage) in *output_dir*.
    9. Return the path to the GeoJSON output file.

    Parameters
    ----------
    aoi_name:
        Study area key: ``"bergen"`` or ``"houston"``.
    start_date:
        Start of date range in ``"YYYY-MM"`` format.
    end_date:
        End of date range in ``"YYYY-MM"`` format.
    model_path:
        Path to a Lightning ``.ckpt`` checkpoint file.
    output_dir:
        Directory where output files will be written (created if absent).
    max_cloud_cover:
        Maximum cloud cover percentage for STAC scene search.
    threshold:
        Probability threshold for change detection.
    min_area_m2:
        Minimum polygon area in square metres for output.
    tile_size:
        Tile size in pixels for inference.
    overlap:
        Tile overlap in pixels for inference.

    Returns
    -------
    Path
        Path to the primary GeoJSON output file.

    Raises
    ------
    PipelineError
        If the AOI name is unknown, no scenes are found, or fewer than two
        monthly composites can be built.
    ValueError
        If *aoi_name* is not a recognised study area.
    """
    # ------------------------------------------------------------------
    # 1. Resolve AOI
    # ------------------------------------------------------------------
    if aoi_name not in _AOI_MAP:
        raise PipelineError(
            f"Unknown AOI name {aoi_name!r}. Choose from: {sorted(_AOI_MAP)}"
        )
    aoi = _AOI_MAP[aoi_name]
    logger.info("Pipeline starting: aoi=%s, start=%s, end=%s", aoi_name, start_date, end_date)

    # ------------------------------------------------------------------
    # 2. Authenticate
    # ------------------------------------------------------------------
    username, password = get_cdse_credentials()
    token_manager = TokenManager(username, password)
    token = token_manager.get_token()
    logger.info("CDSE authentication successful")

    # ------------------------------------------------------------------
    # 3. Search for scenes
    # ------------------------------------------------------------------
    # Expand YYYY-MM to YYYY-MM-01 / YYYY-MM-28 to ensure whole-month coverage
    search_start = f"{start_date}-01"
    search_end = f"{end_date}-28"

    scenes = search_scenes(
        aoi=aoi,
        start_date=search_start,
        end_date=search_end,
        max_cloud_cover=max_cloud_cover,
    )

    if not scenes:
        raise PipelineError(
            f"No scenes found for AOI={aoi_name!r} between {start_date} and {end_date}. "
            "Check date range, cloud cover filter, or CDSE connectivity."
        )
    logger.info("Found %d scenes", len(scenes))

    # ------------------------------------------------------------------
    # 4. Group scenes by month
    # ------------------------------------------------------------------
    month_groups = group_scenes_by_month(scenes)
    sorted_months = sorted(month_groups.keys())
    logger.info("Months with scenes: %s", sorted_months)

    # ------------------------------------------------------------------
    # 5. Build monthly composites
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    composites: dict[str, object] = {}  # year_month -> MonthlyComposite

    for year_month in sorted_months:
        month_scenes = month_groups[year_month]
        logger.info("Processing month %s (%d scenes)", year_month, len(month_scenes))

        # Refresh token before each download batch (long operations may expire it)
        token = token_manager.get_token()

        # Download bands for each scene in this month
        band_paths_list = []
        for scene in month_scenes:
            try:
                band_paths = download_scene_bands(
                    scene=scene,
                    bands=_BANDS,
                    aoi=aoi,
                    token=token,
                    raw_dir=raw_dir,
                )
                band_paths_list.append(band_paths)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download scene %s: %s — skipping", scene.scene_id, exc
                )

        if not band_paths_list:
            logger.warning(
                "No bands downloaded for month %s — skipping composite", year_month
            )
            continue

        composite = create_monthly_composite(band_paths_list, aoi, year_month)
        composite = add_indices_to_composite(composite)

        if not composite.reliable:
            logger.warning(
                "Month %s composite is unreliable (only %d scene(s) contributed). "
                "Results may be noisy.",
                year_month,
                composite.n_scenes,
            )

        composites[year_month] = composite
        logger.info("Built composite for %s (reliable=%s)", year_month, composite.reliable)

    if len(composites) < 2:
        raise PipelineError(
            f"Need at least 2 monthly composites to detect change, "
            f"but only {len(composites)} were built. "
            "Check download success or widen date range."
        )

    # ------------------------------------------------------------------
    # 6. Predict change for each consecutive composite pair
    # ------------------------------------------------------------------
    sorted_composite_months = sorted(composites.keys())
    all_changes: list[gpd.GeoDataFrame] = []

    for i in range(len(sorted_composite_months) - 1):
        month_t1 = sorted_composite_months[i]
        month_t2 = sorted_composite_months[i + 1]
        composite_t1 = composites[month_t1]
        composite_t2 = composites[month_t2]
        detection_period = f"{month_t1}/{month_t2}"

        logger.info("Predicting change: %s -> %s", month_t1, month_t2)

        try:
            prob_map = predict_change(
                model_path=model_path,
                composite_t1=composite_t1,  # type: ignore[arg-type]
                composite_t2=composite_t2,  # type: ignore[arg-type]
                tile_size=tile_size,
                overlap=overlap,
            )
            changes = vectorize_changes(
                prob_map,
                threshold=threshold,
                min_area_m2=min_area_m2,
                detection_period=detection_period,
            )
            logger.info(
                "Detected %d change polygon(s) for %s", len(changes), detection_period
            )
            all_changes.append(changes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Change prediction failed for %s: %s — skipping pair",
                detection_period,
                exc,
            )

    # ------------------------------------------------------------------
    # 7. Concatenate all change polygons
    # ------------------------------------------------------------------
    if all_changes:
        combined = gpd.pd.concat(all_changes, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry")
        # Restore CRS if present in any of the component GDFs
        for gdf in all_changes:
            if gdf.crs is not None:
                combined = combined.set_crs(gdf.crs)
                break
    else:
        combined = gpd.GeoDataFrame(
            columns=["geometry", "confidence", "area_m2", "detection_period"],
            geometry="geometry",
        )

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    safe_aoi = aoi_name.replace(" ", "_")
    geojson_path = output_dir / f"changes_{safe_aoi}.geojson"
    gpkg_path = output_dir / f"changes_{safe_aoi}.gpkg"

    combined.to_file(str(geojson_path), driver="GeoJSON")
    combined.to_file(str(gpkg_path), driver="GPKG")

    logger.info(
        "Saved %d change polygon(s) to %s", len(combined), geojson_path
    )

    return geojson_path
