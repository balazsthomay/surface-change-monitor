"""Run change detection inference on composites.

Usage:
    uv run python scripts/run_inference.py --city bergen --model models/checkpoints/best.ckpt
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import rioxarray
import xarray as xr

from surface_change_monitor.composite import MonthlyComposite
from surface_change_monitor.config import AOI, BERGEN_AOI, HOUSTON_AOI
from surface_change_monitor.model.predict import predict_change, save_prediction
from surface_change_monitor.postprocess import vectorize_changes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AOIS = {
    "bergen": BERGEN_AOI,
    "houston": HOUSTON_AOI,
    "oslo": AOI("oslo", (10.65, 59.85, 10.85, 59.95), 32632),
    "amsterdam": AOI("amsterdam", (4.75, 52.30, 4.95, 52.42), 32631),
    "warsaw": AOI("warsaw", (20.85, 52.15, 21.10, 52.30), 32634),
    "dublin": AOI("dublin", (-6.40, 53.28, -6.15, 53.40), 32629),
}


def run_inference(
    city: str,
    model_path: Path,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("output"),
    tile_size: int = 128,
    threshold: float = 0.5,
    min_area_m2: float = 200.0,
):
    """Run inference on all composite pairs for a city."""
    aoi = AOIS[city]
    composite_dir = data_dir / "composites" / city
    city_output = output_dir / city
    city_output.mkdir(parents=True, exist_ok=True)

    # Find composites sorted by date
    tifs = sorted(composite_dir.glob("*.tif"))
    if len(tifs) < 2:
        log.error(f"Need at least 2 composites, found {len(tifs)}")
        return

    log.info(f"Found {len(tifs)} composites for {city}")

    # Group by epoch (2018 vs 2021)
    t1_tifs = [t for t in tifs if t.stem.startswith("2018")]
    t2_tifs = [t for t in tifs if t.stem.startswith("2021")]

    if not t1_tifs or not t2_tifs:
        log.warning("Missing either 2018 or 2021 composites, using consecutive pairs")
        # Fall back to consecutive pairs
        pairs = [(tifs[i], tifs[i + 1]) for i in range(len(tifs) - 1)]
    else:
        # Cross-epoch pairs: each 2018 month vs corresponding 2021 month
        pairs = []
        for t1 in t1_tifs:
            month = t1.stem.split("-")[1]
            matching = [t2 for t2 in t2_tifs if t2.stem.endswith(f"-{month}")]
            if matching:
                pairs.append((t1, matching[0]))
        if not pairs:
            # Just use first of each
            pairs = [(t1_tifs[0], t2_tifs[0])]

    log.info(f"Processing {len(pairs)} composite pairs")

    all_polygons = []
    for t1_path, t2_path in pairs:
        period = f"{t1_path.stem}_to_{t2_path.stem}"
        log.info(f"  Predicting: {period}")

        c1 = rioxarray.open_rasterio(t1_path)
        c2 = rioxarray.open_rasterio(t2_path)

        # Align if needed
        if c1.shape != c2.shape:
            c1 = c1.rio.reproject_match(c2)

        comp_t1 = MonthlyComposite(
            data=c1, year_month=t1_path.stem, n_scenes=5,
            clear_obs_count=xr.DataArray(), reliable=True, aoi=aoi,
        )
        comp_t2 = MonthlyComposite(
            data=c2, year_month=t2_path.stem, n_scenes=5,
            clear_obs_count=xr.DataArray(), reliable=True, aoi=aoi,
        )

        prob_map = predict_change(model_path, comp_t1, comp_t2, tile_size=tile_size)
        save_prediction(prob_map, city_output / f"prob_{period}.tif")

        polygons = vectorize_changes(prob_map, threshold=threshold, min_area_m2=min_area_m2)
        if len(polygons) > 0:
            polygons["detection_period"] = period
            all_polygons.append(polygons)
            log.info(f"    {len(polygons)} change polygons detected")
        else:
            log.info(f"    No changes detected above threshold")

    if all_polygons:
        import geopandas as gpd
        combined = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True))
        combined.to_file(city_output / "changes.geojson", driver="GeoJSON")
        combined.to_file(city_output / "changes.gpkg", driver="GPKG")
        log.info(f"Saved {len(combined)} total polygons to {city_output}")
    else:
        log.info("No changes detected in any pair")


def main():
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("--city", required=True, choices=list(AOIS.keys()))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=128)
    args = parser.parse_args()

    run_inference(args.city, args.model, tile_size=args.tile_size, threshold=args.threshold)


if __name__ == "__main__":
    main()
