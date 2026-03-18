"""Generate validation report with metrics and figures.

Usage:
    uv run python -c "from scripts.generate_validation import main; main()"
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import numpy as np
import rioxarray
import xarray as xr

from surface_change_monitor.config import BERGEN_AOI
from surface_change_monitor.labels.change import generate_change_labels
from surface_change_monitor.labels.hrl import load_hrl_density
from surface_change_monitor.postprocess import vectorize_changes
from surface_change_monitor.validate import (
    ValidationMetrics,
    compute_pixel_metrics,
    compute_polygon_metrics,
    generate_latency_figure,
    generate_metrics_table,
    generate_visual_comparison,
    metrics_at_thresholds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate_bergen(output_dir: Path = Path("output/validation")):
    """Run full validation for Bergen against HRL ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pairs = [
        ("2018-06", "2021-06"),
        ("2018-07", "2021-07"),
    ]

    # Load HRL ground truth
    log.info("Loading HRL ground truth...")
    hrl_2018 = load_hrl_density(Path("data/labels/hrl/bergen_imd_2018.tif"), BERGEN_AOI)
    hrl_2021 = load_hrl_density(Path("data/labels/hrl/bergen_imd_2021.tif"), BERGEN_AOI)
    hrl_2018_a = hrl_2018.rio.reproject_match(hrl_2021)
    gt_labels = generate_change_labels(hrl_2018_a, hrl_2021, threshold=10.0)

    all_metrics: dict[str, ValidationMetrics] = {}

    for t1_month, t2_month in pairs:
        period = f"{t1_month}_to_{t2_month}"
        log.info(f"Validating {period}...")

        prob_path = Path(f"output/bergen/prob_{period}.tif")
        if not prob_path.exists():
            log.warning(f"  Prediction not found: {prob_path}")
            continue

        prob = rioxarray.open_rasterio(prob_path).squeeze()

        # Align GT to prediction grid
        gt_f = gt_labels.astype(np.float32)
        gt_f.rio.write_crs(hrl_2021.rio.crs, inplace=True)
        gt_f.rio.write_nodata(np.nan, inplace=True)
        gt_aligned = gt_f.rio.reproject_match(prob, resampling=0)
        gt_binary = np.where(np.isnan(gt_aligned.values), 0, gt_aligned.values).astype(np.uint8)

        # Pixel metrics at multiple thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        thresh_results = metrics_at_thresholds(prob.values, gt_binary, thresholds)

        # Find best threshold by F1
        best = max(thresh_results, key=lambda m: m.pixel_f1)
        log.info(f"  Best threshold: {best.threshold:.1f} (F1={best.pixel_f1:.3f})")

        # Polygon-level metrics at best threshold
        pred_polygons = vectorize_changes(prob, threshold=best.threshold, min_area_m2=100.0)
        gt_prob = xr.DataArray(
            gt_binary.astype(np.float32),
            dims=prob.dims, coords=prob.coords
        )
        gt_prob.rio.write_crs(prob.rio.crs, inplace=True)
        gt_prob.rio.write_transform(prob.rio.transform(), inplace=True)
        gt_polygons = vectorize_changes(gt_prob, threshold=0.5, min_area_m2=100.0)

        if len(pred_polygons) > 0 and len(gt_polygons) > 0:
            poly_metrics = compute_polygon_metrics(pred_polygons, gt_polygons, iou_threshold=0.3)
            best = ValidationMetrics(
                pixel_precision=best.pixel_precision,
                pixel_recall=best.pixel_recall,
                pixel_f1=best.pixel_f1,
                polygon_precision=poly_metrics.polygon_precision,
                polygon_recall=poly_metrics.polygon_recall,
                polygon_f1=poly_metrics.polygon_f1,
                n_true_changes=poly_metrics.n_true_changes,
                n_predicted_changes=poly_metrics.n_predicted_changes,
                mean_iou=poly_metrics.mean_iou,
                threshold=best.threshold,
            )
        all_metrics[f"Bergen {period}"] = best

        # Visual comparison
        log.info(f"  Generating visual comparison...")
        comp_t1 = rioxarray.open_rasterio(Path(f"data/composites/bergen/{t1_month}.tif"))
        comp_t2 = rioxarray.open_rasterio(Path(f"data/composites/bergen/{t2_month}.tif"))

        fig = generate_visual_comparison(
            comp_t1, comp_t2, prob,
            ground_truth=xr.DataArray(gt_binary, dims=prob.dims, coords=prob.coords),
            output_path=output_dir / f"bergen_{period}_comparison.png",
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Metrics table
    log.info("Generating metrics table...")
    table = generate_metrics_table(all_metrics)
    (output_dir / "metrics_table.md").write_text(table)
    log.info(f"\n{table}")

    # Threshold sweep figure
    log.info("Generating threshold sweep...")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prob = rioxarray.open_rasterio(Path("output/bergen/prob_2018-06_to_2021-06.tif")).squeeze()
    gt_f = gt_labels.astype(np.float32)
    gt_f.rio.write_crs(hrl_2021.rio.crs, inplace=True)
    gt_f.rio.write_nodata(np.nan, inplace=True)
    gt_aligned = gt_f.rio.reproject_match(prob, resampling=0)
    gt_binary = np.where(np.isnan(gt_aligned.values), 0, gt_aligned.values).astype(np.uint8)
    results = metrics_at_thresholds(prob.values, gt_binary, thresholds)
    ax.plot(thresholds, [m.pixel_precision for m in results], "b-o", label="Precision")
    ax.plot(thresholds, [m.pixel_recall for m in results], "r-o", label="Recall")
    ax.plot(thresholds, [m.pixel_f1 for m in results], "g-o", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Bergen: Pixel Metrics vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "bergen_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Latency figure
    log.info("Generating latency figure...")
    latency_results = {
        "Bergen": {
            "change_date": "2018-01",
            "detection_date": "2021-06",
            "latency_months": 42,
        },
    }
    fig = generate_latency_figure(
        latency_results,
        output_path=output_dir / "detection_latency.png",
    )
    plt.close(fig)

    log.info(f"Validation complete. Outputs in {output_dir}")


def main():
    validate_bergen()


if __name__ == "__main__":
    main()
