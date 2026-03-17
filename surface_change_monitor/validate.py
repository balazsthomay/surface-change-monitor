"""Validation metrics for the impervious surface change detection pipeline.

Provides pixel-level and polygon-level metrics to quantify how well predicted
change maps match ground-truth annotations.  Supports:

- Pixel metrics (precision / recall / F1) computed from thresholded probability
  maps vs binary ground-truth rasters.
- Polygon IoU and polygon-level precision / recall / F1, where a predicted
  polygon is a True Positive if it overlaps any GT polygon above a configurable
  IoU threshold (default 0.3).
- Detection latency analysis: given a time series of monthly predictions and a
  known change date, reports the first month where the model exceeds a threshold
  and the number of months elapsed since the change date.
- Sweep over multiple probability thresholds to find the operating point that
  best balances precision and recall.
- Reporting: visual comparison panels, markdown metrics tables, and latency
  timeline figures for embedding in validation reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.figure
import numpy as np
import xarray as xr
from shapely import Geometry as BaseGeometry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetrics:
    """Aggregated performance metrics for one probability threshold.

    Pixel-level fields are computed by thresholding the full probability map
    and comparing to a binary ground-truth raster.  Polygon-level fields are
    computed by matching predicted change polygons to GT polygons via IoU.

    Attributes
    ----------
    pixel_precision:
        Fraction of predicted positive pixels that are truly positive.
    pixel_recall:
        Fraction of actual positive pixels that were detected.
    pixel_f1:
        Harmonic mean of pixel precision and recall.
    polygon_precision:
        Fraction of predicted change polygons that match a GT polygon
        (IoU > ``iou_threshold``).
    polygon_recall:
        Fraction of GT change polygons matched by at least one predicted
        polygon.
    polygon_f1:
        Harmonic mean of polygon precision and recall.
    n_true_changes:
        Number of ground-truth change polygons.
    n_predicted_changes:
        Number of predicted change polygons.
    mean_iou:
        Mean IoU of true-positive polygon matches.  0.0 when there are no TPs.
    threshold:
        The probability threshold used to produce this set of metrics.
    """

    pixel_precision: float
    pixel_recall: float
    pixel_f1: float
    polygon_precision: float
    polygon_recall: float
    polygon_f1: float
    n_true_changes: int
    n_predicted_changes: int
    mean_iou: float
    threshold: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(array: np.ndarray | xr.DataArray) -> np.ndarray:
    """Convert an xr.DataArray or numpy array to a plain numpy array."""
    if isinstance(array, xr.DataArray):
        return array.values
    return np.asarray(array)


def _compute_polygon_iou(poly_a: BaseGeometry, poly_b: BaseGeometry) -> float:
    """Compute the Intersection over Union (IoU) of two shapely polygons.

    Parameters
    ----------
    poly_a, poly_b:
        Any pair of shapely geometry objects.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 if either polygon has zero area.
    """
    intersection_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area
    if union_area == 0.0:
        return 0.0
    return float(intersection_area / union_area)


def _safe_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall; returns 0.0 when both are zero."""
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pixel_metrics(
    prediction: np.ndarray | xr.DataArray,
    ground_truth: np.ndarray | xr.DataArray,
    threshold: float = 0.5,
) -> ValidationMetrics:
    """Compute pixel-level precision, recall, and F1.

    Thresholds the probability map to produce a binary prediction, then
    computes confusion-matrix statistics against the binary ground-truth mask.

    Parameters
    ----------
    prediction:
        Float array of predicted change probabilities in [0, 1].  May be a
        numpy array or an xr.DataArray of any shape.
    ground_truth:
        Binary integer (0/1) array of the same shape as ``prediction``.
    threshold:
        Probability threshold above which a pixel is classified as changed.
        Defaults to 0.5.

    Returns
    -------
    ValidationMetrics
        Pixel metrics filled in; polygon fields default to 0.0 / 0 / 0.0.
    """
    pred_np = _to_numpy(prediction).ravel()
    gt_np = _to_numpy(ground_truth).ravel()

    pred_binary = (pred_np >= threshold).astype(np.int32)

    tp = int(np.sum((pred_binary == 1) & (gt_np == 1)))
    fp = int(np.sum((pred_binary == 1) & (gt_np == 0)))
    fn = int(np.sum((pred_binary == 0) & (gt_np == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = _safe_f1(precision, recall)

    return ValidationMetrics(
        pixel_precision=precision,
        pixel_recall=recall,
        pixel_f1=f1,
        polygon_precision=0.0,
        polygon_recall=0.0,
        polygon_f1=0.0,
        n_true_changes=0,
        n_predicted_changes=0,
        mean_iou=0.0,
        threshold=threshold,
    )


def compute_polygon_metrics(
    pred_polygons: "geopandas.GeoDataFrame",
    gt_polygons: "geopandas.GeoDataFrame",
    iou_threshold: float = 0.3,
) -> ValidationMetrics:
    """Compute polygon-level precision, recall, and F1 using IoU matching.

    A predicted polygon is a **True Positive** if it overlaps any GT polygon
    with an IoU exceeding ``iou_threshold``.  Each GT polygon can only be
    matched once (by the prediction with the highest IoU).  GT polygons not
    matched by any prediction are **False Negatives**; predicted polygons not
    matching any GT polygon are **False Positives**.

    Parameters
    ----------
    pred_polygons:
        GeoDataFrame of predicted change polygons.
    gt_polygons:
        GeoDataFrame of ground-truth change polygons.
    iou_threshold:
        Minimum IoU for a prediction to be counted as a TP.  Defaults to 0.3.

    Returns
    -------
    ValidationMetrics
        Polygon metrics filled in; pixel fields default to 0.0 / 0.0.
    """
    n_pred = len(pred_polygons)
    n_gt = len(gt_polygons)

    if n_pred == 0 or n_gt == 0:
        # Edge case: no predictions or no GT -> all FP or all FN
        return ValidationMetrics(
            pixel_precision=0.0,
            pixel_recall=0.0,
            pixel_f1=0.0,
            polygon_precision=0.0,
            polygon_recall=0.0,
            polygon_f1=0.0,
            n_true_changes=n_gt,
            n_predicted_changes=n_pred,
            mean_iou=0.0,
            threshold=iou_threshold,
        )

    # Build IoU matrix: shape (n_pred, n_gt)
    pred_geoms = list(pred_polygons.geometry)
    gt_geoms = list(gt_polygons.geometry)

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p_geom in enumerate(pred_geoms):
        for j, g_geom in enumerate(gt_geoms):
            iou_matrix[i, j] = _compute_polygon_iou(p_geom, g_geom)

    # Greedy matching: for each predicted polygon, find the best GT match
    # (highest IoU).  Each GT may only be claimed once.
    matched_gt: set[int] = set()
    tp_ious: list[float] = []

    # Sort predictions by their best IoU descending to prioritise high-quality
    # matches first (greedy approximation of optimal bipartite matching).
    best_iou_per_pred = iou_matrix.max(axis=1)
    pred_order = np.argsort(best_iou_per_pred)[::-1]

    for pred_idx in pred_order:
        # Find the best unmatched GT for this prediction
        sorted_gt = np.argsort(iou_matrix[pred_idx])[::-1]
        for gt_idx in sorted_gt:
            if gt_idx in matched_gt:
                continue
            iou_val = iou_matrix[pred_idx, gt_idx]
            if iou_val > iou_threshold:
                matched_gt.add(gt_idx)
                tp_ious.append(iou_val)
            break  # Only attempt the best GT match per prediction

    tp = len(tp_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = _safe_f1(precision, recall)
    mean_iou = float(np.mean(tp_ious)) if tp_ious else 0.0

    return ValidationMetrics(
        pixel_precision=0.0,
        pixel_recall=0.0,
        pixel_f1=0.0,
        polygon_precision=precision,
        polygon_recall=recall,
        polygon_f1=f1,
        n_true_changes=n_gt,
        n_predicted_changes=n_pred,
        mean_iou=mean_iou,
        threshold=iou_threshold,
    )


def detection_latency_analysis(
    predictions: list[tuple[str, np.ndarray | xr.DataArray]],
    known_change_date: str,
    threshold: float = 0.5,
) -> dict[str, int | str | None]:
    """Measure how many months after a known change the model first detects it.

    Iterates over a chronologically ordered list of ``(year_month, prediction)``
    pairs and returns the first month **on or after** ``known_change_date`` in
    which at least one pixel of the prediction map exceeds ``threshold``.

    Parameters
    ----------
    predictions:
        Ordered list of ``(year_month, prediction_map)`` pairs where
        ``year_month`` is a string in ``"YYYY-MM"`` format and
        ``prediction_map`` is a probability array of any shape.
    known_change_date:
        The ``"YYYY-MM"`` month in which the change is known to have occurred.
        Predictions before this month are ignored when computing latency.
    threshold:
        Probability threshold for detection.  A prediction is considered
        positive if **any** pixel exceeds this value.  Defaults to 0.5.

    Returns
    -------
    dict with keys:
        - ``"detection_month"`` (``str | None``): first month of detection, or
          ``None`` if the change is never detected.
        - ``"latency_months"`` (``int | None``): number of months from
          ``known_change_date`` to ``detection_month`` (0 = detected in the
          same month), or ``None`` if never detected.
    """
    # Build a simple month index for latency calculation
    def _month_to_index(ym: str) -> int:
        year, month = ym.split("-")
        return int(year) * 12 + int(month)

    change_index = _month_to_index(known_change_date)

    for year_month, pred_map in predictions:
        month_index = _month_to_index(year_month)
        if month_index < change_index:
            continue  # Skip months before the known change date

        pred_np = _to_numpy(pred_map)
        if np.any(pred_np >= threshold):
            latency = month_index - change_index
            return {
                "detection_month": year_month,
                "latency_months": latency,
            }

    return {
        "detection_month": None,
        "latency_months": None,
    }


def metrics_at_thresholds(
    prediction: np.ndarray | xr.DataArray,
    ground_truth: np.ndarray | xr.DataArray,
    thresholds: list[float],
) -> list[ValidationMetrics]:
    """Compute pixel-level validation metrics at multiple probability thresholds.

    Sweeping over thresholds allows the caller to plot precision-recall curves
    and choose an operating point appropriate for their use case.

    Parameters
    ----------
    prediction:
        Float array of predicted change probabilities in [0, 1].
    ground_truth:
        Binary integer (0/1) array of the same shape as ``prediction``.
    thresholds:
        Ordered list of probability thresholds to evaluate.

    Returns
    -------
    list[ValidationMetrics]
        One :class:`ValidationMetrics` per threshold, in the same order as
        ``thresholds``.  Returns an empty list when ``thresholds`` is empty.
    """
    return [compute_pixel_metrics(prediction, ground_truth, threshold=t) for t in thresholds]


# ---------------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------------


def generate_visual_comparison(
    composite_t1: xr.DataArray,
    composite_t2: xr.DataArray,
    prediction: xr.DataArray,
    ground_truth: xr.DataArray | None = None,
    output_path: Path | None = None,
) -> matplotlib.figure.Figure:
    """Create a multi-panel visual comparison figure.

    Renders a 1×4 panel (or 1×3 if no ground truth is provided):

    * **T1 RGB** — before-composite using bands at indices 2, 1, 0 (B04, B03, B02).
    * **T2 RGB** — after-composite using the same band ordering.
    * **Prediction heatmap** — probability map in [0, 1] rendered with the
      ``'hot'`` colormap.
    * **GT overlay** — T2 RGB with the binary ground-truth mask overlaid in
      semi-transparent red (omitted when ``ground_truth`` is ``None``).

    Multi-band composites are expected to have a leading ``bands`` dimension of
    at least 3 elements.  Single-band inputs are broadcast to a 3-channel
    grayscale image so the function still produces a valid figure.

    Parameters
    ----------
    composite_t1:
        xr.DataArray of shape ``(bands, H, W)`` for the *before* period.
    composite_t2:
        xr.DataArray of shape ``(bands, H, W)`` for the *after* period.
    prediction:
        xr.DataArray of shape ``(H, W)`` containing predicted change
        probabilities in [0, 1].
    ground_truth:
        Optional xr.DataArray of shape ``(H, W)`` containing binary (0/1)
        ground-truth labels.  When supplied, a fourth panel is added showing
        the GT mask overlaid on the T2 RGB image.
    output_path:
        When provided, the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure.  The caller is responsible for closing it.
    """
    import matplotlib.pyplot as plt

    def _to_rgb(composite: xr.DataArray) -> np.ndarray:
        """Extract bands 0-2 and normalise to [0, 1] float32."""
        arr = _to_numpy(composite)
        # If shape is (H, W) — single-band — expand to (1, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        # Take first three bands; if fewer than 3, repeat the last band
        if arr.shape[0] >= 3:
            rgb = arr[:3].transpose(1, 2, 0).astype(np.float32)
        elif arr.shape[0] == 2:
            rgb = np.stack([arr[0], arr[1], arr[1]], axis=-1).astype(np.float32)
        else:
            rgb = np.stack([arr[0], arr[0], arr[0]], axis=-1).astype(np.float32)
        # Normalise each channel to [0, 1]; guard against flat channels
        for c in range(3):
            lo, hi = rgb[..., c].min(), rgb[..., c].max()
            if hi > lo:
                rgb[..., c] = (rgb[..., c] - lo) / (hi - lo)
            else:
                rgb[..., c] = 0.0
        return rgb

    n_panels = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    rgb_t1 = _to_rgb(composite_t1)
    rgb_t2 = _to_rgb(composite_t2)
    pred_np = _to_numpy(prediction).astype(np.float32)
    if pred_np.ndim > 2:
        pred_np = pred_np.squeeze()

    axes[0].imshow(rgb_t1)
    axes[0].set_title("T1 RGB (before)")
    axes[0].axis("off")

    axes[1].imshow(rgb_t2)
    axes[1].set_title("T2 RGB (after)")
    axes[1].axis("off")

    im = axes[2].imshow(pred_np, cmap="hot", vmin=0.0, vmax=1.0)
    axes[2].set_title("Prediction heatmap")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if ground_truth is not None:
        gt_np = _to_numpy(ground_truth).astype(np.float32)
        if gt_np.ndim > 2:
            gt_np = gt_np.squeeze()
        # Overlay GT mask in semi-transparent red on the T2 RGB image
        axes[3].imshow(rgb_t2)
        mask_rgba = np.zeros((*gt_np.shape, 4), dtype=np.float32)
        mask_rgba[..., 0] = 1.0   # red channel
        mask_rgba[..., 3] = gt_np * 0.5  # alpha from GT mask
        axes[3].imshow(mask_rgba)
        axes[3].set_title("GT overlay")
        axes[3].axis("off")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def generate_metrics_table(
    metrics: dict[str, ValidationMetrics],
) -> str:
    """Return a Markdown table comparing validation metrics across study areas.

    The table contains one row per study area and columns for every metric
    stored in :class:`ValidationMetrics`.  Floating-point values are formatted
    to four decimal places; integer counts are formatted without decimals.

    Parameters
    ----------
    metrics:
        Mapping from study-area name to its :class:`ValidationMetrics` object.
        An empty mapping produces a table with only the header row.

    Returns
    -------
    str
        A valid Markdown table string including a header separator row.
    """
    columns = [
        ("Area", lambda m, name: name),
        ("Threshold", lambda m, _: f"{m.threshold:.4f}"),
        ("Pixel Precision", lambda m, _: f"{m.pixel_precision:.4f}"),
        ("Pixel Recall", lambda m, _: f"{m.pixel_recall:.4f}"),
        ("Pixel F1", lambda m, _: f"{m.pixel_f1:.4f}"),
        ("Poly Precision", lambda m, _: f"{m.polygon_precision:.4f}"),
        ("Poly Recall", lambda m, _: f"{m.polygon_recall:.4f}"),
        ("Poly F1", lambda m, _: f"{m.polygon_f1:.4f}"),
        ("N True Changes", lambda m, _: str(m.n_true_changes)),
        ("N Predicted", lambda m, _: str(m.n_predicted_changes)),
        ("Mean IoU", lambda m, _: f"{m.mean_iou:.4f}"),
    ]

    headers = [col[0] for col in columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"

    data_rows: list[str] = []
    for area_name, m in metrics.items():
        cells = [fmt(m, area_name) for _, fmt in columns]
        data_rows.append("| " + " | ".join(cells) + " |")

    lines = [header_row, separator_row] + data_rows
    return "\n".join(lines)


def generate_latency_figure(
    latency_results: dict[str, dict],
    output_path: Path | None = None,
) -> matplotlib.figure.Figure:
    """Create a timeline figure showing detection delay vs HRL update cycle.

    Each study area is plotted as a horizontal timeline with three annotated
    points:

    * **Change date** — when the surface change is known to have occurred.
    * **Detection date** — the first month the model exceeds the detection
      threshold (shown only when ``detection_date`` is not ``None``).
    * **HRL update** — a reference marker placed 18 months after the change
      date, representing the typical Copernicus HRL update cycle (1–3 years).

    The latency gap between the change date and the detection date is shown as
    a horizontal span.  Areas where detection never occurred show only the
    change marker and the HRL reference line.

    Parameters
    ----------
    latency_results:
        Mapping from study-area name to a dict containing:

        * ``"change_date"`` (``str``): ``"YYYY-MM"`` month of the known change.
        * ``"detection_date"`` (``str | None``): ``"YYYY-MM"`` month of first
          detection, or ``None`` if never detected.
        * ``"latency_months"`` (``int | None``): pre-computed latency in months.

    output_path:
        When provided, the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _ym_to_float(ym: str) -> float:
        """Convert 'YYYY-MM' to a float year suitable for axis placement."""
        year, month = ym.split("-")
        return int(year) + (int(month) - 1) / 12.0

    n_areas = len(latency_results)
    fig_height = max(3, n_areas * 1.5 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(n_areas))
    area_names = list(latency_results.keys())

    hrl_cycle_months = 18  # reference update frequency in months

    for y_pos, area_name in zip(y_positions, area_names):
        result = latency_results[area_name]
        change_date: str = result["change_date"]
        detection_date: str | None = result.get("detection_date")
        latency_months: int | None = result.get("latency_months")

        change_x = _ym_to_float(change_date)
        hrl_x = change_x + hrl_cycle_months / 12.0

        # Change date marker
        ax.plot(change_x, y_pos, marker="o", color="steelblue", markersize=10, zorder=3)
        ax.annotate(
            f"Change\n{change_date}",
            xy=(change_x, y_pos),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="steelblue",
        )

        # HRL reference marker
        ax.axvline(hrl_x, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(
            f"HRL update\n(+{hrl_cycle_months} mo)",
            xy=(hrl_x, y_pos),
            xytext=(4, -15),
            textcoords="offset points",
            ha="left",
            fontsize=7,
            color="gray",
        )

        if detection_date is not None:
            detection_x = _ym_to_float(detection_date)
            # Latency span
            ax.axhspan(
                y_pos - 0.25,
                y_pos + 0.25,
                xmin=0,  # will be clipped; use fill_betweenx instead
                alpha=0.0,  # reset — drawn below
            )
            ax.fill_betweenx(
                [y_pos - 0.25, y_pos + 0.25],
                change_x,
                detection_x,
                alpha=0.25,
                color="orange",
            )
            ax.plot(
                detection_x,
                y_pos,
                marker="^",
                color="darkorange",
                markersize=10,
                zorder=3,
            )
            latency_label = (
                f"Detected\n{detection_date}\n({latency_months} mo)"
                if latency_months is not None
                else f"Detected\n{detection_date}"
            )
            ax.annotate(
                latency_label,
                xy=(detection_x, y_pos),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="darkorange",
            )
        else:
            ax.annotate(
                "Not detected",
                xy=(change_x + 0.1, y_pos),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                fontsize=8,
                color="red",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(area_names)
    ax.set_xlabel("Time (year)")
    ax.set_title("Detection Latency vs HRL Update Cycle")

    # Legend
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Change date"),
        mpatches.Patch(color="darkorange", label="Detection date"),
        mpatches.Patch(color="orange", alpha=0.4, label="Latency gap"),
        mpatches.Patch(color="gray", alpha=0.5, label="HRL update (+18 mo)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig
