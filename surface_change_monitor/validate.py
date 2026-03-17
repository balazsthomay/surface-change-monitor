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
"""

from __future__ import annotations

from dataclasses import dataclass

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
