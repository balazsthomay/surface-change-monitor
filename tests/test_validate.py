"""Tests for validation metrics module.

Tests cover:
  - test_pixel_metrics: Known pred + GT -> correct precision/recall/F1
  - test_polygon_iou: Predicted vs GT polygons -> correct IoU
  - test_polygon_level_metrics: TP = pred polygon with IoU > 0.3 against any GT polygon
  - test_detection_latency: Known change date + detection date -> latency
  - test_metrics_at_thresholds: Metrics at multiple probability thresholds
  - test_generate_visual_comparison: Figure shape, type, and output path
  - test_generate_metrics_table: Markdown structure and content
  - test_generate_latency_figure: Figure type and timeline annotations
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

# Use a non-interactive backend for all tests so figures can be created without
# a display server.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pred_array(values: np.ndarray) -> np.ndarray:
    """Wrap probability values in a float32 numpy array."""
    return values.astype(np.float32)


def _make_gt_array(values: np.ndarray) -> np.ndarray:
    """Wrap ground truth binary values in an int32 numpy array."""
    return values.astype(np.int32)


def _make_pred_xarray(values: np.ndarray) -> xr.DataArray:
    """Wrap probability values in an xr.DataArray."""
    h, w = values.shape
    return xr.DataArray(
        values.astype(np.float32),
        dims=["y", "x"],
        coords={
            "y": np.arange(h, dtype=np.float32),
            "x": np.arange(w, dtype=np.float32),
        },
    )


def _make_gt_xarray(values: np.ndarray) -> xr.DataArray:
    """Wrap binary GT values in an xr.DataArray."""
    h, w = values.shape
    return xr.DataArray(
        values.astype(np.int32),
        dims=["y", "x"],
        coords={
            "y": np.arange(h, dtype=np.float32),
            "x": np.arange(w, dtype=np.float32),
        },
    )


def _make_geodataframe(geometries: list) -> "geopandas.GeoDataFrame":
    """Create a GeoDataFrame from a list of shapely geometries."""
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# 13.1a  test_pixel_metrics
# ---------------------------------------------------------------------------


class TestPixelMetrics:
    def test_perfect_prediction_numpy(self):
        """Perfect prediction: precision=1, recall=1, F1=1."""
        from surface_change_monitor.validate import compute_pixel_metrics

        gt = _make_gt_array(np.array([[1, 0, 1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.1, 0.8, 0.2]], dtype=np.float32))

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_precision == pytest.approx(1.0)
        assert metrics.pixel_recall == pytest.approx(1.0)
        assert metrics.pixel_f1 == pytest.approx(1.0)

    def test_known_values_numpy(self):
        """Known pred + GT -> correct precision/recall/F1.

        GT  = [1, 1, 0, 0]
        pred > 0.5 = [1, 0, 1, 0]

        TP = 1 (pos[0])
        FP = 1 (pos[2])
        FN = 1 (pos[1])
        TN = 1 (pos[3])

        precision = TP / (TP + FP) = 1/2
        recall    = TP / (TP + FN) = 1/2
        F1        = 2 * prec * rec / (prec + rec) = 0.5
        """
        from surface_change_monitor.validate import compute_pixel_metrics

        gt = _make_gt_array(np.array([[1, 1, 0, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.3, 0.7, 0.1]], dtype=np.float32))

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_precision == pytest.approx(0.5)
        assert metrics.pixel_recall == pytest.approx(0.5)
        assert metrics.pixel_f1 == pytest.approx(0.5)

    def test_all_false_negatives(self):
        """All GT positives are missed: recall=0, F1=0."""
        from surface_change_monitor.validate import compute_pixel_metrics

        gt = _make_gt_array(np.array([[1, 1, 1]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.1, 0.1, 0.1]], dtype=np.float32))

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_recall == pytest.approx(0.0)
        assert metrics.pixel_f1 == pytest.approx(0.0)

    def test_all_false_positives(self):
        """All predictions are wrong: precision=0, F1=0."""
        from surface_change_monitor.validate import compute_pixel_metrics

        gt = _make_gt_array(np.array([[0, 0, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.9, 0.9]], dtype=np.float32))

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_precision == pytest.approx(0.0)
        assert metrics.pixel_f1 == pytest.approx(0.0)

    def test_accepts_xarray_inputs(self):
        """compute_pixel_metrics accepts xr.DataArray inputs."""
        from surface_change_monitor.validate import compute_pixel_metrics

        gt_arr = np.array([[1, 0], [0, 1]], dtype=np.int32)
        pred_arr = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

        gt = _make_gt_xarray(gt_arr)
        pred = _make_pred_xarray(pred_arr)

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_precision == pytest.approx(1.0)
        assert metrics.pixel_recall == pytest.approx(1.0)
        assert metrics.pixel_f1 == pytest.approx(1.0)

    def test_custom_threshold(self):
        """Custom threshold changes which predictions are positive."""
        from surface_change_monitor.validate import compute_pixel_metrics

        # GT = [1, 0], pred = [0.4, 0.3]
        # At threshold=0.5: pred_bin = [0, 0] -> TP=0, FN=1 -> recall=0
        # At threshold=0.3: pred_bin = [1, 1] -> TP=1, FP=1, FN=0 -> recall=1, prec=0.5
        gt = _make_gt_array(np.array([[1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.4, 0.3]], dtype=np.float32))

        metrics_05 = compute_pixel_metrics(pred, gt, threshold=0.5)
        metrics_03 = compute_pixel_metrics(pred, gt, threshold=0.3)

        assert metrics_05.pixel_recall == pytest.approx(0.0)
        assert metrics_03.pixel_recall == pytest.approx(1.0)

    def test_threshold_stored_in_metrics(self):
        """The threshold value is stored in ValidationMetrics.threshold."""
        from surface_change_monitor.validate import compute_pixel_metrics

        gt = _make_gt_array(np.array([[1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.1]], dtype=np.float32))

        metrics = compute_pixel_metrics(pred, gt, threshold=0.7)

        assert metrics.threshold == pytest.approx(0.7)

    def test_2d_array(self):
        """Works correctly on 2D arrays (H x W)."""
        from surface_change_monitor.validate import compute_pixel_metrics

        # 3x3 with known TP, FP, FN, TN
        # GT  = [[1,1,0],[0,1,0],[0,0,1]]
        # pred thresholded at 0.5 = [[1,0,1],[0,1,0],[0,0,1]]
        # TP=3 (00,11,22), FP=1 (02), FN=1 (01)
        # precision = 3/4 = 0.75, recall = 3/4 = 0.75
        gt = _make_gt_array(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32))
        pred = _make_pred_array(
            np.array([[0.9, 0.4, 0.7], [0.1, 0.8, 0.2], [0.1, 0.1, 0.9]], dtype=np.float32)
        )

        metrics = compute_pixel_metrics(pred, gt, threshold=0.5)

        assert metrics.pixel_precision == pytest.approx(0.75)
        assert metrics.pixel_recall == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 13.1b  test_polygon_iou
# ---------------------------------------------------------------------------


class TestPolygonIou:
    def test_identical_polygons(self):
        """Identical polygons -> IoU = 1.0."""
        from surface_change_monitor.validate import _compute_polygon_iou

        poly_a = box(0, 0, 1, 1)
        poly_b = box(0, 0, 1, 1)

        iou = _compute_polygon_iou(poly_a, poly_b)

        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping polygons -> IoU = 0.0."""
        from surface_change_monitor.validate import _compute_polygon_iou

        poly_a = box(0, 0, 1, 1)
        poly_b = box(2, 2, 3, 3)

        iou = _compute_polygon_iou(poly_a, poly_b)

        assert iou == pytest.approx(0.0)

    def test_half_overlap(self):
        """50% overlap: intersection=0.5, union=1.5 -> IoU = 1/3."""
        from surface_change_monitor.validate import _compute_polygon_iou

        poly_a = box(0, 0, 1, 1)  # area = 1
        poly_b = box(0.5, 0, 1.5, 1)  # area = 1, overlap = 0.5

        iou = _compute_polygon_iou(poly_a, poly_b)

        # intersection = 0.5, union = 1 + 1 - 0.5 = 1.5
        assert iou == pytest.approx(0.5 / 1.5, rel=1e-5)

    def test_known_iou(self):
        """Compute IoU for known geometry -> exact expected value."""
        from surface_change_monitor.validate import _compute_polygon_iou

        # poly_a: 2x2 = area 4, poly_b: 2x2 shifted 1 unit right
        # overlap: 1x2 = area 2
        # union: 4 + 4 - 2 = 6
        # IoU = 2/6 = 1/3
        poly_a = box(0, 0, 2, 2)
        poly_b = box(1, 0, 3, 2)

        iou = _compute_polygon_iou(poly_a, poly_b)

        assert iou == pytest.approx(1.0 / 3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 13.1c  test_polygon_level_metrics
# ---------------------------------------------------------------------------


class TestPolygonLevelMetrics:
    def test_all_true_positives(self):
        """All predicted polygons overlap GT above threshold -> precision=recall=F1=1."""
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1), box(2, 0, 3, 1)]
        pred_polys = [box(0, 0, 1, 1), box(2, 0, 3, 1)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.polygon_precision == pytest.approx(1.0)
        assert metrics.polygon_recall == pytest.approx(1.0)
        assert metrics.polygon_f1 == pytest.approx(1.0)

    def test_no_matches(self):
        """No predicted polygon overlaps any GT polygon -> precision=recall=F1=0."""
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1)]
        pred_polys = [box(10, 10, 11, 11)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.polygon_precision == pytest.approx(0.0)
        assert metrics.polygon_recall == pytest.approx(0.0)
        assert metrics.polygon_f1 == pytest.approx(0.0)

    def test_iou_threshold_boundary(self):
        """IoU exactly at threshold is a TP; just below is FP/FN.

        poly_a = box(0, 0, 1, 1), area=1
        poly_b = box(0, 0, 1, 0.5), area=0.5
        intersection=0.5, union=1.0
        IoU = 0.5

        At threshold=0.3 -> TP; at threshold=0.6 -> FP
        """
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1)]
        pred_polys = [box(0, 0, 1, 0.5)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics_tp = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)
        metrics_fp = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.6)

        assert metrics_tp.polygon_precision == pytest.approx(1.0)
        assert metrics_fp.polygon_precision == pytest.approx(0.0)

    def test_partial_matches(self):
        """One TP, one FP, one FN -> precision=0.5, recall=0.5.

        GT: [poly_A, poly_B]
        Pred: [poly_A_match, poly_C_no_match]

        TP=1 (poly_A_match matches poly_A with high IoU)
        FP=1 (poly_C_no_match has no GT match)
        FN=1 (poly_B not matched by any pred)

        precision = 1/2 = 0.5
        recall    = 1/2 = 0.5
        """
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1), box(5, 5, 6, 6)]
        pred_polys = [box(0, 0, 1, 1), box(10, 10, 11, 11)]  # first matches GT[0], second no match

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.polygon_precision == pytest.approx(0.5)
        assert metrics.polygon_recall == pytest.approx(0.5)

    def test_counts_stored(self):
        """n_true_changes and n_predicted_changes are stored correctly."""
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
        pred_polys = [box(0, 0, 1, 1)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.n_true_changes == 3
        assert metrics.n_predicted_changes == 1

    def test_mean_iou_computed(self):
        """mean_iou is the average IoU of matched (TP) predictions."""
        from surface_change_monitor.validate import compute_polygon_metrics

        # One TP with perfect IoU=1.0
        gt_polys = [box(0, 0, 1, 1)]
        pred_polys = [box(0, 0, 1, 1)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.mean_iou == pytest.approx(1.0)

    def test_empty_predictions(self):
        """Empty predictions: precision undefined (0), recall=0."""
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys = [box(0, 0, 1, 1)]
        pred_polys: list = []

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.polygon_recall == pytest.approx(0.0)
        assert metrics.n_predicted_changes == 0

    def test_empty_ground_truth(self):
        """Empty GT: recall undefined (0), precision=0 (all FP)."""
        from surface_change_monitor.validate import compute_polygon_metrics

        gt_polys: list = []
        pred_polys = [box(0, 0, 1, 1)]

        gt_gdf = _make_geodataframe(gt_polys)
        pred_gdf = _make_geodataframe(pred_polys)

        metrics = compute_polygon_metrics(pred_gdf, gt_gdf, iou_threshold=0.3)

        assert metrics.polygon_precision == pytest.approx(0.0)
        assert metrics.n_true_changes == 0


# ---------------------------------------------------------------------------
# 13.1d  test_detection_latency
# ---------------------------------------------------------------------------


class TestDetectionLatency:
    def test_first_month_detection(self):
        """Detection at first month -> latency = 0."""
        from surface_change_monitor.validate import detection_latency_analysis

        known_change_date = "2024-01"
        predictions = [
            ("2024-01", _make_pred_array(np.array([[0.9]], dtype=np.float32))),
            ("2024-02", _make_pred_array(np.array([[0.9]], dtype=np.float32))),
        ]

        result = detection_latency_analysis(predictions, known_change_date, threshold=0.5)

        assert result["latency_months"] == 0
        assert result["detection_month"] == "2024-01"

    def test_delayed_detection(self):
        """Change missed for two months then detected -> latency = 2."""
        from surface_change_monitor.validate import detection_latency_analysis

        known_change_date = "2024-01"
        predictions = [
            ("2024-01", _make_pred_array(np.array([[0.1]], dtype=np.float32))),
            ("2024-02", _make_pred_array(np.array([[0.3]], dtype=np.float32))),
            ("2024-03", _make_pred_array(np.array([[0.8]], dtype=np.float32))),
            ("2024-04", _make_pred_array(np.array([[0.9]], dtype=np.float32))),
        ]

        result = detection_latency_analysis(predictions, known_change_date, threshold=0.5)

        assert result["latency_months"] == 2
        assert result["detection_month"] == "2024-03"

    def test_never_detected(self):
        """Change never exceeds threshold -> latency = None, detection_month = None."""
        from surface_change_monitor.validate import detection_latency_analysis

        known_change_date = "2024-01"
        predictions = [
            ("2024-01", _make_pred_array(np.array([[0.1]], dtype=np.float32))),
            ("2024-02", _make_pred_array(np.array([[0.2]], dtype=np.float32))),
        ]

        result = detection_latency_analysis(predictions, known_change_date, threshold=0.5)

        assert result["latency_months"] is None
        assert result["detection_month"] is None

    def test_latency_months_correct_for_gap(self):
        """Months before known_change_date are skipped."""
        from surface_change_monitor.validate import detection_latency_analysis

        # Known change is 2024-03; predictions from 2024-01 through 2024-05
        # Detection at 2024-04 -> latency = 1 month after change date
        known_change_date = "2024-03"
        predictions = [
            ("2024-01", _make_pred_array(np.array([[0.9]], dtype=np.float32))),  # before change
            ("2024-02", _make_pred_array(np.array([[0.9]], dtype=np.float32))),  # before change
            ("2024-03", _make_pred_array(np.array([[0.2]], dtype=np.float32))),  # missed
            ("2024-04", _make_pred_array(np.array([[0.8]], dtype=np.float32))),  # detected
        ]

        result = detection_latency_analysis(predictions, known_change_date, threshold=0.5)

        assert result["latency_months"] == 1
        assert result["detection_month"] == "2024-04"

    def test_2d_prediction_any_pixel(self):
        """Detection succeeds if any pixel in the prediction map exceeds threshold."""
        from surface_change_monitor.validate import detection_latency_analysis

        known_change_date = "2024-01"
        # A 3x3 prediction with only one pixel above threshold
        pred_map = np.array([[0.1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32)
        predictions = [
            ("2024-01", _make_pred_array(pred_map)),
        ]

        result = detection_latency_analysis(predictions, known_change_date, threshold=0.5)

        assert result["latency_months"] == 0


# ---------------------------------------------------------------------------
# 13.1e  test_metrics_at_thresholds
# ---------------------------------------------------------------------------


class TestMetricsAtThresholds:
    def test_returns_list_of_metrics(self):
        """metrics_at_thresholds returns a list of ValidationMetrics, one per threshold."""
        from surface_change_monitor.validate import metrics_at_thresholds

        gt = _make_gt_array(np.array([[1, 0, 1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.1, 0.8, 0.2]], dtype=np.float32))
        thresholds = [0.3, 0.5, 0.7]

        results = metrics_at_thresholds(pred, gt, thresholds)

        assert len(results) == 3

    def test_threshold_increases_precision(self):
        """Higher threshold generally increases precision (fewer FP)."""
        from surface_change_monitor.validate import metrics_at_thresholds

        # GT = [1, 1, 0, 0, 0, 0]
        # pred scores increasingly aligned with GT at high thresholds
        gt = _make_gt_array(np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.8, 0.6, 0.4, 0.2, 0.1]], dtype=np.float32))
        thresholds = [0.3, 0.5, 0.75]

        results = metrics_at_thresholds(pred, gt, thresholds)

        # At threshold 0.3: pred_bin = [1,1,1,1,0,0] -> TP=2, FP=2 -> prec=0.5
        # At threshold 0.5: pred_bin = [1,1,1,0,0,0] -> TP=2, FP=1 -> prec=2/3
        # At threshold 0.75: pred_bin = [1,1,0,0,0,0] -> TP=2, FP=0 -> prec=1.0
        assert results[0].pixel_precision < results[1].pixel_precision
        assert results[1].pixel_precision < results[2].pixel_precision

    def test_each_result_has_correct_threshold(self):
        """Each ValidationMetrics result stores its own threshold value."""
        from surface_change_monitor.validate import metrics_at_thresholds

        gt = _make_gt_array(np.array([[1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.1]], dtype=np.float32))
        thresholds = [0.2, 0.5, 0.8]

        results = metrics_at_thresholds(pred, gt, thresholds)

        for expected_t, result in zip(thresholds, results):
            assert result.threshold == pytest.approx(expected_t)

    def test_metrics_at_thresholds_consistency(self):
        """metrics_at_thresholds results match individually-computed metrics."""
        from surface_change_monitor.validate import compute_pixel_metrics, metrics_at_thresholds

        gt = _make_gt_array(np.array([[1, 0, 1]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.8, 0.3, 0.6]], dtype=np.float32))
        thresholds = [0.4, 0.7]

        batch_results = metrics_at_thresholds(pred, gt, thresholds)

        for t, batch_result in zip(thresholds, batch_results):
            individual = compute_pixel_metrics(pred, gt, threshold=t)
            assert batch_result.pixel_precision == pytest.approx(individual.pixel_precision)
            assert batch_result.pixel_recall == pytest.approx(individual.pixel_recall)
            assert batch_result.pixel_f1 == pytest.approx(individual.pixel_f1)

    def test_empty_threshold_list(self):
        """Empty threshold list returns empty list."""
        from surface_change_monitor.validate import metrics_at_thresholds

        gt = _make_gt_array(np.array([[1, 0]], dtype=np.int32))
        pred = _make_pred_array(np.array([[0.9, 0.1]], dtype=np.float32))

        results = metrics_at_thresholds(pred, gt, thresholds=[])

        assert results == []


# ---------------------------------------------------------------------------
# 13.1f  test_validation_metrics_dataclass
# ---------------------------------------------------------------------------


class TestValidationMetricsDataclass:
    def test_dataclass_fields(self):
        """ValidationMetrics has all required fields."""
        from surface_change_monitor.validate import ValidationMetrics

        m = ValidationMetrics(
            pixel_precision=0.8,
            pixel_recall=0.7,
            pixel_f1=0.747,
            polygon_precision=0.9,
            polygon_recall=0.85,
            polygon_f1=0.874,
            n_true_changes=10,
            n_predicted_changes=9,
            mean_iou=0.65,
            threshold=0.5,
        )

        assert m.pixel_precision == pytest.approx(0.8)
        assert m.pixel_recall == pytest.approx(0.7)
        assert m.pixel_f1 == pytest.approx(0.747)
        assert m.polygon_precision == pytest.approx(0.9)
        assert m.polygon_recall == pytest.approx(0.85)
        assert m.polygon_f1 == pytest.approx(0.874)
        assert m.n_true_changes == 10
        assert m.n_predicted_changes == 9
        assert m.mean_iou == pytest.approx(0.65)
        assert m.threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Helpers shared by reporting tests
# ---------------------------------------------------------------------------


def _make_composite(bands: int = 3, h: int = 8, w: int = 8) -> xr.DataArray:
    """Return a synthetic (bands, H, W) composite DataArray."""
    data = np.random.default_rng(42).random((bands, h, w)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["bands", "y", "x"],
        coords={
            "bands": np.arange(bands),
            "y": np.linspace(0, 1, h),
            "x": np.linspace(0, 1, w),
        },
    )


def _make_pred_da(h: int = 8, w: int = 8) -> xr.DataArray:
    """Return a synthetic (H, W) probability DataArray."""
    data = np.random.default_rng(7).random((h, w)).astype(np.float32)
    return xr.DataArray(data, dims=["y", "x"])


def _make_gt_da(h: int = 8, w: int = 8) -> xr.DataArray:
    """Return a synthetic (H, W) binary DataArray."""
    rng = np.random.default_rng(13)
    data = rng.integers(0, 2, size=(h, w), dtype=np.int32)
    return xr.DataArray(data, dims=["y", "x"])


def _make_metrics(
    pixel_f1: float = 0.75,
    polygon_f1: float = 0.70,
    threshold: float = 0.5,
) -> "ValidationMetrics":
    from surface_change_monitor.validate import ValidationMetrics

    return ValidationMetrics(
        pixel_precision=0.8,
        pixel_recall=pixel_f1,
        pixel_f1=pixel_f1,
        polygon_precision=0.7,
        polygon_recall=polygon_f1,
        polygon_f1=polygon_f1,
        n_true_changes=10,
        n_predicted_changes=9,
        mean_iou=0.6,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 14.2a  test_generate_visual_comparison
# ---------------------------------------------------------------------------


class TestGenerateVisualComparison:
    def test_returns_figure(self):
        """Function returns a matplotlib Figure object."""
        import matplotlib.figure

        from surface_change_monitor.validate import generate_visual_comparison

        fig = generate_visual_comparison(
            composite_t1=_make_composite(),
            composite_t2=_make_composite(),
            prediction=_make_pred_da(),
        )
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_three_panels_without_ground_truth(self):
        """Without GT, the figure has 3 content panels (+ 1 colorbar = 4 axes total)."""
        from surface_change_monitor.validate import generate_visual_comparison

        fig = generate_visual_comparison(
            composite_t1=_make_composite(),
            composite_t2=_make_composite(),
            prediction=_make_pred_da(),
            ground_truth=None,
        )
        try:
            # 3 image panels + 1 colorbar axis
            content_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
            assert len(content_axes) == 3
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_four_panels_with_ground_truth(self):
        """With GT, the figure has 4 content panels (+ 1 colorbar = 5 axes total)."""
        from surface_change_monitor.validate import generate_visual_comparison

        fig = generate_visual_comparison(
            composite_t1=_make_composite(),
            composite_t2=_make_composite(),
            prediction=_make_pred_da(),
            ground_truth=_make_gt_da(),
        )
        try:
            # 4 image panels + 1 colorbar axis
            content_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
            assert len(content_axes) == 4
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_saves_to_output_path(self, tmp_path):
        """Figure is saved to disk when output_path is provided."""
        from surface_change_monitor.validate import generate_visual_comparison

        out = tmp_path / "comparison.png"
        fig = generate_visual_comparison(
            composite_t1=_make_composite(),
            composite_t2=_make_composite(),
            prediction=_make_pred_da(),
            output_path=out,
        )
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_single_band_composite(self):
        """Single-band composites are accepted and produce 3 content panels."""
        from surface_change_monitor.validate import generate_visual_comparison

        fig = generate_visual_comparison(
            composite_t1=_make_composite(bands=1),
            composite_t2=_make_composite(bands=1),
            prediction=_make_pred_da(),
        )
        try:
            content_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
            assert len(content_axes) == 3
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_two_band_composite(self):
        """Two-band composites are accepted without errors."""
        from surface_change_monitor.validate import generate_visual_comparison

        fig = generate_visual_comparison(
            composite_t1=_make_composite(bands=2),
            composite_t2=_make_composite(bands=2),
            prediction=_make_pred_da(),
        )
        try:
            assert isinstance(fig.axes[0].get_images()[0], matplotlib.image.AxesImage)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_creates_parent_directories(self, tmp_path):
        """output_path parent directories are created automatically."""
        from surface_change_monitor.validate import generate_visual_comparison

        out = tmp_path / "subdir" / "nested" / "fig.png"
        fig = generate_visual_comparison(
            composite_t1=_make_composite(),
            composite_t2=_make_composite(),
            prediction=_make_pred_da(),
            output_path=out,
        )
        try:
            assert out.exists()
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


# ---------------------------------------------------------------------------
# 14.2b  test_generate_metrics_table
# ---------------------------------------------------------------------------


class TestGenerateMetricsTable:
    def test_returns_string(self):
        """Function returns a str."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({"Bergen": _make_metrics()})
        assert isinstance(table, str)

    def test_header_present(self):
        """Table contains the expected column headers."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({"Bergen": _make_metrics()})
        assert "Pixel F1" in table
        assert "Poly F1" in table
        assert "Mean IoU" in table
        assert "Area" in table

    def test_separator_row_present(self):
        """Table has a Markdown separator row (--- pattern)."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({"Bergen": _make_metrics()})
        lines = table.splitlines()
        assert any("---" in line for line in lines)

    def test_area_names_in_table(self):
        """Each area name appears in the table."""
        from surface_change_monitor.validate import generate_metrics_table

        areas = {"Bergen": _make_metrics(pixel_f1=0.8), "Houston": _make_metrics(pixel_f1=0.65)}
        table = generate_metrics_table(areas)
        assert "Bergen" in table
        assert "Houston" in table

    def test_empty_metrics_dict(self):
        """Empty dict produces a table with header + separator but no data rows."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({})
        lines = table.splitlines()
        # header row + separator row = 2 lines
        assert len(lines) == 2

    def test_multiple_areas_produce_multiple_rows(self):
        """Three areas yield header + separator + 3 data rows = 5 lines total."""
        from surface_change_monitor.validate import generate_metrics_table

        areas = {
            "Bergen": _make_metrics(),
            "Houston": _make_metrics(),
            "Oslo": _make_metrics(),
        }
        table = generate_metrics_table(areas)
        lines = table.splitlines()
        assert len(lines) == 5

    def test_metric_values_formatted(self):
        """Numeric values appear formatted to 4 decimal places."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({"Bergen": _make_metrics(pixel_f1=0.75)})
        assert "0.7500" in table

    def test_pipe_delimited_format(self):
        """Every non-empty line starts and ends with '|'."""
        from surface_change_monitor.validate import generate_metrics_table

        table = generate_metrics_table({"Bergen": _make_metrics()})
        for line in table.splitlines():
            if line.strip():
                assert line.startswith("|")
                assert line.endswith("|")


# ---------------------------------------------------------------------------
# 14.2c  test_generate_latency_figure
# ---------------------------------------------------------------------------


class TestGenerateLatencyFigure:
    def test_returns_figure(self):
        """Function returns a matplotlib Figure."""
        import matplotlib.figure

        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-03",
                "detection_date": "2024-05",
                "latency_months": 2,
            }
        }
        fig = generate_latency_figure(results)
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_single_area_figure(self):
        """Single study area produces a figure with at least one axis."""
        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-01",
                "detection_date": "2024-03",
                "latency_months": 2,
            }
        }
        fig = generate_latency_figure(results)
        try:
            assert len(fig.axes) >= 1
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_multiple_areas(self):
        """Two study areas produce a valid figure without errors."""
        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-03",
                "detection_date": "2024-05",
                "latency_months": 2,
            },
            "Houston": {
                "change_date": "2024-06",
                "detection_date": "2024-08",
                "latency_months": 2,
            },
        }
        fig = generate_latency_figure(results)
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_area_with_no_detection(self):
        """Areas where detection never occurred are handled without errors."""
        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-03",
                "detection_date": None,
                "latency_months": None,
            }
        }
        fig = generate_latency_figure(results)
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_saves_to_output_path(self, tmp_path):
        """Figure is saved to disk when output_path is provided."""
        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-03",
                "detection_date": "2024-06",
                "latency_months": 3,
            }
        }
        out = tmp_path / "latency.png"
        fig = generate_latency_figure(results, output_path=out)
        try:
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_empty_results(self):
        """Empty results dict produces a valid figure."""
        from surface_change_monitor.validate import generate_latency_figure

        fig = generate_latency_figure({})
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_y_axis_labels_match_area_names(self):
        """Y-axis tick labels correspond to the provided area names."""
        from surface_change_monitor.validate import generate_latency_figure

        results = {
            "Bergen": {
                "change_date": "2024-03",
                "detection_date": "2024-05",
                "latency_months": 2,
            },
            "Houston": {
                "change_date": "2024-06",
                "detection_date": None,
                "latency_months": None,
            },
        }
        fig = generate_latency_figure(results)
        try:
            ax = fig.axes[0]
            tick_labels = [t.get_text() for t in ax.get_yticklabels()]
            assert "Bergen" in tick_labels
            assert "Houston" in tick_labels
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)
