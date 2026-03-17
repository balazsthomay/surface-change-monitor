"""Inference/prediction for bi-temporal change detection.

Implements sliding-window tiled inference with Gaussian overlap blending to
avoid hard edge artefacts at tile boundaries.  For a 10 km × 10 km AOI at
10 m resolution (1000 × 1000 pixels) this produces ~25 overlapping tiles
using the default tile_size=256, overlap=64 (stride=192), which is
comfortably runnable on CPU or Apple MPS without a GPU.

Typical usage
-------------
>>> from surface_change_monitor.model.predict import predict_change, save_prediction
>>> prob_map = predict_change(Path("models/best.ckpt"), composite_t1, composite_t2)
>>> save_prediction(prob_map, Path("output/change_prob.tif"))
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 – registers .rio accessor
import torch
import xarray as xr

from surface_change_monitor.composite import MonthlyComposite
from surface_change_monitor.model.dataset import _BAND_MEAN, _BAND_STD
from surface_change_monitor.model.train import BinaryChangeDetectionTask


def _make_gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:
    """Create a 2D Gaussian weight matrix of shape (size, size).

    The Gaussian is centred at the middle of the kernel and normalised so
    that the peak equals 1 (un-normalised form).  When blending, tiles
    weighted by this kernel will contribute most at their centres and least at
    their edges, eliminating visible seams.

    Parameters
    ----------
    size:
        Side length of the square kernel in pixels.
    sigma:
        Standard deviation in pixels.  Defaults to ``size / 4``.

    Returns
    -------
    np.ndarray
        float64 array of shape ``(size, size)`` with values in ``(0, 1]``.
    """
    if sigma is None:
        sigma = size / 4.0

    centre = (size - 1) / 2.0
    coords = np.arange(size) - centre
    y_grid, x_grid = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel.astype(np.float64)


def _normalize_composite(data: np.ndarray) -> np.ndarray:
    """Apply the same per-channel z-score normalization used during training.

    Parameters
    ----------
    data:
        float32 array of shape ``(C, H, W)``.

    Returns
    -------
    float32 array of shape ``(C, H, W)``.
    """
    data = data.copy()
    np.nan_to_num(data, nan=0.0, copy=False)
    C = data.shape[0]
    mean = _BAND_MEAN[:C].reshape(C, 1, 1)
    std = _BAND_STD[:C].reshape(C, 1, 1)
    return ((data - mean) / (std + 1e-8)).astype(np.float32)


def predict_change(
    model_path: Path,
    composite_t1: MonthlyComposite,
    composite_t2: MonthlyComposite,
    tile_size: int = 256,
    overlap: int = 64,
) -> xr.DataArray:
    """Produce a probability map of imperviousness change between two composites.

    Uses sliding window tiling with Gaussian blending to avoid edge artefacts.
    The model is loaded from a Lightning checkpoint and run in eval mode.

    Parameters
    ----------
    model_path:
        Path to a ``.ckpt`` file saved by :func:`surface_change_monitor.model.train.train`.
    composite_t1:
        Earlier monthly composite (9 bands, shape (C, H, W)).
    composite_t2:
        Later monthly composite (same spatial extent and resolution as *composite_t1*).
    tile_size:
        Size of each square tile in pixels.  Smaller values use less memory;
        larger values provide more spatial context per inference call.
    overlap:
        Number of pixels of overlap between adjacent tiles.  Must be less than
        ``tile_size``.  Larger overlap gives smoother blending but more
        redundant computation.

    Returns
    -------
    xr.DataArray
        Float32 probability map of shape ``(H, W)`` with values in ``[0, 1]``.
        Carries the same CRS and spatial coordinates as the input composites.
    """
    if overlap >= tile_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than tile_size ({tile_size})"
        )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = BinaryChangeDetectionTask.load_from_checkpoint(str(model_path))
    model.eval()

    # ------------------------------------------------------------------
    # Prepare input arrays: (C, H, W) float32, normalised
    # ------------------------------------------------------------------
    t1_np = composite_t1.data.values.astype(np.float32)  # (C, H, W)
    t2_np = composite_t2.data.values.astype(np.float32)  # (C, H, W)

    t1_norm = _normalize_composite(t1_np)
    t2_norm = _normalize_composite(t2_np)

    C, H, W = t1_norm.shape

    # ------------------------------------------------------------------
    # Pad to ensure every tile is full-size (reflect padding avoids border
    # artefacts and is safe even for very small images)
    # ------------------------------------------------------------------
    stride = tile_size - overlap

    pad_h = _required_pad(H, tile_size, stride)
    pad_w = _required_pad(W, tile_size, stride)

    # numpy pad: axis order is (C, H, W) -> pad last two dims
    t1_pad = np.pad(t1_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    t2_pad = np.pad(t2_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, Hp, Wp = t1_pad.shape

    # ------------------------------------------------------------------
    # Accumulation buffers (float64 to avoid precision loss during blending)
    # ------------------------------------------------------------------
    prob_acc = np.zeros((Hp, Wp), dtype=np.float64)
    weight_acc = np.zeros((Hp, Wp), dtype=np.float64)

    gauss_kernel = _make_gaussian_kernel(tile_size)

    # ------------------------------------------------------------------
    # Sliding window inference
    # ------------------------------------------------------------------
    row_starts = _tile_starts(Hp, tile_size, stride)
    col_starts = _tile_starts(Wp, tile_size, stride)

    device = _select_device()

    with torch.no_grad():
        for r in row_starts:
            for c in col_starts:
                # Extract tile: (C, tile_size, tile_size)
                t1_tile = t1_pad[:, r : r + tile_size, c : c + tile_size]
                t2_tile = t2_pad[:, r : r + tile_size, c : c + tile_size]

                # Build batch: (1, 2, C, tile_size, tile_size)
                image = np.stack([t1_tile, t2_tile], axis=0)  # (2, C, H, W)
                image_t = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 2, C, H, W)

                # Forward pass -> logits (1, 1, H, W) -> prob (H, W)
                logits = model(image_t)  # (1, 1, tile_size, tile_size)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (tile_size, tile_size)

                # Accumulate with Gaussian weights
                prob_acc[r : r + tile_size, c : c + tile_size] += gauss_kernel * prob
                weight_acc[r : r + tile_size, c : c + tile_size] += gauss_kernel

    # ------------------------------------------------------------------
    # Normalise by accumulated weights and crop back to original size
    # ------------------------------------------------------------------
    # Avoid division by zero (shouldn't happen, but be defensive)
    weight_acc = np.where(weight_acc > 0, weight_acc, 1.0)
    prob_map = (prob_acc / weight_acc)[:H, :W].astype(np.float32)

    # ------------------------------------------------------------------
    # Wrap in DataArray, preserving georeference from input composite
    # ------------------------------------------------------------------
    ref = composite_t1.data  # (C, H, W) DataArray
    result = xr.DataArray(
        prob_map,
        dims=["y", "x"],
        coords={"y": ref.coords["y"], "x": ref.coords["x"]},
    )

    if ref.rio.crs is not None:
        result = result.rio.write_crs(ref.rio.crs)
    result = result.rio.write_transform()

    return result


def save_prediction(probability_map: xr.DataArray, output_path: Path) -> None:
    """Write a probability map DataArray to a single-band float32 GeoTIFF.

    Parameters
    ----------
    probability_map:
        Float32 DataArray of shape ``(H, W)`` returned by :func:`predict_change`.
    output_path:
        Destination path (parent directory must exist).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # rioxarray expects (band, y, x) for to_raster; expand dims if needed
    da = probability_map.expand_dims("band")
    da.rio.to_raster(str(output_path), dtype="float32")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _required_pad(size: int, tile_size: int, stride: int) -> int:
    """Return how many pixels to add on the right/bottom to cover *size* fully.

    Ensures that sliding window tiles with *stride* cover the full extent
    including any remainder.
    """
    if size <= tile_size:
        return max(0, tile_size - size)
    # Number of strides required to cover size pixels
    n_tiles = math.ceil((size - tile_size) / stride) + 1
    padded = tile_size + (n_tiles - 1) * stride
    return max(0, padded - size)


def _tile_starts(total: int, tile_size: int, stride: int) -> list[int]:
    """Return list of top-left pixel offsets for tiles covering [0, total).

    The last tile is anchored so it does not extend beyond *total* (i.e. the
    padded image size is guaranteed to be a multiple of *stride* + *tile_size*).
    """
    starts: list[int] = []
    pos = 0
    while pos + tile_size <= total:
        starts.append(pos)
        pos += stride
    return starts


def _select_device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
