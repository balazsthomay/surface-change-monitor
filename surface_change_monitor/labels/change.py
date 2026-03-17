"""Change label generation for the impervious surface detection pipeline.

Provides two building blocks for the training dataset:

1. ``generate_change_labels`` — computes a binary mask where imperviousness
   *increased* by at least ``threshold`` percentage points between two epochs.

2. ``extract_patches`` — cuts aligned (t1, t2, label) patches from composites
   and the binary label map using a sliding-window with configurable stride.
   Patches where more than half the pixels are NaN are discarded.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 – registers .rio accessor on xr.DataArray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_change_labels(
    density_t1: xr.DataArray,
    density_t2: xr.DataArray,
    threshold: float = 10.0,
) -> xr.DataArray:
    """Generate a binary change mask from two imperviousness density maps.

    A pixel is labelled as **changed** (1) only when imperviousness *increased*
    by at least ``threshold`` percentage points, i.e.::

        label = uint8(density_t2 - density_t1 >= threshold)

    Decreases in imperviousness are ignored because the downstream use-case
    is detecting new construction / paving that raises flood risk.

    Parameters
    ----------
    density_t1:
        Imperviousness density at time 1 (percentage, 0–100), float32
        DataArray aligned to the same grid as ``density_t2``.
    density_t2:
        Imperviousness density at time 2 (percentage, 0–100).
    threshold:
        Minimum increase (in percentage points) required to label a pixel as
        changed.  Default is 10.0 (i.e. ≥10 pp increase).

    Returns
    -------
    xr.DataArray
        uint8 DataArray with the same spatial coordinates as the inputs.
        Values are 0 (no change or decrease) or 1 (increase ≥ threshold).
    """
    delta: xr.DataArray = density_t2 - density_t1
    labels: xr.DataArray = (delta >= threshold).astype(np.uint8)
    return labels


def extract_patches(
    composite_t1: xr.DataArray,
    composite_t2: xr.DataArray,
    labels: xr.DataArray,
    patch_size: int = 256,
    stride: int = 128,
) -> list[dict]:
    """Extract aligned (t1, t2, label) patches for training.

    A sliding window scans all three inputs simultaneously.  Patches where
    **more than 50 %** of spatial pixels in ``composite_t1`` are NaN are
    discarded to avoid training on mostly-missing data.

    The composites are expected to have shape ``(bands, H, W)`` and the labels
    to have shape ``(H, W)``.  All three must share the same spatial extent and
    pixel grid.

    Parameters
    ----------
    composite_t1:
        Multi-band Sentinel-2 composite at time 1, shape ``(C, H, W)``,
        float32.
    composite_t2:
        Multi-band Sentinel-2 composite at time 2, shape ``(C, H, W)``.
    labels:
        Binary change label map, shape ``(H, W)``, uint8.
    patch_size:
        Spatial size of each square patch (pixels).  Default 256.
    stride:
        Step size between patch origins (pixels).  Default 128.

    Returns
    -------
    list[dict]
        Each element is a dict with keys:

        - ``"t1"``: np.ndarray of shape ``(C, patch_size, patch_size)``
        - ``"t2"``: np.ndarray of shape ``(C, patch_size, patch_size)``
        - ``"label"``: np.ndarray of shape ``(patch_size, patch_size)``
    """
    # Extract numpy arrays from xarray inputs.
    # Composites have shape (bands, H, W); labels have shape (H, W).
    t1_np: np.ndarray = composite_t1.values  # (C, H, W)
    t2_np: np.ndarray = composite_t2.values  # (C, H, W)
    lbl_np: np.ndarray = labels.values       # (H, W)

    if t1_np.ndim == 2:
        # Single-band composite passed without a band dimension
        t1_np = t1_np[np.newaxis, ...]
        t2_np = t2_np[np.newaxis, ...]

    _, height, width = t1_np.shape
    patches: list[dict] = []

    nan_threshold = 0.5  # filter patches with >50 % NaN

    for row_start in range(0, height - patch_size + 1, stride):
        for col_start in range(0, width - patch_size + 1, stride):
            row_end = row_start + patch_size
            col_end = col_start + patch_size

            t1_patch = t1_np[:, row_start:row_end, col_start:col_end]  # (C, P, P)

            # Filter out patches that are mostly NaN.
            nan_count = np.sum(np.isnan(t1_patch))
            total_spatial = patch_size * patch_size
            if nan_count / total_spatial > nan_threshold:
                continue

            t2_patch = t2_np[:, row_start:row_end, col_start:col_end]
            lbl_patch = lbl_np[row_start:row_end, col_start:col_end]

            patches.append(
                {
                    "t1": t1_patch.copy(),
                    "t2": t2_patch.copy(),
                    "label": lbl_patch.copy(),
                }
            )

    return patches
