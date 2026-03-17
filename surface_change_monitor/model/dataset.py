"""Bi-temporal change detection dataset.

Serves (image, mask) pairs for torchgeo's ChangeDetectionTask.

Each sample originates from a ``.npz`` patch file produced by
``extract_patches`` (see ``surface_change_monitor.labels.change``).  The
``.npz`` must contain the keys:

- ``t1``     – float32 array of shape ``(C, H, W)``
- ``t2``     – float32 array of shape ``(C, H, W)``
- ``label``  – uint8 array of shape ``(H, W)``
- ``city``   – scalar string (e.g. ``"bergen"``, ``"houston"``)
- ``source`` – scalar string (e.g. ``"sentinel2"``)

The dataset stacks t1 and t2 into a ``(2, C, H, W)`` tensor keyed ``"image"``
and returns the label as a long tensor keyed ``"mask"`` — the format expected
by torchgeo's ChangeDetectionTask.

Split strategy
--------------
Train  → Bergen, Oslo, Amsterdam, Warsaw  (large, well-labelled cities)
Val    → Dublin                            (held-out European city)
Test   → Houston                           (held-out non-European city)

This is a *spatial* split: each patch belongs to exactly one split based on
its ``city`` metadata, so information cannot leak across the tile boundary.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torchgeo.datasets import NonGeoDataset

# ---------------------------------------------------------------------------
# City → split mapping
# ---------------------------------------------------------------------------

_TRAIN_CITIES: frozenset[str] = frozenset({"bergen", "oslo", "amsterdam", "warsaw"})
_VAL_CITIES: frozenset[str] = frozenset({"dublin"})
_TEST_CITIES: frozenset[str] = frozenset({"houston"})

_SPLIT_CITIES: dict[str, frozenset[str]] = {
    "train": _TRAIN_CITIES,
    "val": _VAL_CITIES,
    "test": _TEST_CITIES,
}

# ---------------------------------------------------------------------------
# Per-channel normalization statistics
# (mean and std estimated from Sentinel-2 reflectance in [0, 1] range)
# Channels: B02, B03, B04, B08, B11, B12, NDVI, NDWI, NDBI
# ---------------------------------------------------------------------------

_BAND_MEAN: np.ndarray = np.array(
    [0.0825, 0.0980, 0.1080, 0.2390, 0.1620, 0.1050, 0.2500, -0.1200, -0.1800],
    dtype=np.float32,
)

_BAND_STD: np.ndarray = np.array(
    [0.0450, 0.0520, 0.0680, 0.0950, 0.0850, 0.0780, 0.2100, 0.1500, 0.1600],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BiTemporalChangeDataset(NonGeoDataset):
    """Serves (t1, t2, label) triplets from ``.npz`` patch files.

    Expects ``.npz`` patches with keys:
    ``t1: (C, 256, 256)``, ``t2: (C, 256, 256)``, ``label: (256, 256)``,
    ``source: str``, ``city: str``

    Parameters
    ----------
    patches_dir:
        Directory containing ``.npz`` patch files.
    split:
        One of ``"train"``, ``"val"``, ``"test"``, or ``None``.
        When ``None`` all patches in the directory are loaded without
        city-based filtering.
    """

    def __init__(
        self,
        patches_dir: str | Path,
        split: str | None = "train",
    ) -> None:
        self.patches_dir = Path(patches_dir)
        self.split = split
        self.patch_paths: list[Path] = self._collect_patches()

    # ------------------------------------------------------------------
    # NonGeoDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return a single sample.

        Returns
        -------
        dict with keys:
            ``"image"`` – float32 tensor of shape ``(2, C, H, W)``
            ``"mask"``  – long tensor of shape ``(H, W)``
        """
        path = self.patch_paths[idx]
        data = np.load(path, allow_pickle=True)

        t1: np.ndarray = data["t1"].astype(np.float32)  # (C, H, W)
        t2: np.ndarray = data["t2"].astype(np.float32)  # (C, H, W)
        label: np.ndarray = data["label"].astype(np.int64)  # (H, W)

        # Replace NaN with 0.0 before any arithmetic
        np.nan_to_num(t1, nan=0.0, copy=False)
        np.nan_to_num(t2, nan=0.0, copy=False)

        # Normalize each band independently using global statistics
        t1 = self._normalize(t1)
        t2 = self._normalize(t2)

        # Stack into (2, C, H, W)
        image = np.stack([t1, t2], axis=0)  # (2, C, H, W)

        image_t = torch.from_numpy(image)    # float32
        mask_t = torch.from_numpy(label)     # int64 / long

        if self.split == "train":
            image_t, mask_t = self._augment(image_t, mask_t)

        return {"image": image_t, "mask": mask_t}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_patches(self) -> list[Path]:
        """Return sorted list of .npz paths matching the requested split."""
        all_paths = sorted(self.patches_dir.glob("*.npz"))

        if self.split is None:
            return all_paths

        allowed_cities = _SPLIT_CITIES.get(self.split)
        if allowed_cities is None:
            raise ValueError(
                f"Unknown split {self.split!r}. Must be one of "
                f"{list(_SPLIT_CITIES.keys())} or None."
            )

        selected: list[Path] = []
        for p in all_paths:
            city = self._read_city(p)
            if city in allowed_cities:
                selected.append(p)
        return selected

    @staticmethod
    def _read_city(path: Path) -> str:
        """Read the ``city`` scalar from a .npz file without loading full arrays."""
        with np.load(path, allow_pickle=True) as data:
            city_val = data["city"]
        # numpy scalars can be 0-d arrays
        if isinstance(city_val, np.ndarray):
            return str(city_val.item())
        return str(city_val)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization using precomputed statistics.

        Parameters
        ----------
        array:
            float32 array of shape ``(C, H, W)``.

        Returns
        -------
        float32 array of shape ``(C, H, W)`` with per-channel zero mean and
        unit variance (approximately).
        """
        C = array.shape[0]
        mean = _BAND_MEAN[:C].reshape(C, 1, 1)
        std = _BAND_STD[:C].reshape(C, 1, 1)
        return (array - mean) / (std + 1e-8)

    @staticmethod
    def _augment(
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent random geometric augmentations to image and mask.

        Augmentations:
        - Random horizontal flip (p=0.5)
        - Random vertical flip (p=0.5)
        - Random 90-degree rotation (k ∈ {0, 1, 2, 3}, uniform)

        The **same** transformation is applied to both temporal images and the
        label so spatial alignment is preserved.

        Parameters
        ----------
        image:
            Float32 tensor of shape ``(2, C, H, W)``.
        mask:
            Long tensor of shape ``(H, W)``.

        Returns
        -------
        Augmented ``(image, mask)`` pair.
        """
        # Horizontal flip
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # Vertical flip
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        # 90-degree rotation (k times)
        k = random.randint(0, 3)
        if k > 0:
            # torch.rot90 rotates the last two dims by default
            image = torch.rot90(image, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])

        return image, mask
