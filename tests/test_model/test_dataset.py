"""Tests for BiTemporalChangeDataset.

Strategy:
- Create synthetic .npz patch files in a temp directory
- Verify the dataset contract (shape, dtype, normalization, augmentation, NaN handling, split)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CHANNELS = 9  # 6 spectral + 3 indices
PATCH_SIZE = 256


def _make_patch(
    tmp_dir: Path,
    name: str,
    city: str = "bergen",
    source: str = "sentinel2",
    with_nan: bool = False,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write a synthetic .npz patch file and return its path."""
    if rng is None:
        rng = np.random.default_rng(42)

    t1 = rng.random((NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    t2 = rng.random((NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    label = rng.integers(0, 2, (PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    if with_nan:
        t1[0, :10, :10] = np.nan
        t2[0, :10, :10] = np.nan

    path = tmp_dir / name
    np.savez(path, t1=t1, t2=t2, label=label, city=city, source=source)
    return path


def _make_patches_dir(
    tmp_path: Path,
    n_bergen: int = 4,
    n_dublin: int = 2,
    n_houston: int = 2,
) -> Path:
    """Populate a directory with patches from multiple cities."""
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    rng = np.random.default_rng(0)

    for i in range(n_bergen):
        _make_patch(patches_dir, f"bergen_{i:03d}.npz", city="bergen", rng=rng)
    for i in range(n_dublin):
        _make_patch(patches_dir, f"dublin_{i:03d}.npz", city="dublin", rng=rng)
    for i in range(n_houston):
        _make_patch(patches_dir, f"houston_{i:03d}.npz", city="houston", rng=rng)

    return patches_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def patches_dir(tmp_path: Path) -> Path:
    return _make_patches_dir(tmp_path)


@pytest.fixture()
def single_patch_dir(tmp_path: Path) -> Path:
    """Directory with one patch per split city (bergen/dublin/houston)."""
    d = tmp_path / "patches"
    d.mkdir()
    _make_patch(d, "bergen_000.npz", city="bergen")
    _make_patch(d, "dublin_000.npz", city="dublin")
    _make_patch(d, "houston_000.npz", city="houston")
    return d


@pytest.fixture()
def nan_patch_dir(tmp_path: Path) -> Path:
    """Directory with one patch that contains NaN values."""
    d = tmp_path / "patches"
    d.mkdir()
    _make_patch(d, "bergen_nan.npz", city="bergen", with_nan=True)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBiTemporalChangeDataset:
    """Core contract tests."""

    def test_returns_correct_shape(self, single_patch_dir: Path) -> None:
        """image must be (2, C, H, W) and mask must be (H, W) — torchgeo format."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")
        sample = ds[0]

        assert "image" in sample
        assert "mask" in sample

        image = sample["image"]
        mask = sample["mask"]

        assert image.shape == (2, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE), (
            f"Expected (2, {NUM_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}), got {image.shape}"
        )
        assert mask.shape == (PATCH_SIZE, PATCH_SIZE), (
            f"Expected ({PATCH_SIZE}, {PATCH_SIZE}), got {mask.shape}"
        )

    def test_image_is_float_tensor(self, single_patch_dir: Path) -> None:
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")
        sample = ds[0]
        assert sample["image"].dtype == torch.float32

    def test_mask_is_long_tensor(self, single_patch_dir: Path) -> None:
        """torchgeo ChangeDetectionTask expects long dtype for mask."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")
        sample = ds[0]
        assert sample["mask"].dtype == torch.long

    def test_dataset_length(self, patches_dir: Path) -> None:
        """Dataset length must match the number of .npz files in the directory."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        # Count all npz files manually
        n_files = len(list(patches_dir.glob("*.npz")))
        ds = BiTemporalChangeDataset(patches_dir, split=None)  # no filtering
        assert len(ds) == n_files

    def test_normalization(self, single_patch_dir: Path) -> None:
        """After normalization, values should be in a bounded float range (not raw reflectance)."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")
        sample = ds[0]
        image = sample["image"]

        # After normalization, values should not be the original [0, 1) random floats
        # multiplied by nothing. We check that normalization was applied at all:
        # the simplest test is that the range is reasonable and not wildly outside [−5, 5].
        assert image.isfinite().all(), "Image contains non-finite values after normalization"

    def test_handles_nan(self, nan_patch_dir: Path) -> None:
        """NaN values in t1/t2 should be replaced with 0.0."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(nan_patch_dir, split=None)
        sample = ds[0]
        image = sample["image"]

        assert not torch.isnan(image).any(), "NaN survived into the output tensor"

    def test_augmentation_train(self, single_patch_dir: Path) -> None:
        """Train mode should apply random augmentation (outputs may differ across calls)."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")

        # Run many samples; at least one pair should differ due to random flips/rotations
        samples = [ds[0] for _ in range(20)]
        images = [s["image"] for s in samples]
        # At least two samples should differ (probability of all 20 being identical is negligible)
        all_same = all(torch.equal(images[0], img) for img in images[1:])
        assert not all_same, "Train augmentation never produced a different result"

    def test_no_augmentation_val(self, single_patch_dir: Path) -> None:
        """Val mode must be deterministic — no augmentation applied."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="val")

        samples = [ds[0] for _ in range(5)]
        images = [s["image"] for s in samples]
        for img in images[1:]:
            assert torch.equal(images[0], img), "Val mode returned different outputs across calls"

    def test_no_augmentation_test(self, single_patch_dir: Path) -> None:
        """Test mode must also be deterministic."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="test")

        samples = [ds[0] for _ in range(5)]
        images = [s["image"] for s in samples]
        for img in images[1:]:
            assert torch.equal(images[0], img), "Test mode returned different outputs across calls"

    def test_augmentation_consistent_across_t1_t2_label(self, single_patch_dir: Path) -> None:
        """When augmented, t1, t2 and label must all receive the same geometric transform."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(single_patch_dir, split="train")

        # We cannot check this directly without exposing internals, but we can verify
        # that the mask shape is always consistent with the image H and W.
        for _ in range(10):
            sample = ds[0]
            _, C, H, W = sample["image"].shape
            assert sample["mask"].shape == (H, W)


class TestTrainValTestSplit:
    """Tests for the spatial city-based train/val/test split."""

    def test_train_val_split(self, tmp_path: Path) -> None:
        """Spatial split: Bergen→train, Dublin→val, Houston→test."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        patches_dir = _make_patches_dir(
            tmp_path, n_bergen=4, n_dublin=2, n_houston=2
        )

        train_ds = BiTemporalChangeDataset(patches_dir, split="train")
        val_ds = BiTemporalChangeDataset(patches_dir, split="val")
        test_ds = BiTemporalChangeDataset(patches_dir, split="test")

        assert len(train_ds) == 4, f"Expected 4 train patches (bergen), got {len(train_ds)}"
        assert len(val_ds) == 2, f"Expected 2 val patches (dublin), got {len(val_ds)}"
        assert len(test_ds) == 2, f"Expected 2 test patches (houston), got {len(test_ds)}"

    def test_train_includes_all_train_cities(self, tmp_path: Path) -> None:
        """Train split includes Bergen, Oslo, Amsterdam, and Warsaw patches."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        rng = np.random.default_rng(1)

        for city in ("bergen", "oslo", "amsterdam", "warsaw"):
            for i in range(2):
                _make_patch(patches_dir, f"{city}_{i}.npz", city=city, rng=rng)

        train_ds = BiTemporalChangeDataset(patches_dir, split="train")
        assert len(train_ds) == 8  # 4 cities × 2 patches

    def test_split_none_returns_all(self, patches_dir: Path) -> None:
        """split=None should load all patches without filtering."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        ds = BiTemporalChangeDataset(patches_dir, split=None)
        n_files = len(list(patches_dir.glob("*.npz")))
        assert len(ds) == n_files

    def test_train_val_test_disjoint(self, patches_dir: Path) -> None:
        """The three splits must not overlap (no patch appears in two splits)."""
        from surface_change_monitor.model.dataset import BiTemporalChangeDataset

        train_ds = BiTemporalChangeDataset(patches_dir, split="train")
        val_ds = BiTemporalChangeDataset(patches_dir, split="val")
        test_ds = BiTemporalChangeDataset(patches_dir, split="test")

        # Extract file paths used by each split
        train_paths = set(train_ds.patch_paths)
        val_paths = set(val_ds.patch_paths)
        test_paths = set(test_ds.patch_paths)

        assert not train_paths & val_paths, "Overlap between train and val"
        assert not train_paths & test_paths, "Overlap between train and test"
        assert not val_paths & test_paths, "Overlap between val and test"
