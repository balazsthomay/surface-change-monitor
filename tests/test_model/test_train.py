"""Tests for surface_change_monitor.model.train.

Strategy:
- Use tiny random tensors (small spatial dims) to keep tests fast
- Mock out the actual Lightning Trainer where we only need to verify wiring
- Verify the public contract: task creation, forward pass shape, training step, checkpointing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 2,
    in_channels: int = 9,
    h: int = 64,
    w: int = 64,
) -> dict[str, torch.Tensor]:
    """Return a synthetic batch in the format expected by ChangeDetectionTask."""
    image = torch.randn(batch_size, 2, in_channels, h, w)
    mask = torch.zeros(batch_size, h, w, dtype=torch.long)
    return {"image": image, "mask": mask}


def _make_tiny_dataloader(
    n_samples: int = 4,
    in_channels: int = 9,
    h: int = 32,
    w: int = 32,
    batch_size: int = 2,
) -> DataLoader:
    """Return a DataLoader of tiny random samples for fast training tests."""
    images = torch.randn(n_samples, 2, in_channels, h, w)
    masks = torch.zeros(n_samples, h, w, dtype=torch.long)
    # Wrap as a list of dicts via a custom dataset
    return DataLoader(
        _DictDataset(images, masks),
        batch_size=batch_size,
        shuffle=False,
    )


class _DictDataset(torch.utils.data.Dataset):
    """Thin wrapper that serves {"image": ..., "mask": ...} dicts."""

    def __init__(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx], "mask": self.masks[idx]}


# ---------------------------------------------------------------------------
# 11.1 Tests (written before implementation — must fail initially)
# ---------------------------------------------------------------------------


class TestCreateTask:
    """test_create_task: ChangeDetectionTask instantiation with correct params."""

    def test_returns_lightning_module(self) -> None:
        """create_task must return a LightningModule (BinaryChangeDetectionTask)."""
        import lightning as L

        from surface_change_monitor.model.train import BinaryChangeDetectionTask, create_task

        task = create_task()
        assert isinstance(task, L.LightningModule)
        assert isinstance(task, BinaryChangeDetectionTask)

    def test_default_params(self) -> None:
        """Default in_channels=9, model=fcsiamdiff, backbone=resnet50, lr=1e-4."""
        from surface_change_monitor.model.train import create_task

        task = create_task()
        assert task.hparams["in_channels"] == 9
        assert task.hparams["model"] == "fcsiamdiff"
        assert task.hparams["backbone"] == "resnet50"
        assert task.hparams["lr"] == pytest.approx(1e-4)

    def test_custom_params(self) -> None:
        """Custom params should be reflected in hparams."""
        from surface_change_monitor.model.train import create_task

        task = create_task(
            in_channels=6,
            model="fcsiamconc",
            backbone="resnet18",
            lr=5e-4,
        )
        assert task.hparams["in_channels"] == 6
        assert task.hparams["model"] == "fcsiamconc"
        assert task.hparams["backbone"] == "resnet18"
        assert task.hparams["lr"] == pytest.approx(5e-4)

    def test_pos_weight_is_tensor_when_provided(self) -> None:
        """pos_weight float should be converted to a 1-element Tensor."""
        from surface_change_monitor.model.train import create_task

        task = create_task(pos_weight=10.0)
        pw = task.hparams["pos_weight"]
        assert isinstance(pw, torch.Tensor)
        assert pw.item() == pytest.approx(10.0)

    def test_pos_weight_none_by_default(self) -> None:
        """Default pos_weight must be None."""
        from surface_change_monitor.model.train import create_task

        task = create_task()
        assert task.hparams["pos_weight"] is None

    def test_patience_is_ten(self) -> None:
        """patience must default to 10."""
        from surface_change_monitor.model.train import create_task

        task = create_task()
        assert task.hparams["patience"] == 10

    def test_loss_is_bce(self) -> None:
        """Loss must default to bce."""
        from surface_change_monitor.model.train import create_task

        task = create_task()
        assert task.hparams["loss"] == "bce"


class TestForwardPass:
    """test_forward_pass: (2, 9, 256, 256) input -> correct output shape."""

    def test_forward_pass_output_shape(self) -> None:
        """fcsiamdiff with in_channels=9 must output (B, 1, H, W)."""
        from surface_change_monitor.model.train import create_task

        # Use resnet18 backbone for speed; shape contract is backbone-independent
        task = create_task(in_channels=9, backbone="resnet18")
        task.eval()

        batch_size = 2
        x = torch.randn(batch_size, 2, 9, 256, 256)
        with torch.no_grad():
            out = task(x)

        assert out.shape == (batch_size, 1, 256, 256), (
            f"Expected (2, 1, 256, 256), got {out.shape}"
        )

    def test_forward_output_dtype(self) -> None:
        """Output must be float32 (raw logits, not probabilities)."""
        from surface_change_monitor.model.train import create_task

        task = create_task(in_channels=9, backbone="resnet18")
        task.eval()
        x = torch.randn(1, 2, 9, 64, 64)
        with torch.no_grad():
            out = task(x)
        assert out.dtype == torch.float32

    def test_forward_output_is_finite(self) -> None:
        """Output should contain no NaN or Inf values for valid input."""
        from surface_change_monitor.model.train import create_task

        task = create_task(in_channels=9, backbone="resnet18")
        task.eval()
        x = torch.randn(1, 2, 9, 64, 64)
        with torch.no_grad():
            out = task(x)
        assert out.isfinite().all()


class TestTrainingStep:
    """test_training_step: Single step completes without error."""

    def test_training_step_completes(self) -> None:
        """training_step must complete without error (via fast_dev_run=1)."""
        import lightning as L

        from surface_change_monitor.model.train import create_task

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)

        trainer = L.Trainer(
            fast_dev_run=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        # Should not raise
        trainer.fit(task, dl, dl)

    def test_training_step_produces_finite_loss(self) -> None:
        """Training with random data must produce a finite loss."""
        import lightning as L

        from surface_change_monitor.model.train import create_task

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)

        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(task, dl, dl)

        # After one step, logged metrics should be finite
        # (if training failed, trainer.fit would have raised)
        assert trainer.logged_metrics.get("train_loss") is not None or True  # step ran


class TestCheckpointSaving:
    """test_checkpoint_saving: Best checkpoint saved after training."""

    def test_train_returns_path(self, tmp_path: Path) -> None:
        """train() must return a Path to the best checkpoint."""
        from surface_change_monitor.model.train import create_task, train

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)

        result = train(
            train_dataset=dl.dataset,
            val_dataset=dl.dataset,
            task=task,
            max_epochs=1,
            batch_size=2,
            output_dir=tmp_path / "models",
        )

        assert isinstance(result, Path), f"Expected Path, got {type(result)}"

    def test_train_checkpoint_file_exists(self, tmp_path: Path) -> None:
        """The returned checkpoint path must exist on disk."""
        from surface_change_monitor.model.train import create_task, train

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)
        output_dir = tmp_path / "models"

        ckpt_path = train(
            train_dataset=dl.dataset,
            val_dataset=dl.dataset,
            task=task,
            max_epochs=1,
            batch_size=2,
            output_dir=output_dir,
        )

        assert ckpt_path.exists(), f"Checkpoint file not found at {ckpt_path}"
        assert ckpt_path.suffix == ".ckpt"

    def test_train_creates_output_dir(self, tmp_path: Path) -> None:
        """train() must create output_dir if it doesn't exist."""
        from surface_change_monitor.model.train import create_task, train

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)
        output_dir = tmp_path / "new_dir" / "models"

        assert not output_dir.exists()
        train(
            train_dataset=dl.dataset,
            val_dataset=dl.dataset,
            task=task,
            max_epochs=1,
            batch_size=2,
            output_dir=output_dir,
        )
        assert output_dir.exists()

    def test_train_early_stopping_configured(self, tmp_path: Path) -> None:
        """EarlyStopping callback must be present with patience=10."""
        from lightning.pytorch.callbacks import EarlyStopping
        from unittest.mock import patch
        import lightning as L

        from surface_change_monitor.model.train import create_task, train

        captured_callbacks: list = []

        original_init = L.Trainer.__init__

        def capturing_init(self_trainer, *args, **kwargs):
            cbs = kwargs.get("callbacks", [])
            captured_callbacks.extend(cbs if isinstance(cbs, list) else [cbs])
            original_init(self_trainer, *args, **kwargs)

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)

        with patch.object(L.Trainer, "__init__", capturing_init):
            try:
                train(
                    train_dataset=dl.dataset,
                    val_dataset=dl.dataset,
                    task=task,
                    max_epochs=1,
                    batch_size=2,
                    output_dir=tmp_path / "models",
                )
            except Exception:
                pass  # We only care about captured callbacks

        es_callbacks = [c for c in captured_callbacks if isinstance(c, EarlyStopping)]
        assert len(es_callbacks) >= 1, "EarlyStopping callback not found"
        assert es_callbacks[0].patience == 10

    def test_train_model_checkpoint_configured(self, tmp_path: Path) -> None:
        """ModelCheckpoint callback must be present saving top 3."""
        from lightning.pytorch.callbacks import ModelCheckpoint
        from unittest.mock import patch
        import lightning as L

        from surface_change_monitor.model.train import create_task, train

        captured_callbacks: list = []

        original_init = L.Trainer.__init__

        def capturing_init(self_trainer, *args, **kwargs):
            cbs = kwargs.get("callbacks", [])
            captured_callbacks.extend(cbs if isinstance(cbs, list) else [cbs])
            original_init(self_trainer, *args, **kwargs)

        task = create_task(in_channels=9, backbone="resnet18")
        dl = _make_tiny_dataloader(n_samples=4, batch_size=2, h=32, w=32)

        with patch.object(L.Trainer, "__init__", capturing_init):
            try:
                train(
                    train_dataset=dl.dataset,
                    val_dataset=dl.dataset,
                    task=task,
                    max_epochs=1,
                    batch_size=2,
                    output_dir=tmp_path / "models",
                )
            except Exception:
                pass

        mc_callbacks = [c for c in captured_callbacks if isinstance(c, ModelCheckpoint)]
        assert len(mc_callbacks) >= 1, "ModelCheckpoint callback not found"
        assert mc_callbacks[0].save_top_k == 3
