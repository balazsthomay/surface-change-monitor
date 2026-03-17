"""Model training for bi-temporal change detection.

Wraps torchgeo's ChangeDetectionTask (a LightningModule) with a
Lightning Trainer configured for production use:

- ModelCheckpoint monitoring val_OverallF1Score, saving top 3
- EarlyStopping with patience=10
- Automatic device selection (MPS on Mac, CUDA on GPU server, CPU fallback)

Typical usage
-------------
>>> from surface_change_monitor.model.train import create_task, train
>>> task = create_task(pos_weight=15.0)
>>> best_ckpt = train(train_ds, val_ds, task, max_epochs=50, output_dir=Path("models/"))

pos_weight guidance
-------------------
Change pixels are ~5-10% of all pixels.  A pos_weight in the range 10-19
compensates for this imbalance when using BCE loss.

Implementation note
-------------------
torchgeo's ChangeDetectionTask uses BCE loss with a model that outputs
(B, 1, H, W), but the standard torchgeo/BiTemporalChangeDataset returns
masks of shape (H, W) — batched to (B, H, W).  PyTorch's BCE loss requires
matching shapes, so BinaryChangeDetectionTask subclasses ChangeDetectionTask
and unsqueezes the mask before computing loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import ChangeDetectionTask

# Metric monitored for checkpointing and early stopping.
# ChangeDetectionTask (via ClassificationMixin) logs this name automatically.
_MONITOR_METRIC = "val_OverallF1Score"


class BinaryChangeDetectionTask(ChangeDetectionTask):
    """ChangeDetectionTask with a mask-shape fix for binary BCE training.

    torchgeo's ChangeDetectionTask outputs ``(B, 1, H, W)`` logits for binary
    change detection, but does not unsqueeze the ``(B, H, W)`` mask before
    computing BCE loss, causing a shape mismatch.  This subclass corrects that
    by unsqueezing to ``(B, 1, H, W)`` before delegating to the parent.
    """

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        # Unsqueeze (B, H, W) mask -> (B, 1, H, W) for BCE shape compatibility
        if self.hparams["loss"] == "bce" and batch["mask"].ndim == 3:
            batch = dict(batch)  # shallow copy to avoid mutating the original
            batch["mask"] = batch["mask"].unsqueeze(1)
        return super()._shared_step(batch, batch_idx, stage)


def create_task(
    in_channels: int = 9,
    model: str = "fcsiamdiff",
    backbone: str = "resnet50",
    lr: float = 1e-4,
    pos_weight: float | None = None,
) -> BinaryChangeDetectionTask:
    """Create a configured BinaryChangeDetectionTask ready for training.

    Parameters
    ----------
    in_channels:
        Number of spectral/index channels per time-step image.  The dataset
        produces 9-channel patches (6 Sentinel-2 bands + NDVI, NDWI, NDBI).
    model:
        torchgeo model name.  ``fcsiamdiff`` (FC-Siam-Diff) takes a
        ``(B, 2, C, H, W)`` Siamese input and is the recommended default.
    backbone:
        timm / smp encoder name (e.g. ``"resnet50"``, ``"resnet18"``).
    lr:
        Initial learning rate for the Adam optimizer.
    pos_weight:
        Scalar weight applied to the positive class in BCE loss.  Pass a
        value in 10-19 to compensate for ~5-10% change-pixel prevalence.
        ``None`` uses unweighted BCE.

    Returns
    -------
    BinaryChangeDetectionTask
        A Lightning module ready to be passed to :func:`train`.
    """
    pw: torch.Tensor | None = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], dtype=torch.float32)

    return BinaryChangeDetectionTask(
        model=model,
        backbone=backbone,
        in_channels=in_channels,
        loss="bce",
        pos_weight=pw,
        lr=lr,
        patience=10,
    )


def train(
    train_dataset: Dataset,
    val_dataset: Dataset,
    task: ChangeDetectionTask,
    max_epochs: int = 50,
    batch_size: int = 8,
    output_dir: Path = Path("models/"),
) -> Path:
    """Train the change detection task and return the path to the best checkpoint.

    Parameters
    ----------
    train_dataset:
        Dataset returning ``{"image": (2, C, H, W), "mask": (H, W)}`` dicts.
    val_dataset:
        Same format as *train_dataset*; used for validation and checkpointing.
    task:
        A configured :class:`BinaryChangeDetectionTask` (e.g. returned by
        :func:`create_task`).
    max_epochs:
        Upper bound on training epochs.  EarlyStopping may stop earlier.
    batch_size:
        DataLoader batch size.  Use 8 on an A100, 2 on an M4 Pro.
    output_dir:
        Directory where checkpoints are saved.  Created automatically if absent.

    Returns
    -------
    Path
        Absolute path to the best checkpoint file (``*.ckpt``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    num_workers = 0  # safe default; increase for production if needed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="change-{epoch:03d}-{val_OverallF1Score:.4f}",
        monitor=_MONITOR_METRIC,
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=False,
    )

    early_stop_cb = EarlyStopping(
        monitor=_MONITOR_METRIC,
        mode="max",
        patience=10,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Trainer - automatic device selection
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=str(output_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit(task, train_loader, val_loader)

    # ------------------------------------------------------------------
    # Return best checkpoint path
    # ------------------------------------------------------------------
    best: str | None = checkpoint_cb.best_model_path
    if best:
        return Path(best)

    # Fallback: return last checkpoint if best was never recorded
    last: str | None = checkpoint_cb.last_model_path
    if last:
        return Path(last)

    # Final fallback: find any .ckpt in output_dir
    ckpts = sorted(output_dir.glob("*.ckpt"))
    if ckpts:
        return ckpts[-1]

    raise RuntimeError(
        f"Training completed but no checkpoint found in {output_dir}. "
        "Check that the trainer saved at least one checkpoint."
    )
