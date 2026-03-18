"""Train the change detection model on extracted patches.

Usage:
    uv run python scripts/train_model.py --patches-dir data/patches --epochs 50 --batch-size 2
"""

import argparse
import logging
from pathlib import Path

from surface_change_monitor.model.dataset import BiTemporalChangeDataset
from surface_change_monitor.model.train import create_task, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument("--patches-dir", type=Path, default=Path("data/patches"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints"))
    args = parser.parse_args()

    # Create datasets with spatial split
    log.info("Loading datasets...")
    train_ds = BiTemporalChangeDataset(args.patches_dir, split="train")
    val_ds = BiTemporalChangeDataset(args.patches_dir, split="val")

    log.info(f"Train: {len(train_ds)} patches, Val: {len(val_ds)} patches")

    if len(train_ds) == 0:
        # Fall back to using all patches for both train and val
        log.warning("No train patches found with spatial split, using all patches")
        train_ds = BiTemporalChangeDataset(args.patches_dir, split=None)
        val_ds = train_ds
        log.info(f"Using all {len(train_ds)} patches for both train and val")

    if len(train_ds) == 0:
        log.error("No patches found!")
        return

    # Create model
    log.info(f"Creating model (in_channels=9, pos_weight={args.pos_weight})")
    task = create_task(in_channels=9, pos_weight=args.pos_weight, lr=args.lr)

    # Train
    log.info(f"Training for {args.epochs} epochs, batch_size={args.batch_size}")
    best_ckpt = train(
        train_ds, val_ds, task,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    log.info(f"Training complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
