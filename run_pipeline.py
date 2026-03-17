"""CLI entry point for the impervious surface change detection pipeline.

Usage
-----
    uv run python run_pipeline.py --aoi bergen --start 2021-01 --end 2021-12 \\
        --model models/checkpoints/best.ckpt --output output/bergen_2021/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Separated from :func:`main` so that tests can import the parser
    without triggering side effects.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with --aoi, --start, --end, --model, --output.
    """
    parser = argparse.ArgumentParser(
        description="Impervious surface change detection pipeline. "
        "Produces monthly change polygons from Sentinel-2 imagery "
        "covering the requested area and date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aoi",
        required=True,
        choices=["bergen", "houston"],
        help="Study area: 'bergen' (Norway) or 'houston' (TX, USA).",
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM",
        help="Start of the date range (inclusive), e.g. 2021-01.",
    )
    parser.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM",
        help="End of the date range (inclusive), e.g. 2021-12.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to a PyTorch Lightning .ckpt model checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory. Will be created if it does not exist.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    parser = build_parser()
    args = parser.parse_args()

    from surface_change_monitor.pipeline import PipelineError, run_pipeline

    try:
        output_path = run_pipeline(
            aoi_name=args.aoi,
            start_date=args.start,
            end_date=args.end,
            model_path=args.model,
            output_dir=args.output,
        )
        print(f"Pipeline complete. Output saved to: {output_path}")
    except PipelineError as exc:
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
