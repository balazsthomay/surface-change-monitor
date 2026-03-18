# Impervious Surface Change Detection for Flood Model Updating

A pipeline that detects imperviousness changes from Sentinel-2 imagery at monthly cadence and 10m resolution, outputting geospatial change polygons that tell a flood modeling system where to re-run hydrodynamic simulations.

Designed to plug into flood modeling pipelines where imperviousness is the primary driver of urban pluvial flooding but upstream data updates every 1-3 years. This fills that gap.

## The Problem

High-resolution flood models depend on accurate imperviousness data for surface runoff estimation, but the available sources lag years behind:

- **Copernicus HRL** (Europe): 10m, updated every 3 years. Latest: 2021 reference year. Underestimates sealed area in Norway by ~33%.
- **NLCD** (US): 30m, annual but far too coarse for building-level detection.

A new parking lot can completely change the flood dynamics of a neighborhood. This pipeline detects that change within a month of it happening, not years later.

## What It Does

Given an area of interest and date range:

1. **Ingests** Sentinel-2 L2A imagery from Copernicus Data Space (B02-B08, B11-B12 at 10m)
2. **Composites** monthly cloud-free medians using SCL masking (handles Bergen's coastal cloud cover)
3. **Detects** per-pixel change probability between bi-temporal composite pairs using a Siamese CNN
4. **Outputs** GeoJSON/GeoPackage polygons with confidence scores, area, and change type classification

The output is a data product, not a visualization. Each polygon is a trigger for selective re-simulation.

## Results

Trained on 5 European cities (Bergen, Oslo, Amsterdam, Warsaw as train; Dublin as validation) using HRL 2018->2021 as labels. Tested on Houston for cross-geography generalization.

### Bergen (2018 -> 2021)

| Metric | Value |
|--------|-------|
| Best pixel F1 | 0.23 (threshold 0.6) |
| Pixel precision | 0.17 |
| Pixel recall | 0.35 |
| HRL change rate | 6.0% of pixels |
| Detected polygons | 115 |

![Threshold sweep](output/validation/bergen_threshold_sweep.png)

### Houston (2018 -> 2021)

| Metric | Value |
|--------|-------|
| Detected polygons | 467 (across 3 month pairs) |
| Cross-geography | European-trained model applied to Texas |

No quantitative Houston validation — NLCD ground truth requires AWS requester-pays access.

### Limitations

- **F1 is low** (0.23). Root causes: narrow tile coverage per city (single S2 tile), only one HRL epoch pair (2018->2021), 1,039 training patches total. More tiles and augmentation would improve this significantly.
- **Polygon IoU is zero** at IoU>0.3 threshold — the model detects change regions but with spatial imprecision. Typical for models trained on coarse labels applied at finer resolution.
- **Seasonal false positives** from vegetation changes are partially mitigated by summer-only compositing but not eliminated.

## Architecture

```
Sentinel-2 L2A (CDSE)
    |
    v
STAC Search -> Band Download (OData Nodes API) -> SCL Cloud Mask
    |
    v
Monthly Median Composite (nanmedian across 3-5 scenes)
    |
    v
Spectral Indices (NDVI, NDBI, NDWI) -> 9-channel input
    |
    v
Bi-Temporal Siamese CNN (fcsiamdiff, ResNet-50 backbone)
    |  - torchgeo ChangeDetectionTask
    |  - BCE loss with pos_weight for class imbalance (~6% change pixels)
    |  - Tiled inference with Gaussian overlap blending
    v
Change Probability Map (0-1, per pixel)
    |
    v
Morphological Cleanup -> Vectorization -> Area Filter (>200m2)
    |
    v
GeoJSON / GeoPackage polygons with confidence, area, change type
```

### Training Data

Binary change labels generated from Copernicus HRL Imperviousness Density:
- Epochs: 2018 and 2021 (the only usable pair with Sentinel-2 coverage)
- Change = imperviousness increase >= 10 percentage points
- Cities: Bergen, Oslo, Amsterdam, Warsaw (train), Dublin (val), Houston (test)
- 1,039 patches at 128x128 pixels, 64-pixel stride

### Model

- **Architecture**: FC-Siam-Diff (Fully Convolutional Siamese Difference) via torchgeo
- **Backbone**: ResNet-50 (ImageNet pretrained)
- **Input**: 9 channels (B02, B03, B04, B08, B11, B12, NDVI, NDBI, NDWI) x 2 temporal images
- **Output**: per-pixel change probability
- **Training**: 30 epochs on MPS (Apple M4 Pro), batch size 4, lr=1e-4, pos_weight=12
- **Val F1**: 0.29 (Dublin held-out city)

## Quick Start

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Copernicus Data Space account (free, for Sentinel-2 access)

### Setup

```bash
git clone <repo-url>
cd surface-change-monitor
uv sync

# Add CDSE credentials
echo "CDSE_USERNAME=your@email.com" >> .env
echo "CDSE_PASSWORD=yourpassword" >> .env
```

### Run the Pipeline

```bash
# Full pipeline (requires trained model checkpoint)
uv run python run_pipeline.py \
    --aoi bergen \
    --start 2021-06 \
    --end 2021-09 \
    --model models/checkpoints/best.ckpt \
    --output output/bergen_2021/
```

### Step-by-Step (Data Acquisition + Training)

```bash
# 1. Download Sentinel-2 composites
uv run python -c "
from scripts.acquire_data import acquire_composites
acquire_composites('bergen', 2021, [6, 7, 8])
acquire_composites('bergen', 2018, [6, 7, 8])
"

# 2. Download HRL labels
uv run python scripts/download_hrl.py
uv run python scripts/download_hrl_2021.py

# 3. Extract training patches
uv run python -c "
from scripts.extract_patches import extract_city_patches
for city in ['bergen', 'oslo', 'amsterdam', 'warsaw', 'dublin']:
    extract_city_patches(city, patch_size=128, stride=64)
"

# 4. Train model
uv run python scripts/train_model.py \
    --patches-dir data/patches \
    --epochs 50 \
    --batch-size 4

# 5. Run inference
uv run python scripts/run_inference.py \
    --city bergen \
    --model models/checkpoints/best.ckpt

# 6. Generate validation report
uv run python -c "from scripts.generate_validation import main; main()"
```

### Tests

```bash
uv run pytest                                          # 340 tests
uv run pytest --cov=surface_change_monitor             # with coverage (96%)
uv run pytest tests/test_config.py -v                  # single module
```

## Project Structure

```
surface_change_monitor/
    config.py          # AOIs, band specs, API endpoints
    auth.py            # CDSE OAuth2 token management
    stac.py            # Sentinel-2 scene discovery
    download.py        # Band download with retry + AOI clipping
    cloud_mask.py      # SCL-based cloud masking
    composite.py       # Monthly median compositing
    indices.py         # NDVI, NDBI, NDWI
    postprocess.py     # Vectorization + change classification
    validate.py        # Metrics + reporting
    pipeline.py        # End-to-end orchestration
    labels/
        hrl.py         # Copernicus HRL processing
        nlcd.py        # NLCD processing
        change.py      # Binary change label generation
    model/
        dataset.py     # BiTemporalChangeDataset
        train.py       # Lightning training loop
        predict.py     # Tiled inference with Gaussian blending

scripts/               # Data acquisition + training helpers
run_pipeline.py        # CLI entry point
tests/                 # 340 tests, 96% coverage
```

## Data Sources

| Source | Coverage | Resolution | Update | Use |
|--------|----------|------------|--------|-----|
| [Sentinel-2 L2A](https://dataspace.copernicus.eu) | Global | 10m | 5 days | Input imagery |
| [Copernicus HRL](https://land.copernicus.eu) | Europe | 10m | 3 years | Training labels |
| [NLCD](https://www.mrlc.gov) | CONUS | 30m | Annual | Houston labels (requires AWS creds) |
| [MS Building Footprints](https://source.coop) | Global | Vector | Periodic | Change type classification |

## Deviations from Original Design

1. **Single HRL epoch pair** (2018->2021) instead of 6 epochs. Sentinel-2 launched late 2015, HRL 2015 is 20m, and 2015-2018 has documented inconsistency.
2. **128x128 patches** instead of 256x256. Narrow Sentinel-2 tile coverage (one tile per city) limits usable width.
3. **OData Nodes API** for band download instead of direct STAC asset URLs. CDSE STAC returns S3 paths requiring separate credentials.
4. **Summer-only compositing** (June-August) to minimize seasonal vegetation artifacts. Bergen August 2018 had zero scenes under 40% cloud.
5. **OSCD pretraining deferred**. Multi-city HRL labels provide more relevant training signal than OSCD's 24 pairs with different band combinations.
