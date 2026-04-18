# Preprocessing Module (`feature/pointcloud-loader`)

Loads LiDAR point cloud data (.LAZ/.LAS), computes geometric features, and prepares training datasets for the sidewalk classifier.

This is the first stage of the Sidewalk Scanner pipeline — everything downstream (classification, boundary extraction, width metrics, visualisation) depends on the outputs from this module.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: `laspy[lazrs]`, `numpy`, `scipy`

## Project structure

```
├── run_preprocessing.py          # CLI entry point — runs the full pipeline
├── preprocessing/
│   ├── loader.py                 # Read/write LAZ/LAS, subsampling
│   ├── features.py               # Eigenvalues, roughness, height, density
│   └── splitter.py               # High/low split, noise removal, train/test prep
├── requirements.txt
└── README.md
```

## Quick start

```bash

# Bologna — needs full feature computation, so subsample first
python3 run_preprocessing.py datasets/bologna_subsampled.laz -o preprocessed/bologna --subsample 0.05

# Riga, Utrecht, Vilnius — eigenvalues precomputed by IFP, runs much faster
python3 run_preprocessing.py datasets/riga_subsampled.laz -o preprocessed/riga
python3 run_preprocessing.py datasets/utrecht_subsampled.laz -o preprocessed/utrecht
python3 run_preprocessing.py datasets/vilnius_subsampled.laz -o preprocessed/vilnius
```

## What the pipeline does

The pipeline runs six steps in order:

1. **Load**: Reads the .LAZ file, normalises field names, auto-detects and shifts absolute coordinates (e.g. Vilnius).
2. **Subsample** (optional): Voxel-grid downsampling to reduce point count for faster iteration.
3. **Compute geometric features**: Eigenvalues, roughness, planarity, linearity, height-above-ground, point density. Skips computation for datasets that already have IFP-precomputed values.
4. **Remove noise**: Filters statistical outliers based on local point density.
5. **High/low split**: Separates street-level points (sidewalks, roads, cars, furniture) from elevated points (buildings, tree canopies) using a height threshold.
6. **Prepare training data**: Exports two formats for the classifier:
   - Flat arrays (`train_data_flat.npz`) for traditional ML (k-means, random forest)
   - Spatial blocks (`train_data_blocks.npz`) for deep learning (PointNet, RandLA-Net)

## Output

Each run produces:

```
preprocessed/<city>/
├── full_featured.laz         # All points with all scalar fields attached
├── low_featured.laz          # Street-level points only (input for classifier)
├── high_featured.laz         # Buildings, trees, etc.
├── train_data_flat.npz       # X_train, X_test, y_train, y_test, feature_names
└── train_data_blocks.npz     # Spatial blocks of 4096 points for DL models
```

The `.laz` files can be opened in [CloudCompare](https://www.cloudcompare.org/) to visually inspect features. The `.npz` files are picked up by the classification module.

## CLI options

```
python3 run_preprocessing.py <input.laz> [options]

Options:
  -o, --output DIR          Output directory (default: preprocessed/)
  --subsample FLOAT         Voxel size in metres (e.g. 0.05). Omit to keep full resolution.
  --radii FLOAT [FLOAT ..]  Radii for eigenvalue computation (default: 0.1 0.3 0.5)
  --height-split FLOAT      Height threshold for high/low split in metres (default: 2.0)
  --block-size FLOAT        Block size for DL training data in metres (default: 5.0)
  --skip-blocks             Skip DL block preparation
```

## Datasets

| City | Points | Precomputed features | Coordinate note |
|------|--------|---------------------|-----------------|
| Bologna | 14.2M | `height_division` only | Shifted (local origin) |
| Riga | 5.0M | Eigenvalues + roughness at 0.1, 0.5, 1.0m | Shifted |
| Utrecht | 6.5M | Eigenvalues + roughness at 0.1, 0.5, 1.0m | Shifted |
| Vilnius | 7.8M | Eigenvalues + roughness at 0.1, 0.5, 1.0m | Absolute (auto-shifted by loader) |

## Computed features

For each point, the following scalar fields are computed (or derived from precomputed values):

**Per radius (0.1m, 0.3m, 0.5m):**
- `eigenvalue_1_{r}`, `eigenvalue_2_{r}`, `eigenvalue_3_{r}` — covariance eigenvalues (descending)
- `linearity_{r}` — (λ1 − λ2) / λ1 — high for poles, wires, kerb edges
- `planarity_{r}` — (λ2 − λ3) / λ1 — high for flat surfaces (ground, walls)
- `scattering_{r}` — λ3 / λ1 — high for foliage, noise
- `omnivariance_{r}` — (λ1 × λ2 × λ3)^(1/3)
- `anisotropy_{r}` — (λ1 − λ3) / λ1
- `roughness_{r}` — distance from point to local best-fit plane

**Height features:**
- `height_above_min`: Z minus lowest Z in local neighbourhood
- `height_range`: max Z − min Z in neighbourhood
- `height_std`: standard deviation of Z in neighbourhood
- `normalized_z`: Z normalised globally to [0, 1]

**Other:**
- `density`: neighbour count within radius
- `intensity_normalized`: laser intensity normalised to [0, 1]

## Using the module in Python

```python
from preprocessing import load_point_cloud, compute_all_features, split_high_low

# Load
cloud = load_point_cloud("bologna_subsampled.laz")

# Compute features (auto-skips if precomputed)
cloud = compute_all_features(cloud, radii=[0.1, 0.3, 0.5])

# Split
low, high = split_high_low(cloud, height_threshold=2.0)

# Prepare training data for classifier
from preprocessing import prepare_training_data
X_train, X_test, y_train, y_test, feature_names = prepare_training_data(low)
```

## Notes for team

- **Classification (Person 2):** Your inputs are `train_data_flat.npz` and `train_data_blocks.npz`. The flat format works for `testing/` branches (sklearn models). The block format works for `feature/sidewalk-classifier` (PointNet/RandLA-Net). Feature names are included in both files.
- **Boundary extraction (Person 3):** Use `low_featured.laz` as your starting point — it contains only street-level points with all features attached.
- **Width metrics (Person 4):** Same — work from `low_featured.laz` or the classified output from Person 2.
- **Visualisation (Person 5):** All `.laz` outputs open directly in CloudCompare with features as scalar fields.
