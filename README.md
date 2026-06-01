# Classification Module (`feature/sidewalk-classifier`)

PIC: Vency

Classifies street-level LiDAR points into **sidewalk**, **street**, and **other** using a two-stage pipeline: a geometric ground filter followed by a deep learning binary classifier.

This is the second stage of the Sidewalk Scanner pipeline. It takes the preprocessed output from Brigitte's module and produces classified point clouds that feed into boundary extraction (Ahmed) and width metrics (Sujeeth).

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: `laspy[lazrs]`, `numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`, `pandas`, `joblib`, `cloth-simulation-filter`

A GPU is recommended for training but not required (the model runs on CPU if CUDA is unavailable).

## Project structure

```
├── classification/
│   ├── utils.py                  # Shared pipeline functions and constants
│   ├── eval_utils.py             # Evaluation helpers
│   ├── dl_classifier.py          # Binary MLP model definition and training
│   ├── dl_train_final_model.py   # Train final model on all labelled cities
│   ├── dl_loco_evaluation.py     # Leave-One-City-Out cross-city evaluation
│   ├── dl_apply_model.py         # Apply trained model to any city
│   ├── run_pipeline.py           # CLI entry point — runs everything
│   ├── preprocessed/             # Brigitte's preprocessed input (NOT in git)
│   ├── models/                   # Saved model files (NOT in git)
│   ├── classified/               # Output .laz files (NOT in git)
│   └── results/                  # Plots and evaluation results (NOT in git)
└── README.md
```

## Quick start

Place Brigitte's preprocessed output at `preprocessed/<city>/low_featured.laz` first, then:

```bash
# Train the model once on all labelled cities
python3 run_pipeline.py --train --epochs 50

# Apply the trained model to a city
python3 run_pipeline.py --apply --cities riga

# Run Leave-One-City-Out cross-city evaluation
python3 run_pipeline.py --loco --epochs 50

# Run everything (train + apply + evaluate)
python3 run_pipeline.py --all --epochs 50
```

The model only needs to be trained once. After training, use `--apply` for each city.

## What the pipeline does

The pipeline runs in two stages so that each model focuses on the part of the problem it handles best.

1. **Cloth Simulation Filter (CSF)**: A geometric ground filter runs on the full point cloud before sampling. It removes buildings, trees, and vehicles, keeping only ground points. Non-ground points are automatically labelled `other` (code 0). This handles the easy geometric separation so the neural network does not have to.
2. **Sample and filter**: Samples 650,000 ground points and applies a roughness filter (points with roughness > 0.05m, likely parked cars or bushes, are removed).
3. **Voxel segmentation**: Points are grouped into 0.6m x 0.6m grid squares. Each square becomes one segment described by the mean of its features and a majority-vote label. A purity filter removes boundary segments where fewer than 85% of points agree on the label.
4. **Context features**: Each segment is enriched with the mean and standard deviation of its 64 nearest neighbouring segments, giving 111 features total (37 per-point x 3: own + context mean + context std).
5. **Binary MLP (sidewalk vs street)**: A multilayer perceptron trained only on ground segments classifies each segment as sidewalk or street. Class weighting handles the imbalance between the minority sidewalk class and the majority street class.
6. **Unlabelled cities only**: Pseudo-label fine-tuning adapts the model to a new city using its own high-confidence predictions (>= 95%), then a connected-component filter removes isolated sidewalk patches smaller than 50 points.

### Why two stages instead of one three-class model?

An initial three-class MLP (sidewalk / street / other) struggled because sidewalk and street are both flat ground surfaces, and their subtle differences in intensity and height were drowned out when the model also had to learn building and tree separation. Letting CSF handle non-ground geometrically lets the binary MLP focus entirely on the hard sidewalk/street distinction.

## Model architecture

```
Input (111 features)
  -> Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  -> Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  -> Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
  -> Dense(2)
Output: 0 = street, 1 = sidewalk
```

## Output

Each run produces:

```
classified/<city>_mlp_classified.laz   # Classified point cloud (input for Ahmed)
models/final_mlp_classifier.pt         # Trained MLP weights
models/final_mlp_scaler.joblib         # Feature scaler
models/common_feat_names.joblib        # Feature names used at training (for inference alignment)
results/                               # Evaluation plots and reports
```

The `.laz` files can be opened in [CloudCompare](https://www.cloudcompare.org/) and coloured by the classification scalar field.

**Classification codes in the output `.laz`:**

| Code | Label |
|------|-------|
| 0 | other |
| 2 | sidewalk |
| 11 | street |

## CLI options

```
python3 run_pipeline.py [options]

Options:
  --train               Train the final model on all labelled cities
  --apply               Apply the trained model to cities
  --cities CITY [..]    City/cities to apply the model to
  --loco                Run Leave-One-City-Out cross-city evaluation
  --all                 Run train + apply + evaluate
  --epochs INT          Number of training epochs (default: 50)
```

## Input

Each city's input is a single file from the preprocessing module:

```
preprocessed/<city>/low_featured.laz
```

This contains street-level points with precomputed geometric features (roughness, planarity, eigenvalues, height features, intensity, density). Labelled cities are auto-detected — a city is treated as labelled if it has more than 1% sidewalk AND street points. No hardcoded city lists are used.

## Datasets

| City | Approx. points | Labelled |
|------|----------------|----------|
| Riga | 1.5M | Yes |
| Vilnius | 3.8M | Yes |
| Warsaw | 1.5M | Yes |
| Bologna | 5.0M | No |
| Utrecht | 3.5M | No |

Bologna has 46 features versus 41 for the other cities (different scanner preprocessing). Common features are selected automatically across all cities at training time so the model uses a consistent feature set.

## Key constants (in `utils.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `SAMPLE_SIZE` | 650,000 | Ground points sampled per city |
| `VOXEL_SIZE` | 0.6m | Size of each segment grid square |
| `ROUGHNESS_THRESH` | 0.05m | Roughness filter cutoff |
| `PURITY_THRESH` | 0.85 | Min label agreement to keep a segment |
| `SMOOTH_THRESHOLD` | 0.6 | Neighbour agreement to override a prediction |
| `SEED` | 42 | Random seed for reproducibility |

## Notes for team

- **Preprocessing (Brigitte):** Input is `preprocessed/<city>/low_featured.laz`. The classifier relies on the geometric features computed in your stage.
- **Boundary extraction (Ahmed):** Your input is `classified/<city>_mlp_classified.laz`. Run `--train` once, then `--apply --cities <city>` for each city.
- **Width metrics (Sujeeth):** Classification is upstream of your module — use Ahmed's boundary output, which derives from these classified clouds.
- **Visualisation (Aaron):** All `.laz` outputs open in CloudCompare. Colour by the classification scalar field to see other / sidewalk / street.
