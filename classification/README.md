# Classification Module

**PIC: Vency Khunt**

Classifies LiDAR point cloud data into **sidewalk**, **street**, and **other** using a deep learning MLP pipeline trained on voxel segment features.

Output `.laz` files go to Ahmed for boundary extraction.

---

## Project Structure

```
classification/
├── utils.py                    # Shared pipeline functions
├── eval_utils.py               # Evaluation helpers
├── dl_classifier.py            # MLP model definition and training functions
├── dl_loco_evaluation.py       # LOCO cross-city evaluation
├── dl_train_final_model.py     # Train final model on all labelled cities
├── dl_apply_model.py           # Apply final model to any city
├── run_pipeline.py             # Single entry point — runs everything
├── preprocessed/               # Brigitte's preprocessed input (NOT in git)
├── models/                     # Saved model files (NOT in git)
├── classified/                 # Output .laz files (NOT in git)
└── results/                    # Plots and evaluation results (NOT in git)
```

---

## Setup

Python 3.9+

```bash
pip install laspy[lazrs] numpy scipy scikit-learn torch matplotlib seaborn pandas joblib cloth-simulation-filter
```

Place Brigitte's preprocessed output at `preprocessed/<city>/low_featured.laz` before running.

---

## Input

Each city's input is a single file: `preprocessed/<city>/low_featured.laz`

This file contains XYZ coordinates plus precomputed geometric features per point (roughness, planarity, eigenvalues, height features, intensity, density etc.), produced by Brigitte's preprocessing step.

---

## How to Run

Everything runs through `run_pipeline.py`.

```bash
# Train the model (run once)
python run_pipeline.py --train --epochs 50

# Apply to cities
python run_pipeline.py --apply --cities <city-name>

# Run LOCO cross-city evaluation
python run_pipeline.py --loco --epochs 50

# Run everything
python run_pipeline.py --all --epochs 50
```

The model only needs to be trained once. After training, use `--apply` for each city.

---

## Pipeline

```
preprocessed/<city>/low_featured.laz
    │
    ▼
CSF (Cloth Simulation Filter)
    Runs on the full point cloud before sampling.
    Removes buildings, trees, vehicles — keeps ground only.
    Non-ground points are labelled other (code 0) automatically.
    │
    ▼
Sample 650k ground points + roughness filter (> 0.05m removed)
    │
    ▼
Voxel segmentation (0.6m × 0.6m grid squares)
    Points in each grid square → one segment (mean features, majority label)
    Purity filter removes boundary segments < 85% label agreement
    │
    ▼
Context features
    Each segment gets mean + std of its 64 nearest neighbour segments
    Features: 37 per-point × 3 (own + ctx_mean + ctx_std) = 111 total
    │
    ▼
Binary MLP — sidewalk vs street
    Input (111) → Dense(256)+BN+ReLU+Drop → Dense(128)+BN+ReLU+Drop
               → Dense(64)+BN+ReLU+Drop → Dense(2)
    │
    ▼
[Unlabelled cities only]
    Pseudo-label fine-tuning — high confidence (≥ 95%) predictions
    used to adapt model to target city (20 epochs, lr=0.0001)
    Connected component filter — removes isolated sidewalk patches < 50 points
    │
    ▼
classified/<city>_mlp_classified.laz
```

---

## Labelled vs Unlabelled Cities

`dl_apply_model.py` automatically detects whether a city has ground truth labels (> 1% sidewalk AND street points). No hardcoded city lists.

| City has labels | Path |
|---|---|
| Yes | Predict → evaluate against ground truth |
| No  | Predict → pseudo-label fine-tune → connected component filter |

---

## Output

| File | Description |
|---|---|
| `classified/{city}_mlp_classified.laz` | Classified point cloud → Ahmed |
| `models/final_mlp_classifier.pt` | Trained MLP weights |
| `models/final_mlp_scaler.joblib` | Feature scaler |
| `models/common_feat_names.joblib` | Feature names used at training (for inference alignment) |

**Classification codes in output `.laz`:**

| Code | Label |
|---|---|
| 0 | other |
| 2 | sidewalk |
| 11 | street |

---

## Datasets

| City | Points

| Riga | 1.5M
| Vilnius | 3.8M
| Warsaw | 1.5M
| Bologna | 5.0M
| Utrecht | 3.5M 

Bologna has 46 features vs 41 for other cities (different scanner preprocessing). Common features are selected automatically across all cities at training time.

---

## Notes for Team

**Ahmed:** Input is `classified/{city}_mlp_classified.laz`. Run `--train` once, then `--apply --cities <city>` for each city.

**Sujeeth:** Classification is upstream of your module — use Ahmed's boundary output.

**Aaron:** All `.laz` outputs open in CloudCompare. Colour by classification scalar field to see other/sidewalk/street.
