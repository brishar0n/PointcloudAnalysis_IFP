# Classification Module (`feature/sidewalk-classifier`)

**PIC: Vency Khunt**

Classifies LiDAR point cloud data into three categories — **street**, **sidewalk**, and **other** — using a deep learning pipeline based on a Multi-Layer Perceptron (MLP) neural network trained on voxel segment features.


---

## Project Structure

```
classification/
├── utils.py                    # Shared functions used by all scripts
├── dl_classifier.py            # Train + evaluate MLP on a single city
├── dl_loco_evaluation.py       # LOCO cross-city evaluation
├── dl_train_final_model.py     # Train final production model on all cities
├── dl_apply_model.py           # Apply final model to any city
├── preprocessed/               # Output from Brigitte's preprocessing (NOT in git)
│   ├── riga/
│   │   ├── low_featured.laz
│   │   ├── full_featured.laz
│   │   └── train_data_flat.npz
│   ├── vilnius/
│   ├── warsaw/
│   ├── utrecht/
│   └── bologna/
├── models/                     # Saved model files (NOT in git)
│   ├── final_mlp_classifier.pt
│   └── final_mlp_scaler.joblib
├── classified/                 # Output classified .laz files (NOT in git)
│   ├── riga_mlp_classified.laz
│   ├── utrecht_mlp_classified.laz
│   └── bologna_mlp_classified.laz
└── results/                    # Evaluation plots and CSV results (NOT in git)
    ├── mlp_loco_results.csv
    └── *.png
```

> **Note:** `preprocessed/`, `models/`, `classified/` and `results/` are excluded from git because they contain large files. See the **Setup** section for how to obtain the preprocessed data.

---

## Setup

### Requirements

Python 3.9+

```bash
pip install laspy[lazrs] numpy scipy scikit-learn torch matplotlib seaborn pandas joblib
```

### Getting the Preprocessed Data

The `preprocessed/` folder is **not included in git** due to large file sizes. You have two options:

**Option A — Download from shared drive:**
Download the `preprocessed/` folder from the team's shared Teams/Google Drive and place it inside the `classification/` folder.

**Option B — Run Brigitte's preprocessing yourself:**
```bash
# From the project root (not classification/)
python run_preprocessing.py datasets/<city-name>_subsampled.laz -o classification/preprocessed/<city-name>
```

---

## Quick Start

### Step 1 — Train the final model (run once)

```bash
cd classification/
python dl_train_final_model.py --epochs 50
```

This trains on Riga + Vilnius + Warsaw combined and saves:
- `models/final_mlp_classifier.pt`
- `models/final_mlp_scaler.joblib`

### Step 2 — Apply to any city

```bash
python dl_apply_model.py --city <city-name>
```

Output saved to `classified/{city}_mlp_classified.laz`

---

## What the Pipeline Does

### Stage 1 — Roughness Pre-filter

Before any machine learning, points with roughness > 0.05m are removed. Roads and sidewalks are smooth surfaces — anything rougher than 5cm is likely a parked car, bush, or damaged surface. This reduces class imbalance before training.

### Stage 2 — Voxel Superpoint Segmentation

The XY plane is divided into a grid of 0.6m × 0.6m squares. All points that fall inside the same square are grouped into one **superpoint**. Each superpoint gets the average features of all its points and a majority-vote label.

This reduces 250,000 points down to approximately 3,000–12,000 segments depending on the city.

**Why segments instead of individual points:** Classifying millions of individual points produces noisy results because each point is classified in isolation with no spatial context. Segments are more stable and allow us to add context features from neighbouring patches.

### Stage 3 — Context Features

For each superpoint, we find its 64 nearest neighbour superpoints using a KD-Tree. We compute the mean and standard deviation of those neighbours' features and add them to the superpoint's own features. This triples the feature count from 41 to 123 per segment.

We tested 8, 16, 32, 64, and 128 neighbours. 64 neighbours gave the best sidewalk F1 score.

**Why context features matter:** A flat smooth patch could be a road or a sidewalk when viewed in isolation. With context — knowing what surrounds it — the model can distinguish them. Sidewalks are typically narrower and adjacent to buildings, roads are wider and more central.

### Stage 4 — MLP Neural Network

A 3-layer Multi-Layer Perceptron classifies each segment into one of 3 classes. Class weights are applied so the model pays extra attention to sidewalk, which is typically the minority class.

**Architecture:**
```
Input (123 features)
→ Dense(256) + BatchNorm + ReLU + Dropout(30%)
→ Dense(128) + BatchNorm + ReLU + Dropout(30%)
→ Dense(64)  + BatchNorm + ReLU + Dropout(20%)
→ Dense(3)   → other / sidewalk / street
```

**Why MLP over Random Forest:** In cross-city LOCO evaluation, MLP achieved 71.2% balanced accuracy vs RF's 68.8%. More importantly, sidewalk F1 improved from 0.460 to 0.576 — a 25% improvement in detecting the most critical class.

**Why dropout:** Dropout randomly turns off neurons during training, preventing the model from memorising one city's patterns. This improves cross-city generalization.

### Stage 5 — Projection Back to Points

After classifying segments, predictions are projected back to individual points. Each point receives the label of its segment. The output is saved as a `.laz` file with the classification field set to IFP standard codes.

---

## Input

| File | Source | Description |
|---|---|---|
| `preprocessed/{city}/low_featured.laz` | Brigitte's preprocessing | Street-level points with all 41 geometric features attached |

**Features used (41 per point, 123 per segment after context):**

| Feature | Description |
|---|---|
| `eigenvalue_1/2/3_{r}` | Covariance eigenvalues at radii 0.1, 0.5, 1.0m |
| `planarity_{r}` | How flat the local surface is |
| `linearity_{r}` | How linear the local surface is |
| `roughness_{r}` | Distance from point to local best-fit plane |
| `height_above_min` | Height above lowest local point |
| `normalized_z` | Global height normalized to [0,1] |
| `density` | Number of nearby points |
| `intensity_normalized` | Laser reflection intensity |

---

## Output

| File | Description |
|---|---|
| `classified/{city}_mlp_classified.laz` | Classified point cloud for Ahmed |
| `models/final_mlp_classifier.pt` | Trained MLP model weights |
| `models/final_mlp_scaler.joblib` | Feature scaler |
| `results/mlp_loco_results.csv` | LOCO evaluation results |
| `results/*.png` | Confusion matrices and training curves |

**Classification labels in output .laz file:**

| Code | Label | Description |
|---|---|---|
| 0 | other | Vegetation, buildings, vehicles, street furniture |
| 2 | sidewalk | Pedestrian surfaces and pavements |
| 11 | street | Carriageway and road surface |

---


## Datasets

| City | Points | Labels | Used For |
|---|---|---|---|
| Riga | 5.0M | ✅ Yes | Training + LOCO evaluation |
| Vilnius | 7.8M | ✅ Yes | Training + LOCO evaluation |
| Warsaw | 2.9M | ✅ Yes | Training + LOCO evaluation |
| Utrecht | 6.5M | ❌ All class 0 | Prediction only |
| Bologna | 14.2M | ❌ All class 0 | Prediction only |

Utrecht and Bologna have no ground truth labels in the provided datasets. The trained model is applied to these cities for prediction. Visual inspection in CloudCompare confirms street detection. Quantitative validation would require IFP to provide a small labelled sample.

---

## CLI Reference

### dl_classifier.py — Single city training and evaluation

```bash
python dl_classifier.py --city <city-name>
python dl_classifier.py --city <city-name> --epochs int
python dl_classifier.py --city <city-name> --sample int
```

| Argument | Default | Description |
|---|---|---|
| `--city` | riga | City name |
| `--epochs` | 50 | Training epochs |
| `--sample` | 250000 | Points to sample |

### dl_loco_evaluation.py — Cross-city LOCO evaluation

```bash
python dl_loco_evaluation.py
python dl_loco_evaluation.py --epochs 50
```

### dl_train_final_model.py — Train production model

```bash
python dl_train_final_model.py
python dl_train_final_model.py --epochs 50
```

### dl_apply_model.py — Apply to new city

```bash
python dl_apply_model.py --city <city-name>
```

---

## Notes for Team


- **Ahmed (boundary extraction):** Your input is `classified/{city}_mlp_classified.laz`. Classification codes: `2=sidewalk`, `11=street`, `0=other`. Run `dl_train_final_model.py` first, then `dl_apply_model.py --city yourcity`.

- **Sujeeth (metrics):** Use Ahmed's boundary output — classification is upstream of your module.

- **Aaron (visualisation):** All classified `.laz` files open directly in CloudCompare with classification as a scalar field. The pipeline entry point is `dl_apply_model.py`.
