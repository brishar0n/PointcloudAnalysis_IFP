# Classification Module

**PIC: Vency Khunt**

Classifies LiDAR point cloud data into three categories — **street**, **sidewalk**, and **other** — using a deep learning MLP pipeline trained on voxel segment features.

---

## Project Structure

```
classification/
├── utils.py                    # Shared pipeline functions
├── eval_utils.py               # Evaluation helpers (metrics, confusion, confidence)
├── dl_classifier.py            # Single city training and evaluation
├── dl_loco_evaluation.py       # LOCO cross-city evaluation
├── dl_train_final_model.py     # Train final model on all labelled cities
├── dl_apply_model.py           # Apply final model to any city
├── run_pipeline.py             # Single entry point — runs everything
├── preprocessed/               # From Brigitte's preprocessing (NOT in git)
├── models/                     # Saved model files (NOT in git)
├── classified/                 # Output .laz files (NOT in git)
└── results/                    # Plots and CSV results (NOT in git)
```

---

## Setup

Python 3.9+

```bash
pip install laspy[lazrs] numpy scipy scikit-learn torch matplotlib seaborn pandas joblib cloth-simulation-filter
```

Place Brigitte's preprocessed output in `classification/preprocessed/<city>/` before running.

---

## How to Run

Everything runs through `run_pipeline.py`. You do not need to call individual scripts directly.

```bash
# Run everything
python run_pipeline.py --all --epochs 50

# Run individual steps
python run_pipeline.py --loco --epochs 50         # LOCO evaluation only
python run_pipeline.py --train --epochs 50        # Train final model only
python run_pipeline.py --apply --cities utrecht bologna  # Apply to cities
```

The model only needs to be trained once. After that just use `--apply`.

---

## What the Pipeline Does

### Step 1 — Roughness Pre-filter
Points with roughness > 0.05m are removed before training. Roads and sidewalks are smooth — rough points are cars, bushes, damaged surfaces. Reduces class imbalance at source.

### Step 2 — CSF Geometry Filter (unlabelled cities only)
For cities without ground truth labels (Utrecht, Bologna), a Cloth Simulation Filter (CSF) is applied first. CSF drops a virtual blanket over the inverted point cloud and keeps only flat ground points — removing buildings, trees, and vehicles before the model sees the data.

### Step 3 — Voxel Superpoint Segmentation
The XY plane is divided into 0.6m x 0.6m grid squares. All points in the same square form one superpoint with averaged features and majority-vote label. Reduces 250,000 points to roughly 3,000–12,000 segments per city.

### Step 4 — Context Features
Each superpoint gets the mean and standard deviation of its 64 nearest neighbour superpoints added as extra features. This triples the feature count from 41 to 123 per segment. Tested with 8, 16, 32, 64, and 128 neighbours — 64 gave the best sidewalk F1.

### Step 5 — MLP Classification
A 3-layer neural network classifies each segment into other / sidewalk / street. Class weights ensure the model pays extra attention to sidewalk (minority class).

```
Input (123 features)
-> Dense(256) + BatchNorm + ReLU + Dropout(30%)
-> Dense(128) + BatchNorm + ReLU + Dropout(30%)
-> Dense(64)  + BatchNorm + ReLU + Dropout(20%)
-> Dense(3)   -> other / sidewalk / street
```

### Step 6 — Pseudo-Label Fine-Tuning (unlabelled cities only)
After initial classification, predictions with >= 95% confidence are used as pseudo-labels to fine-tune the model on the target city. No manual labelling required. The model self-improves on any new city.

### Step 7 — Connected Component Filter (unlabelled cities only)
Isolated sidewalk patches smaller than 50 points are removed. Real sidewalks are large continuous surfaces — small scattered patches are noise.

### Step 8 — Projection Back to Points
Segment predictions are mapped back to individual points and saved as a `.laz` file.

---

## Automatic Label Detection

`dl_apply_model.py` automatically detects whether a city has ground truth labels by checking if more than 1% of points have sidewalk (code 2) or street (code 11) classifications.

| City has labels | Path taken |
|---|---|
| Yes (Riga, Vilnius, Warsaw) | Apply model directly |
| No (Utrecht, Bologna) | CSF + MLP + fine-tune + cleanup |

Output filename is always `classified/{city}_mlp_classified.laz` regardless of path.

---

## Train / Validation / Test Split

| Split | How |
|---|---|
| Train | 60% — used to fit the model |
| Validation | 20% — used to monitor training and save best model |
| Test | LOCO — entire unseen city, stricter than random holdout |

LOCO (Leave-One-City-Out) trains on 2 cities and tests on the 3rd, repeated for each city. This is the honest cross-city accuracy reported as the final test result.

---

## Results

| Model | LOCO Balanced Acc | Sidewalk F1 |
|---|---|---|
| Random Forest | 68.8% | 0.460 |
| MLP (this module) | 71.2% | 0.576 |

MLP achieves 25% better sidewalk F1 than RF in cross-city evaluation.

---

## Datasets

| City | Points | Labels | Used For |
|---|---|---|---|
| Riga | 5.0M | Yes | Training + LOCO |
| Vilnius | 7.8M | Yes | Training + LOCO |
| Warsaw | 2.9M | Yes | Training + LOCO |
| Utrecht | 6.5M | No | Apply only |
| Bologna | 14.2M | No | Apply only |

---

## Output Files

| File | Description |
|---|---|
| `classified/{city}_mlp_classified.laz` | Classified point cloud for Ahmed |
| `models/final_mlp_classifier.pt` | Trained MLP weights |
| `models/final_mlp_scaler.joblib` | Feature scaler |
| `results/mlp_loco_results.csv` | LOCO summary table |
| `results/*_confusion.png` | Confusion matrices |
| `results/*_confidence.png` | Confidence histograms |

**Classification codes in output .laz:**

| Code | Label |
|---|---|
| 0 | other |
| 2 | sidewalk |
| 11 | street |

---

## Notes for Team

**Ahmed (boundary extraction):** Input is `classified/{city}_mlp_classified.laz`. Run `run_pipeline.py --train` once, then `--apply --cities <city>` for each city you need.

**Sujeeth (metrics):** Use Ahmed's boundary output — classification is upstream of your module.

**Aaron (visualisation):** All `.laz` outputs open in CloudCompare with classification as a scalar field. Colour by classification to see other/sidewalk/street.
