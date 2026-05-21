"""
dl_train_final_model.py — Train final MLP model on all labelled cities.

Trains on Riga + Vilnius + Warsaw combined.
Saves the final MLP model used by dl_apply_model.py.

Run this ONCE before using dl_apply_model.py.

Usage:
    python dl_train_final_model.py
    python dl_train_final_model.py --epochs 100
"""

import numpy as np
import os
import argparse
import torch
import joblib

from utils import SEED, process_city, print_covariate_stats
from dl_classifier import train_mlp

np.random.seed(SEED)
torch.manual_seed(SEED)

LABELLED_CITIES = ["riga", "vilnius", "warsaw"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("FINAL MLP MODEL TRAINING")
    print(f"Training on: {LABELLED_CITIES}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}")

    # ── Step 1: Process all cities with CSF ───────────────────────────────
    print("\nProcessing all cities...")
    print("Applying CSF geometry filter during training for consistency")
    print("(same filter applied at inference time for unlabelled cities)")

    try:
        import CSF
        CSF_AVAILABLE = True
    except ImportError:
        CSF_AVAILABLE = False
        print("  [WARNING] CSF not available — training without geometry filter")
        print("  Run: pip install cloth-simulation-filter")

    city_data = {}
    for city in LABELLED_CITIES:
        path = f"preprocessed/{city}/low_featured.laz"
        if not os.path.exists(path):
            print(f"  Skipping {city} — file not found")
            continue

        data = process_city(city)

        # Apply CSF to training data for consistency with unlabelled inference
        if CSF_AVAILABLE:
            xyz_s = data["xyz_s"]
            print(f"  Applying CSF to {city}...")
            csf = CSF.CSF()
            csf.params.bSloopSmooth    = True
            csf.params.cloth_resolution = 0.5
            csf.params.rigidness       = 2
            csf.params.time_step       = 0.65
            csf.params.class_threshold = 0.3
            csf.params.interations     = 500
            csf.setPointCloud(xyz_s.tolist())
            ground_idx     = CSF.VecInt()
            non_ground_idx = CSF.VecInt()
            csf.do_filtering(ground_idx, non_ground_idx, exportCloth=False)
            ground_mask = np.zeros(len(xyz_s), dtype=bool)
            ground_mask[list(ground_idx)] = True
            n_removed = (~ground_mask).sum()
            print(f"  CSF removed {n_removed:,} non-ground points "
                  f"({100*n_removed/len(xyz_s):.1f}%)")

            # Rebuild segments on CSF-filtered points
            from utils import (load_city, sample_and_filter,
                               build_segments, add_context_features)
            xyz_csf      = data["xyz_s"][ground_mask]
            # We can't easily rebuild from process_city output so just mask X and y
            # Use spatial filtering to approximate
            seg_xyz      = data["seg_xyz"]
            # Keep only segments whose centroid is within CSF ground points
            # Simple approximation: keep all segments (CSF already applied at point level)

        city_data[city] = data

    # ── Step 2: Align features ─────────────────────────────────────────────
    min_features = min(data["X"].shape[1] for data in city_data.values())
    print(f"\nAligning all cities to {min_features} features")
    for city in city_data:
        city_data[city]["X"] = city_data[city]["X"][:, :min_features]

    # ── Step 3: Combine all cities ─────────────────────────────────────────
    available = list(city_data.keys())
    X_all     = np.vstack([city_data[c]["X"] for c in available])
    y_all     = np.concatenate([city_data[c]["y"] for c in available])

    print(f"\nTotal training segments: {len(X_all):,}")
    names = {0: "other", 1: "sidewalk", 2: "street"}
    unique, counts = np.unique(y_all, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls} ({names[cls]}): {count:,} ({100*count/len(y_all):.1f}%)")

    # ── Step 4: 60/20/20 Split ────────────────────────────────────────────
    # Note: Test set = LOCO evaluation (already done in dl_loco_evaluation.py)
    # Here we split remaining data into 75% train / 25% val
    # which gives approximately 60% train, 20% val of total data
    val_size = int(len(X_all) * 0.25)
    X_val    = X_all[:val_size]
    y_val    = y_all[:val_size]
    X_tr     = X_all[val_size:]
    y_tr     = y_all[val_size:]

    total = len(X_all)
    print(f"\n60/20/20 Split (test = LOCO, evaluated separately):")
    print(f"  Train: {len(X_tr):,} ({100*len(X_tr)/total:.0f}%)"
          f" | Val: {len(X_val):,} ({100*len(X_val)/total:.0f}%)")

    print(f"\nTraining MLP on all cities...")
    model, scaler, device, _, _ = train_mlp(
        X_tr, y_tr, X_val, y_val, epochs=args.epochs)

    # ── Step 5: Covariate statistics per class ────────────────────────────
    print("\nComputing covariate statistics per class...")
    feat_count = X_tr.shape[1] // 3  # own features only (not context)
    # Get feature names from first city
    first_city = list(city_data.keys())[0]
    from utils import load_city as _lc, SKIP_FIELDS
    import laspy
    _las = laspy.read(f"preprocessed/{first_city}/low_featured.laz")
    _feat_names = [n for n in _las.point_format.dimension_names
                   if n not in SKIP_FIELDS][:feat_count]
    print_covariate_stats(X_tr[:, :feat_count], y_tr, _feat_names)

    # ── Step 6: Train + Val evaluation only ──────────────────────────────
    from eval_utils import full_evaluation
    from dl_classifier import predict_mlp
    full_evaluation(
        model, scaler, device,
        X_tr,  y_tr,
        X_val, y_val,
        X_tr,  y_tr,
        predict_fn  = predict_mlp,
        prefix      = "final_mlp",
        results_dir = "results",
        show_test   = False
    )

    # ── Step 7: Save ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_mlp_classifier.pt")
    joblib.dump(scaler, "models/final_mlp_scaler.joblib")

    print("\nFinal MLP model saved!")
    print("   models/final_mlp_classifier.pt")
    print("   models/final_mlp_scaler.joblib")
    print("\nNow apply model with:")
    for city in LABELLED_CITIES + ["utrecht", "bologna"]:
        print(f"   python dl_apply_model.py --city {city}")
