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

from utils import SEED, process_city
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

    # ── Step 1: Process all cities ─────────────────────────────────────────
    print("\nProcessing all cities...")
    city_data = {}
    for city in LABELLED_CITIES:
        path = f"preprocessed/{city}/low_featured.laz"
        if os.path.exists(path):
            city_data[city] = process_city(city)
        else:
            print(f"  Skipping {city} — file not found")

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

    # ── Step 5: Train + Val evaluation only ──────────────────────────────
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

    # ── Step 6: Save ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_mlp_classifier.pt")
    joblib.dump(scaler, "models/final_mlp_scaler.joblib")

    print("\nFinal MLP model saved!")
    print("   models/final_mlp_classifier.pt")
    print("   models/final_mlp_scaler.joblib")
    print("\nNow apply model with:")
    for city in LABELLED_CITIES + ["utrecht", "bologna"]:
        print(f"   python dl_apply_model.py --city {city}")
