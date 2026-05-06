"""
dl_loco_evaluation.py — Leave-One-City-Out LOCO evaluation for MLP.

Proves the MLP model generalizes to unseen cities.
Does NOT train final model — use dl_train_final_model.py for that.

Usage:
    python dl_loco_evaluation.py
    python dl_loco_evaluation.py --epochs 50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import torch

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from utils import SEED, process_city
from dl_classifier import train_mlp, predict_mlp

np.random.seed(SEED)
torch.manual_seed(SEED)

LABELLED_CITIES = ["riga", "vilnius", "warsaw"]


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("MLP LOCO EVALUATION — Leave One City Out")
    print(f"Cities: {LABELLED_CITIES}")
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

    # ── Step 3: LOCO loop ─────────────────────────────────────────────────
    loco_results = []
    available    = list(city_data.keys())

    for test_city in available:
        print(f"\n{'='*60}")
        print(f"Fold: Hold out {test_city.upper()}")
        print(f"{'='*60}")

        train_cities = [c for c in available if c != test_city]

        X_train = np.vstack([city_data[c]["X"] for c in train_cities])
        y_train = np.concatenate([city_data[c]["y"] for c in train_cities])
        X_test  = city_data[test_city]["X"]
        y_test  = city_data[test_city]["y"]

        print(f"Train: {len(X_train):,} segments from {train_cities}")
        print(f"Test:  {len(X_test):,} segments from {test_city}")

        if not {0,1,2}.issubset(set(np.unique(y_train))):
            print("Skipping — training missing classes")
            continue
        if not {0,1,2}.issubset(set(np.unique(y_test))):
            print("Skipping — test missing classes")
            continue

        # ── 60/20/20 split ────────────────────────────────────────────────
        val_size = int(len(X_train) * 0.25)
        X_val    = X_train[:val_size]
        y_val    = y_train[:val_size]
        X_tr     = X_train[val_size:]
        y_tr     = y_train[val_size:]

        total = len(X_train)
        print(f"\n60/20/20 Split:")
        print(f"  Train: {len(X_tr):,} ({100*len(X_tr)/total:.0f}%)"
              f" | Val: {len(X_val):,} ({100*len(X_val)/total:.0f}%)"
              f" | Test city: {test_city} ({len(X_test):,} segments)")

        start = time.time()
        model, scaler, device, _, _ = train_mlp(
            X_tr, y_tr, X_val, y_val, epochs=args.epochs)
        elapsed = time.time() - start
        print(f"Training done in {elapsed:.1f}s")

        # ── Full evaluation: train / val / test ───────────────────────────
        from eval_utils import full_evaluation
        metrics = full_evaluation(
            model, scaler, device,
            X_tr,    y_tr,
            X_val,   y_val,
            X_test,  y_test,
            predict_fn  = predict_mlp,
            prefix      = f"mlp_loco_{test_city}",
            results_dir = "results"
        )

        loco_results.append({
            "held_out_city" : test_city,
            "n_train"       : len(X_tr),
            "n_val"         : len(X_val),
            "n_test"        : len(X_test),
            "train_acc"     : round(metrics["train_acc"] * 100, 1),
            "val_acc"       : round(metrics["val_acc"]   * 100, 1),
            "bal_acc"       : round(metrics["test_acc"]  * 100, 1),
            "train_sw_f1"   : round(metrics["train_sw_f1"], 3),
            "val_sw_f1"     : round(metrics["val_sw_f1"],   3),
            "sidewalk_f1"   : round(metrics["test_sw_f1"],  3),
        })

    # ── Step 4: Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MLP LOCO SUMMARY")
    print(f"{'='*60}")
    results_df = pd.DataFrame(loco_results)
    print(results_df.to_string(index=False))
    print(f"\nAverage balanced accuracy: {results_df['bal_acc'].mean():.1f}%")
    print(f"Average sidewalk F1:       {results_df['sidewalk_f1'].mean():.3f}")

    results_df.to_csv("results/mlp_loco_results.csv", index=False)
    print(f"\nResults saved to results/mlp_loco_results.csv")
    print(f"Plots saved to results/")
    print(f"\nTo train final model run: python dl_train_final_model.py")
