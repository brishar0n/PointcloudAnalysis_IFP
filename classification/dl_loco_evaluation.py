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

        val_size = int(len(X_train) * 0.1)
        X_val    = X_train[:val_size]
        y_val    = y_train[:val_size]
        X_tr     = X_train[val_size:]
        y_tr     = y_train[val_size:]

        start = time.time()
        model, scaler, device, _, _ = train_mlp(
            X_tr, y_tr, X_val, y_val, epochs=args.epochs)
        elapsed = time.time() - start
        print(f"Training done in {elapsed:.1f}s")

        y_pred  = predict_mlp(model, scaler, X_test, device)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        sw_f1   = f1_score(y_test, y_pred, labels=[1],
                           average=None, zero_division=0)[0]

        print(f"\nResults for {test_city}:")
        print(f"Balanced Accuracy: {bal_acc*100:.1f}%  |  Sidewalk F1: {sw_f1:.3f}")

        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred, labels=[0,1,2],
                                     target_names=["other","sidewalk","street"],
                                     zero_division=0))

        os.makedirs("results", exist_ok=True)
        cm      = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=["other", "sidewalk", "street"],
                    yticklabels=["other", "sidewalk", "street"])
        plt.title(f"MLP LOCO — Hold out {test_city.upper()}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"results/mlp_loco_{test_city}_confusion.png", dpi=150)
        plt.show()

        loco_results.append({
            "held_out_city" : test_city,
            "n_train"       : len(X_train),
            "n_test"        : len(X_test),
            "bal_acc"       : round(bal_acc * 100, 1),
            "sidewalk_f1"   : round(sw_f1, 3),
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
    print(f"\n✅ Results saved to results/mlp_loco_results.csv")
    print(f"✅ Plots saved to results/")
    print(f"\nTo train final model run: python dl_train_final_model.py")
