"""
dl_loco_evaluation.py — Leave-One-City-Out (LOCO) evaluation for binary MLP.

Proves the model generalises to unseen cities by training on N-1 cities
and testing on the held-out city. Repeated for each labelled city.

Matches the final pipeline exactly:
  CSF on full cloud → sample ground-only → binary MLP (sidewalk vs street)

Labelled cities are auto-detected — no hardcoded list.

Usage:
    python dl_loco_evaluation.py
    python dl_loco_evaluation.py --epochs 50
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import laspy

from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

from utils import (SEED, load_city, sample_and_filter, build_segments,
                   add_context_features, get_common_features,
                   MODEL_SIDEWALK, MODEL_STREET)
from dl_classifier import train_mlp_binary, predict_mlp_binary
from dl_train_final_model import detect_labelled_cities, process_city_for_training

np.random.seed(SEED)
torch.manual_seed(SEED)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BINARY MLP LOCO EVALUATION — Leave One City Out")
    print("Pipeline: CSF → ground-only sample → binary sidewalk/street MLP")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}")

    # ── Step 1: Auto-detect labelled cities ───────────────────────────────
    LABELLED_CITIES = detect_labelled_cities()
    if len(LABELLED_CITIES) < 3:
        print(f"\n[ERROR] Need at least 3 labelled cities for LOCO. "
              f"Found: {LABELLED_CITIES}")
        exit(1)
    print(f"\nLOCO cities: {LABELLED_CITIES}")

    # ── Step 2: Check CSF ─────────────────────────────────────────────────
    try:
        import CSF
        CSF_AVAILABLE = True
        print("CSF available — ground filter will run before sampling")
    except ImportError:
        CSF_AVAILABLE = False
        print("[WARNING] CSF not available — install: pip install cloth-simulation-filter")

    # ── Step 3: Common features across all cities ─────────────────────────
    print("\nComputing common features across all labelled cities...")
    COMMON_FEAT_NAMES = get_common_features(LABELLED_CITIES)
    print(f"  Using {len(COMMON_FEAT_NAMES)} common features "
          f"-> {len(COMMON_FEAT_NAMES) * 3} after context")

    # ── Step 4: Process all cities once ───────────────────────────────────
    print("\nProcessing all cities (single pass each)...")
    city_data = {}
    for city in LABELLED_CITIES:
        if not os.path.exists(f"preprocessed/{city}/low_featured.laz"):
            print(f"  Skipping {city} — file not found")
            continue
        result = process_city_for_training(city, COMMON_FEAT_NAMES, CSF_AVAILABLE)
        if result is not None:
            city_data[city] = result

    if len(city_data) < 3:
        print(f"\n[ERROR] Only {len(city_data)} cities processed. Need at least 3.")
        exit(1)

    available = list(city_data.keys())

    # ── Step 5: LOCO loop ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    loco_results = []

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
        print(f"Test:  {len(X_test):,}  segments from {test_city}")

        # Check both classes present
        if len(np.unique(y_train)) < 2:
            print("  Skipping — training data missing a class")
            continue
        if len(np.unique(y_test)) < 2:
            print("  Skipping — test data missing a class")
            continue

        # Shuffle + 75/25 train/val split
        rng      = np.random.RandomState(SEED)
        perm     = rng.permutation(len(X_train))
        X_train  = X_train[perm]
        y_train  = y_train[perm]
        val_size = int(len(X_train) * 0.25)
        X_val    = X_train[:val_size]
        y_val    = y_train[:val_size]
        X_tr     = X_train[val_size:]
        y_tr     = y_train[val_size:]

        print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | "
              f"Test ({test_city}): {len(X_test):,}")

        # Train binary MLP
        start = time.time()
        model, scaler, device, _, _ = train_mlp_binary(
            X_tr, y_tr, X_val, y_val, epochs=args.epochs)
        print(f"  Training done in {time.time() - start:.1f}s")

        # Evaluate on all splits
        results_row = {
            "held_out_city": test_city,
            "train_cities" : "+".join(train_cities),
            "n_train"      : len(X_tr),
            "n_val"        : len(X_val),
            "n_test"       : len(X_test),
        }

        for split_name, X_eval, y_eval in [("train", X_tr,   y_tr),
                                            ("val",   X_val,  y_val),
                                            ("test",  X_test, y_test)]:
            preds   = predict_mlp_binary(model, scaler, X_eval, device)
            bal_acc = balanced_accuracy_score(y_eval, preds)
            sw_f1   = f1_score(y_eval, preds, pos_label=1, zero_division=0)
            print(f"\n  {split_name.upper()} — "
                  f"Balanced Acc: {bal_acc*100:.1f}% | Sidewalk F1: {sw_f1:.3f}")
            print(classification_report(y_eval, preds,
                                        target_names=["street", "sidewalk"],
                                        zero_division=0))
            results_row[f"{split_name}_bal_acc"] = round(bal_acc * 100, 1)
            results_row[f"{split_name}_sw_f1"]   = round(sw_f1, 3)

        loco_results.append(results_row)

    # ── Step 6: Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LOCO SUMMARY — Binary MLP (sidewalk vs street)")
    print(f"{'='*60}")

    df = pd.DataFrame(loco_results)
    display_cols = ["held_out_city", "train_cities",
                    "train_bal_acc", "val_bal_acc", "test_bal_acc",
                    "train_sw_f1",  "val_sw_f1",   "test_sw_f1"]
    print(df[display_cols].to_string(index=False))

    print(f"\nMean test balanced accuracy: {df['test_bal_acc'].mean():.1f}%")
    print(f"Mean test sidewalk F1:       {df['test_sw_f1'].mean():.3f}")

    df.to_csv("results/mlp_loco_results.csv", index=False)
    print(f"\nResults saved to results/mlp_loco_results.csv")
    print(f"\nTo train final model: python dl_train_final_model.py")
