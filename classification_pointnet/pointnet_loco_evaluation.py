"""
pointnet_loco_evaluation.py — LOCO evaluation for PointNet.

Trains on all cities except one, tests on held-out city.
Uses train_data_blocks.npz format (raw point blocks).

Recommended: Run on GPU. On CPU expect ~30-60 min per fold.

Usage:
    python pointnet_loco_evaluation.py --epochs 50
    python pointnet_loco_evaluation.py --epochs 50 --batch 8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                             f1_score, classification_report)

from pointnet import PointNetSeg
from pointnet_classifier import PointCloudBlockDataset, remap, load_blocks
from utils import SEED

np.random.seed(SEED)
torch.manual_seed(SEED)

LABELLED_CITIES = ["riga", "vilnius", "warsaw"]


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"PointNet LOCO EVALUATION")
    print(f"Cities: {LABELLED_CITIES}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}")

    # ── Load all cities ────────────────────────────────────────────────────
    print("\nLoading all cities...")
    city_blocks = {}
    city_features = {}
    for city in LABELLED_CITIES:
        try:
            train_b, test_b, n_feat = load_blocks(city)
            city_blocks[city]   = (train_b, test_b)
            city_features[city] = n_feat
            print(f"  {city}: {len(train_b)} train, {len(test_b)} test blocks")
        except FileNotFoundError as e:
            print(f"  Skipping {city}: {e}")

    available = list(city_blocks.keys())

    # Use minimum feature count for consistency
    min_features = min(city_features.values())
    print(f"\nUsing {min_features} features per point across all cities")

    # ── LOCO loop ──────────────────────────────────────────────────────────
    loco_results = []

    for test_city in available:
        print(f"\n{'='*60}")
        print(f"Fold: Hold out {test_city.upper()}")
        print(f"{'='*60}")

        train_cities = [c for c in available if c != test_city]

        # Combine training blocks from all training cities
        all_train_blocks = []
        for c in train_cities:
            all_train_blocks.extend(city_blocks[c][0])

        test_blocks = city_blocks[test_city][1]

        print(f"Train blocks: {len(all_train_blocks)} from {train_cities}")
        print(f"Test blocks:  {len(test_blocks)} from {test_city}")

        # Val split
        val_size         = max(1, int(len(all_train_blocks) * 0.1))
        val_blocks       = all_train_blocks[:val_size]
        train_blocks_fold = all_train_blocks[val_size:]

        train_loader = DataLoader(
            PointCloudBlockDataset(train_blocks_fold),
            batch_size=args.batch, shuffle=True, drop_last=True)
        val_loader   = DataLoader(
            PointCloudBlockDataset(val_blocks),
            batch_size=args.batch, shuffle=False)
        test_loader  = DataLoader(
            PointCloudBlockDataset(test_blocks),
            batch_size=args.batch, shuffle=False)

        # Class weights
        all_labels = remap(np.concatenate(
            [b[2] for b in train_blocks_fold]))
        classes, counts = np.unique(all_labels, return_counts=True)
        weights    = 1.0 / counts
        weights    = weights / weights.sum() * len(classes)
        weight_vec = np.ones(3, dtype=np.float32)
        for c, w in zip(classes, weights):
            if c < 3:
                weight_vec[c] = w
        class_weights = torch.FloatTensor(weight_vec).to(device)

        # Model
        model     = PointNetSeg(d_in=min_features + 3,
                                num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5)

        # Train
        best_val_acc = 0.0
        start        = time.time()

        for epoch in range(args.epochs):
            model.train()
            for xyz, feat, labels in train_loader:
                xyz, feat = xyz.to(device), feat.to(device)
                labels    = torch.LongTensor(
                    remap(labels.numpy())).to(device)
                optimizer.zero_grad()
                out  = model(xyz, feat).transpose(1, 2)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            preds_v, true_v = [], []
            with torch.no_grad():
                for xyz, feat, labels in val_loader:
                    xyz, feat = xyz.to(device), feat.to(device)
                    labels    = remap(labels.numpy())
                    out       = model(xyz, feat)
                    preds     = out.argmax(dim=-1).cpu().numpy()
                    preds_v.extend(preds.flatten())
                    true_v.extend(labels.flatten())

            val_acc = balanced_accuracy_score(true_v, preds_v)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Val Acc: {val_acc*100:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(),
                           f"models/loco_{test_city}_pointnet.pt")

        elapsed = time.time() - start
        print(f"Training done in {elapsed/60:.1f} min")

        # Test
        model.load_state_dict(torch.load(
            f"models/loco_{test_city}_pointnet.pt",
            map_location=device))
        model.eval()

        preds_t, true_t = [], []
        with torch.no_grad():
            for xyz, feat, labels in test_loader:
                xyz, feat = xyz.to(device), feat.to(device)
                labels    = remap(labels.numpy())
                out       = model(xyz, feat)
                preds     = out.argmax(dim=-1).cpu().numpy()
                preds_t.extend(preds.flatten())
                true_t.extend(labels.flatten())

        bal_acc = balanced_accuracy_score(true_t, preds_t)
        sw_f1   = f1_score(true_t, preds_t, labels=[1],
                           average=None, zero_division=0)[0]

        print(f"\nResults for {test_city}:")
        print(f"Balanced Accuracy: {bal_acc*100:.1f}%  |  "
              f"Sidewalk F1: {sw_f1:.3f}")
        print(classification_report(true_t, preds_t,
                                     target_names=["other","sidewalk","street"],
                                     zero_division=0))

        # Confusion matrix
        os.makedirs("results", exist_ok=True)
        cm      = confusion_matrix(true_t, preds_t, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=["other", "sidewalk", "street"],
                    yticklabels=["other", "sidewalk", "street"])
        plt.title(f"PointNet LOCO — Hold out {test_city.upper()}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"results/pointnet_loco_{test_city}_confusion.png",
                    dpi=150)
        plt.show()

        loco_results.append({
            "held_out_city": test_city,
            "n_train_blocks": len(all_train_blocks),
            "n_test_blocks" : len(test_blocks),
            "bal_acc"       : round(bal_acc * 100, 1),
            "sidewalk_f1"   : round(sw_f1, 3),
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("POINTNET LOCO SUMMARY")
    print(f"{'='*60}")
    results_df = pd.DataFrame(loco_results)
    print(results_df.to_string(index=False))
    print(f"\nAverage balanced accuracy: "
          f"{results_df['bal_acc'].mean():.1f}%")
    print(f"Average sidewalk F1:       "
          f"{results_df['sidewalk_f1'].mean():.3f}")

    results_df.to_csv("results/pointnet_loco_results.csv", index=False)
    print(f"\n✅ Results saved to results/pointnet_loco_results.csv")
