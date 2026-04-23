"""
dl_apply_model.py — Apply trained MLP model to new unlabelled cities.

Loads the final MLP model trained on all labelled cities.
Applies it to any new city (Utrecht, Bologna, or any new scan).
Saves classified .laz file for Ahmed's boundary extraction.

Usage:
    python dl_apply_model.py --city utrecht
    python dl_apply_model.py --city bologna
"""

import numpy as np
import laspy
import joblib
import os
import argparse
import torch

from utils import (
    SEED, SAMPLE_SIZE, VOXEL_SIZE, MIN_POINTS,
    load_city, sample_and_filter, build_segments,
    add_context_features
)
from dl_classifier import SidewalkMLP, predict_mlp

np.random.seed(SEED)
torch.manual_seed(SEED)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city",   required=True)
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MLP Apply Model — {args.city.upper()}")
    print(f"{'='*60}")

    # ── Load final MLP model ───────────────────────────────────────────────
    model_path  = "models/final_mlp_classifier.pt"
    scaler_path = "models/final_mlp_scaler.joblib"

    if not os.path.exists(model_path):
        print("❌ Final MLP model not found!")
        print("   Run dl_loco_evaluation.py first!")
        exit(1)

    scaler = joblib.load(scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load and process city ──────────────────────────────────────────────
    xyz, labels, features, feat_names             = load_city(args.city)
    xyz_s, labels_s, features_s                   = sample_and_filter(
                                                        xyz, labels, features,
                                                        feat_names, args.sample)
    seg_xyz, seg_feats, seg_labels, \
        unique_voxels, inverse_idx                 = build_segments(
                                                        xyz_s, labels_s, features_s)
    X_seg, _                                       = add_context_features(
                                                        seg_xyz, seg_feats)

    # Align features
    expected = scaler.n_features_in_
    if X_seg.shape[1] > expected:
        X_seg = X_seg[:, :expected]
    elif X_seg.shape[1] < expected:
        pad   = np.zeros((X_seg.shape[0], expected - X_seg.shape[1]))
        X_seg = np.hstack([X_seg, pad])

    # Load model
    model = SidewalkMLP(input_dim=expected)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    print("\nClassifying segments...")
    predictions = predict_mlp(model, scaler, X_seg, device)

    # Project back to points
    point_preds = np.zeros(len(xyz_s), dtype=np.uint8)
    seg_counter = 0
    for i in range(len(unique_voxels)):
        mask = inverse_idx == i
        if mask.sum() < MIN_POINTS:
            continue
        pred     = predictions[seg_counter]
        ifp_code = 2 if pred == 1 else (11 if pred == 2 else 0)
        point_preds[mask] = ifp_code
        seg_counter += 1

    # Print distribution
    print("\nPredicted labels distribution:")
    unique_c, counts_c = np.unique(point_preds, return_counts=True)
    names_map = {0: "other", 2: "sidewalk", 11: "street"}
    for cls, count in zip(unique_c, counts_c):
        print(f"  {cls} ({names_map.get(int(cls), 'other')}): "
              f"{count:,} ({100*count/len(point_preds):.1f}%)")

    # Save .laz
    os.makedirs("classified", exist_ok=True)
    header                 = laspy.LasHeader(point_format=0, version="1.2")
    las_out                = laspy.LasData(header)
    las_out.x              = xyz_s[:, 0]
    las_out.y              = xyz_s[:, 1]
    las_out.z              = xyz_s[:, 2]
    las_out.classification = point_preds
    las_out.write(f"classified/{args.city}_mlp_classified.laz")
    print(f"\n✅ Saved classified/{args.city}_mlp_classified.laz for Ahmed!")
