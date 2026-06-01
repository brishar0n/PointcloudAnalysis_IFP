"""
dl_train_final_model.py — Train final binary MLP on all labelled cities.

Two-stage pipeline:
  CSF  — geometry filter, removes buildings/trees/cars (replaces learned Stage 1)
  MLP  — binary classifier: sidewalk(1) vs street(0) on ground-only segments

Why CSF instead of a learned Stage 1?
  Ground vs non-ground is a geometry problem. CSF (Cloth Simulation Filter)
  was designed specifically for this on LiDAR data. A learned Stage 1 would
  just approximate what CSF already does using the same geometric features.
  CSF first also means we sample 650k from ground-only points — every slot
  in our sample budget goes to sidewalk/street, not wasted on buildings.

Why binary instead of 3-class?
  The 3-class MLP confused sidewalk with street because both are flat ground.
  Once CSF handles non-ground, the binary MLP focuses entirely on the hard
  problem: subtle intensity/height differences between sidewalk and street.

Saved model files:
  models/final_mlp_classifier.pt    — binary SidewalkStreetMLP weights
  models/final_mlp_scaler.joblib    — StandardScaler fitted on training data
  models/common_feat_names.joblib   — feature names for alignment at inference

Usage:
    python dl_train_final_model.py
    python dl_train_final_model.py --epochs 100
"""

import numpy as np
import os
import argparse
import torch
import joblib
import laspy

from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

from utils import (SEED, load_city, sample_and_filter, build_segments,
                   add_context_features, get_common_features,
                   print_covariate_stats, SKIP_FIELDS,
                   MODEL_SIDEWALK, MODEL_STREET, MODEL_OTHER)
from dl_classifier import train_mlp_binary, predict_mlp_binary

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── City detection ────────────────────────────────────────────────────────

def detect_labelled_cities(preprocessed_dir="preprocessed", threshold=0.01):
    """Auto-detect cities with both sidewalk AND street labels > threshold."""
    labelled, skipped = [], []
    if not os.path.exists(preprocessed_dir):
        print(f"  [WARNING] preprocessed/ not found")
        return []
    cities = sorted([d for d in os.listdir(preprocessed_dir)
                     if os.path.isdir(os.path.join(preprocessed_dir, d))])
    print(f"\nScanning preprocessed/ for labelled cities...")
    for city in cities:
        laz_path = os.path.join(preprocessed_dir, city, "low_featured.laz")
        if not os.path.exists(laz_path):
            skipped.append(city)
            continue
        try:
            las     = laspy.read(laz_path)
            labels  = np.array(las.classification, dtype=np.int32)
            sw_frac = (labels == 2).sum()  / len(labels)
            st_frac = (labels == 11).sum() / len(labels)
            if sw_frac > threshold and st_frac > threshold:
                labelled.append(city)
                print(f"  {city:12} -> labelled   "
                      f"({100*sw_frac:.1f}% sidewalk, {100*st_frac:.1f}% street)")
            else:
                skipped.append(city)
                print(f"  {city:12} -> skipped    "
                      f"(sw={100*sw_frac:.2f}% st={100*st_frac:.2f}%)")
        except Exception as e:
            skipped.append(city)
            print(f"  {city:12} -> error ({e})")
    print(f"\nLabelled cities found: {labelled}")
    return labelled


# ── CSF helper ────────────────────────────────────────────────────────────

def run_csf(xyz):
    """
    Apply CSF to xyz array. Returns boolean ground mask.
    Parameters tuned for urban street-level TLS scans.
    """
    import CSF
    csf = CSF.CSF()
    csf.params.bSloopSmooth     = True
    csf.params.cloth_resolution = 0.5
    csf.params.rigidness        = 2
    csf.params.time_step        = 0.65
    csf.params.class_threshold  = 0.3
    csf.params.interations      = 500
    csf.setPointCloud(xyz.tolist())
    gi, ngi = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(gi, ngi, exportCloth=False)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[list(gi)] = True
    return mask


# ── Per-city processing ───────────────────────────────────────────────────

def process_city_for_training(city, common_feat_names, csf_available):
    """
    Single clean pass for one city:
      load → select common features → CSF on full cloud → sample from
      ground-only → roughness filter → build segments → context features
      → keep only sidewalk/street segments for binary training

    CSF runs on the FULL cloud before sampling so our 650k sample budget
    is spent entirely on ground points (sidewalk + street), not wasted on
    buildings and trees that would be discarded anyway.

    Returns dict with X (binary features), y (binary labels: 0=street 1=sidewalk),
    seg_xyz, xyz_s, unique_voxels, inverse_idx.
    Returns None if city has too few ground segments.
    """
    print(f"\n{'─'*50}")
    print(f"Processing {city}...")

    # ── Load ──────────────────────────────────────────────────────────────
    xyz_raw, labels_raw, features_raw, feat_names = load_city(city)

    # Select common features by name (not position) — handles Bologna's 46 vs 41
    common_idx      = [feat_names.index(f) for f in common_feat_names
                       if f in feat_names]
    features_raw    = features_raw[:, common_idx]
    feat_names_used = [feat_names[i] for i in common_idx]
    print(f"  Features: {len(feat_names_used)} common "
          f"(dropped {len(feat_names) - len(feat_names_used)} city-specific)")

    # ── CSF on full cloud ─────────────────────────────────────────────────
    # Run before sampling so sample budget goes to ground points only
    if csf_available:
        print(f"  Running CSF on full cloud ({len(xyz_raw):,} points)...")
        ground_mask = run_csf(xyz_raw)
        n_removed   = (~ground_mask).sum()
        print(f"  CSF removed {n_removed:,} non-ground points "
              f"({100*n_removed/len(xyz_raw):.1f}%) "
              f"-> {ground_mask.sum():,} ground points remain")
        xyz_raw      = xyz_raw[ground_mask]
        labels_raw   = labels_raw[ground_mask]
        features_raw = features_raw[ground_mask]
    else:
        print(f"  [WARNING] CSF not available — using all points")

    # ── Sample from ground-only points + roughness filter ─────────────────
    xyz_s, labels_s, features_s = sample_and_filter(
        xyz_raw, labels_raw, features_raw, feat_names_used, training=True)

    # ── Build segments + context features ─────────────────────────────────
    seg_xyz, seg_feats, seg_labels, unique_voxels, inverse_idx, seg_voxel_ids = build_segments(
        xyz_s, labels_s, features_s)
    X, _ = add_context_features(seg_xyz, seg_feats)

    # ── Keep only sidewalk + street segments for binary training ──────────
    # Other-class segments (kerb edges, drain grates etc. that survived CSF)
    # are discarded — binary model only needs to learn sidewalk vs street
    ground_seg_mask = np.isin(seg_labels, [MODEL_SIDEWALK, MODEL_STREET])
    n_discarded     = (~ground_seg_mask).sum()
    X          = X[ground_seg_mask]
    seg_labels = seg_labels[ground_seg_mask]
    seg_xyz    = seg_xyz[ground_seg_mask]
    print(f"  Discarded {n_discarded:,} non-ground segments after CSF "
          f"-> {len(X):,} sidewalk/street segments for binary training")

    if len(X) < 200:
        print(f"  [WARNING] Too few ground segments for {city} — skipping")
        return None

    # ── Remap to binary labels: street=0, sidewalk=1 ──────────────────────
    # MODEL_STREET=2, MODEL_SIDEWALK=1 in 3-class. Binary: street->0, sidewalk->1
    y_binary = (seg_labels == MODEL_SIDEWALK).astype(np.int64)

    sw_count = y_binary.sum()
    st_count = (y_binary == 0).sum()
    print(f"  Binary labels — sidewalk: {sw_count:,} "
          f"({100*sw_count/len(y_binary):.1f}%) | "
          f"street: {st_count:,} ({100*st_count/len(y_binary):.1f}%)")

    return {
        "X"            : X,
        "y"            : y_binary,
        "seg_xyz"      : seg_xyz,
        "xyz_s"        : xyz_s,
        "unique_voxels": unique_voxels,
        "inverse_idx"  : inverse_idx,
    }


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("FINAL BINARY MLP TRAINING")
    print("Pipeline: CSF → ground-only sample → binary sidewalk/street MLP")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}")

    # ── Detect labelled cities ────────────────────────────────────────────
    LABELLED_CITIES = detect_labelled_cities()
    if len(LABELLED_CITIES) < 2:
        print(f"\n[ERROR] Need at least 2 labelled cities. Found: {LABELLED_CITIES}")
        exit(1)
    print(f"\nTraining on: {LABELLED_CITIES}")

    # ── Check CSF ────────────────────────────────────────────────────────
    try:
        import CSF
        CSF_AVAILABLE = True
        print("\nCSF available — will filter full cloud before sampling")
    except ImportError:
        CSF_AVAILABLE = False
        print("\n[WARNING] CSF not available — sampling from all points")
        print("  Install with: pip install cloth-simulation-filter")

    # ── Step 1: Common features across all cities ─────────────────────────
    # Align by feature NAME before anything else — handles Bologna (46 feats)
    # vs Riga/Vilnius/Warsaw (41 feats) without blind index truncation
    print("\nComputing common features across all labelled cities...")
    COMMON_FEAT_NAMES = get_common_features(LABELLED_CITIES)
    print(f"  Training with {len(COMMON_FEAT_NAMES)} common features "
          f"-> {len(COMMON_FEAT_NAMES) * 3} after context (own+mean+std)")

    # ── Step 2: Process each city — single clean pass ─────────────────────
    print("\nProcessing cities (CSF on full cloud, then sample ground-only)...")
    city_data = {}
    for city in LABELLED_CITIES:
        if not os.path.exists(f"preprocessed/{city}/low_featured.laz"):
            print(f"  Skipping {city} — file not found")
            continue
        result = process_city_for_training(city, COMMON_FEAT_NAMES, CSF_AVAILABLE)
        if result is not None:
            city_data[city] = result

    if len(city_data) < 2:
        print(f"\n[ERROR] Only {len(city_data)} cities processed. Need at least 2.")
        exit(1)

    # ── Step 3: Combine ───────────────────────────────────────────────────
    available = list(city_data.keys())
    X_all     = np.vstack([city_data[c]["X"] for c in available])
    y_all     = np.concatenate([city_data[c]["y"] for c in available])

    print(f"\n{'='*60}")
    print(f"Combined training data: {len(X_all):,} ground segments")
    sw = y_all.sum()
    st = (y_all == 0).sum()
    print(f"  Sidewalk: {sw:,} ({100*sw/len(y_all):.1f}%)")
    print(f"  Street:   {st:,} ({100*st/len(y_all):.1f}%)")
    print(f"  Features: {X_all.shape[1]}")

    # ── Step 4: Shuffle then 75/25 train/val split ────────────────────────
    # Shuffle so val set is a mix of all cities, not just the last one stacked
    rng      = np.random.RandomState(SEED)
    perm     = rng.permutation(len(X_all))
    X_all    = X_all[perm]
    y_all    = y_all[perm]

    val_size = int(len(X_all) * 0.25)
    X_val    = X_all[:val_size]
    y_val    = y_all[:val_size]
    X_tr     = X_all[val_size:]
    y_tr     = y_all[val_size:]

    total = len(X_all)
    print(f"\nTrain/Val split (shuffled — mixed cities in val):")
    print(f"  Train: {len(X_tr):,} ({100*len(X_tr)/total:.0f}%) "
          f"| Val: {len(X_val):,} ({100*len(X_val)/total:.0f}%)")

    # ── Step 5: Train binary MLP ──────────────────────────────────────────
    print(f"\nTraining binary MLP (sidewalk vs street)...")
    model, scaler, device, _, _ = train_mlp_binary(
        X_tr, y_tr, X_val, y_val, epochs=args.epochs)

    # ── Step 6: Covariate stats ───────────────────────────────────────────
    print("\nCovariate statistics per class (binary)...")
    feat_count = len(COMMON_FEAT_NAMES)
    print_covariate_stats(X_tr[:, :feat_count], y_tr, COMMON_FEAT_NAMES)

    # ── Step 7: Evaluation ────────────────────────────────────────────────
    for split_name, X_eval, y_eval in [("Train", X_tr, y_tr),
                                        ("Val",   X_val, y_val)]:
        preds   = predict_mlp_binary(model, scaler, X_eval, device)
        bal_acc = balanced_accuracy_score(y_eval, preds)
        sw_f1   = f1_score(y_eval, preds, pos_label=1, zero_division=0)
        print(f"\n{split_name} — Balanced Acc: {bal_acc*100:.1f}% | "
              f"Sidewalk F1: {sw_f1:.3f}")
        print(classification_report(y_eval, preds,
                                    target_names=["street", "sidewalk"],
                                    zero_division=0))

    # ── Step 8: Save ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_mlp_classifier.pt")
    joblib.dump(scaler,            "models/final_mlp_scaler.joblib")
    joblib.dump(COMMON_FEAT_NAMES, "models/common_feat_names.joblib")

    print("\n" + "="*60)
    print("Model saved:")
    print("  models/final_mlp_classifier.pt   — binary MLP weights")
    print("  models/final_mlp_scaler.joblib   — feature scaler")
    print("  models/common_feat_names.joblib  — feature names for inference")
    print("\nApply model with:")
    for city in list(city_data.keys()) + ["utrecht", "bologna"]:
        print(f"  python dl_apply_model.py --city {city}")


