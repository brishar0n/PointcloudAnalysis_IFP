"""
dl_apply_model.py — Apply trained binary MLP to classify a city.

Pipeline (both paths):
  CSF → sample ground-only → segments → binary MLP (sidewalk vs street)
  Non-ground points (removed by CSF) are labelled other=0 automatically.

Path A — Labelled city (has ground truth sidewalk + street labels):
  Apply model directly, evaluate against ground truth.

Path B — Unlabelled city (Utrecht etc.):
  Same pipeline + pseudo-label fine-tuning on high-confidence predictions
  + connected component filter to remove isolated noise patches.

Output:
  classified/{city}_mlp_classified.laz
  Labels: 0=other, 2=sidewalk (IFP), 11=street (IFP)
"""

import numpy as np
import laspy
import joblib
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

try:
    import CSF
    CSF_AVAILABLE = True
except ImportError:
    CSF_AVAILABLE = False
    print("[WARNING] CSF not installed. Run: pip install cloth-simulation-filter")

from utils import (
    SEED, SAMPLE_SIZE, VOXEL_SIZE, MIN_POINTS,
    load_city, sample_and_filter, build_segments, add_context_features,
    MODEL_SIDEWALK, MODEL_STREET, MODEL_OTHER
)
from dl_classifier import SidewalkStreetMLP, predict_mlp_binary, predict_proba_binary

np.random.seed(SEED)
torch.manual_seed(SEED)

LABEL_THRESHOLD   = 0.01
CSF_CLOTH_SIZE    = 0.5
CSF_MAX_ITER      = 500
CSF_CLASS_THRESH  = 0.3
MIN_COMPONENT_PTS = 50


# ── CSF ───────────────────────────────────────────────────────────────────

def csf_ground_filter(xyz, cloth_size=CSF_CLOTH_SIZE):
    """
    Apply CSF to xyz. Returns boolean ground mask.
    Non-ground points = other (code 0) in final output.
    """
    if not CSF_AVAILABLE:
        print("  [WARNING] CSF not available — all points treated as ground")
        return np.ones(len(xyz), dtype=bool)

    print(f"  Running CSF on {len(xyz):,} points...")
    csf = CSF.CSF()
    csf.params.bSloopSmooth     = True
    csf.params.cloth_resolution = cloth_size
    csf.params.rigidness        = 2
    csf.params.time_step        = 0.65
    csf.params.class_threshold  = CSF_CLASS_THRESH
    csf.params.interations      = CSF_MAX_ITER
    csf.setPointCloud(xyz.tolist())
    gi, ngi = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(gi, ngi, exportCloth=False)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[list(gi)] = True
    print(f"  CSF: {mask.sum():,} ground | {(~mask).sum():,} non-ground removed "
          f"({100*(~mask).sum()/len(xyz):.1f}%)")
    return mask


# ── Feature alignment ─────────────────────────────────────────────────────

def align_features_by_name(features, feat_names, common_feat_names):
    """
    Select and reorder features to match training feature set by name.
    Handles cities with different feature counts (e.g. Bologna 46 vs 41).
    Falls back to index truncation with a warning if names don't match.
    """
    # Check if feat_names are available and match
    available = set(feat_names)
    common    = [f for f in common_feat_names if f in available]

    if len(common) < len(common_feat_names) * 0.8:
        # Too many missing — fall back to truncation with warning
        n = len(common_feat_names)
        print(f"  [WARNING] Feature name mismatch. "
              f"Expected {len(common_feat_names)}, found {len(common)} common. "
              f"Falling back to index truncation to {n}.")
        if features.shape[1] >= n:
            return features[:, :n]
        else:
            pad = np.zeros((features.shape[0], n - features.shape[1]))
            return np.hstack([features, pad])

    idx      = [feat_names.index(f) for f in common]
    aligned  = features[:, idx]
    if len(common) < len(common_feat_names):
        missing = len(common_feat_names) - len(common)
        pad     = np.zeros((features.shape[0], missing))
        aligned = np.hstack([aligned, pad])
        print(f"  Feature alignment: {len(feat_names)} -> {aligned.shape[1]} "
              f"({missing} padded)")
    return aligned


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(model_path, scaler_path, feat_names_path, device):
    """
    Load binary MLP, scaler, and common feature names.
    Returns model, scaler, common_feat_names.
    """
    scaler            = joblib.load(scaler_path)
    common_feat_names = joblib.load(feat_names_path)
    expected_features = scaler.n_features_in_
    model             = SidewalkStreetMLP(input_dim=expected_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Model loaded: {expected_features} input features "
          f"({len(common_feat_names)} per-point × 3 context)")
    return model, scaler, common_feat_names


# ── Core processing ───────────────────────────────────────────────────────

def load_and_prepare(city, sample_size, common_feat_names):
    """
    Load city → align features → CSF on full cloud → sample from
    ground-only → build segments → context features.

    Returns:
        xyz_s            — sampled ground points (for saving .laz)
        labels_s         — ground truth labels (may be all-zero for unlabelled)
        X_seg            — segment feature matrix (ready for model)
        seg_xyz          — segment centroids
        unique_voxels    — for point projection
        inverse_idx      — for point projection
        csf_ground_mask  — which of the sampled points survived CSF
                           (non-ground points = other in final output)
    """
    # Load
    xyz_raw, labels_raw, features_raw, feat_names = load_city(city)

    # Align features by name to match training
    features_raw = align_features_by_name(
        features_raw, feat_names, common_feat_names)
    feat_names_aligned = common_feat_names  # for sample_and_filter roughness lookup

    # CSF on full cloud — non-ground becomes other=0 automatically
    csf_mask = csf_ground_filter(xyz_raw)
    xyz_raw      = xyz_raw[csf_mask]
    labels_raw   = labels_raw[csf_mask]
    features_raw = features_raw[csf_mask]

    # Sample from ground-only points + roughness filter
    xyz_s, labels_s, features_s = sample_and_filter(
        xyz_raw, labels_raw, features_raw,
        feat_names_aligned, sample_size, training=False)

    # Build segments + context features
    seg_xyz, seg_feats, seg_labels, unique_voxels, inverse_idx, seg_voxel_ids = build_segments(
        xyz_s, labels_s, features_s)

    # Sanity check — context features need enough neighbours
    from utils import N_NEIGHBORS
    if len(seg_xyz) <= N_NEIGHBORS:
        raise ValueError(
            f"Too few segments ({len(seg_xyz)}) for context features. "
            f"Need > {N_NEIGHBORS}. Check CSF parameters or sample size.")

    X_seg, _ = add_context_features(seg_xyz, seg_feats)
    return xyz_s, labels_s, X_seg, seg_xyz, unique_voxels, inverse_idx, seg_voxel_ids


# ── Point projection ──────────────────────────────────────────────────────

def project_to_points(binary_preds, unique_voxels, inverse_idx,
                      n_points, seg_voxel_ids):
    """
    Map binary segment predictions (0=street, 1=sidewalk) back to points.
    Uses seg_voxel_ids to correctly map each segment to its voxel,
    even after the purity filter has removed some segments.

    Without seg_voxel_ids, sequential iteration would assign prediction[i]
    to the i-th voxel, which is wrong when boundary segments were removed.

    Output uses IFP codes: 0=other, 2=sidewalk, 11=street.
    Points in voxels that were purity-filtered get label 0 (other).
    """
    point_preds = np.zeros(n_points, dtype=np.uint8)

    for seg_idx, voxel_idx in enumerate(seg_voxel_ids):
        mask = inverse_idx == voxel_idx
        if mask.sum() < MIN_POINTS:
            continue
        pred              = binary_preds[seg_idx]
        ifp_code          = 2 if pred == 1 else 11   # 1=sidewalk->IFP2, 0=street->IFP11
        point_preds[mask] = ifp_code

    return point_preds


# ── Label detection ───────────────────────────────────────────────────────

def has_ground_truth_labels(city):
    """Dynamically check if city has both sidewalk + street labels > 1%."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path     = os.path.join(base_dir, "preprocessed", city, "low_featured.laz")
    las      = laspy.read(path)
    labels   = np.array(las.classification, dtype=np.int32)
    sw_frac  = (labels == 2).sum()  / len(labels)
    st_frac  = (labels == 11).sum() / len(labels)
    print(f"  Label check: sidewalk={100*sw_frac:.2f}% street={100*st_frac:.2f}%",
          end="")
    has = sw_frac >= LABEL_THRESHOLD and st_frac >= LABEL_THRESHOLD
    print(f" -> {'Has ground truth' if has else 'Unlabelled'}")
    return has


# ── Pseudo-label fine-tuning (Path B) ────────────────────────────────────

def run_pseudo_label_finetune(model, scaler, X_seg, device,
                               confidence_threshold=0.95,
                               epochs=20, batch_size=256, lr=0.0001):
    """
    Fine-tune binary model on high-confidence pseudo-labels from unlabelled city.

    Steps:
      1. Predict probabilities on all segments
      2. Keep segments with max_prob >= confidence_threshold
      3. Fine-tune model on those pseudo-labels
      4. Re-predict everything with updated model
    """
    # Step 1 — probabilities
    print(f"\n  Getting initial predictions...")
    all_probs   = predict_proba_binary(model, scaler, X_seg, device)
    predictions = all_probs.argmax(axis=1)
    max_probs   = all_probs.max(axis=1)

    # Step 2 — select confident pseudo-labels
    confident_mask = max_probs >= confidence_threshold
    n_confident    = confident_mask.sum()
    n_total        = len(predictions)
    print(f"  Pseudo-label selection: {n_confident:,}/{n_total:,} "
          f"confident (≥{confidence_threshold*100:.0f}%)")

    if n_confident < 100:
        print(f"  [WARNING] Too few confident predictions. "
              f"Try lowering --confidence. Skipping fine-tuning.")
        return model, predictions

    pseudo_labels = predictions[confident_mask]
    X_pseudo      = X_seg[confident_mask]

    names = {0: "street", 1: "sidewalk"}
    unique, counts = np.unique(pseudo_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {names[cls]:10s}: {count:,} ({100*count/n_confident:.1f}%)")

    if len(unique) < 2:
        print(f"  [WARNING] Only one class in pseudo-labels — skipping fine-tuning.")
        return model, predictions

    # Step 3 — fine-tune
    print(f"\n  Fine-tuning on {n_confident:,} pseudo-labelled segments "
          f"({epochs} epochs)...")
    X_sc = scaler.transform(X_pseudo).astype(np.float32)

    classes, c_counts = np.unique(pseudo_labels, return_counts=True)
    weights    = 1.0 / c_counts
    weights    = weights / weights.sum() * len(classes)
    weight_vec = np.ones(2, dtype=np.float32)
    for c, w in zip(classes, weights):
        weight_vec[c] = w
    class_weights = torch.FloatTensor(weight_vec).to(device)

    X_t      = torch.FloatTensor(X_sc).to(device)
    y_t      = torch.LongTensor(pseudo_labels).to(device)
    loader   = DataLoader(TensorDataset(X_t, y_t),
                          batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f}")
    print(f"  Fine-tuning complete.")

    # Step 4 — re-predict
    model.eval()
    new_preds = predict_mlp_binary(model, scaler, X_seg, device)
    orig = dict(zip(*np.unique(predictions,  return_counts=True)))
    new  = dict(zip(*np.unique(new_preds,    return_counts=True)))
    print(f"\n  Before vs after fine-tuning:")
    print(f"  {'Class':10} {'Before':>10} {'After':>10}")
    for cls in [0, 1]:
        print(f"  {names[cls]:10} "
              f"{orig.get(cls, 0):>10,} {new.get(cls, 0):>10,}")
    return model, new_preds


# ── Connected component filter ────────────────────────────────────────────

def remove_isolated_patches(xyz_s, point_preds, min_points=MIN_COMPONENT_PTS):
    """
    Remove isolated sidewalk patches too small to be real sidewalks.
    Reclassifies small isolated patches as other (IFP 0).
    """
    cleaned       = point_preds.copy()
    sidewalk_mask = point_preds == 2   # IFP code 2 = sidewalk
    n_before      = sidewalk_mask.sum()

    if n_before == 0:
        print("  No sidewalk points to filter.")
        return cleaned

    sw_xyz    = xyz_s[sidewalk_mask]
    sw_idx    = np.where(sidewalk_mask)[0]
    radius    = VOXEL_SIZE * 3
    tree      = KDTree(sw_xyz[:, :2])
    visited   = np.zeros(len(sw_xyz), dtype=bool)
    components = []

    for i in range(len(sw_xyz)):
        if visited[i]:
            continue
        comp  = [i]
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            for nb in tree.query_radius([sw_xyz[curr, :2]], r=radius)[0]:
                if not visited[nb]:
                    visited[nb] = True
                    comp.append(nb)
                    queue.append(nb)
        components.append(comp)

    removed = 0
    for comp in components:
        if len(comp) < min_points:
            for idx in comp:
                cleaned[sw_idx[idx]] = 0
                removed += 1

    print(f"\n  Connected component filter:")
    print(f"    Sidewalk before: {n_before:,}")
    print(f"    Noise removed:   {removed:,}")
    print(f"    Sidewalk after:  {n_before - removed:,}")
    return cleaned


# ── Output helpers ────────────────────────────────────────────────────────

def print_distribution(point_preds, title="Label distribution"):
    names_map        = {0: "other", 2: "sidewalk", 11: "street"}
    unique_c, cnts_c = np.unique(point_preds, return_counts=True)
    total            = len(point_preds)
    print(f"\n{title}:")
    for cls, count in zip(unique_c, cnts_c):
        print(f"  {cls:2d} ({names_map.get(int(cls),'other'):10s}): "
              f"{count:,} ({100*count/total:.1f}%)")


def save_laz(city, xyz_s, point_preds):
    os.makedirs("classified", exist_ok=True)
    out_path               = f"classified/{city}_mlp_classified.laz"
    header                 = laspy.LasHeader(point_format=0, version="1.2")
    las_out                = laspy.LasData(header)
    las_out.x              = xyz_s[:, 0]
    las_out.y              = xyz_s[:, 1]
    las_out.z              = xyz_s[:, 2]
    las_out.classification = point_preds
    las_out.write(out_path)
    print(f"\n[OK] Saved: {out_path}")
    print(f"     Labels: 0=other  2=sidewalk  11=street")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city",            required=True)
    parser.add_argument("--sample",          type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--confidence",      type=float, default=0.95)
    parser.add_argument("--finetune-epochs", type=int,   default=20)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MLP Apply — {args.city.upper()}")
    print(f"Pipeline: CSF → ground-only sample → binary MLP")
    print(f"{'='*60}")

    # ── Check model files ─────────────────────────────────────────────────
    model_path      = "models/final_mlp_classifier.pt"
    scaler_path     = "models/final_mlp_scaler.joblib"
    feat_names_path = "models/common_feat_names.joblib"

    for path in [model_path, scaler_path, feat_names_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            print("  Run: python dl_train_final_model.py")
            exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model, scaler, common_feat_names = load_model(
        model_path, scaler_path, feat_names_path, device)

    # ── Detect labels ─────────────────────────────────────────────────────
    print(f"\nChecking labels for {args.city}...")
    labelled = has_ground_truth_labels(args.city)

    # ── Load + prepare (same pipeline for both paths) ─────────────────────
    print(f"\nLoading and preparing {args.city}...")
    xyz_s, labels_s, X_seg, seg_xyz, unique_voxels, inverse_idx, seg_voxel_ids = \
        load_and_prepare(args.city, args.sample, common_feat_names)

    # ── Path A: labelled — predict + evaluate ─────────────────────────────
    if labelled:
        print(f"\n[Path A] Labelled city — predicting and evaluating...")
        binary_preds = predict_mlp_binary(model, scaler, X_seg, device)
        point_preds  = project_to_points(
            binary_preds, unique_voxels, inverse_idx, len(xyz_s), seg_voxel_ids)

        # Evaluate against ground truth (ground segments only)
        from utils import remap_labels
        from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
        y_true_3class = remap_labels(labels_s)
        # Only evaluate on sidewalk/street points (binary problem)
        ground_mask  = np.isin(point_preds, [2, 11])
        if ground_mask.sum() > 0:
            # Map IFP codes back to binary for eval
            pred_binary = (point_preds[ground_mask] == 2).astype(int)
            # Ground truth binary (1=sidewalk, 0=street) for same points
            true_binary = (y_true_3class[ground_mask] == MODEL_SIDEWALK).astype(int)
            bal_acc = balanced_accuracy_score(true_binary, pred_binary)
            sw_f1   = f1_score(true_binary, pred_binary, pos_label=1,
                                zero_division=0)
            print(f"\nEvaluation (ground segments only):")
            print(f"  Balanced Acc: {bal_acc*100:.1f}% | Sidewalk F1: {sw_f1:.3f}")
            print(classification_report(true_binary, pred_binary,
                                        target_names=["street", "sidewalk"],
                                        zero_division=0))

    # ── Path B: unlabelled — fine-tune + cleanup ──────────────────────────
    else:
        print(f"\n[Path B] Unlabelled city — pseudo-label fine-tuning + cleanup")

        model, binary_preds = run_pseudo_label_finetune(
            model, scaler, X_seg, device,
            confidence_threshold=args.confidence,
            epochs=args.finetune_epochs)

        point_preds = project_to_points(
            binary_preds, unique_voxels, inverse_idx, len(xyz_s), seg_voxel_ids)

        print(f"\n── Connected Component Filter ──")
        point_preds = remove_isolated_patches(xyz_s, point_preds)

    # ── Save ──────────────────────────────────────────────────────────────
    print_distribution(point_preds, title="Final label distribution")
    save_laz(args.city, xyz_s, point_preds)
