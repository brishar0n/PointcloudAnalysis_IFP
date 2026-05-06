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
    print("[WARNING]  CSF not installed. Run: pip install cloth-simulation-filter")
    print("   Geometry filtering will be skipped.")

from utils import (
    SEED, SAMPLE_SIZE, VOXEL_SIZE, MIN_POINTS,
    load_city, sample_and_filter, build_segments,
    add_context_features
)
from dl_classifier import SidewalkMLP, predict_mlp

np.random.seed(SEED)
torch.manual_seed(SEED)

LABEL_THRESHOLD   = 0.01   # >1% non-zero -> city has ground truth labels
CSF_CLOTH_SIZE    = 0.5    # cloth resolution in metres
CSF_MAX_ITER      = 500    # cloth simulation iterations
CSF_CLASS_THRESH  = 0.3    # ground/non-ground distance threshold
MIN_COMPONENT_PTS = 50     # min sidewalk patch size to keep

def csf_ground_filter(xyz):
    """
    Apply Cloth Simulation Filter (CSF) — the blanket method.

    Drops a virtual cloth over the inverted point cloud.
    Points touching the cloth = ground (roads + sidewalks).
    Points above the cloth = non-ground (buildings, trees, cars) -> removed.

    Returns:
        ground_mask : boolean array, True = ground point
    """
    if not CSF_AVAILABLE:
        print("  [WARNING]  CSF not available — skipping geometry filter")
        return np.ones(len(xyz), dtype=bool)

    print("\n  Running CSF geometry filter (blanket method)...")
    csf = CSF.CSF()

    # Parameters tuned for urban street-level TLS scans
    csf.params.bSloopSmooth    = True   # smooth cloth on slopes
    csf.params.cloth_resolution = CSF_CLOTH_SIZE
    csf.params.rigidness       = 2      # 1=steep, 2=normal, 3=flat
    csf.params.time_step       = 0.65
    csf.params.class_threshold = CSF_CLASS_THRESH
    csf.params.interations     = CSF_MAX_ITER

    csf.setPointCloud(xyz.tolist())
    ground_idx     = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx, exportCloth=False)

    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[list(ground_idx)] = True

    n_ground  = ground_mask.sum()
    n_removed = (~ground_mask).sum()
    print(f"  CSF: {n_ground:,} ground points kept | "
          f"{n_removed:,} non-ground removed "
          f"({100*n_removed/len(xyz):.1f}%)")
    return ground_mask


# ── Connected Component Filter ────────────────────────────────────────────

def remove_isolated_patches(xyz_s, point_preds, min_points=MIN_COMPONENT_PTS):
    """
    Remove isolated sidewalk patches too small to be real sidewalks.

    Real sidewalks are large continuous surfaces.
    Small isolated sidewalk patches are noise — reclassify as other.

    Returns:
        cleaned_preds : point predictions with noise removed
    """
    from utils import VOXEL_SIZE as VS

    cleaned       = point_preds.copy()
    sidewalk_mask = point_preds == 2   # IFP code 2 = sidewalk
    n_before      = sidewalk_mask.sum()

    if n_before == 0:
        print("  No sidewalk points to filter.")
        return cleaned

    sw_xyz  = xyz_s[sidewalk_mask]
    sw_idx  = np.where(sidewalk_mask)[0]
    radius  = VS * 3   # connect points within 3 voxel widths

    tree    = KDTree(sw_xyz[:, :2])
    visited = np.zeros(len(sw_xyz), dtype=bool)

    components = []
    for i in range(len(sw_xyz)):
        if visited[i]:
            continue
        component  = [i]
        queue      = [i]
        visited[i] = True
        while queue:
            curr      = queue.pop(0)
            neighbors = tree.query_radius([sw_xyz[curr, :2]], r=radius)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    component.append(nb)
                    queue.append(nb)
        components.append(component)

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


# ── Label detection ───────────────────────────────────────────────────────

def has_ground_truth_labels(city):
    """
    Check if a city has usable ground truth labels.
    Only counts IFP sidewalk (code 2) and street (code 11) as real labels.
    Other non-zero codes (vegetation, buildings etc.) are ignored.
    Returns True if more than 1% of points have sidewalk or street labels.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path     = os.path.join(base_dir, "preprocessed", city, "low_featured.laz")
    las      = laspy.read(path)
    labels   = np.array(las.classification, dtype=np.int32)
    frac     = (np.isin(labels, [2, 11])).sum() / len(labels)
    print(f"  Label check: {100*frac:.2f}% sidewalk/street labels", end="")
    if frac > LABEL_THRESHOLD:
        print(f" -> Has ground truth labels")
        return True
    else:
        print(f" -> No ground truth labels (unlabelled city)")
        return False


# ── Core apply function ───────────────────────────────────────────────────

def load_model(model_path, scaler_path, device):
    """Load final MLP model and scaler from disk."""
    scaler   = joblib.load(scaler_path)
    expected = scaler.n_features_in_
    model    = SidewalkMLP(input_dim=expected)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, scaler, expected


def process_city_data(city, sample_size, expected, csf_mask=None):
    """
    Load, sample, segment and build features for a city.
    If csf_mask is provided, only ground points (from CSF) are used.

    Feature alignment happens BEFORE context features are added.
    expected is the number of per-point features the model was trained on.
    After context (mean + std of neighbours), final feature count = expected * 3.
    """
    xyz, labels, features, feat_names         = load_city(city)
    xyz_s, labels_s, features_s               = sample_and_filter(
                                                    xyz, labels, features,
                                                    feat_names, sample_size)

    # Apply CSF ground mask if provided
    if csf_mask is not None:
        n_before   = len(xyz_s)
        xyz_s      = xyz_s[csf_mask]
        labels_s   = labels_s[csf_mask]
        features_s = features_s[csf_mask]
        print(f"  CSF mask applied: {n_before:,} -> {len(xyz_s):,} points")

    # ── Align per-point features BEFORE segmentation ──────────────────────
    # expected * 3 = final segment feature count (own + ctx_mean + ctx_std)
    # So per-point features should match expected // 3
    n_point_feats = expected // 3
    if features_s.shape[1] > n_point_feats:
        print(f"  Feature alignment: {features_s.shape[1]} -> {n_point_feats} "
              f"per-point features (trimmed)")
        features_s = features_s[:, :n_point_feats]
    elif features_s.shape[1] < n_point_feats:
        pad        = np.zeros((features_s.shape[0],
                               n_point_feats - features_s.shape[1]))
        features_s = np.hstack([features_s, pad])
        print(f"  Feature alignment: padded to {n_point_feats} per-point features")

    seg_xyz, seg_feats, seg_labels, \
        unique_voxels, inverse_idx             = build_segments(
                                                    xyz_s, labels_s, features_s)
    X_seg, _                                   = add_context_features(
                                                    seg_xyz, seg_feats)

    # Final check — should match exactly now
    if X_seg.shape[1] != expected:
        pad   = np.zeros((X_seg.shape[0], expected - X_seg.shape[1]))
        X_seg = np.hstack([X_seg, pad])

    return xyz_s, labels_s, X_seg, seg_xyz, unique_voxels, inverse_idx


def project_to_points(predictions, unique_voxels, inverse_idx, n_points):
    """Map segment-level predictions back to individual points."""
    point_preds = np.zeros(n_points, dtype=np.uint8)
    seg_counter = 0
    for i in range(len(unique_voxels)):
        mask = inverse_idx == i
        if mask.sum() < MIN_POINTS:
            continue
        pred              = predictions[seg_counter]
        ifp_code          = 2 if pred == 1 else (11 if pred == 2 else 0)
        point_preds[mask] = ifp_code
        seg_counter      += 1
    return point_preds


def print_distribution(point_preds, title="Label distribution"):
    """Print class distribution of point predictions."""
    names_map        = {0: "other", 2: "sidewalk", 11: "street"}
    unique_c, cnts_c = np.unique(point_preds, return_counts=True)
    print(f"\n{title}:")
    for cls, count in zip(unique_c, cnts_c):
        print(f"  {cls:2d} ({names_map.get(int(cls), 'other'):10s}): "
              f"{count:,} ({100*count/len(point_preds):.1f}%)")


def save_laz(city, xyz_s, point_preds):
    """Save classified point cloud to classified/{city}_mlp_classified.laz."""
    os.makedirs("classified", exist_ok=True)
    out_path               = f"classified/{city}_mlp_classified.laz"
    header                 = laspy.LasHeader(point_format=0, version="1.2")
    las_out                = laspy.LasData(header)
    las_out.x              = xyz_s[:, 0]
    las_out.y              = xyz_s[:, 1]
    las_out.z              = xyz_s[:, 2]
    las_out.classification = point_preds
    las_out.write(out_path)
    print(f"\n[OK] Saved {out_path}")
    print(f"   Labels: 0=other, 2=sidewalk, 11=street")


# ── Pseudo-label fine-tuning ──────────────────────────────────────────────

def run_pseudo_label_finetune(model, scaler, X_seg, device,
                               confidence_threshold, epochs,
                               batch_size=256, lr=0.0001):
    """
    Fine-tune model on high-confidence pseudo-labels from unlabelled city.

    Steps:
        1. Get predictions + probabilities from current model
        2. Keep only segments where confidence >= threshold
        3. Fine-tune model on those pseudo-labels
        4. Return updated model + new predictions for all segments
    """

    # ── Step 1: Get initial predictions with probabilities ────────────────
    print(f"\n  Getting initial predictions...")
    model.eval()
    X_scaled = scaler.transform(X_seg).astype(np.float32)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 512):
            batch  = X_tensor[i:i+512]
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs   = np.vstack(all_probs)
    predictions = all_probs.argmax(axis=1)
    max_probs   = all_probs.max(axis=1)

    # ── Step 2: Select high-confidence pseudo-labels ──────────────────────
    confident_mask = max_probs >= confidence_threshold
    n_confident    = confident_mask.sum()
    n_total        = len(predictions)

    print(f"\n  Pseudo-label selection:")
    print(f"    Total segments:     {n_total:,}")
    print(f"    Confident (≥{confidence_threshold*100:.0f}%): "
          f"{n_confident:,} ({100*n_confident/n_total:.1f}%)")
    print(f"    Discarded:          {n_total - n_confident:,} (too uncertain)")

    if n_confident < 100:
        print(f"\n  [WARNING]  Too few confident predictions ({n_confident}).")
        print(f"     Try lowering --confidence. Skipping fine-tuning.")
        return model, predictions

    pseudo_labels = predictions[confident_mask]
    X_pseudo      = X_seg[confident_mask]

    unique, counts = np.unique(pseudo_labels, return_counts=True)
    names          = {0: "other", 1: "sidewalk", 2: "street"}
    print(f"\n  Pseudo-label class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    {cls} ({names[cls]:10s}): {count:,} "
              f"({100*count/n_confident:.1f}%)")

    if len(unique) < 2:
        print(f"\n  [WARNING]  Only one class in pseudo-labels — skipping fine-tuning.")
        return model, predictions

    # ── Step 3: Fine-tune on pseudo-labels ────────────────────────────────
    print(f"\n  Fine-tuning on {n_confident:,} pseudo-labelled segments...")
    print(f"  Epochs: {epochs} | LR: {lr}")

    X_pseudo_scaled = scaler.transform(X_pseudo).astype(np.float32)

    # Class weights
    classes, c_counts = np.unique(pseudo_labels, return_counts=True)
    weights    = 1.0 / c_counts
    weights    = weights / weights.sum() * len(classes)
    weight_vec = np.ones(3, dtype=np.float32)
    for c, w in zip(classes, weights):
        weight_vec[c] = w
    class_weights = torch.FloatTensor(weight_vec).to(device)

    X_t      = torch.FloatTensor(X_pseudo_scaled).to(device)
    y_t      = torch.LongTensor(pseudo_labels).to(device)
    dataset  = TensorDataset(X_t, y_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f}")

    print(f"  Fine-tuning complete!")

    # ── Step 4: Re-predict with fine-tuned model ──────────────────────────
    print(f"\n  Re-predicting with fine-tuned model...")
    new_predictions = predict_mlp(model, scaler, X_seg, device)

    # Show before vs after comparison
    orig_dict = dict(zip(*np.unique(predictions,     return_counts=True)))
    new_dict  = dict(zip(*np.unique(new_predictions, return_counts=True)))
    print(f"\n  Comparison before vs after fine-tuning:")
    print(f"  {'Class':12} {'Before':>10} {'After':>10}")
    print(f"  {'-'*35}")
    for cls in [0, 1, 2]:
        print(f"  {names[cls]:12} "
              f"{orig_dict.get(cls, 0):>10,} "
              f"{new_dict.get(cls, 0):>10,}")

    return model, new_predictions


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city",             required=True)
    parser.add_argument("--sample",           type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--confidence",       type=float, default=0.95,
                        help="Confidence threshold for pseudo-labels (default 0.95)")
    parser.add_argument("--finetune-epochs",  type=int,   default=20,
                        help="Fine-tuning epochs for unlabelled cities (default 20)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MLP Apply Model — {args.city.upper()}")
    print(f"{'='*60}")

    # ── Check model exists ────────────────────────────────────────────────
    model_path  = "models/final_mlp_classifier.pt"
    scaler_path = "models/final_mlp_scaler.joblib"

    if not os.path.exists(model_path):
        print("[ERROR] Final MLP model not found!")
        print("   Run: python run_pipeline.py --train")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Detect whether city has ground truth labels ───────────────────────
    print(f"\nChecking labels for {args.city}...")
    labelled = has_ground_truth_labels(args.city)

    # ── Load model ────────────────────────────────────────────────────────
    model, scaler, expected = load_model(model_path, scaler_path, device)

    # ── Path A: Labelled city — apply directly ────────────────────────────
    if labelled:
        print(f"\n[Path A] Labelled city — applying model directly...")
        xyz_s, labels_s, X_seg, \
            seg_xyz, unique_voxels, inverse_idx = process_city_data(
                                                    args.city, args.sample, expected)
        predictions = predict_mlp(model, scaler, X_seg, device)
        point_preds = project_to_points(
            predictions, unique_voxels, inverse_idx, len(xyz_s))

    # ── Path B: Unlabelled city — CSF -> MLP -> fine-tune -> cleanup ────────
    else:
        print(f"\n[Path B] Unlabelled city — 3-step pipeline:")
        print(f"  Step 1: CSF geometry filter (blanket)")
        print(f"  Step 2: MLP classification + pseudo-label fine-tuning")
        print(f"  Step 3: Connected component noise removal")

        # Load city ONCE — share same sampled data across all steps
        from utils import load_city as _load_city, sample_and_filter as _saf
        xyz_raw, labels_raw, features_raw, feat_names = _load_city(args.city)
        xyz_sampled, labels_sampled, features_sampled = _saf(
            xyz_raw, labels_raw, features_raw, feat_names, args.sample)

        # Step 1 — CSF on the already-sampled points
        print(f"\n── Step 1: CSF Geometry Filter ──")
        csf_mask = csf_ground_filter(xyz_sampled)

        # Apply CSF mask to sampled data
        xyz_s      = xyz_sampled[csf_mask]
        labels_s   = labels_sampled[csf_mask]
        features_s = features_sampled[csf_mask]
        print(f"  Ground points after CSF: {len(xyz_s):,}")

        # Step 2 — Build segments and run MLP + fine-tune on CSF-filtered points
        print(f"\n-- Step 2: MLP Classification + Fine-Tuning --")
        from utils import build_segments as _bs, add_context_features as _acf

        # Align per-point features BEFORE segmentation
        n_point_feats = expected // 3
        if features_s.shape[1] > n_point_feats:
            print(f"  Feature alignment: {features_s.shape[1]} -> {n_point_feats} "
                  f"per-point features (trimmed)")
            features_s = features_s[:, :n_point_feats]
        elif features_s.shape[1] < n_point_feats:
            pad        = np.zeros((features_s.shape[0],
                                   n_point_feats - features_s.shape[1]))
            features_s = np.hstack([features_s, pad])

        seg_xyz, seg_feats, seg_labels, \
            unique_voxels, inverse_idx = _bs(xyz_s, labels_s, features_s)
        X_seg, _                       = _acf(seg_xyz, seg_feats)

        # Final check
        if X_seg.shape[1] != expected:
            pad   = np.zeros((X_seg.shape[0], expected - X_seg.shape[1]))
            X_seg = np.hstack([X_seg, pad])

        model, predictions = run_pseudo_label_finetune(
            model, scaler, X_seg, device,
            confidence_threshold=args.confidence,
            epochs=args.finetune_epochs
        )
        point_preds = project_to_points(
            predictions, unique_voxels, inverse_idx, len(xyz_s))

        # Step 3 — Remove isolated noise patches
        print(f"\n── Step 3: Connected Component Filter ──")
        point_preds = remove_isolated_patches(xyz_s, point_preds)

    # ── Print distribution and save ───────────────────────────────────────
    print_distribution(point_preds, title="Final label distribution")
    save_laz(args.city, xyz_s, point_preds)
