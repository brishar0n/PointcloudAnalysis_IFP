"""
utils.py — Shared functions for the sidewalk classification pipeline.

Contains all core functions used by:
- segment_classifier.py (single city classification)
- loco_evaluation.py (multi-city LOCO evaluation)

Functions:
    load_city()              - Load low_featured.laz for a city
    remap_labels()           - Collapse IFP codes to 3 classes
    sample_and_filter()      - Sample points + roughness pre-filter
    build_segments()         - Build voxel superpoints
    add_context_features()   - Add neighbour context features
    apply_graph_smoothing()  - Fix isolated misclassifications
    process_city()           - Full pipeline for one city
    train_rf()               - Train Random Forest
    evaluate()               - Print metrics and return results
"""

import os
import numpy as np
import laspy
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             balanced_accuracy_score,
                             f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

# ── Constants ─────────────────────────────────────────────────────────────
SEED             = 42
SAMPLE_SIZE      = 650_000
VOXEL_SIZE       = 0.6      # metres — size of each superpoint square
MIN_POINTS       = 8        # minimum points per segment
N_NEIGHBORS      = 64       # context neighbours (optimised from 8,16,32,64,128)
SMOOTH_THRESHOLD = 0.6      # 60% neighbour agreement to override prediction
ROUGHNESS_THRESH = 0.05     # roughness filter threshold
PURITY_THRESH    = 0.85     # min % of points agreeing on label to keep segment
OTHER_RATIO      = 2.0      # max ratio of other:sidewalk segments (undersampling)

# IFP Class Mapping Constants
IFP_OTHER        = 0
IFP_SIDEWALK     = 2
IFP_STREET       = 11

MODEL_OTHER      = 0
MODEL_SIDEWALK   = 1
MODEL_STREET     = 2

SKIP_FIELDS = {
    'X','Y','Z','x','y','z','red','green','blue',
    'intensity','classification','gps_time',
    'return_number','number_of_returns',
    'scan_direction_flag','edge_of_flight_line',
    'scan_angle_rank','user_data','point_source_id',
    'synthetic','key_point','withheld'
}

label_names = {0: "other", 1: "sidewalk", 2: "street"}
np.random.seed(SEED)


# ── Functions ─────────────────────────────────────────────────────────────

def get_common_features(city_names, preprocessed_dir="preprocessed"):
    """
    Compute intersection of feature names across all cities.
    Returns features common to ALL cities in the order they appear
    in the city with fewest features — ensures consistent alignment
    across cities with different preprocessing (e.g. Bologna has 49 vs 41).
    """
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    city_feats = {}
    for city in city_names:
        path = os.path.join(base_dir, preprocessed_dir, city, "low_featured.laz")
        las  = laspy.read(path)
        feat_names = [n for n in las.point_format.dimension_names
                      if n not in SKIP_FIELDS]
        city_feats[city] = feat_names

    # Use city with fewest features as reference order
    ref_city   = min(city_feats, key=lambda c: len(city_feats[c]))
    ref_order  = city_feats[ref_city]
    common_set = set.intersection(*[set(f) for f in city_feats.values()])
    common     = [f for f in ref_order if f in common_set]
    print(f"  Common features across {city_names}: {len(common)} "
          f"(reference: {ref_city})")
    return common


def load_city(city_name):
    """
    Load low_featured.laz for a city.
    Returns xyz, labels, features, feature_names.

    We use low_featured.laz because:
    - Already split to street level by Bri's high/low filter
    - Has all precomputed features attached
    - Has XYZ coordinates needed for spatial segmentation
    """
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    path       = os.path.join(base_dir, "preprocessed", city_name, "low_featured.laz")
    print(f"  Loading {city_name} from {path}...")
    las        = laspy.read(path)
    xyz        = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    labels     = np.array(las.classification, dtype=np.int32)
    feat_names = [n for n in las.point_format.dimension_names
                  if n not in SKIP_FIELDS]
    features   = np.column_stack([np.array(getattr(las, n), dtype=np.float32)
                                   for n in feat_names])
    features   = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {len(xyz):,} points | {len(feat_names)} features")
    return xyz, labels, features, feat_names


def remap_labels(y):
    """
    Collapse IFP codes into 3 classes:
    0 = other
    1 = sidewalk (was IFP code 2)
    2 = street   (was IFP code 11)
    """
    y_new = np.full_like(y, fill_value=MODEL_OTHER)
    y_new[y == IFP_SIDEWALK] = MODEL_SIDEWALK
    y_new[y == IFP_STREET]   = MODEL_STREET
    return y_new


def sample_and_filter(xyz, labels, features, feat_names,
                      sample_size=SAMPLE_SIZE, training=False):
    """
    Sample points and apply roughness pre-filter to remove non-ground objects
    like parked cars and bushes (roughness > 0.05m).

    training=True  : also apply other class undersampling (for training only)
    training=False : skip undersampling (for inference/apply)
    """
    idx        = (np.random.choice(len(xyz), sample_size, replace=False)
                  if len(xyz) > sample_size else np.arange(len(xyz)))
    xyz_s      = xyz[idx]
    labels_s   = remap_labels(labels[idx])
    features_s = features[idx]

    roughness_idx = next((i for i, f in enumerate(feat_names)
                          if "roughness_0.1" in str(f)), None)
    if roughness_idx is not None:
        mask       = features_s[:, roughness_idx] <= ROUGHNESS_THRESH
        removed    = (~mask).sum()
        xyz_s      = xyz_s[mask]
        labels_s   = labels_s[mask]
        features_s = features_s[mask]
        print(f"  Roughness filter removed {removed:,} points -> {len(xyz_s):,} remain")

    # Other class undersampling — only during training
    # Prevents model from defaulting to other due to class imbalance
    if training:
        minority_count = (labels_s != MODEL_OTHER).sum()
        other_count    = (labels_s == MODEL_OTHER).sum()
        max_other      = int(minority_count * OTHER_RATIO)
        if other_count > max_other and minority_count > 100:
            other_idx  = np.where(labels_s == MODEL_OTHER)[0]
            keep_idx   = np.random.choice(other_idx, max_other, replace=False)
            non_other  = np.where(labels_s != MODEL_OTHER)[0]
            selected   = np.sort(np.concatenate([keep_idx, non_other]))
            removed    = other_count - max_other
            xyz_s      = xyz_s[selected]
            labels_s   = labels_s[selected]
            features_s = features_s[selected]
            print(f"  Other undersampling removed {removed:,} other points "
                  f"-> {len(xyz_s):,} remain "
                  f"(ratio {OTHER_RATIO:.1f}x minority)")
        else:
            print(f"  Other undersampling skipped "
                  f"(minority classes already balanced)")

    return xyz_s, labels_s, features_s


def build_segments(xyz_s, labels_s, features_s, purity_thresh=PURITY_THRESH):
    """
    Build voxel superpoints by dividing the XY plane into VOXEL_SIZE squares.
    Assigns mean features and majority vote labels to each superpoint.

    Purity filter: segments where less than purity_thresh of points agree
    on the majority label are discarded — these are boundary voxels that
    mix sidewalk and road and confuse the model.
    """
    voxel_coords               = np.floor(xyz_s[:, :2] / VOXEL_SIZE).astype(np.int32)
    voxel_keys                 = voxel_coords[:, 0] * 1_000_000 + voxel_coords[:, 1]
    unique_voxels, inverse_idx = np.unique(voxel_keys, return_inverse=True)

    seg_xyz, seg_feats, seg_labels, seg_purity, seg_voxel_ids = [], [], [], [], []

    for i in range(len(unique_voxels)):
        mask = inverse_idx == i
        if mask.sum() < MIN_POINTS:
            continue
        pts   = xyz_s[mask]
        feats = features_s[mask]
        lbls  = labels_s[mask]
        unique_lbls, counts = np.unique(lbls, return_counts=True)
        majority_count = counts.max()
        purity         = majority_count / mask.sum()
        seg_xyz.append(pts.mean(axis=0))
        seg_feats.append(feats.mean(axis=0))
        seg_labels.append(unique_lbls[np.argmax(counts)])
        seg_purity.append(purity)
        seg_voxel_ids.append(i)  # track which unique_voxel index this segment maps to

    seg_xyz       = np.array(seg_xyz)
    seg_feats     = np.array(seg_feats)
    seg_labels    = np.array(seg_labels)
    seg_purity    = np.array(seg_purity)
    seg_voxel_ids = np.array(seg_voxel_ids, dtype=np.int32)

    # Apply purity filter — only for labelled segments (skip if all other)
    # seg_voxel_ids keeps track of which voxels survive so projection stays correct
    has_labels = np.isin(seg_labels, [1, 2]).any()
    if has_labels and purity_thresh > 0:
        pure_mask     = seg_purity >= purity_thresh
        removed       = (~pure_mask).sum()
        seg_xyz       = seg_xyz[pure_mask]
        seg_feats     = seg_feats[pure_mask]
        seg_labels    = seg_labels[pure_mask]
        seg_voxel_ids = seg_voxel_ids[pure_mask]
        print(f"  Segments: {len(seg_xyz):,} "
              f"(purity filter removed {removed:,} boundary segments "
              f"< {purity_thresh*100:.0f}% pure)")
    else:
        print(f"  Segments: {len(seg_xyz):,}")

    return seg_xyz, seg_feats, seg_labels, unique_voxels, inverse_idx, seg_voxel_ids


def add_context_features(seg_xyz, seg_feats):
    """
    Add spatial context features (mean and std) from N_NEIGHBORS nearest segments
    to help disambiguate similar local patches.
    """
    tree              = KDTree(seg_xyz[:, :2])
    _, indices        = tree.query(seg_xyz[:, :2], k=N_NEIGHBORS + 1)
    neighbour_indices = indices[:, 1:]

    ctx_mean = np.vstack([seg_feats[neighbour_indices[i]].mean(axis=0)
                          for i in range(len(seg_xyz))])
    ctx_std  = np.vstack([seg_feats[neighbour_indices[i]].std(axis=0)
                          for i in range(len(seg_xyz))])

    X = np.nan_to_num(np.hstack([seg_feats, ctx_mean, ctx_std]),
                      nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X.shape} (own + ctx_mean + ctx_std)")
    return X, neighbour_indices


def apply_graph_smoothing(y_pred, seg_xyz_subset, threshold=SMOOTH_THRESHOLD):
    """
    Fix isolated misclassifications using spatial graph smoothing.
    If a segment prediction disagrees with >= threshold of its neighbors,
    it is overridden with the neighbor majority vote.
    """
    tree              = KDTree(seg_xyz_subset[:, :2])
    _, indices        = tree.query(seg_xyz_subset[:, :2], k=N_NEIGHBORS + 1)
    neighbour_indices = indices[:, 1:]

    y_smoothed = y_pred.copy()
    changed    = 0
    for i in range(len(y_smoothed)):
        nbr        = y_pred[neighbour_indices[i]]
        vals, cnts = np.unique(nbr, return_counts=True)
        maj_pred   = vals[np.argmax(cnts)]
        maj_frac   = cnts.max() / len(nbr)
        if maj_pred != y_smoothed[i] and maj_frac >= threshold:
            y_smoothed[i] = maj_pred
            changed += 1
    print(f"  Graph smoothing: {changed:,} predictions changed "
          f"({100*changed/len(y_smoothed):.1f}%)")
    return y_smoothed


def process_city(city_name):
    """
    Full pipeline for one city — used by LOCO evaluation.
    Returns dict with X, y, seg_xyz, unique_voxels, inverse_idx.
    """
    print(f"\nProcessing {city_name}...")
    xyz, labels, features, feat_names             = load_city(city_name)
    xyz_s, labels_s, features_s                   = sample_and_filter(
                                                        xyz, labels, features, feat_names,
                                                        training=True)
    seg_xyz, seg_feats, seg_labels, \
        unique_voxels, inverse_idx, \
        seg_voxel_ids                              = build_segments(
                                                        xyz_s, labels_s, features_s)
    X, nbr_indices                                 = add_context_features(
                                                        seg_xyz, seg_feats)
    return {
        "city"         : city_name,
        "X"            : X,
        "y"            : seg_labels,
        "seg_xyz"      : seg_xyz,
        "nbr_indices"  : nbr_indices,
        "xyz_s"        : xyz_s,
        "unique_voxels": unique_voxels,
        "inverse_idx"  : inverse_idx,
        "seg_voxel_ids": seg_voxel_ids,
    }


def train_rf(X_train, y_train):
    """
    Train Random Forest with balanced_subsample class weights.
    balanced_subsample ensures each tree sees equal class counts.
    Fixes class imbalance during training.
    """
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    clf    = RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED)
    clf.fit(X_tr, y_train)
    return clf, scaler


def print_covariate_stats(X, y, feat_names):
    """
    Print mean and std of each feature per class.
    Answers James's question: what does the model think each class looks like?
    """
    names = {0: "other", 1: "sidewalk", 2: "street"}
    print(f"\n{'='*70}")
    print(f"  COVARIATE STATISTICS PER CLASS")
    print(f"{'='*70}")
    print(f"  {'Feature':<30} {'Other mean':>12} {'SW mean':>10} {'ST mean':>10}")
    print(f"  {'─'*64}")
    for i, feat in enumerate(feat_names):
        means = []
        for cls in [0, 1, 2]:
            mask = y == cls
            if mask.sum() > 0:
                means.append(X[mask, i].mean())
            else:
                means.append(float('nan'))
        print(f"  {feat:<30} {means[0]:>12.4f} {means[1]:>10.4f} {means[2]:>10.4f}")
    print(f"{'='*70}")


def evaluate(y_true, y_pred, title=""):
    """
    Print classification report and return metrics dict.
    Returns balanced_accuracy and sidewalk_f1.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sw_f1   = f1_score(y_true, y_pred, labels=[1],
                       average=None, zero_division=0)[0]
    if title:
        print(f"\n{title}")
    print(f"Balanced Accuracy: {bal_acc*100:.1f}%  |  Sidewalk F1: {sw_f1:.3f}")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2],
                                 target_names=["other", "sidewalk", "street"],
                                 zero_division=0))
    return {"bal_acc": bal_acc, "sidewalk_f1": sw_f1}
