"""
splitter.py — Splits point clouds for the classification pipeline.

Two kinds of splitting:
1. Spatial: separate "high" (buildings, trees) from "low" (streets, cars,
   sidewalks) based on height. This preprocessing step is described in the
   project chapters doc.
2. Train/test: prepare labelled data for the classifier (Person 2).

Usage:
    from preprocessing.splitter import split_high_low, prepare_training_data
    low_cloud, high_cloud = split_high_low(cloud)
    X_train, X_test, y_train, y_test = prepare_training_data(cloud)
"""

import numpy as np
from typing import Tuple


# ------------------------------------------------------------------ #
#  Classification labels used by IFP                                  #
#  (standard LAS classification + IFP custom labels)                  #
# ------------------------------------------------------------------ #
CLASS_LABELS = {
    0: "never_classified",
    1: "unclassified",
    2: "ground",          # includes street + sidewalk surfaces
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    7: "low_point",       # noise
    8: "reserved",
    9: "water",
    10: "rail",
    11: "road_surface",
    12: "reserved",
    13: "wire_guard",
    14: "wire_conductor",
    15: "transmission_tower",
    16: "wire_structure",
    17: "bridge_deck",
    18: "high_noise",
    # IFP may use custom values above 18 for sidewalk, kerb, etc.
    # These should be confirmed with Hendrik/James.
}

# For the binary "street vs sidewalk" classification task
SIDEWALK_CLASSES = {
    "street": 0,
    "sidewalk": 1,
    "other": 2,
}


def split_high_low(cloud: dict,
                   height_threshold: float = 2.0,
                   use_relative_height: bool = True) -> Tuple[dict, dict]:
    """
    Split cloud into "low" and "high" subsets.

    The project docs describe this as: buildings, tree canopies → high;
    streets, cars, obstacles → low.

    Parameters
    ----------
    cloud : dict from loader
    height_threshold : cutoff in metres
    use_relative_height : if True, use height_above_min (local relative
        height). If False, use raw Z. Relative height is better because
        streets on hills still count as "low".

    Returns
    -------
    (low_cloud, high_cloud) : two cloud dicts
    """
    if use_relative_height and "height_above_min" in cloud:
        height = cloud["height_above_min"]
    else:
        z = cloud["xyz"][:, 2]
        z_ground = np.percentile(z, 5)  # approximate ground level
        height = z - z_ground

    low_mask = height <= height_threshold
    high_mask = ~low_mask

    low_cloud = _apply_mask(cloud, low_mask)
    high_cloud = _apply_mask(cloud, high_mask)

    print(f"Split: {low_mask.sum():,} low points, "
          f"{high_mask.sum():,} high points "
          f"(threshold={height_threshold}m)")

    return low_cloud, high_cloud


def filter_by_classification(cloud: dict,
                             keep_classes: list) -> dict:
    """
    Keep only points with classification values in `keep_classes`.
    Useful for isolating ground, buildings, etc. from manually
    classified clouds.
    """
    mask = np.isin(cloud["classification"], keep_classes)
    filtered = _apply_mask(cloud, mask)
    print(f"Filtered: kept {mask.sum():,} / {len(mask):,} points "
          f"(classes {keep_classes})")
    return filtered


def remove_noise(cloud: dict, z_score_threshold: float = 3.0) -> dict:
    """
    Remove statistical outlier points (likely scan noise).
    Uses Z-score on local density: points with very few neighbours
    relative to the mean are removed.
    """
    if "density" not in cloud:
        raise ValueError("Run compute_density first (in features.py)")

    density = cloud["density"]
    mean_d = density.mean()
    std_d = density.std()

    # Keep points that are not extreme outliers
    mask = np.abs(density - mean_d) < (z_score_threshold * std_d)
    # Also remove any points with zero density (isolated)
    mask &= density > 0

    cleaned = _apply_mask(cloud, mask)
    removed = (~mask).sum()
    print(f"Noise removal: removed {removed:,} points "
          f"({100*removed/len(mask):.2f}%)")
    return cleaned


def prepare_training_data(cloud: dict,
                          feature_keys: list = None,
                          test_ratio: float = 0.2,
                          random_seed: int = 42) -> Tuple:
    """
    Prepare feature matrix X and label vector y for the classifier.

    Parameters
    ----------
    cloud : dict with computed features and classification labels
    feature_keys : list of field names to use as features.
        If None, auto-detects all numeric scalar fields.
    test_ratio : fraction of data for testing
    random_seed : for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Auto-detect feature columns
    if feature_keys is None:
        skip = {"xyz", "rgb", "classification", "gps_time", "header",
                "return_number", "num_returns", "intensity",
                "normal_x_0.1", "normal_y_0.1", "normal_z_0.1",
                "normal_x_0.3", "normal_y_0.3", "normal_z_0.3",
                "normal_x_0.5", "normal_y_0.5", "normal_z_0.5"}
        feature_keys = [
            k for k, v in cloud.items()
            if isinstance(v, np.ndarray) and v.ndim == 1
            and v.dtype in (np.float32, np.float64)
            and k not in skip
        ]
        # Also include normalised intensity and XYZ-derived
        if "intensity_normalized" in cloud:
            feature_keys.append("intensity_normalized")

    feature_keys = sorted(set(feature_keys))
    print(f"Using {len(feature_keys)} features: {feature_keys}")

    # Build feature matrix
    X = np.column_stack([cloud[k] for k in feature_keys])
    y = cloud["classification"]

    # Remove unlabelled points (class 0 = never classified)
    labelled_mask = y > 0
    X = X[labelled_mask]
    y = y[labelled_mask]
    print(f"Labelled points: {labelled_mask.sum():,} / {len(labelled_mask):,}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        name = CLASS_LABELS.get(cls, f"custom_{cls}")
        print(f"  {cls:3d} ({name:20s}): {count:>10,} "
              f"({100*count/len(y):.1f}%)")

    # Shuffle and split
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))

    X_train = X[indices[:split_idx]]
    X_test = X[indices[split_idx:]]
    y_train = y[indices[:split_idx]]
    y_test = y[indices[split_idx:]]

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, feature_keys


def prepare_training_blocks(cloud: dict,
                            block_size: float = 5.0,
                            feature_keys: list = None,
                            n_points_per_block: int = 4096,
                            test_ratio: float = 0.2,
                            random_seed: int = 42) -> Tuple:
    """
    Prepare data in spatial blocks for deep learning models like
    PointNet/RandLA-Net that expect fixed-size point subsets.

    Divides the XY plane into blocks of `block_size` metres and samples
    `n_points_per_block` from each. This is the standard approach for
    training point cloud DL models.

    Returns
    -------
    train_blocks, test_blocks : lists of (points, features, labels) tuples
    feature_names : list of feature column names
    """
    if feature_keys is None:
        skip = {"xyz", "rgb", "classification", "gps_time", "header",
                "return_number", "num_returns", "intensity"}
        feature_keys = sorted([
            k for k, v in cloud.items()
            if isinstance(v, np.ndarray) and v.ndim == 1
            and v.dtype in (np.float32, np.float64)
            and k not in skip
        ])

    xyz = cloud["xyz"]
    features = np.column_stack([cloud[k] for k in feature_keys])
    labels = cloud["classification"]

    # Divide into spatial blocks
    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)

    blocks = []
    x_start = xy_min[0]
    while x_start < xy_max[0]:
        y_start = xy_min[1]
        while y_start < xy_max[1]:
            mask = (
                (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_start + block_size) &
                (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_start + block_size)
            )
            n_in_block = mask.sum()
            if n_in_block < 100:
                y_start += block_size
                continue

            block_xyz = xyz[mask]
            block_feat = features[mask]
            block_labels = labels[mask]

            # Sample to fixed size
            if n_in_block >= n_points_per_block:
                idx = np.random.choice(n_in_block, n_points_per_block,
                                       replace=False)
            else:
                idx = np.random.choice(n_in_block, n_points_per_block,
                                       replace=True)

            # Normalize XYZ within block (center at origin)
            centered_xyz = block_xyz[idx] - block_xyz[idx].mean(axis=0)

            blocks.append((
                centered_xyz.astype(np.float32),
                block_feat[idx].astype(np.float32),
                block_labels[idx],
            ))

            y_start += block_size
        x_start += block_size

    # Shuffle and split blocks
    rng = np.random.default_rng(random_seed)
    rng.shuffle(blocks)
    split_idx = int(len(blocks) * (1 - test_ratio))

    train_blocks = blocks[:split_idx]
    test_blocks = blocks[split_idx:]

    print(f"Created {len(blocks)} blocks of {n_points_per_block} points each")
    print(f"  Block size: {block_size}m × {block_size}m")
    print(f"  Train: {len(train_blocks)} blocks, Test: {len(test_blocks)} blocks")

    return train_blocks, test_blocks, feature_keys


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _apply_mask(cloud: dict, mask: np.ndarray) -> dict:
    """Apply a boolean mask to all arrays in a cloud dict."""
    new_cloud = {}
    for key, val in cloud.items():
        if key == "header":
            new_cloud[key] = val.copy()
        elif isinstance(val, np.ndarray):
            new_cloud[key] = val[mask]
        else:
            new_cloud[key] = val
    if "header" in new_cloud:
        new_cloud["header"]["point_count"] = mask.sum()
    return new_cloud
