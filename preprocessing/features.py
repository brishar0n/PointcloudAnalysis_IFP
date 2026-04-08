"""
features.py — Geometric feature computation for point clouds.

Computes per-point features based on local neighbourhood geometry.
These become the input features for the sidewalk classifier (Person 2).

The key idea: for each point, find its neighbours within a sphere of
radius r, fit a covariance matrix, and decompose it into eigenvalues.
The ratios between eigenvalues tell you the local shape:
  - All similar      → spherical (e.g. foliage, noise)
  - One dominant     → linear (e.g. poles, wires, kerb edges)
  - Two dominant     → planar (e.g. ground, walls, sidewalk surface)

Usage:
    from preprocessing.features import compute_all_features
    cloud = load_point_cloud("bologna_subsampled.laz")
    cloud = compute_all_features(cloud, radii=[0.1, 0.3, 0.5])
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List


def _derive_from_existing_eigenvalues(cloud: dict, radius: float):
    """
    When eigenvalues are already precomputed (Riga, Utrecht, Vilnius),
    derive the secondary features (linearity, planarity, etc.) from them.
    """
    r = float(radius)
    λ1 = cloud.get(f"eigenvalue_1_{r}")
    λ2 = cloud.get(f"eigenvalue_2_{r}")
    λ3 = cloud.get(f"eigenvalue_3_{r}")

    if λ1 is None or λ2 is None or λ3 is None:
        return

    λ1_safe = np.maximum(λ1, 1e-10)

    derived = {
        f"linearity_{r}": ((λ1 - λ2) / λ1_safe).astype(np.float32),
        f"planarity_{r}": ((λ2 - λ3) / λ1_safe).astype(np.float32),
        f"scattering_{r}": (λ3 / λ1_safe).astype(np.float32),
        f"omnivariance_{r}": np.cbrt(λ1 * λ2 * λ3).astype(np.float32),
        f"anisotropy_{r}": ((λ1 - λ3) / λ1_safe).astype(np.float32),
    }

    added = []
    for key, val in derived.items():
        if key not in cloud:
            cloud[key] = val
            added.append(key)
    if added:
        print(f"  Derived from existing eigenvalues: {added}")


def compute_eigenvalues(xyz: np.ndarray, radius: float,
                        max_neighbours: int = 50) -> dict:
    """
    For each point, find neighbours within `radius`, compute covariance
    eigenvalues, and derive geometric descriptors.

    Parameters
    ----------
    xyz : (N, 3) array of point coordinates
    radius : search radius in metres
    max_neighbours : cap on neighbours to keep computation manageable

    Returns
    -------
    dict of (N,) arrays:
        "eigenvalue_1_{r}"  : largest eigenvalue (λ1)
        "eigenvalue_2_{r}"  : middle eigenvalue  (λ2)
        "eigenvalue_3_{r}"  : smallest eigenvalue (λ3)
        "linearity_{r}"     : (λ1 - λ2) / λ1
        "planarity_{r}"     : (λ2 - λ3) / λ1
        "scattering_{r}"    : λ3 / λ1
        "omnivariance_{r}"  : (λ1 * λ2 * λ3) ^ (1/3)
        "anisotropy_{r}"    : (λ1 - λ3) / λ1
        "verticality_{r}"   : 1 - |normal_z|  (0 = horizontal, 1 = vertical)
    where {r} is the radius label (e.g. "0.3")
    """
    n_points = len(xyz)
    r_label = f"{radius}"

    # Build KD-tree
    print(f"  Building KD-tree for radius {radius}m...")
    tree = KDTree(xyz)

    eigenvalues = np.zeros((n_points, 3), dtype=np.float64)
    normals = np.zeros((n_points, 3), dtype=np.float64)

    print(f"  Computing eigenvalues for {n_points:,} points...")
    # Process in chunks to show progress
    chunk_size = 50_000
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        if start % 200_000 == 0:
            print(f"    Progress: {start:,}/{n_points:,} "
                  f"({100*start/n_points:.0f}%)")

        batch_xyz = xyz[start:end]
        neighbours_list = tree.query_ball_point(batch_xyz, r=radius,
                                                workers=-1)

        for i, neighbours in enumerate(neighbours_list):
            idx = start + i
            if len(neighbours) < 3:
                # Not enough neighbours — leave as zeros
                continue

            # Cap neighbours for performance
            if len(neighbours) > max_neighbours:
                neighbours = neighbours[:max_neighbours]

            pts = xyz[neighbours]
            centroid = pts.mean(axis=0)
            centered = pts - centroid

            # Covariance matrix
            cov = (centered.T @ centered) / len(neighbours)

            # Eigendecomposition (returns ascending order)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Ensure non-negative
            eigvals = np.maximum(eigvals, 0.0)

            # Store in descending order: λ1 ≥ λ2 ≥ λ3
            eigenvalues[idx] = eigvals[::-1]

            # Normal is eigenvector of smallest eigenvalue
            # Orient normals upward (positive Z)
            normal = eigvecs[:, 0]  # smallest eigenvalue's eigenvector
            if normal[2] < 0:
                normal = -normal
            normals[idx] = normal

    # Derived features
    λ1 = eigenvalues[:, 0]
    λ2 = eigenvalues[:, 1]
    λ3 = eigenvalues[:, 2]

    # Avoid division by zero
    λ1_safe = np.maximum(λ1, 1e-10)

    features = {
        f"eigenvalue_1_{r_label}": λ1.astype(np.float32),
        f"eigenvalue_2_{r_label}": λ2.astype(np.float32),
        f"eigenvalue_3_{r_label}": λ3.astype(np.float32),
        f"linearity_{r_label}": ((λ1 - λ2) / λ1_safe).astype(np.float32),
        f"planarity_{r_label}": ((λ2 - λ3) / λ1_safe).astype(np.float32),
        f"scattering_{r_label}": (λ3 / λ1_safe).astype(np.float32),
        f"omnivariance_{r_label}": np.cbrt(λ1 * λ2 * λ3).astype(np.float32),
        f"anisotropy_{r_label}": ((λ1 - λ3) / λ1_safe).astype(np.float32),
        f"verticality_{r_label}": (1.0 - np.abs(normals[:, 2])).astype(np.float32),
        f"normal_x_{r_label}": normals[:, 0].astype(np.float32),
        f"normal_y_{r_label}": normals[:, 1].astype(np.float32),
        f"normal_z_{r_label}": normals[:, 2].astype(np.float32),
    }

    return features


def compute_roughness(xyz: np.ndarray, radius: float) -> np.ndarray:
    """
    Roughness = distance from each point to the best-fit plane of its
    neighbours. High roughness → uneven surface, kerb edges, obstacles.
    Low roughness → flat ground, smooth sidewalk.
    """
    n_points = len(xyz)
    roughness = np.zeros(n_points, dtype=np.float32)

    tree = KDTree(xyz)
    print(f"  Computing roughness (radius {radius}m)...")

    chunk_size = 50_000
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        batch_xyz = xyz[start:end]
        neighbours_list = tree.query_ball_point(batch_xyz, r=radius,
                                                workers=-1)

        for i, neighbours in enumerate(neighbours_list):
            idx = start + i
            if len(neighbours) < 3:
                continue
            pts = xyz[neighbours]
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            cov = (centered.T @ centered) / len(neighbours)
            _, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # smallest eigenvalue direction
            # Distance from the point to the local plane
            roughness[idx] = abs(np.dot(xyz[idx] - centroid, normal))

    return roughness


def compute_height_features(xyz: np.ndarray, radius: float = 1.0) -> dict:
    """
    Height-based features relative to local neighbourhood.
    These are very useful for separating ground from objects.

    Returns
    -------
    dict:
        "height_above_min" : Z - min(Z in neighbourhood)
        "height_range"     : max(Z) - min(Z) in neighbourhood
        "height_std"       : std of Z in neighbourhood
        "normalized_z"     : global Z normalized to [0, 1]
    """
    n_points = len(xyz)
    z = xyz[:, 2]

    tree = KDTree(xyz[:, :2])  # 2D tree for height comparison
    print(f"  Computing height features (radius {radius}m)...")

    height_above_min = np.zeros(n_points, dtype=np.float32)
    height_range = np.zeros(n_points, dtype=np.float32)
    height_std = np.zeros(n_points, dtype=np.float32)

    chunk_size = 50_000
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        batch_xy = xyz[start:end, :2]
        neighbours_list = tree.query_ball_point(batch_xy, r=radius,
                                                workers=-1)

        for i, neighbours in enumerate(neighbours_list):
            idx = start + i
            if len(neighbours) < 2:
                continue
            z_neighbours = z[neighbours]
            height_above_min[idx] = z[idx] - z_neighbours.min()
            height_range[idx] = z_neighbours.max() - z_neighbours.min()
            height_std[idx] = z_neighbours.std()

    # Global normalised height
    z_min, z_max = z.min(), z.max()
    normalized_z = ((z - z_min) / max(z_max - z_min, 1e-6)).astype(np.float32)

    return {
        "height_above_min": height_above_min,
        "height_range": height_range,
        "height_std": height_std,
        "normalized_z": normalized_z,
    }


def compute_density(xyz: np.ndarray, radius: float = 0.3) -> np.ndarray:
    """
    Local point density = number of neighbours within radius.
    Useful for detecting scan artefacts (sparse regions) and
    differentiating surface types.
    """
    tree = KDTree(xyz)
    print(f"  Computing point density (radius {radius}m)...")
    counts = tree.query_ball_point(xyz, r=radius, workers=-1,
                                   return_length=True)
    return np.array(counts, dtype=np.float32)


def compute_all_features(cloud: dict,
                         radii: List[float] = [0.1, 0.3, 0.5],
                         height_radius: float = 1.0,
                         density_radius: float = 0.3) -> dict:
    """
    Master function: compute all geometric features and add them to the
    cloud dict. This is the main entry point for Person 1's pipeline.

    Parameters
    ----------
    cloud : dict from loader.load_point_cloud
    radii : list of radii for eigenvalue computation (multi-scale).
            The project docs suggest eigenvalues at multiple radii to
            capture both fine (kerb edge at 0.1m) and coarse (road
            surface at 0.5m) geometry.
    height_radius : radius for height-above-ground features
    density_radius : radius for point density

    Returns
    -------
    The same cloud dict with new feature arrays added.
    """
    xyz = cloud["xyz"]
    print(f"\n{'='*60}")
    print(f"Computing geometric features for {len(xyz):,} points")
    print(f"{'='*60}")

    # Multi-scale eigenvalue features
    for r in radii:
        # Check if eigenvalues already exist (precomputed by IFP in CC)
        key_check = f"eigenvalue_1_{float(r)}"
        if key_check in cloud:
            print(f"\n--- Eigenvalues at radius {r}m: ALREADY PRESENT, skipping ---")
            # Still derive linearity/planarity etc. if missing
            _derive_from_existing_eigenvalues(cloud, r)
        else:
            print(f"\n--- Eigenvalue features at radius {r}m ---")
            eigen_features = compute_eigenvalues(xyz, radius=r)
            cloud.update(eigen_features)

        roughness_key = f"roughness_{float(r)}"
        if roughness_key in cloud:
            print(f"--- Roughness at radius {r}m: ALREADY PRESENT, skipping ---")
        else:
            print(f"--- Roughness at radius {r}m ---")
            cloud[roughness_key] = compute_roughness(xyz, radius=r)

    # Height features
    print(f"\n--- Height features ---")
    height_features = compute_height_features(xyz, radius=height_radius)
    cloud.update(height_features)

    # Density
    print(f"\n--- Point density ---")
    cloud["density"] = compute_density(xyz, radius=density_radius)

    # Intensity normalisation (useful as a feature too)
    if "intensity" in cloud:
        intensity = cloud["intensity"]
        i_max = intensity.max()
        if i_max > 0:
            cloud["intensity_normalized"] = (intensity / i_max).astype(np.float32)

    n_features = sum(1 for k, v in cloud.items()
                     if isinstance(v, np.ndarray) and v.ndim == 1
                     and k not in ("classification",))
    print(f"\nDone. Total scalar fields: {n_features}")

    return cloud
