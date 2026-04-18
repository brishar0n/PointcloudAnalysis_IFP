"""
loader.py — Point cloud I/O module
Reads .LAZ/.LAS files and returns structured numpy arrays.

Usage:
    from preprocessing.loader import load_point_cloud, save_point_cloud
    cloud = load_point_cloud("datasets/bologna_subsampled.laz")
    print(cloud["xyz"].shape)        # (N, 3)
    print(cloud["classification"])   # array of uint8
"""

import laspy
import numpy as np
import re
from pathlib import Path


def _normalize_field_name(name: str) -> str:
    """
    Normalise IFP's CloudCompare field names into clean Python keys.

    Examples:
        "1st eigenvalue (1)"   → "eigenvalue_1_1.0"
        "2nd eigenvalue (0.5)" → "eigenvalue_2_0.5"
        "3rd eigenvalue (0.1)" → "eigenvalue_3_0.1"
        "Roughness (0.5)"      → "roughness_0.5"
        "Original cloud index" → "original_cloud_index"
        "height_division"      → "height_division"  (unchanged)
        "HasColor"             → "has_color"
    """
    # Eigenvalue patterns: "1st eigenvalue (1)" etc.
    m = re.match(r"(\d)(?:st|nd|rd|th)\s+eigenvalue\s+\(([^)]+)\)", name)
    if m:
        return f"eigenvalue_{m.group(1)}_{float(m.group(2))}"

    # Roughness pattern: "Roughness (0.5)"
    m = re.match(r"[Rr]oughness\s+\(([^)]+)\)", name)
    if m:
        return f"roughness_{float(m.group(1))}"

    # CamelCase → snake_case, spaces → underscores, lowercase
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^a-zA-Z0-9._]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name


def _detect_and_apply_shift(xyz: np.ndarray) -> tuple:
    """
    Detect if coordinates are absolute (large values like Vilnius:
    X~4.9M, Y~3.2M) and shift to local origin if so.

    Returns (shifted_xyz, shift_vector). If no shift needed,
    shift_vector is [0, 0, 0].
    """
    centroid = xyz.mean(axis=0)
    if np.any(np.abs(centroid) > 10_000):
        # Absolute coordinates — shift to local origin
        shift = centroid.copy()
        shift[2] = 0  # keep Z absolute (height matters)
        shifted = xyz - shift
        print(f"  ⚠ Absolute coordinates detected — applied shift "
              f"[{shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f}]")
        return shifted, shift
    return xyz, np.zeros(3)


def load_point_cloud(filepath: str, auto_shift: bool = True) -> dict:
    """
    Load a .LAZ or .LAS file and return a dict of arrays.

    Returns
    -------
    dict with keys:
        "xyz"            : (N, 3) float64   — X, Y, Z coordinates
        "rgb"            : (N, 3) float32   — normalised to [0, 1]
        "intensity"      : (N,)   float32   — laser return intensity
        "classification" : (N,)   uint8     — manual class labels
        "gps_time"       : (N,)   float64   — GPS timestamp
        "return_number"  : (N,)   uint8
        "num_returns"    : (N,)   uint8
        + any extra fields found in the file (e.g. "height_division")
    Also includes metadata:
        "header"         : dict with scale, offset, point_count, crs info
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    las = laspy.read(str(filepath))

    # Core coordinates (with optional auto-shift for absolute coords)
    xyz = np.vstack((las.x, las.y, las.z)).T  # (N, 3)
    global_shift = np.zeros(3)
    if auto_shift:
        xyz, global_shift = _detect_and_apply_shift(xyz)

    # RGB — LAS stores as uint16, normalise to [0, 1]
    rgb = None
    if hasattr(las, "red"):
        rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
        max_val = rgb.max()
        if max_val > 0:
            rgb /= max_val

    # Standard fields
    cloud = {
        "xyz": xyz,
        "rgb": rgb,
        "intensity": np.array(las.intensity, dtype=np.float32),
        "classification": np.array(las.classification, dtype=np.uint8),
    }

    # Optional standard fields
    if hasattr(las, "gps_time"):
        cloud["gps_time"] = np.array(las.gps_time, dtype=np.float64)
    if hasattr(las, "return_number"):
        cloud["return_number"] = np.array(las.return_number, dtype=np.uint8)
    if hasattr(las, "number_of_returns"):
        cloud["num_returns"] = np.array(las.number_of_returns, dtype=np.uint8)

    # Extra fields (eigenvalues, roughness, etc.) with normalised names
    standard_names = {
        "X", "Y", "Z", "x", "y", "z",
        "red", "green", "blue",
        "intensity", "classification",
        "gps_time", "return_number", "number_of_returns",
        "scan_angle_rank", "user_data", "point_source_id",
        "scan_direction_flag", "edge_of_flight_line",
        "raw_classification", "synthetic", "key_point", "withheld",
    }
    precomputed_fields = []
    for dim_name in las.point_format.dimension_names:
        if dim_name not in standard_names and dim_name not in cloud:
            clean_name = _normalize_field_name(dim_name)
            cloud[clean_name] = np.array(getattr(las, dim_name))
            precomputed_fields.append(clean_name)

    # Header metadata
    cloud["header"] = {
        "point_count": len(las.points),
        "scale": list(las.header.scales),
        "offset": list(las.header.offsets),
        "mins": list(las.header.mins),
        "maxs": list(las.header.maxs),
        "point_format": las.header.point_format.id,
        "version": f"{las.header.version.major}.{las.header.version.minor}",
        "global_shift_applied": global_shift.tolist(),
        "source_file": filepath.name,
    }

    print(f"Loaded {cloud['header']['point_count']:,} points from {filepath.name}")
    print(f"  XYZ range: X[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}]  "
          f"Y[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}]  "
          f"Z[{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}]")
    if precomputed_fields:
        print(f"  Precomputed fields: {precomputed_fields}")
    print(f"  All fields: {[k for k in cloud if k != 'header']}")

    return cloud


def save_point_cloud(cloud: dict, filepath: str):
    """
    Save a cloud dict back to .LAZ or .LAS format.
    Preserves classification and any extra scalar fields.
    """
    filepath = Path(filepath)
    n_points = cloud["xyz"].shape[0]

    # Determine point format (3 if RGB present, 1 if not)
    has_rgb = cloud.get("rgb") is not None
    has_gps = "gps_time" in cloud
    if has_rgb and has_gps:
        point_format = 3
    elif has_gps:
        point_format = 1
    elif has_rgb:
        point_format = 2
    else:
        point_format = 0

    header = laspy.LasHeader(point_format=point_format, version="1.2")

    # Add extra dimensions for any non-standard fields
    skip_keys = {"xyz", "rgb", "intensity", "classification", "gps_time",
                 "return_number", "num_returns", "header"}
    extra_fields = [k for k in cloud if k not in skip_keys and isinstance(cloud[k], np.ndarray)]
    for field_name in extra_fields:
        arr = cloud[field_name]
        if arr.dtype in (np.float32, np.float64):
            header.add_extra_dim(laspy.ExtraBytesParams(name=field_name, type=np.float32))
        elif arr.dtype in (np.uint8, np.int8):
            header.add_extra_dim(laspy.ExtraBytesParams(name=field_name, type=np.uint8))
        else:
            header.add_extra_dim(laspy.ExtraBytesParams(name=field_name, type=np.float32))

    las = laspy.LasData(header)
    las.x = cloud["xyz"][:, 0]
    las.y = cloud["xyz"][:, 1]
    las.z = cloud["xyz"][:, 2]
    las.intensity = cloud["intensity"].astype(np.uint16)
    las.classification = cloud["classification"]

    if has_rgb:
        rgb_uint16 = (cloud["rgb"] * 65535).astype(np.uint16)
        las.red = rgb_uint16[:, 0]
        las.green = rgb_uint16[:, 1]
        las.blue = rgb_uint16[:, 2]

    if has_gps:
        las.gps_time = cloud["gps_time"]

    for field_name in extra_fields:
        setattr(las, field_name, cloud[field_name].astype(np.float32))

    las.write(str(filepath))
    print(f"Saved {n_points:,} points to {filepath.name}")


def subsample(cloud: dict, voxel_size: float = 0.05) -> dict:
    """
    Voxel-grid subsampling. Keeps one point per voxel cube.
    Useful for reducing density before heavy computation.

    Parameters
    ----------
    cloud : dict from load_point_cloud
    voxel_size : edge length of each voxel in metres

    Returns
    -------
    New cloud dict with fewer points.
    """
    xyz = cloud["xyz"]
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    # Unique voxels — keep first point in each
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original ordering

    new_cloud = {}
    for key, val in cloud.items():
        if key == "header":
            new_cloud[key] = val.copy()
        elif isinstance(val, np.ndarray):
            new_cloud[key] = val[unique_idx]
        else:
            new_cloud[key] = val

    new_cloud["header"]["point_count"] = len(unique_idx)
    print(f"Subsampled: {len(xyz):,} → {len(unique_idx):,} points "
          f"(voxel size {voxel_size}m)")
    return new_cloud
