from __future__ import annotations

"""
separate_boundary_sides.py

Separates a combined sidewalk boundary point cloud into:
- kerb-side boundary
- frontage/HFE-side boundary

Method:
Boundary points closer to street-class points are treated as kerb-side.
Boundary points farther from street-class points are treated as frontage/HFE-side.

This script is designed to work with any classified LAS/LAZ file and any
combined boundary LAS/LAZ file.

Outputs:
- sidewalk_boundary_kerb.obj
- sidewalk_boundary_hfe.obj
- sidewalk_boundary_kerb.las
- sidewalk_boundary_hfe.las

The OBJ files are used by the width metrics module.
The LAS files are easier to open and visualise in CloudCompare.
"""

import argparse
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree


STREET_LABEL = 11

KERB_VIS_LABEL = 21
HFE_VIS_LABEL = 22


def load_points_from_las(path: Path) -> tuple[np.ndarray, laspy.LasData]:
    """
    Load point coordinates from a LAS/LAZ file.
    """
    las = laspy.read(path)
    points = np.column_stack((las.x, las.y, las.z))
    return points, las


def get_street_points(classified_las: laspy.LasData, street_label: int) -> np.ndarray:
    """
    Extract street/road points from the classified point cloud.

    These street points are used as a reference to decide which boundary points
    are closer to the road side.
    """
    labels = np.asarray(classified_las.classification)
    mask = labels == street_label

    street_points = np.column_stack((
        np.asarray(classified_las.x)[mask],
        np.asarray(classified_las.y)[mask],
        np.asarray(classified_las.z)[mask],
    ))

    if len(street_points) == 0:
        raise ValueError(f"No street points found using classification label {street_label}.")

    return street_points


def split_boundary_by_street_distance(
    boundary_points: np.ndarray,
    street_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Split boundary points using distance to nearest street point.

    Closer boundary points are treated as kerb-side.
    Farther boundary points are treated as frontage/HFE-side.
    """
    street_tree = cKDTree(street_points[:, :2])
    distances, _ = street_tree.query(boundary_points[:, :2], k=1)

    threshold = float(np.median(distances))

    kerb_mask = distances <= threshold
    hfe_mask = distances > threshold

    kerb_points = boundary_points[kerb_mask]
    hfe_points = boundary_points[hfe_mask]

    if len(kerb_points) == 0 or len(hfe_points) == 0:
        raise ValueError("Boundary separation failed: one side has no points.")

    return kerb_points, hfe_points, threshold


def save_obj(points: np.ndarray, output_path: Path, name: str) -> None:
    """
    Save points as a simple OBJ vertex file.

    The width metrics module reads OBJ vertex lines beginning with 'v '.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"# {name}\n")
        for x, y, z in points:
            file.write(f"v {x} {y} {z}\n")

    print(f"Saved {name}: {output_path} ({len(points):,} points)")


def save_las(points: np.ndarray, output_path: Path, classification: int) -> None:
    """
    Save separated boundary points as a LAS file for CloudCompare visualisation.

    Different classification values are used so kerb and HFE/frontage sides
    can be visually distinguished after loading.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=0, version="1.2")
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = np.full(len(points), classification, dtype=np.uint8)

    las.write(str(output_path))

    print(f"Saved LAS visualisation file: {output_path} ({len(points):,} points)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate combined sidewalk boundary into kerb and HFE/frontage sides."
    )

    parser.add_argument(
        "--classified",
        required=True,
        help="Classified LAS/LAZ file containing street label 11.",
    )
    parser.add_argument(
        "--boundary",
        required=True,
        help="Combined sidewalk boundary LAS/LAZ file.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for separated boundary files.",
    )
    parser.add_argument(
        "--street-label",
        type=int,
        default=STREET_LABEL,
        help="Street/road classification label. Default is 11.",
    )

    args = parser.parse_args()

    classified_path = Path(args.classified)
    boundary_path = Path(args.boundary)
    output_dir = Path(args.output)

    print("=== Boundary Side Separation ===")
    print(f"Classified input: {classified_path}")
    print(f"Boundary input:   {boundary_path}")

    _, classified_las = load_points_from_las(classified_path)
    boundary_points, _ = load_points_from_las(boundary_path)

    street_points = get_street_points(classified_las, args.street_label)

    print(f"Loaded boundary points: {len(boundary_points):,}")
    print(f"Loaded street points:   {len(street_points):,}")

    kerb_points, hfe_points, threshold = split_boundary_by_street_distance(
        boundary_points,
        street_points,
    )

    print(f"Distance threshold used: {threshold:.3f}m")
    print(f"Kerb-side points:        {len(kerb_points):,}")
    print(f"HFE/frontage points:     {len(hfe_points):,}")

    kerb_obj_path = output_dir / "sidewalk_boundary_kerb.obj"
    hfe_obj_path = output_dir / "sidewalk_boundary_hfe.obj"

    kerb_las_path = output_dir / "sidewalk_boundary_kerb.las"
    hfe_las_path = output_dir / "sidewalk_boundary_hfe.las"

    save_obj(
        kerb_points,
        kerb_obj_path,
        "Kerb-side sidewalk boundary",
    )

    save_obj(
        hfe_points,
        hfe_obj_path,
        "HFE/frontage-side sidewalk boundary",
    )

    save_las(
        kerb_points,
        kerb_las_path,
        classification=KERB_VIS_LABEL,
    )

    save_las(
        hfe_points,
        hfe_las_path,
        classification=HFE_VIS_LABEL,
    )

    print("\nDone.")
    print("Generated:")
    print(f" - {kerb_obj_path}")
    print(f" - {hfe_obj_path}")
    print(f" - {kerb_las_path}")
    print(f" - {hfe_las_path}")


if __name__ == "__main__":
    main()