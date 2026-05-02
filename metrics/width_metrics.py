from __future__ import annotations

"""
width_metrics.py

Sujeeth Gunasekaran - Metrics & Analysis

This module calculates sidewalk accessibility metrics from the team pipeline.

It supports two modes:

1. Point-based metrics
   - Input: classified .laz/.las file
   - Uses class labels:
        2  = sidewalk
        11 = street
        0, 5, 8, 13, 15 = obstacles / other objects
   - Outputs segment-level width, usable width, obstacle count and slope.

2. Boundary-based metrics
   - Input: kerb .obj and HFE .obj files from Ahmed's boundary extraction module
   - Calculates the distance between kerb and HFE boundary points.
   - This is the more realistic sidewalk width method once boundary extraction is ready.
"""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Team label format
SIDEWALK_LABEL = 2
STREET_LABEL = 11

# Extra labels seen in the team visualisation notebook.
# These are treated as possible obstacles when they appear inside a sidewalk segment.
OBSTACLE_LABELS = {0, 5, 8, 13, 15}


def load_laz_points(file_path: Path) -> pd.DataFrame:
    """
    Load a classified LAS/LAZ point cloud into a DataFrame.

    This keeps only the columns needed for metrics:
    x, y, z and classification.
    """
    las = laspy.read(file_path)

    return pd.DataFrame({
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
        "classification": np.asarray(las.classification).astype(int),
    })


def get_sidewalk_points(points: pd.DataFrame) -> pd.DataFrame:
    """
    Return only sidewalk points.

    In the team pipeline, label 2 represents sidewalk/pedestrian surface.
    """
    sidewalk = points[points["classification"] == SIDEWALK_LABEL].copy()

    if sidewalk.empty:
        raise ValueError("No sidewalk points found using label 2.")

    return sidewalk


def get_obstacle_points(points: pd.DataFrame) -> pd.DataFrame:
    """
    Return points that may block pedestrian movement.

    This includes vegetation, furniture, vehicles and unclassified objects.
    """
    return points[points["classification"].isin(OBSTACLE_LABELS)].copy()


def segment_sidewalk(points: pd.DataFrame, segment_size: float = 1.0) -> pd.DataFrame:
    """
    Split sidewalk points into small segments along the x-axis.

    This is a baseline segmentation method. It gives us local measurements instead
    of only one global width value for the entire sidewalk.
    """
    points = points.copy()
    x_min = points["x"].min()
    points["segment_id"] = ((points["x"] - x_min) / segment_size).astype(int)
    return points


def compute_width(segment: pd.DataFrame) -> float:
    """
    Estimate segment width using the y-axis range.

    This is the fallback point-cloud method. The stronger method is the boundary
    method further below, which uses kerb and HFE polylines.
    """
    return float(segment["y"].max() - segment["y"].min())


def compute_slope(segment: pd.DataFrame) -> float:
    """
    Estimate slope percentage from z-change over x-distance.

    This is a simple local slope estimate for each segment.
    """
    x_range = segment["x"].max() - segment["x"].min()

    if x_range == 0:
        return 0.0

    z_range = segment["z"].max() - segment["z"].min()
    return float((z_range / x_range) * 100)


def compute_usable_width(segment: pd.DataFrame, all_points: pd.DataFrame) -> tuple[float, int]:
    """
    Estimate usable width after accounting for obstacles.

    Logic:
    - Start with the total sidewalk width.
    - Find obstacle points inside the same x/y corridor as the segment.
    - Subtract the obstacle span from the total width.
    - Never return a negative width.
    """
    total_width = compute_width(segment)
    obstacles = get_obstacle_points(all_points)

    if obstacles.empty:
        return total_width, 0

    x_min = segment["x"].min()
    x_max = segment["x"].max()
    y_min = segment["y"].min()
    y_max = segment["y"].max()

    segment_obstacles = obstacles[
        (obstacles["x"] >= x_min) &
        (obstacles["x"] <= x_max) &
        (obstacles["y"] >= y_min) &
        (obstacles["y"] <= y_max)
    ]

    if segment_obstacles.empty:
        return total_width, 0

    obstacle_span = segment_obstacles["y"].max() - segment_obstacles["y"].min()
    usable_width = max(total_width - obstacle_span, 0.0)

    return float(usable_width), int(len(segment_obstacles))


def compute_segment_metrics(points: pd.DataFrame, segment_size: float = 1.0) -> pd.DataFrame:
    """
    Generate all point-based metrics for each sidewalk segment.

    Output per segment:
    - overall width
    - usable width
    - obstacle count
    - slope percentage
    """
    sidewalk = get_sidewalk_points(points)
    sidewalk = segment_sidewalk(sidewalk, segment_size)

    rows = []

    for segment_id, group in sidewalk.groupby("segment_id"):
        overall_width = compute_width(group)
        slope_percent = compute_slope(group)
        usable_width, obstacle_count = compute_usable_width(group, points)

        rows.append({
            "segment_id": int(segment_id),
            "point_count": int(len(group)),
            "overall_width_m": overall_width,
            "usable_width_m": usable_width,
            "obstacle_count": obstacle_count,
            "slope_percent": slope_percent,
        })

    return pd.DataFrame(rows)


def summarise_segment_metrics(metrics: pd.DataFrame) -> dict:
    """
    Create a summary dictionary from the segment-level metrics table.
    """
    return {
        "segment_count": int(len(metrics)),
        "average_overall_width_m": float(metrics["overall_width_m"].mean()),
        "minimum_overall_width_m": float(metrics["overall_width_m"].min()),
        "maximum_overall_width_m": float(metrics["overall_width_m"].max()),
        "average_usable_width_m": float(metrics["usable_width_m"].mean()),
        "minimum_usable_width_m": float(metrics["usable_width_m"].min()),
        "maximum_usable_width_m": float(metrics["usable_width_m"].max()),
        "average_slope_percent": float(metrics["slope_percent"].mean()),
        "maximum_slope_percent": float(metrics["slope_percent"].max()),
        "total_obstacle_points": int(metrics["obstacle_count"].sum()),
    }


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """
    Read vertex points from an OBJ file.

    Ahmed's boundary extraction saves kerb and HFE boundaries as OBJ files.
    OBJ vertex lines look like:
        v x y z
    """
    vertices = []

    with open(obj_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])

    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {obj_path}")

    return np.asarray(vertices)


def compute_boundary_widths(kerb_points: np.ndarray, hfe_points: np.ndarray) -> dict:
    """
    Compute sidewalk width using kerb and HFE boundary points.

    This is the more realistic width method:
    - Kerb = road-side boundary
    - HFE = building/frontage-side boundary

    For each kerb point, the nearest HFE point is found.
    Those distances are then summarised.
    """
    tree = cKDTree(hfe_points[:, :2])
    distances, _ = tree.query(kerb_points[:, :2], k=1)

    return {
        "boundary_average_width_m": float(np.mean(distances)),
        "boundary_minimum_width_m": float(np.min(distances)),
        "boundary_maximum_width_m": float(np.max(distances)),
        "boundary_median_width_m": float(np.median(distances)),
        "boundary_sample_count": int(len(distances)),
    }


def save_point_metric_outputs(
    metrics: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    """
    Save point-based segment metrics to CSV and summary metrics to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "sidewalk_segment_metrics.csv"
    summary_path = output_dir / "sidewalk_metrics_summary.json"

    metrics.to_csv(metrics_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved segment metrics: {metrics_path}")
    print(f"Saved summary metrics: {summary_path}")


def save_boundary_metric_outputs(summary: dict, output_dir: Path) -> None:
    """
    Save boundary-based width summary to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "boundary_width_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved boundary width summary: {summary_path}")


def run_point_metrics(input_path: Path, output_dir: Path, segment_size: float) -> None:
    """
    Run the point-based metric workflow.
    """
    points = load_laz_points(input_path)
    metrics = compute_segment_metrics(points, segment_size)
    summary = summarise_segment_metrics(metrics)

    summary["input_file"] = str(input_path)
    summary["method"] = "Point-based segmented width, usable width, obstacle count and slope"

    save_point_metric_outputs(metrics, summary, output_dir)


def run_boundary_metrics(kerb_obj: Path, hfe_obj: Path, output_dir: Path) -> None:
    """
    Run the boundary-based metric workflow using Ahmed's kerb/HFE outputs.
    """
    kerb_points = load_obj_vertices(kerb_obj)
    hfe_points = load_obj_vertices(hfe_obj)

    summary = compute_boundary_widths(kerb_points, hfe_points)

    summary["kerb_obj"] = str(kerb_obj)
    summary["hfe_obj"] = str(hfe_obj)
    summary["method"] = "Boundary-based width between kerb and HFE polylines"

    save_boundary_metric_outputs(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sidewalk width, usable width, obstacle and slope metrics."
    )

    parser.add_argument(
        "--input",
        help="Classified LAZ/LAS file for point-based metrics.",
    )
    parser.add_argument(
        "--kerb-obj",
        help="Kerb boundary OBJ file from boundary extraction.",
    )
    parser.add_argument(
        "--hfe-obj",
        help="HFE boundary OBJ file from boundary extraction.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/width_metrics",
        help="Output directory.",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=1.0,
        help="Segment size in metres for point-based metrics.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    has_point_input = args.input is not None
    has_boundary_input = args.kerb_obj is not None and args.hfe_obj is not None

    if has_point_input:
        run_point_metrics(
            input_path=Path(args.input),
            output_dir=output_dir,
            segment_size=args.segment_size,
        )

    if has_boundary_input:
        run_boundary_metrics(
            kerb_obj=Path(args.kerb_obj),
            hfe_obj=Path(args.hfe_obj),
            output_dir=output_dir,
        )

    if not has_point_input and not has_boundary_input:
        raise ValueError(
            "Please provide either --input for point-based metrics, "
            "or both --kerb-obj and --hfe-obj for boundary-based metrics."
        )


if __name__ == "__main__":
    main()