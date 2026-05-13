from __future__ import annotations

"""
width_metrics.py

Sujeeth Gunasekaran - Width Metrics / Analysis

This file calculates practical sidewalk access measurements from the point cloud
pipeline used in the IFP pedestrian access project.

The goal is to produce useful sidewalk metrics for pedestrian access analysis,
while avoiding unrealistic results caused by noisy classification or large open
areas being labelled as sidewalk.

Supported analysis modes:

1. Point-based analysis
   - Uses classified .laz/.las files.
   - Uses sidewalk-labelled points to estimate walking direction.
   - Measures width across the sidewalk using PCA projection.
   - Uses percentile ranges to reduce the impact of outliers.
   - Filters out unrealistic width segments above a safe threshold.
   - Estimates usable width after accounting for obstacle points.

2. Boundary-based analysis
   - Uses KERB and HFE .obj boundary files.
   - Measures distances between boundary points.
"""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SIDEWALK_LABEL = 2
STREET_LABEL = 11
OBSTACLE_LABELS = {0, 5, 8, 13, 15}

# Any segment wider than this is treated as likely classifier spillover.
# Real sidewalks may vary, but 15m is a safer upper bound for this stage.
MAX_REASONABLE_WIDTH_M = 15.0


def load_laz_points(file_path: Path) -> pd.DataFrame:
    las = laspy.read(file_path)

    return pd.DataFrame({
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
        "classification": np.asarray(las.classification).astype(int),
    })


def get_sidewalk_points(points: pd.DataFrame) -> pd.DataFrame:
    sidewalk = points[points["classification"] == SIDEWALK_LABEL].copy()

    if sidewalk.empty:
        raise ValueError("No sidewalk points found using label 2.")

    return sidewalk


def get_obstacle_points(points: pd.DataFrame) -> pd.DataFrame:
    return points[points["classification"].isin(OBSTACLE_LABELS)].copy()


def estimate_sidewalk_axes(sidewalk_points: pd.DataFrame):
    """
    Estimate the main walking direction of the sidewalk.

    This avoids measuring width using raw x/y ranges, which can give unrealistic
    values when the street is diagonal or curved in the scan.
    """
    xy = sidewalk_points[["x", "y"]].to_numpy()

    origin = xy.mean(axis=0)
    centred = xy - origin

    _, _, vh = np.linalg.svd(centred, full_matrices=False)

    along_axis = vh[0]
    across_axis = np.array([-along_axis[1], along_axis[0]])

    return origin, along_axis, across_axis


def add_projection_columns(
    points: pd.DataFrame,
    origin: np.ndarray,
    along_axis: np.ndarray,
    across_axis: np.ndarray,
) -> pd.DataFrame:
    points = points.copy()

    xy = points[["x", "y"]].to_numpy()
    centred = xy - origin

    points["along_m"] = centred @ along_axis
    points["across_m"] = centred @ across_axis

    return points


def robust_range(values: pd.Series, lower: float = 5, upper: float = 95) -> float:
    """
    Calculate a percentile range instead of max-min.

    This reduces the effect of outlier points.
    """
    if values.empty:
        return 0.0

    return float(np.percentile(values, upper) - np.percentile(values, lower))


def segment_sidewalk(points: pd.DataFrame, segment_size: float = 1.0) -> pd.DataFrame:
    """
    Split sidewalk points into small sections along the walking direction.
    """
    points = points.copy()
    start = points["along_m"].min()

    points["segment_id"] = ((points["along_m"] - start) / segment_size).astype(int)

    return points


def compute_width(segment: pd.DataFrame) -> float:
    """
    Measure sidewalk width across the walking direction.
    """
    return robust_range(segment["across_m"], lower=5, upper=95)


def compute_slope(segment: pd.DataFrame) -> float:
    """
    Estimate slope percentage for one sidewalk segment.
    """
    along_range = robust_range(segment["along_m"], lower=5, upper=95)

    if along_range == 0:
        return 0.0

    z_range = robust_range(segment["z"], lower=5, upper=95)

    return float((z_range / along_range) * 100)


def compute_usable_width(
    segment: pd.DataFrame,
    projected_obstacles: pd.DataFrame,
) -> tuple[float, int]:
    """
    Estimate remaining usable width after obstacle points are considered.
    """
    total_width = compute_width(segment)

    if projected_obstacles.empty:
        return total_width, 0

    along_min = segment["along_m"].min()
    along_max = segment["along_m"].max()

    across_min = np.percentile(segment["across_m"], 5)
    across_max = np.percentile(segment["across_m"], 95)

    segment_obstacles = projected_obstacles[
        (projected_obstacles["along_m"] >= along_min)
        & (projected_obstacles["along_m"] <= along_max)
        & (projected_obstacles["across_m"] >= across_min)
        & (projected_obstacles["across_m"] <= across_max)
    ]

    if segment_obstacles.empty:
        return total_width, 0

    obstacle_span = robust_range(segment_obstacles["across_m"], lower=5, upper=95)
    usable_width = max(total_width - obstacle_span, 0.0)

    return float(usable_width), int(len(segment_obstacles))


def compute_segment_metrics(points: pd.DataFrame, segment_size: float = 1.0) -> tuple[pd.DataFrame, int]:
    sidewalk = get_sidewalk_points(points)

    origin, along_axis, across_axis = estimate_sidewalk_axes(sidewalk)

    projected_points = add_projection_columns(
        points,
        origin,
        along_axis,
        across_axis,
    )

    sidewalk = get_sidewalk_points(projected_points)
    obstacles = get_obstacle_points(projected_points)

    sidewalk = segment_sidewalk(sidewalk, segment_size)

    rows = []
    skipped_unrealistic_segments = 0

    for segment_id, group in sidewalk.groupby("segment_id"):
        if len(group) < 20:
            continue

        overall_width = compute_width(group)

        # Client-safe filter:
        # very large "sidewalk" widths are usually caused by classifier spillover,
        # plazas, open areas, or disconnected surfaces being grouped together.
        if overall_width > MAX_REASONABLE_WIDTH_M:
            skipped_unrealistic_segments += 1
            continue

        slope_percent = compute_slope(group)
        usable_width, obstacle_count = compute_usable_width(group, obstacles)

        rows.append({
            "segment_id": int(segment_id),
            "point_count": int(len(group)),
            "overall_width_m": overall_width,
            "usable_width_m": usable_width,
            "obstacle_count": obstacle_count,
            "slope_percent": slope_percent,
        })

    if not rows:
        raise ValueError("No valid sidewalk segments found after filtering.")

    return pd.DataFrame(rows), skipped_unrealistic_segments


def summarise_segment_metrics(metrics: pd.DataFrame, skipped_unrealistic_segments: int) -> dict:
    return {
        "segment_count": int(len(metrics)),
        "skipped_unrealistic_segments": int(skipped_unrealistic_segments),
        "max_reasonable_width_m": float(MAX_REASONABLE_WIDTH_M),
        "average_overall_width_m": float(metrics["overall_width_m"].mean()),
        "minimum_overall_width_m": float(metrics["overall_width_m"].min()),
        "maximum_overall_width_m": float(metrics["overall_width_m"].max()),
        "average_usable_width_m": float(metrics["usable_width_m"].mean()),
        "minimum_usable_width_m": float(metrics["usable_width_m"].min()),
        "maximum_usable_width_m": float(metrics["usable_width_m"].max()),
        "average_slope_percent": float(metrics["slope_percent"].mean()),
        "maximum_slope_percent": float(metrics["slope_percent"].max()),
        "total_obstacle_points": int(metrics["obstacle_count"].sum()),
        "quality_note": (
            "Segments above the reasonable width threshold were excluded because "
            "they are likely caused by classifier spillover or large open areas."
        ),
    }


def load_obj_vertices(obj_path: Path) -> np.ndarray:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "sidewalk_segment_metrics.csv"
    summary_path = output_dir / "sidewalk_metrics_summary.json"

    metrics.to_csv(metrics_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved segment metrics: {metrics_path}")
    print(f"Saved summary metrics: {summary_path}")


def save_boundary_metric_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "boundary_width_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved boundary width summary: {summary_path}")


def run_point_metrics(input_path: Path, output_dir: Path, segment_size: float) -> None:
    points = load_laz_points(input_path)
    metrics, skipped_unrealistic_segments = compute_segment_metrics(points, segment_size)

    summary = summarise_segment_metrics(metrics, skipped_unrealistic_segments)
    summary["input_file"] = str(input_path)
    summary["method"] = (
        "PCA-projected point-based width metrics using robust percentiles "
        "with filtering for unrealistic sidewalk widths"
    )

    save_point_metric_outputs(metrics, summary, output_dir)


def run_boundary_metrics(kerb_obj: Path, hfe_obj: Path, output_dir: Path) -> None:
    kerb_points = load_obj_vertices(kerb_obj)
    hfe_points = load_obj_vertices(hfe_obj)

    summary = compute_boundary_widths(kerb_points, hfe_points)

    summary["kerb_obj"] = str(kerb_obj)
    summary["hfe_obj"] = str(hfe_obj)
    summary["method"] = "Boundary-based width between kerb and HFE points"

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