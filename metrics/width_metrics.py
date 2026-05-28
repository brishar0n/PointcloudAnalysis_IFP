from __future__ import annotations
import os
"""
width_metrics.py

Sujeeth Gunasekaran - Width Metrics / Analysis
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

MAX_REASONABLE_WIDTH_M = 15.0
MIN_REASONABLE_BOUNDARY_WIDTH_M = 0.8
MAX_REASONABLE_BOUNDARY_WIDTH_M = 15.0


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
    if values.empty:
        return 0.0

    return float(np.percentile(values, upper) - np.percentile(values, lower))


def segment_sidewalk(points: pd.DataFrame, segment_size: float = 1.0) -> pd.DataFrame:
    points = points.copy()
    start = points["along_m"].min()
    points["segment_id"] = ((points["along_m"] - start) / segment_size).astype(int)
    return points


def compute_width(segment: pd.DataFrame) -> float:
    return robust_range(segment["across_m"], lower=5, upper=95)


def compute_slope(segment: pd.DataFrame) -> float:
    along_range = robust_range(segment["along_m"], lower=5, upper=95)

    if along_range == 0:
        return 0.0

    z_range = robust_range(segment["z"], lower=5, upper=95)
    return float((z_range / along_range) * 100)


def compute_usable_width(
    segment: pd.DataFrame,
    projected_obstacles: pd.DataFrame,
) -> tuple[float, int]:
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

    segmented_sidewalk = segment_sidewalk(sidewalk, segment_size)

    rows = []
    skipped_unrealistic_segments = 0

    for segment_id, group in segmented_sidewalk.groupby("segment_id"):
        if len(group) < 20:
            continue

        overall_width = compute_width(group)

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

    metrics_df = pd.DataFrame(rows)
    return metrics_df, skipped_unrealistic_segments, segmented_sidewalk


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


def load_boundary_points(boundary_path: Path) -> np.ndarray:
    """
    Load boundary points from OBJ, LAS, or LAZ files.
    """
    suffix = boundary_path.suffix.lower()

    if suffix == ".obj":
        return load_obj_vertices(boundary_path)

    if suffix in {".las", ".laz"}:
        las = laspy.read(boundary_path)
        return np.column_stack((
            np.asarray(las.x),
            np.asarray(las.y),
            np.asarray(las.z),
        ))

    raise ValueError(
        f"Unsupported boundary file format: {boundary_path}. "
        "Use .obj, .las, or .laz."
    )


def compute_boundary_widths(kerb_points: np.ndarray, hfe_points: np.ndarray) -> dict:
    tree = cKDTree(hfe_points[:, :2])
    distances, _ = tree.query(kerb_points[:, :2], k=1)

    filtered_distances = distances[
        (distances >= MIN_REASONABLE_BOUNDARY_WIDTH_M)
        & (distances <= MAX_REASONABLE_BOUNDARY_WIDTH_M)
    ]

    summary = {
        "boundary_average_width_m": float(np.mean(distances)),
        "boundary_minimum_width_m": float(np.min(distances)),
        "boundary_maximum_width_m": float(np.max(distances)),
        "boundary_median_width_m": float(np.median(distances)),
        "boundary_sample_count": int(len(distances)),
        "boundary_filter_min_m": float(MIN_REASONABLE_BOUNDARY_WIDTH_M),
        "boundary_filter_max_m": float(MAX_REASONABLE_BOUNDARY_WIDTH_M),
        "boundary_filtered_sample_count": int(len(filtered_distances)),
        "boundary_removed_outlier_count": int(len(distances) - len(filtered_distances)),
        "quality_note": (
            "Raw boundary distances are included, but filtered values exclude "
            "very small overlaps and very large likely mismatches."
        ),
    }

    if len(filtered_distances) > 0:
        summary.update({
            "boundary_filtered_average_width_m": float(np.mean(filtered_distances)),
            "boundary_filtered_minimum_width_m": float(np.min(filtered_distances)),
            "boundary_filtered_maximum_width_m": float(np.max(filtered_distances)),
            "boundary_filtered_median_width_m": float(np.median(filtered_distances)),
        })
    else:
        summary.update({
            "boundary_filtered_average_width_m": None,
            "boundary_filtered_minimum_width_m": None,
            "boundary_filtered_maximum_width_m": None,
            "boundary_filtered_median_width_m": None,
        })

    return summary


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

def save_segmented_points_as_laz(
    output_path: Path,
    original_las_path: Path,
    all_points: pd.DataFrame,
    segmented_sidewalk: pd.DataFrame,
    segment_metrics: pd.DataFrame,
) -> None:
    """Saves all points with new segment/width attributes on sidewalk points."""
    final_points = all_points.copy()

    if not segmented_sidewalk.empty and not segment_metrics.empty:
        # This DataFrame will have the metrics for each sidewalk point
        sidewalk_with_metrics = pd.merge(
            segmented_sidewalk,
            segment_metrics[["segment_id", "overall_width_m", "usable_width_m", "slope_percent"]],
            on="segment_id",
            how="left"
        )
        sidewalk_with_metrics.fillna({
        "segment_id": -1,
        "overall_width_m": -1.0,
        "usable_width_m": -1.0,
        "slope_percent": -1.0
        }, inplace=True)
    
        #TEMPORARILY COMMENTED OUT
        temp_df=final_points[final_points["classification"] != SIDEWALK_LABEL]
        temp_df["segment_id"] = -2
        temp_df["overall_width_m"] = -1.0
        temp_df["usable_width_m"] = -1.0
        temp_df["slope_percent"] = -1.0
        final_points=pd.concat([temp_df,sidewalk_with_metrics])

    # Create a new LAZ file from the original's header to preserve offsets/scales
    source_las = laspy.read(original_las_path)
    header = laspy.LasHeader(version=source_las.header.version, point_format=source_las.header.point_format.id)
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales

    # Add the extra dimensions for our new features
    header.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="segment_overall_width", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="segment_usable_width", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="segment_slope_percent", type=np.float32))

    las_out = laspy.LasData(header)
    # final_points=final_points.loc[final_points["classification"] == SIDEWALK_LABEL]
    # final_points=final_sidewalk_with_metrics
    # final_points=sidewalk_with_metrics
    
    las_out.x, las_out.y, las_out.z = final_points["x"], final_points["y"], final_points["z"]
    las_out.classification = final_points["classification"]
    las_out.segment_id = final_points["segment_id"].astype(np.int32)
    las_out.segment_overall_width = final_points["overall_width_m"].astype(np.float32)
    las_out.segment_usable_width = final_points["usable_width_m"].astype(np.float32)
    las_out.segment_slope_percent = final_points["slope_percent"].astype(np.float32)

    las_out.write(output_path)
    print(f"Saved all points with sidewalk width data: {output_path}")


def save_boundary_metric_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "boundary_width_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved boundary width summary: {summary_path}")


def run_point_metrics(input_path: Path, output_dir: Path, segment_size: float) -> None:
    points = load_laz_points(input_path)
    metrics, skipped_unrealistic_segments, segmented_sidewalk = compute_segment_metrics(points, segment_size)

    summary = summarise_segment_metrics(metrics, skipped_unrealistic_segments)
    summary["input_file"] = str(input_path)
    summary["method"] = (
        "PCA-projected point-based width metrics using robust percentiles "
        "with filtering for unrealistic sidewalk widths"
    )

    save_point_metric_outputs(metrics, summary, output_dir)
    
    # save outputs to a laz file
    laz_output_path = output_dir / "sidewalk_segmented_points.laz"
    save_segmented_points_as_laz(
        output_path=laz_output_path,
        original_las_path=input_path,
        all_points=points,
        segmented_sidewalk=segmented_sidewalk,
        segment_metrics=metrics,
    )


def run_boundary_metrics(kerb_file: Path, hfe_file: Path, output_dir: Path) -> None:
    kerb_points = load_boundary_points(kerb_file)
    hfe_points = load_boundary_points(hfe_file)

    summary = compute_boundary_widths(kerb_points, hfe_points)

    summary["kerb_file"] = str(kerb_file)
    summary["hfe_file"] = str(hfe_file)
    summary["method"] = (
        "Boundary-based width between kerb/KI and HFE points with raw and filtered summaries"
    )

    save_boundary_metric_outputs(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sidewalk width, usable width, obstacle and slope metrics."
    )

    parser.add_argument(
        "--input-metric",
        help="Classified LAZ/LAS file for point-based metrics.",
    )
    parser.add_argument(
        "--kerb-file",
        help="Kerb/KI boundary file from boundary extraction. Supports .obj, .las, or .laz.",
    )
    parser.add_argument(
        "--hfe-file",
        help="HFE/frontage boundary file from boundary extraction. Supports .obj, .las, or .laz.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/",
        help="Output directory.",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=1.0,
        help="Segment size in metres for point-based metrics.",
    )
    parser.add_argument(
        "--metric-city",
        default='bologna',
        help="city_name of the input file.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir=Path(os.path.join(output_dir,args.metric_city))

    has_point_input = args.input_metric is not None
    has_boundary_input = args.kerb_file is not None and args.hfe_file is not None

    if has_point_input:
        run_point_metrics(
            input_path=Path(args.input_metric),
            output_dir=output_dir,
            segment_size=args.segment_size,
        )

    if has_boundary_input:
        run_boundary_metrics(
            kerb_file=Path(args.kerb_file),
            hfe_file=Path(args.hfe_file),
            output_dir=output_dir,
        )

    if not has_point_input and not has_boundary_input:
        raise ValueError(
            "Please provide either --input-metric for point-based metrics, "
            "or both --kerb-file and --hfe-file for boundary-based metrics."
        )


if __name__ == "__main__":
    main()