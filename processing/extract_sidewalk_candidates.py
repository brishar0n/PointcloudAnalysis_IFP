from __future__ import annotations

import argparse
from pathlib import Path
import re

import laspy
import numpy as np



# Sidewalk candidate extraction / boundary extraction baseline.
# Input comes from the preprocessing module:
#   preprocessed/<city>/low_featured.laz
#
# This file already contains street-level points and geometric features
# such as height, planarity, roughness, density, and eigenvalue-derived fields.
#
# This script filters likely sidewalk candidate points and saves them
# for later boundary/width analysis.

def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("(", "_").replace(")", "")
    s = s.replace(".", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def get_dimension_names(las: laspy.LasData) -> list[str]:
    return [str(n) for n in las.point_format.dimension_names]


def build_name_map(las: laspy.LasData) -> dict[str, str]:
    return {normalize_name(name): name for name in get_dimension_names(las)}


def get_field(
    las: laspy.LasData,
    name_map: dict[str, str],
    candidates: list[str],
) -> np.ndarray | None:
    for candidate in candidates:
        if candidate in name_map:
            return np.asarray(las[name_map[candidate]])
    return None


def get_planarity(
    las: laspy.LasData,
    name_map: dict[str, str],
    radius: str,
) -> np.ndarray | None:
    planarity = get_field(las, name_map, [
        f"planarity_{radius}",
        f"planarity_{radius}m",
    ])

    if planarity is not None:
        return planarity.astype(np.float32)

    eig1 = get_field(las, name_map, [
        f"eigenvalue_1_{radius}",
        f"1st_eigenvalue_{radius}",
    ])
    eig2 = get_field(las, name_map, [
        f"eigenvalue_2_{radius}",
        f"2nd_eigenvalue_{radius}",
    ])
    eig3 = get_field(las, name_map, [
        f"eigenvalue_3_{radius}",
        f"3rd_eigenvalue_{radius}",
    ])

    if eig1 is None or eig2 is None or eig3 is None:
        return None

    eig1 = eig1.astype(np.float32)
    eig2 = eig2.astype(np.float32)
    eig3 = eig3.astype(np.float32)

    eps = 1e-8
    return (eig2 - eig3) / np.maximum(eig1, eps)


def get_roughness(
    las: laspy.LasData,
    name_map: dict[str, str],
    radius: str,
) -> np.ndarray | None:
    roughness = get_field(las, name_map, [
        f"roughness_{radius}",
        f"roughness_{radius}m",
    ])

    if roughness is None:
        return None

    return roughness.astype(np.float32)


def get_height_feature(
    las: laspy.LasData,
    name_map: dict[str, str],
) -> np.ndarray:
    height = get_field(las, name_map, [
        "height_above_min",
        "height_division",
        "normalized_z",
        "z",
    ])

    if height is None:
        return np.asarray(las.z).astype(np.float32)

    return height.astype(np.float32)


def clean_feature(feature: np.ndarray, fill_value: float) -> np.ndarray:
    return np.nan_to_num(
        feature,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    ).astype(np.float32)


def get_classifier_mask(
    las: laspy.LasData,
    use_classifier: bool,
    sidewalk_label: int,
) -> np.ndarray:
    if not use_classifier:
        return np.ones(len(las.points), dtype=bool)

    labels = np.asarray(las.classification)
    return labels == sidewalk_label


def save_filtered_cloud(
    las: laspy.LasData,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    filtered = laspy.create(
        file_version=las.header.version,
        point_format=las.header.point_format,
    )
    filtered.points = las.points[mask]
    filtered.write(output_path)


def save_summary(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    height_threshold: float,
    kept: int,
    total: int,
    used_classifier: bool,
) -> None:
    summary_path = output_path.with_suffix(".txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sidewalk candidate extraction summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Radius used: {args.radius}\n")
        f.write(f"Used classifier labels: {used_classifier}\n")
        f.write(f"Sidewalk label: {args.sidewalk_label}\n")
        f.write(f"Height percentile: {args.height_max_percentile}\n")
        f.write(f"Height threshold: {height_threshold:.4f}\n")
        f.write(f"Planarity minimum: {args.planarity_min}\n")
        f.write(f"Roughness maximum: {args.roughness_max}\n")
        f.write(f"Points kept: {kept} / {total} ({100 * kept / total:.2f}%)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first-pass sidewalk candidate points"
    )

    parser.add_argument(
        "input",
        help="Path to feature-enriched or classified LAZ/LAS file",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/sidewalk_candidates.laz",
        help="Output LAZ path",
    )
    parser.add_argument(
        "--radius",
        default="0.1",
        help="Feature radius to use, e.g. 0.1, 0.3, 0.5",
    )
    parser.add_argument(
        "--planarity-min",
        type=float,
        default=0.35,
        help="Minimum planarity threshold",
    )
    parser.add_argument(
        "--roughness-max",
        type=float,
        default=0.06,
        help="Maximum roughness threshold",
    )
    parser.add_argument(
        "--height-max-percentile",
        type=float,
        default=65.0,
        help="Keep points below this height percentile",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Use classification label to keep predicted sidewalk points first",
    )
    parser.add_argument(
        "--sidewalk-label",
        type=int,
        default=2,
        help="Classification label for sidewalk. Current team format: 2 = sidewalk",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    las = laspy.read(input_path)

    print("\nAvailable dimensions:")
    for name in get_dimension_names(las):
        print(f" - {name}")

    name_map = build_name_map(las)
    radius_key = args.radius.replace(".", "_")

    height = get_height_feature(las, name_map)
    planarity = get_planarity(las, name_map, radius_key)
    roughness = get_roughness(las, name_map, radius_key)

    if planarity is None:
        raise ValueError(f"Could not find planarity for radius {args.radius}")

    if roughness is None:
        raise ValueError(f"Could not find roughness for radius {args.radius}")

    height = clean_feature(height, fill_value=np.nanmedian(height))
    planarity = clean_feature(planarity, fill_value=0.0)
    roughness = clean_feature(roughness, fill_value=np.nanmax(roughness))

    classifier_mask = get_classifier_mask(
        las,
        use_classifier=args.use_classifier,
        sidewalk_label=args.sidewalk_label,
    )

    if classifier_mask.sum() == 0:
        raise ValueError(
            "No points matched the classifier sidewalk label. "
            "Check the label value or input file."
        )

    height_threshold = np.percentile(
        height[classifier_mask],
        args.height_max_percentile,
    )

    print("\nFeature summary:")
    print(f" - height min/max: {height.min():.4f} / {height.max():.4f}")
    print(f" - planarity min/max: {planarity.min():.4f} / {planarity.max():.4f}")
    print(f" - roughness min/max: {roughness.min():.4f} / {roughness.max():.4f}")

    print("\nUsing thresholds:")
    print(f" - classifier used = {args.use_classifier}")
    print(f" - sidewalk label = {args.sidewalk_label}")
    print(f" - height <= percentile {args.height_max_percentile} -> {height_threshold:.4f}")
    print(f" - planarity >= {args.planarity_min}")
    print(f" - roughness <= {args.roughness_max}")

    mask = (
        classifier_mask &
        (height <= height_threshold) &
        (planarity >= args.planarity_min) &
        (roughness <= args.roughness_max)
    )

    kept = int(mask.sum())
    total = int(len(mask))

    print(f"\nKept {kept:,} / {total:,} points ({100 * kept / total:.2f}%)")

    save_filtered_cloud(las, mask, output_path)
    save_summary(
        input_path=input_path,
        output_path=output_path,
        args=args,
        height_threshold=height_threshold,
        kept=kept,
        total=total,
        used_classifier=args.use_classifier,
    )

    print(f"Saved sidewalk candidates -> {output_path.resolve()}")
    print(f"Saved summary -> {output_path.with_suffix('.txt').resolve()}")

    # TODO:
    # - turn sidewalk candidates into actual boundary lines
    # - improve road/sidewalk separation
    # - test on more classified city outputs
    # - pass boundary output to width metrics


if __name__ == "__main__":
    main()