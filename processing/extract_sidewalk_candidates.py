from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import laspy


# first pass at extracting sidewalk candidate points
# still a baseline and will need more refinement later


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
    raw_names = get_dimension_names(las)
    return {normalize_name(name): name for name in raw_names}


def get_field(
    las: laspy.LasData,
    name_map: dict[str, str],
    candidates: list[str]
) -> np.ndarray | None:
    for candidate in candidates:
        if candidate in name_map:
            raw_name = name_map[candidate]
            return np.asarray(las[raw_name])
    return None


def get_planarity(
    las: laspy.LasData,
    name_map: dict[str, str],
    radius: str
) -> np.ndarray | None:
    # first try direct planarity field
    direct_candidates = [
        f"planarity_{radius}",
        f"planarity_{radius}m",
    ]

    planarity = get_field(las, name_map, direct_candidates)
    if planarity is not None:
        return planarity.astype(np.float32)

    # if not present, derive it from eigenvalues
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
    radius: str
) -> np.ndarray | None:
    candidates = [
        f"roughness_{radius}",
        f"roughness_{radius}m",
    ]
    roughness = get_field(las, name_map, candidates)
    if roughness is None:
        return None
    return roughness.astype(np.float32)


def get_height_feature(
    las: laspy.LasData,
    name_map: dict[str, str]
) -> np.ndarray:
    # prefer processed height fields if available
    candidates = [
        "normalized_z",
        "height_above_min",
        "height_division",
        "z",
    ]
    height = get_field(las, name_map, candidates)

    if height is None:
        return np.asarray(las.z).astype(np.float32)

    return height.astype(np.float32)


def get_classification_mask(
    las: laspy.LasData,
    allowed_classes: list[int] | None
) -> np.ndarray:
    if allowed_classes is None:
        return np.ones(len(las.points), dtype=bool)

    class_values = np.asarray(las.classification)
    return np.isin(class_values, allowed_classes)


def save_filtered_cloud(
    las: laspy.LasData,
    mask: np.ndarray,
    output_path: Path
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
    total: int
) -> None:
    summary_path = output_path.with_suffix(".txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sidewalk candidate extraction summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Radius used: {args.radius}\n")
        f.write(f"Planarity minimum: {args.planarity_min}\n")
        f.write(f"Roughness maximum: {args.roughness_max}\n")
        f.write(f"Height percentile: {args.height_max_percentile}\n")
        f.write(f"Height threshold value: {height_threshold:.4f}\n")
        f.write(f"Allowed classes: {args.allowed_classes}\n")
        f.write(f"Points kept: {kept} / {total} ({100 * kept / total:.2f}%)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first-pass sidewalk candidate points"
    )
    parser.add_argument(
        "input",
        help="Path to feature-enriched LAZ/LAS file"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/sidewalk_candidates.laz",
        help="Output LAZ path"
    )
    parser.add_argument(
        "--radius",
        default="0.1",
        help="Feature radius to use, e.g. 0.1, 0.3, 0.5"
    )
    parser.add_argument(
        "--planarity-min",
        type=float,
        default=0.35,
        help="Minimum planarity threshold"
    )
    parser.add_argument(
        "--roughness-max",
        type=float,
        default=0.06,
        help="Maximum roughness threshold"
    )
    parser.add_argument(
        "--height-max-percentile",
        type=float,
        default=65.0,
        help="Keep points below this height percentile"
    )
    parser.add_argument(
        "--allowed-classes",
        nargs="+",
        type=int,
        default=None,
        help="Optional classification labels to keep before filtering"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    las = laspy.read(input_path)

    raw_names = get_dimension_names(las)
    print("\nAvailable dimensions:")
    for name in raw_names:
        print(f" - {name}")

    name_map = build_name_map(las)

    # allow both 0.1 and 0_1 style input
    radius_key = args.radius.replace(".", "_")

    height = get_height_feature(las, name_map)
    planarity = get_planarity(las, name_map, radius_key)
    roughness = get_roughness(las, name_map, radius_key)

    if planarity is None:
        raise ValueError(
            f"Could not find or derive planarity for radius '{args.radius}'."
        )

    if roughness is None:
        raise ValueError(
            f"Could not find roughness for radius '{args.radius}'."
        )

    print("\nFeature summary:")
    print(f" - height min/max: {height.min():.4f} / {height.max():.4f}")
    print(f" - planarity min/max: {planarity.min():.4f} / {planarity.max():.4f}")
    print(f" - roughness min/max: {roughness.min():.4f} / {roughness.max():.4f}")

    class_mask = get_classification_mask(las, args.allowed_classes)
    height_threshold = np.percentile(height[class_mask], args.height_max_percentile)

    print("\nUsing thresholds:")
    print(f" - height <= percentile {args.height_max_percentile} -> {height_threshold:.4f}")
    print(f" - planarity >= {args.planarity_min}")
    print(f" - roughness <= {args.roughness_max}")
    print(f" - allowed classes = {args.allowed_classes}")

    mask = (
        class_mask &
        (height <= height_threshold) &
        (planarity >= args.planarity_min) &
        (roughness <= args.roughness_max)
    )

    kept = int(mask.sum())
    total = int(len(mask))
    print(f"\nKept {kept:,} / {total:,} points ({100 * kept / total:.2f}%)")

    save_filtered_cloud(las, mask, output_path)
    save_summary(input_path, output_path, args, height_threshold, kept, total)

    print(f"Saved sidewalk candidates -> {output_path.resolve()}")
    print(f"Saved summary -> {output_path.with_suffix('.txt').resolve()}")

    # TODO:
    # - better separate sidewalk from road
    # - try kerb / local elevation change features
    # - compare different thresholds across cities
    # - later combine this with classifier output


if __name__ == "__main__":
    main()