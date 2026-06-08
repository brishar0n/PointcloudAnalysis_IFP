"""
highlight_sidewalk.py
=====================
Visualisation helper – Sidewalk Highlighting
Author : Ahmed Bassam Hamdan Almasri

Takes:
  • The original LAZ  (full RGB colours, no classification)
  • The classified LAZ (same points, with classification labels from ML)

Matches the two clouds by XYZ coordinates (they are the same points),
then writes a new LAZ containing ALL original points in their original
RGB — except sidewalk points, which are forced to neon green so they
are clearly visible in CloudCompare.

Dependencies (install once):
    pip install laspy[lazrs] numpy scipy

Usage (run from the  processing/  folder):
    python highlight_sidewalk.py
    python highlight_sidewalk.py --sidewalk-label 2
"""

import argparse
import sys
from pathlib import Path

try:
    import laspy
except ImportError:
    sys.exit("ERROR: laspy not installed.\nRun:  pip install laspy[lazrs]")

import numpy as np
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PATHS  ░░
# ─────────────────────────────────────────────────────────────────────────────
INPUT_ORIGINAL   = Path(
    r"C:\Users\1ahme\OneDrive\Desktop\lidar_sidewalks"
    r"\lidar_sidewalks-main\data\utrecht_subsampled.laz"
)
INPUT_CLASSIFIED = Path(
    r"C:\Users\1ahme\OneDrive\Desktop\lidar_sidewalks"
    r"\PointcloudAnalysis_IFP\data\utrecht_classified.laz"
)
OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "utrecht_sidewalk_highlighted.laz"

# ─────────────────────────────────────────────────────────────────────────────
# ░░  CONFIG  ░░
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SIDEWALK_LABEL = 2

# Neon green in 16-bit LAS colour space (0–65535)
NEON_GREEN = (0, 65535, 0)


# ─────────────────────────────────────────────────────────────────────────────
# ░░  MAIN  ░░
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Highlight classified sidewalk points neon green in the original LAZ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sidewalk-label", type=int, default=DEFAULT_SIDEWALK_LABEL,
                   help="Classification value for sidewalk in the ML output file")
    return p.parse_args()


def main():
    args = parse_args()

    for p in (INPUT_ORIGINAL, INPUT_CLASSIFIED):
        if not p.exists():
            sys.exit(f"ERROR: File not found:\n  {p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Sidewalk Highlighter")
    print(f"  Original   : {INPUT_ORIGINAL.name}")
    print(f"  Classified : {INPUT_CLASSIFIED.name}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"{'─'*60}")

    # 1. load original ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading original LAZ …")
    orig_las = laspy.read(str(INPUT_ORIGINAL))
    orig_xyz = np.column_stack([
        np.asarray(orig_las.x, dtype=np.float64),
        np.asarray(orig_las.y, dtype=np.float64),
        np.asarray(orig_las.z, dtype=np.float64),
    ])
    print(f"  Points : {len(orig_xyz):,}")

    # grab original RGB (16-bit)
    try:
        rgb = np.column_stack([
            np.asarray(orig_las.red,   dtype=np.uint16),
            np.asarray(orig_las.green, dtype=np.uint16),
            np.asarray(orig_las.blue,  dtype=np.uint16),
        ])
        print(f"  RGB    : found ✓")
    except AttributeError:
        print("  WARNING: No RGB in original file — output will be grey + green.")
        grey = np.full(len(orig_xyz), 32768, dtype=np.uint16)
        rgb  = np.column_stack([grey, grey, grey])

    # 2. load classified ───────────────────────────────────────────────────────
    print("\n[2/4] Loading classified LAZ …")
    cls_las = laspy.read(str(INPUT_CLASSIFIED))
    cls_xyz = np.column_stack([
        np.asarray(cls_las.x, dtype=np.float64),
        np.asarray(cls_las.y, dtype=np.float64),
        np.asarray(cls_las.z, dtype=np.float64),
    ])
    cls_labels = np.asarray(cls_las.classification, dtype=np.int32)
    print(f"  Points : {len(cls_xyz):,}")

    unique, counts = np.unique(cls_labels, return_counts=True)
    print("  Labels:")
    for u, c in zip(unique, counts):
        tag = " ← sidewalk" if u == args.sidewalk_label else ""
        print(f"    label {u:3d}  →  {c:,} pts{tag}")

    # 3. match sidewalk points to original by XYZ ──────────────────────────────
    # The classified file is the same point cloud — coordinates are identical.
    # Build a KDTree on the original, query each classified sidewalk point,
    # and take its nearest neighbour (will be the exact same point).
    print("\n[3/4] Matching sidewalk points to original cloud …")

    sw_mask = cls_labels == args.sidewalk_label
    n_sw    = sw_mask.sum()
    if n_sw == 0:
        sys.exit(f"ERROR: No points with label {args.sidewalk_label} found.")
    print(f"  Sidewalk points to highlight : {n_sw:,}")

    sw_xyz = cls_xyz[sw_mask]

    tree = cKDTree(orig_xyz)
    _, indices = tree.query(sw_xyz, k=1, workers=-1)

    # Mark those indices in the colour array
    rgb[indices, 0] = NEON_GREEN[0]
    rgb[indices, 1] = NEON_GREEN[1]
    rgb[indices, 2] = NEON_GREEN[2]

    print(f"  Highlighted : {len(indices):,} points → neon green")

    # 4. write output LAZ ──────────────────────────────────────────────────────
    print("\n[4/4] Writing output LAZ …")

    header = laspy.LasHeader(version="1.2", point_format=2)  # format 2 = XYZ + RGB
    header.offsets = orig_las.header.offsets
    header.scales  = orig_las.header.scales

    out       = laspy.LasData(header=header)
    out.x     = orig_xyz[:, 0]
    out.y     = orig_xyz[:, 1]
    out.z     = orig_xyz[:, 2]
    out.red   = rgb[:, 0]
    out.green = rgb[:, 1]
    out.blue  = rgb[:, 2]

    out.write(str(OUTPUT_PATH))
    print(f"  → saved  {OUTPUT_PATH}")

    print(f"""
  ┌─ Summary ─────────────────────────────────────────┐
  │  Total points            : {len(orig_xyz):>10,}             │
  │  Sidewalk (neon green)   : {len(indices):>10,}             │
  │  Other   (original RGB)  : {len(orig_xyz) - len(indices):>10,}             │
  └───────────────────────────────────────────────────┘

  In CloudCompare:
    File → Open → utrecht_sidewalk_highlighted.laz
    Make sure display is set to RGB colours.
Done.
""")


if __name__ == "__main__":
    main()