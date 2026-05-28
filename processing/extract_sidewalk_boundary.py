"""
Sidewalk Boundary Extraction Tool
Ahmed Bassam Hamdan Almasri — RMIT University
Pointcloud Analysis for Pedestrian Access

This script takes a classified LAZ file and extracts the sidewalk boundary lines.
I built this pipeline in five main stages:

    1. Load & Down Sample   — Reads in LAZ files and retains one data point every 0.25 meters to create a grid.
    2. Remove Noise         — Eliminates random, clustered points likely caused by sensor error or other sources.
    3. Alpha Shape          — Identifies the outer edge of all sidewalk points.
    4. Smart Line Fitting   — Divides sidewalks into sections and uses line-fitting methods to define each section’s two boundaries.
    4a. HFE/KI              — Determines which side of the street is the building frontage (HFE) and which is the kerb (KI).
    4b. Buffer Zones        — Calculates the distance from kerb to road centreline. 
    5. Stitch & Close       — Bridges gaps and completes polygon edges to form complete streets.

Outputs go to ../outputs/:
  sidewalk_boundary_ALL.laz   — closed boundary polygons (label 10)
  sidewalk_HFE.laz            — building-side lines (label 10)
  sidewalk_KI.laz             — kerb-side lines (label 11)
  sidewalk_centreline.laz     — walking centreline between HFE and KI (label 12)
  sidewalk_buffer_zones.csv   — buffer width categories per strip
                                (Not present / Narrow <20cm / Medium 20-70cm / Wide >70cm)

Usage:
  python extract_sidewalk_boundary.py ../data/utrecht_classified.laz
  python extract_sidewalk_boundary.py ../data/utrecht_classified.laz --interactive


CHANGELOG:
    v1 - Initial basic alpha shape boundary extraction code producing a single closed polygon
    v2 - PCA-based slice of sidewalk data to isolate individual left/right edges as separate lines from the original single row
    v3 - Implemented HFE/KI side assignment using street point distance comparison
    After researching other methods (mean distance), I found that the use of the smallest distance provided more reliable results in determining the kerb-side
    v4 - Developed an algorithm to determine whether or not there is a buffer zone, and if so which category it falls under (Not Present/Narrow (< 20 cm)/Medium (20-70 cm)/Wide (> 70 cm)) as per Hendrik’s width specifications document
    v5 - Included a strip merge function to account for sidewalks being broken apart due to scan gaps
    I researched both clustering and direction-matching for merging strips; however, I did not find either approach effective enough
    v6 - Provided an interactive mode with real-time matplotlib preview allowing easier parameter adjustment
    I also tried using RDP Line Simplification and found it to be less desirable than my previous implementation (Rolling Average Smoothing) in terms of performance on curvy areas
    v7 - Improved error checking for missing files and incorrect labels
    Completed final ring closure for more stable polygon output in Stage 5
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SIDEWALK_LABEL = 2
STREET_LABEL   = 11


# --- Stage 1: Load & Downsample ---

def load_sidewalk(path, label=SIDEWALK_LABEL, voxel=0.25):
    print("\n[Stage 1] Loading sidewalk points and downsampling:")

    if not Path(path).exists():
        sys.exit(f"ERROR: File not found: {path}")

    try:
        las = laspy.read(path)
    except Exception as e:
        sys.exit(f"ERROR: Could not read LAZ file: {e}")

    labels = np.asarray(las.classification)

    print("  Labels in the file:")
    for lbl, count in zip(*np.unique(labels, return_counts=True)):
        tag = " <-- sidewalk" if lbl == label else ""
        print(f"    Label {lbl}: {count:,} points{tag}")

    if label not in np.unique(labels):
        sys.exit(f"ERROR: Label {label} not found in file. "
                 f"Check --sidewalk-label matches your ML model output.")

    mask = labels == label
    pts  = np.column_stack((
        np.asarray(las.x)[mask],
        np.asarray(las.y)[mask],
        np.asarray(las.z)[mask]
    ))
    print(f"  Loaded {len(pts):,} sidewalk points")

    if len(pts) < 50:
        sys.exit(f"ERROR: Only {len(pts)} sidewalk points found. "
                 f"Check --sidewalk-label or the quality of the ML classification.")

    # grid-based downsampling — keeps one point per voxel cell, uses median Z
    # this reduces noise from overlapping scan lines without losing coverage
    grid   = np.floor(pts[:, :2] / voxel).astype(np.int64)
    _, inv = np.unique(grid, axis=0, return_inverse=True)

    downsampled = []
    for i in range(inv.max() + 1):
        members = pts[inv == i]
        row     = members[0].copy()
        row[2]  = np.median(members[:, 2])
        downsampled.append(row)

    result = np.array(downsampled)
    print(f"  Downsampled to {len(result):,} points (voxel = {voxel}m)")
    return result


def load_street(path, street_label=STREET_LABEL):
    try:
        las    = laspy.read(path)
        labels = np.asarray(las.classification)
        mask   = labels == street_label
        if mask.sum() == 0:
            print(f"  WARNING: No points with label {street_label} found. "
                  f"HFE/KI assignment may be wrong — check --street-label.")
            return np.empty((0, 3))
        pts = np.column_stack((
            np.asarray(las.x)[mask],
            np.asarray(las.y)[mask],
            np.asarray(las.z)[mask]
        ))
        print(f"  Loaded {len(pts):,} street points (label {street_label})")
        return pts
    except Exception as e:
        print(f"  WARNING: Could not load street points: {e}")
        return np.empty((0, 3))


# --- Stage 2: Noise Removal ---

def remove_noise(pts, dist=5.0, min_pts=30):
    print("\n[Stage 2] Removing noise clusters:")

    if len(pts) < min_pts:
        print("  Too few points, skipping noise removal")
        return pts

    tree    = cKDTree(pts[:, :2])
    visited = np.zeros(len(pts), dtype=bool)
    keep    = []

    for start in range(len(pts)):
        if visited[start]:
            continue
        stack   = [start]
        visited[start] = True
        cluster = []
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for nb in tree.query_ball_point(pts[cur, :2], r=dist):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(cluster) >= min_pts:
            keep.extend(cluster)

    result = pts[keep]
    print(f"  Removed {len(pts) - len(result):,} noise points → {len(result):,} left")
    return result


# --- Stage 3: Alpha Shape ---

def find_edges(pts2d, alpha=0.3):
    print(f"\n[Stage 3] Computing alpha shape (alpha={alpha})")

    # alpha shape uses Delaunay triangulation — keeps only triangles whose
    # circumradius is smaller than 1/alpha. smaller alpha = tighter shape.
    # if it fails or produces too few edges, we fall back to convex hull.
    try:
        tri        = Delaunay(pts2d)
        edge_count = {}

        for simplex in tri.simplices:
            p1, p2, p3 = pts2d[simplex]
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p2 - p3)
            c = np.linalg.norm(p3 - p1)
            s = (a + b + c) / 2.0
            area_sq = s * (s-a) * (s-b) * (s-c)
            if area_sq <= 0:
                continue
            r = (a * b * c) / (4.0 * np.sqrt(area_sq))
            if r < (1.0 / alpha):
                for e in [
                    tuple(sorted((simplex[0], simplex[1]))),
                    tuple(sorted((simplex[1], simplex[2]))),
                    tuple(sorted((simplex[2], simplex[0]))),
                ]:
                    edge_count[e] = edge_count.get(e, 0) + 1

        edges = [e for e, cnt in edge_count.items() if cnt == 1]

        if len(edges) < 8:
            print("  Not enough edges, falling back to convex hull")
            return _hull_edges(pts2d)

        print(f"  Found {len(edges):,} boundary edges")
        return edges

    except Exception as e:
        print(f"  Alpha shape failed ({e}), using convex hull instead")
        return _hull_edges(pts2d)


def _hull_edges(pts2d):
    hull = ConvexHull(pts2d)
    v    = hull.vertices
    return [(v[i], v[(i+1) % len(v)]) for i in range(len(v))]


def filter_edges(pts, edges, min_len=0.1):
    return [(a, b) for a, b in edges
            if np.linalg.norm(pts[a, :2] - pts[b, :2]) >= min_len]


# --- Stage 4: Smart Line Fitting ---

def find_strips(pts, cluster_dist=2.5, min_pts=30):
    print(f"  Clustering points into sidewalk strips (dist={cluster_dist}m)...")

    tree  = cKDTree(pts[:, :2])
    pairs = tree.query_pairs(r=cluster_dist, output_type='ndarray')

    if len(pairs) == 0:
        return [pts]

    n   = len(pts)
    mat = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n)
    )
    n_comp, labels = connected_components(mat, directed=False)

    strips = []
    for i in range(n_comp):
        mask = labels == i
        if mask.sum() >= min_pts:
            strips.append(pts[mask])

    print(f"  Found {len(strips)} strips")
    return strips


def merge_strips(strips, merge_dist=8.0, angle_threshold=0.85):
    """
    Merge strip segments that represent the same walkway but were separated by scanning gaps.
    The decision whether or not to merge two strips is based on both the orientation of the strips, and the length from an end point for each strip.
    The angle threshold determines how parallel the strips have to be in order to merge them (an angle of .85 radians will result in a merged line with a maximum angle of 32 degrees).
    """
    if len(strips) <= 1:
        return strips

    def get_direction(pts):
        centre  = pts[:, :2].mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((pts[:, :2] - centre).T))
        return vecs[:, -1]

    def get_ends(pts):
        centre  = pts[:, :2].mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((pts[:, :2] - centre).T))
        along   = vecs[:, -1]
        proj    = (pts[:, :2] - centre) @ along
        return pts[np.argmin(proj)], pts[np.argmax(proj)]

    result = [s.copy() for s in strips]
    merged = True

    while merged:
        merged     = False
        used       = [False] * len(result)
        new_strips = []

        for i in range(len(result)):
            if used[i]:
                continue
            cur    = result[i]
            dir_i  = get_direction(cur)
            ends_i = get_ends(cur)

            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                other  = result[j]
                dir_j  = get_direction(other)
                ends_j = get_ends(other)

                if abs(np.dot(dir_i, dir_j)) < angle_threshold:
                    continue

                close = any(np.linalg.norm(a[:2] - b[:2]) < merge_dist
                            for a in ends_i for b in ends_j)

                if close:
                    cur    = np.vstack([cur, other])
                    used[j] = True
                    merged  = True
                    dir_i   = get_direction(cur)
                    ends_i  = get_ends(cur)

            new_strips.append(cur)
            used[i] = True

        result = new_strips

    print(f"  After merging: {len(result)} strips")
    return result


def fit_edge(x, y, z_vals, straightness_threshold):
    spread = np.std(y)
    center = np.mean(y)

    if spread < straightness_threshold:
        # side is straight — draw one line at the average lateral position.
        # tried linear regression here first but it still zigzagged slightly
        # when points were uneven. mean position is simpler and cleaner.
        line_pts = np.array([[x[0], center], [x[-1], center]])
        z_out    = np.array([z_vals[0], z_vals[-1]])
        return line_pts, z_out, True
    else:
        # side is curved — apply rolling average to reduce noise but keep shape.
        # tried RDP (Ramer-Douglas-Peucker) simplification here first but it
        # made sharp corners at bends look wrong. rolling average is smoother.
        window  = max(3, len(y) // 10)
        window += (1 - window % 2)
        padded  = np.pad(y, window // 2, mode="edge")
        smoothed = np.array([padded[i:i+window].mean() for i in range(len(y))])
        return np.column_stack([x, smoothed]), z_vals, False


def fit_strip(strip_pts, slice_step, straightness_threshold, edge_percentile):
    if len(strip_pts) < 10:
        return None

    strip_2d = strip_pts[:, :2]
    strip_z  = strip_pts[:, 2]

    centre  = strip_2d.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((strip_2d - centre).T))
    along   = vecs[:, -1]
    across  = vecs[:, -2]

    shifted    = strip_2d - centre
    along_proj = shifted @ along
    side_proj  = shifted @ across

    proj_min, proj_max = along_proj.min(), along_proj.max()
    if proj_max - proj_min < slice_step * 2:
        return None

    cuts = np.arange(proj_min, proj_max + slice_step, slice_step)
    left_pts, right_pts, left_z, right_z = [], [], [], []

    for i in range(len(cuts) - 1):
        mask = (along_proj >= cuts[i]) & (along_proj < cuts[i+1])
        if mask.sum() < 3:
            continue

        ac  = side_proj[mask]
        zs  = strip_z[mask]
        mid = (cuts[i] + cuts[i+1]) / 2.0

        
        edge_hi = np.percentile(ac, 100.0 - edge_percentile)
        edge_lo = np.percentile(ac, edge_percentile)

        left_pts.append([mid, ac[np.argmin(np.abs(ac - edge_hi))]])
        right_pts.append([mid, ac[np.argmin(np.abs(ac - edge_lo))]])
        left_z.append(zs[np.argmin(np.abs(ac - edge_hi))])
        right_z.append(zs[np.argmin(np.abs(ac - edge_lo))])

    if len(left_pts) < 2:
        return None

    left_arr  = np.array(left_pts)
    right_arr = np.array(right_pts)
    left_z_arr  = np.array(left_z)
    right_z_arr = np.array(right_z)

    left_fit,  left_z_out,  left_straight  = fit_edge(left_arr[:,  0], left_arr[:,  1], left_z_arr,  straightness_threshold)
    right_fit, right_z_out, right_straight = fit_edge(right_arr[:, 0], right_arr[:, 1], right_z_arr, straightness_threshold)

    def unproject(local_2d, z_out):
        xy = centre + local_2d[:, 0:1] * along + local_2d[:, 1:2] * across
        return np.column_stack([xy, z_out])

    return (unproject(left_fit,  left_z_out),
            unproject(right_fit, right_z_out),
            left_straight, right_straight)


def fit_lines(pts, slice_step=0.5, straightness=1.5,
              edge_percentile=2.0, cluster_dist=2.5, merge_dist=8.0):
    print("\n[Stage 4] Fitting lines to sidewalk strips:")

    strips = find_strips(pts, cluster_dist=cluster_dist)
    strips = merge_strips(strips, merge_dist=merge_dist)

    all_lines = []
    all_sides = []

    for i, strip in enumerate(strips):
        res = fit_strip(strip, slice_step, straightness, edge_percentile)
        if res is None:
            print(f"  Strip {i}: too small, skipped")
            continue

        line_a, line_b, left_straight, right_straight = res
        print(f"  Strip {i}: {len(strip):,} pts | "
              f"left={'straight' if left_straight else 'curved'} | "
              f"right={'straight' if right_straight else 'curved'}")

        closed = np.vstack([line_a, line_b[::-1], line_a[[0]]])
        all_lines.append(closed)
        all_sides.append((line_a, line_b))

    if not all_lines:
        print("  Warning: No valid lines generated")
        return None, []

    print(f"  Generated {len(all_lines)} closed boundaries")
    return all_lines, all_sides


# --- Stage 4b: HFE / KI Assignment ---

def label_sides(all_sides, street_pts):
    print("\n[Stage 4b] Assigning HFE (building) and KI (kerb) sides:")

    if len(street_pts) == 0:
        print("  No street points available, skipping side assignment")
        return [], []

    tree      = cKDTree(street_pts[:, :2])
    hfe_lines = []
    ki_lines  = []

    for i, (line_a, line_b) in enumerate(all_sides):
        # that were far from the road, pulling the average up even though the
        # line was clearly on the road side. min() correctly identifies the
        # side that actually touches the road edge.
        min_dist_a = tree.query(line_a[:, :2], k=1)[0].min()
        min_dist_b = tree.query(line_b[:, :2], k=1)[0].min()

        if min_dist_a < min_dist_b:
            ki, hfe = line_a, line_b
        else:
            ki, hfe = line_b, line_a

        hfe_lines.append(hfe)
        ki_lines.append(ki)
        print(f"  Strip {i}: KI min dist = {min(min_dist_a, min_dist_b):.2f}m | "
              f"HFE min dist = {max(min_dist_a, min_dist_b):.2f}m")

    return hfe_lines, ki_lines


# --- Stage 4c: Buffer Zones ---

def calc_buffers(ki_lines, street_pts):
    print("\n[Stage 4c] Calculating buffer zones between kerb and road:")

    if len(street_pts) == 0:
        print("  No street data, skipping buffer calculation")
        return []

    tree    = cKDTree(street_pts[:, :2])
    results = []

    for i, ki in enumerate(ki_lines):
        # minimum distance from any KI point to the road
        # this gives the actual physical gap (buffer zone width)
        dists       = tree.query(ki[:, :2], k=1)[0]
        min_dist_m  = dists.min()
        min_dist_cm = min_dist_m * 100

        if min_dist_cm < 5:
            category = "Not present"
        elif min_dist_cm < 20:
            category = "Narrow (<20cm)"
        elif min_dist_cm <= 70:
            category = "Medium (20-70cm)"
        else:
            category = "Wide (>70cm)"

        print(f"  Strip {i}: {min_dist_cm:.1f}cm => {category}")
        results.append({
            "strip":       i,
            "min_dist_cm": round(min_dist_cm, 1),
            "category":    category,
        })

    return results


# --- Stage 4d: Walking Centreline ---

def compute_centrelines(hfe_lines, ki_lines):
    """
    For each strip, compute the walking centreline as the midpoint
    between the HFE and KI lines at regular intervals.
    This is what Sujeeth needs for width measurements along the walking direction.
    """
    print("\n[Stage 4d] Computing walking centrelines...")

    centrelines = []

    for i, (hfe, ki) in enumerate(zip(hfe_lines, ki_lines)):
        # take midpoints directly
        n = max(len(hfe), len(ki))

        def resample(line, n):
            # spread n evenly spaced points along the line
            dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))])
            total = dists[-1]
            if total < 1e-9:
                return line
            targets = np.linspace(0, total, n)
            resampled = np.zeros((n, 3))
            for k in range(3):
                resampled[:, k] = np.interp(targets, dists, line[:, k])
            return resampled

        hfe_r = resample(hfe, n)
        ki_r  = resample(ki,  n)

        # midpoint between the two lines
        centre = (hfe_r + ki_r) / 2.0
        centrelines.append(centre)
        print(f"  Strip {i}: centreline with {n} points")

    return centrelines


def save_buffer_csv(buffer_results, out_path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strip", "min_dist_cm", "category"])
        writer.writeheader()
        writer.writerows(buffer_results)
    print(f"  Saved buffer zones to {out_path.name}")


# --- Stage 5: Stitch & Close ---

def stitch_lines(polys, close_gap=8.0, stitch_gap=6.0):
    print(f"\n[Stage 5] Stitching and closing boundaries:")

    result = [c.copy() for c in polys]

    # first pass: close any rings where start and end are nearly touching
    for i, poly in enumerate(result):
        if 0 < np.linalg.norm(poly[0, :2] - poly[-1, :2]) < close_gap:
            result[i] = np.vstack([poly, poly[0]])

    # second pass: stitch nearby open endpoints across different polylines
    for _ in range(20):
        merged = False
        used   = [False] * len(result)
        new    = []

        for i in range(len(result)):
            if used[i]:
                continue
            segment = result[i]

            if np.linalg.norm(segment[0, :2] - segment[-1, :2]) < 0.5:
                new.append(segment)
                used[i] = True
                continue

            match_j, match_dist, join_mode = -1, stitch_gap, None
            for j in range(len(result)):
                if i == j or used[j]:
                    continue
                other     = result[j]
                is_closed = np.linalg.norm(other[0, :2] - other[-1, :2]) < 0.5

                candidates = [
                    (np.linalg.norm(segment[-1, :2] - other[0, :2]),  "es"),
                    (np.linalg.norm(segment[-1, :2] - other[-1, :2]), "ee") if not is_closed else (stitch_gap, ""),
                    (np.linalg.norm(segment[0, :2]  - other[0, :2]),  "ss") if not is_closed else (stitch_gap, ""),
                    (np.linalg.norm(segment[0, :2]  - other[-1, :2]), "se") if not is_closed else (stitch_gap, ""),
                ]

                for d, mode in candidates:
                    if mode and d < match_dist:
                        match_dist, match_j, join_mode = d, j, mode

            if match_j >= 0:
                other = result[match_j]
                if join_mode == "es":   segment = np.vstack([segment, other])
                elif join_mode == "ee": segment = np.vstack([segment, other[::-1]])
                elif join_mode == "ss": segment = np.vstack([segment[::-1], other])
                elif join_mode == "se": segment = np.vstack([other, segment])
                used[match_j] = True
                merged = True

            new.append(segment)
            used[i] = True

        result = new
        if not merged:
            break

    # final pass: close any rings that are now within gap after stitching
    for i, c in enumerate(result):
        if 0 < np.linalg.norm(c[0, :2] - c[-1, :2]) < close_gap:
            result[i] = np.vstack([c, c[0]])

    print(f"  Final boundaries: {len(result)}")
    return result


# --- Save as LAZ ---

def save_laz(polys, out_path, source_las, label=10, spacing=0.05):
    def densify(coords):
        pts = [coords[0]]
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i+1]
            seg  = np.linalg.norm(b - a)
            if seg < 1e-9:
                continue
            n = max(1, int(np.ceil(seg / spacing)))
            for t in np.linspace(0, 1, n, endpoint=False)[1:]:
                pts.append(a + t * (b - a))
        pts.append(coords[-1])
        return np.array(pts)

    all_pts = np.vstack([densify(c) for c in polys])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header         = laspy.LasHeader(version="1.2", point_format=0)
    header.offsets = source_las.header.offsets
    header.scales  = source_las.header.scales

    out                = laspy.LasData(header=header)
    out.x              = all_pts[:, 0]
    out.y              = all_pts[:, 1]
    out.z              = all_pts[:, 2]
    out.classification = np.full(len(all_pts), label, dtype=np.uint8)
    out.write(str(out_path))
    print(f"  Saved {out_path.name} ({len(all_pts):,} points, label {label})")


# --- Debug Preview ---

def show_preview(pts, raw_edges, final_polys, hfe_lines=None, ki_lines=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Sidewalk Boundary Extraction Preview")

    axes[0].set_title("Raw Alpha Shape Edges")
    axes[0].scatter(pts[:, 0], pts[:, 1], s=0.5, c="red", alpha=0.3)
    for a, b in raw_edges:
        axes[0].plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                     lw=0.5, c="steelblue", alpha=0.6)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X (metres)")
    axes[0].set_ylabel("Y (metres)")

    axes[1].set_title("Final Boundaries")
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.5, c="red", alpha=0.3)
    for c in final_polys:
        axes[1].plot(c[:, 0], c[:, 1], lw=1.5, color="lime")
    if hfe_lines:
        for c in hfe_lines:
            axes[1].plot(c[:, 0], c[:, 1], lw=2.0, color="dodgerblue")
    if ki_lines:
        for c in ki_lines:
            axes[1].plot(c[:, 0], c[:, 1], lw=2.0, color="orange")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("X (metres)")
    axes[1].set_ylabel("Y (metres)")
    axes[1].text(0.02, 0.98, "Green = boundary\nBlue  = HFE (building)\nOrange = KI (kerb)",
                 transform=axes[1].transAxes, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


# --- Main Pipeline ---

def run_pipeline(input_path, params):
    source_las = laspy.read(input_path)
    pts        = load_sidewalk(input_path, voxel=params["voxel_size"])
    street_pts = load_street(input_path, params["street_label"])

    pts   = remove_noise(pts)
    edges = find_edges(pts[:, :2], alpha=params["alpha"])
    edges = filter_edges(pts, edges)

    all_lines, all_sides = fit_lines(
        pts,
        slice_step=params["slice_step"],
        straightness=params["straightness"],
        edge_percentile=params["edge_percentile"],
        cluster_dist=params["cluster_dist"],
        merge_dist=params["merge_dist"]
    )

    result         = all_lines if all_lines is not None else []
    hfe_lines, ki_lines = label_sides(all_sides, street_pts)
    centrelines    = compute_centrelines(hfe_lines, ki_lines)
    buffer_results = calc_buffers(ki_lines, street_pts)

    return source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results


# --- Interactive Mode ---

def interactive_loop(input_path, out_dir, params):
    print("\n" + "="*65)
    print("          INTERACTIVE MODE - tweak parameters live")
    print("="*65)

    while True:
        source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results = run_pipeline(input_path, params)

        print("\nCurrent parameters:")
        for k, v in params.items():
            print(f"   {k:<18} = {v}")

        print("\nOptions:")
        print("   1 -> Show preview")
        print("   2 -> Change parameter")
        print("   3 -> Save outputs and exit")
        print("   4 -> Exit without saving")
        choice = input("   Enter choice: ").strip()

        if choice == "1":
            show_preview(pts, edges, result, hfe_lines, ki_lines)

        elif choice == "2":
            print("\nWhich parameter?")
            print("   1 alpha          2 straightness     3 cluster_dist")
            print("   4 merge_dist     5 slice_step       6 edge_percentile")
            print("   7 voxel_size")
            p = input("   Choice: ").strip()

            param_map = {
                "1": "alpha",      "2": "straightness",   "3": "cluster_dist",
                "4": "merge_dist", "5": "slice_step",     "6": "edge_percentile",
                "7": "voxel_size"
            }
            if p in param_map:
                key     = param_map[p]
                new_val = input(f"   New value for {key} (current={params[key]}): ").strip()
                if new_val:
                    params[key] = float(new_val)
            else:
                print("   Invalid option")

        elif choice == "3":
            save_all_outputs(result, hfe_lines, ki_lines, centrelines, buffer_results, out_dir, source_las)
            print("Files saved.")
            break

        elif choice == "4":
            print("   Exiting without saving.")
            break


def save_all_outputs(result, hfe_lines, ki_lines, centrelines, buffer_results, out_dir, source_las):
    print("\nSaving all outputs...")
    if result:
        save_laz(result,      out_dir / "sidewalk_boundary_ALL.laz", source_las, label=10)
    if hfe_lines:
        save_laz(hfe_lines,   out_dir / "sidewalk_HFE.laz",          source_las, label=10)
    if ki_lines:
        save_laz(ki_lines,    out_dir / "sidewalk_KI.laz",           source_las, label=11)
    if centrelines:
        save_laz(centrelines, out_dir / "sidewalk_centreline.laz",   source_las, label=12)
    if buffer_results:
        save_buffer_csv(buffer_results, out_dir / "sidewalk_buffer_zones.csv")


# --- Entry Point ---

def main():

    parser = argparse.ArgumentParser(description="Sidewalk Boundary Extraction")
    parser.add_argument("--input-processing",help="Path to classified LAZ file")
    parser.add_argument("--voxel-size",        type=float, default=0.25)
    parser.add_argument("--alpha",             type=float, default=0.3)
    parser.add_argument("--slice-step",        type=float, default=0.5)
    parser.add_argument("--straightness",      type=float, default=1.5)
    parser.add_argument("--edge-percentile",   type=float, default=2.0)
    parser.add_argument("--cluster-dist",      type=float, default=2.5)
    parser.add_argument("--merge-dist",        type=float, default=8.0)
    parser.add_argument("--street-label",      type=int,   default=STREET_LABEL)
    parser.add_argument("--interactive",       action="store_true")
    parser.add_argument("--boundary-city", default=None)
    args = parser.parse_args()

    params = {
        "input_processing": args.input_processing,
        "voxel_size":      args.voxel_size,
        "alpha":           args.alpha,
        "slice_step":      args.slice_step,
        "straightness":    args.straightness,
        "edge_percentile": args.edge_percentile,
        "cluster_dist":    args.cluster_dist,
        "merge_dist":      args.merge_dist,
        "street_label":    args.street_label,
    }

    print("=== Sidewalk Boundary Extraction Tool ===")
    
    out_dir = Path(f"outputs/{args.boundary_city}")

    if args.interactive:
        interactive_loop(args.input_processing, out_dir, params)
    else:
        source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results = run_pipeline(args.input_processing, params)
        save_all_outputs(result, hfe_lines, ki_lines, centrelines, buffer_results, out_dir, source_las)
        print("Done")


if __name__ == "__main__":
    main()