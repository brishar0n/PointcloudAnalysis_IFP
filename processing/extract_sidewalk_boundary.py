"""
Sidewalk Boundary Extraction Module
Author: Ahmed Bassam Hamdan Almasri
RMIT University
Final Project - Pointcloud Analysis for Pedestrian Access
"""
from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

SIDEWALK_LABEL = 2


def load_classified_sidewalk_points(path: Path, sidewalk_label: int = SIDEWALK_LABEL) -> np.ndarray:
    las = laspy.read(path)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    labels = np.asarray(las.classification)
    mask = labels == sidewalk_label
    points = np.column_stack((x[mask], y[mask], z[mask]))
    print(f"Loaded {len(points):,} sidewalk points (label {sidewalk_label})")
    if len(points) < 50:
        raise ValueError("Not enough sidewalk points.")
    return points


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.25) -> np.ndarray:
    grid = np.floor(points[:, :2] / voxel_size).astype(np.int64)
    unique_cells, inverse = np.unique(grid, axis=0, return_inverse=True)
    result = np.empty((len(unique_cells), 3), dtype=points.dtype)
    for i in range(len(unique_cells)):
        members = points[inverse == i]
        result[i, :2] = members[0, :2]
        result[i, 2] = np.median(members[:, 2])
    print(f"Downsampled from {len(points)} to {len(result)} points")
    return result


def filter_small_clusters(points: np.ndarray, distance: float = 1.5, min_points: int = 100) -> np.ndarray:
    if len(points) < min_points:
        print("Not enough points for clustering, skipping...")
        return points
    tree = cKDTree(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    kept_indices = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cluster = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            for n in tree.query_ball_point(points[current, :2], r=distance):
                if not visited[n]:
                    visited[n] = True
                    stack.append(n)
        if len(cluster) >= min_points:
            kept_indices.extend(cluster)
    kept_points = points[kept_indices]
    print(f"After cleaning noise: {len(kept_points):,} points left")
    return kept_points


def extract_boundaries(points_2d: np.ndarray, alpha: float = 0.3) -> list[tuple[int, int]]:
    try:
        tri = Delaunay(points_2d)
        edge_count = {}
        for simplex in tri.simplices:
            p1, p2, p3 = points_2d[simplex]
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p2 - p3)
            c = np.linalg.norm(p3 - p1)
            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                continue
            area = np.sqrt(area_sq)
            circum_r = (a * b * c) / (4.0 * area)
            if circum_r < (1.0 / alpha):
                for edge in [
                    tuple(sorted((simplex[0], simplex[1]))),
                    tuple(sorted((simplex[1], simplex[2]))),
                    tuple(sorted((simplex[2], simplex[0]))),
                ]:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
        boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        if len(boundary_edges) < 8:
            print("Alpha shape not great, switching to Convex Hull...")
            return _convex_hull_edges(points_2d)
        return boundary_edges
    except Exception as e:
        print(f"Alpha shape had problem ({e}), using Convex Hull instead")
        return _convex_hull_edges(points_2d)


def _convex_hull_edges(points_2d: np.ndarray) -> list[tuple[int, int]]:
    hull = ConvexHull(points_2d)
    verts = hull.vertices
    return [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]


def remove_short_edges(points: np.ndarray, edges: list, min_length: float = 0.1):
    cleaned = []
    for a, b in edges:
        if np.linalg.norm(points[a, :2] - points[b, :2]) >= min_length:
            cleaned.append((a, b))
    return cleaned


def edges_to_polylines(edges: list) -> list[list[int]]:
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    unused = {tuple(sorted(e)) for e in edges}
    polylines = []
    while unused:
        a, b = unused.pop()
        line = [a, b]
        while True:
            curr = line[-1]
            prev = line[-2]
            options = [n for n in adjacency.get(curr, [])
                       if n != prev and tuple(sorted((curr, n))) in unused]
            if not options:
                break
            nxt = options[0]
            unused.discard(tuple(sorted((curr, nxt))))
            line.append(nxt)
        while True:
            curr = line[0]
            prev = line[1]
            options = [n for n in adjacency.get(curr, [])
                       if n != prev and tuple(sorted((curr, n))) in unused]
            if not options:
                break
            nxt = options[0]
            unused.discard(tuple(sorted((curr, nxt))))
            line.insert(0, nxt)
        if len(line) >= 3:
            polylines.append(line)
    return polylines


def smooth_polyline(coords: np.ndarray, window: int = 25) -> np.ndarray:
    if len(coords) < window:
        return coords
    half = window // 2
    out = coords.copy()
    is_closed = np.linalg.norm(coords[0, :2] - coords[-1, :2]) < 0.5
    pad_mode = "wrap" if is_closed else "edge"
    for col in range(2):
        padded = np.pad(coords[:, col], half, mode=pad_mode)
        out[:, col] = np.array([padded[i: i + window].mean() for i in range(len(coords))])
    return out


def rdp_simplify(coords: np.ndarray, epsilon: float) -> np.ndarray:
    if len(coords) < 3:
        return coords
    pts = coords[:, :2]
    start, end = pts[0], pts[-1]
    d = end - start
    denom = np.linalg.norm(d)
    dists = (np.linalg.norm(pts - start, axis=1) if denom < 1e-12
             else np.abs(np.cross(pts - start, d)) / denom)
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = rdp_simplify(coords[: idx + 1], epsilon)
        right = rdp_simplify(coords[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([coords[0], coords[-1]])


def close_and_stitch_polylines(coord_list: list[np.ndarray], close_gap: float = 8.0, stitch_gap: float = 6.0) -> list[np.ndarray]:
    result = [c.copy() for c in coord_list]
    for i, coords in enumerate(result):
        gap = np.linalg.norm(coords[0, :2] - coords[-1, :2])
        if 0 < gap < close_gap:
            result[i] = np.vstack([coords, coords[0]])
    for _ in range(20):
        merged = False
        used = [False] * len(result)
        new_result = []
        for i in range(len(result)):
            if used[i]:
                continue
            current = result[i]
            if np.linalg.norm(current[0, :2] - current[-1, :2]) < 0.5:
                new_result.append(current)
                used[i] = True
                continue
            best_j, best_d, best_mode = -1, stitch_gap, None
            for j in range(len(result)):
                if i == j or used[j]:
                    continue
                other = result[j]
                other_closed = np.linalg.norm(other[0, :2] - other[-1, :2]) < 0.5
                d = np.linalg.norm(current[-1, :2] - other[0, :2])
                if d < best_d:
                    best_d, best_j, best_mode = d, j, "es"
                if not other_closed:
                    d = np.linalg.norm(current[-1, :2] - other[-1, :2])
                    if d < best_d:
                        best_d, best_j, best_mode = d, j, "ee"
                if not other_closed:
                    d = np.linalg.norm(current[0, :2] - other[0, :2])
                    if d < best_d:
                        best_d, best_j, best_mode = d, j, "ss"
                if not other_closed:
                    d = np.linalg.norm(current[0, :2] - other[-1, :2])
                    if d < best_d:
                        best_d, best_j, best_mode = d, j, "se"
            if best_j >= 0:
                other = result[best_j]
                if best_mode == "es":
                    current = np.vstack([current, other])
                elif best_mode == "ee":
                    current = np.vstack([current, other[::-1]])
                elif best_mode == "ss":
                    current = np.vstack([current[::-1], other])
                elif best_mode == "se":
                    current = np.vstack([other, current])
                used[best_j] = True
                merged = True
            new_result.append(current)
            used[i] = True
        result = new_result
        if not merged:
            break
    for i, coords in enumerate(result):
        gap = np.linalg.norm(coords[0, :2] - coords[-1, :2])
        if 0 < gap < close_gap:
            result[i] = np.vstack([coords, coords[0]])
    print(f"  After stitch+close: {len(result)} polylines")
    return result


def smooth_and_simplify_polylines(points: np.ndarray, polylines: list[list[int]], smooth_window: int, rdp_epsilon: float, min_length: float = 15.0) -> list[np.ndarray]:
    result = []
    for line in polylines:
        coords = points[line]
        seg_lengths = np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1)
        if seg_lengths.sum() < min_length:
            continue
        gap = np.linalg.norm(coords[0, :2] - coords[-1, :2])
        if gap < 2.0 and not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        coords = smooth_polyline(coords, smooth_window)
        coords = rdp_simplify(coords, rdp_epsilon)
        if len(coords) >= 2:
            result.append(coords)
    print(f"  {len(polylines)} polylines → {len(result)} kept → {sum(len(c) for c in result):,} vertices")
    return result


def densify(coords: np.ndarray, spacing: float) -> np.ndarray:
    pts = [coords[0]]
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        seg_len = np.linalg.norm(b - a)
        if seg_len < 1e-9:
            continue
        n = max(1, int(np.ceil(seg_len / spacing)))
        for t in np.linspace(0, 1, n, endpoint=False)[1:]:
            pts.append(a + t * (b - a))
    pts.append(coords[-1])
    return np.array(pts)


def save_laz(coord_list: list[np.ndarray], output_path: Path, source_las: laspy.LasData, classification: int, spacing: float = 0.05):
    all_pts = [densify(c, spacing) for c in coord_list]
    if not all_pts:
        return
    all_pts = np.vstack(all_pts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(version="1.2", point_format=0)
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales
    out = laspy.LasData(header=header)
    out.x = all_pts[:, 0]
    out.y = all_pts[:, 1]
    out.z = all_pts[:, 2]
    out.classification = np.full(len(all_pts), classification, dtype=np.uint8)
    out.write(str(output_path))
    print(f"  → {output_path.name}  ({len(all_pts):,} pts)")


def main():
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _OUT_DIR = _SCRIPT_DIR.parent / "outputs"

    parser = argparse.ArgumentParser(description="Sidewalk Boundary Extraction")
    parser.add_argument("input",           help="Classified LAZ file")
    parser.add_argument("--voxel-size",    type=float, default=0.25)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--smooth-window", type=int,   default=25)
    parser.add_argument("--rdp-epsilon",   type=float, default=0.5)
    parser.add_argument("--min-length",    type=float, default=15.0)
    parser.add_argument("--close-gap",     type=float, default=8.0)
    parser.add_argument("--stitch-gap",    type=float, default=6.0)

    args = parser.parse_args()

    print("=== Starting Sidewalk Boundary Extraction ===")

    source_las = laspy.read(args.input)
    points     = load_classified_sidewalk_points(Path(args.input))
    points     = voxel_downsample(points, args.voxel_size)
    points     = filter_small_clusters(points, distance=5.0, min_points=30)

    edges     = extract_boundaries(points[:, :2], args.alpha)
    edges     = remove_short_edges(points, edges)
    polylines = edges_to_polylines(edges)

    smooth_all = smooth_and_simplify_polylines(points, polylines, args.smooth_window, args.rdp_epsilon, args.min_length)
    smooth_all = close_and_stitch_polylines(smooth_all, close_gap=args.close_gap, stitch_gap=args.stitch_gap)

    print("\nSaving output ...")
    save_laz(smooth_all, _OUT_DIR / "sidewalk_boundary_ALL.laz", source_las, classification=10)
    print("\nDone! Output: sidewalk_boundary_ALL.laz")


if __name__ == "__main__":
    main()