from __future__ import annotations
import argparse
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

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
        print(f"    Label {lbl}: {count:,} points")

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
    grid        = np.floor(pts[:, :2] / voxel).astype(np.int64)
    _, inv, cnt = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    n_cells     = cnt.shape[0]
    xy_sum      = np.zeros((n_cells, 2))
    z_vals      = [[] for _ in range(n_cells)]
    np.add.at(xy_sum, inv, pts[:, :2])
    for idx, cell in enumerate(inv):
        z_vals[cell].append(pts[idx, 2])
    result        = np.zeros((n_cells, 3))
    result[:, :2] = xy_sum / cnt[:, None]
    result[:, 2]  = np.array([np.median(z) for z in z_vals])

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


def filter_edges(pts, edges, min_len=0.1, max_len=3.0):
    return [(a, b) for a, b in edges
            if min_len <= np.linalg.norm(pts[a, :2] - pts[b, :2]) <= max_len]


# --- Stage 4: Classify Boundary Edges ---

def chain_edges(edges, pts):
    """Walk connected alpha shape edges into ordered polylines."""
    from collections import defaultdict

    if not edges:
        return []

    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    all_nodes   = set(adj.keys())
    visited     = set()
    chains      = []
    endpoints   = [n for n in all_nodes if len(adj[n]) == 1]
    start_queue = endpoints + [n for n in all_nodes if len(adj[n]) != 1]

    for start in start_queue:
        if start in visited:
            continue
        chain   = [start]
        visited.add(start)
        current = start
        while True:
            nxt = [n for n in adj[current] if n not in visited]
            if not nxt:
                break
            current = nxt[0]
            chain.append(current)
            visited.add(current)
        if len(chain) >= 2:
            chains.append(pts[np.array(chain)])

    return chains


def smooth_chains(edges, pts, window=5, min_component=10):
    """
    Chain alpha shape edges into one polyline per connected component,
    then apply a rolling average to smooth out the wiggle.
    """
    from collections import defaultdict
    print("\n[Stage 4a] Chaining and smoothing alpha shape edges:")

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited    = set()
    components = []
    for start in adj:
        if start in visited:
            continue
        comp  = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(adj[node] - visited)
        if len(comp) >= min_component:
            components.append(comp)

    w = max(3, window)
    w += (1 - w % 2)

    smoothed = []
    for comp in components:
        comp_edges  = [(a, b) for a, b in edges if a in comp]
        comp_chains = chain_edges(comp_edges, pts)
        for chain in comp_chains:
            if len(chain) < w:
                smoothed.append(chain)
                continue
            out = chain.copy()
            for col in range(2):
                padded      = np.pad(chain[:, col], w // 2, mode='edge')
                out[:, col] = np.array([padded[i:i+w].mean() for i in range(len(chain))])
            smoothed.append(out)

    print(f"  {len(smoothed)} boundary chains (smoothing window={w})")
    return smoothed


def classify_boundary(pts, edges, street_pts, min_component=10):
    """
    Split alpha shape edges into HFE (building-side) and KI (kerb-side) polylines.
    Processes each connected component (= one sidewalk strip) independently so
    edges from different strips are never mixed into the same chain.
    """
    from collections import defaultdict

    print("\n[Stage 4] Classifying boundary into HFE / KI:")

    if not edges or len(street_pts) == 0:
        print("  No edges or street points — cannot classify")
        return [], []

    tree = cKDTree(street_pts[:, :2])

    # Find connected components of the alpha shape so each strip is isolated
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited    = set()
    components = []
    for start in adj:
        if start in visited:
            continue
        comp  = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(adj[node] - visited)
        if len(comp) >= min_component:
            components.append(comp)

    print(f"  Found {len(components)} boundary components")

    matched_hfe, matched_ki = [], []

    for i, comp_nodes in enumerate(components):
        comp_edges = [(a, b) for a, b in edges if a in comp_nodes]

        dists = np.array([
            tree.query((pts[a, :2] + pts[b, :2]) / 2.0, k=1)[0]
            for a, b in comp_edges
        ])

        # 2-cluster k-means on distances — finds natural KI/HFE groups
        # regardless of whether the two sides have equal numbers of edges
        centers = np.array([dists.min(), dists.max()])
        for _ in range(20):
            labels  = np.argmin(np.abs(dists[:, None] - centers), axis=1)
            new_c   = np.array([dists[labels == j].mean() if (labels == j).any()
                                 else centers[j] for j in range(2)])
            if np.allclose(centers, new_c):
                break
            centers = new_c

        ki_cluster = int(np.argmin(centers))
        ki_edges   = [e for e, l in zip(comp_edges, labels) if l == ki_cluster]
        hfe_edges  = [e for e, l in zip(comp_edges, labels) if l != ki_cluster]

        ki_chains  = chain_edges(ki_edges,  pts)
        hfe_chains = chain_edges(hfe_edges, pts)

        if not ki_chains or not hfe_chains:
            continue

        ki_best  = max(ki_chains,  key=len)
        hfe_best = max(hfe_chains, key=len)

        # Final sanity check — if KI ended up further from road than HFE, swap them
        ki_mean  = tree.query(ki_best[:,  :2], k=1)[0].mean()
        hfe_mean = tree.query(hfe_best[:, :2], k=1)[0].mean()
        if ki_mean > hfe_mean:
            ki_best, hfe_best = hfe_best, ki_best
            ki_mean, hfe_mean = hfe_mean, ki_mean

        matched_hfe.append(hfe_best)
        matched_ki.append(ki_best)
        print(f"  Strip {i}: KI avg={ki_mean:.2f}m | HFE avg={hfe_mean:.2f}m")

    return matched_hfe, matched_ki


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
        n = max(len(hfe), len(ki))

        def resample(line, n):
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

    for i, poly in enumerate(result):
        if 0 < np.linalg.norm(poly[0, :2] - poly[-1, :2]) < close_gap:
            result[i] = np.vstack([poly, poly[0]])

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
    edges = filter_edges(pts, edges, max_len=params["max_edge_len"])

    result              = smooth_chains(edges, pts, window=params["smooth_window"])
    hfe_lines, ki_lines = classify_boundary(pts, edges, street_pts)

    centrelines    = compute_centrelines(hfe_lines, ki_lines)
    buffer_results = calc_buffers(ki_lines, street_pts)

    return source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results


# --- Interactive Mode ---

def interactive_loop(input_path, out_dir, params):
    print("\n" + "="*65)
    print("          INTERACTIVE MODE")
    print("="*65)

    source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results = run_pipeline(input_path, params)

    while True:
        print("\nCurrent parameters:")
        for k, v in params.items():
            print(f"   {k:<18} = {v}")

        print("\nOptions:")
        print("   1 -> Show preview")
        print("   2 -> Change parameter (re-runs pipeline)")
        print("   3 -> Save outputs and exit")
        print("   4 -> Exit without saving")
        choice = input("   Enter choice: ").strip()

        if choice == "1":
            show_preview(pts, edges, result, hfe_lines, ki_lines)

        elif choice == "2":
            print("\nWhich parameter?")
            print("   1 alpha       2 voxel_size       3 max_edge_len       4 smooth_window")
            p = input("   Choice: ").strip()

            param_map = {"1": "alpha", "2": "voxel_size", "3": "max_edge_len", "4": "smooth_window"}
            if p in param_map:
                key     = param_map[p]
                new_val = input(f"   New value for {key} (current={params[key]}): ").strip()
                if new_val:
                    params[key] = float(new_val)
                    source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results = run_pipeline(input_path, params)
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
    script_dir = Path(__file__).resolve().parent
    out_dir    = script_dir.parent / "outputs"

    parser = argparse.ArgumentParser(description="Sidewalk Boundary Extraction")
    parser.add_argument("input",               help="Path to classified LAZ file")
    parser.add_argument("--voxel-size",        type=float, default=0.25)
    parser.add_argument("--alpha",             type=float, default=0.3)
    parser.add_argument("--max-edge-len",      type=float, default=3.0)
    parser.add_argument("--smooth-window",     type=int,   default=5)
    parser.add_argument("--slice-step",        type=float, default=0.5)
    parser.add_argument("--straightness",      type=float, default=1.5)
    parser.add_argument("--edge-percentile",   type=float, default=2.0)
    parser.add_argument("--cluster-dist",      type=float, default=2.5)
    parser.add_argument("--merge-dist",        type=float, default=8.0)
    parser.add_argument("--street-label",      type=int,   default=STREET_LABEL)
    parser.add_argument("--interactive",       action="store_true")
    args = parser.parse_args()

    params = {
        "voxel_size":      args.voxel_size,
        "alpha":           args.alpha,
        "max_edge_len":    args.max_edge_len,
        "smooth_window":   args.smooth_window,
        "slice_step":      args.slice_step,
        "straightness":    args.straightness,
        "edge_percentile": args.edge_percentile,
        "cluster_dist":    args.cluster_dist,
        "merge_dist":      args.merge_dist,
        "street_label":    args.street_label,
    }

    print("=== Sidewalk Boundary Extraction Tool ===")

    if args.interactive:
        interactive_loop(args.input, out_dir, params)
    else:
        source_las, pts, edges, result, hfe_lines, ki_lines, centrelines, buffer_results = run_pipeline(args.input, params)
        save_all_outputs(result, hfe_lines, ki_lines, centrelines, buffer_results, out_dir, source_las)
        print("Done")


if __name__ == "__main__":
    main()