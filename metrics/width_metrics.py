import laspy
import numpy as np
import pandas as pd


SIDEWALK_LABEL = 2
STREET_LABEL = 11


def load_laz(file_path):
    las = laspy.read(file_path)

    points = pd.DataFrame({
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
        "classification": np.asarray(las.classification)
    })

    return points


def get_sidewalk_points(points):
    sidewalk = points[points["classification"] == SIDEWALK_LABEL].copy()

    if sidewalk.empty:
        raise ValueError("No sidewalk points found.")

    return sidewalk


def segment_sidewalk(points, segment_size=1.0):
    x_min = points["x"].min()
    points["segment_id"] = ((points["x"] - x_min) / segment_size).astype(int)
    return points


def compute_width(segment):
    return segment["y"].max() - segment["y"].min()


def compute_slope(segment):
    z = segment["z"]
    x = segment["x"]

    if x.max() - x.min() == 0:
        return 0

    return ((z.max() - z.min()) / (x.max() - x.min())) * 100


def compute_metrics(points):
    results = []

    for seg_id, group in points.groupby("segment_id"):
        width = compute_width(group)
        slope = compute_slope(group)

        results.append({
            "segment_id": seg_id,
            "point_count": len(group),
            "width": width,
            "slope_percent": slope
        })

    return pd.DataFrame(results)


def summarize(metrics):
    return {
        "avg_width": metrics["width"].mean(),
        "min_width": metrics["width"].min(),
        "max_width": metrics["width"].max(),
        "avg_slope": metrics["slope_percent"].mean()
    }


def run(file_path):
    points = load_laz(file_path)
    sidewalk = get_sidewalk_points(points)
    segmented = segment_sidewalk(sidewalk)

    metrics = compute_metrics(segmented)
    summary = summarize(metrics)

    print("=== Segment Metrics ===")
    print(metrics.head())

    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    run("data/sample.laz")  # replace later