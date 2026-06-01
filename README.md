# Width Metrics Module (`feature/width-metrics`)

PIC: Sujeeth

Computes sidewalk accessibility metrics — overall width, usable width, obstacle count, and slope — from classified point clouds and extracted boundaries. This is the final analysis stage that turns geometry into the policy-relevant numbers the IFP cares about.

This is the fourth stage of the Sidewalk Scanner pipeline. It takes either the classified points from Vency's module or the boundary lines from Ahmed's module, and produces per-segment metrics and summary statistics.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: `laspy[lazrs]`, `numpy`, `scipy`, `pandas`

## Project structure

```
├── metrics/
│   └── width_metrics.py          # Width, usable width, slope, obstacle metrics
├── processing/                   # Boundary extraction scripts (shared with Ahmed's module)
├── Width_notebook/
│   └── test_width_metrics.ipynb  # Analysis and visualisation notebook
├── requirements.txt
└── README.md
```

## Quick start

The module supports two modes: **point-based** (from classified points) and **boundary-based** (from extracted boundary lines).

```bash
# Point-based — measure width directly from classified sidewalk points
python3 -m metrics.width_metrics --input-metric classified/utrecht_mlp_classified.laz --output outputs/width_metrics --metric-city utrecht

# Boundary-based — measure width between extracted kerb and frontage lines
python3 -m metrics.width_metrics --kerb-file outputs/sidewalk_boundary_kerb.obj --hfe-file outputs/sidewalk_boundary_hfe.obj --output outputs/width_metrics --metric-city utrecht
```

## What the module does

### Point-based metrics

Working directly from classified sidewalk points, the module measures width using a principal-axis approach:

1. **Find the sidewalk axis**: Singular Value Decomposition (SVD) on the sidewalk points finds the principal axis — the *along* direction of travel — and the perpendicular *across* axis.
2. **Segment**: Sidewalk points are divided into segments along the walking axis (default 1m each).
3. **Overall width**: For each segment, width is the robust range (5th–95th percentile) of across-axis positions. Using percentiles instead of min/max avoids sensitivity to stray outlier points.
4. **Usable width**: The across-axis span occupied by obstacle-class points within the segment is subtracted from the overall width. Obstacle classes are codes 0, 5, 8, 13, 15.
5. **Slope**: Computed as the vertical (Z) range over the along-axis range, expressed as a percentage.
6. **Quality filtering**: Segments with fewer than 20 points, or with implausible widths over 15m (likely classifier spillover into open areas), are excluded and counted separately.

### Boundary-based metrics

Given the kerb (KI) and frontage (HFE) boundary lines from Ahmed's module, the module measures the perpendicular distance between them, filtering to a reasonable range (0.8m to 15m) to discard artefacts.

## Input

| Input | Source | Mode |
|-------|--------|------|
| `classified/<city>_mlp_classified.laz` | Vency's classifier | Point-based |
| `sidewalk_boundary_kerb.obj` / `.las` | Ahmed's boundary extraction | Boundary-based |
| `sidewalk_boundary_hfe.obj` / `.las` | Ahmed's boundary extraction | Boundary-based |

**Classification labels used:**

| Code | Meaning |
|------|---------|
| 2 | sidewalk |
| 11 | street |
| 0, 5, 8, 13, 15 | obstacle classes |

## Output

```
outputs/<city>/
├── sidewalk_segment_metrics.csv     # Per-segment: width, usable width, obstacles, slope
├── sidewalk_metrics_summary.json    # Aggregate statistics across all segments
├── sidewalk_segmented_points.laz    # Points tagged with segment width data
└── boundary_width_summary.json      # Boundary-based width stats (boundary mode only)
```

The summary JSON includes average/min/max overall width, average/min/max usable width, average and maximum slope, total obstacle points, and the count of segments excluded as unrealistic. The `.laz` output opens in [CloudCompare](https://www.cloudcompare.org/) for visual inspection.

## CLI options

```
python3 -m metrics.width_metrics [options]

Options:
  --input-metric PATH    Classified LAZ/LAS file for point-based metrics
  --kerb-file PATH       Kerb/KI boundary file (.obj, .las, or .laz)
  --hfe-file PATH        HFE/frontage boundary file (.obj, .las, or .laz)
  -o, --output DIR       Output directory (default: outputs/)
  --segment-size FLOAT   Segment size in metres for point-based metrics (default: 1.0)
  --metric-city NAME     City name of the input file (default: bologna)
```

Provide either `--input-metric` for point-based metrics, or both `--kerb-file` and `--hfe-file` for boundary-based metrics.

## Key thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Segment size | 1.0m | Length of each measured segment |
| Robust range | 5th–95th percentile | Width measurement, ignoring outliers |
| Max reasonable width | 15.0m | Segments wider are excluded as spillover |
| Min boundary width | 0.8m | Lower bound for boundary-based filtering |
| Min segment points | 20 | Segments with fewer points are skipped |

## Datasets

| City | Status |
|------|--------|
| Utrecht | Tested (point-based and boundary-based) |
| Riga | Supported |
| Vilnius | Supported |
| Bologna | Default city in CLI |

## Notes for team

- **Classification (Vency):** Point-based mode consumes your `classified/<city>_mlp_classified.laz` directly. The sidewalk (2) and obstacle (0, 5, 8, 13, 15) codes must match your model output.
- **Boundary extraction (Ahmed):** Boundary-based mode consumes your kerb and HFE files. The `.obj` exports are used directly; `.las`/`.laz` are also supported.
- **Preprocessing (Brigitte):** Indirectly upstream — your features feed the classifier whose output this module measures.
- **Visualisation (Aaron):** The `sidewalk_segmented_points.laz` carries per-segment width as a scalar field, ready to render as a width heatmap. The summary JSON and CSV provide the curated datapoints for reports.
