And and do not change any meanings of words, dates or facts.

Do not return anything but a redone version of this text!

# sidewalk boundary extraction

Extracts hfe (Building-side) and ki (Kerb-side) boundary lines from a classified LiDAR point cloud. The output of the ML classifier is used to create georeferenced boundary polylines, a walking centerline and Buffer zone measurements which align to IFP accessibility standards.

---

## requirements

- python 3.10+

```
Pip install laspy[lazrs] numpy scipy matplotlib
```

---
### Standalone

```
Python extract_sidewalk_boundary.py
```

### via pipeline

```
Python extract_sidewalk_boundary.py --input-processing --boundary-city
```

### interactive mode (parameter tuning)

```
Python extract_sidewalk_boundary.py --interactive
```

Interactive mode allows you to view results and modify Parameters without having to run everything again each time.

---
## Parameters

| parameter | default value | description |
| --- | --- | --- |
| `--Voxel-size` | 0.25 | downsampling Voxel grid size in metres |
| `--Alpha` | 0.3 | Alpha shape tightness - lower = looser boundary higher = tighter |
| `--max-edge-len` | 3.0 | edge length max (in metres) - filters out road crossing edges |
| `--smooth-window` | 5 | window smoothing width of the arc length interpolation for boundary smoothness |
| `--street-Label` | 11 | classification Label of points classified as roads/streets |
| `--boundary-city` | — | city name - sets the sub-folder where outputs are stored under `outputs/` |

---

## input

A classified laz/las File from step 2 of the ML classification process with the following classifications:

| Label | class |
| --- | --- |
| 2 | sidewalk |
| 11 | Street / Road |

---

## output

All files are saved in `outputs//` (or `outputs/` if no city is specified).

| File | description |
| --- | --- |
| `sidewalk_hfe.laz` | Building-side boundary line |
| `sidewalk_KI.laz` | Kerb-side boundary line |
| `sidewalk_centreline.laz` | centre-line of walkway (mid-point between hfe and ki) |
| `sidewalk_boundary_all.laz` | full boundaries |
| `sidewalk_buffer_zones.csv` | gap between Kerb and road for each strip with IFP accessibility category |

---

### Buffer zone categories (IFP specification)

| category | gap |
| --- | --- |
| not present | less than 5cm |
| Narrow | 5 - 20cm |
| medium | 20 - 70cm |
| wide | greater than 70cm |

---

## methodology

This script runs a 5 Stage pipe-line:

**Stage 1 - load & down sample**

Reads the classified laz File, select only points classified as sidewalk (Label=2), and apply Voxel grid down sampling to reduce point density while preserving shape.

**Stage 2 - noise removal**

Uses connected component analysis (flood-fill with KD-trees) to remove isolated cluster(s) of points. Any cluster containing fewer than 30 points within a radius of 5 meters is discarded.

**Stage 3 - Alpha shape**

Uses Delaunay triangulation with circum-radius filtering to compute concave hull of sidewalk points using α = 0.3. Longer than max_edge_len are removed to avoid creating artifacts due to crossing roadways. Will fall back to convex hull if the number of edges produced by the Alpha-shapes is too small.

**Stage 4 - Classification & Analysis**

- `classify_boundary`: Groups Alpha-shape edges into connected components (one per sidewalk strip); then performs one-dimensional k-means clustering on each edge distance to nearest roadway. Closer cluster becomes ki; farther becomes hfe. Uses a swap guard to correct cases where assignments invert.
- `calc_buffers`: calculates minimum distance from each ki line to the road surface; then assigns an IFP accessibility category based on that distance;
- `compute_centrelines`: samples hfe and ki lines to equal length using arc-length interpolation; then computes mid-point between them as walking center-line.

**Stage 5 - stitch & close**: Connects broken segments of boundary using greedy nearest end-point matching. Will close any near complete rings where start & end points are within close gap threshold.

---
## how this fits together

This module exists between steps 2 (ML classifier) and 4 (width metrics) in the larger pipeline.

```bash
Pipeline.py
└── classification/run_pipeline.py (step 2 - ML classifier)
└── processing/extract_sidewalk_boundary.py (step 3 - this module)
Outputs: sidewalk_hfe.laz, sidewalk_KI.laz -> metrics/width_metrics.py (step 4)
```

The `sidewalk_hfe.laz` and `sidewalk_KI.laz` outputs are passed directly to the width metrics module as inputs for calculation of per-segment width values.

---
## known issues

- separation of hfe and ki works best when there is high dense point cloud classification. Poor classification quality may produce overlapping boundaries rather than parallel boundaries.
- the Alpha shape parameter (`--Alpha`) may need adjustment depending on scan density per city. More dense scans can use a higher value for Alpha shape parameter for a tighter fit.