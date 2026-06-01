# Boundary Extraction Module (`feature/boundary-extraction`)

PIC: Ahmed

Extracts sidewalk candidate points and boundary lines from classified LiDAR point cloud data. This module turns a collection of classified sidewalk points into usable geometry — boundary polygons, building-side and kerb-side lines, walking centrelines, and buffer-zone measurements.

This is the third stage of the Sidewalk Scanner pipeline. It takes the classified output from Vency's module (or geometric features directly from Brigitte's preprocessing) and produces boundary geometry that feeds into the width metrics module (Sujeeth).

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: `laspy[lazrs]`, `numpy`, `scipy`, `matplotlib`

## Project structure

```
├── processing/
│   ├── extract_sidewalk_candidates.py   # Candidate point extraction (geometric or classifier-based)
│   └── extract_sidewalk_boundary.py     # Full boundary line extraction pipeline
├── requirements.txt
└── README.md
```

## Quick start

There are two scripts, used in sequence or independently.

### Candidate extraction

Filters likely sidewalk points, either geometrically or from classifier output.

```bash
# Geometric baseline — filters using height, planarity, roughness
python3 processing/extract_sidewalk_candidates.py "preprocessed/riga/low_featured.laz" -o "outputs/riga_sidewalk_geometry.laz"

# Classifier + cleanup — uses ML labels with geometric cleanup
python3 processing/extract_sidewalk_candidates.py "classified/riga_mlp_classified.laz" -o "outputs/riga_sidewalk_classifier_cleaned.laz" --use-classifier --sidewalk-label 2
```

### Boundary extraction

Takes a classified cloud and extracts the full set of boundary lines.

```bash
python3 processing/extract_sidewalk_boundary.py data/utrecht_classified.laz

# Interactive mode — real-time matplotlib preview for tuning parameters
python3 processing/extract_sidewalk_boundary.py data/utrecht_classified.laz --interactive
```

## What the pipeline does

The boundary extraction script runs in five main stages:

1. **Load & downsample**: Reads the LAZ file and retains one point per 0.25m grid square, creating an even grid from the raw sidewalk points.
2. **Remove noise**: Eliminates random, clustered points likely caused by sensor error or misclassification.
3. **Alpha shape**: Computes the outer edge of all sidewalk points using a Delaunay-triangulation-based alpha shape (keeps triangles whose circumradius is smaller than 1/alpha; smaller alpha gives a tighter outline).
4. **Smart line fitting**: Divides the sidewalk into strips and fits lines to define each strip's two boundaries. This stage also:
   - **(4b) HFE/KI assignment**: Determines which boundary is the building frontage (Housing Front Equivalent, HFE) and which is the kerb side (Kerb Inside, KI) by comparing distances to street-class points — the smaller distance identifies the kerb side.
   - **(4c) Buffer zones**: Calculates the distance from kerb to road centreline and categorises it (Not Present / Narrow < 20cm / Medium 20–70cm / Wide > 70cm) per the IFP width specification.
   - **(4d) Walking centreline**: Computes the centreline between the HFE and KI boundaries.
5. **Stitch & close**: Bridges gaps caused by scan breaks and closes polygon rings to form complete street boundaries.

## Input

The module accepts two kinds of input:

| Input | Source | Use |
|-------|--------|-----|
| `preprocessed/<city>/low_featured.laz` | Brigitte's preprocessing | Geometric baseline (height, planarity, roughness) |
| `classified/<city>_mlp_classified.laz` | Vency's classifier | Classifier-based extraction |

**Classification labels used (from the classifier):**

| Code | Label |
|------|-------|
| 0 | other |
| 2 | sidewalk |
| 11 | street/road |

## Output

The candidate extraction script produces:

```
outputs/<city>_sidewalk_*.laz    # Extracted sidewalk candidate points
outputs/<city>_*_summary.txt     # Thresholds and statistics
```

The boundary extraction script produces:

```
outputs/sidewalk_boundary_ALL.laz    # Closed boundary polygons (label 10)
outputs/sidewalk_HFE.laz             # Building-side lines (label 10)
outputs/sidewalk_KI.laz              # Kerb-side lines (label 11)
outputs/sidewalk_centreline.laz      # Walking centreline between HFE and KI (label 12)
outputs/sidewalk_buffer_zones.csv    # Buffer width category per strip
```

Buffer-zone categories in the CSV: Not Present / Narrow (< 20cm) / Medium (20–70cm) / Wide (> 70cm).

All `.laz` outputs can be opened in [CloudCompare](https://www.cloudcompare.org/).

## CLI options

```
python3 processing/extract_sidewalk_candidates.py <input.laz> [options]

Options:
  -o, --output PATH         Output .laz path
  --use-classifier          Use ML classification labels instead of geometric filtering
  --sidewalk-label INT      Classification code for sidewalk (default: 2)
```

```
python3 processing/extract_sidewalk_boundary.py <input.laz> [options]

Options:
  --interactive             Real-time matplotlib preview for parameter tuning
  --sidewalk-label INT      Classification code for sidewalk (default: 2)
  --street-label INT        Classification code for street (default: 11)
```

## Datasets

| City | Status |
|------|--------|
| Riga | Geometric baseline tested |
| Utrecht | Boundary extraction tested |
| Vilnius | Supported |
| Bologna | Avoided where unlabelled |

The geometric baseline was implemented and tested first (on Riga), with classifier integration added once classified outputs became available from Vency's module.

## Current status

- Geometric candidate baseline implemented and tested (Riga)
- Classifier-based candidate extraction implemented
- Full boundary pipeline implemented through all five stages (alpha shape, HFE/KI assignment, buffer zones, centreline, stitch & close)
- Interactive preview mode available for parameter tuning

## Notes for team

- **Preprocessing (Brigitte):** The geometric baseline depends on your `low_featured.laz` output (height, planarity, roughness fields).
- **Classification (Vency):** The main path uses your `classified/<city>_mlp_classified.laz`. The `--sidewalk-label` must match your model's output code (2).
- **Width metrics (Sujeeth):** Your input is the boundary output from this module — the HFE and KI lines (as OBJ/LAS) and the classified sidewalk points. The buffer-zone CSV is also directly useful for your analysis.
- **Visualisation (Aaron):** All `.laz` outputs open in CloudCompare. The boundary lines, centreline, and polygons each carry distinct labels (10, 11, 12) for easy colour separation.
