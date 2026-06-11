# Pointcloud Analysis for Pedestrian Access

PIC: Team HN-677 

End-to-end pipeline that analyses LiDAR point cloud scans of urban streets to extract pedestrian sidewalk accessibility metrics. Developed for the International Federation of Pedestrians (IFP).

The system runs as five connected stages — preprocessing, classification, boundary extraction, width metrics, and visualisation — chained together by a single `pipeline.py`. Each stage has its own module README with full detail; this document covers running the whole pipeline.

## Setup

Requires Python 3.9+ (Windows for the Potree visualisation step).

```bash
pip install -r requirements.txt
```

## Project structure

Datasets must be placed in a `datasets/` folder following this hierarchy:

```text
POINTCLOUDANALYSIS_IFP/
├── classification/
├── datasets/
├── metrics/
├── preprocessing/
├── processing/
├── visualisation/
│   ├── potree_vis/
│   │   ├── build/
│   │   ├── libs/
│   │   └── index.html
│   └── PotreeConverter_1.7_windows_x64/
├── .gitignore
├── pipeline.py
├── README.md
└── requirements.txt
```

## Quick start

Run the entire pipeline with a single command:

```bash
python .\pipeline.py <path to dataset> --city <name of city> --all
```

Example:

```bash
python .\pipeline.py datasets\bologna_subsampled.laz --city bologna --all
```

This runs all five stages in order: preprocessing → classification → boundary extraction → width metrics → visualisation.

## What the pipeline does

| Stage | Module | Role |
|-------|--------|------|
| 1. Preprocessing | `preprocessing/` | Reads the scan, computes geometric features, exports training data |
| 2. Classification | `classification/` | Classifies points as sidewalk, street, or other |
| 3. Boundary extraction | `processing/` | Traces sidewalk boundary lines (frontage and kerb sides) |
| 4. Width metrics | `metrics/` | Measures overall width, usable width, slope, and obstacles |
| 5. Visualisation | `visualisation/` | Converts clouds for the interactive Potree 3D viewer |

Each stage has a dedicated README in its module folder with full setup, options, and output details.

## Skipping preprocessing (optional)

Preprocessing is the most time-consuming stage. If the preprocessed files are already available, you can skip it: comment out the `preprocessing()` function call in the `main` function of `pipeline.py`. The pipeline will then run classification → boundary extraction → width metrics → visualisation directly.

Preprocessed files live in `classification/preprocessed/`, and each city's classification input is a single file:

```text
preprocessed/<city>/low_featured.laz
```

## Viewing the visualisation

Once `pipeline.py` has finished, the converted point clouds are written to `visualisation/potree_vis/pointclouds/`, following this hierarchy:

```text
visualisation/
├── potree_vis/
│   ├── build/
│   ├── libs/
│   ├── pointclouds/
│   │   └── <city>/
│   │       ├── city        # all points in the city
│   │       └── sidewalk    # sidewalk points only
│   └── index.html
└── PotreeConverter_1.7_windows_x64/
```

To launch the viewer, change into the Potree directory and start a local server:

```bash
cd visualisation\potree_vis
python -m http.server 8000
```

Then open the visualisation in a browser at **http://localhost:8000/**.
