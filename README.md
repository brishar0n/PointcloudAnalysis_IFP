# Sidewalk Candidate Extraction (Boundary Extraction)

## Overview
This script extracts sidewalk candidate points from LiDAR point cloud data.  
It supports two approaches:
- Geometric filtering (baseline)
- Classifier-based filtering with geometric cleanup

## Script
`processing/extract_sidewalk_candidates.py`

## Inputs
### Option 1 – Geometric (baseline)
Preprocessed file with geometric features:
- height
- planarity
- roughness

Example:
```
preprocessed/riga/low_featured.laz
```

### Option 2 – Classifier
Output from ML model:
```
classified/{city}_mlp_classified.laz
```

Labels used:
- 0 = other
- 2 = sidewalk
- 11 = road

## Usage

### 1. Geometric baseline
```
python processing/extract_sidewalk_candidates.py "preprocessed/riga/low_featured.laz" -o "outputs/riga_sidewalk_geometry.laz"
```

### 2. Classifier + cleanup
```
python processing/extract_sidewalk_candidates.py "classified/riga_mlp_classified.laz" -o "outputs/riga_sidewalk_classifier_cleaned.laz" --use-classifier --sidewalk-label 2
```

## Connection with preprocessing module

This script depends on the preprocessing module output.


## Output
- `.laz` file with extracted sidewalk candidate points
- `.txt` summary file with thresholds and statistics

## Current Status
- Geometric baseline implemented and tested (Riga dataset)
- Classifier integration implemented (waiting for classified outputs to test)

## Notes
- Preprocessing step is required to generate geometric features
- Bologna and Utrecht datasets are currently avoided (unlabelled)
- Results can be visualised in CloudCompare

The preprocessing stage creates:

```text
preprocessed/<city>/low_featured.laz
