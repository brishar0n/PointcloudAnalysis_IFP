"""
run_preprocessing.py — End-to-end preprocessing pipeline.

Run this on a .LAZ file to produce feature-enriched data ready
for the classifier (Person 2).

Usage:
    python run_preprocessing.py datasets/bologna_subsampled.laz --output preprocessed/
    python run_preprocessing.py datasets/bologna_subsampled.laz --subsample 0.05 --radii 0.1 0.3 0.5
"""

import argparse
import time
from pathlib import Path
import numpy as np

from preprocessing.loader import load_point_cloud, save_point_cloud, subsample
from preprocessing.features_query_method import compute_all_features # changed to new file name
from preprocessing.splitter import (
    split_high_low, remove_noise,
    prepare_training_data, prepare_training_blocks,
)


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Preprocess a LiDAR point cloud for sidewalk analysis"
    )
    parser.add_argument("input", help="Path to .LAZ or .LAS file")
    parser.add_argument("--output", "-o", default="preprocessed",
                        help="Output directory")
    parser.add_argument("--subsample", type=float, default=None,
                        help="Voxel size for subsampling (e.g. 0.05)")
    parser.add_argument("--radii", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5],
                        help="Radii for eigenvalue computation")
    parser.add_argument("--height-split", type=float, default=2.0,
                        help="Height threshold for high/low split")
    parser.add_argument("--block-size", type=float, default=5.0,
                        help="Block size for DL training data")
    parser.add_argument("--skip-blocks", action="store_true",
                        help="Skip DL block preparation")
    args = parser.parse_args(args_list)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Load 
    print("\n[1/6] Loading point cloud...")
    cloud = load_point_cloud(args.input)

    # Step 2: Subsample (optional) 
    if args.subsample:
        print(f"\n[2/6] Subsampling at {args.subsample}m...")
        cloud = subsample(cloud, voxel_size=args.subsample)
    else:
        print("\n[2/6] Subsampling: skipped")

    # Step 3: Compute geometric features
    print(f"\n[3/6] Computing geometric features (radii={args.radii})...")
    cloud = compute_all_features(cloud, radii=args.radii)

    # Step 4: Remove noise 
    print("\n[4/6] Removing noise...")
    cloud = remove_noise(cloud)

    # Step 5: High/low split 
    print(f"\n[5/6] Splitting high/low (threshold={args.height_split}m)...")
    low_cloud, high_cloud = split_high_low(
        cloud, height_threshold=args.height_split
    )

    save_point_cloud(cloud, out_dir / "full_featured.laz")
    save_point_cloud(low_cloud, out_dir / "low_featured.laz")
    save_point_cloud(high_cloud, out_dir / "high_featured.laz")

    # Step 6: Prepare training data 
    print("\n[6/6] Preparing training data...")

    # Traditional ML format (for Vency's testing/ branches)
    result = prepare_training_data(low_cloud)
    X_train, X_test, y_train, y_test, feature_names = result
    np.savez(
        out_dir / "train_data_flat.npz",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
    )
    print(f"  Saved flat training data → {out_dir / 'train_data_flat.npz'}")

    # Deep learning block format (for PointNet/RandLA-Net)
    if not args.skip_blocks:
        train_blocks, test_blocks, feat_names = prepare_training_blocks(
            low_cloud, block_size=args.block_size
        )
        np.savez(
            out_dir / "train_data_blocks.npz",
            train_xyz=[b[0] for b in train_blocks],
            train_features=[b[1] for b in train_blocks],
            train_labels=[b[2] for b in train_blocks],
            test_xyz=[b[0] for b in test_blocks],
            test_features=[b[1] for b in test_blocks],
            test_labels=[b[2] for b in test_blocks],
            feature_names=feat_names,
            allow_pickle=True,
        )
        print(f"  Saved block training data → {out_dir / 'train_data_blocks.npz'}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Preprocessing complete in {elapsed/60:.1f} minutes")
    print(f"Outputs in: {out_dir.resolve()}")
    print(f"{'='*60}")
    
    return out_dir


if __name__ == "__main__":
    main()
