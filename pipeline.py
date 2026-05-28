import argparse
from preprocessing import run_preprocessing
import subprocess
import sys
import os 

def preprocessing(args):
    target_output = f"classification/preprocessed/{args.city}_temp"
    
    print(f"STARTING PIPELINE for {args.city}")
    
    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    print("\n--- STEP 1: Preprocessing ---")
    
    # Build the argument list dynamically
    args_list = [
        args.input, 
        "--output", target_output,
        "--height-split", str(args.height_split),
        "--block-size", str(args.block_size)
        ]
    
    if args.subsample is not None:
        args_list.extend(["--subsample", str(args.subsample)])
    if args.skip_blocks:
        args_list.append("--skip-blocks")
        
    # Call the main function directly, passing the arguments as a list
    outdir=run_preprocessing.main(args_list)
    
def classification(args):
    
    print("\n--- STEP 2: Classification ---")
    
    class_cmd = [sys.executable, "run_pipeline.py"]
    if args.all: 
        class_cmd.append("--all")
        if args.cities is not None:
            class_cmd.append("--cities")
            class_cmd.extend(args.cities)
        else:
            class_cmd.append("--cities")
            class_cmd.extend([args.city])
    if args.loco:
        class_cmd.append("--loco")
    if args.train:
        class_cmd.append("--train")
    if args.apply:
        class_cmd.append("--apply")
        if args.cities is not None:
            class_cmd.append("--cities")
            class_cmd.extend(args.cities)
        else:
            class_cmd.append("--cities")
            class_cmd.extend([args.city])
    if args.epochs:
        class_cmd.extend(["--epochs",str(args.epochs)])
    
    print(class_cmd)
    subprocess.run(class_cmd, cwd="classification", check=True)
    
def boundary_extraction(args):
    
    print("\n--- STEP 3: Boundary Extraction---")
    
    # Use the provided input or default to the output of the classification step
    input_file = args.input_processing or f"./classification/classified/{args.city}_mlp_classified.laz"
    abs_input_path = os.path.abspath(input_file)
    
    class_cmd = [sys.executable, "extract_sidewalk_boundary.py"]
    # if args.input_processing: 
    if args.input_processing is None:
        class_cmd.extend(["--input-processing",str(abs_input_path)])
    else:
        class_cmd.extend(["--input-processing",str(args.input_processing)])
    if args.voxel_size: 
        class_cmd.extend(["--voxel-size",str(args.voxel_size)])
    if args.alpha:
        class_cmd.extend(["--alpha",str(args.alpha)])
    if args.slice_step:
        class_cmd.extend(["--slice-step",str(args.slice_step)])
    if args.straightness:
        class_cmd.extend(["--straightness",str(args.straightness)])
    if args.edge_percentile:
        class_cmd.extend(["--edge-percentile",str(args.edge_percentile)])
    if args.cluster_dist:
        class_cmd.extend(["--cluster-dist",str(args.cluster_dist)])
    if args.merge_dist:
        class_cmd.extend(["--merge-dist",str(args.merge_dist)])
    if args.street_label:
        class_cmd.extend(["--street-label",str(args.street_label)])
    if args.interactive:
        class_cmd.extend(["--interactive",str(args.interactive)])
    class_cmd.extend(["--boundary-city",str(args.city)])
    
    subprocess.run(class_cmd, cwd="./processing", check=True)
    
def width_calc(args):
    # SEE THE OUTPUT FOLDER CONFIG - IF YOU WANT TO REMOVE IT OR NOT
    print("\n--- STEP 4: Width Metrics Calculation  ---")
    
    if args.input_metric:
        input_path = args.input_metric
    else:
        input_path = f"./classification/classified/{args.city}_mlp_classified.laz"

    abs_input_path = os.path.abspath(input_path)
    print(f'Getting file: {abs_input_path}')
    
    class_cmd = [sys.executable, "width_metrics.py"]
    class_cmd.extend(["--input-metric", str(abs_input_path)])
    if args.kerb_file: 
        class_cmd.extend(["--kerb-file",str(args.kerb_file)])
    else:
        input_path = f"./processing/outputs/{args.city}/sidewalk_KI.laz"
    if args.hfe_file :
        class_cmd.extend(["--hfe-file",str(args.hfe_file)])
    else:
        input_path = f"./processing/outputs/{args.city}/sidewalk_HFE.laz"
    if args.segment_size:
        class_cmd.extend(["--segment-size",str(args.segment_size)])
    class_cmd.extend(["--metric-city",str(args.city)])
    
    subprocess.run(class_cmd, cwd="./metrics", check=True)
    
def visualise(args):
    
    print("\n--- STEP 5: Visualisation ---")
    input_file = os.path.abspath(f'./classification/classified/{args.city}_mlp_classified.laz')
    output_dir = os.path.abspath(f'./visualisation/potree_vis/pointclouds/{args.city}/city')
    
    converter_path = os.path.join(
        "visualisation", 
        "PotreeConverter_1.7_windows_x64", 
        "PotreeConverter.exe"
    )

    cmd = [converter_path, input_file, "-o", output_dir]
    subprocess.run(cmd)

def main():
    STREET_LABEL   = 11
    
    parser = argparse.ArgumentParser(description="Main Pipeline")
    parser.add_argument("input", 
                        help="Path to input .LAZ or .LAS file")
    # parser.add_argument("--output", "-o", 
    #                     default="preprocessed/", 
    #                     help="Output directory")
    parser.add_argument("--subsample", 
                        type=float, 
                        default=None, 
                        help="Voxel size for subsampling (e.g. 0.05)")
    parser.add_argument("--height-split", 
                        type=float, 
                        default=2.0, 
                        help="Height threshold for high/low split")
    parser.add_argument("--block-size", type=float, default=5.0,
                        help="Block size for DL training data")
    parser.add_argument("--skip-blocks", action="store_true",
                        help="Skip DL block preparation")
    parser.add_argument("--city", required=True, help="City name (e.g., bologna)")
    
    # Tags required for classification
    parser.add_argument("--all", action="store_true",
                        help="Run everything: LOCO -> train final model -> apply to cities")
    parser.add_argument(
        "--loco",
        action="store_true",
        help="Run LOCO cross-city evaluation"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the final model on all labelled cities"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply final model to unlabelled cities (use with --cities)"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        # default=["utrecht", "bologna"],
        metavar="CITY",
        help="Cities to apply the model to (default: utrecht bologna)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for LOCO and final model (default: 50)"
    )
    
    # Tags used for boundary extraction
    parser.add_argument("--input-processing", default=None, help="Classified LAZ file")
    parser.add_argument("--voxel-size",    type=float, default=0.25)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--slice-step", type=int,   default=0.5)
    parser.add_argument("--straightness",   type=float, default=1.5)
    parser.add_argument("--edge-percentile",    type=float, default=2.0)
    parser.add_argument("--cluster-dist",     type=float, default=2.5)
    parser.add_argument("--merge-dist",    type=float, default=8.0)
    parser.add_argument("--street-label",    type=int, default=STREET_LABEL)
    parser.add_argument("--interactive", action="store_true")
    
    # Tags used for width_metrics
    parser.add_argument(
        "--input-metric",
        help="Classified LAZ/LAS file for point-based metrics.",
    )
    parser.add_argument(
        "--kerb-file",
        help="Kerb boundary file from boundary extraction.",
    )
    parser.add_argument(
        "--hfe-file",
        help="HFE boundary file from boundary extraction.",
    )
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     default="outputs/width_metrics",
    #     help="Output directory.",
    # )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=1.0,
        help="Segment size in metres for point-based metrics.",
    )
    
    args = parser.parse_args()
    
    
    # preprocessing(args)
    classification(args)
    boundary_extraction(args)
    width_calc(args)
    visualise(args)
    
    print("\n FULL PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
