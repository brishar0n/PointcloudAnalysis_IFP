import argparse
import run_preprocessing
import subprocess
import sys
import os 

def preprocessing(args):
    target_output = f"classification/preprocessed/{args.city}"
    
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
    if args.loco:
        class_cmd.append("--loco")
    if args.train:
        class_cmd.append("--train")
    if args.apply:
        class_cmd.append("--apply")
        if args.cities is not None:
            class_cmd.append("--cities")
            class_cmd.extend(args.cities)
    if args.epochs:
        class_cmd.extend(["--epochs",str(args.epochs)])
    
    subprocess.run(class_cmd, cwd="classification", check=True)
    
    # TODO MAKE SURE THAT USER IS ABLE TO APPLY THEIR OWN MODEL
    
def boundary_extraction(args):
    
    print("\n--- STEP 3: Boundary Extraction---")
    
    input_path=f"./classification/classified/{args.city}_mlp_classified.laz"
    # input_path = args.input_processing
    abs_input_path = os.path.abspath(input_path)
    
    class_cmd = [sys.executable, "extract_sidewalk_boundary.py"]
    # if args.input_processing: 
    class_cmd.extend(["--input-processing",str(abs_input_path)])
    if args.voxel_size: 
        class_cmd.extend(["--voxel-size",str(args.voxel_size)])
    if args.alpha:
        class_cmd.extend(["--alpha",str(args.alpha)])
    if args.smooth_window:
        class_cmd.extend(["--smooth-window",str(args.smooth_window)])
    if args.rdp_epsilon:
        class_cmd.extend(["--rdp-epsilon",str(args.rdp_epsilon)])
    if args.min_length:
        class_cmd.extend(["--min-length",str(args.min_length)])
    if args.close_gap:
        class_cmd.extend(["--close-gap",str(args.close_gap)])
    if args.stitch_gap:
        class_cmd.extend(["--stitch-gap",str(args.stitch_gap)])
    
    subprocess.run(class_cmd, cwd="./processing", check=True)
    
    
def main():
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
        default=["utrecht", "bologna"],
        metavar="CITY",
        help="Cities to apply the model to (default: utrecht bologna)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for LOCO and final model (default: 50)"
    )
    parser.add_argument("--input-processing", default=None, help="Classified LAZ file")
    parser.add_argument("--voxel-size",    type=float, default=0.25)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--smooth-window", type=int,   default=25)
    parser.add_argument("--rdp-epsilon",   type=float, default=0.5)
    parser.add_argument("--min-length",    type=float, default=15.0)
    parser.add_argument("--close-gap",     type=float, default=8.0)
    parser.add_argument("--stitch-gap",    type=float, default=6.0)
    
    args = parser.parse_args()
    
    
    preprocessing(args)
    classification(args)
    boundary_extraction(args)
    
    print("\n FULL PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
