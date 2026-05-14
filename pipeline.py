import argparse
import run_preprocessing
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Main Pipeline")
    parser.add_argument("input", 
                        help="Path to input .LAZ or .LAS file")
    parser.add_argument("--output", "-o", 
                        default="preprocessed/", 
                        help="Output directory")
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
    
    args = parser.parse_args()
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
    
    print("\n--- CONFIGURING CLASSIFICATION ---")
    
    user_epochs = input("Enter number of epochs (default 50): ").strip() or "50"
    do_loco = input("Run LOCO evaluation? (y/n): ").strip().lower() == 'y'
    do_train = input("Train final model? (y/n): ").strip().lower() == 'y'
    
    class_cmd = [sys.executable, "run_pipeline.py", "--apply", "--cities", args.city]
    
    if do_loco:
        class_cmd.append("--loco")
    if do_train:
        class_cmd.append("--train")
        
    class_cmd.extend(["--epochs", user_epochs])
    
    subprocess.run(class_cmd, cwd="classification", check=True)
    
    print("\n FULL PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
