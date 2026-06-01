"""
run_pipeline.py — Single entry point for the full classification pipeline.

Chains together LOCO evaluation, final model training, and city application.
Run individual steps or everything at once.

Usage:
    python run_pipeline.py --all                          # run everything
    python run_pipeline.py --all --epochs 100             # custom epochs

    python run_pipeline.py --loco                         # LOCO evaluation only
    python run_pipeline.py --train                        # train final model only
    python run_pipeline.py --apply --cities utrecht       # apply to one city
    python run_pipeline.py --apply --cities utrecht bologna  # apply to multiple

    python run_pipeline.py --train --apply --cities utrecht bologna  # train + apply
    python run_pipeline.py --loco --train                 # validate then train

    python run_pipeline.py --all --epochs 100             # all steps, 100 epochs
"""

import argparse
import subprocess
import sys
import os


# ── Helpers ───────────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_step(step, total, label):
    print(f"\n[{step}/{total}] {label}")
    print("-" * 60)


def ask_continue(step_name, error_msg):
    """Ask user what to do when a step fails."""
    print(f"\n[ERROR] STEP FAILED: {step_name}")
    print(f"   Error: {error_msg}")
    print(f"\n   What would you like to do?")
    print(f"   [c] Continue to next step")
    print(f"   [s] Stop the pipeline")

    while True:
        choice = input("\n   Your choice (c/s): ").strip().lower()
        if choice == "c":
            print("   [SKIP]  Skipping — continuing to next step...")
            return True   # continue
        elif choice == "s":
            print("   [STOPPED] Pipeline stopped by user.")
            sys.exit(1)
        else:
            print("   Please enter 'c' to continue or 's' to stop.")


def run_step(step_name, cmd):
    """
    Run a subprocess command.
    On failure, ask the user whether to continue or stop.
    Returns True if step succeeded or user chose to continue.
    """
    print(f"\n▶  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=os.environ.copy())

    if result.returncode != 0:
        return ask_continue(step_name, f"exit code {result.returncode}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the MLP sidewalk classification pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all
  python run_pipeline.py --all --epochs 100
  python run_pipeline.py --loco --epochs 30
  python run_pipeline.py --train
  python run_pipeline.py --apply --cities utrecht bologna
  python run_pipeline.py --train --apply --cities utrecht
        """
    )

    # ── What to run ───────────────────────────────────────────────────────
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run everything: LOCO -> train final model -> apply to cities"
    )
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

    # ── Options ───────────────────────────────────────────────────────────
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

    args = parser.parse_args()

    # ── Validate at least one flag was given ──────────────────────────────
    if not any([args.all, args.loco, args.train, args.apply]):
        parser.print_help()
        print("\n[ERROR] Please specify at least one step: --all, --loco, --train, --apply")
        sys.exit(1)

    # ── Resolve which steps to run ────────────────────────────────────────
    run_loco  = args.all or args.loco
    run_train = args.all or args.train
    run_apply = args.all or args.apply

    steps = []
    if run_loco:
        steps.append("LOCO evaluation")
    if run_train:
        steps.append("Train final model")
    if run_apply:
        for city in args.cities:
            steps.append(f"Apply model -> {city}")

    total = len(steps)

    # ── Print plan ────────────────────────────────────────────────────────
    print_header("MLP Sidewalk Classification Pipeline")
    print(f"\n  Steps to run ({total} total):")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")
    print(f"\n  Epochs: {args.epochs}")
    if run_apply:
        print(f"  Cities to apply: {', '.join(args.cities)}")

    # ── Run steps ─────────────────────────────────────────────────────────
    step_num = 1
    python   = sys.executable   # use same python env as current process

    # Step: LOCO
    if run_loco:
        print_step(step_num, total, "LOCO Cross-City Evaluation")
        run_step(
            "LOCO evaluation",
            [python, "dl_loco_evaluation.py", "--epochs", str(args.epochs)]
        )
        step_num += 1

    # Step: Train final model
    if run_train:
        print_step(step_num, total, "Training Final Model")
        run_step(
            "Train final model",
            [python, "dl_train_final_model.py", "--epochs", str(args.epochs)]
        )
        step_num += 1

    # Step: Apply to cities
    if run_apply:
        for city in args.cities:
            print_step(step_num, total, f"Applying Model -> {city.upper()}")
            run_step(
                f"Apply model to {city}",
                [python, "dl_apply_model.py", "--city", city]
            )
            step_num += 1

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [OK] Pipeline complete!")
    print(f"{'='*60}")
    if run_apply:
        print(f"\n  Classified files saved to: classified/")
        for city in args.cities:
            print(f"    -> classified/{city}_mlp_classified.laz")
    if run_loco:
        print(f"\n  LOCO results saved to: results/mlp_loco_results.csv")
    if run_train:
        print(f"\n  Final model saved to:  models/final_mlp_classifier.pt")
    print()


if __name__ == "__main__":
    main()
