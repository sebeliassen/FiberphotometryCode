
"""
Panel 1h Unified Driver

This script serves as the main entry point for running the Panel 1h replication pipeline.
It orchestrates the execution of digestion and plotting steps for the specified project 
and analysis type.

Usage:
    python run_panel_1h_unified.py --project {ifdn, amph} --type {peak, trace, all}

Arguments:
    --project: Target dataset ('ifdn' or 'amph').
    --type: Analysis mode ('peak', 'trace', or 'all' for both).
    --limit: Optional limit on number of sessions to process.
    --min-trials: Minimum trials required to plot a syllable.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_name, args):
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run Panel 1h Unified Pipeline")
    parser.add_argument("--project", type=str, default="ifdn", choices=["ifdn", "amph"])
    parser.add_argument("--limit", type=str, default=None, help="Limit number of sessions (optional)")
    parser.add_argument("--type", type=str, default="peak", choices=["peak", "trace", "all"],
                        help="Analysis type: 'peak' (default), 'trace' (heatmap), or 'all'")
    parser.add_argument("--min-trials", type=str, default=None, help="Min trials for plotting")

    args = parser.parse_args()
    types_to_run = ["peak", "trace"] if args.type == "all" else [args.type]

    for run_type in types_to_run:
        print(f"\n>>>> STARTING ANALYSIS TYPE: {run_type.upper()} <<<<\n")

        # 1. Digest
        digest_args = ["--project", args.project, "--type", run_type]
        if args.limit:
            digest_args.extend(["--limit", args.limit])
        run_step("digest_panel_1h_unified.py", digest_args)

        # 2. Plot
        plot_args = ["--project", args.project, "--type", run_type]
        if args.min_trials:
            plot_args.extend(["--min-trials", args.min_trials])
        elif run_type == "trace":
            plot_args.extend(["--min-trials", "50"])

        run_step("plot_panel_1h_unified.py", plot_args)

    print("\n=== Panel 1h Pipeline Complete ===")

if __name__ == "__main__":
    main()
