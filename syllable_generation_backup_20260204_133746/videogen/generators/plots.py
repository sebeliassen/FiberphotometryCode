"""
syllable_generation/videogen/generators/plots.py

Generates static trajectory plots (PDFs) for each syllable using Keypoint MoSeq.
These PDFs are cached and displayed by the `SyllableGridGenerator`.

Usage:
    python -m syllable_generation.videogen.generators.plots --checkpoint <path> --output <dir>
"""

import argparse
import keypoint_moseq as kpms
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from jax_moseq.utils import unbatch

def generate_plots(checkpoint_path: str, output_dir: str, test_mode: bool = False):
    """
    Loads KPMS checkpoint and config, aligns coordinates, and calls `generate_trajectory_plots`.
    Sets min_frequency=0 for maximum syllable coverage.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    project_dir = checkpoint_path.parent.parent.resolve()
    model_name = checkpoint_path.parent.name
    config_path = project_dir / "config.yml"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        return

    # Load Model & Data
    try:
        model, data, metadata, iteration = kpms.load_checkpoint(path=str(checkpoint_path))
        conf = kpms.load_config(str(project_dir))
        fps = conf.get('fps', 30)
        print(f"Loaded config. Using FPS: {fps}")
    except Exception as e:
        print(f"Error loading checkpoint/config: {e}")
        return

    # Fix for coordinate dimension mismatch:
    # Ensure config['bodyparts'] aligns with loaded data dimensions
    if 'use_bodyparts' in conf:
        conf['bodyparts'] = conf['use_bodyparts']

    print("Extracting results and coordinates...")
    results = kpms.extract_results(model, metadata, str(project_dir), str(model_name), save_results=False)
    coordinates = unbatch(data['Y'], *metadata)
    
    if test_mode:
        print("\n--- TEST MODE ENABLED ---")
        # Slice to just first 2 sessions
        all_keys = list(coordinates.keys())
        subset_keys = all_keys[:2]
        print(f"Limiting to first {len(subset_keys)} sessions: {subset_keys}")
        
        coordinates = {k: coordinates[k] for k in subset_keys}
        # Filter results too (results is dict of dicts usually keyed by session)
        results = {k: results[k] for k in subset_keys if k in results}
        print(f"Subset Coordinates: {len(coordinates)}")
        print(f"Subset Results: {len(results)}")
        
        # Verify Lengths
        for key in subset_keys:
            c_len = len(coordinates[key])
            if key in results:
                # results[key] is dict with 'syllable'
                try:
                    r_len = len(results[key]['syllable'])
                    print(f"Session {key[:20]}...: Coords={c_len}, Results={r_len}")
                    if c_len != r_len:
                        print("WARNING: Length Mismatch!")
                except KeyError:
                    print(f"Session {key} missing 'syllable' in results")
            else:
                 print(f"Session {key} missing from results")
                 
        print("-------------------------\n")
    elif isinstance(coordinates, dict):
         print(f"Num Sessions: {len(coordinates)}")

    print("Generating Trajectory Plots...")
    print("calling kpms.generate_trajectory_plots... (This may take a few minutes)")
    print(f"Project Dir: {project_dir}")
    print(f"Output Dir: {output_dir}")
    
    # Generate Plots
    # Critical: min_frequency=0 ensures ALL identified syllables get a plot, regardless of rarity.
    kpms.generate_trajectory_plots(
        coordinates=coordinates,
        results=results,
        project_dir=str(project_dir),
        output_dir=str(output_dir),
        min_frequency=0, # Maximum leniency
        min_duration=0,  # Maximum leniency
        save_individually=True, 
        save_gifs=True, 
        save_mp4s=False,
        density_sample=False, # Disable density sampling to avoid "minimum 50 neighbors" filter
        **conf
    )
    
    print(f"Done. Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Keypoint-MoSeq Trajectory Plots")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.h5")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--test", action="store_true", help="Run in test mode (fewer sessions for speed)")
    
    args = parser.parse_args()
    
    generate_plots(args.checkpoint, args.output, test_mode=args.test)

if __name__ == "__main__":
    main()
