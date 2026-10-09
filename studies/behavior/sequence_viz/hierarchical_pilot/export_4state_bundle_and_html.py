#!/usr/bin/env python3
"""
export_4state_bundle_and_html.py
Fits the master M=4 ControlledHierarchicalModel on the full 30-min cohort,
saves the results bundle, and generates the interactive HTML visualizer.
"""

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.hierarchical_pilot.generate_4state_interactive_html import (
    build_4state_interactive_html,
)
from syllable_generation.analysis.sequence_viz.hierarchical_pilot.run_controlled_upper_timescale_experiment import (
    OUTPUT_DIR,
    ControlledHierarchicalModel,
    extract_tracks_with_posteriors,
    load_30min_cohort,
)


def main():
    bundle_path = OUTPUT_DIR / "hierarchical_model_results_bundle_4state.pkl"
    if bundle_path.exists():
        print(f"Loading existing bundle from {bundle_path.name}...")
        with open(bundle_path, "rb") as f:
            bundle_data = pickle.load(f)
        sessions_data = bundle_data["sessions_data"]
    else:
        print("Loading 30-min cohort...")
        sessions, params = load_30min_cohort(max_duration_sec=1800.0)

        print("Fitting Master M=4 Model on Full Cohort (25 iterations)...")
        master_model = ControlledHierarchicalModel(M=4, K=6, S=3, n_features=2, random_seed=42)
        master_model.fit_multi_sequence_joint(sessions, max_iters=25)

        exemplar_indices = [1, 2, 4, 0]
        print(f"Extracting exemplar tracks: {exemplar_indices}...")
        sessions_data = []
        for idx in exemplar_indices:
            s_data = extract_tracks_with_posteriors(sessions[idx], master_model)
            sessions_data.append(s_data)

        with open(bundle_path, "wb") as f:
            pickle.dump({
                "master_model": master_model,
                "sessions_data": sessions_data,
                "norm_params": params
            }, f)
        print(f"✓ Saved Bundle: {bundle_path.name}")

    # Generate Interactive HTML
    html_path = OUTPUT_DIR / "hierarchical_timeline_4state_interactive.html"
    build_4state_interactive_html(sessions_data, html_path)
    print(f"✓ Saved 4-State Interactive HTML: {html_path.name}")

if __name__ == "__main__":
    main()
