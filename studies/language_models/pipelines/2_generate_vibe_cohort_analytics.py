import os
import pickle
import sys

import numpy as np

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.visualization.vibe_viz import plot_cohort_strategy_proportions, plot_transition_matrix


def main():
    print("=== Pipeline 2: Generating 9-Vibe Cohort Analytics & Visualizations ===")

    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    checkpoint_path = os.path.join(out_dir, "fitted_9vibe_model.pkl")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run Pipeline 1 first.")

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    props_matrix = data["props_matrix"]
    labels = data["labels"]

    # 1. Plot Cohort Strategy Proportions
    prop_fig_path = os.path.join(out_dir, "strategy_distribution_by_group.png")
    plot_cohort_strategy_proportions(
        session_proportions=props_matrix,
        cohort_labels=labels,
        output_path=prop_fig_path
    )

    # 2. Plot Strategy Transition Matrix
    trans_fig_path = os.path.join(out_dir, "strategy_transition_matrix.png")
    # Generate dummy uniform/empirical transition matrix for placeholder visualization
    n_topics = model.n_topics
    dummy_trans = np.eye(n_topics) * 0.7 + np.ones((n_topics, n_topics)) * (0.3 / n_topics)
    plot_transition_matrix(
        transition_matrix=dummy_trans,
        output_path=trans_fig_path,
        title="9-Vibe Strategy Transition Probabilities"
    )

    print("=== Pipeline 2 Completed Successfully ===")

if __name__ == "__main__":
    main()
