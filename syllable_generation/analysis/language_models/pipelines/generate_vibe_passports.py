import os
import pickle
import sys

import numpy as np

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.models.strategies import StrategyBERTopic
from analysis.language_models.lib.visualization.vibe_passports import generate_vibe_passports_figure


def main():
    print("=== Generating 9-Vibe Kinematic Passports ('Behavioral Moods') ===")

    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")

    dataset = SessionDataset(cache_path=cache_path)
    win_inputs = dataset.get_window_inputs()
    n_win_per_session = dataset.n_windows_per_session

    # Fit 9-vibe sweetspot model
    model = StrategyBERTopic(
        vectorizer_type="transition",
        n_neighbors=30,
        min_cluster_size=300,
        random_state=42,
        vocab_size=300
    )
    model.fit(win_inputs)
    strat_ids, _ = model.transform(win_inputs)

    # Load session dataframes
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    session_id_list = [meta_tuple[6] for meta_tuple in dataset.session_window_meta]
    session_dfs = {}
    for s in raw_sessions:
        exact_id = s.session_id
        for s_id in session_id_list:
            if s_id.startswith(exact_id) or exact_id.startswith(s_id):
                session_dfs[s_id] = s.all_data_df
                break

    frame_strat_dict = {}
    win_offset = 0
    for s_idx, s_id in enumerate(session_id_list):
        n_win = n_win_per_session[s_idx]
        s_wins = strat_ids[win_offset : win_offset + n_win]
        win_offset += n_win
        frame_strat_dict[s_id] = np.repeat(s_wins, 800)

    save_fig_path = os.path.join(out_dir, "vibe_passports.png")
    generate_vibe_passports_figure(session_dfs, frame_strat_dict, model.n_topics, save_fig_path)
    print("=== Completed 9-Vibe Kinematic Passports Generation ===")

if __name__ == "__main__":
    main()
