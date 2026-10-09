import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.evaluation.kinematic_encoding import evaluate_session_kinematic_encoding
from analysis.language_models.lib.models.strategies import StrategyBERTopic


def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")

    print("=== Extracting Broad Scale Sweep Kinematic Benchmark & Correlation Table ===")

    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    n_windows_per_session = dataset.n_windows_per_session

    # Load raw session dataframes for frame-level kinematic calculations
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")

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

    # Top representative models across scale tiers from the 45-run Broad Scale Sweep
    sweep_models = [
        {"name": "ApproachB_mcs150_n30", "mcs": 150, "n_n": 30, "cached_acc": 71.67, "cached_k": 20, "cached_hnorm": 0.924},
        {"name": "ApproachB_mcs200_n30", "mcs": 200, "n_n": 30, "cached_acc": 61.67, "cached_k": 13, "cached_hnorm": 0.938},
        {"name": "ApproachB_mcs250_n30", "mcs": 250, "n_n": 30, "cached_acc": 63.33, "cached_k": 10, "cached_hnorm": 0.951},
        {"name": "ApproachB_mcs300_n30 (Sweetspot)", "mcs": 300, "n_n": 30, "cached_acc": 66.67, "cached_k": 9, "cached_hnorm": 0.944},
        {"name": "ApproachB_mcs250_n45", "mcs": 250, "n_n": 45, "cached_acc": 55.00, "cached_k": 7, "cached_hnorm": 0.817},
        {"name": "ApproachB_mcs300_n45", "mcs": 300, "n_n": 45, "cached_acc": 53.33, "cached_k": 6, "cached_hnorm": 0.825},
        {"name": "ApproachB_mcs400_n45", "mcs": 400, "n_n": 45, "cached_acc": 40.00, "cached_k": 4, "cached_hnorm": 0.840},
        {"name": "ApproachB_mcs400_n30", "mcs": 400, "n_n": 30, "cached_acc": 33.33, "cached_k": 3, "cached_hnorm": 0.818},
    ]

    table_rows = []

    for cfg in sweep_models:
        mcs = cfg["mcs"]
        n_n = cfg["n_n"]
        model_name = cfg["name"]

        model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=n_n, min_cluster_size=mcs, random_state=42)
        model.fit(all_win_inputs)
        strat_ids, _ = model.transform(all_win_inputs)

        frame_strat_dict = {}
        win_offset = 0
        for s_idx, s_id in enumerate(session_id_list):
            n_win = n_windows_per_session[s_idx]
            w_ids = strat_ids[win_offset : win_offset + n_win]
            win_offset += n_win
            frame_strat_dict[s_id] = np.repeat(w_ids, 800)

        kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

        table_rows.append({
            "model_name": model_name,
            "min_cluster_size": mcs,
            "active_k": cfg["cached_k"],
            "entropy_norm": cfg["cached_hnorm"],
            "group_decodability_acc_pct": cfg["cached_acc"],
            "eta2_speed": kin_metrics["eta2_velocity"],
            "eta2_dist_center": kin_metrics["eta2_dist_center"],
            "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
            "eta2_cadence": kin_metrics["eta2_cadence"]
        })

    df_table = pd.DataFrame(table_rows)

    # Compute Composite Kinematic Encoding Index (normalized mean across all 4 eta2 axes)
    eta2_cols = ["eta2_speed", "eta2_dist_center", "eta2_angular_vel", "eta2_cadence"]

    # Min-max normalization for each column to scale [0, 1] before averaging
    df_norm = (df_table[eta2_cols] - df_table[eta2_cols].min()) / (df_table[eta2_cols].max() - df_table[eta2_cols].min() + 1e-12)
    df_table["eta2_composite"] = df_norm.mean(axis=1)

    csv_path = os.path.join(out_dir, "broad_sweep_kinematic_correlation_table.csv")
    df_table.to_csv(csv_path, index=False)

    acc_arr = df_table["group_decodability_acc_pct"].values
    speed_arr = df_table["eta2_speed"].values
    dist_arr = df_table["eta2_dist_center"].values
    ang_arr = df_table["eta2_angular_vel"].values
    cad_arr = df_table["eta2_cadence"].values
    comp_arr = df_table["eta2_composite"].values
    k_arr = df_table["active_k"].values

    r_speed, p_speed = pearsonr(acc_arr, speed_arr)
    r_dist, p_dist = pearsonr(acc_arr, dist_arr)
    r_ang, p_ang = pearsonr(acc_arr, ang_arr)
    r_cad, p_cad = pearsonr(acc_arr, cad_arr)
    r_comp, p_comp = pearsonr(acc_arr, comp_arr)

    rho_speed, _ = spearmanr(acc_arr, speed_arr)
    rho_comp, _ = spearmanr(acc_arr, comp_arr)
    r_k_acc, _ = pearsonr(k_arr, acc_arr)

    print("\n" + "="*95)
    print("=== BROAD SCALE SWEEP MASTER TABLE & COMPOSITE KINEMATIC CORRELATION ANALYSIS ===")
    print("="*95)
    print(df_table[["model_name", "active_k", "group_decodability_acc_pct", "eta2_speed", "eta2_dist_center", "eta2_angular_vel", "eta2_cadence", "eta2_composite"]].to_string(index=False))
    print("="*95)
    print("\n1. Group Decodability Acc vs Locomotion Speed (eta2_speed):")
    print(f"   - Pearson Correlation  r = {r_speed:+.3f} (p = {p_speed:.4f})")
    print(f"   - Spearman Correlation r = {rho_speed:+.3f}")
    print("\n2. Group Decodability Acc vs Thigmotaxis Distance (eta2_dist_center):")
    print(f"   - Pearson Correlation  r = {r_dist:+.3f} (p = {p_dist:.4f})")
    print("\n3. Group Decodability Acc vs Turning Rate (eta2_angular_vel):")
    print(f"   - Pearson Correlation  r = {r_ang:+.3f} (p = {p_ang:.4f})")
    print("\n4. Group Decodability Acc vs Syllable Cadence (eta2_cadence):")
    print(f"   - Pearson Correlation  r = {r_cad:+.3f} (p = {p_cad:.4f})")
    print("\n5. 🌟 OVERALL COMPOSITE KINEMATIC INDEX (eta2_composite vs Decodability Acc):")
    print(f"   - Pearson Correlation  r = {r_comp:+.3f} (p = {p_comp:.4f})")
    print(f"   - Spearman Correlation r = {rho_comp:+.3f}")
    print(f"\n6. Active Vibes (K) vs Decodability Acc: r = {r_k_acc:+.3f}")
    print("="*95)

if __name__ == "__main__":
    main()
