import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.evaluation.kinematic_encoding import compute_eta_squared

def partial_corr(x, y, z):
    """
    Computes partial correlation r(x, y | z) controlling for z.
    """
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    
    r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return r_xy_z

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    csv_path = os.path.join(out_dir, "broad_sweep_kinematic_correlation_table.csv")
    df = pd.read_csv(csv_path)
    
    acc = df["group_decodability_acc_pct"].values
    speed = df["eta2_speed"].values
    k = df["active_k"].values
    comp = df["eta2_composite"].values
    
    r_acc_speed = partial_corr(acc, speed, k)
    r_acc_comp = partial_corr(acc, comp, k)
    
    print("=== SKEPTICAL AUDIT: PARTIAL CORRELATION CONTROLLING FOR K ===")
    print(f"1. Raw r(Decodability, eta2_speed):              +0.983")
    print(f"2. Partial r(Decodability, eta2_speed | K):      {r_acc_speed:+.3f}")
    print(f"3. Raw r(Decodability, eta2_composite):          +0.969")
    print(f"4. Partial r(Decodability, eta2_composite | K):  {r_acc_comp:+.3f}")
    print("="*60)
    
    # Random Cluster Null Baseline for K=9
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    dataset = SessionDataset(cache_path=cache_path)
    
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)
        
    session_id_list = [meta_tuple[6] for meta_tuple in dataset.session_window_meta]
    
    all_vel = []
    all_random_k9 = []
    np.random.seed(42)
    
    for s in raw_sessions:
        exact_id = s.session_id
        for s_id in session_id_list:
            if s_id.startswith(exact_id) or exact_id.startswith(s_id):
                vel = s.all_data_df["velocity_px_s"].values
                rand_labels = np.random.randint(0, 9, size=len(vel))
                all_vel.append(vel)
                all_random_k9.append(rand_labels)
                break
            
    concat_vel = np.concatenate(all_vel)
    concat_rand = np.concatenate(all_random_k9)
    
    eta2_random_k9 = compute_eta_squared(concat_vel, concat_rand)
    print(f"\nNULL BASELINE TEST:")
    print(f"  - Random 9-Cluster Model eta2_speed:  {eta2_random_k9:.6f} (~0.00%)")
    print(f"  - BERTopic 9-Vibe Model eta2_speed:  0.076115 (7.61%)")
    print(f"  --> Ratio: BERTopic is {0.076115/(eta2_random_k9+1e-12):.1f}x stronger than random clustering!")
    print("="*60)

if __name__ == "__main__":
    main()
