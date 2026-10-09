import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path

# Setup Pathing
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
parent_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

from analysis.replications.panel_1z.lib import loader, processing, utils

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_TRANSITIONS = 100 # Minimum transitions per animal to include in entropy calc
MIN_ANIMALS = 3

def calculate_conditional_entropy(syllables):
    """
    Calculates H(S_t+1 | S_t).
    H = - sum_i P(i) * sum_j P(j|i) * log2(P(j|i))
    """
    # 1. Clean data (remove NaNs)
    syllables = syllables[~np.isnan(syllables)].astype(int)
    if len(syllables) < MIN_TRANSITIONS:
        return np.nan
    
    # 2. Get Bigrams
    s1 = syllables[:-1]
    s2 = syllables[1:]
    
    # 3. Build Transition Counts
    # We use a dataframe to pivot easily
    df = pd.DataFrame({'s_t': s1, 's_t_plus_1': s2})
    counts = df.groupby(['s_t', 's_t_plus_1']).size().unstack(fill_value=0)
    
    # 4. Probabilities
    # Row sums are P(i) * total_transitions
    row_sums = counts.sum(axis=1)
    p_i = row_sums / row_sums.sum()
    
    # Transition probabilities P(j|i)
    # Divide each row by its sum
    p_j_given_i = counts.div(row_sums, axis=0)
    
    # 5. Entropy calculation
    # sum_j P(j|i) * log2(P(j|i))
    # We use log2 to get "bits"
    # p*log(p) where p=0 is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log2(p_j_given_i)
        logs[np.isinf(logs)] = 0
        h_per_row = -(p_j_given_i * logs).sum(axis=1)
    
    # Weighted average by P(i)
    h_cond = (p_i * h_per_row).sum()
    
    return h_cond

def run_analysis(sessions, output_dir, project_name):
    logger.info(f"Starting Syllable Transition Entropy Analysis for {project_name}...")
    
    entropy_results = []
    
    for s in sessions:
        meta = s.metadata
        animal = meta.get("animal_id", "UNK")
        # Handle group mapping (Amph vs Saline or WT vs KI)
        geno = meta.get("group", meta.get("genotype", "UNK"))
        
        df = getattr(s, "all_data_df", None)
        if df is None or "syllable" not in df.columns:
            continue
            
        # We only care about unique syllable entries (transitions happen at boundaries)
        # But conditional entropy is usually calculated over the sequence.
        # If the mouse stays in a syllable for 10 frames, is that a transition to self?
        # Standard behavioral entropy often uses the *syllable sequence* (labels only).
        
        # Get sequence of syllables (remove consecutive repeats to analyze transitions between different syllables)
        # Or keep them to analyze "self-transitions" (dwell time randomness).
        # Let's keep them as the user might want to capture transition timing.
        # Actually, standard "Transition Entropy" usually looks at the "Syntax" (s1 -> s2 where s1 != s2).
        
        syll_full = df['syllable'].values
        # Ensure it is a clean numpy array (handle nullable pandas dtypes)
        if hasattr(syll_full, "to_numpy"):
            syll_full = syll_full.to_numpy(dtype=float, na_value=np.nan)
        else:
            syll_full = np.array(syll_full, dtype=float)
            
        # 1. Sequence Entropy (Syllable Syntax)
        # Remove consecutive duplicates
        valid_mask = ~np.isnan(syll_full)
        syll_clean = syll_full[valid_mask]
        
        if len(syll_clean) < 2: continue
        
        # Identify changes
        changes = np.concatenate(([True], syll_clean[1:] != syll_clean[:-1]))
        syll_seq = syll_clean[changes]
        
        h_syntax = calculate_conditional_entropy(syll_seq)
        
        # 2. Time-step Entropy (Full sequence, includes self-transitions)
        h_time = calculate_conditional_entropy(syll_clean)
        
        if not np.isnan(h_syntax):
            entropy_results.append({
                "Animal": animal,
                "Genotype": geno,
                "Entropy_Syntax": h_syntax,
                "Entropy_Time": h_time,
                "N_Transitions": len(syll_seq) - 1
            })
            
    if not entropy_results:
        logger.warning("No valid entropy data found!")
        return
        
    df_entropy = pd.DataFrame(entropy_results)
    
    # Output pathing
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Stats Calculation
    stats_rows = []
    groups = df_entropy["Genotype"].unique()
    
    # Plotting for both metrics
    for metric, label in [("Entropy_Syntax", "Syntactic Entropy (Syllable Sequence)"), 
                          ("Entropy_Time", "Temporal Entropy (Full Time-Series)")]:
        
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df_entropy, x="Genotype", y=metric, hue="Genotype", showfliers=False, palette="Set2")
        sns.stripplot(data=df_entropy, x="Genotype", y=metric, hue="Genotype", jitter=True, size=8, alpha=0.7, palette="Set2")
        
        # Pairwise Stats
        import itertools
        p_vals = {}
        for g1, g2 in itertools.combinations(sorted(groups), 2):
            v1 = df_entropy[df_entropy["Genotype"] == g1][metric]
            v2 = df_entropy[df_entropy["Genotype"] == g2][metric]
            if len(v1) >= MIN_ANIMALS and len(v2) >= MIN_ANIMALS:
                stat, p = stats.mannwhitneyu(v1, v2)
                p_vals[f"{g1}_vs_{g2}"] = p
                
                stats_rows.append({
                    "Project": project_name,
                    "Metric": label,
                    "Comparison": f"{g1}_vs_{g2}",
                    "p_value": p,
                    "Mean_G1": v1.mean(),
                    "Mean_G2": v2.mean()
                })
        
        title = f"{project_name}: {label}"
        if p_vals:
            min_p = min(p_vals.values())
            title += f"\n(Min p={min_p:.1e})"
        
        plt.title(title)
        plt.ylabel("Entropy (Bits)")
        plt.tight_layout()
        
        m_tag = metric.lower()
        plt.savefig(os.path.join(output_dir, "plots", f"entropy_{m_tag}_{project_name}.png"))
        plt.close()

    # Save Stats CSV
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        csv_path = os.path.join(output_dir, f"entropy_stats_{project_name}.csv")
        df_stats.to_csv(csv_path, index=False)
        logger.info(f"Saved stats to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, choices=["ifdn", "amph"])
    args = parser.parse_args()
    
    # Set paths based on project
    if args.project == "ifdn":
        pkl = "data/jonathan_ifdn_sessions_w_sylls.pkl"
    else:
        pkl = "data/jonathan_amph_sessions_v2_w_sylls.pkl"
        
    out = "analysis/deep_research_experiments/syllable_entropy/output"
    
    sessions = loader.load_sessions(args.project, pkl)
    
    # For AMPH, apply post-injection filter
    if args.project == "amph":
        sessions = processing.filter_post_injection(sessions, delay_mins=10)
        
    run_analysis(sessions, out, args.project)
