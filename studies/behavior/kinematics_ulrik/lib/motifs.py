import copy
import os

import numpy as np
import pandas as pd
from fpcode.behavior.bouts import run_boundaries
from fpcode.common.paths import get_repo_root


def get_label_mapping(sessions, project="AMPH", style="standard", *, manual_labels_path=None) -> dict:
    """
    Returns a dict mapping syllable ID (int) to a string label.
    Styles: 'raw', 'standard', 'alphabet'
    """
    if style == "alphabet":
        all_counts = pd.Series(dtype=int)
        for s in sessions:
            if "syllable" in s.all_data_df.columns:
                counts = s.all_data_df["syllable"].value_counts()
                all_counts = all_counts.add(counts, fill_value=0)
        top_20 = all_counts.sort_values(ascending=False).head(20).index.astype(int).tolist()

        import string
        letters = string.ascii_uppercase
        mapping = {}
        for i, syll in enumerate(top_20):
            if i < len(letters):
                mapping[syll] = letters[i]
        return mapping

    from .label_mapping import load_manual_labels

    csv_path = (manual_labels_path if manual_labels_path is not None else
                get_repo_root() / "syllable_generation/analysis/ulrik_kinematics/manual_labels.csv")
    raw_mappings = load_manual_labels(csv_path)
    if project not in raw_mappings:
        print(f"Warning: Project {project} not found in manual labels.")
        return {}

    mapping = raw_mappings[project].copy()

    if style == "standard":
        for sid in list(mapping.keys()):
            lbl = str(mapping[sid]).lower()
            if "left" in lbl: mapping[sid] = "left turn"
            elif "right" in lbl or "run" in lbl: mapping[sid] = "right turn"
            elif "walk" in lbl or "dash" in lbl: mapping[sid] = "walk/dash"
            elif "rear" in lbl or "lean" in lbl: mapping[sid] = "rear"
            elif "misc" in lbl or "no video" in lbl:
                del mapping[sid]

    return mapping

def extract_ngrams(session, n: int = 3, mapping: dict = None) -> pd.Series:
    """
    Extracts n-length sequence motifs from a session's syllable stream.
    If mapping is provided, converts IDs to strings BEFORE run-length encoding
    to collapse true semantic transitions (e.g. walk -> walk).
    """
    df = session.all_data_df
    if "syllable" not in df.columns: return pd.Series()

    raw_seq = df["syllable"].fillna(-1).to_numpy()
    if len(raw_seq) < n: return pd.Series()

    semantic_seq = []
    for x in raw_seq:
        if x == -1:
            semantic_seq.append(None)
            continue

        if mapping is not None:
             lbl = mapping.get(int(x), None)
             semantic_seq.append(lbl)
        else:
             semantic_seq.append(int(x))

    semantic_seq = np.array(semantic_seq)

    # Run-length encode (collapse identical adjacent labels)
    starts, _ = run_boundaries(semantic_seq)
    transitions = semantic_seq[starts]

    if len(transitions) < n: return pd.Series()

    windows = zip(*[transitions[i:len(transitions)-n+1+i] for i in range(n)])

    valid_ngrams = [seq for seq in windows if None not in seq]

    if not valid_ngrams: return pd.Series(dtype=int)

    ngram_strings = [" -> ".join(map(str, seq)) for seq in valid_ngrams]
    return pd.Series(ngram_strings).value_counts()

def get_top_motifs(sessions, n: int = 3, top_k: int = 10, mapping: dict = None) -> pd.DataFrame:
    """
    Aggregates motifs across multiple sessions.
    """
    all_counts = pd.Series(dtype=int)

    for s in sessions:
        counts = extract_ngrams(s, n, mapping=mapping)
        if counts.empty: continue
        all_counts = all_counts.add(counts, fill_value=0)

    if all_counts.empty: return pd.DataFrame()

    # Sort and get top K
    top_counts = all_counts.sort_values(ascending=False).head(top_k)

    # Normalize to probability (relative to total valid n-grams found)
    total_ngrams = all_counts.sum()

    df = pd.DataFrame({
        "Motif": top_counts.index,
        "Support": top_counts.values.astype(int),
        "Probability": top_counts.values / total_ngrams
    })
    return df

def calculate_sequence_entropy(sessions) -> float:
    """Wrapper to calculate stationary entropy via existing metrics"""
    from syllable_generation.analysis.ulrik_kinematics.lib import metrics
    # Assuming run_analysis path logic allows this

    h_sum = 0
    count = 0
    for s in sessions:
        res = metrics.calculate_entropy_metrics(s)
        h = res.get("entropy_transition", np.nan)
        if not np.isnan(h):
            h_sum += h
            count += 1
    return h_sum / count if count > 0 else np.nan


def run_discovery(sessions, project_name="AMPH", style="standard", plot_mode="all"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import mannwhitneyu

    print(f"\n--- Running Motif Discovery ({style} style, mode={plot_mode}) ---")
    base_project = project_name.split("_")[0]  # Strip phase suffix (e.g. AMPH_pre -> AMPH)
    mapping = get_label_mapping(sessions, project=base_project, style=style)

    # -------------------------------------------------------------
    # PHASE A: RAW TOP MOTIFS (Unfiltered implementation)
    # -------------------------------------------------------------
    if plot_mode in ["all", "top_motifs"]:
        df_rows_raw = []
        for s in sessions:
            g = s.metadata.get("genotype")
            d = s.metadata.get("drug", "N/A")
            if g in ["None", "UNK", None]: continue
            if d.lower().startswith("amph"): d = "Amphetamine"
            elif d.lower().startswith("sal"): d = "Saline"

            counts = extract_ngrams(s, n=3, mapping=mapping)
            total_s_tri = counts.sum() if not counts.empty else 0
            for motif, c in counts.items():
                if total_s_tri > 0:
                    df_rows_raw.append({
                        "Session": s.session_id,
                        "Genotype": g,
                        "Drug": d,
                        "Motif": motif,
                        "Probability": c / total_s_tri
                    })

        if df_rows_raw:
            df_raw = pd.DataFrame(df_rows_raw)

            # --- Project-Specific Layout Logic ---
            if base_project.upper() == "AMPH":
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                fig.suptitle(f"Top 6 Unfiltered Sequences by Group ({style})", fontsize=18, fontweight='bold')

                groups = [
                    ("WT", "Saline", axes[0, 0], "#2ecc71"),
                    ("KI", "Saline", axes[0, 1], "#e74c3c"),
                    ("WT", "Amphetamine", axes[1, 0], "#27ae60"),
                    ("KI", "Amphetamine", axes[1, 1], "#c0392b")
                ]

                for gen_flag, drug_flag, ax, color in groups:
                    df_group = df_raw[(df_raw["Genotype"] == gen_flag) & (df_raw["Drug"] == drug_flag)]
                    if df_group.empty:
                        ax.set_visible(False)
                        continue

                    mean_probs = df_group.groupby("Motif")["Probability"].mean().sort_values(ascending=False).head(6)
                    sns.barplot(x=mean_probs.values, y=mean_probs.index, ax=ax, color=color, orient="h")
                    ax.set_title(f"{gen_flag} - {drug_flag}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("")

            else: # IFDN or other non-drug projects
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                fig.suptitle(f"Top 6 Unfiltered Sequences by Genotype: {project_name} ({style})", fontsize=18, fontweight='bold')

                groups = [
                    ("WT", axes[0], "#2ecc71"),
                    ("KI", axes[1], "#e74c3c")
                ]

                for gen_flag, ax, color in groups:
                    df_group = df_raw[df_raw["Genotype"] == gen_flag]
                    if df_group.empty:
                        ax.set_visible(False)
                        continue

                    mean_probs = df_group.groupby("Motif")["Probability"].mean().sort_values(ascending=False).head(6)
                    sns.barplot(x=mean_probs.values, y=mean_probs.index, ax=ax, color=color, orient="h")
                    ax.set_title(f"Genotype: {gen_flag}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_dir = os.path.join(os.path.dirname(__file__), "../output/motifs")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_top_motifs_combined_{style}.png"), bbox_inches="tight")
            plt.close()

            print(f"Saved horizontal layout unfiltered Top Motif plots for {project_name} ({style}).")

    if plot_mode not in ["all", "divergence"]:
        return

    # -------------------------------------------------------------
    # PHASE B: DIVERGENCE MOTIFS (Two-Step Markov Filtered)
    # -------------------------------------------------------------
    # STEP 1: Global Filter (Expected vs Observed Markov Null Model)
    bigrams_all = pd.Series(dtype=float)
    trigrams_all = pd.Series(dtype=float)
    unigrams_all = pd.Series(dtype=float)
    for s in sessions:
        b = extract_ngrams(s, n=2, mapping=mapping)
        if not b.empty: bigrams_all = bigrams_all.add(b, fill_value=0)
        t = extract_ngrams(s, n=3, mapping=mapping)
        if not t.empty: trigrams_all = trigrams_all.add(t, fill_value=0)
        u = extract_ngrams(s, n=1, mapping=mapping)
        if not u.empty: unigrams_all = unigrams_all.add(u, fill_value=0)

    total_uni = unigrams_all.sum()
    total_tri = trigrams_all.sum()

    intentional_motifs = []
    ratio_map = {}
    for motif, count in trigrams_all.items():
        if count < 10: continue
        parts = [p.strip() for p in motif.split("->")]
        if len(parts) != 3: continue
        A, B, C = parts[0], parts[1], parts[2]

        p_A = unigrams_all.get(A, 0) / total_uni if total_uni > 0 else 0
        starts_with_A = sum(v for k, v in bigrams_all.items() if k.startswith(f"{A} ->"))
        p_B_given_A = bigrams_all.get(f"{A} -> {B}", 0) / starts_with_A if starts_with_A > 0 else 0

        starts_with_B = sum(v for k, v in bigrams_all.items() if k.startswith(f"{B} ->"))
        p_C_given_B = bigrams_all.get(f"{B} -> {C}", 0) / starts_with_B if starts_with_B > 0 else 0

        p_exp = p_A * p_B_given_A * p_C_given_B
        p_obs = count / total_tri if total_tri > 0 else 0
        ratio = p_obs / p_exp if p_exp > 0 else 0

        if ratio >= 1.5:  # The non-random filter requirement!
            intentional_motifs.append(motif)
            ratio_map[motif] = ratio

    print(f"Step 1 Complete: Filtered down to {len(intentional_motifs)} highly intentional motifs.")

    # STEP 2: Cross-Phenotype Session Extraction & Mann-Whitney Test
    df_rows = []
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"

        counts = extract_ngrams(s, n=3, mapping=mapping)
        total_s_tri = counts.sum() if not counts.empty else 0
        for motif in intentional_motifs:
            prob = counts.get(motif, 0) / total_s_tri if total_s_tri > 0 else 0
            df_rows.append({
                "Session": s.session_id,
                "Genotype": g,
                "Drug": d,
                "Motif": motif,
                "Probability": prob,
                "Ratio": ratio_map[motif]
            })

    if not df_rows:
        print("No valid intentional motifs found!")
        return

    df_all = pd.DataFrame(df_rows)

    # Plot individually per drug
    for drug in ["Saline", "Amphetamine"]:
        df_drug = df_all[df_all["Drug"] == drug]
        if df_drug.empty: continue

        # Find Top 6 dominant motifs within this specific drug environment
        mean_probs = df_drug.groupby("Motif")["Probability"].mean().sort_values(ascending=False)
        top_6 = mean_probs.head(6).index.tolist()

        df_plot = df_drug[df_drug["Motif"].isin(top_6)]

        # Calculate stats for the * markers
        stat_results = {}
        for m in top_6:
            wt_vals = df_plot[(df_plot["Motif"] == m) & (df_plot["Genotype"] == "WT")]["Probability"].values
            ki_vals = df_plot[(df_plot["Motif"] == m) & (df_plot["Genotype"] == "KI")]["Probability"].values
            if len(wt_vals) >= 3 and len(ki_vals) >= 3:
                # We specifically test if KI is DIFFERENT than WT (two-sided)
                stat, p_val = mannwhitneyu(wt_vals, ki_vals, alternative='two-sided')
                stat_results[m] = p_val
            else:
                stat_results[m] = 1.0

        plt.figure(figsize=(15, 8))
        ax = sns.barplot(
            data=df_plot, x="Motif", y="Probability", hue="Genotype",
            palette=["#2ecc71", "#e74c3c"] if df_plot["Genotype"].iloc[0]=='WT' else ["#e74c3c", "#2ecc71"],
            err_kws={'linewidth': 1.5}, capsize=0.1
        )

        # Annotate Significance Stars
        for i, m in enumerate(top_6):
            pval = stat_results.get(m, 1.0)
            if pval < 0.05:
                # Adjust asterisk height to be just above the bar/error, but capped at the top axis limit
                star = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
                y_max = df_plot[df_plot["Motif"] == m]["Probability"].max()
                y_pos = min(y_max * 1.02, ax.get_ylim()[1] * 0.95)
                ax.text(i, y_pos, star, ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')

        plt.title(f"Genotype Divergence: {drug} Motifs (Markov Filter > 1.5 | {style} labels)", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylabel("Probability of Occurrence", fontsize=14)
        plt.xlabel("")
        plt.tight_layout()

        out_dir = os.path.join(os.path.dirname(__file__), "../output/motifs")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_{drug.lower()}_divergence_{style}.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved 2-step divergence plots for {project_name} ({style}).")

def check_significance(sessions, project="AMPH", style="standard"):
    """
    Implements Method 1: The Markov Null Model checking.
    Calculates the 'Random Walk Ratio' by deriving expected Trigram probabilities
    from unigram bounds and bigram transition probabilities.
    """
    print(f"\n--- Motif Significance vs Markov Random Walk ({style}) ---")
    mapping = get_label_mapping(sessions, project=project, style=style)

    bigrams_all = pd.Series(dtype=int)
    trigrams_all = pd.Series(dtype=int)
    unigrams_all = pd.Series(dtype=int)

    for s in sessions:
        b = extract_ngrams(s, n=2, mapping=mapping)
        if not b.empty: bigrams_all = bigrams_all.add(b, fill_value=0)
        t = extract_ngrams(s, n=3, mapping=mapping)
        if not t.empty: trigrams_all = trigrams_all.add(t, fill_value=0)
        u = extract_ngrams(s, n=1, mapping=mapping)
        if not u.empty: unigrams_all = unigrams_all.add(u, fill_value=0)

    total_uni = unigrams_all.sum()
    total_tri = trigrams_all.sum()

    results = []

    # Assess the top 15 most frequent trigrams globally
    top_t = trigrams_all.sort_values(ascending=False).head(15)

    for motif, count in top_t.items():
        parts = [p.strip() for p in motif.split("->")]
        if len(parts) != 3: continue
        A, B, C = parts[0], parts[1], parts[2]

        p_A = unigrams_all.get(A, 0) / total_uni

        starts_with_A = sum(v for k, v in bigrams_all.items() if k.startswith(f"{A} ->"))
        p_B_given_A = bigrams_all.get(f"{A} -> {B}", 0) / starts_with_A if starts_with_A > 0 else 0

        starts_with_B = sum(v for k, v in bigrams_all.items() if k.startswith(f"{B} ->"))
        p_C_given_B = bigrams_all.get(f"{B} -> {C}", 0) / starts_with_B if starts_with_B > 0 else 0

        p_exp = p_A * p_B_given_A * p_C_given_B
        p_obs = count / total_tri

        ratio = p_obs / p_exp if p_exp > 0 else 0

        results.append({
            "Motif": motif,
            "Count": count,
            "Observed_%": p_obs * 100,
            "Expected_%": p_exp * 100,
            "Ratio (Obs/Exp)": ratio
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False, floatfmt=".3f"))

def verify_motifs_module(sessions):
    """
    Paired Verification Check: True Data vs Sequence Shuffle Perturbation.
    Prints diagnostic tables to standard output.
    """
    from syllable_generation.analysis.ulrik_kinematics import config
    print("\n" + "="*50)
    print(" MODULE VERIFICATION: MOTIFS & SHUFFLE PERTURBATION")
    print("="*50)

    # --- BASELINE RUN ---
    print("\n[1/2] Computing True Baseline Metrics...")
    top_true = get_top_motifs(sessions, n=config.MOTIF_LENGTH, top_k=config.TOP_K_MOTIFS)
    h_true = calculate_sequence_entropy(sessions)

    true_top_prob = top_true["Probability"].iloc[0] if not top_true.empty else 0

    # --- PERTURBATION RUN ---
    print("[2/2] Applying Sequence Shuffle Perturbation...")
    shuffled_sessions = copy.deepcopy(sessions)

    for s in shuffled_sessions:
        if "syllable" in s.all_data_df.columns:
            # We shuffle the syllables column to destroy temporal order
            # Note: This dramatically increases the # of transitions due to breaking bouts
            vals = s.all_data_df["syllable"].values
            np.random.shuffle(vals)
            s.all_data_df["syllable"] = vals

    top_shuff = get_top_motifs(shuffled_sessions, n=config.MOTIF_LENGTH, top_k=config.TOP_K_MOTIFS)
    h_shuff = calculate_sequence_entropy(shuffled_sessions)

    shuff_top_prob = top_shuff["Probability"].iloc[0] if not top_shuff.empty else 0

    # --- REPORTING ---
    print("\n" + "-"*30)
    print("       DIAGNOSTIC RESULTS       ")
    print("-"*30)

    table = pd.DataFrame({
        "Condition": ["True Data", "Shuffled Data (Perturbation)"],
        "Sequence_Entropy": [h_true, h_shuff],
        "Top_Motif_Probability": [true_top_prob, shuff_top_prob]
    })

    # Calculate % drop
    drop_pct = (1 - (shuff_top_prob / true_top_prob)) * 100 if true_top_prob > 0 else 0

    print(table.to_markdown(index=False, floatfmt=".3f"))

    print("\n> INVARIANT CHECKS:")
    print(f"  - Entropy Constraint: Shuffled Entropy ({h_shuff:.2f}) > True Entropy ({h_true:.2f}) -> {'PASS' if h_shuff > h_true else 'FAIL'}")
    print(f"  - Probability Constraint: Shuffled Top Motif prob dropped by {drop_pct:.1f}% -> {'PASS' if drop_pct > 80 else 'WARN'}")
    print("="*50 + "\n")
