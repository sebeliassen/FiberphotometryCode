import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statannotations.Annotator import Annotator

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add root to path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader, metrics

# Setup plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_METRICS = ["mean_velocity", "mean_acceleration", "entropy_transition", "stereotypy", "entropy_duration"]
MIN_INSTANCES = 15  # Minimum bouts for a syllable to be robust

def remove_outliers_iqr(df, column, factor=3.0):
    """
    Remove outliers from a DataFrame based on IQR.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column to filter.
        factor (float): IQR multiplier (default 3.0 for robust exclusion).
        
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    if df.empty or column not in df.columns: return df
    
    # Calculate IQR
    # Use dropna to compute stats
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    
    # Filter
    return df[(df[column] >= lower) & (df[column] <= upper)]

def add_stats_stars(ax, df, x, y, order, pairs, hue=None):
    """
    Add statistical significance stars to a box/violin plot.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        df (pd.DataFrame): Data.
        x (str): X-axis variable.
        y (str): Y-axis variable.
        order (list): Order of x-axis categories.
        pairs (list): List of pairs to compare.
        hue (str, optional): Hue variable.
    """
    try:
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, hue=hue, order=order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Stats error for {y}: {e}")

def plot_usage(df, project_name, ordered_sylls=None):
    """
    Plot syllable usage barplot for top syllables.
    
    Args:
        df (pd.DataFrame): Usage data.
        project_name (str): Name of the project (e.g., 'IFDN').
        ordered_sylls (list, optional): List of syllable IDs to force order/inclusion.
    """
    # Filter "None" / "UNK"
    df = df[~df["Genotype"].astype(str).isin(["None", "UNK", "nan"])]
    
    if ordered_sylls is not None:
        top_sylls = ordered_sylls
    else:
        # Fallback to dynamic rank-sum calculation
        # 1. Avg frequency per group per syllable
        group_cols = ["Genotype"]
        if "Drug" in df.columns: group_cols.append("Drug")
        grouped = df.groupby(group_cols + ["Syllable"])["Frequency"].mean().reset_index()
        # 2. Rank within group
        grouped["Rank"] = grouped.groupby(group_cols)["Frequency"].rank(method='min', ascending=False)
        # 3. Sum ranks and take top 15
        top_sylls = grouped.groupby("Syllable")["Rank"].sum().sort_values().head(15).index.tolist()

    plot_df = df[df["Syllable"].isin(top_sylls)]
    # Ensure categorical sorting for X-axis matching the rank order
    plot_df['Syllable'] = pd.Categorical(plot_df['Syllable'], categories=top_sylls, ordered=True)
    
    plt.figure(figsize=(15, 6))
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        # Complex case: Combine Genotype and Drug for hue
        plot_df["Group_Tag"] = plot_df["Genotype"] + " " + plot_df["Drug"]
        hue_order = ["WT Saline", "WT Amphetamine", "KI Saline", "KI Amphetamine"]
        # Filters to ensure hue_order exists in data
        existing_groups = [g for g in hue_order if g in plot_df["Group_Tag"].unique()]
        sns.barplot(data=plot_df, x="Syllable", y="Frequency", hue="Group_Tag", hue_order=existing_groups)
    else:
        sns.barplot(data=plot_df, x="Syllable", y="Frequency", hue="Genotype", hue_order=["WT", "KI"])
        
    plt.title(f"Syllable Usage (Fair Rank Sort) - {project_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_usage.png"))
    plt.close()


def plot_transition_deltas(sessions, project_name, ordered_sylls=None):
    """
    Plot Delta Heatmaps (Amph - Saline) per Genotype.
    
    Args:
        sessions (list): List of Session objects.
        project_name (str): Project identifier.
        ordered_sylls (list, optional): Syllables to include/order.
    """
    if project_name.lower() != "amph": return
    
    # 1. Group sessions by (Genotype, Drug)
    groups = {}
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        
        # Standardize labels for plotting
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        
        key = (g, d)
        if key not in groups: groups[key] = []
        groups[key].append(s)
        
    genotypes = sorted(list(set([k[0] for k in groups.keys()])))
    
    # Use provided syllables or top 12 global
    if ordered_sylls is not None:
        top_sylls_list = ordered_sylls[:12] # Limit for heatmap readability
    else:
        all_sylls = []
        for s in sessions:
            if "syllable" in s.all_data_df.columns:
                all_sylls.append(s.all_data_df["syllable"].dropna())
        global_counts = pd.concat(all_sylls).value_counts()
        top_sylls_list = global_counts.head(12).index.tolist()
    
    fig, axes = plt.subplots(1, len(genotypes), figsize=(6 * len(genotypes), 5))
    if len(genotypes) == 1: axes = [axes]
    
    for i, geno in enumerate(genotypes):
        sess_sal = groups.get((geno, "Saline"), [])
        sess_amph = groups.get((geno, "Amphetamine"), [])
        
        ax = axes[i]
        if not sess_sal or not sess_amph: 
            ax.set_title(f"{geno}: Insufficient Data")
            continue
        
        m_sal = metrics.get_group_transition_matrix(sess_sal, ordered_syllables=top_sylls_list)
        m_amph = metrics.get_group_transition_matrix(sess_amph, ordered_syllables=top_sylls_list)
        
        delta = m_amph - m_sal
        sns.heatmap(delta, ax=ax, cmap="coolwarm", center=0, annot=False)
        ax.set_title(f"{geno}: Amph - Saline")
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_transition_deltas.png"))
    plt.close()

def plot_transition_matrices(sessions, project_name, ordered_sylls=None):
    """
    Plot transition matrix heatmaps for each major group.
    
    Args:
        sessions (list): List of Session objects.
        project_name (str): Project identifier.
        ordered_sylls (list, optional): Syllables to include/order.
    """
    groups = {}
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        
        key = (g, d)
        if key not in groups: groups[key] = []
        groups[key].append(s)
    
    # Sort keys
    keys = sorted(groups.keys())
    
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    if len(keys) == 1: axes = [axes]
    
    # Determine syllables to use for heatmap
    if ordered_sylls is not None:
        heatmap_sylls = ordered_sylls[:12] # Heatmaps get messy with too many states
    else:
        heatmap_sylls = None
    
    for i, key in enumerate(keys):
        matrix = metrics.get_group_transition_matrix(groups[key], top_n=12, ordered_syllables=heatmap_sylls)
        sns.heatmap(matrix, ax=axes[i], cmap="rocket_r", annot=False)
        
        # Clean title: omit drug if N/A
        title = f"{key[0]}"
        if key[1] != "N/A":
            title += f" {key[1]}"
        axes[i].set_title(title)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_transitions.png"))
    plt.close()

def run_ifdn():
    print("Loading IFDN Dataset...")
    sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

    rows_ifdn = []
    for s in tqdm(sessions_ifdn, desc="Processing IFDN"):
        rows_ifdn.append(metrics.analyze_session(s))
        
    df_ifdn = pd.DataFrame(rows_ifdn)
    # Filter
    df_ifdn = df_ifdn[~df_ifdn["Genotype"].astype(str).isin(["None", "UNK", "nan"])]
    df_ifdn.to_csv(os.path.join(OUTPUT_DIR, "ifdn_kinematics_stats.csv"), index=False)

    # 1. Summary Boxplots with Stats
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    order = ["WT", "KI"]
    pairs = [("WT", "KI")]
    
    for i, m in enumerate(PLOT_METRICS):
        ax = axes[i]
        sns.boxplot(data=df_ifdn, x="Genotype", y=m, ax=ax, width=0.5, palette="Set2", showfliers=False, order=order)
        sns.stripplot(data=df_ifdn, x="Genotype", y=m, ax=ax, color="black", alpha=0.7, order=order)
        add_stats_stars(ax, df_ifdn, "Genotype", m, order, pairs)
        ax.set_title(m)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ifdn_summary_panel.png"))
    plt.close()

    # 2. Usage Barplot
    # Fair Top list for IFDN
    ifdn_fair_sylls = [0, 1, 2, 3, 57, 36, 15, 25, 11, 43, 27, 12, 10, 34, 24, 56, 32, 64, 45, 16]
    df_usage = metrics.get_group_usage(sessions_ifdn, min_instances=MIN_INSTANCES)
    plot_usage(df_usage, "IFDN", ordered_sylls=ifdn_fair_sylls)
    
    # 3. Transitions
    plot_transition_matrices(sessions_ifdn, "IFDN", ordered_sylls=ifdn_fair_sylls)

def run_amph():
    print("Loading AMPH Dataset (Post-Injection Filtered)...")
    sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)

    rows_amph = []
    for s in tqdm(sessions_amph, desc="Processing AMPH"):
        rows_amph.append(metrics.analyze_session(s))

    df_amph = pd.DataFrame(rows_amph)
    # Filter
    df_amph = df_amph[~df_amph["Genotype"].astype(str).isin(["None", "UNK", "nan"])]
    df_amph.to_csv(os.path.join(OUTPUT_DIR, "amph_kinematics_stats.csv"), index=False)

    # 1. Summary Boxplots
    # Fix pairs for stats
    # Group_Tag is crucial
    df_amph["Drug"] = df_amph["Drug"].fillna("N/A")
    df_amph["Group_Tag"] = df_amph["Genotype"] + " " + df_amph["Drug"]
    
    # Pairs for T-Tests inside the plot
    # Logic: Boxplot x=Genotype, hue=Drug. 
    # Annotator needs x=Genotype, hue=Drug if we want standard dodging.
    # BUT annotating across hues is tricky.
    # Easier: x="Group_Tag", no hue? No, user wants grouped boxplots.
    # Using x=Genotype, hue=Drug with Annotator support.
    
    pairs = [
        (("WT", "Saline"), ("WT", "Amphetamine")),
        (("KI", "Saline"), ("KI", "Amphetamine")),
        (("WT", "Saline"), ("KI", "Saline")),
        (("WT", "Amphetamine"), ("KI", "Amphetamine"))
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for i, m in enumerate(PLOT_METRICS):
        ax = axes[i]
        sns.boxplot(data=df_amph, x="Genotype", y=m, hue="Drug", ax=ax, palette="Set1", showfliers=False)
        sns.stripplot(data=df_amph, x="Genotype", y=m, hue="Drug", ax=ax, dodge=True, color="black", alpha=0.5)
        
        # Add stars
        add_stats_stars(ax, df_amph, "Genotype", m, order=["WT", "KI"], pairs=pairs, hue="Drug")
        
        ax.set_title(m)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "amph_summary_panel.png"))
    plt.close()

    # 2. Usage
    # Fair Top list for AMPH (T=7)
    amph_fair_sylls = [0, 1, 2, 3, 58, 4, 35, 65, 62, 29, 70, 86, 63, 80, 95, 83, 72, 89, 43, 96]
    df_usage = metrics.get_group_usage(sessions_amph, min_instances=MIN_INSTANCES)
    plot_usage(df_usage, "AMPH", ordered_sylls=amph_fair_sylls)
    
    # 3. Transitions & Deltas !
    plot_transition_matrices(sessions_amph, "AMPH", ordered_sylls=amph_fair_sylls)
    plot_transition_deltas(sessions_amph, "AMPH", ordered_sylls=amph_fair_sylls)

def plot_manual_usage(df, project_name):
    """Syllable usage barplot for MANUAL labels (Time and Onsets)."""
    
    # FILTERING
    # Remove "Other", "no video", empty strings, NaNs
    excluded = ["Other", "no video", "", "nan"]
    df = df[~df["Label"].isin(excluded)]
    df = df[~df["Label"].isna()]
    
    # 1. Plot Time Spent
    # Order by total mean Time_Pct
    order_time = df.groupby("Label")["Time_Pct"].mean().sort_values(ascending=False).index.tolist()
    
    plt.figure(figsize=(15, 6))
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        df["Group_Tag"] = df["Genotype"] + " " + df["Drug"]
        hue_order = ["WT Saline", "WT Amphetamine", "KI Saline", "KI Amphetamine"]
        existing = [g for g in hue_order if g in df["Group_Tag"].unique()]
        sns.barplot(data=df, x="Label", y="Time_Pct", hue="Group_Tag", hue_order=existing, order=order_time)
    else:
        sns.barplot(data=df, x="Label", y="Time_Pct", hue="Genotype", hue_order=["WT", "KI"], order=order_time)
        
    plt.title(f"Manual Label Usage (Time Spent %) - {project_name}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Time Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_manual_usage_time.png"))
    plt.close()
    
    # 2. Plot N Onsets
    # Order by total mean Onset_Count
    order_onset = df.groupby("Label")["Onset_Count"].mean().sort_values(ascending=False).index.tolist()
    
    plt.figure(figsize=(15, 6))
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        sns.barplot(data=df, x="Label", y="Onset_Count", hue="Group_Tag", hue_order=existing, order=order_onset)
    else:
        sns.barplot(data=df, x="Label", y="Onset_Count", hue="Genotype", hue_order=["WT", "KI"], order=order_onset)
        
    plt.title(f"Manual Label Usage (N Onsets) - {project_name}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Onsets (Bouts per Session)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_manual_usage_onsets.png"))
    plt.close()

def get_significant_changes(matrix_delta, dist_a, dist_b, n_limit=10, interaction=False):
    """
    Find significant transitions using Mann-Whitney U test.
    Returns sorted top N significant changes.
    """
    # 1. Stack Delta
    stacked = matrix_delta.stack().reset_index()
    stacked.columns = ["Source", "Dest", "Delta"]
    stacked["AbsDelta"] = stacked["Delta"].abs()
    
    # 2. Calculate P-Values for ALL transitions
    p_values = []
    
    for _, row in stacked.iterrows():
        src, dst = row["Source"], row["Dest"]
        
        # dist_a/b are lists of matrices (aligned)
        vals_a = [m.loc[src, dst] for m in dist_a]
        vals_b = [m.loc[src, dst] for m in dist_b]
        
        # Mann-Whitney U Test
        # Requires at least one value
        if len(vals_a) > 0 and len(vals_b) > 0:
            try:
                # alternative='two-sided' is default but explicit is good
                _, p = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            except ValueError:
                # e.g. all values identical
                p = 1.0
        else:
            p = 1.0
            
        p_values.append(p)
        
    stacked["PValue"] = p_values
    
    # 3. Filter Significant only
    sig = stacked[stacked["PValue"] < 0.05].copy()
    
    # 4. Sort by Magnitude
    sig = sig.sort_values("AbsDelta", ascending=False)
    
    # 5. Limit to Top N
    top = sig.head(n_limit).copy()
    
    # Add metadata
    top["Direction"] = top["Delta"].apply(lambda x: "Increase" if x > 0 else "Decrease")
    top["Transition"] = top["Source"] + " -> " + top["Dest"]
    
    return top

def plot_bar_changes(top_df, title, filename):
    """Auxiliary plotter for top changes with p-value stars."""
    
    # Check if empty
    if top_df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No Significant Changes (p < 0.05)", ha='center', va='center')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        return

    # Dynamic Height
    height = max(5, len(top_df) * 0.6)
    plt.figure(figsize=(10, height))
    
    palette = {"Increase": "tab:red", "Decrease": "tab:blue"}
    ax = sns.barplot(data=top_df, x="Delta", y="Transition", hue="Direction", dodge=False, palette=palette)
    plt.title(title)
    plt.axvline(0, color="black", linewidth=1)
    
    # Annotate stars (Since we filter p<0.05, all are at least *)
    for i, (idx, row) in enumerate(top_df.iterrows()):
        p = row["PValue"]
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        else: sig = "ns" # Should not happen given filter
        
        x_val = row["Delta"]
        offset = 0.01 * x_val if x_val != 0 else 0.005
        # Ensure offset is away from zero
        if x_val > 0: offset = 0.002
        else: offset = -0.002
        
        ha = 'left' if x_val >= 0 else 'right'
        
        # Add numeric p-value optionally? No, sig stars requested previously.
        ax.text(x_val + offset, i, sig, va='center', ha=ha, fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_top_manual_changes(sessions, mapping, project_name):
    """
    Calculate and plot top 10 SIGNIFICANT changing transitions.
    Using Mann-Whitney U test.
    """
    # 1. Group sessions
    groups = {}
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        
        # Standardize labels
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        
        key = (g, d)
        if key not in groups: groups[key] = []
        groups[key].append(s)
        
    keys = sorted(groups.keys())
    
    # 2. Calculate Matrices and Distributions
    matrices = {}
    distributions = {} # {key: list of matrices}
    all_labels = set()
    
    for k in keys:
        res = metrics.get_manual_group_transition_matrix(groups[k], mapping, return_distributions=True)
        m_mean, m_dist = res
        
        # Filter unwanted labels
        to_drop = [l for l in ["Other", "no video", "nan"] if l in m_mean.index]
        # Drop columns/rows
        m_mean = m_mean.drop(to_drop, axis=0).drop(to_drop, axis=1)
        # Drop from dists
        cleaned_dists = []
        for m in m_dist:
            # We must be careful to drop if exists
            m_clean = m.copy()
            valid_drop = [x for x in to_drop if x in m_clean.index]
            if valid_drop:
                m_clean = m_clean.drop(valid_drop, axis=0).drop(valid_drop, axis=1)
            cleaned_dists.append(m_clean)
            
        matrices[k] = m_mean
        distributions[k] = cleaned_dists
        all_labels.update(m_mean.index)
        
    all_labels = sorted(list(all_labels))
    
    # 3. Align Matrices and Distributions
    aligned_matrices = {}
    aligned_dists = {}
    for k in matrices:
        aligned_matrices[k] = matrices[k].reindex(index=all_labels, columns=all_labels).fillna(0)
        # Align dists
        curr_dists = distributions[k]
        aligned_list = []
        for m in curr_dists:
            aligned_list.append(m.reindex(index=all_labels, columns=all_labels).fillna(0))
        aligned_dists[k] = aligned_list
        
    # 5. Stats & Plotting Logic (Replaced calc_stats)
    
    # 4. Deltas and Plots
    if project_name == "IFDN":
        k_wt, k_ki = ("WT", "N/A"), ("KI", "N/A")
        if k_wt in aligned_matrices and k_ki in aligned_matrices:
            delta = aligned_matrices[k_ki] - aligned_matrices[k_wt]
            # Use new function
            top = get_significant_changes(delta, aligned_dists[k_wt], aligned_dists[k_ki])
            plot_bar_changes(top, "IFDN: KI - WT (Significant Changes p<0.05)", "ifdn_manual_top_changes.png")
            
    elif project_name == "AMPH":
        k_wt_sal, k_wt_amph = ("WT", "Saline"), ("WT", "Amphetamine")
        k_ki_sal, k_ki_amph = ("KI", "Saline"), ("KI", "Amphetamine")
        
        # WT Delta
        if k_wt_sal in aligned_matrices and k_wt_amph in aligned_matrices:
            delta_wt = aligned_matrices[k_wt_amph] - aligned_matrices[k_wt_sal]
            top_wt = get_significant_changes(delta_wt, aligned_dists[k_wt_sal], aligned_dists[k_wt_amph])
            plot_bar_changes(top_wt, "WT: Amph - Saline (Sig p<0.05)", "amph_manual_top_changes_wt.png")
        else: delta_wt = None

        # KI Delta
        if k_ki_sal in aligned_matrices and k_ki_amph in aligned_matrices:
            delta_ki = aligned_matrices[k_ki_amph] - aligned_matrices[k_ki_sal]
            top_ki = get_significant_changes(delta_ki, aligned_dists[k_ki_sal], aligned_dists[k_ki_amph])
            plot_bar_changes(top_ki, "KI: Amph - Saline (Sig p<0.05)", "amph_manual_top_changes_ki.png")
        else: delta_ki = None
            
        # Interaction
        if delta_wt is not None and delta_ki is not None:
            # Stats for interaction
            # Interaction T-Test -> Mann Whitney on Deltas?
            # Response_G = [ (S_amph - Mean(S_sal_group)) ... ]
            
            # Calculate mean saline matrices
            mean_wt_sal = aligned_matrices[k_wt_sal]
            mean_ki_sal = aligned_matrices[k_ki_sal]
            
            # Helper to get deltas relative to group mean baseline
            def get_group_deltas(dists_amph, mean_sal):
                return [m - mean_sal for m in dists_amph]
                
            dist_delta_wt = get_group_deltas(aligned_dists[k_wt_amph], mean_wt_sal)
            dist_delta_ki = get_group_deltas(aligned_dists[k_ki_amph], mean_ki_sal)
            
            # delta_ki vs delta_wt
            diff_diff = aligned_matrices[k_ki_amph] - aligned_matrices[k_ki_sal] - (aligned_matrices[k_wt_amph] - aligned_matrices[k_wt_sal])
            
            # Mann Whitney on the deltas
            top_int = get_significant_changes(diff_diff, dist_delta_wt, dist_delta_ki, interaction=True)
            plot_bar_changes(top_int, "Interaction (KI Delta - WT Delta) (Sig p<0.05)", "amph_manual_top_changes_interaction.png")

def plot_manual_transitions(sessions, mapping, project_name):
    """Plot transition matrix heatmaps for manual labels."""
    groups = {}
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        
        key = (g, d)
        if key not in groups: groups[key] = []
        groups[key].append(s)
    
    keys = sorted(groups.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 6))
    if len(keys) == 1: axes = [axes]
    
    # To keep matrices comparable, we need a fixed order of labels used in mapping
    # excluding "Other" if desired, or all unique values.
    # Let's order specific labels by name or user definition order? 
    # CSV order?
    # Let's trust metrics.get_manual which sorts alphabetically. 
    # Or better: Sort by alphabetical but put 'Other' last?
    
    for i, key in enumerate(keys):
        matrix = metrics.get_manual_group_transition_matrix(groups[key], mapping)
        # Drop Other?
        if "Other" in matrix.index:
            try:
                matrix = matrix.drop("Other", axis=0).drop("Other", axis=1)
            except: pass
            
        sns.heatmap(matrix, ax=axes[i], cmap="rocket_r", annot=False)
        
        # Clean title
        title = f"{key[0]}"
        if key[1] != "N/A":
            title += f" {key[1]}"
        axes[i].set_title(title)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_manual_transitions.png"))
    plt.close()

def run_manual_viz():
    print("Loading Manual Labels...")
    csv_path = os.path.join(os.path.dirname(__file__), "manual_labels.csv")
    mappings = metrics.load_manual_labels(csv_path)
    
    print(mappings.keys())
    
    # Fix 11: Standardize labels in-memory ('left' -> 'left turn', etc)
    # This ensures they cluster correctly in plots without changing the source CSV.
    for proj in mappings:
        for sid in list(mappings[proj].keys()):
            lbl = mappings[proj][sid].lower()
            if "left" in lbl:
                mappings[proj][sid] = "left turn"
            elif "right" in lbl or "run" in lbl:
                mappings[proj][sid] = "right turn"
            elif "misc" in lbl or "no video" in lbl:
                # Remove from mapping so it defaults to "Other" which is filtered later
                del mappings[proj][sid]
            elif "walk" in lbl or "dash" in lbl:
                mappings[proj][sid] = "walk/dash"
            elif "rear" in lbl or "lean" in lbl:
                # Group 'rear', 'lean rear/lean', 'lean/rear' into 'rear'
                mappings[proj][sid] = "rear"
    
    # IFDN
    if "IFDN" in mappings:
        print("Running Manual Viz for IFDN...")
        sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
        df_usage = metrics.get_manual_group_usage(sessions_ifdn, mappings["IFDN"])
        plot_manual_usage(df_usage, "IFDN")
        plot_manual_transitions(sessions_ifdn, mappings["IFDN"], "IFDN")
        plot_top_manual_changes(sessions_ifdn, mappings["IFDN"], "IFDN")
        
    # AMPH
    if "AMPH" in mappings:
        print("Running Manual Viz for AMPH...")
        sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
        df_usage = metrics.get_manual_group_usage(sessions_amph, mappings["AMPH"])
        # Manual labels are already robust-ish, but for consistency:
        # User said "all plots". But manual labels are merged syllables.
        # I will keep manual behavior as is unless asked specifically to filter underlying components.
        plot_manual_usage(df_usage, "AMPH")
        plot_manual_transitions(sessions_amph, mappings["AMPH"], "AMPH")
        plot_top_manual_changes(sessions_amph, mappings["AMPH"], "AMPH")

def plot_quantitative_matrix(df, project_name, metric="Time_Pct"):
    """
    Plot 3x3 Heatmaps for Quantitative Analysis.
    Rows: Vigor (High, Medium, Low) - Visual Y-Axis.
    Cols: Turn (Left, Middle, Right) - Visual X-Axis.
    Values: Mean Metric per group.
    """
    # Order
    # Heatmap plots index 0 at top. So for Y-axis "Vigor" to go Low->High (bottom->top), 
    # we need row order to be High->Medium->Low.
    v_order = ["High", "Medium", "Low"]
    t_order = ["Left", "Middle", "Right"]
    
    # 1. Group by Condition (Genotype/Drug) -> Matrix
    # We want one heatmap per Condition.
    # Conditions:
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        # Combined
        df["Condition"] = df["Genotype"] + " " + df["Drug"]
        # Filter None
        df = df[df["Condition"].notna()]
        conditions = sorted(df["Condition"].astype(str).unique())
        # Sort logic: WT Saline, WT Amph, KI Saline, KI Amph
        # Custom sort if possible
    else:
        # Filter None
        df = df[df["Genotype"].notna()]
        conditions = sorted(df["Genotype"].astype(str).unique())
        df["Condition"] = df["Genotype"]
        
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))
    if len(conditions) == 1: axes = [axes]
    
    # Global Min/Max for colorbar
    vmin = df[metric].min()
    vmax = df[metric].max()
    
    for i, cond in enumerate(conditions):
        sub = df[df["Condition"] == cond]
        
        # Pivot to 3x3
        # index=Vigor, columns=Turn
        mat = sub.groupby(["Turn", "Vigor"])[metric].mean().unstack(level="Turn")
        
        # Reindex to force order
        mat = mat.reindex(index=v_order, columns=t_order)
        
        ax = axes[i]
        sns.heatmap(mat, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=".2f")
        ax.set_title(f"{cond} ({metric})")
        ax.set_xlabel("Turn (Left <-> Right)")
        ax.set_ylabel("Vigor (High ^)")
        
    plt.tight_layout()
    metric_name = "time" if "Time" in metric else "onsets"
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_quantitative_matrix_{metric_name}.png"))
    plt.close()

def plot_quantitative_delta_matrix(df, project_name, metric="Time_Pct"):
    """
    Plot 3x3 Heatmaps of (Amph - Saline) per Genotype.
    Rows: Vigor, Cols: Turn.
    """
    if "Drug" not in df.columns or df["Drug"].nunique() <= 1:
        print("Delta analysis requires Drug column.")
        return

    # Filter None
    df = df[df["Genotype"].notna()]
    genotypes = sorted(df["Genotype"].astype(str).unique())
    
    v_order = ["High", "Medium", "Low"]
    t_order = ["Left", "Middle", "Right"]
    
    fig, axes = plt.subplots(1, len(genotypes), figsize=(5 * len(genotypes), 5))
    if len(genotypes) == 1: axes = [axes]
    
    # Calculate global range for symmetric colormap
    # We need to compute all deltas first to find min/max
    deltas = {}
    
    all_vals = []
    
    for g in genotypes:
        sub = df[df["Genotype"] == g]
        # Pivot: Mean per cell per drug
        # We need drug split
        # Group by Drug, Turn, Vigor
        means = sub.groupby(["Drug", "Turn", "Vigor"])[metric].mean().unstack(level=["Turn", "Vigor"])
        
        # Check if we have Saline and Amph
        drugs = means.index
        # Use loose matching
        sal = [d for d in drugs if "sal" in d.lower()]
        amph = [d for d in drugs if "amph" in d.lower()]
        
        if not sal or not amph:
            print(f"Skipping {g} due to missing drug conditions.")
            continue
            
        sal_vec = means.loc[sal[0]]
        amph_vec = means.loc[amph[0]]
        
        delta = amph_vec - sal_vec
        # Reshape to Matrix (Rows=Vigor, Cols=Turn)
        # unstack produced multi-index columns (Turn, Vigor).
        # Series -> Unstack Vigor?
        # Let's pivot Series
        mat = delta.reset_index() # Turn, Vigor, 0
        mat.columns = ["Turn", "Vigor", "Delta"]
        mat = mat.pivot(index="Vigor", columns="Turn", values="Delta")
        mat = mat.reindex(index=v_order, columns=t_order)
        
        deltas[g] = mat
        all_vals.extend(mat.values.flatten())
        
    # Symmetric limit
    max_abs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 0.1
    vmin, vmax = -max_abs, max_abs
    
    for i, g in enumerate(genotypes):
        ax = axes[i]
        if g in deltas:
            sns.heatmap(deltas[g], ax=ax, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax, annot=True, fmt=".2f")
            ax.set_title(f"{g} Delta (Amph - Saline) - {metric}")
            ax.set_xlabel("Turn (Left <-> Right)")
            ax.set_ylabel("Vigor (High ^)")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')
            
    plt.tight_layout()
    metric_name = "time" if "Time" in metric else "onsets"
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_quantitative_delta_{metric_name}.png"))
    plt.close()

def plot_quantitative_genotype_delta(df, project_name, metric="Time_Pct"):
    """
    Plot 3x3 Heatmap of (KI - WT) for projects without drugs (like IFDN).
    """
    # Filter None
    df = df[df["Genotype"].notna()]
    
    v_order = ["High", "Medium", "Low"]
    t_order = ["Left", "Middle", "Right"]
    
    # Calculate group means
    means = df.groupby(["Genotype", "Turn", "Vigor"])[metric].mean().unstack(level=["Turn", "Vigor"])
    
    if "WT" not in means.index or "KI" not in means.index:
        print(f"Skipping Genotype Delta for {project_name}: missing WT or KI.")
        return
        
    delta = means.loc["KI"] - means.loc["WT"]
    
    # Reshape to Matrix
    mat_df = delta.reset_index()
    mat_df.columns = ["Turn", "Vigor", "Delta"]
    mat = mat_df.pivot(index="Vigor", columns="Turn", values="Delta")
    mat = mat.reindex(index=v_order, columns=t_order)
    
    # Symmetric limit
    max_abs = abs(mat.values).max() if not mat.dropna().empty else 0.1
    vmin, vmax = -max_abs, max_abs
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax, annot=True, fmt=".2f")
    plt.title(f"{project_name} Genotype Delta (KI - WT) - {metric}")
    plt.xlabel("Turn (Left <-> Right)")
    plt.ylabel("Vigor (High ^)")
    
    plt.tight_layout()
    metric_name = "time" if "Time" in metric else "onsets"
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_quantitative_genotype_delta_{metric_name}.png"))
    plt.close()

def plot_violin_stats(df, project_name, metric, title, log_scale=False):
    """
    Stratified Violin Plot with Stats and Outlier Removal.
    
    Args:
        df (pd.DataFrame): Data.
        project_name (str): Project context.
        metric (str): Column name to plot.
        title (str): Plot title.
        log_scale (bool): Whether to log-scale Y axis.
    """
    # 1. Remove Outliers
    df = remove_outliers_iqr(df, metric, factor=3.0)
    
    plt.figure(figsize=(10, 6))
    
    # Stratification
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        # AMPH style: Genotype X in hue=Drug
        df = df[df["Genotype"].notna()]
        ax = sns.violinplot(data=df, x="Genotype", y=metric, hue="Drug", split=True, inner="quartile", palette="Set2")
        plt.title(f"{title} (Stratified by Drug)")
        
        # Stats Pairs
        genos = sorted(df["Genotype"].unique())
        drugs = sorted(df["Drug"].unique())
        if len(genos) == 2 and len(drugs) == 2:
           # WT-Sal vs WT-Amph, KI-Sal vs KI-Amph
           pairs = [
               ((genos[0], drugs[0]), (genos[0], drugs[1])),
               ((genos[1], drugs[0]), (genos[1], drugs[1])),
               ((genos[0], drugs[0]), (genos[1], drugs[0])), # Baseline G1 vs G2
               ((genos[0], drugs[1]), (genos[1], drugs[1]))  # Drug G1 vs G2
           ]
           add_stats_stars(ax, df, "Genotype", metric, genos, pairs, hue="Drug")
           
    else:
        # IFDN style: x=Genotype
        df = df[df["Genotype"].notna()]
        ax = sns.violinplot(data=df, x="Genotype", y=metric, palette="Set2", inner="quartile")
        plt.title(f"{title} (Genotype)")
        
        genos = sorted(df["Genotype"].unique())
        if len(genos) == 2:
            pairs = [(genos[0], genos[1])]
            add_stats_stars(ax, df, "Genotype", metric, genos, pairs)
        
    if log_scale: plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_{metric}.png"))
    plt.close()

def plot_scatter_stats(df, project_name, x_metric, y_metric, title):
    """
    Stratified Scatter Plot with Regression, Correlation Stats, and Outlier Removal.
    
    Args:
        df (pd.DataFrame): Data.
        project_name (str): Project context.
        x_metric (str): X-axis column.
        y_metric (str): Y-axis column.
        title (str): Chart title.
    """
    # Clean data for stats
    clean = df.dropna(subset=[x_metric, y_metric])
    
    # 1. Remove Outliers (on both axes independently or joint?)
    # Independent generic filtering usually sufficient to remove extreme artifacts
    clean = remove_outliers_iqr(clean, x_metric, factor=3.0)
    clean = remove_outliers_iqr(clean, y_metric, factor=3.0)
    
    g = sns.FacetGrid(clean, col="Genotype", hue="Drug" if "Drug" in clean.columns else None, height=5, legend_out=True)
    
    # Custom mapping function to add regression and stats
    def map_reg_stats(x, y, **kwargs):
        # 1. Scatter
        sns.scatterplot(x=x, y=y, **kwargs, alpha=0.5)
        
        # 2. Regression
        # Filter for current group kwargs (color)
        color = kwargs.get("color", "k")
        label = kwargs.get("label", None)
        
        # Calculate Stats
        if len(x) > 2:
            res = stats.linregress(x, y)
            r, p = res.rvalue, res.pvalue
            
            # Draw line
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = res.slope * x_range + res.intercept
            plt.plot(x_range, y_pred, color=color, linestyle="--", linewidth=2)
            
            # Annotation text
            # Move text based on group? Hard to avoid overlap dynamically without heavy lifting.
            # Just put it in the legend or title?
            # Let's annotate on plot.
            # To avoid overlap, we use label in legend.
            
            stat_str = f"r={r:.2f}, p={p:.1e}"
            # Attach to existing label if possible?
            # Or print on screen?
            
            # Simple approach: Text annotation at top/bottom corner depending on slope?
            # Or just append to label for Legend.
            # But Legend is drawn by FactGrid logic. 
            # Let's add text directly.
            
            ax = plt.gca()
            # Random vertical jitter for text to avoid overlap if multiple groups?
            # If multiple hues (AMPH), they share the axis.
            # If hue is present, we have multiple colors.
            # Use transform=transAxes
            
            # Get current artist count to offset?
            # HACK: Use label to identify index?
            # Or just y-offset based on correlation?
            
            # Let's just create a custom label and put it in a corner text box
            # But FacetGrid iterates.
            pass

    # Better approach: Iterate manually after mapping
    g.map(sns.scatterplot, x_metric, y_metric, alpha=0.5)
    
    if "Drug" in clean.columns:
        g.add_legend()
        
    # Manually add regression lines and stats
    for ax_idx, ax in enumerate(g.axes.flatten()):
        # Get data for this facet (Genotype)
        # FacetGrid stores data? No, we filter.
        # Assuming col_order matches
        geno = g.col_names[ax_idx]
        sub = clean[clean["Genotype"] == geno]
        
        if "Drug" in sub.columns:
            groups = sub.groupby("Drug")
        else:
            groups = [("All", sub)]
            
        text_y_pos = 0.95
        
        for name, group in groups:
            x = group[x_metric]
            y = group[y_metric]
            if len(x) > 2:
                res = stats.linregress(x, y)
                r, p = res.rvalue, res.pvalue
                
                # Determine color matches scatter
                # Hard without accessing seaborn's color cycle state.
                # Just use black/grey/dashed for regression to be safe, or get color from palette.
                
                # Plot Line
                x_rng = np.linspace(x.min(), x.max(), 100)
                y_pred = res.slope * x_rng + res.intercept
                
                # Choose color
                c = "red" if (name and "Amph" in str(name)) else "blue"
                if "Drug" not in sub.columns: c = "black"
                
                ax.plot(x_rng, y_pred, color=c, linestyle="--", lw=1.5)
                
                # Add Text
                sig = "*" if p < 0.05 else "ns"
                txt = f"{name}: r={r:.2f} ({sig})"
                ax.text(0.05, text_y_pos, txt, transform=ax.transAxes, color=c, fontsize=10, fontweight="bold")
                text_y_pos -= 0.08
                
    g.set_axis_labels(x_metric, y_metric)
    g.fig.suptitle(title, y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_{x_metric}_vs_{y_metric}.png"))
    plt.close()

def plot_radar_fingerprint(df, project_name):
    """
    Radar Chart / Movement Signature Heatmap.
    
    Args:
        df (pd.DataFrame): Normalized metric dataframe.
        project_name (str): Project context.
    """
    """
    Radar Chart for Top 5 Syllables.
    Dims: [Mean Vel, Mean Accel (Expl), Straightness, Duration]
    Normalized to 0-1 across the dataset.
    """
    # 1. Select Metrics
    # Fix 17: Use Straightness directly
    metrics = ["mean_velocity", "mean_explosiveness", "mean_straightness", "mean_duration", "duration_cv"]
    
    # Normalize
    norm_df = df.copy()
    
    # Force Numeric
    for m in metrics:
        norm_df[m] = pd.to_numeric(norm_df[m], errors='coerce')
        
    # Fix 6: Use Z-score normalization (StandardScaler)
    for m in metrics:
        mean_val = norm_df[m].mean()
        std_val = norm_df[m].std()
        if pd.isna(std_val) or std_val == 0:
             norm_df[m] = 0
        else:
             norm_df[m] = (norm_df[m] - mean_val) / std_val
        
    # Ready for plotting
    
    final_metrics = ["mean_velocity", "mean_explosiveness", "mean_straightness", "mean_duration", "duration_cv"]
    labels = ["Vel", "Accel", "Straight", "Dur", "CV"]
    
    # Agg by Genotype (IFDN) or Geno+Drug (AMPH)
    if "Drug" in df.columns and df["Drug"].nunique() > 1:
        group_cols = ["Genotype", "Drug"]
    else:
        group_cols = ["Genotype"]
        
    # Filter None keys
    norm_df = norm_df.dropna(subset=group_cols)
    
    # 2-step agg: First mean per session (already done?), then mean per group.
    # Current df has one row per session? No, one row per syllable per session?
    # get_fingerprint_data returns one row per syllable per session.
    # We want mean across all sessions/syllables for that group??
    # "Movement Signature" implies aggregating over ALL top syllables?
    # Or keeping syllables distinct?
    # "Dims: [Mean Vel...]" computed ACROSS top behavioral states.
    # Yes, collapse syllables -> One signature per Animal Group.
    
    summary = norm_df.groupby(group_cols)[final_metrics].mean().reset_index()
    
    # Pivot for Heatmap: Index=Metrics, Cols=Groups
    # Transform to long
    melted = summary.melt(id_vars=group_cols, var_name="Metric", value_name="Score")
    
    # Create compound label for cols if needed
    if "Drug" in summary.columns:
        melted["Group"] = melted["Genotype"].astype(str) + " " + melted["Drug"].astype(str)
    else:
        melted["Group"] = melted["Genotype"].astype(str)
        
    pivot = melted.pivot(index="Metric", columns="Group", values="Score")
    
    # Sort cols
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    # DEBUG
    print("Pivot dtypes:")
    print(pivot.dtypes)
    # Force float
    pivot = pivot.astype(float)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f"{project_name} Movement Signature (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{project_name.lower()}_movement_signature.png"))
    plt.close()

def run_advanced_kinematics_viz():
    print("\nCalculated Advanced Kinematics...")
    
    # Load All Data again to compute advanced metrics per session
    # IFDN
    s_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
    df_ifdn = metrics.get_fingerprint_data(s_ifdn, top_n=999, min_instances=MIN_INSTANCES) 
    
    # AMPH
    s_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
    df_amph = metrics.get_fingerprint_data(s_amph, top_n=999, min_instances=MIN_INSTANCES)
    
    # Metrics to Loop
    feats = [
        ("duration_cv", "Duration CV (Roboticness)", False),
        ("mean_explosiveness", "Explosiveness (P95 Accel)", True),
        ("mean_straightness", "Straightness (Disp/Path)", False),
        ("mean_velocity", "Mean Velocity", False),
        ("mean_duration", "Mean Duration", False)
    ]
    
    for feat, title, log in feats:
        plot_violin_stats(df_ifdn, "IFDN", feat, title, log)
        plot_violin_stats(df_amph, "AMPH", feat, title, log)
        
    # Scatters
    plot_scatter_stats(df_ifdn, "IFDN", "mean_velocity", "mean_duration", "Ballistic Check")
    plot_scatter_stats(df_amph, "AMPH", "mean_velocity", "mean_duration", "Ballistic Check")
    
    plot_scatter_stats(df_ifdn, "IFDN", "mean_velocity", "mean_straightness", "Vigor vs Straightness")
    plot_scatter_stats(df_amph, "AMPH", "mean_velocity", "mean_straightness", "Vigor vs Straightness")
    
    # Signatures
    plot_radar_fingerprint(df_ifdn, "IFDN")
    plot_radar_fingerprint(df_amph, "AMPH")

    print("Advanced Kinematics plots saved.")
    
def run_quantitative_viz():
    print("Loading Quantitative Map...")
    # Map file is at syllable_generation/quantitative_map.csv
    # Script runs from syllable_generation/
    # run_analysis.py is in subfolder.
    # Base dir is provided
    map_path = os.path.join(root_dir, "syllable_generation", "quantitative_map.csv")
    mapping = metrics.load_quantitative_map(map_path)
    
    if not mapping:
        print("Quantitative map not found. Skipping.")
        return

    print(f"Loaded {len(mapping)} classified syllables.")
    
    # Run IFDN
    print("Running Quantitative Viz for IFDN...")
    sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
    df_ifdn = metrics.get_quantitative_group_usage(sessions_ifdn, mapping, min_instances=MIN_INSTANCES)
    plot_quantitative_matrix(df_ifdn, "IFDN", "Time_Pct")
    plot_quantitative_matrix(df_ifdn, "IFDN", "Onset_Count")
    # Genotype Delta
    plot_quantitative_genotype_delta(df_ifdn, "IFDN", "Time_Pct")
    plot_quantitative_genotype_delta(df_ifdn, "IFDN", "Onset_Count")
    
    # Run AMPH
    print("Running Quantitative Viz for AMPH...")
    sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
    df_amph = metrics.get_quantitative_group_usage(sessions_amph, mapping, min_instances=MIN_INSTANCES)
    plot_quantitative_matrix(df_amph, "AMPH", "Time_Pct")
    plot_quantitative_matrix(df_amph, "AMPH", "Onset_Count")
    
    # Delta Plots
    plot_quantitative_delta_matrix(df_amph, "AMPH", "Time_Pct")
    plot_quantitative_delta_matrix(df_amph, "AMPH", "Onset_Count")

if __name__ == "__main__":
    # Standard Runs
    run_ifdn()
    run_amph()
    
    # Manual Run
    run_manual_viz()

    # Quantitative Run
    run_quantitative_viz()
    
    # Advanced Kinematics
    run_advanced_kinematics_viz()
