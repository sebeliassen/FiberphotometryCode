import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from scipy import stats
from statannotations.Annotator import Annotator

# Suppress warnings
warnings.filterwarnings('ignore')

# Add root to path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader, metrics

# Setup plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_METRICS_DIR = os.path.join(OUTPUT_DIR, "global_metrics")

def get_output_dir(project_name, stat_type):
    proj = project_name.lower().split("_")[0]
    d = os.path.join(GLOBAL_METRICS_DIR, proj, stat_type)
    os.makedirs(d, exist_ok=True)
    return d

PLOT_METRICS = ["straightness", "explosiveness", "entropy_transition", "duration", "entropy_duration"]
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
        geno_order = sorted(plot_df["Genotype"].dropna().unique().tolist())
        sns.barplot(data=plot_df, x="Syllable", y="Frequency", hue="Genotype", hue_order=geno_order)
        
    plt.title(f"Syllable Usage (Fair Rank Sort) - {project_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(get_output_dir(project_name, "usage"), f"{project_name.lower()}_usage.png"))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_transition_deltas.png"))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_transitions.png"))
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
    df_ifdn.to_csv(os.path.join(get_output_dir("ifdn", "stats"), "ifdn_kinematics_stats.csv"), index=False)

    # 1. Summary Boxplots with Stats
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    order = ["WT", "KI"]
    pairs = [("WT", "KI")]
    
    METRIC_TITLES = {
        "straightness": "Straightness\n(Path efficiency: Disp / Dist)",
        "explosiveness": "Explosiveness\n(Motor Vigor: P95 Accel)",
        "entropy_transition": "Transition Entropy\n(Behavioral Unpredictability)",
        "duration": "Duration\n(Avg Bout Sustain Time, sec)",
        "entropy_duration": "Duration Entropy\n(Bout Length Variability)"
    }
    
    for i, m in enumerate(PLOT_METRICS):
        ax = axes[i]
        sns.boxplot(data=df_ifdn, x="Genotype", y=m, ax=ax, width=0.5, palette="Set2", showfliers=False, order=order)
        sns.stripplot(data=df_ifdn, x="Genotype", y=m, ax=ax, color="black", alpha=0.7, order=order)
        add_stats_stars(ax, df_ifdn, "Genotype", m, order, pairs)
        ax.set_title(METRIC_TITLES.get(m, m.replace("_", " ").title()), fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(get_output_dir("ifdn", "stats"), "ifdn_summary_panel.png"))
    plt.close()

    # 2. Usage Barplot
    # Fair Top list for IFDN
    ifdn_fair_sylls = [0, 1, 2, 3, 57, 36, 15, 25, 11, 43, 27, 12, 10, 34, 24, 56, 32, 64, 45, 16]
    df_usage = metrics.get_group_usage(sessions_ifdn, min_instances=MIN_INSTANCES)
    plot_usage(df_usage, "IFDN", ordered_sylls=ifdn_fair_sylls)
    
    # 3. Transitions
    plot_transition_matrices(sessions_ifdn, "IFDN", ordered_sylls=ifdn_fair_sylls)

def run_karen():
    print("Loading KAREN Dataset...")
    sessions_karen = loader.load_dataset("karen", base_dir=root_dir)

    rows_karen = []
    for s in tqdm(sessions_karen, desc="Processing KAREN"):
        rows_karen.append(metrics.analyze_session(s))
        
    df_karen = pd.DataFrame(rows_karen)
    # Filter
    df_karen = df_karen[~df_karen["Genotype"].astype(str).isin(["None", "UNK", "nan"])]
    df_karen.to_csv(os.path.join(get_output_dir("karen", "stats"), "karen_kinematics_stats.csv"), index=False)

    # 1. Summary Boxplots with Stats
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    order = ["Monomer", "PFF"]
    pairs = [("Monomer", "PFF")]
    
    METRIC_TITLES = {
        "straightness": "Straightness\n(Path efficiency: Disp / Dist)",
        "explosiveness": "Explosiveness\n(Motor Vigor: P95 Accel)",
        "entropy_transition": "Transition Entropy\n(Behavioral Unpredictability)",
        "duration": "Duration\n(Avg Bout Sustain Time, sec)",
        "entropy_duration": "Duration Entropy\n(Bout Length Variability)"
    }
    
    for i, m in enumerate(PLOT_METRICS):
        ax = axes[i]
        sns.boxplot(data=df_karen, x="Genotype", y=m, ax=ax, width=0.5, palette="Set2", showfliers=False, order=order)
        sns.stripplot(data=df_karen, x="Genotype", y=m, ax=ax, color="black", alpha=0.7, order=order)
        add_stats_stars(ax, df_karen, "Genotype", m, order, pairs)
        ax.set_title(METRIC_TITLES.get(m, m.replace("_", " ").title()), fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(get_output_dir("karen", "stats"), "karen_summary_panel.png"))
    plt.close()

    # 2. Usage Barplot
    # We dynamically select top syllables by frequency rank fallback
    df_usage = metrics.get_group_usage(sessions_karen, min_instances=MIN_INSTANCES)
    plot_usage(df_usage, "KAREN", ordered_sylls=None)
    
    # 3. Transitions
    plot_transition_matrices(sessions_karen, "KAREN", ordered_sylls=None)

def run_amph():
    print("Loading AMPH Dataset...")
    
    # Run phased global metrics
    for phase in ["pre", "post"]:
        print(f"  Processing AMPH phase: {phase}")
        sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)

        rows_amph = []
        for s in tqdm(sessions_amph, desc=f"Extracting {phase} Stats"):
            rows_amph.append(metrics.analyze_session(s))

        df_amph = pd.DataFrame(rows_amph)
        # Filter
        df_amph = df_amph[~df_amph["Genotype"].astype(str).isin(["None", "UNK", "nan"])]
        df_amph.to_csv(os.path.join(get_output_dir("amph", "stats"), f"amph_{phase}_kinematics_stats.csv"), index=False)

        # 1. Summary Boxplots
        df_amph["Drug"] = df_amph["Drug"].fillna("N/A")
        df_amph["Group_Tag"] = df_amph["Genotype"] + " " + df_amph["Drug"]
        
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
        plt.savefig(os.path.join(get_output_dir("amph", "stats"), f"amph_{phase}_summary_panel.png"))
        plt.close()

        # 2. Raw Usage and Transitions
        amph_fair_sylls = [0, 1, 2, 3, 58, 4, 35, 65, 62, 29, 70, 86, 63, 80, 95, 83, 72, 89, 43, 96]
        df_usage = metrics.get_group_usage(sessions_amph, min_instances=MIN_INSTANCES)
        plot_usage(df_usage, f"AMPH_{phase}", ordered_sylls=amph_fair_sylls)
        
        plot_transition_matrices(sessions_amph, f"AMPH_{phase}", ordered_sylls=amph_fair_sylls)
        if phase == "post":
            plot_transition_deltas(sessions_amph, f"AMPH_{phase}", ordered_sylls=amph_fair_sylls)
            
        # 3. Alphabet Labels Generation
        from analysis.ulrik_kinematics.lib import motifs
        alphabet_map = motifs.get_label_mapping(sessions_amph, project="AMPH", style="alphabet")
        
        # We can hijack manual_usage logic to plot alphabet
        # by passing the alphabet mapping.
        df_alpha_usage = metrics.get_manual_group_usage(sessions_amph, alphabet_map)
        plot_manual_usage(df_alpha_usage, f"AMPH_{phase}_ALPHABET")
        
        m_alpha, session_matrices = metrics.get_manual_group_transition_matrix(sessions_amph, alphabet_map, return_distributions=True)
        plot_manual_transitions(sessions_amph, alphabet_map, f"AMPH_{phase}_ALPHABET")

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
    plt.savefig(os.path.join(get_output_dir(project_name, "usage"), f"{project_name.lower()}_manual_usage_time.png"))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "usage"), f"{project_name.lower()}_manual_usage_onsets.png"))
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

def plot_bar_changes(top_df, title, filename, project_name):
    """Auxiliary plotter for top changes with p-value stars."""
    
    # Check if empty
    if top_df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No Significant Changes (p < 0.05)", ha='center', va='center')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(get_output_dir(project_name, "usage"), filename))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "usage"), filename))
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
            plot_bar_changes(top, "IFDN: KI - WT (Significant Changes p<0.05)", "ifdn_manual_top_changes.png", project_name)
            
    elif project_name == "AMPH":
        k_wt_sal, k_wt_amph = ("WT", "Saline"), ("WT", "Amphetamine")
        k_ki_sal, k_ki_amph = ("KI", "Saline"), ("KI", "Amphetamine")
        
        # WT Delta
        if k_wt_sal in aligned_matrices and k_wt_amph in aligned_matrices:
            delta_wt = aligned_matrices[k_wt_amph] - aligned_matrices[k_wt_sal]
            top_wt = get_significant_changes(delta_wt, aligned_dists[k_wt_sal], aligned_dists[k_wt_amph])
            plot_bar_changes(top_wt, "WT: Amph - Saline (Sig p<0.05)", "amph_manual_top_changes_wt.png", project_name)
        else: delta_wt = None

        # KI Delta
        if k_ki_sal in aligned_matrices and k_ki_amph in aligned_matrices:
            delta_ki = aligned_matrices[k_ki_amph] - aligned_matrices[k_ki_sal]
            top_ki = get_significant_changes(delta_ki, aligned_dists[k_ki_sal], aligned_dists[k_ki_amph])
            plot_bar_changes(top_ki, "KI: Amph - Saline (Sig p<0.05)", "amph_manual_top_changes_ki.png", project_name)
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
            plot_bar_changes(top_int, "Interaction (KI Delta - WT Delta) (Sig p<0.05)", "amph_manual_top_changes_interaction.png", project_name)

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
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_manual_transitions.png"))
    plt.close()

def run_manual_viz(dataset="all"):
    print("Loading Manual Labels...")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "manual_labels.csv")
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
    if "IFDN" in mappings and dataset in ["ifdn", "all"]:
        print("Running Manual Viz for IFDN...")
        sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
        df_usage = metrics.get_manual_group_usage(sessions_ifdn, mappings["IFDN"])
        plot_manual_usage(df_usage, "IFDN")
        plot_manual_transitions(sessions_ifdn, mappings["IFDN"], "IFDN")
        plot_top_manual_changes(sessions_ifdn, mappings["IFDN"], "IFDN")
        
    # AMPH
    if "AMPH" in mappings and dataset in ["amph", "all"]:
        print("Running Manual Viz for AMPH...")
        for phase in ["pre", "post"]:
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)
            df_usage = metrics.get_manual_group_usage(sessions_amph, mappings["AMPH"])
            plot_manual_usage(df_usage, f"AMPH_{phase}")
            plot_manual_transitions(sessions_amph, mappings["AMPH"], f"AMPH_{phase}")
            if phase == "post":
                plot_top_manual_changes(sessions_amph, mappings["AMPH"], f"AMPH_{phase}")

