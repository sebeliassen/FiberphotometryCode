import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from analysis.ulrik_kinematics.lib.legacy_viz import (
    remove_outliers_iqr, add_stats_stars,
    get_output_dir, MIN_INSTANCES,
    root_dir
)
from analysis.ulrik_kinematics.lib import loader, metrics


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
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_quantitative_matrix_{metric_name}.png"))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_quantitative_delta_{metric_name}.png"))
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
    
    genos = sorted(means.index.unique().tolist())
    if len(genos) < 2:
        print(f"Skipping Genotype Delta for {project_name}: need at least 2 genotypes, found {genos}.")
        return
    
    # Use first two genotypes sorted alphabetically: delta = second - first
    g_base, g_test = genos[0], genos[1]
    delta = means.loc[g_test] - means.loc[g_base]
    
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
    plt.title(f"{project_name} Genotype Delta ({g_test} - {g_base}) - {metric}")
    plt.xlabel("Turn (Left <-> Right)")
    plt.ylabel("Vigor (High ^)")
    
    plt.tight_layout()
    metric_name = "time" if "Time" in metric else "onsets"
    plt.savefig(os.path.join(get_output_dir(project_name, "transitions"), f"{project_name.lower()}_quantitative_genotype_delta_{metric_name}.png"))
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
    # 1. Remove Outliers (standard Tukey 1.5x IQR fence)
    df = remove_outliers_iqr(df, metric, factor=1.5)
    
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
    plt.savefig(os.path.join(get_output_dir(project_name, "kinematics"), f"{project_name.lower()}_{metric}.png"))
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
    plt.savefig(os.path.join(get_output_dir(project_name, "kinematics"), f"{project_name.lower()}_{x_metric}_vs_{y_metric}.png"))
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
    melted = summary.melt(id_vars=group_cols, var_name="Metric", value_name="Score")
    
    if "Drug" in summary.columns:
        melted["Group"] = melted["Genotype"].astype(str) + " " + melted["Drug"].astype(str)
        pivot = melted.pivot(index="Metric", columns="Group", values="Score").astype(float)
        
        # Calculate Delta: Amph - Saline
        delta_pivot = pd.DataFrame(index=pivot.index)
        for geno in summary["Genotype"].unique():
            sal_col = f"{geno} Saline"
            amph_col = f"{geno} Amphetamine"
            if sal_col in pivot.columns and amph_col in pivot.columns:
                delta_pivot[f"{geno} Drug Effect"] = pivot[amph_col] - pivot[sal_col]
                
        # Sort cols
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", ax=axes[0])
        axes[0].set_title(f"{project_name} Raw Signatures (Z-Score)")
        
        if not delta_pivot.empty:
            max_abs = abs(delta_pivot.values).max() if not delta_pivot.empty else 1.0
            sns.heatmap(delta_pivot, cmap="coolwarm", center=0, vmin=-max_abs, vmax=max_abs, annot=True, fmt=".2f", ax=axes[1])
            axes[1].set_title(f"{project_name} Drug Effect (Amph - Saline)")
        else:
            axes[1].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(get_output_dir(project_name, "kinematics"), f"{project_name.lower()}_movement_signature.png"))
        plt.close()
    else:
        melted["Group"] = melted["Genotype"].astype(str)
        pivot = melted.pivot(index="Metric", columns="Group", values="Score").astype(float)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
        plt.title(f"{project_name} Movement Signature (Z-Score)")
        plt.tight_layout()
        plt.savefig(os.path.join(get_output_dir(project_name, "kinematics"), f"{project_name.lower()}_movement_signature.png"))
        plt.close()

def run_advanced_kinematics_viz(dataset="all"):
    print("\nCalculated Advanced Kinematics...")
    
    # Metrics to Loop
    feats = [
        ("duration_cv", "Duration CV (Roboticness)", False),
        ("mean_explosiveness", "Explosiveness (P95 Accel)", False),
        ("mean_straightness", "Straightness (Disp/Path)", False),
        ("mean_velocity", "Mean Velocity", False),
        ("mean_duration", "Mean Duration", False)
    ]
    
    # Load All Data again to compute advanced metrics per session
    # IFDN
    if dataset in ["ifdn", "all"]:
        s_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
        df_ifdn = metrics.get_fingerprint_data(s_ifdn, top_n=999, min_instances=MIN_INSTANCES) 
        
        # Run IFDN
        for feat, title, log in feats:
            plot_violin_stats(df_ifdn, "IFDN", feat, title, log)
        plot_scatter_stats(df_ifdn, "IFDN", "mean_velocity", "mean_duration", "Ballistic Check")
        plot_scatter_stats(df_ifdn, "IFDN", "mean_velocity", "mean_straightness", "Vigor vs Straightness")
        plot_radar_fingerprint(df_ifdn, "IFDN")
    
    if dataset in ["karen", "all"]:
        s_karen = loader.load_dataset("karen", base_dir=root_dir)
        df_karen = metrics.get_fingerprint_data(s_karen, top_n=999, min_instances=MIN_INSTANCES)
        
        # Run KAREN
        for feat, title, log in feats:
            plot_violin_stats(df_karen, "KAREN", feat, title, log)
        plot_scatter_stats(df_karen, "KAREN", "mean_velocity", "mean_duration", "Ballistic Check")
        plot_scatter_stats(df_karen, "KAREN", "mean_velocity", "mean_straightness", "Vigor vs Straightness")
        plot_radar_fingerprint(df_karen, "KAREN")

    if dataset in ["amph", "all"]:
        # AMPH
        for phase in ["pre", "post"]:
            print(f"  Calculating Advanced Kinematics for AMPH {phase}...")
            s_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)
            df_amph = metrics.get_fingerprint_data(s_amph, top_n=999, min_instances=MIN_INSTANCES)
            
            for feat, title, log in feats:
                plot_violin_stats(df_amph, f"AMPH_{phase}", feat, title, log)
                
            plot_scatter_stats(df_amph, f"AMPH_{phase}", "mean_velocity", "mean_duration", "Ballistic Check")
            plot_scatter_stats(df_amph, f"AMPH_{phase}", "mean_velocity", "mean_straightness", "Vigor vs Straightness")
            plot_radar_fingerprint(df_amph, f"AMPH_{phase}")

    print("Advanced Kinematics plots saved.")
    
def run_quantitative_viz(dataset="all"):
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
    if dataset in ["ifdn", "all"]:
        print("Running Quantitative Viz for IFDN...")
    sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
    df_ifdn = metrics.get_quantitative_group_usage(sessions_ifdn, mapping, min_instances=MIN_INSTANCES)
    plot_quantitative_matrix(df_ifdn, "IFDN", "Time_Pct")
    plot_quantitative_matrix(df_ifdn, "IFDN", "Onset_Count")
    # Genotype Delta
    plot_quantitative_genotype_delta(df_ifdn, "IFDN", "Time_Pct")
    plot_quantitative_genotype_delta(df_ifdn, "IFDN", "Onset_Count")
    print("Quantitative Viz completed.")
    
    # Run AMPH
    if dataset in ["amph", "all"]:
        print("Running Quantitative Viz for AMPH...")
    for phase in ["pre", "post"]:
        sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)
        df_amph = metrics.get_quantitative_group_usage(sessions_amph, mapping, min_instances=MIN_INSTANCES)
        plot_quantitative_matrix(df_amph, f"AMPH_{phase}", "Time_Pct")
        plot_quantitative_matrix(df_amph, f"AMPH_{phase}", "Onset_Count")
        
        # We can calculate genotype deltas for both pre and post
        plot_quantitative_genotype_delta(df_amph, f"AMPH_{phase}", "Time_Pct")
        plot_quantitative_genotype_delta(df_amph, f"AMPH_{phase}", "Onset_Count")
        
        if phase == "post":
            # Drug Deltas (Amph - Saline) only exist for post
            plot_quantitative_delta_matrix(df_amph, f"AMPH_{phase}", "Time_Pct")
            plot_quantitative_delta_matrix(df_amph, f"AMPH_{phase}", "Onset_Count")
