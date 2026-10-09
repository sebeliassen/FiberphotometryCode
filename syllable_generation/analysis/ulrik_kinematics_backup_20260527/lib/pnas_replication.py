import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from analysis.ulrik_kinematics.lib import metrics, networks

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/pnas_replication")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")


def _get_group_tag(s):
    geno = s.metadata.get("genotype")
    drug = s.metadata.get("drug", "N/A")
    if geno in ["None", "UNK", None]: return None

    if drug.lower().startswith("amph"): drug_str = "Amphetamine"
    elif drug.lower().startswith("sal"): drug_str = "Saline"
    else: drug_str = drug
    return f"{geno} {drug_str}".strip()


def plot_fig3c_syllable_usage(sessions, project_name):
    """
    Fig 3C: Initiations per minute boxplots for the 20 most frequent syllables.
    """
    print("Plotting Fig 3C: Syllable Usage Boxplots")
    all_bouts = []
    for s in sessions:
        all_bouts.append(metrics.get_syllable_instances(s.all_data_df))
    if not all_bouts: return

    global_bouts = pd.concat(all_bouts).groupby(level=0).sum()
    top_sylls = global_bouts.sort_values(ascending=False).head(20).index.tolist()

    # Map top 20 to letters a-t (a = most frequent, t = 20th frequent)
    letters = 'abcdefghijklmnopqrst'
    letter_map = {syll: letters[i] for i, syll in enumerate(top_sylls)}

    rows = []
    for s in sessions:
        counts = metrics.get_syllable_instances(s.all_data_df)
        session_mins = len(s.all_data_df) / s.metadata.get("fs", 80.0) / 60.0
        group_tag = _get_group_tag(s)
        if not group_tag: continue

        for syll in top_sylls:
            inits = counts.get(syll, 0)
            rows.append({
                "Animal": s.metadata.get("animal_id"),
                "Group": group_tag,
                "Syllable": letter_map[syll],
                "Syllable_ID": syll,
                "Initiations_Per_Min": inits / session_mins if session_mins > 0 else 0
            })

    df = pd.DataFrame(rows)
    if df.empty: return

    ordered_letters = [letter_map[s] for s in top_sylls][::-1]

    hue_order = ["WT Saline", "WT Amphetamine", "KI Saline", "KI Amphetamine"]
    existing_hues = [h for h in hue_order if h in df["Group"].unique()]

    for group in existing_hues:
        sub_df = df[df["Group"] == group]
        plt.figure(figsize=(14, 6))

        ax = sns.boxplot(data=sub_df, x="Syllable", y="Initiations_Per_Min",
                         order=ordered_letters, showfliers=True,
                         flierprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 5, 'alpha': 0.6},
                         width=0.7, color='lightgray')

        plt.title(f"Syllable Usage (Initiations per min) - {group}")
        plt.ylabel("Average initiations per min.")
        plt.xlabel("Syllables")
        sns.despine(trim=True)
        plt.tight_layout()

        group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
        os.makedirs(group_dir, exist_ok=True)

        grp_filename = group.lower().replace(" ", "_")
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig3c_{grp_filename}.png"), dpi=300)
        plt.close()


def plot_fig3h_kinematic_similarity(sessions, project_name, global_kinematics=None):
    """
    Fig 3H: Hierarchically clustered heatmap of typical kinematic trajectories.
    Follows Keypoint MoSeq syllable_similarity paradigm.
    Also saves top similar pairs for use by the paradox twin plots.
    """
    print("Plotting Fig 3H: Kinematic Similarity Dendrogram")
    all_bouts = []
    for s in sessions:
        all_bouts.append(metrics.get_syllable_instances(s.all_data_df))
    if not all_bouts: return

    if global_kinematics is not None:
        dist_matrix, valid_sylls, Z, top_sylls = global_kinematics
    else:
        global_bouts = pd.concat(all_bouts).groupby(level=0).sum()
        top_sylls = global_bouts.sort_values(ascending=False).head(20).index.tolist()
        dist_matrix, valid_sylls, Z = get_kinematic_similarity(sessions, project_name, top_sylls)

    if dist_matrix is None:
        print("  [Warning] Could not compute kinematic similarity matrix.")
        return

    # Build alphabet labels (consistent with motifs module)
    letters = 'abcdefghijklmnopqrst'
    letter_map = {syll: letters[i] for i, syll in enumerate(top_sylls)}
    valid_labels = [letter_map.get(s, str(s)) for s in valid_sylls]
    syll_to_letter = {s: letter_map.get(s, str(s)) for s in valid_sylls}

    spearman_corr = 1.0 - dist_matrix


    # We will use col_linkage=Z to restore the standard symmetric matrix look.
    clustergrid = sns.clustermap(spearman_corr, row_linkage=Z, col_linkage=Z,
                                 xticklabels=valid_labels, yticklabels=valid_labels,
                                 cmap="viridis", figsize=(8, 9), vmin=0, vmax=1)

    # Move Y labels to the left side (same side as the row dendrogram)
    clustergrid.ax_heatmap.yaxis.tick_left()
    clustergrid.ax_heatmap.yaxis.set_label_position("left")

    # By default, sns.clustermap puts X labels on bottom, X dendrogram on top.
    # The user requested column dendrogram and letters AT THE BOTTOM

    # Increase the gap between the heatmap and the dendrograms
    dx, dy = 0.07, 0.07

    # 1. Heatmap
    h_pos = clustergrid.ax_heatmap.get_position()
    clustergrid.ax_heatmap.set_position([h_pos.x0 + dx, h_pos.y0 + dy, h_pos.width, h_pos.height])

    # 2. Row Dendrogram (Left) -> Shift up to match heatmap rows
    r_pos = clustergrid.ax_row_dendrogram.get_position()
    clustergrid.ax_row_dendrogram.set_position([r_pos.x0, r_pos.y0 + dy, r_pos.width, r_pos.height])

    # 3. Column Dendrogram (Top -> Bottom)
    # Move it directly beneath the heatmap, matching heatmap width/x-position
    c_pos = clustergrid.ax_col_dendrogram.get_position()
    clustergrid.ax_col_dendrogram.set_position([h_pos.x0 + dx, h_pos.y0 + dy - c_pos.height - 0.05, h_pos.width, c_pos.height])
    # Invert the Y-axis of the column dendrogram so the "branches" reach UP towards the heatmap
    clustergrid.ax_col_dendrogram.invert_yaxis()

    # Prevent arm clipping for left dendrogram
    r_xlim = clustergrid.ax_row_dendrogram.get_xlim()
    clustergrid.ax_row_dendrogram.set_xlim(r_xlim[0] + (r_xlim[0] - r_xlim[1]) * 0.05, r_xlim[1])

    # Extend row arms
    row_extend = r_xlim[0] * -0.15
    for coll in clustergrid.ax_row_dendrogram.collections:
        coll.set_clip_on(False)
        paths = coll.get_segments()
        for path in paths:
            for pt in path:
                if pt[0] == 0:
                    pt[0] = row_extend
        coll.set_segments(paths)

    # Col extend (arms reaching UP towards Y=0)
    # the root is at y=max, arms terminate at y=0. Because we inverted y-axis, the limits are flipped.
    # We want them to extend past y=0 into negative space (towards heatmap).
    for coll in clustergrid.ax_col_dendrogram.collections:
        coll.set_clip_on(False)
        paths = coll.get_segments()
        for path in paths:
            for pt in path:
                if pt[1] == 0:
                    pt[1] = -c_pos.height * 0.15
        coll.set_segments(paths)

    # Triple line thickness
    for collection in clustergrid.ax_row_dendrogram.collections:
        collection.set_linewidth(3.0)
    for collection in clustergrid.ax_col_dendrogram.collections:
        collection.set_linewidth(3.0)

    clustergrid.fig.suptitle("Kinematic similarity", fontsize=28, y=1.02)

    # Increase tick label font size by 2pt
    plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=0, fontsize=14)
    plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=14)

    # Mirror horizontally ("flipped horinztalle") across the Y-axis!
    clustergrid.ax_heatmap.invert_xaxis()
    clustergrid.ax_col_dendrogram.invert_xaxis()

    shared_dir = os.path.join(OUTPUT_DIR, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    clustergrid.savefig(os.path.join(shared_dir, f"{project_name.lower()}_fig3h_kinematic_similarity.png"), dpi=300)
    plt.close()


def plot_fig4e_relative_change(sessions, project_name):
    """
    Fig 4E: Boxplot of change in syllable frequency (Amph / Saline baseline)
    for WT and KI groups.
    """
    print("Plotting Fig 4E: Relative Change in Syllable Usage")
    all_bouts = []
    for s in sessions:
        all_bouts.append(metrics.get_syllable_instances(s.all_data_df))
    if not all_bouts: return
    global_bouts = pd.concat(all_bouts).groupby(level=0).sum()
    top_sylls = global_bouts.sort_values(ascending=False).head(20).index.tolist()
    letters = 'abcdefghijklmnopqrst'
    letter_map = {syll: letters[i] for i, syll in enumerate(top_sylls)}

    counts_dict = {"WT": {"Saline": [], "Amphetamine": []}, "KI": {"Saline": [], "Amphetamine": []}}

    for s in sessions:
        counts = metrics.get_syllable_instances(s.all_data_df)
        session_mins = len(s.all_data_df) / s.metadata.get("fs", 80.0) / 60.0
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")
        if geno not in counts_dict: continue

        if drug.lower().startswith("sal"): d = "Saline"
        elif drug.lower().startswith("amph"): d = "Amphetamine"
        else: continue

        c = {syll: (counts.get(syll, 0) / session_mins) for syll in top_sylls}
        c["animal_id"] = s.metadata.get("animal_id")
        counts_dict[geno][d].append(c)

    rows = []
    ordered_letters = [letter_map[s] for s in top_sylls][::-1]

    for geno in ["WT", "KI"]:
        sal_data = pd.DataFrame(counts_dict[geno]["Saline"])
        amph_data = pd.DataFrame(counts_dict[geno]["Amphetamine"])

        if sal_data.empty or amph_data.empty: continue

        # Baseline = mean for each syllable in saline
        sal_mean = sal_data[top_sylls].mean()

        for _, row in amph_data.iterrows():
            animal = row.get("animal_id", "UNK")
            for syll in top_sylls:
                base = sal_mean[syll]
                if base == 0: continue
                fc = row[syll] / base
                rows.append({
                    "Genotype": geno,
                    "Animal": animal,
                    "Syllable": letter_map[syll],
                    "Relative_Change": fc
                })

    df = pd.DataFrame(rows)
    if df.empty: return

    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(data=df, x="Syllable", y="Relative_Change", hue="Genotype",
                     hue_order=["WT", "KI"], order=ordered_letters, showfliers=True,
                     flierprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 5, 'alpha': 0.6},
                     palette="Set2")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Genotype")

    plt.axhline(1.0, color='black', linestyle='--')
    plt.title(f"Change from Saline Vehicle - {project_name}")
    plt.ylabel("Relative change (Fold over mean Saline)")
    sns.despine(trim=True)
    plt.tight_layout()

    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)
    plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig4e_relative_change.png"), dpi=300)
    plt.close()


def plot_fig4l_transition_entropy(sessions, project_name):
    """
    Fig 4L (PNAS 2023): Transition Entropy of syllable sequencing.
    Paired boxplots Saline -> Amphetamine.
    Higher entropy = more randomized sequencing (less orderly transitions).
    Reference: Jørgensen et al. 2023, PNAS Fig 4L.
    """
    print("Plotting Fig 4L: Transition Entropy (Syllable Sequencing Orderliness)")
    rows = []
    for s in sessions:
        res = metrics.calculate_entropy_metrics(s)
        ent = res.get("entropy_transition", np.nan)
        if pd.isna(ent): continue
        group_tag = _get_group_tag(s)
        if not group_tag: continue

        rows.append({
            "Animal": s.metadata.get("animal_id"),
            "Genotype": s.metadata.get("genotype"),
            "Drug": "Saline" if "Saline" in group_tag else "Amphetamine",
            "Transition_Entropy": ent
        })

    df = pd.DataFrame(rows)
    if df.empty or "Drug" not in df.columns: return

    for geno in ["WT", "KI"]:
        sub = df[df["Genotype"] == geno]
        if sub.empty: continue

        plt.figure(figsize=(5, 5))
        ax = sns.boxplot(data=sub, x="Drug", y="Transition_Entropy", order=["Saline", "Amphetamine"],
                         palette="pastel", showfliers=False, width=0.5)

        pt = sub.groupby(["Animal", "Drug"])["Transition_Entropy"].mean().unstack("Drug").dropna()
        if pt.empty:
            plt.close()
            continue

        sal_vals = []
        amph_vals = []

        for idx, row in pt.iterrows():
            sal_val = row.get("Saline")
            amph_val = row.get("Amphetamine")
            if pd.notna(sal_val) and pd.notna(amph_val):
                sal_vals.append(sal_val)
                amph_vals.append(amph_val)
                color = "#4c72b0" if geno == "WT" else "#55a868"
                plt.plot([0, 1], [sal_val, amph_val], marker='o', color=color, alpha=0.7)

        # Statistical Test
        if len(sal_vals) > 2:
            stat, p = stats.ttest_rel(sal_vals, amph_vals)
            if p < 0.05:
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                y_max = max(max(sal_vals), max(amph_vals)) * 1.05
                plt.plot([0, 0, 1, 1], [y_max, y_max*1.02, y_max*1.02, y_max], lw=1.5, color='black')
                plt.text(0.5, y_max*1.03, sig, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

        plt.title(f"Fig 4L: Transition Entropy (Sequencing Orderliness)\n{geno} | {project_name}")
        plt.ylabel("Transition Entropy (bits)")
        plt.xlabel("Drug Condition")
        sns.despine(trim=True)
        plt.tight_layout()

        group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
        os.makedirs(group_dir, exist_ok=True)
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig4l_{geno.lower()}_transition_entropy.png"), dpi=300)
        plt.close()

def plot_supplementary_variability(sessions, project_name):
    """
    Supplementary paired plots for Saline -> Amphetamine.
    Includes bounds on Usage Entropy, Duration Entropy, Initiations/Min, and Stereotypy.
    """
    print("Plotting Supplementary Variability Metrics")
    rows = []
    for s in sessions:
        group_tag = _get_group_tag(s)
        if not group_tag: continue

        # 1. Base Metrics
        res = metrics.analyze_session(s)

        # 2. Initiations per minute
        df = s.all_data_df
        total_bouts = 0
        if "syllable" in df.columns:
            sylls = df["syllable"].fillna(-1).to_numpy()
            bouts_mask = np.concatenate(([True], sylls[1:] != sylls[:-1]))
            bouts = sylls[bouts_mask]
            total_bouts = len(bouts[bouts != -1])

        duration_mins = len(df) / s.metadata.get("fs", 80.0) / 60.0
        init_rate = total_bouts / duration_mins if duration_mins > 0 else np.nan

        # 3. Usage Entropy (Overall time distribution)
        usage = metrics.get_syllable_usage(s)
        usage_probs = usage.values
        usage_probs = usage_probs[usage_probs > 0]
        usage_entropy = -np.sum(usage_probs * np.log2(usage_probs)) if len(usage_probs) > 0 else np.nan

        rows.append({
            "Animal": s.metadata.get("animal_id"),
            "Genotype": s.metadata.get("genotype"),
            "Drug": "Saline" if "Saline" in group_tag else "Amphetamine",
            "Stereotypy": res.get("stereotypy", np.nan),
            "Duration_Entropy": res.get("entropy_duration", np.nan),
            "Usage_Entropy": usage_entropy,
            "Initiations_Per_Min": init_rate
        })

    df = pd.DataFrame(rows)
    if df.empty or "Drug" not in df.columns: return

    metrics_to_plot = {
        "Stereotypy":         ("Supp",  "Stereotypy (1 - Norm. Transition Entropy)"),
        "Duration_Entropy":   ("Supp",  "Duration Entropy (bits)"),
        "Usage_Entropy":      ("Fig4F", "Fig 4F: Usage Entropy (Syllable Repertoire Diversity)"),
        "Initiations_Per_Min":("Supp",  "Initiations Per Minute"),
    }

    for metric, (fig_label, axis_label) in metrics_to_plot.items():
        for geno in ["WT", "KI"]:
            sub = df[(df["Genotype"] == geno) & (df[metric].notna())]
            if sub.empty: continue

            plt.figure(figsize=(5, 5))
            ax = sns.boxplot(data=sub, x="Drug", y=metric, order=["Saline", "Amphetamine"],
                             palette="pastel", showfliers=False, width=0.5)

            pt = sub.groupby(["Animal", "Drug"])[metric].mean().unstack("Drug").dropna()
            if pt.empty:
                plt.close()
                continue

            sal_vals = []
            amph_vals = []

            for idx, row in pt.iterrows():
                sal_val = row.get("Saline")
                amph_val = row.get("Amphetamine")
                if pd.notna(sal_val) and pd.notna(amph_val):
                    sal_vals.append(sal_val)
                    amph_vals.append(amph_val)
                    color = "#4c72b0" if geno == "WT" else "#55a868"
                    plt.plot([0, 1], [sal_val, amph_val], marker='o', color=color, alpha=0.7)

            # Statistical Test
            if len(sal_vals) > 2:
                stat, p = stats.ttest_rel(sal_vals, amph_vals)
                if p < 0.05:
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    y_max = max(max(sal_vals), max(amph_vals))
                    y_range = max(sal_vals + amph_vals) - min(sal_vals + amph_vals)
                    buffer = y_range * 0.05 if y_range > 0 else y_max * 0.05
                    top_line = y_max + buffer

                    plt.plot([0, 0, 1, 1], [top_line, top_line + buffer*0.5, top_line + buffer*0.5, top_line], lw=1.5, color='black')
                    plt.text(0.5, top_line + buffer*0.6, sig, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

            plt.title(f"{axis_label}\n{geno} | {project_name}")
            plt.ylabel(axis_label)
            plt.xlabel("Drug Condition")
            sns.despine(trim=True)
            plt.tight_layout()

            group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
            os.makedirs(group_dir, exist_ok=True)

            # Usage Entropy → fig4f, others → supp_fig
            if fig_label == "Fig4F":
                fname = f"{project_name.lower()}_fig4f_{geno.lower()}_usage_entropy.png"
            else:
                fname = f"{project_name.lower()}_supp_{geno.lower()}_{metric.lower()}.png"

            plt.savefig(os.path.join(group_dir, fname), dpi=300)
            plt.close()



def plot_fig4k_transition_scatter(sessions, project_name):
    """
    Fig 4K: Change in transition freq (Amph vs Saline) scatter plot with least square lines.
    """
    print("Plotting Fig 4K: Transition Frequency Scatter")
    groups = {}
    for s in sessions:
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")
        if geno in ["None", "UNK", None]: continue

        if drug.lower().startswith("sal"): d = "Saline"
        elif drug.lower().startswith("amph"): d = "Amphetamine"
        else: continue

        key = (geno, d)
        if key not in groups: groups[key] = []
        groups[key].append(s)

    all_bouts = []
    for s in sessions:
        all_bouts.append(metrics.get_syllable_instances(s.all_data_df))
    if not all_bouts: return
    top_sylls = pd.concat(all_bouts).groupby(level=0).sum().sort_values(ascending=False).head(20).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, geno in enumerate(["WT", "KI"]):
        ax = axes[i]
        key_sal = (geno, "Saline")
        key_amph = (geno, "Amphetamine")

        if key_sal not in groups or key_amph not in groups:
            ax.set_title(f"{geno}: Missing Data")
            continue

        m_sal = metrics.get_group_transition_matrix(groups[key_sal], ordered_syllables=top_sylls)
        m_amph = metrics.get_group_transition_matrix(groups[key_amph], ordered_syllables=top_sylls)

        m_sal_flat = m_sal.values.flatten()
        m_amph_flat = m_amph.values.flatten()

        mask = (m_sal_flat > 0)
        x = m_sal_flat[mask]
        y_raw = m_amph_flat[mask]
        y = y_raw / x

        # We need positive values for log, and some y_raw might be 0, so y is 0
        valid_log = y > 0
        x = x[valid_log]
        y = y[valid_log]

        if len(x) == 0: continue

        ax.scatter(x, y, color='grey', alpha=0.7)
        ax.axhline(1.0, color='black', linestyle='--')

        # Fit on log-log
        slope, intercept, _, p_value, _ = stats.linregress(np.log10(x), np.log10(y))

        # Plot fit line
        x_fit = np.array([np.min(x), np.max(x)])
        # log(y) = intercept + slope*log(x)  => y = 10^(intercept) * x^slope
        y_fit = (10**intercept) * (x_fit**slope)
        ax.plot(x_fit, y_fit, color='black', alpha=0.8)

        ax.set_xscale('log')
        ax.set_yscale('log')

        # Force specific ticks formatting
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val*100:g}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:g}"))

        ax.text(0.6, 0.9, f"Slope: p={p_value:.3f}", transform=ax.transAxes)
        ax.set_xlabel("Transition freq. after vehicle (%)")
        ax.set_ylabel("Change in transition freq. (fold-change)")
        ax.set_title(f"{geno}")

    plt.tight_layout()
    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)
    plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig4k_transition_scatter.png"), dpi=300)
    plt.close()


def plot_fig4ij_transition_networks(sessions, project_name):
    """
    Fig 4I/J: Circular diagram of syllable transitions.
    """
    print("Plotting Fig 4I/J: Syllable Transition Networks")
    groups = {}
    for s in sessions:
        tag = _get_group_tag(s)
        if not tag: continue
        if tag not in groups: groups[tag] = []
        groups[tag].append(s)

    all_bouts = []
    for s in sessions:
        all_bouts.append(metrics.get_syllable_instances(s.all_data_df))
    if not all_bouts: return
    top_sylls = pd.concat(all_bouts).groupby(level=0).sum().sort_values(ascending=False).head(20).index.tolist()

    letters = 'abcdefghijklmnopqrst'
    letter_map = {syll: letters[i] for i, syll in enumerate(top_sylls)}

    freq_series = pd.concat(all_bouts).groupby(level=0).sum()
    total = freq_series.sum()
    norm_freq = (freq_series / total).loc[top_sylls]
    norm_freq.index = [letter_map[x] for x in norm_freq.index]

    keys = ["WT Saline", "WT Amphetamine", "KI Saline", "KI Amphetamine"]
    keys = [k for k in keys if k in groups]

    if not keys: return

    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 7))
    if len(keys) == 1: axes = [axes]

    for i, tag in enumerate(keys):
        ax = axes[i]
        mat = metrics.get_group_transition_matrix(groups[tag], ordered_syllables=top_sylls, normalize="global")

        mat.index = [letter_map[x] for x in mat.index]
        mat.columns = [letter_map[x] for x in mat.columns]

        networks.EDGE_CUTOFF = 0.005 # 0.5% Global threshold
        networks.plot_directed_network(mat, norm_freq, title=f"Syllable transitions\n{tag}", ax=ax)

    plt.tight_layout()
    group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
    os.makedirs(group_dir, exist_ok=True)
    plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig4ij_transition_networks.png"), facecolor='white', dpi=300)
    plt.close()

    # Split into Genotype-specific PNGs for slides
    for geno in ["WT", "KI"]:
        geno_keys = [k for k in keys if k.startswith(geno)]
        if not geno_keys: continue

        fig, axes = plt.subplots(1, len(geno_keys), figsize=(7 * len(geno_keys), 7))
        if len(geno_keys) == 1: axes = [axes]

        for i, tag in enumerate(geno_keys):
            ax = axes[i]
            mat = metrics.get_group_transition_matrix(groups[tag], ordered_syllables=top_sylls, normalize="global")
            mat.index = [letter_map[x] for x in mat.index]
            mat.columns = [letter_map[x] for x in mat.columns]

            networks.EDGE_CUTOFF = 0.005 # 0.5% Global threshold
            networks.plot_directed_network(mat, norm_freq, title=f"Syllable transitions\n{tag}", ax=ax)

        plt.tight_layout()
        group_dir = os.path.join(OUTPUT_DIR, project_name.lower())
        os.makedirs(group_dir, exist_ok=True)
        plt.savefig(os.path.join(group_dir, f"{project_name.lower()}_fig4ij_{geno.lower()}_transitions.png"), facecolor='white', dpi=300)
        plt.close()


def run_replication(sessions, project_name, global_kinematics=None):
    print(f"=== Starting PNAS Plot Replications for {project_name} ===")
    if not sessions or len(sessions) == 0:
        return

    plot_fig3c_syllable_usage(sessions, project_name)
    plot_fig4ij_transition_networks(sessions, project_name)

    if "amph" in project_name.lower() and "both" not in project_name.lower():
        plot_fig4l_transition_entropy(sessions, project_name)
        plot_supplementary_variability(sessions, project_name)

    print(f"Completed {project_name} PNAS Replications.")


# Re-export from dedicated module for backward compatibility
from analysis.ulrik_kinematics.lib.kinematic_similarity import (
    get_kinematic_similarity,
)

