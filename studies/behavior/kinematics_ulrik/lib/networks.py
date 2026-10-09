import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from analysis.ulrik_kinematics.lib import motifs

# Tunable threshold: only draw arrows that account for > EDGE_CUTOFF (3%) of outgoing transitions
EDGE_CUTOFF = 0.03

# Delta cutoff (absolute shift in probability required to render a delta edge)
EDGE_DELTA_CUTOFF = 0.02

def build_transition_matrix(sessions, mapping=None, top_n=10):
    """
    Extracts global Markov transition probabilities across the provided sessions.
    Returns:
       - TransDF: (NxN dataframe) The Probability of Row -> Col
       - FreqDF: (Series) Relative node size (total usage frequency)
    """
    all_transitions = []

    # 1. Gather all sequence pairs, matching motifs logic
    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns: continue

        raw_seq = df["syllable"].fillna(-1).astype(float).to_numpy()
        mask = np.concatenate(([True], raw_seq[1:] != raw_seq[:-1]))
        seq = raw_seq[mask]

        for i in range(len(seq) - 1):
            src = seq[i]
            dst = seq[i+1]
            if src == -1 or dst == -1: continue

            src_lbl = mapping.get(src, "UNK") if mapping else src
            dst_lbl = mapping.get(dst, "UNK") if mapping else dst

            if src_lbl == "UNK" or dst_lbl == "UNK" or pd.isna(src_lbl) or pd.isna(dst_lbl):
                continue

            all_transitions.append((src_lbl, dst_lbl))

    if not all_transitions:
        return pd.DataFrame(), pd.Series()

    bg_df = pd.DataFrame(all_transitions, columns=["src", "dst"])

    # Calculate Node Sizes (Bout occurrences, not frame counts)
    src_counts = bg_df["src"].value_counts()

    # Keep only the top N syllables to make the diagram readable
    valid_nodes = src_counts.head(top_n).index.tolist()
    bg_df = bg_df[bg_df["src"].isin(valid_nodes) & bg_df["dst"].isin(valid_nodes)]

    # Recompute sizes post-filter
    final_counts = bg_df["src"].value_counts(normalize=True)

    trans_matrix = pd.crosstab(bg_df["src"], bg_df["dst"], normalize="index")

    return trans_matrix, final_counts

def plot_directed_network(trans_df, freq_series, title, ax):
    """
    Draws the NetworkX visualization onto the provided matplotlib axis.
    """
    if trans_df.empty:
        ax.set_title("Insufficient data")
        return

    G = nx.DiGraph()

    for lbl in trans_df.index:
        G.add_node(lbl, weight=freq_series.get(lbl, 0))

    for src in trans_df.index:
        for dst in trans_df.columns:
            prob = trans_df.loc[src, dst]
            if prob > EDGE_CUTOFF:
                G.add_edge(src, dst, weight=prob)

    # Isolate singletons without edges
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    if G.number_of_nodes() == 0:
        ax.set_title("No connections > Threshold")
        return

    # Use a circular layout (Polygon) as requested
    pos = nx.circular_layout(G, scale=2.0)

    node_size_dict = {n: G.nodes[n]['weight'] * 15000 for n in G.nodes()}
    node_sizes = [node_size_dict[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='lightblue', edgecolors='black', linewidths=2)

    global_mode = EDGE_CUTOFF < 0.01
    width_scale = 160.0 if global_mode else 8.0
    alpha_scale = 15.0 if global_mode else 1.5

    # Draw edges with varying opacity/width based on transition probability.
    # min_target_margin is scaled to node size so arrowheads land just outside
    # large nodes instead of being hidden underneath them.
    for u, v, data in G.edges(data=True):
        w = data['weight']
        edge_alpha = np.clip(w * alpha_scale, 0.2, 1.0)
        edge_width = w * width_scale
        target_size = node_size_dict.get(v, 300)
        # sqrt gives a reasonable pixel-space radius estimate from the area-based size
        target_margin = np.sqrt(target_size) * 0.6
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            width=edge_width, alpha=edge_alpha,
            edge_color='gray', arrowsize=20,
            connectionstyle='arc3,rad=0.15',
            min_target_margin=target_margin
        )

    # Offset labels radially so they don't overlap the polygon nodes
    label_pos = {}
    for node, (x, y) in pos.items():
        # Move outward from center for radial labeling
        # Using a slight push outward based on the normalized vector
        norm = np.sqrt(x**2 + y**2)
        if norm > 0:
            label_pos[node] = (x + (x/norm)*0.5, y + (y/norm)*0.5)
        else:
            label_pos[node] = (x, y)

    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=11, font_weight="bold", font_color="darkred")

    # Expand margins so labels don't get cropped by the matplotlib figure boundary
    ax.margins(0.2)
    ax.axis('off')

    # Build a custom legend for thickness -> probability mapping
    from matplotlib.lines import Line2D
    if global_mode:
        legend_elements = [
            Line2D([0], [0], color='gray', lw=EDGE_CUTOFF * width_scale, label=f'{EDGE_CUTOFF*100:.1f}% (Cutoff)'),
            Line2D([0], [0], color='gray', lw=0.01 * width_scale, label='1% Global Prob'),
            Line2D([0], [0], color='gray', lw=0.03 * width_scale, label='3% Global Prob'),
            Line2D([0], [0], color='gray', lw=0.05 * width_scale, label='5% Global Prob'),
        ]
        legend_title = "Global Transition\n(Edge Thickness)"
    else:
        legend_elements = [
            Line2D([0], [0], color='gray', lw=EDGE_CUTOFF * width_scale, label=f'{EDGE_CUTOFF*100:.1f}% (Cutoff)'),
            Line2D([0], [0], color='gray', lw=0.1 * width_scale, label='10% Probability'),
            Line2D([0], [0], color='gray', lw=0.3 * width_scale, label='30% Probability'),
            Line2D([0], [0], color='gray', lw=0.6 * width_scale, label='60% Probability'),
        ]
        legend_title = "Transition Probability\n(Edge Thickness)"

    # Place legend outside the main graph area
    ax.legend(handles=legend_elements, loc='upper right', title=legend_title, fontsize=9, title_fontsize=10)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.text(0.99, 0.01, f"Cutoff: {EDGE_CUTOFF*100:.2f}%", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

def plot_delta_network(delta_df, freq_series, title, ax):
    """
    Draws a specific differential network where edges can be negative or positive.
    Red = Positive Delta, Blue = Negative Delta.
    """
    if delta_df.empty:
        ax.set_title("Insufficient data")
        return

    G = nx.DiGraph()

    for lbl in delta_df.index:
        G.add_node(lbl, weight=freq_series.get(lbl, 0))

    for src in delta_df.index:
        for dst in delta_df.columns:
            val = delta_df.loc[src, dst]
            if abs(val) > EDGE_DELTA_CUTOFF:
                G.add_edge(src, dst, weight=val)

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    if G.number_of_nodes() == 0:
        ax.set_title("No connections > Delta Threshold")
        return

    pos = nx.circular_layout(G, scale=2.0)
    node_size_dict = {n: G.nodes[n]['weight'] * 15000 for n in G.nodes()}
    node_sizes = [node_size_dict[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='lightgray', edgecolors='black', linewidths=2)

    for u, v, data in G.edges(data=True):
        w = data['weight']
        edge_color = 'red' if w > 0 else 'blue'
        edge_alpha = np.clip(abs(w) * 3.0, 0.2, 1.0)
        edge_width = abs(w) * 15.0
        target_size = node_size_dict.get(v, 300)
        target_margin = np.sqrt(target_size) * 0.6
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            width=edge_width, alpha=edge_alpha,
            edge_color=edge_color, arrowsize=20,
            connectionstyle='arc3,rad=0.15',
            min_target_margin=target_margin
        )

    label_pos = {}
    for node, (x, y) in pos.items():
        norm = np.sqrt(x**2 + y**2)
        if norm > 0:
            label_pos[node] = (x + (x/norm)*0.5, y + (y/norm)*0.5)
        else:
            label_pos[node] = (x, y)

    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=11, font_weight="bold", font_color="black")
    ax.margins(0.2)
    ax.axis('off')

    # Legend for Delta
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='+ Increase'),
        Line2D([0], [0], color='blue', lw=3, label='- Decrease'),
        Line2D([0], [0], color='gray', lw=EDGE_DELTA_CUTOFF * 15.0, label=f'{EDGE_DELTA_CUTOFF*100:.1f}% Cutoff')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Transition Shift", fontsize=9, title_fontsize=10)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.text(0.99, 0.01, f"Delta Cutoff: ±{EDGE_DELTA_CUTOFF*100:.1f}%", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

def run_discovery(sessions, project_name="AMPH", style="alphabet"):
    print(f"\n--- Running Networks Discovery ({style.capitalize()} Mode) ---")
    base_project = project_name.split("_")[0]
    mapping = motifs.get_label_mapping(sessions, project=base_project, style=style)

    drugs = ["Saline", "Amphetamine"] if base_project == "AMPH" else ["Baseline"]

    mats = {}
    freqs = {}

    for drug in drugs:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(f"Behavioral Transition Networks: {project_name} ({drug}) - [{style.capitalize()}]", fontsize=18, fontweight="bold")

        # Partition data
        d_val = drug if base_project == "AMPH" else None

        s_wt = [s for s in sessions if s.metadata.get("genotype") == "WT" and (d_val is None or s.metadata.get("drug", "").lower().startswith(d_val[:3].lower()))]
        s_ki = [s for s in sessions if s.metadata.get("genotype") == "KI" and (d_val is None or s.metadata.get("drug", "").lower().startswith(d_val[:3].lower()))]

        if not s_wt or not s_ki: continue

        t_wt, f_wt = build_transition_matrix(s_wt, mapping=mapping, top_n=10)
        t_ki, f_ki = build_transition_matrix(s_ki, mapping=mapping, top_n=10)


        out_dir = os.path.join(os.path.dirname(__file__), "../output/networks")
        os.makedirs(out_dir, exist_ok=True)

        # We need these saved independently to compute Deltas afterwards
        if drug == "Saline" or base_project == "IFDN":
            mats["wt_base"] = t_wt
            mats["ki_base"] = t_ki
            freqs["base"] = (f_wt + f_ki) / 2
        elif drug == "Amphetamine":
            mats["wt_amph"] = t_wt
            mats["ki_amph"] = t_ki
            freqs["amph"] = (f_wt + f_ki) / 2

        plot_directed_network(t_wt, f_wt, "Wild Type (WT)", axes[0])
        plot_directed_network(t_ki, f_ki, "Knock In (KI)", axes[1])

        plt.tight_layout()
        out_file = os.path.join(out_dir, f"{project_name.lower()}_{drug.lower()}_{style}_network.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

    # --- PLOT DELTAS ---
    if "wt_base" in mats and "ki_base" in mats:
        # Genotype Baseline Delta (KI Baseline - WT Baseline)
        delta_base = mats["ki_base"].subtract(mats["wt_base"], fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 9))
        fig.suptitle(f"Genotype Baseline Delta (KI - WT) - [{style.capitalize()}]", fontsize=16, fontweight="bold")
        plot_delta_network(delta_base, freqs["base"], "Red = KI > WT  |  Blue = WT > KI", ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_delta_baseline_{style}_network.png"), dpi=300, bbox_inches="tight")
        plt.close()

    if base_project == "AMPH" and all(k in mats for k in ["wt_base", "ki_base", "wt_amph", "ki_amph"]):
        # Delta of Deltas: (KI Amph - KI Saline) - (WT Amph - WT Saline)
        wt_shift = mats["wt_amph"].subtract(mats["wt_base"], fill_value=0)
        ki_shift = mats["ki_amph"].subtract(mats["ki_base"], fill_value=0)
        dod = ki_shift.subtract(wt_shift, fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 9))
        fig.suptitle(f"Delta-of-Deltas: Phenotypic Drug Response - [{style.capitalize()}]", fontsize=16, fontweight="bold")
        plot_delta_network(dod, (freqs["base"] + freqs["amph"])/2, "Red = Amph amplifies in KI uniquely\nBlue = Amph suppresses in KI uniquely", ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_dod_amphetamine_{style}_network.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- NEW: Drug Effect Comparison (WT Effect vs KI Effect) ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Drug-Induced Transition Shifts (Amphetamine - Saline) - [{style.capitalize()}]", fontsize=18, fontweight="bold")

        plot_delta_network(wt_shift, (freqs["base"] + freqs["amph"])/2, "WT: Drug Effect (Amph - Saline)", axes[0])
        plot_delta_network(ki_shift, (freqs["base"] + freqs["amph"])/2, "KI: Drug Effect (Amph - Saline)", axes[1])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_drug_effect_comparison_{style}_network.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- STATS LOGGING ---
        stats_file = os.path.join(out_dir, f"{project_name.lower()}_network_stats.txt")
        with open(stats_file, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"PROJECT: {project_name} | STYLE: {style}\n")
            f.write(f"{'='*50}\n\n")

            f.write("--- WT AMPHETIC SHIFT (WT Amph - WT Saline) ---\n")
            # Flatten, drop self-loops, sort
            wt_s_flat = wt_shift.stack().reset_index()
            wt_s_flat.columns = ["From", "To", "Delta"]
            wt_s_flat = wt_s_flat[wt_s_flat["From"] != wt_s_flat["To"]]
            for _, row in wt_s_flat.sort_values("Delta", ascending=False).head(5).iterrows():
                f.write(f"  (+) {row['From']:<12} -> {row['To']:<12} | +{row['Delta']*100:.1f}%\n")
            for _, row in wt_s_flat.sort_values("Delta", ascending=True).head(5).iterrows():
                f.write(f"  (-) {row['From']:<12} -> {row['To']:<12} | {row['Delta']*100:.1f}%\n")

            f.write("\n--- KI AMPHETIC SHIFT (KI Amph - KI Saline) ---\n")
            ki_s_flat = ki_shift.stack().reset_index()
            ki_s_flat.columns = ["From", "To", "Delta"]
            ki_s_flat = ki_s_flat[ki_s_flat["From"] != ki_s_flat["To"]]
            for _, row in ki_s_flat.sort_values("Delta", ascending=False).head(5).iterrows():
                f.write(f"  (+) {row['From']:<12} -> {row['To']:<12} | +{row['Delta']*100:.1f}%\n")
            for _, row in ki_s_flat.sort_values("Delta", ascending=True).head(5).iterrows():
                f.write(f"  (-) {row['From']:<12} -> {row['To']:<12} | {row['Delta']*100:.1f}%\n")

            f.write("\n--- DELTA OF DELTAS (KI Shift - WT Shift) ---\n")
            f.write(" Positive: Amph amplifies this pathway in KI uniquely.\n")
            f.write(" Negative: Amph suppresses this pathway in KI uniquely.\n")
            dod_flat = dod.stack().reset_index()
            dod_flat.columns = ["From", "To", "Delta"]
            dod_flat = dod_flat[dod_flat["From"] != dod_flat["To"]]
            for _, row in dod_flat.sort_values("Delta", ascending=False).head(5).iterrows():
                f.write(f"  [AMPLIFIED]   {row['From']:<12} -> {row['To']:<12} | {row['Delta']*100:+.1f}%\n")
            for _, row in dod_flat.sort_values("Delta", ascending=True).head(5).iterrows():
                f.write(f"  [SUPPRESSED]  {row['From']:<12} -> {row['To']:<12} | {row['Delta']*100:+.1f}%\n")


    print(f"Saved Network Topology {style} plots to output/networks/")

def verify_networks_module(sessions):
    """
    Mathematical defense test for Transition Probability Matrices.
    Asserts row conservation (sum of outgoing probabilities must equal 1.0).
    """
    print("\n" + "="*50)
    print(" VERIFICATION: MARKOV MATRIX INVARIANCE")
    print("="*50)

    mapping = motifs.get_label_mapping(sessions, style="alphabet")
    trans_matrix, _ = build_transition_matrix(sessions[:2], mapping=mapping, top_n=5)

    if trans_matrix.empty:
        print("FAIL: Insufficient data to build transition matrix.")
        return

    row_sums = trans_matrix.sum(axis=1)
    max_error = np.max(np.abs(row_sums - 1.0))

    print(f"Matrix Dimension: {trans_matrix.shape}")
    print(f"Maximum Row Error (Deviation from 1.0): {max_error:.6f}")

    if max_error < 0.0001:
        print("STATUS: PASS -> All paths strictly conserve probability.")
    else:
        print("STATUS: FAIL -> Math leakage detected.")
    print("="*50 + "\n")
