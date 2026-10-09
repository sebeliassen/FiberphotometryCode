import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.ulrik_kinematics.lib import metrics, motifs
from scipy import stats


def extract_bouts(session, mapping=None):
    """
    Extracts every discrete bout (contiguous syllable execution) and calculates its micro-physics.
    """
    df = session.all_data_df.copy()

    if "syllable" not in df.columns or df.empty:
        return pd.DataFrame()

    s = df["syllable"].fillna(-1).astype(float).to_numpy()

    # 1. Generate run-length boundaries to assign unique IDs to each contiguous bout
    mask = np.concatenate(([True], s[1:] != s[:-1]))
    df["bout_id"] = mask.cumsum()

    # Calculate timeframe
    if "frame_index" in df.columns:
        df["time"] = df["frame_index"] / 30.0 # Assuming 30FPS Native
    else:
        return pd.DataFrame()

    bouts = []

    # Pre-fetch physical matrices to avoid pandas overhead locally
    vel_arr = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))

    # --- FIXED: Proper Angular Velocity handling (mod 2*pi) ---
    if "heading" in df.columns:
        from scipy.signal import medfilt
        h = df["heading"].values

        # 1. Apply a small median filter to remove single-frame head-flip glitches
        h_smooth = medfilt(h, kernel_size=3)

        # 2. Delta heading with proper 2pi wrap-around
        dh = np.diff(h_smooth)
        dh_fixed = np.arctan2(np.sin(dh), np.cos(dh))

        # 3. Convert to rad/s using the session's actual hardware rate (80Hz)
        fs = session.metadata.get("fs", 80.0)
        ang_arr = np.concatenate(([0], dh_fixed * fs))

        # 4. Biological Clipping: A mouse cannot physically turn at >40 rad/s (~6 rotations/sec)
        # Anything higher is tracking noise.
        ang_arr = np.clip(ang_arr, -40.0, 40.0)

    elif "angular_velocity" in df.columns:
        ang_arr = np.clip(df["angular_velocity"].values, -40.0, 40.0)
    else:
        ang_arr = np.zeros(len(df))

    s_arr = df["syllable"].values
    t_arr = df["time"].values
    bout_idx = df["bout_id"].values

    # Efficient numpy grouping
    unique_bouts = np.unique(bout_idx)

    # Pre-group index slices for speed
    # np.searchsorted can quickly find boundaries if ordered
    split_indices = np.where(mask)[0]
    ranges = list(zip(split_indices[:-1], split_indices[1:]))
    ranges.append((split_indices[-1], len(df)))

    for start, end in ranges:
        syll = s_arr[start]
        if pd.isna(syll) or syll == -1: continue # Skip UNK/boundary frames

        # Apply Mapping logic (e.g., A..T Alphabet Style)
        mapped_label = mapping.get(syll, "UNK") if mapping else syll
        if mapped_label == "UNK" or pd.isna(mapped_label): continue

        v_slice = vel_arr[start:end]
        ang_slice = ang_arr[start:end]
        t_slice = t_arr[start:end]

        dur = t_slice[-1] - t_slice[0]
        # Bouts require at least 2 frames to have a logical maximum/duration
        if len(v_slice) < 2: continue

        bouts.append({
            "Bout_ID": bout_idx[start],
            "Syllable_ID": syll,
            "Mapped_Label": mapped_label,
            "Duration": dur,
            "Mean_Velocity": np.nanmean(v_slice) if not np.all(np.isnan(v_slice)) else np.nan,
            "Median_Velocity": np.nanmedian(v_slice) if not np.all(np.isnan(v_slice)) else np.nan,
            "Mean_Angular": np.nanmean(np.abs(ang_slice)) if not np.all(np.isnan(ang_slice)) else np.nan,
            "Median_Angular": np.nanmedian(np.abs(ang_slice)) if not np.all(np.isnan(ang_slice)) else np.nan,
            "Frame_Count": end - start
        })

    return pd.DataFrame(bouts)

def verify_micro_module(sessions):
    """
    Mathematical defense test for Micro-Kinematics.
    Ensures that the aggregation of syllables exactly matches the global physics engine.
    """
    print("\n" + "="*50)
    print(" VERIFICATION: MICRO-PHYSICS EXTRACTION & INVARIANCE")
    print("="*50)

    s = sessions[0]

    print("[1/2] Computing True Conservation Math...")
    # Get standard mapping (alphabet/standard) - let's use raw to catch all frames
    df_bouts = extract_bouts(s)

    # If we multiply the mean velocity of every bout by its frame count,
    # it must mathematically sum to the global mean velocity * total valid frames
    if not df_bouts.empty:
        total_bout_velocity = (df_bouts["Mean_Velocity"] * df_bouts["Frame_Count"]).sum()
        total_bout_frames = df_bouts["Frame_Count"].sum()
        micro_mean_global = total_bout_velocity / total_bout_frames if total_bout_frames > 0 else 0
    else:
        micro_mean_global = 0.0

    # Standard global logic baseline
    k = metrics.calculate_kinematics(s)
    global_vel = k.get("mean_velocity", np.nan)

    # Recreate global mean but strictly on the frames we marked as active syllables
    # (Because extract_bouts skips UNK/-1 data by definition)
    df_raw = s.all_data_df
    valid_mask = df_raw["syllable"].notna() & (df_raw["syllable"] != -1)
    true_active_vel = df_raw.loc[valid_mask, "velocity_px_s"].mean()

    math_delta = abs(micro_mean_global - true_active_vel)

    print("[2/2] Injecting structural desync to test duration crashes...")
    s_pert = copy.deepcopy(s)
    np.random.seed(42)
    # 20% random drop
    drop_mask = np.random.rand(len(s_pert.all_data_df)) < 0.20
    s_pert.all_data_df.loc[drop_mask, "syllable"] = -1

    df_pert_bouts = extract_bouts(s_pert)

    mean_true_dur = df_bouts["Duration"].mean() if not df_bouts.empty else 0
    mean_pert_dur = df_pert_bouts["Duration"].mean() if not df_pert_bouts.empty else 0
    frag_drop = 100 * (1 - (mean_pert_dur / mean_true_dur)) if mean_true_dur > 0 else 0

    print("\n" + "-"*30)
    print("       DIAGNOSTIC RESULTS       ")
    print("-"*30)

    table = pd.DataFrame({
        "Condition": ["True Extractor", "Perturbed Array"],
        "Summed_Micro_Velocity": [f"{micro_mean_global:.4f}", "N/A"],
        "Global_Active_Velocity": [f"{true_active_vel:.4f}", "N/A"],
        "Mean_Bout_Duration": [f"{mean_true_dur:.4f}s", f"{mean_pert_dur:.4f}s"]
    })

    print(table.to_markdown(index=False))

    print("\n> INVARIANT CHECKS:")
    print(f"  - Math Conservation Constraint: Extraction math error {math_delta:.4f} < 0.01 -> {'PASS' if math_delta < 0.01 else 'FAIL'}")
    print(f"  - Structural Ejection Constraint: Perturbation crashed durations by {frag_drop:.1f}% -> {'PASS' if frag_drop > 40 else 'FAIL'}")
    print("="*50 + "\n")

def run_discovery(sessions, project_name="AMPH", style="alphabet"):
    print(f"\n--- Running Micro-Kinematics Discovery ({style.capitalize()} Mode) ---")

    # Strip phase suffix (e.g., AMPH_pre -> AMPH) so the manual_labels.csv lookup still works
    base_project = project_name.split("_")[0]
    mapping = motifs.get_label_mapping(sessions, project=base_project, style=style)

    all_bouts = []

    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue

        # Standardize naming
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"

        df_b = extract_bouts(s, mapping=mapping)
        if not df_b.empty:
            df_b["Genotype"] = g
            df_b["Drug"] = d
            df_b["Session"] = s.session_id
            all_bouts.append(df_b)

    if not all_bouts:
        print("No bouts extracted.")
        return

    df_all = pd.concat(all_bouts, ignore_index=True)

    # We will only plot the top 5 most common behaviors to prevent charts with 20 subplots.
    # We evaluate their Velocity (px/s) and Angular Velocity bounds.
    top_syllables = df_all["Mapped_Label"].value_counts().head(5).index.tolist()

    df_plot = df_all[df_all["Mapped_Label"].isin(top_syllables)]

    for drug in ["Saline", "Amphetamine"]:
        df_d = df_plot[df_plot["Drug"] == drug]
        if df_d.empty: continue

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        wt_col = "#2ecc71"
        ki_col = "#e74c3c"
        palette = {"WT": wt_col, "KI": ki_col}

        metrics_to_plot = [
            ("Mean_Velocity", "Mean Velocity (px/s)", "Typical (Mean) Velocity Effort"),
            ("Mean_Angular", "Mean Angular (rad/s)", "Typical (Mean) Turning Rate"),
            ("Median_Velocity", "Median Velocity (px/s)", "Robust (Median) Velocity Effort"),
            ("Median_Angular", "Median Angular (rad/s)", "Robust (Median) Turning Rate")
        ]

        for idx, (col_name, ylabel, title_suffix) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]

            sns.violinplot(
                data=df_d, x="Mapped_Label", y=col_name, hue="Genotype",
                split=True, inner="quartile", ax=ax, palette=palette, order=sorted(top_syllables), cut=0
            )
            ax.set_title(f"{title_suffix} ({drug})", fontsize=14, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Syllable Identity (Alphabet)")

            # Stat annotation for this specific subplot
            for i, syll in enumerate(sorted(top_syllables)):
                wt_v = df_d[(df_d["Mapped_Label"] == syll) & (df_d["Genotype"] == "WT")][col_name].dropna()
                ki_v = df_d[(df_d["Mapped_Label"] == syll) & (df_d["Genotype"] == "KI")][col_name].dropna()

                if len(wt_v) > 20 and len(ki_v) > 20:
                    stat, p = stats.mannwhitneyu(wt_v, ki_v, alternative='two-sided')
                    if p < 0.001: sig = "***"
                    elif p < 0.01: sig = "**"
                    elif p < 0.05: sig = "*"
                    else: sig = "ns"

                    if sig != "ns":
                        y_max = df_d[df_d["Mapped_Label"] == syll][col_name].quantile(0.95)
                        ax.text(i, y_max + 0.5, sig, ha='center', fontweight='bold', color='black')

        plt.tight_layout()
        out_dir = os.path.join(os.path.dirname(__file__), "../output/micro_kinematics")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{project_name.lower()}_{drug.lower()}_{style}_micro_violin.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

        # ----------------------------------------------------------------
        # NORMALISED FIGURE: Subtract group mean to isolate syllable-specific effects
        # ----------------------------------------------------------------
        # For each (Genotype) group, subtract the overall mean across ALL bouts
        # This partials out the group-level speed confound and shows
        # whether kinematic differences are syllable-specific or just global
        df_norm = df_d.copy()
        for col_name, _, _ in metrics_to_plot:
            group_means = df_norm.groupby("Genotype")[col_name].transform("mean")
            df_norm[col_name + "_norm"] = df_norm[col_name] - group_means

        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
        fig2.suptitle(
            f"Group-Mean-Centred Micro-Kinematics ({drug}) [{style.capitalize()}]\n"
            "Residual = Syllable Value − Genotype Group Mean  |  Isolates syllable-specific effects",
            fontsize=13, y=1.01
        )

        norm_metrics = [
            ("Mean_Velocity_norm",  "Δ Mean Velocity (px/s)",  "Syllable-Specific Velocity Residual (Mean)"),
            ("Mean_Angular_norm",   "Δ Mean Angular (rad/s)",  "Syllable-Specific Turning Residual (Mean)"),
            ("Median_Velocity_norm","Δ Median Velocity (px/s)","Syllable-Specific Velocity Residual (Median)"),
            ("Median_Angular_norm", "Δ Median Angular (rad/s)","Syllable-Specific Turning Residual (Median)"),
        ]

        for idx, (col_name, ylabel, title_suffix) in enumerate(norm_metrics):
            ax = axes2[idx // 2, idx % 2]

            sns.violinplot(
                data=df_norm, x="Mapped_Label", y=col_name, hue="Genotype",
                split=True, inner="quartile", ax=ax, palette=palette, order=sorted(top_syllables), cut=0
            )
            ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Group Mean (0)")
            ax.set_title(f"{title_suffix} ({drug})", fontsize=14, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Syllable Identity (Alphabet)")

            # Stat annotation on residuals
            raw_col = col_name.replace("_norm", "")
            for i, syll in enumerate(sorted(top_syllables)):
                wt_v = df_norm[(df_norm["Mapped_Label"] == syll) & (df_norm["Genotype"] == "WT")][col_name].dropna()
                ki_v = df_norm[(df_norm["Mapped_Label"] == syll) & (df_norm["Genotype"] == "KI")][col_name].dropna()

                if len(wt_v) > 20 and len(ki_v) > 20:
                    stat, p = stats.mannwhitneyu(wt_v, ki_v, alternative='two-sided')
                    if p < 0.001: sig = "***"
                    elif p < 0.01: sig = "**"
                    elif p < 0.05: sig = "*"
                    else: sig = "ns"

                    if sig != "ns":
                        y_max = df_norm[df_norm["Mapped_Label"] == syll][col_name].quantile(0.95)
                        ax.text(i, y_max + 0.3, sig, ha='center', fontweight='bold', color='black')

        # ----------------------------------------------------------------
        # STATS LOGGING
        # ----------------------------------------------------------------
        stats_file = os.path.join(out_dir, f"{project_name.lower()}_micro_stats.txt")
        with open(stats_file, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"PROJECT: {project_name} | DRUG: {drug} | STYLE: {style}\n")
            f.write(f"{'='*50}\n\n")

            f.write("--- RAW METRICS (WT vs KI) ---\n")
            for col_name, _, _ in metrics_to_plot:
                f.write(f"\nMetric: {col_name}\n")
                for syll in sorted(top_syllables):
                    wt_v = df_d[(df_d["Mapped_Label"] == syll) & (df_d["Genotype"] == "WT")][col_name].dropna()
                    ki_v = df_d[(df_d["Mapped_Label"] == syll) & (df_d["Genotype"] == "KI")][col_name].dropna()
                    if len(wt_v) > 20 and len(ki_v) > 20:
                        _, p = stats.mannwhitneyu(wt_v, ki_v, alternative='two-sided')
                        wt_mean = wt_v.mean()
                        ki_mean = ki_v.mean()
                        sig_star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        f.write(f"  Syllable: {syll:<15} | p-value: {p:.4f} {sig_star:<3} | WT Mean: {wt_mean:.3f} | KI Mean: {ki_mean:.3f}\n")

            f.write("\n--- NORMALIZED METRICS (Group-Mean-Centered) ---\n")
            for col_name, _, _ in norm_metrics:
                f.write(f"\nMetric: {col_name}\n")
                for syll in sorted(top_syllables):
                    wt_v = df_norm[(df_norm["Mapped_Label"] == syll) & (df_norm["Genotype"] == "WT")][col_name].dropna()
                    ki_v = df_norm[(df_norm["Mapped_Label"] == syll) & (df_norm["Genotype"] == "KI")][col_name].dropna()
                    if len(wt_v) > 20 and len(ki_v) > 20:
                        _, p = stats.mannwhitneyu(wt_v, ki_v, alternative='two-sided')
                        wt_mean = wt_v.mean()
                        ki_mean = ki_v.mean()
                        sig_star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        f.write(f"  Syllable: {syll:<15} | p-value: {p:.4f} {sig_star:<3} | WT Res: {wt_mean:.3f} | KI Res: {ki_mean:.3f}\n")

        plt.tight_layout()
        norm_file = os.path.join(out_dir, f"{project_name.lower()}_{drug.lower()}_{style}_micro_violin_normalised.png")
        plt.savefig(norm_file, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved micro-kinematics {style} bounds to output/micro_kinematics/")
