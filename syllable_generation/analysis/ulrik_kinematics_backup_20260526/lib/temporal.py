import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from analysis.ulrik_kinematics import config
from analysis.ulrik_kinematics.lib import metrics

def get_rolling_metrics(session, window_sec, step_sec, align_to_injection=False):
    """
    Slices the session into temporal chunks and feeds them into the verified metrics engine.
    """
    df = session.all_data_df.copy()
    if "frame_index" not in df.columns: return pd.DataFrame()
    
    # 1. MASKING: Explicitly restrict analysis to only the active tracking span
    if "syllable" in df.columns and not df["syllable"].isna().all():
        valid_idx = df["syllable"].dropna().index
        start_idx = valid_idx[0]
        end_idx = valid_idx[-1]
        df = df.loc[start_idx:end_idx].copy()
        
    if df.empty: return pd.DataFrame()
    
    # 2. ALIGNMENT: Calculate normalized local time (seconds)
    if align_to_injection and session.metadata.get("injection_frame") is not None:
        t_zero_frame = session.metadata.get("injection_frame")
    else:
        t_zero_frame = df["frame_index"].iloc[0]
        
    df["time"] = (df["frame_index"] - t_zero_frame) / 30.0
    
    max_time = df["time"].max()
    min_time = df["time"].min()
    windows = []
    
    # Align the starting grid to a clean multiple of step_sec relative to t=0
    start_t = np.floor(min_time / step_sec) * step_sec
    
    while start_t < max_time:
        end_t = start_t + window_sec
        chunk = df[(df["time"] >= start_t) & (df["time"] < end_t)]
        
        # Need at least ~10 seconds of data (300 frames at 30fps) for robust metrics
        if len(chunk) > 300: 
            s_mock = copy.copy(session)
            s_mock.all_data_df = chunk
            
            # The entropy metrics calculator natively accepts robust boundaries
            res = metrics.calculate_entropy_metrics(s_mock)
            
            # Calculate physical kinematics directly for this window
            mean_vel = chunk["velocity_px_s"].mean() if "velocity_px_s" in chunk.columns else np.nan
            
            # Reconstruct durations for the window
            if "instance_id" in chunk.columns:
                durations = chunk.groupby("instance_id").apply(lambda x: x["time"].max() - x["time"].min())
                mean_dur = durations.mean() if not durations.empty else np.nan
            else:
                mean_dur = np.nan
                
            windows.append({
                "Window_Start": start_t,
                "Window_Center": start_t + (window_sec / 2),
                "Sequence_Entropy": res.get("entropy_transition", np.nan),
                "Mean_Velocity": mean_vel,
                "Mean_Duration": mean_dur
            })
            
        start_t += step_sec
        
    return pd.DataFrame(windows)

def verify_temporal_module(sessions):
    """
    Tests the temporal sliding window by ensuring it breaks mathematically
    if the data is physically corrupted (hardware desynchronization).
    """
    print("\n" + "="*50)
    print(" VERIFICATION: TEMPORAL DYNAMICS & DESYNC PERTURBATION")
    print("="*50)
    
    # Use the first session as our test subject
    s = sessions[0]
    
    print(f"\n[1/3] Computing true rolling window boundary metrics...")
    df_true = get_rolling_metrics(s, config.TEMPORAL_WINDOW_SEC, config.TEMPORAL_STEP_SEC)
    mean_rolling_dur = df_true["Mean_Duration"].mean()
    mean_rolling_vel = df_true["Mean_Velocity"].mean()
    mean_rolling_ent = df_true["Sequence_Entropy"].mean()
    
    # --- IDENTITY CHECK ---
    print("[2/3] Performing Identity Check vs Global Session Metrics...")
    global_kin = metrics.calculate_kinematics(s)
    global_ent = metrics.calculate_entropy_metrics(s)
    
    g_vel = global_kin.get("mean_velocity", np.nan)
    g_ent = global_ent.get("entropy_transition", np.nan)
    
    # Global Duration: compute directly from the full session using same logic as rolling engine
    df_full = s.all_data_df.copy()
    if "syllable" in df_full.columns and not df_full["syllable"].isna().all():
        valid_idx = df_full["syllable"].dropna().index
        df_full = df_full.loc[valid_idx[0]:valid_idx[-1]].copy()
    first_frame = df_full["frame_index"].iloc[0]
    df_full["_time"] = (df_full["frame_index"] - first_frame) / 30.0
    if "instance_id" in df_full.columns:
        g_dur = df_full.groupby("instance_id").apply(lambda x: x["_time"].max() - x["_time"].min()).mean()
    else:
        g_dur = np.nan
    
    # --- PERTURBATION RUN ---
    print("[3/3] Applying 15% random desync to simulate hardware camera failure...")
    s_pert = copy.deepcopy(s)
    
    # Inject camera failure (syllable = -1) across 15% of random frames
    mask = np.random.rand(len(s_pert.all_data_df)) < 0.15
    s_pert.all_data_df.loc[mask, "syllable"] = -1
    
    # Re-calculate syllable instances since we forcefully injected boundary breaks!
    new_syll = s_pert.all_data_df["syllable"].values
    transitions = np.concatenate(([True], new_syll[1:] != new_syll[:-1]))
    s_pert.all_data_df["instance_id"] = transitions.cumsum()
    
    df_pert = get_rolling_metrics(s_pert, config.TEMPORAL_WINDOW_SEC, config.TEMPORAL_STEP_SEC)
    mean_pert_dur = df_pert["Mean_Duration"].mean()
    
    # --- REPORTING ---
    print("\n" + "-"*30)
    print("       DIAGNOSTIC RESULTS       ")
    print("-"*30)
    
    drop_p = (1 - (mean_pert_dur / mean_rolling_dur)) * 100 if mean_rolling_dur > 0 else 0
    
    # Identity deltas
    vel_delta = abs(mean_rolling_vel - g_vel) / g_vel * 100 if g_vel > 0 else 0
    ent_delta = abs(mean_rolling_ent - g_ent) / g_ent * 100 if g_ent > 0 else 0
    dur_delta = abs(mean_rolling_dur - g_dur) / g_dur * 100 if (not np.isnan(g_dur) and g_dur > 0) else np.nan
    
    table = pd.DataFrame({
        "Metric": ["Velocity (px/s)", "Sequence Entropy (Drop expected)", "Sustained Duration"],
        "Global_Value": [f"{g_vel:.2f}", f"{g_ent:.2f}", f"{g_dur:.2f}s" if not np.isnan(g_dur) else "N/A"],
        "Rolling_Average": [f"{mean_rolling_vel:.2f}", f"{mean_rolling_ent:.2f}", f"{mean_rolling_dur:.2f}s"],
        "Identity_Match": [f"{100-vel_delta:.1f}%", "N/A (drop expected)", f"{100-dur_delta:.1f}%" if not np.isnan(dur_delta) else "N/A"]
    })
    
    print(table.to_markdown(index=False))
    
    print("\n> INVARIANT CHECKS:")
    print(f"  - Identity Constraint (Vel): Rolling avg matches global within 10% -> {'PASS' if vel_delta < 10 else 'FAIL'}")
    print(f"  - Identity Constraint (Dur): Rolling avg matches global within 15% -> {'PASS' if (not np.isnan(dur_delta) and dur_delta < 15) else 'FAIL or N/A'}")
    print(f"  - Entropy Drop (Ent):        Rolling entropy < Global entropy -> {'PASS (expected drop)' if mean_rolling_ent < g_ent else 'FAIL'}")
    print(f"  - Fragmentation Constraint: Desync test broke sustained duration by {drop_p:.1f}% -> {'PASS' if drop_p > 20 else 'FAIL'}")
    print("="*50 + "\n")
    
    # --- VISUALIZATIONS ---
    print("Generating Verification Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # 1. Identity Check: Velocity
    sns.barplot(x=["Global (30m)", "Rolling Avg (2m)"], y=[g_vel, mean_rolling_vel], ax=axes[0], palette=["#bdc3c7", "#2ecc71"])
    axes[0].set_title("Identity Check: Velocity", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Mean Velocity (px/s)")
    for i, v in enumerate([g_vel, mean_rolling_vel]):
        axes[0].text(i, v + 0.5, f"{v:.1f}", color='black', ha="center")

    # 2. Entropy Drop (expected to be lower in rolling windows)
    sns.barplot(x=["Global (30m)", "Rolling Avg (2m)"], y=[g_ent, mean_rolling_ent], ax=axes[1], palette=["#bdc3c7", "#e74c3c"])
    axes[1].set_title("Entropy Drop (Rolling < Global Expected)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Sequence Entropy")
    for i, v in enumerate([g_ent, mean_rolling_ent]):
        axes[1].text(i, v + 0.05, f"{v:.2f}", color='black', ha="center")

    # 3. Identity Check: Behavioral Duration
    g_dur_plot = g_dur if not np.isnan(g_dur) else 0.0
    sns.barplot(x=["Global (30m)", "Rolling Avg (2m)"], y=[g_dur_plot, mean_rolling_dur], ax=axes[2], palette=["#bdc3c7", "#3498db"])
    axes[2].set_title("Identity Check: Behavioral Duration", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Mean Bout Duration (s)")
    for i, v in enumerate([g_dur_plot, mean_rolling_dur]):
        axes[2].text(i, v + 0.02, f"{v:.2f}s", color='black', ha="center")

    # 4. Perturbation Crash: Hardware Desync 
    sns.barplot(x=["True Execution", "Hardware Desync"], y=[mean_rolling_dur, mean_pert_dur], ax=axes[3], palette=["#3498db", "#9b59b6"])
    axes[3].set_title("Perturbation Crash Check", fontsize=14, fontweight="bold")
    axes[3].set_ylabel("Sustained Duration (s)")
    for i, v in enumerate([mean_rolling_dur, mean_pert_dur]):
        axes[3].text(i, v + 0.05, f"{v:.2f}s", color='black', ha="center")
        
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "../output/temporal")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "temporal_verification_diagnostics.png"), dpi=150)
    plt.close()
    print("Saved temporal_verification_diagnostics.png")

def run_discovery(sessions, project_name="AMPH"):
    print(f"\n--- Running Temporal Dynamics Discovery ---")
    all_rows = []
    
    for s in sessions:
        g = s.metadata.get("genotype")
        d = s.metadata.get("drug", "N/A")
        if g in ["None", "UNK", None]: continue
        
        # Standardize naming
        if d.lower().startswith("amph"): d = "Amphetamine"
        elif d.lower().startswith("sal"): d = "Saline"
        
        df_roll = get_rolling_metrics(s, config.TEMPORAL_WINDOW_SEC, config.TEMPORAL_STEP_SEC)
        if not df_roll.empty:
            df_roll["Session"] = s.session_id
            df_roll["Genotype"] = g
            df_roll["Drug"] = d
            all_rows.append(df_roll)
            
    if not all_rows: 
        print("No valid temporal data generated.")
        return
        
    df_all = pd.concat(all_rows, ignore_index=True)
    
    # Generate unified timeline plots
    for drug in ["Saline", "Amphetamine"]:
        df_plot = df_all[df_all["Drug"] == drug]
        if df_plot.empty: continue
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Convert window center from seconds to minutes for cleaner X-axis
        df_plot["Time_Min"] = df_plot["Window_Center"] / 60.0
        
        # DYNAMIC TRUNCATION: Avoid 'dropout artifacts' at the end of the plot.
        # We calculate the 10th percentile of session durations, which ensures 
        # that the plot ends where at least 90% of the mice are still tracking.
        session_max_times = df_plot.groupby("Session")["Time_Min"].max()
        if not session_max_times.empty:
            cutoff_min = session_max_times.quantile(0.10)
            print(f"  -> {drug}: Dynamic cutoff applied at {cutoff_min:.2f} min (90% cohort coverage)")
            df_plot = df_plot[df_plot["Time_Min"] <= cutoff_min].copy()
            
        color_map = {g: ("#2ecc71" if g == "WT" else "#e74c3c") for g in df_plot["Genotype"].unique()}
        
        from scipy.stats import linregress
        def add_regression(ax, df, x_col, y_col):
            lines_text = []
            for g, color in color_map.items():
                dg = df[df["Genotype"] == g].dropna(subset=[x_col, y_col])
                if dg.empty or len(dg) < 3: continue
                
                res = linregress(dg[x_col], dg[y_col])
                x_vals = np.array([dg[x_col].min(), dg[x_col].max()])
                y_vals = res.intercept + res.slope * x_vals
                
                ax.plot(x_vals, y_vals, color=color, linestyle="--", linewidth=2, alpha=0.8)
                
                sign = "+" if res.slope >= 0 else ""
                lines_text.append(f"{g}: slope={sign}{res.slope:.4f}, r={res.rvalue:.2f}")
                
            if lines_text:
                # Place annotation in bottom left (0.02, 0.05) to avoid overlap with 'upper right' legend
                ax.text(0.02, 0.05, "\n".join(lines_text), transform=ax.transAxes, 
                        fontsize=11, fontweight='bold', verticalalignment='bottom',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))
        
        # 1. Sequence Entropy Plot
        sns.lineplot(data=df_plot, x="Time_Min", y="Sequence_Entropy", hue="Genotype", 
                     palette=color_map, err_style="band", errorbar="ci", ax=axes[0], linewidth=3)
        axes[0].set_title(f"Temporal Dynamics: Sequence Entropy ({drug})", fontsize=15, fontweight="bold")
        axes[0].set_ylabel("Entropy (bits/step)")
        axes[0].legend(loc='upper right', frameon=True, framealpha=0.85)
        add_regression(axes[0], df_plot, "Time_Min", "Sequence_Entropy")
        
        # 2. Velocity Plot
        sns.lineplot(data=df_plot, x="Time_Min", y="Mean_Velocity", hue="Genotype", 
                     palette=color_map, err_style="band", errorbar="ci", ax=axes[1], linewidth=3)
        axes[1].set_title(f"Temporal Dynamics: Velocity ({drug})", fontsize=15, fontweight="bold")
        axes[1].set_ylabel("Velocity")
        axes[1].legend(loc='upper right', frameon=True, framealpha=0.85)
        add_regression(axes[1], df_plot, "Time_Min", "Mean_Velocity")
        
        # 3. Behavioral Sustained Duration Plot
        sns.lineplot(data=df_plot, x="Time_Min", y="Mean_Duration", hue="Genotype", 
                     palette=color_map, err_style="band", errorbar="ci", ax=axes[2], linewidth=3)
        axes[2].set_title(f"Temporal Dynamics: Behavioral Duration ({drug})", fontsize=15, fontweight="bold")
        axes[2].set_ylabel("Duration (seconds)")
        axes[2].set_xlabel("Session Timeline (Minutes)", fontsize=14)
        axes[2].legend(loc='upper right', frameon=True, framealpha=0.85)
        add_regression(axes[2], df_plot, "Time_Min", "Mean_Duration")
        
        plt.tight_layout()
        out_dir = os.path.join(os.path.dirname(__file__), "../output/temporal")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{project_name.lower()}_{drug.lower()}_rolling_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Saved temporal discovery plots to output/temporal/")

def run_full_session_discovery(sessions, project_name="AMPH"):
    print(f"\n--- Running Full-Session Temporal Dynamics Discovery ---")
    
    # We generate two separate plots: one for the Amphetamine group and one for the Saline group
    for drug_target in ["Amphetamine", "Saline"]:
        print(f" > Processing Treatment Group: {drug_target}")
        all_rows = []
        
        for s in sessions:
            # 1. Normalize Drug and Genotype
            d = s.metadata.get("drug", "UNK")
            if d.lower().startswith("amph"): d = "Amphetamine"
            elif d.lower().startswith("sal"): d = "Saline"
            
            if d != drug_target: continue
            
            g = s.metadata.get("genotype")
            if g in ["None", "UNK", None]: continue
            
            # 2. Calculate rolling metrics
            df_roll = get_rolling_metrics(s, config.TEMPORAL_WINDOW_SEC, config.TEMPORAL_STEP_SEC, align_to_injection=True)
            if not df_roll.empty:
                df_roll["Session"] = s.session_id
                df_roll["Genotype"] = g
                all_rows.append(df_roll)
                
        if not all_rows: 
            print(f"No valid temporal data for {drug_target} group. Skipping.")
            continue
            
        df_all = pd.concat(all_rows, ignore_index=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        df_plot = df_all.copy()
        df_plot["Time_Min"] = df_plot["Window_Center"] / 60.0
        
        # Filter first 5 minutes of tracking per session (acclimatization)
        df_filtered = []
        for s_id, group in df_plot.groupby("Session"):
            min_time = group["Time_Min"].min()
            df_filtered.append(group[group["Time_Min"] >= min_time + 5])
        df_plot = pd.concat(df_filtered, ignore_index=True)
        
        # 3. Coordinate Shift: Remove the gap and pull the right side left
        df_plot = df_plot[df_plot["Time_Min"] <= 45].copy()
        df_plot = df_plot[(df_plot["Time_Min"] <= -5) | (df_plot["Time_Min"] >= 15)].copy()
        mask_post = df_plot["Time_Min"] >= 15
        df_plot.loc[mask_post, "Time_Min"] -= 20
                
        color_map = {g: ("#2ecc71" if g == "WT" else "#e74c3c") for g in df_plot["Genotype"].unique()}
        
        metrics_to_plot = [
            ("Sequence_Entropy", "Entropy (bits/step)", f"Temporal Dynamics: Sequence Entropy ({drug_target} Group)"),
            ("Mean_Velocity", "Velocity (px/s)", f"Temporal Dynamics: Velocity ({drug_target} Group)"),
            ("Mean_Duration", "Duration (seconds)", f"Temporal Dynamics: Behavioral Duration ({drug_target} Group)")
        ]
        
        import matplotlib.ticker as ticker
        from scipy.stats import linregress
        
        def format_broken_axis(x, pos):
            if x <= -5:
                return f"{int(x)}"
            else:
                return f"{int(x + 20)}"
                
        for i, (col, ylabel, title) in enumerate(metrics_to_plot):
            ax = axes[i]
            sns.lineplot(data=df_plot, x="Time_Min", y=col, hue="Genotype", 
                         palette=color_map, err_style="band", errorbar="ci", ax=ax, linewidth=3)
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.set_ylabel(ylabel)
            
            # Regression Logic
            for genotype in ["WT", "KI"]:
                if genotype not in df_plot["Genotype"].values: continue
                c = color_map[genotype]
                y_pos_low = 0.12 if genotype == "WT" else 0.04
                
                # Saline Phase (Left side)
                mask_sal = (df_plot["Genotype"] == genotype) & (df_plot["Time_Min"] <= -5)
                df_sal = df_plot.loc[mask_sal, ["Time_Min", col]].dropna()
                
                if len(df_sal) > 2:
                    res_sal = linregress(df_sal["Time_Min"], df_sal[col])
                    x_line_sal = np.array([df_sal["Time_Min"].min(), df_sal["Time_Min"].max()])
                    y_line_sal = res_sal.intercept + res_sal.slope * x_line_sal
                    ax.plot(x_line_sal, y_line_sal, color=c, linestyle="--", linewidth=2.5, alpha=0.8)
                    
                    p_text_sal = f"p<0.001" if res_sal.pvalue < 0.001 else f"p={res_sal.pvalue:.3f}"
                    text_sal = f"{genotype} Saline: m={res_sal.slope:.3f}, R={res_sal.rvalue:.2f}, {p_text_sal}"
                    ax.text(0.02, y_pos_low, text_sal, transform=ax.transAxes, color='black', fontsize=10, fontweight="bold",
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                            
                # Amph Phase (Right side)
                mask_amph = (df_plot["Genotype"] == genotype) & (df_plot["Time_Min"] > -5)
                df_amph = df_plot.loc[mask_amph, ["Time_Min", col]].dropna()
                
                if len(df_amph) > 2:
                    res_amph = linregress(df_amph["Time_Min"], df_amph[col])
                    x_line_amph = np.array([df_amph["Time_Min"].min(), df_amph["Time_Min"].max()])
                    y_line_amph = res_amph.intercept + res_amph.slope * x_line_amph
                    ax.plot(x_line_amph, y_line_amph, color=c, linestyle="--", linewidth=2.5, alpha=0.8)
                    
                    p_text_amph = f"p<0.001" if res_amph.pvalue < 0.001 else f"p={res_amph.pvalue:.3f}"
                    text_amph = f"{genotype} Amph: m={res_amph.slope:.3f}, R={res_amph.rvalue:.2f}, {p_text_amph}"
                    ax.text(0.98, y_pos_low, text_amph, transform=ax.transAxes, color='black', 
                            ha='right', fontsize=10, fontweight="bold",
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_broken_axis))
            ax.axvline(-5, color='k', linestyle=':', linewidth=3.0, alpha=0.8, zorder=0)
            
            handles, labels = ax.get_legend_handles_labels()
            import matplotlib.lines as mlines
            inj_line = mlines.Line2D([], [], color='k', linestyle=':', linewidth=3.0, alpha=0.8, label='Injection Gap')
            
            filtered_handles, filtered_labels = [], []
            for h, l in zip(handles, labels):
                if l not in filtered_labels and not l.startswith("_"):
                    filtered_handles.append(h)
                    filtered_labels.append(l)
                    
            ax.legend(handles=filtered_handles + [inj_line], labels=filtered_labels + ['Injection Gap'], 
                      loc='upper right', frameon=True, framealpha=0.9)
            
        axes[-1].set_xlabel("Time Relative to Injection (Minutes)", fontsize=14)
        plt.tight_layout()
        
        out_dir = os.path.join(os.path.dirname(__file__), "../output/temporal")
        os.makedirs(out_dir, exist_ok=True)
        # Reconsidered Naming: Include the drug in the filename
        out_path = os.path.join(out_dir, f"{project_name.lower()}_full_session_temporal_{drug_target.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved temporal plot to {out_path}")
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "../output/temporal")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{project_name.lower()}_full_session_rolling_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved full session temporal plots to {out_path}")
