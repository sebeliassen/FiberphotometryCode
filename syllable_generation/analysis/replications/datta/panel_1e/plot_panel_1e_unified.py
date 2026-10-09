"""
Plot Results: Generates final correlation plots with statistical markers.
- Multi-panel visualizations (Primary: Velocity/Transition; Secondary: Others)
- Gray Chance Bands (95% Shuffled Quantile)
- Holm-corrected significance markers (horizontal bars)
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from lib import config

# --- CONFIG ---
sns.set(style="whitegrid", context="talk", font_scale=0.8)
PLOT_HEIGHT = 3.0
PLOT_ASPECT = 1.1

LABEL_MAP = {
    "velocity_px_s": "vel.", "angular_velocity": "ang. vel.", "acceleration": "acc.", 
    "jerk": "jerk", "transition": "trans.", "duration": "dur."
}
FIXED_PALETTE = {
    "velocity_px_s": "C0", "transition": "C1", "angular_velocity": "C2", 
    "acceleration": "C3", "jerk": "C4", "duration": "C5"
}

# --- HELPERS ---
def normalize_keys(df):
    """Normalize categorical columns to match legacy naming conventions."""
    if df is None: return None
    
    # Check Index vs Columns
    is_idx = isinstance(df.index, pd.MultiIndex)
    if is_idx:
        df = df.reset_index()
        
    for col in ["brain_reg", "neural_agg", "feature", "genotype", "hemisphere", "orig_letter"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Map Full Names to Abbreviations
            if col == "hemisphere":
                df[col] = df[col].replace({"Left": "L", "Right": "R", "left": "L", "right": "R"})
            if col == "orig_letter":
                df[col] = df[col].replace({"Green": "G", "Red": "R"})
                
    if "timescale" in df.columns:
        df["timescale"] = pd.to_numeric(df["timescale"], errors="coerce").fillna(0).astype(int)
        
    if is_idx:
        # Restore index? Usually we process in long form now.
        pass
    return df

def apply_color_hotfix_stats(df):
    """Replicates legacy logic to infer Color (G/R) from Fiber Num if missing."""
    if df is None: return None
    
    # Flatten if MultiIndex
    was_multi = isinstance(df.index, pd.MultiIndex)
    if was_multi:
        df = df.reset_index()
        
    if "fiber_num" in df.columns and "orig_letter" not in df.columns:
        f_nums = pd.to_numeric(df["fiber_num"], errors="coerce")
        df["orig_letter"] = np.where(f_nums % 2 == 0, "G", "R")
        
    if was_multi:
        # Re-set index if needed? Legacy does:
        # df.index = pd.MultiIndex.from_frame(idx_df)
        pass
        
    return df

def load_data():
    prefix = "DEBUG_" if config.DEBUG_MODE else ""
    
    # Load Observed
    obs_path = config.OUTPUT_DIR / f"{prefix}OBSERVED.pkl"
    if not obs_path.exists():
        print(f"Missing {obs_path}")
        return None, {}
    obs_df = pd.read_pickle(obs_path)
    if "r" in obs_df.columns:
        obs_df["r"] = pd.to_numeric(obs_df["r"], errors="coerce").astype("float32")
    
    # Load Stats
    stats = {}
    for mode in ["roll", "permute"]:
        stats_path = config.OUTPUT_DIR / f"{prefix}STATS_{mode}.pkl"
        if stats_path.exists():
            # digest_results saves (chance_df, p_dist_df)
            stats[mode] = pd.read_pickle(stats_path)
        else:
            print(f"Missing stats for {mode}")
            
    return obs_df, stats

# Pathing
this_file = Path(__file__).resolve()
repo_root = this_file.parents[5] 
datta_dir = this_file.parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(datta_dir) not in sys.path:
    sys.path.insert(0, str(datta_dir))

def plot_panel(geno, title_str, obs_g, chance_g, p_dist_g, features, output_path):
    # if output_path.exists(): return

    valid_feats = [f for f in features if f in obs_g.columns]
    if not valid_feats: return

    # 1. P-Values calculation (Holm Corrected)
    final_p_long = pd.DataFrame()
    
    # Note: p_dist_g here is likely a DataFrame where columns are features?
    # In digest_results: p_dist_df = pop_means.unstack("feature")
    # So p_dist_g is MultiIndex [timescale, brain_reg, neural_agg, shuffle_index] -> Cols [Features]
    
    if p_dist_g is not None:
        p_vals = {}
        # Ensure p_dist_g is normalized
        p_dist_flat = normalize_keys(p_dist_g.copy()) # Flattened
        
        # Prepare Obs Mean for comparison
        # obs_g is Wide (Index=meta, Cols=Features)
        obs_flat = normalize_keys(obs_g.reset_index())
        obs_mean = obs_flat.groupby(["timescale", "brain_reg", "neural_agg"])[valid_feats].mean().reset_index()
        
        # Prepare Shuffle Mean (per shuffle index)
        # p_dist_flat has 'shuffle_index'
        shuff_mean = p_dist_flat.groupby(["timescale", "brain_reg", "neural_agg", "shuffle_index"])[valid_feats].mean().reset_index()
        
        # Merge
        merged = pd.merge(shuff_mean, obs_mean, on=["timescale", "brain_reg", "neural_agg"], suffixes=("_shuff", "_obs"))
        
        for ft in valid_feats:
            if f"{ft}_shuff" not in merged.columns: continue
            
            # Check Extreme
            merged["extreme"] = merged[f"{ft}_shuff"].abs() >= merged[f"{ft}_obs"].abs()
            
            # P = Mean(Extreme)
            sub_p = merged.groupby(["timescale", "brain_reg", "neural_agg"])["extreme"].mean()
            p_vals[ft] = sub_p
            
        if p_vals:
            p_df = pd.DataFrame(p_vals)
            # Holm Correction per Brain Region
            corrected = []
            if "brain_reg" in p_df.index.names:
                for reg, sub in p_df.groupby(level="brain_reg"):
                    if str(reg) == "UNK": continue
                    flat_p = sub.stack().values
                    if len(flat_p) == 0: continue
                    _, adj_p, _, _ = multipletests(flat_p, method="holm")
                    corrected.append(pd.Series(adj_p, index=sub.stack().index))
            
            if corrected:
                final_p_long = pd.concat(corrected).reset_index()
                final_p_long.columns = ["timescale", "brain_reg", "neural_agg", "variable", "p"]
                final_p_long = normalize_keys(final_p_long)

    # 2. Plotting Data
    # obs_g is [Index -> Features]
    plot_idx = obs_g.reset_index()
    plot_idx = normalize_keys(plot_idx)
    
    reset_cols = ["timescale", "brain_reg", "neural_agg", "animal_id"]
    
    plot_data = plot_idx.melt(
        id_vars=[c for c in reset_cols if c in plot_idx.columns],
        value_vars=valid_feats,
        var_name="variable", value_name="value"
    )
    
    plot_data = plot_data[(plot_data["brain_reg"] != "UNK") & (plot_data["neural_agg"] != "UNK")]
    if plot_data.empty: return

    # 3. Draw
    g = sns.relplot(
        data=plot_data, x="timescale", y="value", hue="variable",
        row="neural_agg", col="brain_reg", kind="line",
        errorbar=("ci", 68),
        height=PLOT_HEIGHT, aspect=PLOT_ASPECT, palette=FIXED_PALETTE, legend="auto"
    )
    
    if g.legend:
        g.legend.set_title("")
        for t in g.legend.get_texts():
            if t.get_text() in LABEL_MAP: t.set_text(LABEL_MAP[t.get_text()])

    g.set_axis_labels("Timescale (s)", "Correlation (r)")
    g.set(xlim=(0, 60))
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # 4. Decorations (Chance Band + P-Vals)
    for ax in g.axes.flat:
        if ax is None: continue
        title = ax.get_title()
        if "|" not in title: continue
        r_name, c_name = [t.strip() for t in title.split("|")]
        
        pretty_r = LABEL_MAP.get(r_name, r_name)
        ax.set_title(f"{pretty_r} | {c_name}")
        ax.axhline(0, color="gray", linewidth=0.8, zorder=0)

        # Draw Chance Band
        if chance_g is not None:
             try:
                 # Flatten for easier filtering
                 c_flat = normalize_keys(chance_g.reset_index()) if isinstance(chance_g, (pd.DataFrame, pd.Series)) else chance_g
                 if isinstance(c_flat, pd.Series): c_flat = c_flat.reset_index()
                 
                 if "r" in c_flat.columns: c_flat = c_flat.rename(columns={"r": "val"})
                 if 0 in c_flat.columns: c_flat = c_flat.rename(columns={0: "val"})
                 
                 mask = (c_flat["brain_reg"] == c_name) & (c_flat["neural_agg"] == r_name) & (c_flat["feature"].isin(valid_feats))
                 panel_chance = c_flat[mask]
                 
                 if not panel_chance.empty:
                     envelope = panel_chance.groupby("timescale")["val"].max()
                     ax.fill_between(envelope.index, -envelope.values, envelope.values,
                                     color="#cccccc", alpha=0.3, linewidth=0, zorder=1.5)
                 else:
                     # print(f"  [WARN] No band data for {r_name}|{c_name}")
                     pass
             except Exception as e:
                 print(f"Error drawing chance band: {e}")
                 pass

        # Draw P-Values
        if not final_p_long.empty:
            p_sub = final_p_long[(final_p_long["brain_reg"] == c_name) & (final_p_long["neural_agg"] == r_name)]
            if not p_sub.empty:
                y_lims = ax.get_ylim()
                base_y = y_lims[1] - 0.05 * (y_lims[1]-y_lims[0])
                for i, ft in enumerate(valid_feats):
                    sig_ts = p_sub[(p_sub["variable"] == ft) & (p_sub["p"] < 0.05)]["timescale"]
                    if not sig_ts.empty:
                        ts = sig_ts.values
                        ax.hlines(y=[base_y - (i*0.06*(y_lims[1]-y_lims[0]))]*len(ts),
                                  xmin=ts-0.52, xmax=ts+0.52, colors=FIXED_PALETTE.get(ft, "k"),
                                  linewidth=3, zorder=10)

    g.fig.suptitle(title_str, y=1.02)
    g.fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"      [SAVED] {output_path.name}")
    plt.close(g.fig)

def main():
    obs_df, stats = load_data()
    if obs_df is None: return

    # Pre-process Observed
    # Normalize Keys
    obs_df = normalize_keys(obs_df)
    obs_df = apply_color_hotfix_stats(obs_df)
    
    # Pivot to Wide [Index -> Features] for Slicing
    idx_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "animal_id", "hemisphere", "orig_letter"]
    idx_cols = [c for c in idx_cols if c in obs_df.columns]
    
    obs_wide = obs_df.pivot_table(index=idx_cols, columns="feature", values="r")
    
    # Configuration Space
    # We want: Genotype -> Color -> Hemisphere -> Mode
    genotypes = obs_wide.index.get_level_values("genotype").unique()
    colors = ["G", "R"]
    hemispheres = ["L", "R"]
    
    plot_dir = config.OUTPUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    for mode in ["roll", "permute"]:
        if mode not in stats: continue
        print(f"   MODE: {mode}")
        
        # Stats are (chance_df, p_dist_df)
        chance_df, p_dist_df = stats[mode]
        # Normalize Stats
        # Chance
        chance_df = normalize_keys(chance_df) # Likely Series->DF
        # P Dist
        # p_dist_df is [Index... -> Features (Cols)]
        p_dist_df = normalize_keys(p_dist_df)
        
        chance_df = apply_color_hotfix_stats(chance_df)
        p_dist_df = apply_color_hotfix_stats(p_dist_df)
        
        # Helper to slice DF by Meta
        def get_slice(df, geno, col, hemi):
            if df is None: return None
            # Assumes columns exist.
            # If MultiIndex, reset already done by normalize_keys (mostly)
            # But pivot_table outputs MultiIndex.
            
            # For Obs_Wide (MultiIndex)
            if isinstance(df.index, pd.MultiIndex):
                try:
                    return df.xs(geno, level="genotype").xs(col, level="orig_letter").xs(hemi, level="hemisphere")
                except KeyError: return None
            
            # For Flat DF (Chance, P_Dist)
            mask = (df["genotype"] == geno) & (df["orig_letter"] == col) & (df["hemisphere"] == hemi)
            sub = df[mask]
            if sub.empty: return None
            return sub

        for geno in genotypes:
            if str(geno) == "UNK": continue
            for color in colors:
                for hemi in hemispheres:
                     
                    curr_obs = get_slice(obs_wide, geno, color, hemi)
                    if curr_obs is None or curr_obs.empty: continue
                    
                    curr_chance = get_slice(chance_df, geno, color, hemi)
                    curr_p = get_slice(p_dist_df, geno, color, hemi)
                    
                    title = f"{geno} | {color}-{hemi} | {mode} | datta"
                    fname = f"scalar_{geno}_{color}_{hemi}_{mode}_datta.png"
                    
                    cols = curr_obs.columns
                    prim = [f for f in ["velocity_px_s", "transition"] if f in cols]
                    sec = [f for f in ["angular_velocity", "acceleration", "jerk", "duration"] if f in cols]
                    
                    if prim:
                        plot_panel(geno, title, curr_obs, curr_chance, curr_p, prim, plot_dir / f"PRIMARY_{fname}")
                    if sec:
                        plot_panel(geno, title, curr_obs, curr_chance, curr_p, sec, plot_dir / f"SECONDARY_{fname}")

if __name__ == "__main__":
    main()
