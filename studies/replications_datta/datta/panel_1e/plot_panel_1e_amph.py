"""
Plot Results (AMPH): Generates Correlation Plots for AMPH Dataset (Panel 1e).
- Filter: Green fibers only.
- Groups: Genotype x Drug.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lib import config
from statsmodels.stats.multitest import multipletests

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

# --- HELPERS (Copied from plot_results.py) ---
def normalize_keys(df):
    if df is None: return None
    is_idx = isinstance(df.index, pd.MultiIndex)
    if is_idx: df = df.reset_index()

    for col in ["brain_reg", "neural_agg", "feature", "genotype", "hemisphere", "orig_letter", "drug"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            if col == "hemisphere":
                df[col] = df[col].replace({"Left": "L", "Right": "R", "left": "L", "right": "R"})
            if col == "orig_letter":
                df[col] = df[col].replace({"Green": "G", "Red": "R"})

    if "timescale" in df.columns:
        df["timescale"] = pd.to_numeric(df["timescale"], errors="coerce").fillna(0).astype(int)
    return df

def apply_color_hotfix_stats(df):
    # Same logic
    if df is None: return None
    was_multi = isinstance(df.index, pd.MultiIndex)
    if was_multi: df = df.reset_index()
    if "fiber_num" in df.columns and "orig_letter" not in df.columns:
        f_nums = pd.to_numeric(df["fiber_num"], errors="coerce")
        df["orig_letter"] = np.where(f_nums % 2 == 0, "G", "R")
    return df

def load_data():
    prefix = "DEBUG_" if config.DEBUG_MODE else ""
    # Override Output Dir
    target_dir = config.PANEL_DIRS["1e_amph"]

    # Load Observed
    obs_path = target_dir / f"{prefix}OBSERVED.pkl"
    if not obs_path.exists():
        print(f"Missing {obs_path}")
        return None, {}
    obs_df = pd.read_pickle(obs_path)
    if "r" in obs_df.columns:
        obs_df["r"] = pd.to_numeric(obs_df["r"], errors="coerce").astype("float32")

    stats = {}
    for mode in ["roll", "permute"]:
        stats_path = target_dir / f"{prefix}STATS_{mode}.pkl"
        if stats_path.exists():
            stats[mode] = pd.read_pickle(stats_path)
        else:
            print(f"Missing stats for {mode}")

    return obs_df, stats

def plot_panel(group_label, title_str, obs_g, chance_g, p_dist_g, features, output_path):
    # Same plotting logic
    valid_feats = [f for f in features if f in obs_g.columns]
    if not valid_feats: return

    final_p_long = pd.DataFrame()

    if p_dist_g is not None:
        p_vals = {}
        p_dist_flat = normalize_keys(p_dist_g.copy())
        obs_flat = normalize_keys(obs_g.reset_index())
        obs_mean = obs_flat.groupby(["timescale", "brain_reg", "neural_agg"])[valid_feats].mean().reset_index()
        shuff_mean = p_dist_flat.groupby(["timescale", "brain_reg", "neural_agg", "shuffle_index"])[valid_feats].mean().reset_index()
        merged = pd.merge(shuff_mean, obs_mean, on=["timescale", "brain_reg", "neural_agg"], suffixes=("_shuff", "_obs"))

        for ft in valid_feats:
            if f"{ft}_shuff" not in merged.columns: continue
            merged["extreme"] = merged[f"{ft}_shuff"].abs() >= merged[f"{ft}_obs"].abs()
            sub_p = merged.groupby(["timescale", "brain_reg", "neural_agg"])["extreme"].mean()
            p_vals[ft] = sub_p

        if p_vals:
            p_df = pd.DataFrame(p_vals)
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

    plot_idx = obs_g.reset_index()
    plot_idx = normalize_keys(plot_idx)
    reset_cols = ["timescale", "brain_reg", "neural_agg", "animal_id", "drug"] # Include Drug

    plot_data = plot_idx.melt(
        id_vars=[c for c in reset_cols if c in plot_idx.columns],
        value_vars=valid_feats,
        var_name="variable", value_name="value"
    )
    plot_data = plot_data[(plot_data["brain_reg"] != "UNK") & (plot_data["neural_agg"] != "UNK")]
    if plot_data.empty: return

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

    # Chance Bands
    for ax in g.axes.flat:
        if ax is None: continue
        title = ax.get_title()
        if "|" not in title: continue
        r_name, c_name = [t.strip() for t in title.split("|")]
        pretty_r = LABEL_MAP.get(r_name, r_name)
        ax.set_title(f"{pretty_r} | {c_name}")
        ax.axhline(0, color="gray", linewidth=0.8, zorder=0)

        if chance_g is not None:
             try:
                 c_flat = normalize_keys(chance_g.reset_index()) if isinstance(chance_g, (pd.DataFrame, pd.Series)) else chance_g
                 if isinstance(c_flat, pd.Series): c_flat = c_flat.reset_index()
                 if "r" in c_flat.columns: c_flat = c_flat.rename(columns={"r": "val"})
                 if 0 in c_flat.columns: c_flat = c_flat.rename(columns={0: "val"})

                 # Chance grouping implicitly included Drug?
                 # chance_g was sliced by Drug before passing here.

                 mask = (c_flat["brain_reg"] == c_name) & (c_flat["neural_agg"] == r_name) & (c_flat["feature"].isin(valid_feats))
                 panel_chance = c_flat[mask]

                 if not panel_chance.empty:
                     envelope = panel_chance.groupby("timescale")["val"].max()
                     ax.fill_between(envelope.index, -envelope.values, envelope.values,
                                     color="#cccccc", alpha=0.3, linewidth=0, zorder=1.5)
             except Exception as e:
                 print(f"Error drawing chance band: {e}")

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

    obs_df = normalize_keys(obs_df)
    obs_df = apply_color_hotfix_stats(obs_df)

    # Pivot
    idx_cols = ["genotype", "drug", "timescale", "brain_reg", "neural_agg", "animal_id", "hemisphere", "orig_letter"]
    idx_cols = [c for c in idx_cols if c in obs_df.columns]
    obs_wide = obs_df.pivot_table(index=idx_cols, columns="feature", values="r")

    plot_dir = config.PANEL_DIRS["1e_amph"] / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Iterate Groups
    genotypes = obs_wide.index.get_level_values("genotype").unique()
    drugs = obs_wide.index.get_level_values("drug").unique() if "drug" in obs_wide.index.names else ["UNK"]
    hemispheres = ["L", "R"]

    for mode in ["roll", "permute"]:
        if mode not in stats: continue
        print(f"   MODE: {mode}")

        chance_df, p_dist_df = stats[mode]
        chance_df = apply_color_hotfix_stats(normalize_keys(chance_df))
        p_dist_df = apply_color_hotfix_stats(normalize_keys(p_dist_df))

        def get_slice(df, geno, drug, hemi):
            # Assumes Green Only ("G")
            color = "G"

            mask = (df["genotype"] == geno) & (df["orig_letter"] == color) & (df["hemisphere"] == hemi)
            if "drug" in df.columns: mask &= (df["drug"] == drug)
            if "drug" in df.index.names: mask &= (df.index.get_level_values("drug") == drug)

            if isinstance(df.index, pd.MultiIndex):
                # Using boolean mask on MultiIndex is tricky, better to reset or use xs if simple.
                # xs doesn't handle custom levels easily if not sorted/named perfectly.
                # Simplest: ensure we work with flat DF for slicing or use query.
                flat = df.reset_index()
                sub = flat[
                     (flat["genotype"] == geno) &
                     (flat["orig_letter"] == color) &
                     (flat["hemisphere"] == hemi) &
                     (flat.get("drug", "UNK") == drug) # Safe check
                ]
                # Re-index to match expected format for plotting [Index -> Features]
                if sub.empty: return None
                return sub.set_index(["timescale", "brain_reg", "neural_agg", "animal_id"] if "animal_id" in sub.columns else ["timescale", "brain_reg", "neural_agg"])
            else:
                return df[mask]

        for geno in genotypes:
            for drug in drugs:
                for hemi in hemispheres:
                    # Slice logic needs to be robust for pivot table (obs_wide)
                    # Obs_wide is MultiIndex.
                    try:
                        # Slice Obs
                        # We want: Genotype=geno, Drug=drug, Color=G, Hemi=hemi.
                        # Using xs is cleanest if levels exist.
                        temp = obs_wide.xs(geno, level="genotype").xs("G", level="orig_letter").xs(hemi, level="hemisphere")
                        if "drug" in temp.index.names:
                            temp = temp.xs(drug, level="drug")
                        curr_obs = temp
                    except KeyError:
                        continue

                    if curr_obs.empty: continue

                    # Slice Stats (Chance/P)
                    # These are likely flat or different structure.
                    # Let's simple filter.
                    curr_chance = get_slice(chance_df, geno, drug, hemi)
                    curr_p = get_slice(p_dist_df, geno, drug, hemi)

                    title = f"{geno} | {drug} | G-{hemi} | {mode}"
                    fname = f"scalar_{geno}_{drug}_G_{hemi}_{mode}_amph.png"

                    cols = curr_obs.columns
                    prim = [f for f in ["velocity_px_s", "transition"] if f in cols]
                    sec = [f for f in ["angular_velocity", "acceleration", "jerk", "duration"] if f in cols]

                    if prim:
                        plot_panel(f"{geno}-{drug}", title, curr_obs, curr_chance, curr_p, prim, plot_dir / f"PRIMARY_{fname}")
                    if sec:
                        plot_panel(f"{geno}-{drug}", title, curr_obs, curr_chance, curr_p, sec, plot_dir / f"SECONDARY_{fname}")

if __name__ == "__main__":
    main()
