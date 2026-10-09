
"""
Unified Plot Panel 1f: Syllable vs Movement Initiation Traces.
Supports both IFDN and AMPH projects.
"""
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Pathing
this_file = Path(__file__).resolve()
repo_root = this_file.parents[5] 
datta_dir = this_file.parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(datta_dir) not in sys.path:
    sys.path.insert(0, str(datta_dir))

from lib import config

def clean_axes(ax):
    sns.despine(ax=ax, trim=True)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)

def main():
    parser = argparse.ArgumentParser(description="Unified Plot Panel 1f")
    parser.add_argument("--project", type=str, required=True, choices=["ifdn", "amph"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    print(f"--- Plotting Panel 1f ({args.project.upper()}) ---")
    
    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        # Resolve from config panel dirs if possible, or fallback
        d_key = "1f_amph" if args.project == "amph" else "panel_1f_ifdn"
        # Check if config has custom mapping or use default relative to output root
        data_dir = config.OUTPUT_ROOT / d_key
        if not data_dir.exists():
            # Try alternative name
            data_dir = config.OUTPUT_ROOT / ("panel_1f" if args.project == "ifdn" else "panel_1f_amph")
            
    path_obs = data_dir / "datta_1f_OBSERVED.parquet"
    path_shuff = data_dir / "datta_1f_SHUFFLE_BANDS.parquet"
    path_stats = data_dir / "datta_1f_STATS.csv"
    
    if not path_obs.exists():
        print(f"Missing {path_obs}")
        return
        
    df_obs = pd.read_parquet(path_obs)
    df_shuff = pd.read_parquet(path_shuff)
    try:
        df_stats = pd.read_csv(path_stats)
    except:
        df_stats = pd.DataFrame()
        
    plot_dir = data_dir / f"plots{args.suffix}"
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Dimensions
    genotypes = df_obs["genotype"].unique()
    hemispheres = df_obs["hemisphere"].unique() 
    methods = df_obs["method"].unique() 
    drugs = df_obs["drug"].unique() if "drug" in df_obs.columns else ["UNK"]
    
    # IFDN requires specific regions to be attempted (DMS, VS, DLS)
    # AMPH uses auto-discovered regions
    if args.project == "ifdn":
        target_regions = ["DMS", "VS", "DLS"]
    else:
        target_regions = df_obs["brain_reg"].unique()
        
    color_map = {
        "syllables_offline": "#2ca02c", # Green
        "movement": "#9467bd" # Purple
    }
    
    for geno in genotypes:
        for drug in drugs:
            for reg in target_regions:
                for hemi in hemispheres:
                    for method in methods:
                        # Filter
                        mask_obs = (df_obs["genotype"] == geno) & \
                                   (df_obs["brain_reg"] == reg) & \
                                   (df_obs["hemisphere"] == hemi) & \
                                   (df_obs["method"] == method)
                        if drug != "UNK": mask_obs &= (df_obs["drug"] == drug)
                        
                        obs_sub = df_obs[mask_obs]
                        
                        mask_shuff = (df_shuff["genotype"] == geno) & \
                                     (df_shuff["brain_reg"] == reg) & \
                                     (df_shuff["hemisphere"] == hemi) & \
                                     (df_shuff["method"] == method)
                        if drug != "UNK": mask_shuff &= (df_shuff["drug"] == drug)
                                     
                        shuff_sub = df_shuff[mask_shuff]
                        
                        if obs_sub.empty and shuff_sub.empty:
                            continue

                        print(f"Plotting {args.project} {geno} {drug if drug != 'UNK' else ''} {reg} {hemi} - {method}...")
                            
                        fig, ax = plt.subplots(figsize=(4, 3))
                            
                        # Global Shuffle Limits
                        if not shuff_sub.empty:
                            global_lo = shuff_sub["lo"].mean()
                            global_hi = shuff_sub["hi"].mean()
                            ax.axhspan(global_lo, global_hi, color="#bdbdbd", alpha=0.3, zorder=-200, label="95% Shuffle CI", linewidth=0)

                        texts = []
                        for init_type in ["syllables_offline", "movement"]:
                            c = color_map.get(init_type, "gray")
                            short_name = "Syll" if init_type == "syllables_offline" else "Mov"
                                
                            o_dat = obs_sub[obs_sub["init_type"] == init_type].sort_values("x")
                            if not o_dat.empty:
                                y_col = "z_shuffle" if "z_shuffle" in o_dat.columns else "mean"
                                
                                if y_col == "z_shuffle" and "z_shuffle" in o_dat.columns:
                                    # Propagate SEM to Z-space: Z_sem = raw_sem / std_shuff
                                    if "std_shuff" in o_dat.columns and "sem" in o_dat.columns:
                                        z_sem = o_dat["sem"] / o_dat["std_shuff"]
                                        ax.fill_between(o_dat["x"], o_dat[y_col] - 1.96*z_sem, o_dat[y_col] + 1.96*z_sem,
                                                        color=c, alpha=0.3, zorder=90, linewidth=0)
                                    ax.plot(o_dat["x"], o_dat[y_col], color=c, label=init_type, linewidth=2, zorder=100)
                                else:
                                    ax.fill_between(o_dat["x"], o_dat["mean"] - 1.96*o_dat["sem"], o_dat["mean"] + 1.96*o_dat["sem"], 
                                                    color=c, alpha=0.3, zorder=90, lw=0)
                                    ax.plot(o_dat["x"], o_dat["mean"], color=c, label=init_type, linewidth=2, zorder=100)
                                    
                                n_val = int(o_dat[o_dat["x"] == 0]["n"].iloc[0]) if not o_dat[o_dat["x"] == 0].empty else 0
                                texts.append(f"n({short_name})={n_val}")
                            else:
                                texts.append(f"n({short_name})=0")

                        # Annotate n values
                        if texts:
                            ax.text(0.95, 0.05, "\n".join(texts), transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

                        # Stats
                        if not df_stats.empty:
                            st_mask = (df_stats["genotype"] == geno) & \
                                      (df_stats["brain_reg"] == reg) & \
                                      (df_stats["hemisphere"] == hemi) & \
                                      (df_stats["method"] == method)
                            if drug != "UNK": st_mask &= (df_stats["drug"] == drug)
                                        
                            st = df_stats[st_mask]
                            if not st.empty:
                                pval = st.iloc[0]["pvalue"]
                                txt = f"p={pval:.2e}" if pval < 0.001 else f"p={pval:.4f}"
                                ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8)

                        clean_axes(ax)
                        d_title = f" {drug}" if drug != "UNK" else ""
                        ax.set_title(f"{args.project.upper()} {geno}{d_title}\n{reg} ({hemi})\n{method}", fontsize=8)
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("DA (Shuffle STDs)") if "z_shuffle" in df_obs.columns else ax.set_ylabel("Z-score (renorm)")
                        ax.legend(fontsize=6, loc="upper right")
                        
                        drug_fn = f"_{drug}" if drug != "UNK" else ""
                        fname = f"{args.project}_1f_{geno}{drug_fn}_{reg}_{hemi}_{method}".replace(" ", "_").replace("(", "").replace(")", "")
                        fig.savefig(plot_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
                        plt.close(fig)

    print("Done.")

if __name__ == "__main__":
    main()
