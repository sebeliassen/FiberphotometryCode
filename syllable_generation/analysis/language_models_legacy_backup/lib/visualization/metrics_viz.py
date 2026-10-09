import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_perplexity_by_group(df: pd.DataFrame, save_path: str):
    """
    Boxplot comparing sequence perplexity across genotypes and drug groups.
    df columns: ["Session", "Genotype", "Drug", "Perplexity"]
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Genotype", y="Perplexity", hue="Drug", palette="Set2")
    plt.title("Sequence Grammar Perplexity by Experimental Group")
    plt.ylabel("Language Model Perplexity")
    plt.xlabel("Genotype")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved perplexity boxplot to {save_path}")

def plot_optimization_results_m1(df: pd.DataFrame, save_path: str):
    """ Plots Method 1 Bayesian search results on a semi-log scale. """
    plt.figure(figsize=(9, 6))
    
    # Sort by percentile to draw landscape line
    df_sorted = df.sort_values("percentile")
    plt.plot(df_sorted["percentile"], df_sorted["composite_score"], color="royalblue", alpha=0.5, linestyle="-")
    
    # Differentiate random vs guided points
    random_pts = df[df["type"] == "random"]
    guided_pts = df[df["type"] == "guided"]
    
    plt.scatter(random_pts["percentile"], random_pts["composite_score"], color="dimgray", marker="o", s=80, alpha=0.8, label="Initial Random Points")
    plt.scatter(guided_pts["percentile"], guided_pts["composite_score"], color="forestgreen", marker="D", s=80, alpha=0.8, label="GP Guided Points")
    
    # Highlight optimal point
    opt_idx = df["composite_score"].idxmax()
    opt_p = df.loc[opt_idx, "percentile"]
    opt_s = df.loc[opt_idx, "composite_score"]
    plt.scatter([opt_p], [opt_s], color='red', marker='*', s=250, zorder=10, label=f"Optimal (p={opt_p:.6f}, S={opt_s:.3f})")
    
    plt.xscale("log")
    plt.title("Method 1 (Slow Syllables Percentile) Bayesian Sweep (Log-Scale)")
    plt.xlabel("Percentile Cutoff (Log Scale)")
    plt.ylabel("Score")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Method 1 sweep plot to {save_path}")

def plot_optimization_results_m2(df: pd.DataFrame, save_path: str):
    """ Plots Method 2 2D Bayesian trajectory and evaluation scores on log-log axes. """
    plt.figure(figsize=(10, 8))
    
    # Draw trajectory line in sequence of iteration
    plt.plot(df["idle_vel"], df["min_duration"], color="gray", linestyle="--", alpha=0.4, zorder=1)
    
    # Scatter plot colored by composite score
    sc = plt.scatter(df["idle_vel"], df["min_duration"], c=df["composite_score"], cmap="viridis", s=120, edgecolors='black', zorder=2)
    cbar = plt.colorbar(sc)
    cbar.set_label("Composite Score S")
    
    # Label the sequence of evaluations
    for _, row in df.iterrows():
        plt.text(row["idle_vel"] * 1.05, row["min_duration"] * 1.05, str(int(row["iteration"])), fontsize=8, color="black", weight="bold")
        
    # Highlight the optimal point
    opt_idx = df["composite_score"].idxmax()
    opt_v = df.loc[opt_idx, "idle_vel"]
    opt_d = df.loc[opt_idx, "min_duration"]
    opt_s = df.loc[opt_idx, "composite_score"]
    
    plt.scatter([opt_v], [opt_d], color='red', marker='*', s=350, zorder=10, edgecolors='black', label=f"Optimal (vel={opt_v:.3f}, dur={opt_d}, S={opt_s:.3f})")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Method 2 (Kinematic Idleness) 2D Bayesian Search Trajectory")
    plt.xlabel("Idle Velocity Cutoff (px/s, Log Scale)")
    plt.ylabel("Minimum Idle Duration (Frames, Log Scale)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Method 2 2D sweep scatter plot to {save_path}")
