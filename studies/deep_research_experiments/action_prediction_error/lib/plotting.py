import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_slope_variance(res_df, out_dir, filename_suffix="", ylabel="slope_variance"):
    """
    Plot violin plots of slope variance by Genotype/Drug.
    """
    plt.figure(figsize=(10, 6))

    # Create combined group column
    if 'drug' in res_df.columns:
        res_df['Group'] = res_df['genotype'] + ' - ' + res_df['drug']
    else:
        res_df['Group'] = res_df['genotype']

    # Define pairs for statistical comparison: Only meaningful ones
    # 1. Within WT: Saline vs Amphetamine
    # 2. Within KI: Saline vs Amphetamine
    # 3. Between Genotypes: WT Saline vs KI Saline
    # 4. Between Genotypes: WT Amphetamine vs KI Amphetamine

    pairs = [
        ('WT - Saline', 'WT - Amphetamine'),
        ('KI - Saline', 'KI - Amphetamine'),
        ('WT - Saline', 'KI - Saline'),
        ('WT - Amphetamine', 'KI - Amphetamine')
    ]

    # Filter pairs to those existing in the data to avoid errors
    groups = res_df['Group'].unique()
    pairs = [p for p in pairs if p[0] in groups and p[1] in groups]

    ax = sns.violinplot(data=res_df, x='Group', y='slope_variance', inner='quartile')
    sns.stripplot(data=res_df, x='Group', y='slope_variance', color='k', alpha=0.5, size=4)

    # Add statistical annotations
    try:
        from statannotations.Annotator import Annotator
        if pairs:
            annotator = Annotator(ax, pairs, data=res_df, x='Group', y='slope_variance')
            # hide_non_significant = True is the key request
            annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2, hide_non_significant=True)
            annotator.apply_and_annotate()
    except ImportError:
        print("statannotations package not found. Skipping statistical stars.")
    except Exception as e:
        print(f"Stats annotation failed: {e}")

    plt.title(f'{ylabel} by Group')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"slope_variance_violin{filename_suffix}.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def plot_example_traces(traces, label, out_dir):
    """
    Plot raw traces to visualize variance.
    traces: (N, T) array
    """
    plt.figure(figsize=(8, 4))
    time = np.arange(traces.shape[1]) # frames

    # Plot individual traces
    for i in range(min(len(traces), 50)):
        plt.plot(time, traces[i], color='gray', alpha=0.3, linewidth=1)

    # Plot mean
    mu = np.mean(traces, axis=0)
    plt.plot(time, mu, color='red', linewidth=2, label='Mean')

    plt.title(f'Example DLS Traces: {label}')
    plt.xlabel('Frames post-onset')
    plt.ylabel('dF/F (z-score)')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"trace_example_{label}.png")
    plt.savefig(save_path)
    plt.close()
