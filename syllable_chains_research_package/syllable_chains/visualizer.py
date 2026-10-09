#!/usr/bin/env python3
"""
syllable_chains/visualizer.py
=============================================================================
Compact, Publication-Grade Multi-Panel Visualizations for Syllable Chain Dynamics.
Space-efficient: consolidates metrics into unified multi-panel summaries
rather than generating sprawling directories of individual plots.
=============================================================================
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class ChainVisualizer:
    """
    Generates unified, high-density visualization figures for inter-chain organization
    and intra-chain execution metrics.
    """

    def __init__(self, out_dir: Path, style: str = "whitegrid"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style=style, font_scale=1.0)
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # 1. INTER-CHAIN ORGANISATION MASTER FIGURE
    # -------------------------------------------------------------------------
    def plot_inter_chain_summary(
        self,
        repeat_df: pd.DataFrame,
        hazard_df: pd.DataFrame,
        trans_matrices: Dict[str, np.ndarray],
        chain_ids: List[str],
        lead_vs_rep_df: pd.DataFrame,
        title_prefix: str = "Cohort"
    ) -> Path:
        """
        Creates a 4-panel master figure of inter-chain organization:
        Panel A: Perseveration Index & Repeat Rates
        Panel B: Continuation Hazard Curves
        Panel C: Inter-Chain Transition Matrix / Differences
        Panel D: Lead vs. Repetition Contrast
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title_prefix}: Inter-Chain Organization & Sequence Dynamics", fontsize=16, fontweight="bold", y=0.98)

        # Panel A: Perseveration Index across groups
        ax_a = axes[0, 0]
        if not repeat_df.empty and "perseveration_index" in repeat_df.columns:
            sns.boxplot(
                data=repeat_df,
                x="genotype",
                y="perseveration_index",
                hue="condition" if "condition" in repeat_df.columns and repeat_df["condition"].nunique() > 1 else None,
                ax=ax_a,
                palette="Set2"
            )
            sns.stripplot(
                data=repeat_df,
                x="genotype",
                y="perseveration_index",
                color="black",
                alpha=0.6,
                jitter=0.2,
                ax=ax_a
            )
            ax_a.set_title("A. Perseveration Index (Self-Loop Probability)", fontweight="bold", fontsize=12)
            ax_a.set_ylabel("P(Chain Repeat)")
            ax_a.set_xlabel("Genotype / Cohort")
        else:
            ax_a.text(0.5, 0.5, "Insufficient Repeat Data", ha="center", va="center")

        # Panel B: Continuation Hazard
        ax_b = axes[0, 1]
        if not hazard_df.empty:
            haz_melt = pd.melt(
                hazard_df,
                id_vars=["genotype", "chain_id"],
                value_vars=[c for c in ["p_continue_1_to_2", "p_continue_2_to_3", "p_continue_3_to_4"] if c in hazard_df.columns],
                var_name="transition",
                value_name="hazard"
            )
            haz_melt["transition"] = haz_melt["transition"].str.replace("p_continue_", "").str.replace("_to_", " -> ")
            sns.barplot(
                data=haz_melt,
                x="transition",
                y="hazard",
                hue="genotype",
                ax=ax_b,
                palette="tab10"
            )
            ax_b.set_title("B. Continuation Probability across Successive Cycles", fontweight="bold", fontsize=12)
            ax_b.set_ylabel("P(Cycle k+1 | Cycle k)")
            ax_b.set_xlabel("Cycle Transition")
            ax_b.set_ylim(0, 1.0)
        else:
            ax_b.text(0.5, 0.5, "No Multi-Cycle Bouts Found", ha="center", va="center")

        # Panel C: Transition Matrix (first group or difference)
        ax_c = axes[1, 0]
        groups = list(trans_matrices.keys())
        if len(groups) >= 2:
            g1, g2 = groups[0], groups[1]
            diff_mat = trans_matrices[g2] - trans_matrices[g1]
            vmax = max(0.15, np.max(np.abs(diff_mat)))
            sns.heatmap(
                diff_mat,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                xticklabels=chain_ids,
                yticklabels=chain_ids,
                ax=ax_c,
                cbar_kws={"label": f"Δ P ({g2} - {g1})"}
            )
            ax_c.set_title(f"C. Inter-Chain Transition Shift: {g2} minus {g1}", fontweight="bold", fontsize=12)
            ax_c.set_xlabel("Destination Chain")
            ax_c.set_ylabel("Source Chain")
        elif len(groups) == 1:
            g1 = groups[0]
            sns.heatmap(
                trans_matrices[g1],
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=chain_ids,
                yticklabels=chain_ids,
                ax=ax_c,
                cbar_kws={"label": "Transition Probability"}
            )
            ax_c.set_title(f"C. Inter-Chain Transition Matrix: {g1}", fontweight="bold", fontsize=12)
            ax_c.set_xlabel("Destination Chain")
            ax_c.set_ylabel("Source Chain")
        else:
            ax_c.text(0.5, 0.5, "No Transition Data", ha="center", va="center")

        # Panel D: Lead vs Repetition Contrast
        ax_d = axes[1, 1]
        if not lead_vs_rep_df.empty:
            lead_melt = pd.melt(
                lead_vs_rep_df,
                id_vars=["chain_id"],
                value_vars=[c for c in ["lead_mean_speed", "repeat_mean_speed"] if c in lead_vs_rep_df.columns],
                var_name="execution_type",
                value_name="mean_speed"
            )
            lead_melt["execution_type"] = lead_melt["execution_type"].map({
                "lead_mean_speed": "Lead (Cycle 1)",
                "repeat_mean_speed": "Repeat (Cycle >= 2)"
            })
            sns.barplot(
                data=lead_melt,
                x="chain_id",
                y="mean_speed",
                hue="execution_type",
                ax=ax_d,
                palette="coolwarm"
            )
            ax_d.set_title("D. Kinematic Contrast: Lead vs. Repetition Speed", fontweight="bold", fontsize=12)
            ax_d.set_ylabel("Mean Speed (pix/frame or cm/s)")
            ax_d.set_xlabel("Chain ID")
        else:
            ax_d.text(0.5, 0.5, "Insufficient Multi-Bout Data", ha="center", va="center")

        plt.tight_layout()
        out_path = self.out_dir / "inter_chain_organization_summary.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # -------------------------------------------------------------------------
    # 2. INTRA-CHAIN EXECUTION MASTER FIGURE
    # -------------------------------------------------------------------------
    def plot_intra_chain_summary(
        self,
        consistency_df: pd.DataFrame,
        completion_df: pd.DataFrame,
        hazard_df: pd.DataFrame,
        variation_df: pd.DataFrame,
        pacing_df: pd.DataFrame,
        title_prefix: str = "Cohort"
    ) -> Path:
        """
        Creates a 4-panel master figure of intra-chain execution fidelity:
        Panel A: Biomechanical Consistency (DTW Distance & Profile Correlation)
        Panel B: Completion Rate & Early Abort Hazard
        Panel C: Structural Variations (Insertions, Deletions, Stutters)
        Panel D: Stage-by-Stage Pacing & Relative Durations
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title_prefix}: Intra-Chain Execution Fidelity & Kinematic Dynamics", fontsize=16, fontweight="bold", y=0.98)

        # Panel A: Consistency (DTW Distance)
        ax_a = axes[0, 0]
        if not consistency_df.empty and "mean_pairwise_dtw_dist" in consistency_df.columns:
            sns.barplot(
                data=consistency_df,
                x="chain_id",
                y="mean_pairwise_dtw_dist",
                hue="genotype",
                ax=ax_a,
                palette="mako"
            )
            ax_a.set_title("A. Execution Consistency (DTW Distance, Lower = More Stereotyped)", fontweight="bold", fontsize=12)
            ax_a.set_ylabel("Mean Dynamic Time Warping Dist")
            ax_a.set_xlabel("Chain ID")
        else:
            ax_a.text(0.5, 0.5, "Insufficient Complete Executions", ha="center", va="center")

        # Panel B: Completion Rate
        ax_b = axes[0, 1]
        if not completion_df.empty:
            sns.barplot(
                data=completion_df,
                x="chain_id",
                y="completion_rate_pct",
                hue="genotype",
                ax=ax_b,
                palette="viridis"
            )
            ax_b.set_title("B. Conditional Progression Rate (% Reaching Suffix)", fontweight="bold", fontsize=12)
            ax_b.set_ylabel("Progression Rate (%)")
            ax_b.set_xlabel("Chain ID")
            ax_b.set_ylim(0, 100)
        else:
            ax_b.text(0.5, 0.5, "No Attempts Found", ha="center", va="center")

        # Panel C: Structural Variations
        ax_c = axes[1, 0]
        if not variation_df.empty:
            var_melt = pd.melt(
                variation_df,
                id_vars=["chain_id", "genotype"],
                value_vars=[c for c in ["mean_insertions", "mean_deletions", "mean_repetitions_stutter"] if c in variation_df.columns],
                var_name="variation_type",
                value_name="mean_count"
            )
            var_melt["variation_type"] = var_melt["variation_type"].str.replace("mean_", "").str.capitalize()
            sns.barplot(
                data=var_melt,
                x="chain_id",
                y="mean_count",
                hue="variation_type",
                ax=ax_c,
                palette="flare"
            )
            ax_c.set_title("C. Structural Variations (Insertions, Deletions, Stutters)", fontweight="bold", fontsize=12)
            ax_c.set_ylabel("Mean Count per Attempt")
            ax_c.set_xlabel("Chain ID")
        else:
            ax_c.text(0.5, 0.5, "No Variation Data", ha="center", va="center")

        # Panel D: Stage Pacing
        ax_d = axes[1, 1]
        if not pacing_df.empty:
            pacing_df_plot = pacing_df.copy()
            pacing_df_plot["stage_label"] = pacing_df_plot["chain_id"] + " S" + pacing_df_plot["stage_index"].astype(str)
            sns.barplot(
                data=pacing_df_plot,
                x="stage_label",
                y="stage_relative_fraction",
                hue="genotype",
                ax=ax_d,
                palette="rocket"
            )
            ax_d.set_title("D. Temporal Pacing (Stage Duration Fraction of Total Chain)", fontweight="bold", fontsize=12)
            ax_d.set_ylabel("Fraction of Chain Duration")
            plt.setp(ax_d.get_xticklabels(), rotation=45, ha="right")
        else:
            ax_d.text(0.5, 0.5, "No Pacing Data", ha="center", va="center")

        plt.tight_layout()
        out_path = self.out_dir / "intra_chain_execution_summary.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # -------------------------------------------------------------------------
    # 3. CONTEXT-DEPENDENT SYLLABLE MOVEMENT FIGURE
    # -------------------------------------------------------------------------
    def plot_context_dependent_movement(
        self,
        context_df: pd.DataFrame,
        title_prefix: str = "Cohort"
    ) -> Path:
        """
        Plots kinematic modulation of identical syllables executed inside chains vs baseline.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{title_prefix}: Context-Dependent Syllable Kinematics (Inside Chain vs. Baseline)", fontsize=14, fontweight="bold")

        ax1 = axes[0]
        if not context_df.empty and "speed_modulation_ratio" in context_df.columns:
            sns.barplot(
                data=context_df,
                x="syllable_token",
                y="speed_modulation_ratio",
                hue="genotype",
                ax=ax1,
                palette="mako"
            )
            ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
            ax1.set_title("A. Speed Modulation Ratio (In-Chain / Outside)", fontweight="bold", fontsize=11)
            ax1.set_ylabel("Speed Ratio (In-Chain / Outside)")
            ax1.set_xlabel("Syllable Token")
        else:
            ax1.text(0.5, 0.5, "No Context Data", ha="center", va="center")

        ax2 = axes[1]
        if not context_df.empty and "duration_modulation_ratio" in context_df.columns:
            sns.barplot(
                data=context_df,
                x="syllable_token",
                y="duration_modulation_ratio",
                hue="genotype",
                ax=ax2,
                palette="viridis"
            )
            ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
            ax2.set_title("B. Duration Modulation Ratio (In-Chain / Outside)", fontweight="bold", fontsize=11)
            ax2.set_ylabel("Duration Ratio (In-Chain / Outside)")
            ax2.set_xlabel("Syllable Token")
        else:
            ax2.text(0.5, 0.5, "No Context Data", ha="center", va="center")

        ax3 = axes[2]
        if not context_df.empty and "in_chain_mean_speed" in context_df.columns:
            sns.barplot(
                data=context_df,
                x="syllable_token",
                y="in_chain_mean_speed",
                hue="genotype",
                ax=ax3,
                palette="magma"
            )
            ax3.set_title("C. Absolute In-Chain Speed", fontweight="bold", fontsize=11)
            ax3.set_ylabel("Mean Speed")
            ax3.set_xlabel("Syllable Token")
        else:
            ax3.text(0.5, 0.5, "No Context Data", ha="center", va="center")

        plt.tight_layout()
        out_path = self.out_dir / "context_dependent_syllable_kinematics.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_intra_syllable_micro_kinematics(
        self,
        micro_df: pd.DataFrame,
        title_prefix: str = "Cohort"
    ) -> Optional[Path]:
        """
        Plots intra-frame micro-kinematics across normalized time tau in [0.0, 1.0].
        """
        if micro_df.empty:
            return None
            
        unique_tokens = sorted(micro_df["syllable_token"].unique())[:4]
        fig, axes = plt.subplots(1, len(unique_tokens), figsize=(4.5 * len(unique_tokens), 4), sharey=True)
        if len(unique_tokens) == 1:
            axes = [axes]
        fig.suptitle(f"{title_prefix}: Intra-Frame Micro-Kinematics (Velocity Across Normalized Syllable Phase)", fontsize=13, fontweight="bold")
        
        for ax, tok in zip(axes, unique_tokens):
            sub = micro_df[micro_df["syllable_token"] == tok]
            sns.lineplot(
                data=sub,
                x="tau",
                y="speed_mean",
                hue="genotype",
                style="context",
                ax=ax,
                errorbar=None,
                linewidth=2.0
            )
            ax.set_title(f"Syllable {tok}", fontweight="bold", fontsize=11)
            ax.set_xlabel("Normalized Phase (tau)")
            ax.set_ylabel("Speed")
            ax.grid(True, linestyle=":", alpha=0.5)
            
        plt.tight_layout()
        out_path = self.out_dir / "intra_syllable_micro_kinematics.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
