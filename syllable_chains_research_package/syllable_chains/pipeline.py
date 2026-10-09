#!/usr/bin/env python3
"""
syllable_chains/pipeline.py
=============================================================================
Unified Pipeline Runner for Syllable Chain Dynamics Analysis (Project B).
Executes Inter-Chain Organization, Intra-Chain Execution Fidelity,
Context-Dependent Kinematics, and Generates Space-Efficient Figures & Reports.

Cohort-Agnostic:
- Works directly on Amph KPMS HDF5 / CSV
- Works directly on Karen Tandem/Multiview Session Pickles
- Works on custom arbitrary behavioral cohorts
=============================================================================
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .core import ChainDefinition, ChainFinder, ChainOccurrence, CohortDataAdapter, auto_discover_chains
from .inter_chain import InterChainAnalyzer
from .intra_chain import IntraChainAnalyzer
from .visualizer import ChainVisualizer

# -----------------------------------------------------------------------------
# Canonical Chain Definitions for Built-In Cohorts
# -----------------------------------------------------------------------------
AMPH_CANONICAL_CHAINS = [
    ChainDefinition(
        chain_id="M1_Rear_Descent",
        name="Rear -> Descent",
        tokens=[14, 4, 3],
        category="Vertical Rearing",
        description="Rearing up -> sniffing/extending -> descending back down"
    ),
    ChainDefinition(
        chain_id="M5_Slalom_Sprint",
        name="Slalom Sprint",
        tokens=[6, 17, 2],
        category="Locomotion",
        description="High-speed progression -> darting -> slalom sprint"
    ),
    ChainDefinition(
        chain_id="M8_Turn_Wall_Turn",
        name="Turn -> Wall -> Turn",
        tokens=[11, 1, 30],
        category="Wall Exploration",
        description="Right turn -> running along wall -> right turn"
    ),
    ChainDefinition(
        chain_id="M10_Pause_Scan",
        name="Sniff -> Look -> Pause",
        tokens=[20, 34, 49],
        category="Investigative / Scan",
        description="Sniff Right (20) -> Look Left Subtle (34) -> Stationary Pause (49)"
    ),
    ChainDefinition(
        chain_id="M6_Dash_Redirect",
        name="Dash & Redirect",
        tokens=[21, 2, 0],
        category="Locomotion",
        description="Rapid forward dart -> steer -> settling orient"
    ),
    ChainDefinition(
        chain_id="M7_Wall_Exploration",
        name="Wall Exploration",
        tokens=[16, 14, 7],
        category="Exploration",
        description="Wall sniff -> unsupported rear -> explore perimeter"
    )
]

KAREN_21KP_CANONICAL_CHAINS = [
    ChainDefinition(
        chain_id="K1_Rear_Settle",
        name="Rear Dumpster -> Rear -> Rear Dumpster",
        tokens=[29, 1, 29],
        category="Postural/Rear",
        description="Elevated rear posture (S29/S01) settle loop"
    ),
    ChainDefinition(
        chain_id="K2_Rear_Pivot",
        name="Rear Dumpster -> S30 -> Rear Dumpster",
        tokens=[29, 30, 29],
        category="Postural/Pivot",
        description="Rear posture -> postural transition (S30) -> rear"
    ),
    ChainDefinition(
        chain_id="K3_Vigilance_Pause",
        name="Sniff -> Still Observe -> Sniff",
        tokens=[15, 25, 15],
        category="Investigation/Vigilance",
        description="Observant sniff (S15) -> motionless arrest/observe (S25) -> resume sniff (S15)"
    ),
    ChainDefinition(
        chain_id="K4_Look_Scrunch_Look",
        name="Look Left -> Scrunch -> Look Left",
        tokens=[56, 58, 56],
        category="Postural/Orienting",
        description="Glance left with slight sniff (S56) -> bodily scrunch (S58) -> glance left (S56)"
    ),
    ChainDefinition(
        chain_id="K5_Sniff_Step_Sniff",
        name="Sniff -> Look Ahead Step -> Sniff",
        tokens=[15, 53, 15],
        category="Investigation/Stepping",
        description="Observant sniff (S15) -> forward glance step (S53) -> observant sniff (S15)"
    ),
    ChainDefinition(
        chain_id="K6_Dash_Steer_Dash",
        name="Dash -> Turn Left -> Dash",
        tokens=[42, 71, 42],
        category="Locomotion/Steering",
        description="Active locomotor dash (S42) -> leftward pivot (S71) -> continued dash (S42)"
    )
]


def run_pipeline(
    data_type: str,
    h5_path: Optional[str] = None,
    meta_csv: Optional[str] = None,
    pkl_path: Optional[str] = None,
    out_dir: Union[str, Path] = "chain_analysis_results",
    cohort_name: str = "Cohort",
    syllable_col: str = "syllable",
    gap_threshold_s: float = 1.0,
    chain_defs: Optional[List[ChainDefinition]] = None,
    auto_discover: bool = False,
    auto_discover_n: int = 6
) -> Dict[str, Any]:
    """
    Executes the comprehensive Project B Syllable Chain analysis workflow.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_dir = out_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"🚀 INITIATING PROJECT B SYLLABLE CHAIN ANALYSIS: {cohort_name.upper()}")
    print("=" * 78)

    # 1. Load Data
    if data_type.lower() == "amph":
        if h5_path is None:
            h5_path = "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/results.h5"
        if meta_csv is None:
            meta_csv = "syllable_generation/analysis/sequence_viz/data/session_metadata_ground_truth.csv"
        session_dict = CohortDataAdapter.load_from_kpms_h5(h5_path, meta_csv=meta_csv, fps_default=40.0)
        default_chains = AMPH_CANONICAL_CHAINS
    elif data_type.lower() == "karen":
        if pkl_path is None:
            pkl_path = "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl"
        session_dict = CohortDataAdapter.load_from_pickle(pkl_path, fps_default=60.0, syllable_col=syllable_col)
        default_chains = KAREN_21KP_CANONICAL_CHAINS
    elif data_type.lower() == "pickle":
        if pkl_path is None:
            raise ValueError("pkl_path must be specified for pickle data_type.")
        session_dict = CohortDataAdapter.load_from_pickle(pkl_path, fps_default=60.0, syllable_col=syllable_col)
        default_chains = None
    elif data_type.lower() == "kpms_h5":
        if h5_path is None:
            raise ValueError("h5_path must be specified for kpms_h5 data_type.")
        session_dict = CohortDataAdapter.load_from_kpms_h5(h5_path, meta_csv=meta_csv, fps_default=40.0)
        default_chains = None
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    if auto_discover:
        print(f"[Pipeline] Auto-discovering top {auto_discover_n} recurring 3-syllable chains from data...")
        chain_defs = auto_discover_chains(session_dict, top_n=auto_discover_n, chain_length=3)
    elif chain_defs is None:
        if default_chains is None:
            raise ValueError("chain_defs must be provided or use --auto-discover.")
        chain_defs = default_chains

    print(f"Loaded {len(session_dict)} sessions. Target chains: {[c.chain_id for c in chain_defs]}")

    # 2. Extract Chain Occurrences
    finder = ChainFinder(chain_defs)
    all_occurrences: List[ChainOccurrence] = []

    for sid, sdata in session_dict.items():
        occs = finder.extract_chain_occurrences(
            session_id=sid,
            syllables=sdata["syllables"],
            velocity=sdata["velocity"],
            centroid=sdata["centroid"],
            animal_id=sdata["animal_id"],
            genotype=sdata["genotype"],
            condition=sdata["condition"],
            fps=sdata["fps"]
        )
        all_occurrences.extend(occs)

    print(f"Extracted {len(all_occurrences)} total chain occurrences across {len(session_dict)} sessions.")

    # Convert occurrences to summary table
    occ_summary = []
    for o in all_occurrences:
        occ_summary.append({
            "occurrence_id": o.occurrence_id,
            "session_id": o.session_id,
            "animal_id": o.animal_id,
            "genotype": o.genotype,
            "condition": o.condition,
            "chain_id": o.chain_id,
            "start_frame": o.start_frame,
            "end_frame": o.end_frame,
            "duration_s": o.duration_s,
            "is_complete": o.is_complete,
            "abort_stage": o.abort_stage,
            "edit_distance": o.edit_distance,
            "n_insertions": len(o.insertions),
            "n_repetitions": len(o.repetitions),
            "mean_speed": o.mean_speed,
            "peak_speed": o.peak_speed,
            "is_lead": o.is_lead,
            "bout_cycle_index": o.bout_cycle_index,
            "prev_chain_id": o.prev_chain_id,
            "prev_chain_gap_s": o.prev_chain_gap_s
        })
    df_occs = pd.DataFrame(occ_summary)
    df_occs.to_csv(csv_dir / "chain_occurrences_master.csv", index=False)

    # 3. Inter-Chain Organization Analysis
    print("\n--- Part 1: Inter-Chain Organization Analysis ---")
    inter_analyzer = InterChainAnalyzer(chain_defs, all_occurrences, continuation_gap_threshold_s=gap_threshold_s)

    repeat_df = inter_analyzer.compute_repeat_statistics()
    repeat_df.to_csv(csv_dir / "inter_chain_repeats.csv", index=False)

    hazard_df = inter_analyzer.compute_continuation_hazard()
    hazard_df.to_csv(csv_dir / "inter_chain_continuation_hazard.csv", index=False)

    trans_matrices = inter_analyzer.compute_transition_matrices()
    for g, mat in trans_matrices.items():
        pd.DataFrame(mat, index=[c.chain_id for c in chain_defs], columns=[c.chain_id for c in chain_defs]).to_csv(
            csv_dir / f"inter_chain_transition_matrix_{g}.csv"
        )

    syntax_entropy_df = inter_analyzer.compute_syntax_entropy()
    syntax_entropy_df.to_csv(csv_dir / "inter_chain_syntax_entropy.csv", index=False)

    lead_vs_rep_df = inter_analyzer.compare_lead_vs_repeat_kinematics()
    lead_vs_rep_df.to_csv(csv_dir / "inter_chain_lead_vs_repeat.csv", index=False)

    # 4. Intra-Chain Execution Analysis
    print("\n--- Part 2: Intra-Chain Execution & Kinematics Analysis ---")
    intra_analyzer = IntraChainAnalyzer(chain_defs, all_occurrences, session_dict=session_dict)

    consistency_df = intra_analyzer.compute_consistency()
    consistency_df.to_csv(csv_dir / "intra_chain_consistency.csv", index=False)

    completion_df, abort_hazard_df = intra_analyzer.compute_completion_and_aborts()
    completion_df.to_csv(csv_dir / "intra_chain_completion.csv", index=False)
    abort_hazard_df.to_csv(csv_dir / "intra_chain_stage_hazard.csv", index=False)

    variation_df = intra_analyzer.compute_variations()
    variation_df.to_csv(csv_dir / "intra_chain_variations.csv", index=False)

    pacing_df = intra_analyzer.compute_pacing_profiles()
    pacing_df.to_csv(csv_dir / "intra_chain_pacing.csv", index=False)

    context_df = intra_analyzer.compute_context_dependent_movement()
    context_df.to_csv(csv_dir / "context_dependent_movement.csv", index=False)

    micro_df = intra_analyzer.compute_intra_syllable_micro_kinematics()
    if not micro_df.empty:
        micro_df.to_csv(csv_dir / "intra_syllable_micro_kinematics.csv", index=False)

    # 5. Visualizations (Unified Multi-Panel Plots)
    print("\n--- Part 3: Generating Visualizations ---")
    visualizer = ChainVisualizer(out_path)
    chain_ids = [c.chain_id for c in chain_defs]

    p_inter = visualizer.plot_inter_chain_summary(
        repeat_df=repeat_df,
        hazard_df=hazard_df,
        trans_matrices=trans_matrices,
        chain_ids=chain_ids,
        lead_vs_rep_df=lead_vs_rep_df,
        title_prefix=cohort_name
    )
    print(f"Saved: {p_inter}")

    p_intra = visualizer.plot_intra_chain_summary(
        consistency_df=consistency_df,
        completion_df=completion_df,
        hazard_df=abort_hazard_df,
        variation_df=variation_df,
        pacing_df=pacing_df,
        title_prefix=cohort_name
    )
    print(f"Saved: {p_intra}")

    p_context = visualizer.plot_context_dependent_movement(
        context_df=context_df,
        title_prefix=cohort_name
    )
    print(f"Saved: {p_context}")

    if not micro_df.empty:
        p_micro = visualizer.plot_intra_syllable_micro_kinematics(
            micro_df=micro_df,
            title_prefix=cohort_name
        )
        if p_micro:
            print(f"Saved: {p_micro}")

    # 6. Generate Markdown Report
    report_path = out_path / "CHAIN_ANALYSIS_REPORT.md"
    _generate_markdown_report(
        report_path=report_path,
        cohort_name=cohort_name,
        chain_defs=chain_defs,
        n_sessions=len(session_dict),
        n_occurrences=len(all_occurrences),
        repeat_df=repeat_df,
        hazard_df=hazard_df,
        syntax_entropy_df=syntax_entropy_df,
        lead_vs_rep_df=lead_vs_rep_df,
        consistency_df=consistency_df,
        completion_df=completion_df,
        variation_df=variation_df,
        pacing_df=pacing_df,
        context_df=context_df
    )
    print(f"Report written to: {report_path}")
    print("=" * 78)
    print(f"✅ ANALYSIS COMPLETE FOR {cohort_name.upper()}")
    print("=" * 78)

    return {
        "out_dir": str(out_path),
        "n_occurrences": len(all_occurrences),
        "figures": [str(p_inter), str(p_intra), str(p_context)],
        "report": str(report_path)
    }


def _generate_markdown_report(
    report_path: Path,
    cohort_name: str,
    chain_defs: List[ChainDefinition],
    n_sessions: int,
    n_occurrences: int,
    repeat_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
    syntax_entropy_df: pd.DataFrame,
    lead_vs_rep_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    variation_df: pd.DataFrame,
    pacing_df: pd.DataFrame,
    context_df: pd.DataFrame
):
    """Writes a publication-ready markdown report summarizing all chain analysis metrics."""
    with open(report_path, "w") as f:
        f.write(f"# Project B: Syllable Chain Dynamics Report — {cohort_name}\n\n")
        f.write(f"**Cohort:** {cohort_name}  \n")
        f.write(f"**Sessions Analyzed:** {n_sessions}  \n")
        f.write(f"**Total Chain Executions Detected:** {n_occurrences:,}  \n\n")

        f.write("## 1. Canonical Behavioral Chains Analyzed\n\n")
        f.write("| Chain ID | Name | Category | Syllables | Description |\n")
        f.write("|---|---|---|---|---|\n")
        for cd in chain_defs:
            f.write(f"| `{cd.chain_id}` | {cd.name} | {cd.category} | `{cd.to_string()}` | {cd.description} |\n")
        f.write("\n---\n\n")

        f.write("## 2. Inter-Chain Organization Summary\n\n")
        f.write("![Inter-Chain Organization](inter_chain_organization_summary.png)\n\n")

        f.write("### 2.1 Perseveration Index & Chain Repeat Rates\n")
        if not repeat_df.empty:
            grp_rep = repeat_df.groupby("genotype")["perseveration_index"].agg(["count", "mean", "std"]).reset_index()
            f.write(grp_rep.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 2.2 Continuation Hazard Rates Across Successive Cycles\n")
        if not hazard_df.empty:
            f.write(hazard_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 2.3 Syntax Transition Entropy Rate\n")
        if not syntax_entropy_df.empty:
            grp_ent = syntax_entropy_df.groupby("genotype")["chain_transition_entropy"].agg(["count", "mean", "std"]).reset_index()
            f.write(grp_ent.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 2.4 Initiations: Lead vs. Repetition Kinematics\n")
        if not lead_vs_rep_df.empty:
            f.write(lead_vs_rep_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("---\n\n")
        f.write("## 3. Intra-Chain Execution Fidelity & Dynamics\n\n")
        f.write("![Intra-Chain Execution](intra_chain_execution_summary.png)\n\n")

        f.write("### 3.1 Execution Consistency (DTW Distance & Profile Correlation)\n")
        if not consistency_df.empty:
            f.write(consistency_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 3.2 Completion Rates\n")
        if not completion_df.empty:
            f.write(completion_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 3.3 Structural Variations (Insertions, Deletions, Stutters)\n")
        if not variation_df.empty:
            f.write(variation_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("### 3.4 Temporal Stage Pacing Profiles\n")
        if not pacing_df.empty:
            f.write(pacing_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("---\n\n")
        f.write("## 4. Context-Dependent Movement\n\n")
        f.write("![Context Dependent Movement](context_dependent_syllable_kinematics.png)\n\n")
        if not context_df.empty:
            f.write(context_df.to_markdown(index=False))
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Run Project B Syllable Chain Analysis.")
    parser.add_argument("--cohort", type=str, default="amph", choices=["amph", "karen", "pickle", "kpms_h5"],
                        help="Cohort type: amph, karen, pickle, kpms_h5")
    parser.add_argument("--h5-path", type=str, default=None, help="Path to KPMS results.h5")
    parser.add_argument("--meta-csv", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--pkl-path", type=str, default=None, help="Path to session pickle")
    parser.add_argument("--out-dir", type=str, default="syllable_chain_results", help="Output directory")
    parser.add_argument("--cohort-name", type=str, default=None, help="Cohort display name")
    parser.add_argument("--syllable-col", type=str, default="syllable", help="Syllable column name in DataFrame")
    parser.add_argument("--gap-threshold", type=float, default=1.0, help="Continuation gap threshold in seconds")
    parser.add_argument("--auto-discover", action="store_true", help="Auto-discover top recurring 3-syllable chains from data")
    parser.add_argument("--auto-discover-n", type=int, default=6, help="Number of recurring chains to auto-discover")

    args = parser.parse_args()

    c_name = args.cohort_name if args.cohort_name else args.cohort.capitalize()

    run_pipeline(
        data_type=args.cohort,
        h5_path=args.h5_path,
        meta_csv=args.meta_csv,
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
        cohort_name=c_name,
        syllable_col=args.syllable_col,
        gap_threshold_s=args.gap_threshold,
        auto_discover=args.auto_discover,
        auto_discover_n=args.auto_discover_n
    )


if __name__ == "__main__":
    main()
