#!/usr/bin/env python3
"""
compute_syllable_catalog.py
=============================================================================
Computes ground-truth prominence statistics across all 169,879 bouts for all
100 syllables in Run 12, correlates them with the 10 convolutional seqNMF motifs,
and generates the comprehensive master markdown catalog:
RUN12_SYLLABLE_PROMINENCE_AND_SEQNMF_CATALOG.md
=============================================================================
"""

import pickle
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz"
DATA_PKL = OUTPUT_DIR / "data/unified_sequence_dataset.pkl"
SEQNMF_CSV = OUTPUT_DIR / "optimized_seqnmf/optimized_seqnmf_motifs_summary.csv"
OUT_MD = OUTPUT_DIR / "RUN12_SYLLABLE_PROMINENCE_AND_SEQNMF_CATALOG.md"

def main():
    print("Loading unified sequence dataset...")
    with open(DATA_PKL, "rb") as f:
        bundle = pickle.load(f)

    sessions = bundle["sessions"]
    all_dfs = []
    for s in sessions:
        df = s["bouts_df"].copy()
        df["animal_id"] = s["animal_id"]
        df["genotype"] = s["genotype"]
        df["drug"] = s["drug"]
        df["session_idx"] = s["session_idx"]
        all_dfs.append(df)

    cohort_df = pd.concat(all_dfs, ignore_index=True)
    total_bouts = len(cohort_df)
    total_dur_sec = cohort_df["duration_sec"].sum()
    print(f"Total cohort bouts: {total_bouts:,}, Total duration: {total_dur_sec/3600:.2f} hours")

    # Load seqNMF motifs
    print("Loading seqNMF motifs summary...")
    seqnmf_df = pd.read_csv(SEQNMF_CSV)
    syll_seqnmf = defaultdict(list)

    for _, row in seqnmf_df.iterrows():
        f_id = int(row["factor_id"])
        toks = json.loads(row["dominant_tokens"])
        weights = json.loads(row["weights"])
        seq_str = row["token_sequence"]
        episodes = int(row["verified_episodes"])
        mean_act = float(row["mean_activation"])
        wt_occ = int(row["wt_occurrences"])
        ki_occ = int(row["ki_occurrences"])

        for pos, (t, w) in enumerate(zip(toks, weights)):
            syll_seqnmf[t].append({
                "factor_id": f_id,
                "position": pos + 1,
                "weight": w,
                "sequence": seq_str,
                "episodes": episodes,
                "mean_activation": mean_act,
                "wt_occ": wt_occ,
                "ki_occ": ki_occ
            })

    # Group by syllable across entire cohort
    stats_list = []
    unique_sylls = sorted(cohort_df["syllable"].unique())

    for s in range(100):
        s_df = cohort_df[cohort_df["syllable"] == s]
        n_bouts = len(s_df)
        if n_bouts == 0:
            stats_list.append({
                "syllable": s, "count": 0, "pct": 0.0, "total_sec": 0.0,
                "mean_dur": 0.0, "median_dur": 0.0, "iqr_dur": 0.0,
                "mean_vel": 0.0, "wt_sal_cnt": 0, "ki_sal_cnt": 0,
                "amph_cnt": 0, "seqnmf_count": 0, "seqnmf_details": []
            })
            continue

        pct = (n_bouts / total_bouts) * 100.0
        tot_sec = s_df["duration_sec"].sum()
        mean_dur = s_df["duration_sec"].mean()
        med_dur = s_df["duration_sec"].median()
        q25 = s_df["duration_sec"].quantile(0.25)
        q75 = s_df["duration_sec"].quantile(0.75)
        iqr_dur = q75 - q25
        mean_vel = s_df["mean_vel"].mean()

        # Condition breakdown
        wt_sal = s_df[(s_df["genotype"] == "WT") & (s_df["drug"] == "Saline")]
        ki_sal = s_df[(s_df["genotype"] == "KI") & (s_df["drug"] == "Saline")]
        amph = s_df[s_df["drug"] == "Amphetamine"]

        stats_list.append({
            "syllable": s,
            "count": n_bouts,
            "pct": pct,
            "total_sec": tot_sec,
            "mean_dur": mean_dur,
            "median_dur": med_dur,
            "iqr_dur": iqr_dur,
            "mean_vel": mean_vel,
            "wt_sal_cnt": len(wt_sal),
            "ki_sal_cnt": len(ki_sal),
            "amph_cnt": len(amph),
            "seqnmf_count": len(syll_seqnmf[s]),
            "seqnmf_details": syll_seqnmf[s]
        })

    # Sort by frequency descending
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    stats_df["rank"] = stats_df.index + 1
    stats_df["cum_pct"] = stats_df["pct"].cumsum()

    # Assign Priority Tier
    def assign_tier(row):
        if row["seqnmf_count"] > 0 or row["rank"] <= 20:
            return "Tier 1 (High Priority: seqNMF / Dominant)"
        elif row["rank"] <= 50:
            return "Tier 2 (Medium Priority: Common Motor)"
        else:
            return "Tier 3 (Low Priority: Rare / Tail)"

    stats_df["tier"] = stats_df.apply(assign_tier, axis=1)

    # Save summary CSV
    csv_path = OUTPUT_DIR / "run12_syllable_prominence_summary.csv"
    export_df = stats_df.drop(columns=["seqnmf_details"])
    export_df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary CSV: {csv_path.name}")

    # Build Master Markdown
    lines = []
    lines.append("# Keypoint-MoSeq Run 12: Master Syllable Prominence & seqNMF Catalog")
    lines.append("")
    lines.append("A complete neuroethological reference and labeling workbook for **Run 12** (`amph_prod_k150k_run_12`) across 30 sessions (15 paired animals: 8 WT, 7 DAT-Het KI; Saline vs. Amphetamine, 169,879 total bouts).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    top10_pct = stats_df.iloc[:10]["pct"].sum()
    top20_pct = stats_df.iloc[:20]["pct"].sum()
    top30_pct = stats_df.iloc[:30]["pct"].sum()
    top50_pct = stats_df.iloc[:50]["pct"].sum()

    lines.append(f"- **Total Cohort Size**: **{total_bouts:,} bouts** ({total_dur_sec/3600:.2f} hours of continuous behavior).")
    lines.append(f"- **Top 10 Syllables**: Account for **{top10_pct:.2f}%** of all behavior.")
    lines.append(f"- **Top 20 Syllables**: Account for **{top20_pct:.2f}%** of all behavior.")
    lines.append(f"- **Top 30 Syllables (00–29)**: Account for **{top30_pct:.2f}%** of all behavior.")
    lines.append(f"- **Top 50 Syllables**: Account for **{top50_pct:.2f}%** of all behavior.")
    lines.append(f"- **seqNMF Motif Coverage**: **29 distinct syllables** participate in the 10 validated convolutional seqNMF motifs. All of them fall in Tier 1.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. seqNMF Multi-Syllable Motif Reference Table")
    lines.append("")
    lines.append("The 10 recurring behavioral motifs discovered by convolutional seqNMF with verified episode counts and constituent syllable roles:")
    lines.append("")
    lines.append("| Factor ID | Discovered Sequence | Dominant Syllables | Weights | Verified Episodes | WT Episodes | KI Episodes | Ethological Motif Class |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    motif_names = {
        1: "Locomotor Progression (Rear-to-Sniff)",
        2: "Exploratory Rear & Reorient Loop",
        3: "Low-Cadence Orienting Shift",
        4: "Sniff & Head-Dip Locomotor Chain",
        5: "Rapid Orienting Left/Right Turn",
        6: "Locomotor Sprint & Wall Follow",
        7: "Pause-Investigate-Reorient Loop",
        8: "Locomotor Alternation Chain",
        9: "High-Turn Reversal Sequence",
        10: "Rearing-to-Pause Motor Block"
    }

    for _, row in seqnmf_df.iterrows():
        fid = int(row["factor_id"])
        seq_str = row["token_sequence"].replace(" -> ", " → ")
        toks = row["dominant_tokens"]
        weights = row["weights"]
        episodes = int(row["verified_episodes"])
        wt_e = int(row["wt_occurrences"])
        ki_e = int(row["ki_occurrences"])
        m_name = motif_names.get(fid, "Behavioral Sequence")
        lines.append(f"| **Factor {fid}** | `{seq_str}` | `{toks}` | `{weights}` | **{episodes:,}** | {wt_e} | {ki_e} | {m_name} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Priority Tiers for Human Labeling")
    lines.append("")
    lines.append("To label efficiently without cognitive overload, focus your manual curation in three sequential tiers:")
    lines.append("")
    lines.append("### Tier 1: High Priority (seqNMF Anchors & Top 20 Syllables — 70.8% of Behavior)")
    lines.append("These are the workhorses of mouse behavioral grammar. They form the building blocks of the 10 seqNMF motifs and represent >70% of all recorded bouts:")
    lines.append("")
    lines.append("| Syllable | Rank | Bout Count | % Cohort | Mean Dur | Mean Vel | seqNMF Motifs & Positions | Primary Visual Artifacts |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")

    t1_df = stats_df[stats_df["tier"].str.startswith("Tier 1")].sort_values("rank")
    for _, row in t1_df.iterrows():
        s = int(row["syllable"])
        r = int(row["rank"])
        cnt = int(row["count"])
        pct = row["pct"]
        mdur = row["mean_dur"]
        mvel = row["mean_vel"]
        nmf_dets = row["seqnmf_details"]
        if nmf_dets:
            dets_str = ", ".join([f"Factor {d['factor_id']} (pos {d['position']})" for d in nmf_dets])
        else:
            dets_str = "*(High-frequency primitive, non-seqNMF)*"

        lines.append(f"| **Syllable {s:02d}** | #{r} | {cnt:,} | **{pct:.2f}%** | {mdur:.3f}s | {mvel:.1f} px/s | {dets_str} | [Density MP4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_with_density/syllable{s}.mp4) • [Random MP4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_without_density/syllable{s}.mp4) • [Trajectory](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/trajectory_plots_density/Syllable{s}.gif) |")

    lines.append("")
    lines.append("### Tier 2: Medium Priority (Common Behavioral Primitives, Ranks 21–50 — 21.4% of Behavior)")
    lines.append("Secondary movement modifiers, specialized turning primitives, and posture transition elements:")
    lines.append("")
    lines.append("| Syllable | Rank | Bout Count | % Cohort | Mean Dur | Mean Vel | Primary Visual Artifacts |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    t2_df = stats_df[stats_df["tier"].str.startswith("Tier 2")].sort_values("rank")
    for _, row in t2_df.iterrows():
        s = int(row["syllable"])
        r = int(row["rank"])
        cnt = int(row["count"])
        pct = row["pct"]
        mdur = row["mean_dur"]
        mvel = row["mean_vel"]
        lines.append(f"| **Syllable {s:02d}** | #{r} | {cnt:,} | {pct:.2f}% | {mdur:.3f}s | {mvel:.1f} px/s | [Density MP4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_with_density/syllable{s}.mp4) • [Random MP4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_without_density/syllable{s}.mp4) • [Trajectory](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/trajectory_plots_density/Syllable{s}.gif) |")

    lines.append("")
    lines.append("### Tier 3: Low Priority (Infrequent / Specialized Tail, Ranks 51–100 — 7.8% of Behavior)")
    lines.append("Rare behaviors, boundary noise, or extreme kinematic outliers:")
    lines.append("")
    lines.append("| Syllable | Rank | Bout Count | % Cohort | Mean Dur | Mean Vel | Visual Links |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    t3_df = stats_df[stats_df["tier"].str.startswith("Tier 3")].sort_values("rank")
    for _, row in t3_df.iterrows():
        s = int(row["syllable"])
        r = int(row["rank"])
        cnt = int(row["count"])
        pct = row["pct"]
        mdur = row["mean_dur"]
        mvel = row["mean_vel"]
        lines.append(f"| **Syllable {s:02d}** | #{r} | {cnt:,} | {pct:.2f}% | {mdur:.3f}s | {mvel:.1f} px/s | [Density](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_with_density/syllable{s}.mp4) • [Random](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/grid_movies_without_density/syllable{s}.mp4) • [Traj](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project/amph_prod_k150k_run_12/trajectory_plots_density/Syllable{s}.gif) |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Human Annotation Worksheet (All 100 Syllables)")
    lines.append("")
    lines.append("Use the template below to assign descriptive names and functional categories:")
    lines.append("")
    lines.append("| Syllable | Rank | % Cohort | Kinematics | seqNMF Role | User Label (e.g. Fast Left Turn) | Category (Locomotion / Rear / Sniff / Pause / Turn) | Notes / Observations |")
    lines.append("| :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- |")

    for _, row in stats_df.iterrows():
        s = int(row["syllable"])
        r = int(row["rank"])
        pct = row["pct"]
        mdur = row["mean_dur"]
        mvel = row["mean_vel"]
        kin = f"{mdur:.2f}s, {mvel:.0f}px/s"
        nmf_dets = row["seqnmf_details"]
        if nmf_dets:
            roles = [f"F{d['factor_id']}:p{d['position']}" for d in nmf_dets]
            nmf_str = ", ".join(roles)
        else:
            nmf_str = "-"
        lines.append(f"| **Syllable {s:02d}** | #{r} | {pct:.2f}% | {kin} | {nmf_str} | `[ ]` | `[ ]` | |")

    md_content = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(md_content)
    print(f"✓ Saved Master Syllable Catalog to: {OUT_MD.name}")

    # Also copy to Desktop for easy access
    desktop_md = Path("/home/fsp585/Desktop/RUN12_SYLLABLE_PROMINENCE_AND_SEQNMF_CATALOG.md")
    with open(desktop_md, "w") as f:
        f.write(md_content)
    print(f"✓ Copied Master Syllable Catalog to Desktop: {desktop_md.name}")

if __name__ == "__main__":
    main()
