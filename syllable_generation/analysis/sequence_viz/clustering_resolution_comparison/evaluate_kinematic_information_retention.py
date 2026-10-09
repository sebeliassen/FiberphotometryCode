#!/usr/bin/env python3
"""
evaluate_kinematic_information_retention.py

Rigorous Leave-One-Animal-Out Cross-Validation (LOAO-CV) measuring retained
kinematic information across 4 clustering resolutions (Levels 1, 2, 3, 4).

Addresses collaborator prompt:
1. Fix boundaries to original KPMS bouts.
2. For each resolution, predict heading change (deg), speed (px/s), duration (s),
   and displacement (px) from the assigned categorical label on 14 training mice.
3. Test on held-out 15th mouse; repeat across all 15 mice.
4. Report errors in interpretable units, averaged equally across animals, with
   uncertainty and paired differences between levels.
5. Identify which specific merges account for information loss.
6. Export full per-episode CSV for the matched 3-step benchmark.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import load_chain_movie_resources
from syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_level_definitions,
    get_verified_metadata,
)

EXP_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison"
AUDIT_DIR = EXP_DIR / "diagnostic_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def extract_all_physical_bouts(results):
    """
    Extracts every single original syllable bout across all 30 recordings
    with exact frame boundaries and true continuous kinematics.
    """
    print("[EXTRACTION] Extracting physical bouts and kinematics across 30 sessions...", flush=True)
    bouts = []

    for rk, data in results.items():
        if "centroid" not in data or "heading" not in data or "syllable" not in data:
            continue
        cents = data["centroid"]
        heads = data["heading"]
        sylls = data["syllable"]
        n_frames = min(len(cents), len(heads), len(sylls))

        cents = cents[:n_frames]
        heads = heads[:n_frames]
        sylls = sylls[:n_frames]

        aid, geno, drug = get_verified_metadata(rk)

        # Identify run-length collapsed syllable bouts
        mask = np.concatenate([[True], sylls[1:] != sylls[:-1]])
        starts = np.where(mask)[0].tolist()
        ends = starts[1:] + [n_frames]

        step_d = np.linalg.norm(np.diff(cents, axis=0), axis=1) # frame-by-frame step dist
        dh = np.diff(heads)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi # unwrapped angular step in radians

        for sf, ef in zip(starts, ends):
            dur_f = ef - sf
            if dur_f < 2:
                continue # exclude 1-frame transient glitches

            sub_step_d = step_d[sf:min(ef - 1, len(step_d))]
            sub_dh = dh[sf:min(ef - 1, len(dh))]

            # 1. Heading change in degrees (accumulated signed heading travel)
            turn_deg = float(np.degrees(np.sum(sub_dh))) if len(sub_dh) > 0 else 0.0

            # 2. Mean centroid speed (px/s at 40 fps)
            speed_px_s = float(np.mean(sub_step_d) * 40.0) if len(sub_step_d) > 0 else 0.0

            # 3. Bout duration (seconds)
            dur_sec = float(dur_f / 40.0)

            # 4. Net displacement (px)
            net_disp = float(np.linalg.norm(cents[ef - 1] - cents[sf]))

            # 5. KPMS syllable ID
            sid = int(sylls[sf])

            bouts.append({
                "recording_key": rk,
                "animal_id": aid,
                "genotype": geno,
                "drug": drug,
                "start_frame": sf,
                "end_frame": ef,
                "duration_frames": dur_f,
                "syllable_id": sid,
                "turn_deg": turn_deg,
                "speed_px_s": speed_px_s,
                "dur_sec": dur_sec,
                "net_disp_px": net_disp
            })

    df = pd.DataFrame(bouts)
    print(f"[EXTRACTION] Successfully extracted {len(df):,} physical bouts across {df['animal_id'].nunique()} distinct animals.", flush=True)
    return df


def run_loao_cross_validation(df_bouts, levels):
    """
    Leave-One-Animal-Out Cross-Validation (LOAO-CV).
    For each held-out animal:
      - Train on other 14 animals: compute per-label median predictor.
      - Predict on held-out animal.
      - Evaluate MAE, MedAE, and RMSE for heading change, speed, duration, net displacement.
    """
    animals = sorted(df_bouts["animal_id"].unique())
    n_animals = len(animals)
    print(f"\n[LOAO-CV] Running 15-Fold Leave-One-Animal-Out Cross-Validation across {n_animals} mice...", flush=True)

    # Pre-map syllable IDs to level state labels
    for lvl in levels:
        lid = lvl["level_id"]
        m_arr = lvl["map_arr"]
        df_bouts[f"label_lvl_{lid}"] = m_arr[df_bouts["syllable_id"].values]

    features = [
        ("turn_deg", "Heading Change (deg)", "deg"),
        ("speed_px_s", "Centroid Speed (px/s)", "px/s"),
        ("dur_sec", "Bout Duration (s)", "s"),
        ("net_disp_px", "Net Displacement (px)", "px")
    ]

    # Storage for per-animal error records
    animal_results = []

    for a_idx, held_out_aid in enumerate(animals):
        train_mask = df_bouts["animal_id"] != held_out_aid
        test_mask = df_bouts["animal_id"] == held_out_aid

        df_train = df_bouts[train_mask]
        df_test = df_bouts[test_mask]

        # Evaluate overall baseline (unconditional median across all training data)
        base_record = {
            "held_out_animal": held_out_aid,
            "n_test_bouts": len(df_test)
        }

        for feat, feat_name, unit in features:
            base_pred = df_train[feat].median()
            base_mae = float(np.mean(np.abs(df_test[feat].values - base_pred)))
            base_record[f"base_mae_{feat}"] = base_mae

        for lvl in levels:
            lid = lvl["level_id"]
            lvl_name = lvl["name"]
            lbl_col = f"label_lvl_{lid}"

            # Compute training per-label median
            grouped_medians = df_train.groupby(lbl_col)[ [f[0] for f in features] ].median()

            for feat, feat_name, unit in features:
                mapper = grouped_medians[feat].to_dict()
                global_med = df_train[feat].median()

                # Predict test labels
                y_true = df_test[feat].values
                y_pred = df_test[lbl_col].map(mapper).fillna(global_med).values

                mae = float(np.mean(np.abs(y_true - y_pred)))
                medae = float(np.median(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

                # R^2 relative to held-out animal variance
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = float(1.0 - (ss_res / max(1e-8, ss_tot)))

                base_record[f"lvl_{lid}_mae_{feat}"] = mae
                base_record[f"lvl_{lid}_medae_{feat}"] = medae
                base_record[f"lvl_{lid}_rmse_{feat}"] = rmse
                base_record[f"lvl_{lid}_r2_{feat}"] = r2

        animal_results.append(base_record)

    df_res = pd.DataFrame(animal_results)
    return df_res


def compute_paired_uncertainty_and_summary(df_res, levels):
    """
    Computes equally animal-weighted error means, standard errors, and
    paired differences between adjacent levels with 95% Confidence Intervals.
    """
    print("\n" + "=" * 80)
    print("LOAO-CV PREDICTION ERROR SUMMARY (EQUAL ANIMAL WEIGHTING, N=15 MICE)")
    print("=" * 80)

    features = [
        ("turn_deg", "Heading Change", "deg"),
        ("speed_px_s", "Centroid Speed", "px/s"),
        ("dur_sec", "Bout Duration", "s"),
        ("net_disp_px", "Net Displacement", "px")
    ]

    summary_rows = []

    for feat, feat_name, unit in features:
        print(f"\n--- {feat_name} ({unit}) ---")
        base_maes = df_res[f"base_mae_{feat}"].values
        print(f"  Unconditional Population Baseline: {np.mean(base_maes):.2f} ± {stats.sem(base_maes):.2f} {unit}")

        lvl_maes = {}
        for lvl in levels:
            lid = lvl["level_id"]
            vals = df_res[f"lvl_{lid}_mae_{feat}"].values
            r2_vals = df_res[f"lvl_{lid}_r2_{feat}"].values
            lvl_maes[lid] = vals
            mean_val = np.mean(vals)
            sem_val = stats.sem(vals)
            mean_r2 = np.mean(r2_vals) * 100.0

            print(f"  {lvl['name']} ({lvl['vocab_size']} states): MAE = {mean_val:.2f} ± {sem_val:.2f} {unit} | R² = {mean_r2:.1f}%")

            summary_rows.append({
                "Feature": feat_name,
                "Unit": unit,
                "Level": lvl["name"],
                "Level_ID": lid,
                "Vocab_Size": lvl["vocab_size"],
                "MAE_Mean": mean_val,
                "MAE_SEM": sem_val,
                "MAE_95CI_Low": mean_val - 1.96 * sem_val,
                "MAE_95CI_High": mean_val + 1.96 * sem_val,
                "R2_Percent_Mean": mean_r2
            })

        # Paired differences: Level 4 vs Level 3, Level 3 vs Level 2, Level 2 vs Level 1
        print("\n  [Paired Differences across 15 Mice]:")
        pairs = [
            (3, 4, "Level 4 vs Level 3 (Cost of Pooling to 14 States)"),
            (2, 3, "Level 3 vs Level 2 (Cost of Pooling to 22 States)"),
            (1, 2, "Level 2 vs Level 1 (Cost of Synonym Merges)")
        ]
        for l_fine, l_coarse, desc in pairs:
            diffs = lvl_maes[l_coarse] - lvl_maes[l_fine] # positive means coarse has HIGHER error
            d_mean = np.mean(diffs)
            d_sem = stats.sem(diffs)
            t_stat, p_val = stats.ttest_rel(lvl_maes[l_coarse], lvl_maes[l_fine])
            ci_low = d_mean - 2.145 * d_sem # t_{0.025, df=14} = 2.145
            ci_high = d_mean + 2.145 * d_sem
            pct_chg = (d_mean / np.mean(lvl_maes[l_fine])) * 100.0

            sig_star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"    {desc}:")
            print(f"      ΔMAE = +{d_mean:.3f} {unit} (+{pct_chg:.1f}%), 95% CI [{ci_low:.3f}, {ci_high:.3f}], p = {p_val:.4e} ({sig_star})")

    df_summary = pd.DataFrame(summary_rows)
    return df_summary


def breakdown_information_loss_by_verb(df_bouts, levels):
    """
    Analyzes specific merges to pinpoint which action classes lose information
    between Level 3 and Level 4.
    """
    print("\n" + "=" * 80)
    print("ACTION-CLASS BREAKDOWN: SPECIFIC KINEMATIC MERGES AUDIT")
    print("=" * 80)

    # Define Level 3 functional classes
    action_classes = {
        "Angular Turning": [29, 30, 6, 2],           # Walk_Sharp_L/R, Turn_L/R
        "Vertical Reorientation": [4, 7, 37, 12, 14, 31, 16], # Descents & Rears
        "Locomotor Propulsion": [0, 13, 17, 21, 26, 33, 41, 1, 3, 11, 19, 23, 38, 55], # Sprint & Walk
        "Sensory Head-Casting": [5, 22, 28, 34, 40, 42, 15], # Looks & Sniff_Up
        "Stationary / Quiescent": [8, 9, 10, 18, 20, 24, 25, 32, 43, 49, 27, 52] # Sniff, Rest, Groom
    }

    class_records = []
    for c_name, sids in action_classes.items():
        sub = df_bouts[df_bouts["syllable_id"].isin(sids)]
        if len(sub) == 0: continue

        for feat, feat_name, unit in [("turn_deg", "Heading Change", "deg"),
                                      ("speed_px_s", "Speed", "px/s"),
                                      ("dur_sec", "Duration", "s")]:

            # Measure within-cluster standard deviation (residual dispersion)
            std_l1 = np.sqrt(np.mean(sub.groupby("label_lvl_1")[feat].var().dropna()))
            std_l2 = np.sqrt(np.mean(sub.groupby("label_lvl_2")[feat].var().dropna()))
            std_l3 = np.sqrt(np.mean(sub.groupby("label_lvl_3")[feat].var().dropna()))
            std_l4 = np.sqrt(np.mean(sub.groupby("label_lvl_4")[feat].var().dropna()))

            class_records.append({
                "Action_Class": c_name,
                "Feature": feat_name,
                "Unit": unit,
                "N_Bouts": len(sub),
                "Within_State_SD_L1": std_l1,
                "Within_State_SD_L2": std_l2,
                "Within_State_SD_L3": std_l3,
                "Within_State_SD_L4": std_l4,
                "SD_Inflation_L3_to_L4_Pct": ((std_l4 - std_l3) / std_l3) * 100.0,
                "SD_Inflation_L2_to_L3_Pct": ((std_l3 - std_l2) / std_l2) * 100.0
            })

    df_class = pd.DataFrame(class_records)
    print(df_class[["Action_Class", "Feature", "Within_State_SD_L3", "Within_State_SD_L4", "SD_Inflation_L3_to_L4_Pct"]].to_string(index=False))
    return df_class


def export_matched_action_chains_episode_csv(results, levels):
    """
    Exports the comprehensive per-episode CSV for the matched 3-step benchmark:
    1. Rear -> Descent -> Walk
    2. Sprint -> Turn -> Sprint
    3. Sniff -> Rest -> Sniff
    """
    print("\n" + "=" * 80)
    print("EXPORTING FULL PER-EPISODE CSV FOR MATCHED 3-STEP BENCHMARK")
    print("=" * 80)

    matched_themes = [
        {
            "theme_id": "matched_rear_descent_walk",
            "theme_title": "Rear -> Descent -> Walk",
            "level_queries": {
                1: [[12, 14, 16, 31], [4, 7, 37], [1, 3, 11, 19, 23, 38]],
                2: [[12, 14, 16, 31], [4, 7, 37], [1, 3, 19]], # synonyms mapped
                3: [[12, 13], [4, 5, 14], [0, 1, 2, 8, 9, 10]], # L3 indices
                4: [[6], [7], [0, 1, 2]] # L4 indices
            }
        },
        {
            "theme_id": "matched_sprint_turn_sprint",
            "theme_title": "Sprint -> Turn -> Sprint",
            "level_queries": {
                1: [[0, 13, 17, 21, 26, 33, 41], [2, 6, 29, 30], [0, 13, 17, 21, 26, 33, 41]],
                2: [[0, 13, 17, 21, 26, 33, 41], [2, 6, 29, 30], [0, 13, 17, 21, 26, 33, 41]],
                3: [[11], [2, 3, 0, 1], [11]],
                4: [[3], [4, 5], [3]]
            }
        },
        {
            "theme_id": "matched_sniff_rest_sniff",
            "theme_title": "Sniff -> Rest -> Sniff",
            "level_queries": {
                1: [[8, 9, 10, 15, 18, 20, 24, 25, 36, 39], [32, 43, 49], [8, 9, 10, 15, 18, 20, 24, 25, 36, 39]],
                2: [[8, 10, 15, 18, 20, 24, 25, 36, 39], [32], [8, 10, 15, 18, 20, 24, 25, 36, 39]],
                3: [[15, 16, 17], [19], [15, 16, 17]],
                4: [[8], [11], [8]]
            }
        }
    ]

    all_matched_episodes = []

    for tm in matched_themes:
        th_id = tm["theme_id"]
        th_title = tm["theme_title"]

        # Collect candidate feature vectors across all levels to build pooled reference
        theme_cands_by_level = {1: [], 2: [], 3: [], 4: []}

        for lvl in levels:
            lid = lvl["level_id"]
            lvl_name = lvl["name"]
            m_arr = lvl["map_arr"]
            stages = tm["level_queries"][lid]

            for rk, data in results.items():
                if "centroid" not in data or "heading" not in data or "syllable" not in data or "latent_state" not in data:
                    continue
                cents = data["centroid"]
                heads = data["heading"]
                latents = data["latent_state"]
                sylls = data["syllable"]
                n_f = min(len(cents), len(heads), len(sylls), len(latents))

                cents = cents[:n_f]
                heads = heads[:n_f]
                sylls = sylls[:n_f]
                latents = latents[:n_f]

                mapped = m_arr[sylls]
                mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
                c_seq = mapped[mask].tolist()
                c_starts = np.where(mask)[0].tolist()
                c_ends = c_starts[1:] + [n_f]
                aid, geno, drug = get_verified_metadata(rk)

                for i in range(len(c_seq) - 2):
                    if (c_seq[i] in stages[0] and
                        c_seq[i + 1] in stages[1] and
                        c_seq[i + 2] in stages[2]):

                        sf = c_starts[i]
                        ef = c_ends[i + 2]
                        bounds = [c_starts[i], c_starts[i + 1], c_starts[i + 2], ef]
                        durs = np.diff(bounds)

                        if any(d < 2 for d in durs) or (ef - sf) > 160:
                            continue
                        if ef >= len(cents):
                            continue

                        sub_cents = cents[sf:ef]
                        step_d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                        if np.any(step_d > 25.0):
                            continue # tracking glitch check

                        # Kinematics
                        path_len = float(np.sum(step_d))
                        net_disp = float(np.linalg.norm(sub_cents[-1] - sub_cents[0]))
                        avg_speed = float(np.mean(step_d) * 40.0)
                        dur_sec = float((ef - sf) / 40.0)

                        dh = np.diff(heads[sf:ef])
                        dh = (dh + np.pi) % (2 * np.pi) - np.pi
                        turn_deg = float(np.degrees(np.sum(dh)))

                        # Feature vector (T=20 points)
                        t_orig = np.linspace(0, 1, ef - sf)
                        t_targ = np.linspace(0, 1, 20)
                        from scipy.interpolate import interp1d
                        z_res = interp1d(t_orig, latents[sf:ef], axis=0)(t_targ).flatten()
                        v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d * 40.0)(t_targ)
                        v_norm = (v_res - 60.0) / 45.0
                        dh_res = interp1d(np.linspace(0, 1, len(dh)), dh * 40.0)(t_targ)
                        dh_norm = dh_res / 5.0

                        feat_vec = np.concatenate([z_res, v_norm * 2.0, dh_norm * 2.0])

                        cand_obj = {
                            "level_id": lid,
                            "level_name": lvl_name,
                            "theme_id": th_id,
                            "theme_title": th_title,
                            "recording_key": rk,
                            "animal_id": aid,
                            "genotype": geno,
                            "drug": drug,
                            "start_frame": int(sf),
                            "end_frame": int(ef),
                            "stage_1_end": int(bounds[1]),
                            "stage_2_end": int(bounds[2]),
                            "duration_sec": dur_sec,
                            "avg_speed_px_s": avg_speed,
                            "path_length_px": path_len,
                            "net_displacement_px": net_disp,
                            "signed_turn_deg": turn_deg,
                            "feat_vec": feat_vec
                        }
                        theme_cands_by_level[lid].append(cand_obj)

        # Build unbiased reference from pooled candidates across all 4 levels
        all_theme_feats = []
        for lid in [1, 2, 3, 4]:
            for c in theme_cands_by_level[lid]:
                all_theme_feats.append(c["feat_vec"])

        if len(all_theme_feats) > 0:
            ref_vec = np.mean(all_theme_feats, axis=0)
            ref_norm = np.linalg.norm(ref_vec)

            for lid in [1, 2, 3, 4]:
                for c in theme_cands_by_level[lid]:
                    f = c["feat_vec"]
                    sim = float(np.dot(f, ref_vec) / (np.linalg.norm(f) * ref_norm + 1e-8))
                    c["similarity_score"] = sim
                    c["exceeds_p70"] = bool(sim >= 0.70)
                    c["exceeds_p75"] = bool(sim >= 0.75)
                    c["exceeds_p80"] = bool(sim >= 0.80)
                    c["exceeds_p85"] = bool(sim >= 0.85)
                    del c["feat_vec"] # remove large vector before saving to CSV
                    all_matched_episodes.append(c)

    df_episodes = pd.DataFrame(all_matched_episodes)
    out_csv = AUDIT_DIR / "matched_action_chains_per_episode_audit.csv"
    df_episodes.to_csv(out_csv, index=False)
    print(f"[EXPORT] Saved {len(df_episodes):,} matched benchmark candidate episodes to {out_csv.relative_to(REPO_ROOT)}", flush=True)

    # Print summary counts at Sim >= 0.75
    print("\n--- MATCHED BENCHMARK AT SIM >= 0.75 ---")
    summary_p75 = df_episodes[df_episodes["exceeds_p75"]].groupby(["theme_title", "level_name"]).agg(
        n_episodes=("start_frame", "count"),
        n_animals=("animal_id", "nunique")
    ).reset_index()
    print(summary_p75.to_string(index=False))

    return df_episodes


def main():
    print("=" * 80)
    print("STARTING COMPREHENSIVE KINEMATIC INFORMATION RETENTION & AUDIT PIPELINE")
    print("=" * 80)
    t0 = time.time()

    res = load_chain_movie_resources()
    results = res["results"]
    levels = get_level_definitions()

    # 1. Extract all physical bouts
    df_bouts = extract_all_physical_bouts(results)

    # 2. Run Leave-One-Animal-Out Cross-Validation (LOAO-CV)
    df_loao_results = run_loao_cross_validation(df_bouts, levels)
    loao_csv = AUDIT_DIR / "loao_cross_validation_per_animal_errors.csv"
    df_loao_results.to_csv(loao_csv, index=False)
    print(f"[EXPORT] Saved per-animal LOAO-CV results to {loao_csv.relative_to(REPO_ROOT)}")

    # 3. Compute paired uncertainty and summary statistics
    df_summary = compute_paired_uncertainty_and_summary(df_loao_results, levels)
    summary_csv = AUDIT_DIR / "loao_kinematic_information_retention_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[EXPORT] Saved kinematic summary statistics to {summary_csv.relative_to(REPO_ROOT)}")

    # 4. Breakdown by action class to identify specific lossy merges
    df_class = breakdown_information_loss_by_verb(df_bouts, levels)
    class_csv = AUDIT_DIR / "action_class_kinematic_dispersion_breakdown.csv"
    df_class.to_csv(class_csv, index=False)
    print(f"[EXPORT] Saved action class breakdown to {class_csv.relative_to(REPO_ROOT)}")

    # 5. Export full per-episode CSV for the matched 3-step benchmark
    df_matched = export_matched_action_chains_episode_csv(results, levels)

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PIPELINE SUCCESSFULLY COMPLETED IN {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
