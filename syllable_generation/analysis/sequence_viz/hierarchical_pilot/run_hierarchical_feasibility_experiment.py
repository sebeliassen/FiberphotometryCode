#!/usr/bin/env python3
"""
run_hierarchical_feasibility_experiment.py
=============================================================================
Master Feasibility Experiment: Hierarchical Shared-Subroutine Behavioral Model.
Rigorous, boundary-aware multi-sequence execution addressing reviewer audit:

Repairs & Enhancements:
1. Multi-Sequence Fitting: Independent forward-backward per recording session,
   eliminating artificial cross-session joins during parameter updates.
2. Boundary-Safe Segmentation: Routines and macro-visits segmented strictly
   within sessions; boundary-truncated units flagged; no cross-recording spans.
3. Accurate Animal Breadth: Correctly counts unique mice (max 15), no 'mixed' ID.
4. Unbiased 1st-Order Surrogates (Control A): Preserves empirical transition
   sparsity, enforces zero self-transitions (no s -> s), identical fitting budget.
5. Prior Sensitivity (Control B): E[D] = 4, 6, 10 bouts across multiple seeds.
6. The Progression Experiment (Free vs. Tied Stage Emissions):
   Held-out cross-validation comparing:
   - Free Stage Progression: theta_{k, s} (ordered chains across stages)
   - Tied Stage Mixture: theta_{k, 0} = theta_{k, 1} = theta_{k, 2} = theta_k
   Evaluates held-out log-likelihood per bout on unseen sessions.
=============================================================================
"""

import os
import sys
import time
import json
import pickle
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.hierarchical_pilot.hierarchical_subroutine_model import HierarchicalSubroutineModel

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"
LOG_FILE = OUTPUT_DIR / "hierarchical_pilot.log"
NTFY_TOPIC = "https://ntfy.sh/woody_videos_seb"


def log(msg, notify=False, ntfy_title=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    if notify:
        try:
            req = urllib.request.Request(
                NTFY_TOPIC,
                data=msg.encode("utf-8"),
                headers={"Title": ntfy_title or "Hierarchical Model Pilot"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


def load_cohort_sessions(max_bouts_per_session=1500):
    """Loads session list (one dictionary per recording session)."""
    with open(DATA_PKL, "rb") as f:
        bundle = pickle.load(f)

    sessions_raw = bundle["sessions"]
    cluster_map = np.array(bundle["clusters_0idx"])

    session_list = []
    for s_idx, sess in enumerate(sessions_raw):
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        df = sess["bouts_df"]

        sub_df = df.head(max_bouts_per_session)
        sylls = sub_df["syllable"].values.astype(int)
        durs = sub_df["duration_sec"].values.astype(float)

        session_list.append({
            "session_idx": s_idx,
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "syllables": sylls,
            "durations": durs,
            "total_bouts": len(sylls)
        })

    return session_list, cluster_map


def segment_session_organization(macro_seq, routine_seq, stage_seq, durations, animal_id, genotype, session_idx):
    """
    Segments decoded hidden states strictly within a single recording session.
    Flags boundary-truncated occurrences at start (t=0) and end (t=T).
    """
    T = len(macro_seq)
    routine_occurrences = []
    curr_rout = routine_seq[0]
    curr_macro = macro_seq[0]
    start_t = 0

    for t in range(1, T):
        if routine_seq[t] != curr_rout or (stage_seq[t] == 0 and stage_seq[t-1] == 2):
            end_t = t
            bouts = end_t - start_t
            sec = float(np.sum(durations[start_t:end_t]))
            is_start_trunc = (start_t == 0)
            is_end_trunc = False

            routine_occurrences.append({
                "session_idx": session_idx,
                "animal_id": animal_id,
                "genotype": genotype,
                "routine_id": int(curr_rout),
                "macro_state": int(curr_macro),
                "start_t": start_t,
                "end_t": end_t,
                "bouts": bouts,
                "duration_sec": sec,
                "truncated": is_start_trunc or is_end_trunc
            })

            start_t = t
            curr_rout = routine_seq[t]
            curr_macro = macro_seq[t]

    # Final routine segment of session
    bouts = T - start_t
    sec = float(np.sum(durations[start_t:T]))
    routine_occurrences.append({
        "session_idx": session_idx,
        "animal_id": animal_id,
        "genotype": genotype,
        "routine_id": int(curr_rout),
        "macro_state": int(curr_macro),
        "start_t": start_t,
        "end_t": T,
        "bouts": bouts,
        "duration_sec": sec,
        "truncated": True # ends at session boundary
    })

    df_rout = pd.DataFrame(routine_occurrences)

    # Immediate repetitions (k -> k) within this session
    rep_runs = []
    if len(df_rout) > 0:
        r_ids = df_rout["routine_id"].values
        change_idx = np.where(r_ids[1:] != r_ids[:-1])[0] + 1
        starts = np.r_[0, change_idx]
        ends = np.r_[change_idx, len(r_ids)]

        for st, en in zip(starts, ends):
            rep_runs.append({
                "session_idx": session_idx,
                "animal_id": animal_id,
                "genotype": genotype,
                "routine_id": int(r_ids[st]),
                "consecutive_occurrences": int(en - st),
                "total_bouts": int(df_rout.iloc[st:en]["bouts"].sum()),
                "total_sec": float(df_rout.iloc[st:en]["duration_sec"].sum()),
                "macro_state": int(df_rout.iloc[st]["macro_state"]),
                "contains_boundary_truncation": bool(df_rout.iloc[st:en]["truncated"].any())
            })
    df_rep = pd.DataFrame(rep_runs)

    # Macro-state visits strictly within this session
    macro_visits = []
    change_idx = np.where(macro_seq[1:] != macro_seq[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(macro_seq)]

    for st, en in zip(starts, ends):
        m_id = int(macro_seq[st])
        sub_routs = df_rout[(df_rout["start_t"] >= st) & (df_rout["end_t"] <= en)]
        rout_count = len(sub_routs)
        sec = float(np.sum(durations[st:en]))
        bouts = int(en - st)
        is_trunc = (st == 0) or (en == T)

        macro_visits.append({
            "session_idx": session_idx,
            "animal_id": animal_id,
            "genotype": genotype,
            "macro_state": m_id,
            "start_t": st,
            "end_t": en,
            "bouts": bouts,
            "duration_sec": sec,
            "routine_occurrences": rout_count,
            "unique_routines": int(sub_routs["routine_id"].nunique()) if rout_count > 0 else 0,
            "truncated": is_trunc
        })
    df_macro = pd.DataFrame(macro_visits)

    # Return gaps strictly within this session
    return_gaps = []
    for k in sorted(df_rout["routine_id"].unique()):
        k_indices = np.where(df_rout["routine_id"].values == k)[0]
        if len(k_indices) > 1:
            for idx1, idx2 in zip(k_indices[:-1], k_indices[1:]):
                gap_routs = idx2 - idx1 - 1
                t_gap_start = df_rout.iloc[idx1]["end_t"]
                t_gap_end = df_rout.iloc[idx2]["start_t"]
                gap_bouts = t_gap_end - t_gap_start
                gap_sec = float(np.sum(durations[t_gap_start:t_gap_end]))
                return_gaps.append({
                    "session_idx": session_idx,
                    "animal_id": animal_id,
                    "genotype": genotype,
                    "routine_id": k,
                    "intervening_routines": gap_routs,
                    "intervening_bouts": gap_bouts,
                    "gap_sec": gap_sec
                })
    df_gaps = pd.DataFrame(return_gaps)

    return df_rout, df_rep, df_macro, df_gaps


def generate_sparse_first_order_surrogate(session_list, random_seed=42):
    """
    Generates a 1st-order Markov surrogate preserving exact observed transitions
    (no artificial pseudocounts on unobserved pairs, no self-transitions s -> s).
    """
    rng = np.random.default_rng(random_seed)
    surrogate_sessions = []

    for s_info in session_list:
        orig_sylls = s_info["syllables"]
        orig_durs = s_info["durations"]
        aid = s_info["animal_id"]
        gt = s_info["genotype"]
        T_sess = len(orig_sylls)

        # Empirical transition counts
        trans_counts = {}
        dur_pool = {}
        for i in range(T_sess - 1):
            s_curr, s_next = orig_sylls[i], orig_sylls[i+1]
            if s_curr not in trans_counts:
                trans_counts[s_curr] = {}
            trans_counts[s_curr][s_next] = trans_counts[s_curr].get(s_next, 0) + 1

            if s_curr not in dur_pool:
                dur_pool[s_curr] = []
            dur_pool[s_curr].append(orig_durs[i])

        if orig_sylls[-1] not in dur_pool:
            dur_pool[orig_sylls[-1]] = [orig_durs[-1]]

        # Transition probability distributions
        P_trans = {}
        unigram_counts = np.bincount(orig_sylls, minlength=100)
        unigram_probs = unigram_counts / unigram_counts.sum()

        for s_curr, dest_dict in trans_counts.items():
            dests = list(dest_dict.keys())
            weights = np.array([dest_dict[d] for d in dests], dtype=float)
            P_trans[s_curr] = (dests, weights / weights.sum())

        # Generate sequence
        synth_sylls = np.zeros(T_sess, dtype=int)
        synth_durs = np.zeros(T_sess, dtype=float)

        curr_s = orig_sylls[0]
        synth_sylls[0] = curr_s
        synth_durs[0] = rng.choice(dur_pool[curr_s])

        for t in range(1, T_sess):
            if curr_s in P_trans:
                dests, probs = P_trans[curr_s]
                next_s = rng.choice(dests, p=probs)
            else:
                # Syllable had no outgoing transitions in this session: sample from unigram
                next_s = rng.choice(100, p=unigram_probs)
                while next_s == curr_s: # enforce RLE property (no s -> s)
                    next_s = rng.choice(100, p=unigram_probs)

            synth_sylls[t] = next_s
            synth_durs[t] = rng.choice(dur_pool[next_s]) if next_s in dur_pool else 0.35
            curr_s = next_s

        surrogate_sessions.append({
            "session_idx": s_info["session_idx"],
            "animal_id": aid,
            "genotype": gt,
            "syllables": synth_sylls,
            "durations": synth_durs,
            "total_bouts": T_sess
        })

    return surrogate_sessions


def main():
    log("=" * 80, notify=True, ntfy_title="Hierarchical Pilot (Repaired)")
    log("STARTING BOUNDARY-AWARE HIERARCHICAL EXPERIMENT & PROGRESSION COMPARISON")
    log(f"Tail command: tail -f {LOG_FILE}")
    log("=" * 80)

    # 1. Load multi-sequence cohort
    log("Loading cohort recordings as independent sequences...")
    session_list, cluster_map = load_cohort_sessions(max_bouts_per_session=1500)
    train_sequences = [s["syllables"] for s in session_list]
    tot_bouts = sum(len(s) for s in train_sequences)
    unique_animals = sorted(list(set(s["animal_id"] for s in session_list)))
    log(f"Loaded {len(session_list)} recording sessions across {len(unique_animals)} animals ({tot_bouts:,} total bouts).")

    # =========================================================================
    # PART 1: FIT REAL MULTI-SEQUENCE MODEL (BOUNDARY-SAFE)
    # =========================================================================
    log("\n--- PART 1: FITTING BOUNDARY-SAFE MULTI-SEQUENCE MODEL (25 ITERATIONS) ---", notify=True)
    M, K, S = 3, 6, 3
    model_free = HierarchicalSubroutineModel(M=M, K=K, S=S, n_syllables=100, init_p=0.5, tied_stage_emissions=False, random_seed=42)

    t0 = time.time()
    history_free = model_free.fit_multi_sequence(train_sequences, max_iter=25, tol=1e-3, verbose=True)
    fit_time_free = time.time() - t0
    tot_ll_free, ll_bout_free = model_free.score(train_sequences)
    log(f"✓ Free-stage model fitted in {fit_time_free:.2f}s. Final Total LL={tot_ll_free:.2f} (LL/bout={ll_bout_free:.4f})")

    # Decode each session independently and assemble boundary-safe datasets
    log("\nDecoding states independently per recording session (no join contamination)...")
    all_rout_dfs = []
    all_rep_dfs = []
    all_macro_dfs = []
    all_gap_dfs = []

    for s_info in session_list:
        seq = s_info["syllables"]
        durs = s_info["durations"]
        aid = s_info["animal_id"]
        gt = s_info["genotype"]
        s_idx = s_info["session_idx"]

        m_seq, r_seq, st_seq = model_free.decode_session_viterbi(seq)
        df_rout, df_rep, df_macro, df_gaps = segment_session_organization(
            m_seq, r_seq, st_seq, durs, aid, gt, s_idx
        )
        all_rout_dfs.append(df_rout)
        all_rep_dfs.append(df_rep)
        all_macro_dfs.append(df_macro)
        all_gap_dfs.append(df_gaps)

    df_routines = pd.concat(all_rout_dfs, ignore_index=True)
    df_rep = pd.concat(all_rep_dfs, ignore_index=True)
    df_macro = pd.concat(all_macro_dfs, ignore_index=True)
    df_gaps = pd.concat(all_gap_dfs, ignore_index=True)

    df_routines.to_csv(OUTPUT_DIR / "real_routine_occurrences_boundary_clean.csv", index=False)
    df_rep.to_csv(OUTPUT_DIR / "real_routine_repetitions_boundary_clean.csv", index=False)
    df_macro.to_csv(OUTPUT_DIR / "real_macro_state_visits_boundary_clean.csv", index=False)
    df_gaps.to_csv(OUTPUT_DIR / "real_routine_return_gaps_boundary_clean.csv", index=False)

    # Tabulate Level 1 Summary (Completed, non-truncated occurrences)
    log("\n=== LEVEL 1: BOUNDARY-CLEAN ROUTINE SUMMARY (NON-TRUNCATED DWELLS) ===")
    rout_summary = []
    tot_valid_min = sum(s["durations"].sum() for s in session_list) / 60.0

    # Total bouts in cohort
    tot_cohort_bouts = sum(len(s["syllables"]) for s in session_list)

    for k in range(K):
        sub_all = df_routines[df_routines["routine_id"] == k]
        # Dwells from non-truncated occurrences
        sub_clean = sub_all[~sub_all["truncated"]]

        n_occ_clean = len(sub_clean)
        n_occ_all = len(sub_all)
        n_anim = sub_all["animal_id"].nunique() # true unique animals (max 15)

        b_med = float(sub_clean["bouts"].median()) if n_occ_clean > 0 else 0.0
        b_mean = float(sub_clean["bouts"].mean()) if n_occ_clean > 0 else 0.0
        b_p90 = float(sub_clean["bouts"].quantile(0.90)) if n_occ_clean > 0 else 0.0
        s_med = float(sub_clean["duration_sec"].median()) if n_occ_clean > 0 else 0.0
        s_p90 = float(sub_clean["duration_sec"].quantile(0.90)) if n_occ_clean > 0 else 0.0
        tot_bouts_k = int(sub_all["bouts"].sum())
        bout_share_pct = (tot_bouts_k / tot_cohort_bouts) * 100.0

        rate_per_min = n_occ_all / tot_valid_min

        # Immediate repetition
        sub_rep = df_rep[df_rep["routine_id"] == k]
        rep_frac = float(np.mean(sub_rep["consecutive_occurrences"] > 1)) if len(sub_rep) > 0 else 0.0

        # Return gap
        sub_gap = df_gaps[df_gaps["routine_id"] == k]
        gap_med = float(sub_gap["intervening_bouts"].median()) if len(sub_gap) > 0 else 0.0

        p_k = float(model_free.p_advance[k])
        e_dur = 3.0 / p_k

        rout_summary.append({
            "routine_id": k + 1,
            "learned_p_adv": round(p_k, 3),
            "expected_bouts": round(e_dur, 2),
            "clean_occurrences": n_occ_clean,
            "rate_per_min": round(rate_per_min, 2),
            "median_bouts": b_med,
            "mean_bouts": round(b_mean, 2),
            "p90_bouts": b_p90,
            "median_sec": round(s_med, 2),
            "p90_sec": round(s_p90, 2),
            "bout_share_pct": round(bout_share_pct, 1),
            "repeat_run_frac": round(rep_frac, 3),
            "median_return_gap_bouts": gap_med,
            "contributing_animals": n_anim
        })

    df_rout_sum = pd.DataFrame(rout_summary)
    df_rout_sum.to_csv(OUTPUT_DIR / "real_level1_routine_summary_boundary_clean.csv", index=False)
    log(df_rout_sum.to_string(index=False))

    # Level 2 Summary (Macro visits strictly within recording sessions)
    log("\n=== LEVEL 2: BOUNDARY-CLEAN MACRO-STATE SUMMARY (WITHIN-SESSION VISITS) ===")
    macro_summary = []
    for u in range(M):
        sub_m_all = df_macro[df_macro["macro_state"] == u]
        sub_m_clean = sub_m_all[~sub_m_all["truncated"]]

        n_vis = len(sub_m_all)
        med_routs = float(sub_m_clean["routine_occurrences"].median()) if len(sub_m_clean) > 0 else 0.0
        p90_routs = float(sub_m_clean["routine_occurrences"].quantile(0.90)) if len(sub_m_clean) > 0 else 0.0
        med_sec = float(sub_m_clean["duration_sec"].median()) if len(sub_m_clean) > 0 else 0.0
        mean_sec = float(sub_m_clean["duration_sec"].mean()) if len(sub_m_clean) > 0 else 0.0
        max_sec = float(sub_m_clean["duration_sec"].max()) if len(sub_m_clean) > 0 else 0.0
        tot_min = float(sub_m_all["duration_sec"].sum() / 60.0)

        macro_summary.append({
            "macro_state": u + 1,
            "total_visits": n_vis,
            "clean_visits": len(sub_m_clean),
            "median_routines_per_visit": med_routs,
            "p90_routines_per_visit": p90_routs,
            "median_visit_sec": round(med_sec, 1),
            "mean_visit_sec": round(mean_sec, 1),
            "max_visit_sec": round(max_sec, 1),
            "total_occupied_min": round(tot_min, 1)
        })
    df_macro_sum = pd.DataFrame(macro_summary)
    df_macro_sum.to_csv(OUTPUT_DIR / "real_level2_macro_summary_boundary_clean.csv", index=False)
    log(df_macro_sum.to_string(index=False))

    # =========================================================================
    # PART 2: THE PROGRESSION EXPERIMENT (FREE vs. TIED STAGE EMISSIONS)
    # =========================================================================
    log("\n" + "=" * 80, notify=True)
    log("PART 2: THE PROGRESSION EXPERIMENT (FREE vs. TIED STAGE EMISSIONS)")
    log("Question: Does a routine contain a reproducible sequential progression,")
    log("or does it mainly group bouts with similar stationary syllable usage?")
    log("=" * 80)

    # 5-fold cross-validation on recordings (6 test sessions per fold)
    n_sessions = len(session_list)
    n_splits = 5
    fold_size = n_sessions // n_splits
    cv_results = []

    for fold_i in range(n_splits):
        test_indices = list(range(fold_i * fold_size, (fold_i + 1) * fold_size))
        train_indices = [i for i in range(n_sessions) if i not in test_indices]

        train_seqs = [session_list[i]["syllables"] for i in train_indices]
        test_seqs = [session_list[i]["syllables"] for i in test_indices]

        # Fit Model A: Free Stage Emissions (Sequential Progressions)
        m_free_cv = HierarchicalSubroutineModel(M=M, K=K, S=S, n_syllables=100, init_p=0.5, tied_stage_emissions=False, random_seed=42 + fold_i)
        m_free_cv.fit_multi_sequence(train_seqs, max_iter=25, tol=1e-3, verbose=False)
        _, test_ll_free = m_free_cv.score(test_seqs)

        # Fit Model B: Tied Stage Emissions (Stationary Mixtures)
        m_tied_cv = HierarchicalSubroutineModel(M=M, K=K, S=S, n_syllables=100, init_p=0.5, tied_stage_emissions=True, random_seed=42 + fold_i)
        m_tied_cv.fit_multi_sequence(train_seqs, max_iter=25, tol=1e-3, verbose=False)
        _, test_ll_tied = m_tied_cv.score(test_seqs)

        delta_ll = test_ll_free - test_ll_tied
        cv_results.append({
            "fold": fold_i + 1,
            "test_sessions": len(test_indices),
            "free_stage_heldout_ll_per_bout": test_ll_free,
            "tied_stage_heldout_ll_per_bout": test_ll_tied,
            "delta_ll_per_bout": delta_ll,
            "advantage": "Free Progression" if delta_ll > 0 else "Tied Mixture"
        })
        log(f"  Fold {fold_i+1}/5 | Free LL/bout: {test_ll_free:.4f} | Tied LL/bout: {test_ll_tied:.4f} | Δ: {delta_ll:+.4f} ({cv_results[-1]['advantage']})")

    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(OUTPUT_DIR / "progression_free_vs_tied_cv_results.csv", index=False)

    mean_delta = df_cv["delta_ll_per_bout"].mean()
    std_delta = df_cv["delta_ll_per_bout"].std()
    log("\n=== PROGRESSION EXPERIMENT SUMMARY ===")
    log(f"Mean Held-Out LL/bout Advantage for Free Progressions: {mean_delta:+.4f} ± {std_delta:.4f}")
    if mean_delta > 0:
        log("✓ CONCLUSION: Free-stage progressions achieve higher held-out likelihood than tied mixtures.")
        log("  Routines contain genuine reproducible stage-to-stage sequential structure, not just stationary clusters.")
    else:
        log("✓ CONCLUSION: Tied-stage mixtures perform comparably or better.")
        log("  Routines primarily group bouts into stationary behavioural mixtures of characteristic duration.")

    # =========================================================================
    # PART 3: CONTROL A (UNBIASED 1ST-ORDER MARKOV SURROGATES)
    # =========================================================================
    log("\n" + "=" * 80, notify=True)
    log("PART 3: CONTROL A (10 UNBIASED 1ST-ORDER MARKOV SURROGATES)")
    log("Preserving exact transition sparsity & RLE constraints; identical 25-iteration budget.")
    log("=" * 80)

    N_SURROGATES = 10
    surrogate_results = []

    for s_i in range(N_SURROGATES):
        t_s0 = time.time()
        surr_sessions = generate_sparse_first_order_surrogate(session_list, random_seed=2000 + s_i)
        surr_seqs = [s["syllables"] for s in surr_sessions]

        m_surr = HierarchicalSubroutineModel(M=M, K=K, S=S, n_syllables=100, init_p=0.5, tied_stage_emissions=False, random_seed=42)
        surr_hist = m_surr.fit_multi_sequence(surr_seqs, max_iter=25, tol=1e-3, verbose=False)
        surr_tot_ll, surr_ll_bout = m_surr.score(surr_seqs)

        # Decode surrogate sessions
        s_rout_dfs = []
        s_rep_dfs = []
        s_macro_dfs = []
        for s_info in surr_sessions:
            seq = s_info["syllables"]
            durs = s_info["durations"]
            aid = s_info["animal_id"]
            gt = s_info["genotype"]
            s_idx = s_info["session_idx"]

            m_seq, r_seq, st_seq = m_surr.decode_session_viterbi(seq)
            df_r, df_rp, df_m, _ = segment_session_organization(m_seq, r_seq, st_seq, durs, aid, gt, s_idx)
            s_rout_dfs.append(df_r)
            s_rep_dfs.append(df_rp)
            s_macro_dfs.append(df_m)

        s_rout_all = pd.concat(s_rout_dfs, ignore_index=True)
        s_rep_all = pd.concat(s_rep_dfs, ignore_index=True)
        s_macro_all = pd.concat(s_macro_dfs, ignore_index=True)

        s_clean_routs = s_rout_all[~s_rout_all["truncated"]]
        s_clean_macros = s_macro_all[~s_macro_all["truncated"]]

        mean_bouts = float(s_clean_routs["bouts"].mean()) if len(s_clean_routs) > 0 else 0.0
        mean_rep_len = float(s_rep_all["consecutive_occurrences"].mean()) if len(s_rep_all) > 0 else 1.0
        mean_macro_routs = float(s_clean_macros["routine_occurrences"].mean()) if len(s_clean_macros) > 0 else 0.0

        elapsed_s = time.time() - t_s0
        log(f"  Surrogate {s_i+1:02d}/10 | Mean Rout Bouts: {mean_bouts:.2f} | Routs/MacroVisit: {mean_macro_routs:.2f} | LL/bout: {surr_ll_bout:.4f} ({elapsed_s:.1f}s)")

        surrogate_results.append({
            "surrogate_id": s_i + 1,
            "loglik_per_bout": surr_ll_bout,
            "mean_routine_bouts": mean_bouts,
            "mean_repetition_length": mean_rep_len,
            "mean_routines_per_macro_visit": mean_macro_routs
        })

    df_surr_summary = pd.DataFrame(surrogate_results)
    df_surr_summary.to_csv(OUTPUT_DIR / "control_a_surrogate_summary_repaired.csv", index=False)

    # Compare Real vs Unbiased Surrogates
    real_clean_routs = df_routines[~df_routines["truncated"]]
    real_clean_macros = df_macro[~df_macro["truncated"]]

    real_mean_bouts = float(real_clean_routs["bouts"].mean())
    real_mean_rep = float(df_rep["consecutive_occurrences"].mean())
    real_mean_macro_routs = float(real_clean_macros["routine_occurrences"].mean())

    log("\n=== CONTROL A (REPAIRED): REAL DATA vs. UNBIASED 1ST-ORDER NULL ===")
    log(f"Metric                      | Real Data | Surrogates (Mean ± Std) | Direction / Effect")
    log("-" * 80)
    surr_b_mean = df_surr_summary["mean_routine_bouts"].mean()
    surr_b_std = df_surr_summary["mean_routine_bouts"].std()
    log(f"Mean Routine Dwell (bouts)  | {real_mean_bouts:9.2f} | {surr_b_mean:6.2f} ± {surr_b_std:4.2f}      | Real vs Null: {real_mean_bouts - surr_b_mean:+.2f}")

    surr_rep_mean = df_surr_summary["mean_repetition_length"].mean()
    surr_rep_std = df_surr_summary["mean_repetition_length"].std()
    log(f"Immediate Repetition Length | {real_mean_rep:9.2f} | {surr_rep_mean:6.2f} ± {surr_rep_std:4.2f}      | Real vs Null: {real_mean_rep - surr_rep_mean:+.2f}")

    surr_m_mean = df_surr_summary["mean_routines_per_macro_visit"].mean()
    surr_m_std = df_surr_summary["mean_routines_per_macro_visit"].std()
    log(f"Routines per Upper Visit    | {real_mean_macro_routs:9.2f} | {surr_m_mean:6.2f} ± {surr_m_std:4.2f}      | Real vs Null: {real_mean_macro_routs - surr_m_mean:+.2f}")

    surr_ll_mean = df_surr_summary["loglik_per_bout"].mean()
    log(f"Log-Likelihood per Bout     | {ll_bout_free:9.4f} | {surr_ll_mean:6.4f} ± {df_surr_summary['loglik_per_bout'].std():4.4f}      | Real vs Null: {ll_bout_free - surr_ll_mean:+.4f}")

    # =========================================================================
    # PART 4: CONTROL B (INITIALIZATION SENSITIVITY IN MULTI-SEQUENCE)
    # =========================================================================
    log("\n" + "=" * 80, notify=True)
    log("PART 4: CONTROL B (INITIALIZATION SENSITIVITY IN MULTI-SEQUENCE)")
    log("=" * 80)

    init_dur_regimes = [
        {"name": "Short (4 bouts)", "init_p": 0.75},
        {"name": "Medium (6 bouts)", "init_p": 0.50},
        {"name": "Long (10 bouts)", "init_p": 0.30}
    ]
    seeds = [11, 42, 101]
    init_results = []

    for regime in init_dur_regimes:
        r_name = regime["name"]
        init_p = regime["init_p"]
        for s in seeds:
            m_init = HierarchicalSubroutineModel(M=M, K=K, S=S, n_syllables=100, init_p=init_p, tied_stage_emissions=False, random_seed=s)
            h_init = m_init.fit_multi_sequence(train_sequences, max_iter=25, tol=1e-3, verbose=False)
            final_p = m_init.p_advance.copy()
            final_exp_durs = 3.0 / final_p

            init_results.append({
                "regime": r_name,
                "seed": s,
                "initial_p": init_p,
                "initial_expected_duration": round(3.0 / init_p, 1),
                "learned_p_mean": float(np.mean(final_p)),
                "learned_expected_duration_mean": float(np.mean(final_exp_durs)),
                "learned_expected_duration_min": float(np.min(final_exp_durs)),
                "learned_expected_duration_max": float(np.max(final_exp_durs)),
                "routine_1_expected_duration": float(final_exp_durs[0]),
                "routines_2_to_6_mean_expected_duration": float(np.mean(final_exp_durs[1:]))
            })
            log(f"  [{r_name} | Seed {s:3d}]: Initial E[D]={3.0/init_p:.1f} -> Learned E[D]={np.mean(final_exp_durs):.2f} bouts (Rout 1: {final_exp_durs[0]:.1f}b, Routs 2-6: {np.mean(final_exp_durs[1:]):.1f}b)")

    df_init = pd.DataFrame(init_results)
    df_init.to_csv(OUTPUT_DIR / "control_b_initialization_sensitivity_repaired.csv", index=False)

    log("\n=== CONTROL B SUMMARY ===")
    for r_name in [r["name"] for r in init_dur_regimes]:
        sub_i = df_init[df_init["regime"] == r_name]
        mean_init = sub_i["initial_expected_duration"].iloc[0]
        mean_learned = sub_i["learned_expected_duration_mean"].mean()
        std_learned = sub_i["learned_expected_duration_mean"].std()
        r1_mean = sub_i["routine_1_expected_duration"].mean()
        r26_mean = sub_i["routines_2_to_6_mean_expected_duration"].mean()
        log(f"  • {r_name:18s}: Initial={mean_init:.1f}b -> Learned={mean_learned:.2f} ± {std_learned:.2f}b (Rout 1: {r1_mean:.1f}b, Routs 2-6: {r26_mean:.1f}b)")

    # Save repaired bundle
    bundle_path = OUTPUT_DIR / "hierarchical_model_results_bundle_repaired.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump({
            "model_free": model_free,
            "progression_cv": df_cv,
            "surrogate_summary": df_surr_summary,
            "init_summary": df_init,
            "df_routines": df_routines,
            "df_rep": df_rep,
            "df_macro": df_macro,
            "df_gaps": df_gaps,
            "df_rout_sum": df_rout_sum,
            "df_macro_sum": df_macro_sum
        }, f)

    log(f"\n✓ Saved repaired results bundle to: {bundle_path.name}")
    log("\n" + "=" * 80, notify=True, ntfy_title="Repaired Experiment Complete!")
    log("REPAIRED HIERARCHICAL EXPERIMENT & PROGRESSION AUDIT FINISHED SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    main()
