#!/usr/bin/env python3
"""
run_controlled_upper_timescale_experiment.py
=============================================================================
Comprehensive Controlled Upper-Timescale Experiment & M=4 State Resolution:

1. Bug fix: Routine advance probability (p_advance) update includes all 3 stages,
   including staying in and advancing/exiting from stage S-1.
2. M=4 Upper States: Fits 4 upper states (U1..U4) with neutral numerical labels.
3. Upper-Timescale Intervention: Compares:
   - Free A_uu (unconstrained learned persistence)
   - Fixed A_uu = 0.98 (expected 50 routines / ~150s)
   - Fixed A_uu = 0.90 (expected 10 routines / ~30s)
   - Fixed A_uu = 0.80 (expected 5 routines / ~15s)
   Evaluates held-out cross-validation log-likelihood per bout.
4. Posterior Uncertainty: Computes gamma_t(u) posterior probabilities alongside
   hard Viterbi assignments to measure certainty vs. ambiguity.
5. Produces:
   - fig_controlled_timescale_comparison.png (Diagnostic benchmark)
   - fig_hierarchical_timeline_4state_overview.png (30-min 4-state overview)
   - fig_hierarchical_timeline_4state_detail.png (120s detail with posterior certainty)
   - hierarchical_timeline_4state_interactive.html (Interactive viewer with posteriors)
   - controlled_timescale_diagnostic_summary.csv
=============================================================================
"""

import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.hierarchical_pilot.hierarchical_subroutine_model import (
    HierarchicalSubroutineModel,
    build_transition_matrix_np,
    jax_forward_backward,
    jax_forward_loglik_only,
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82")
DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"

# Palettes
ROUTINE_COLORS = {
    0: "#E41A1C", # R1: Vivid Crimson
    1: "#377EB8", # R2: Sapphire Blue
    2: "#4DAF4A", # R3: Emerald Green
    3: "#984EA3", # R4: Royal Amethyst
    4: "#FF7F00", # R5: Amber Orange
    5: "#00A896"  # R6: Deep Teal
}

# Neutral 4 Upper States Palette
UPPER_COLORS_4 = {
    0: "#6C8EBF", # U1: Slate Blue
    1: "#82B366", # U2: Sage Green
    2: "#D6B656", # U3: Warm Ochre
    3: "#B85450"  # U4: Muted Terracotta
}

rng = np.random.default_rng(12345)
hues = np.linspace(0, 1, 100, endpoint=False)
rng.shuffle(hues)
SYLLABLE_COLORS = {}
for s in range(100):
    rgb = mcolors.hsv_to_rgb([hues[s], 0.75, 0.85])
    SYLLABLE_COLORS[s] = mcolors.to_hex(rgb)


def load_30min_cohort(max_duration_sec=1800.0):
    with open(DATA_PKL, "rb") as f:
        bundle = pickle.load(f)

    sessions_raw = bundle["sessions"]
    session_list = []

    all_log_dur = []
    all_log_vel = []

    for s_idx, sess in enumerate(sessions_raw):
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        df = sess["bouts_df"]

        cum_durs = df["duration_sec"].cumsum().values
        valid_mask = cum_durs <= max_duration_sec
        if not np.any(valid_mask):
            valid_mask = np.zeros(len(df), dtype=bool)
            valid_mask[0] = True
        sub_df = df[valid_mask]

        sylls = sub_df["syllable"].values.astype(int)
        durs = sub_df["duration_sec"].values.astype(float)
        vels = sub_df["mean_vel"].values.astype(float)

        log_dur = np.log(np.clip(durs, 1e-3, None))
        log_vel = np.log(np.clip(vels + 1.0, 1e-3, None))

        all_log_dur.extend(log_dur)
        all_log_vel.extend(log_vel)

        session_list.append({
            "session_idx": s_idx,
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "syllables": sylls,
            "durations": durs,
            "velocities": vels,
            "log_dur": log_dur,
            "log_vel": log_vel,
            "total_sec": float(np.sum(durs)),
            "total_bouts": len(sylls)
        })

    dur_mean = float(np.mean(all_log_dur))
    dur_std = float(np.std(all_log_dur))
    vel_mean = float(np.mean(all_log_vel))
    vel_std = float(np.std(all_log_vel))

    for sess in session_list:
        norm_dur = (sess["log_dur"] - dur_mean) / dur_std
        norm_vel = (sess["log_vel"] - vel_mean) / vel_std
        sess["features"] = np.column_stack([norm_dur, norm_vel])

    return session_list, (dur_mean, dur_std, vel_mean, vel_std)


class ControlledHierarchicalModel(HierarchicalSubroutineModel):
    """
    Hierarchical Model supporting:
    - Arbitrary M upper states (e.g. M=4)
    - Bug-fixed p_advance update including all stages (0, 1, ..., S-1)
    - Joint categorical + continuous Gaussian emissions
    - Optional fixed diagonal A_uu constraint for controlled timescale experiments
    - Posterior probability extraction gamma_t(u)
    """
    def __init__(self, M=4, K=6, S=3, n_features=2, min_variance=0.05, fixed_A_diag=None, **kwargs):
        self.n_features = n_features
        self.min_variance = min_variance
        self.fixed_A_diag = fixed_A_diag  # None or float in (0, 1)
        self.mu_feat = None
        self.var_feat = None
        super().__init__(M=M, K=K, S=S, **kwargs)

    def _initialize_parameters(self):
        super()._initialize_parameters()
        rng = np.random.default_rng(self.random_seed)
        self.mu_feat = rng.normal(0.0, 0.2, size=(self.K, self.S, self.n_features)).astype(np.float32)
        self.var_feat = np.ones((self.K, self.S, self.n_features), dtype=np.float32)

        if self.fixed_A_diag is not None:
            diag_val = float(self.fixed_A_diag)
            off_val = (1.0 - diag_val) / (self.M - 1)
            A = np.full((self.M, self.M), off_val, dtype=np.float32)
            np.fill_diagonal(A, diag_val)
            self.A_macro = A
            self.T_mat = self._build_transition_matrix()

    def _build_transition_matrix(self):
        return build_transition_matrix_np(self.M, self.K, self.S, self.p_advance, self.A_macro, self.B_routine)

    def get_log_emission_seq_joint(self, syllables, features):
        T = len(syllables)
        log_theta = np.log(np.clip(self.theta, 1e-12, 1.0))
        cat_log_emiss = log_theta[:, :, syllables].transpose(2, 0, 1)  # (T, K, S)

        f_exp = features[:, None, None, :]
        mu_exp = self.mu_feat[None, :, :, :]
        var_exp = self.var_feat[None, :, :, :]

        diff_sq = (f_exp - mu_exp) ** 2
        gauss_log = -0.5 * np.sum(np.log(2.0 * np.pi * var_exp) + diff_sq / var_exp, axis=-1)

        joint_ks = cat_log_emiss + gauss_log  # (T, K, S)
        joint_all = np.repeat(joint_ks[:, None, :, :], self.M, axis=1)
        return joint_all.reshape(T, self.N_states)

    def fit_multi_sequence_joint(self, sequence_list, max_iters=25, tol=1e-4):
        self._initialize_parameters()
        log_history = []

        for it in range(max_iters):
            T_mat_j = jnp.array(self.T_mat)
            log_trans = jnp.log(jnp.clip(T_mat_j, 1e-12, 1.0))
            init_log_pi = jnp.log(jnp.clip(jnp.array(self.pi_init), 1e-12, 1.0))

            accum_xi = np.zeros_like(self.T_mat)
            accum_gamma = np.zeros(self.N_states, dtype=np.float64)
            accum_theta_counts = np.zeros_like(self.theta)

            accum_feat_w = np.zeros((self.K, self.S), dtype=np.float64)
            accum_feat_sum = np.zeros((self.K, self.S, self.n_features), dtype=np.float64)
            accum_feat_sq = np.zeros((self.K, self.S, self.n_features), dtype=np.float64)

            total_cohort_log_lik = 0.0

            for sess in sequence_list:
                sylls = sess["syllables"]
                feats = sess["features"]
                log_emiss = self.get_log_emission_seq_joint(sylls, feats)

                log_gamma, log_xi_sum, sess_ll = jax_forward_backward(
                    log_trans, jnp.array(log_emiss), init_log_pi
                )

                gamma = np.array(jnp.exp(log_gamma))
                xi_sum = np.array(jnp.exp(log_xi_sum))
                total_cohort_log_lik += float(sess_ll)

                accum_xi += xi_sum
                accum_gamma += np.sum(gamma, axis=0)

                gamma_mks = gamma.reshape(-1, self.M, self.K, self.S)
                gamma_ks = np.sum(gamma_mks, axis=1)

                for t_idx, y in enumerate(sylls):
                    accum_theta_counts[:, :, y] += gamma_ks[t_idx]

                for k in range(self.K):
                    for s in range(self.S):
                        w = gamma_ks[:, k, s]
                        accum_feat_w[k, s] += np.sum(w)
                        accum_feat_sum[k, s] += np.sum(w[:, None] * feats, axis=0)
                        accum_feat_sq[k, s] += np.sum(w[:, None] * (feats ** 2), axis=0)

            log_history.append(total_cohort_log_lik)

            # --- M-step: 1. BUG-FIXED p_advance across ALL stages (0..S-1) ---
            for k in range(self.K):
                n_self = 0.0
                n_adv = 0.0
                # Stages 0 to S-2: advance to s+1
                for s in range(self.S - 1):
                    for m in range(self.M):
                        curr_idx = self.tuple_to_state(m, k, s)
                        self_idx = curr_idx
                        adv_idx = self.tuple_to_state(m, k, s + 1)
                        n_self += accum_xi[curr_idx, self_idx]
                        n_adv += accum_xi[curr_idx, adv_idx]

                # Stage S-1 (final stage): stay in S-1 vs. exit routine
                for m in range(self.M):
                    exit_idx = self.tuple_to_state(m, k, self.S - 1)
                    n_self += accum_xi[exit_idx, exit_idx]
                    for m_prime in range(self.M):
                        for k_prime in range(self.K):
                            entry_idx = self.tuple_to_state(m_prime, k_prime, 0)
                            n_adv += accum_xi[exit_idx, entry_idx]

                a = self.alpha_p + n_adv
                b = self.beta_p + n_self
                self.p_advance[k] = float(np.clip(a / (a + b), 0.05, 0.95))

            # --- M-step: 2. Syllable emissions ---
            pseudocount = 0.1
            numer_theta = accum_theta_counts + pseudocount
            denom_theta = np.sum(numer_theta, axis=-1, keepdims=True)
            self.theta = numer_theta / denom_theta

            # --- M-step: 3. Continuous feature parameters ---
            denom_w = np.maximum(accum_feat_w[:, :, None], 1e-6)
            new_mu = accum_feat_sum / denom_w
            new_var = (accum_feat_sq / denom_w) - (new_mu ** 2)
            self.mu_feat = new_mu
            self.var_feat = np.maximum(new_var, self.min_variance)

            # --- M-step: 4. Upper transitions and routine selection ---
            trans_exit = np.zeros((self.M, self.M), dtype=np.float64)
            routine_sel = np.zeros((self.M, self.K), dtype=np.float64)

            for m in range(self.M):
                for k in range(self.K):
                    exit_idx = self.tuple_to_state(m, k, self.S - 1)
                    for m_prime in range(self.M):
                        for k_prime in range(self.K):
                            entry_idx = self.tuple_to_state(m_prime, k_prime, 0)
                            flow = accum_xi[exit_idx, entry_idx]
                            trans_exit[m, m_prime] += flow
                            routine_sel[m_prime, k_prime] += flow

            prior_B = 1.0
            num_B = routine_sel + prior_B
            self.B_routine = num_B / np.sum(num_B, axis=1, keepdims=True)

            if self.fixed_A_diag is not None:
                # Controlled timescale: fix A_uu and normalize off-diagonals proportionally
                diag_val = float(self.fixed_A_diag)
                new_A = np.zeros((self.M, self.M), dtype=np.float64)
                for u in range(self.M):
                    new_A[u, u] = diag_val
                    off_counts = np.array([trans_exit[u, v] + 0.5 for v in range(self.M) if v != u])
                    off_probs = (off_counts / np.sum(off_counts)) * (1.0 - diag_val)
                    idx = 0
                    for v in range(self.M):
                        if v != u:
                            new_A[u, v] = off_probs[idx]
                            idx += 1
                self.A_macro = new_A
            else:
                prior_A = 2.0
                num_A = trans_exit + prior_A
                self.A_macro = num_A / np.sum(num_A, axis=1, keepdims=True)

            self.T_mat = self._build_transition_matrix()

            if it > 0 and abs(log_history[-1] - log_history[-2]) < tol:
                break

        return log_history

    def compute_session_posteriors(self, syllables, features):
        """Returns posterior gamma_macro: shape (T, M) and certainty: shape (T,)."""
        T_mat_j = jnp.array(self.T_mat)
        log_trans = jnp.log(jnp.clip(T_mat_j, 1e-12, 1.0))
        init_log_pi = jnp.log(jnp.clip(jnp.array(self.pi_init), 1e-12, 1.0))
        log_emiss = self.get_log_emission_seq_joint(syllables, features)

        log_gamma, _, _ = jax_forward_backward(log_trans, jnp.array(log_emiss), init_log_pi)
        gamma = np.array(jnp.exp(log_gamma))  # (T, N_states)

        gamma_mks = gamma.reshape(-1, self.M, self.K, self.S)
        gamma_macro = np.sum(gamma_mks, axis=(2, 3))  # (T, M)
        certainty = np.max(gamma_macro, axis=-1)  # (T,)
        return gamma_macro, certainty

    def decode_session_viterbi_joint(self, syllables, features):
        T = len(syllables)
        log_emiss = self.get_log_emission_seq_joint(syllables, features)
        log_trans = np.log(np.clip(self.T_mat, 1e-12, 1.0))
        log_pi = np.log(np.clip(self.pi_init, 1e-12, 1.0))

        V = np.zeros((T, self.N_states), dtype=np.float32)
        ptr = np.zeros((T, self.N_states), dtype=np.int32)

        V[0] = log_pi + log_emiss[0]
        for t in range(1, T):
            trans_matrix = V[t - 1, :, None] + log_trans
            ptr[t] = np.argmax(trans_matrix, axis=0)
            V[t] = trans_matrix[ptr[t], np.arange(self.N_states)] + log_emiss[t]

        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = ptr[t + 1, best_path[t + 1]]

        macro_seq = np.zeros(T, dtype=np.int32)
        routine_seq = np.zeros(T, dtype=np.int32)
        stage_seq = np.zeros(T, dtype=np.int32)
        for t in range(T):
            m, k, s = self.state_to_tuple(best_path[t])
            macro_seq[t] = m
            routine_seq[t] = k
            stage_seq[t] = s

        return macro_seq, routine_seq, stage_seq

    def score_sequence(self, syllables, features):
        log_trans = jnp.log(jnp.clip(jnp.array(self.T_mat), 1e-12, 1.0))
        init_log_pi = jnp.log(jnp.clip(jnp.array(self.pi_init), 1e-12, 1.0))
        log_emiss = self.get_log_emission_seq_joint(syllables, features)
        return float(jax_forward_loglik_only(log_trans, jnp.array(log_emiss), init_log_pi))


def run_controlled_timescale_sweep(sessions):
    """
    Compares 4 conditions using 5-fold cross-validation:
    1. Free A_uu
    2. Fixed A_uu = 0.98 (expected 50 routines)
    3. Fixed A_uu = 0.90 (expected 10 routines)
    4. Fixed A_uu = 0.80 (expected 5 routines)
    """
    conditions = [
        ("Free A_uu", None),
        ("A_uu = 0.98 (~50 rout)", 0.98),
        ("A_uu = 0.90 (~10 rout)", 0.90),
        ("A_uu = 0.80 (~5 rout)", 0.80)
    ]

    print("=" * 80)
    print("RUNNING CONTROLLED UPPER-TIMESCALE SWEEP (5-FOLD CV ACROSS SESSIONS)")
    print("=" * 80)

    n_sessions = len(sessions)
    fold_size = n_sessions // 5
    results = []

    for cond_name, fixed_diag in conditions:
        print(f"\nEvaluating Condition: {cond_name}...")
        fold_test_lls = []
        fold_visit_durs = []
        fold_visit_counts = []
        fold_certainties = []

        for fold in range(5):
            test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
            train_indices = [i for i in range(n_sessions) if i not in test_indices]

            train_sessions = [sessions[i] for i in train_indices]
            test_sessions = [sessions[i] for i in test_indices]

            model = ControlledHierarchicalModel(
                M=4, K=6, S=3, n_features=2, fixed_A_diag=fixed_diag, random_seed=42 + fold
            )
            model.fit_multi_sequence_joint(train_sessions, max_iters=20)

            # Evaluate on held-out test sessions
            for test_s in test_sessions:
                ll = model.score_sequence(test_s["syllables"], test_s["features"])
                ll_per_bout = ll / test_s["total_bouts"]
                fold_test_lls.append(ll_per_bout)

                # Decode test session
                m_seq, r_seq, _ = model.decode_session_viterbi_joint(test_s["syllables"], test_s["features"])
                _, cert = model.compute_session_posteriors(test_s["syllables"], test_s["features"])
                fold_certainties.append(float(np.mean(cert)))

                # Count macro visits
                visits = []
                curr_m = m_seq[0]
                st = 0
                for t in range(1, len(m_seq)):
                    if m_seq[t] != curr_m:
                        visits.append(np.sum(test_s["durations"][st:t]))
                        curr_m = m_seq[t]
                        st = t
                visits.append(np.sum(test_s["durations"][st:]))
                fold_visit_counts.append(len(visits))
                fold_visit_durs.extend(visits)

        mean_ll = float(np.mean(fold_test_lls))
        std_ll = float(np.std(fold_test_lls))
        med_dur = float(np.median(fold_visit_durs))
        q90_dur = float(np.percentile(fold_visit_durs, 90))
        mean_vis = float(np.mean(fold_visit_counts))
        mean_cert = float(np.mean(fold_certainties))

        print(f"  • Held-out LL/bout: {mean_ll:.4f} ± {std_ll:.4f}")
        print(f"  • Median Visit Dur: {med_dur:.1f}s (90th%: {q90_dur:.1f}s) | Mean Visits/Session: {mean_vis:.1f}")
        print(f"  • Mean Posterior Certainty: {mean_cert*100:.1f}%")

        results.append({
            "condition": cond_name,
            "fixed_A_diag": fixed_diag if fixed_diag else "Free",
            "held_out_ll_per_bout_mean": round(mean_ll, 4),
            "held_out_ll_per_bout_std": round(std_ll, 4),
            "median_visit_sec": round(med_dur, 2),
            "q90_visit_sec": round(q90_dur, 2),
            "mean_visits_per_session": round(mean_vis, 1),
            "mean_posterior_certainty_pct": round(mean_cert * 100, 1)
        })

    df_res = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "controlled_timescale_diagnostic_summary.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n✓ Saved Controlled Timescale Summary to: {csv_path.name}")
    return df_res


def extract_tracks_with_posteriors(session_info, model):
    sylls = session_info["syllables"]
    durs = session_info["durations"]
    feats = session_info["features"]
    aid = session_info["animal_id"]
    gt = session_info["genotype"]
    s_idx = session_info["session_idx"]
    drug = session_info["drug"]

    T = len(sylls)
    offsets = np.cumsum(durs)
    onsets = np.r_[0.0, offsets[:-1]]

    m_seq, r_seq, st_seq = model.decode_session_viterbi_joint(sylls, feats)
    gamma_macro, cert = model.compute_session_posteriors(sylls, feats)

    syll_blocks = []
    for t in range(T):
        syll_blocks.append({
            "syllable": int(sylls[t]),
            "start_sec": float(onsets[t]),
            "end_sec": float(offsets[t]),
            "duration_sec": float(durs[t]),
            "stage": int(st_seq[t]),
            "routine": int(r_seq[t]),
            "macro": int(m_seq[t]),
            "certainty": float(cert[t])
        })

    rout_blocks = []
    curr_r = r_seq[0]
    curr_m = m_seq[0]
    r_start = 0
    for t in range(1, T):
        if r_seq[t] != curr_r or (st_seq[t] == 0 and st_seq[t-1] == 2):
            rout_blocks.append({
                "routine_id": int(curr_r),
                "macro_state": int(curr_m),
                "start_sec": float(onsets[r_start]),
                "end_sec": float(offsets[t - 1]),
                "duration_sec": float(np.sum(durs[r_start:t])),
                "bouts": int(t - r_start),
                "truncated": (r_start == 0)
            })
            curr_r = r_seq[t]
            curr_m = m_seq[t]
            r_start = t

    rout_blocks.append({
        "routine_id": int(curr_r),
        "macro_state": int(curr_m),
        "start_sec": float(onsets[r_start]),
        "end_sec": float(offsets[T - 1]),
        "duration_sec": float(np.sum(durs[r_start:T])),
        "bouts": int(T - r_start),
        "truncated": True
    })

    macro_blocks = []
    curr_m = m_seq[0]
    m_start = 0
    for t in range(1, T):
        if m_seq[t] != curr_m:
            macro_blocks.append({
                "macro_state": int(curr_m),
                "start_sec": float(onsets[m_start]),
                "end_sec": float(offsets[t - 1]),
                "duration_sec": float(np.sum(durs[m_start:t])),
                "bouts": int(t - m_start),
                "mean_certainty": float(np.mean(cert[m_start:t])),
                "truncated": (m_start == 0)
            })
            curr_m = m_seq[t]
            m_start = t

    macro_blocks.append({
        "macro_state": int(curr_m),
        "start_sec": float(onsets[m_start]),
        "end_sec": float(offsets[T - 1]),
        "duration_sec": float(np.sum(durs[m_start:T])),
        "bouts": int(T - m_start),
        "mean_certainty": float(np.mean(cert[m_start:T])),
        "truncated": True
    })

    return {
        "session_idx": s_idx,
        "animal_id": aid,
        "genotype": gt,
        "drug": drug,
        "total_sec": float(offsets[-1]),
        "total_bouts": T,
        "syll_blocks": syll_blocks,
        "rout_blocks": rout_blocks,
        "macro_blocks": macro_blocks,
        "gamma_macro": gamma_macro,
        "certainty": cert,
        "onsets": onsets,
        "offsets": offsets
    }


def draw_tracks_4state(ax, s_data, time_window, show_labels=True):
    t_min, t_max = time_window
    span = t_max - t_min

    # 1. TOP: Macro state (M=4)
    for b in s_data["macro_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        m = b["macro_state"]
        rect = patches.Rectangle(
            (st, 2.15), w, 0.85,
            facecolor=UPPER_COLORS_4.get(m, "#AAA"), edgecolor="black", linewidth=0.8,
            hatch="///" if b["truncated"] else None
        )
        ax.add_patch(rect)
        if w > span * 0.035:
            cert_pct = int(b.get("mean_certainty", 1.0) * 100)
            label = f"U{m+1} ({cert_pct}%)" if span <= 300 else f"U{m+1}"
            ax.text(st + w / 2.0, 2.57, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111")

    # 2. MIDDLE: Routines with solid dividers
    for b in s_data["rout_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        r = b["routine_id"]
        rect = patches.Rectangle(
            (st, 1.05), w, 0.90,
            facecolor=ROUTINE_COLORS.get(r, "#777"), edgecolor="black", linewidth=0.9,
            hatch="///" if b["truncated"] else None
        )
        ax.add_patch(rect)
        if show_labels and w > span * 0.022:
            ax.text(st + w / 2.0, 1.50, f"R{r+1}", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # 3. BOTTOM: Syllables
    for b in s_data["syll_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        s = b["syllable"]
        rect = patches.Rectangle(
            (st, 0.0), w, 0.85,
            facecolor=SYLLABLE_COLORS.get(s, "#CCC"), edgecolor=(0, 0, 0, 0.25), linewidth=0.25
        )
        ax.add_patch(rect)
        if span <= 180 and w > span * 0.02:
            ax.text(st + w / 2.0, 0.42, f"{s}", ha="center", va="center", fontsize=7, color="#222")


def render_4state_visualizations(sessions_data, model):
    # Overview: 30 Minutes (0 to 1,800s)
    n_sess = len(sessions_data)
    fig, axes = plt.subplots(n_sess, 1, figsize=(22, 3.6 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_tracks_4state(ax, s_data, (0, 1800), show_labels=False)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']}, {s_data['drug']})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=80, va="center")
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables", "Routines", "Upper State (M=4)"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(0, 1800)
        ax.set_xticks(np.arange(0, 1801, 300))
        ax.set_xticks(np.arange(0, 1801, 60), minor=True)
        ax.grid(True, which="major", axis="x", alpha=0.4, linestyle="-", color="#777")
        ax.grid(True, which="minor", axis="x", alpha=0.15, linestyle=":")
        if i == 0:
            ax.set_title("30-Minute Hierarchical Timeline with M=4 Upper States (0 to 1,800s / 0.5 Hour)\nBug-Fixed Advance Updates & Joint Continuous Kinematics", fontsize=13, fontweight="bold", pad=15)

    tick_locs = np.arange(0, 1801, 300)
    tick_labels = [f"{s}s\n({s//60}m)" for s in tick_locs]
    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Elapsed Time in Recording: Seconds (Minutes)", fontsize=11, fontweight="bold")

    upper_patches = [patches.Patch(facecolor=UPPER_COLORS_4[u], edgecolor="black", label=f"U{u+1}") for u in range(4)]
    routine_patches = [patches.Patch(facecolor=ROUTINE_COLORS[r], edgecolor="black", label=f"R{r+1}") for r in range(6)]
    trunc_patch = patches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Boundary Truncated")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.28, 0.98), ncol=4, frameon=True, fontsize=9, title="Upper Macro-States (M=4)")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=6, frameon=True, fontsize=9, title="Shared Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    out_overview = OUTPUT_DIR / "fig_hierarchical_timeline_4state_overview.png"
    fig.savefig(out_overview, dpi=300)
    plt.close(fig)
    print(f"✓ Saved 4-State Overview: {out_overview.name}")

    # Detail View: 120 Seconds (600s to 720s) with Posterior Certainty Track
    fig, axes = plt.subplots(n_sess * 2, 1, figsize=(22, 2.5 * n_sess * 2), sharex=True, gridspec_kw={'height_ratios': [3, 1] * n_sess})

    for idx, s_data in enumerate(sessions_data):
        ax_main = axes[idx * 2]
        ax_prob = axes[idx * 2 + 1]

        # Main 3-track
        draw_tracks_4state(ax_main, s_data, (600, 720), show_labels=True)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']})"
        ax_main.set_ylabel(margin_text, fontsize=9, fontweight="bold", rotation=0, labelpad=75, va="center")
        ax_main.set_yticks([0.42, 1.50, 2.57])
        ax_main.set_yticklabels(["Syllables", "Routines", "Upper State"], fontsize=8, fontweight="bold")
        ax_main.set_ylim(-0.15, 3.15)
        ax_main.set_xlim(600, 720)
        ax_main.grid(True, axis="x", alpha=0.35, linestyle="--")

        # Probability ribbon: posterior gamma_t(u) for u=0..3
        t_mask = (s_data["onsets"] >= 590) & (s_data["offsets"] <= 730)
        t_pts = s_data["onsets"][t_mask]
        gamma_pts = s_data["gamma_macro"][t_mask]

        if len(t_pts) > 0:
            for u in range(4):
                ax_prob.plot(t_pts, gamma_pts[:, u], color=UPPER_COLORS_4[u], label=f"P(U{u+1})", linewidth=1.5)

        ax_prob.set_ylabel("Posterior\nProb P(U)", fontsize=8, fontweight="bold", rotation=0, labelpad=75, va="center")
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.set_xlim(600, 720)
        ax_prob.set_yticks([0.0, 0.5, 1.0])
        ax_prob.grid(True, alpha=0.3, linestyle=":")

        if idx == 0:
            ax_main.set_title("120-Second Detail View with Upper-State Posterior Probabilities (600s to 720s)\nResolving Certainty vs. Ambiguity across M=4 States", fontsize=12, fontweight="bold", pad=15)
            ax_prob.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)

    axes[-1].set_xlabel("Elapsed Time in Recording (Seconds)", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0.06, 0.03, 0.98, 0.94])
    out_detail = OUTPUT_DIR / "fig_hierarchical_timeline_4state_detail.png"
    fig.savefig(out_detail, dpi=300)
    plt.close(fig)
    print(f"✓ Saved 4-State Detail with Posteriors: {out_detail.name}")

    import shutil
    shutil.copy(out_overview, ARTIFACT_DIR / out_overview.name)
    shutil.copy(out_detail, ARTIFACT_DIR / out_detail.name)
    print("✓ Copied 4-state figures to artifact directory for inline display.")


def plot_timescale_diagnostic_figure(df_res):
    """Plots controlled timescale sweep: Held-out LL vs. Visit Duration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    conds = df_res["condition"].values
    lls = df_res["held_out_ll_per_bout_mean"].values
    ll_err = df_res["held_out_ll_per_bout_std"].values
    durs = df_res["median_visit_sec"].values
    certs = df_res["mean_posterior_certainty_pct"].values

    x = np.arange(len(conds))

    # Panel 1: Held-out LL
    bars1 = ax1.bar(x, lls, yerr=ll_err, capsize=5, color=["#4A90E2", "#50E3C2", "#F5A623", "#D0021B"], edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds, fontsize=9, fontweight="bold", rotation=15)
    ax1.set_ylabel("Held-Out Log-Likelihood / Bout", fontsize=10, fontweight="bold")
    ax1.set_title("A. Predictive Generalization on Unseen Sessions\n(Higher is Better)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3, linestyle=":")

    for b, ll in zip(bars1, lls):
        ax1.text(b.get_x() + b.get_width()/2, ll - 0.04, f"{ll:.3f}", ha="center", va="top", fontsize=9, fontweight="bold", color="black")

    # Panel 2: Visit Duration & Posterior Certainty
    ax2_cert = ax2.twinx()
    b2 = ax2.bar(x - 0.2, durs, width=0.4, color="#34495E", edgecolor="black", label="Median Visit Duration (s)")
    l2 = ax2_cert.plot(x + 0.2, certs, color="#E74C3C", marker="o", linewidth=2.5, markersize=8, label="Mean Posterior Certainty (%)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(conds, fontsize=9, fontweight="bold", rotation=15)
    ax2.set_ylabel("Median Upper-Visit Duration (s)", fontsize=10, fontweight="bold", color="#34495E")
    ax2_cert.set_ylabel("Mean Posterior Certainty (%)", fontsize=10, fontweight="bold", color="#E74C3C")
    ax2.set_title("B. Upper-State Timescale & Posterior Certainty", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")

    for b, d in zip(b2, durs):
        ax2.text(b.get_x() + b.get_width()/2, d + 5, f"{d:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for px, pc in zip(x + 0.2, certs):
        ax2_cert.text(px, pc + 1.0, f"{pc:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#E74C3C")

    plt.tight_layout()
    out_diag = OUTPUT_DIR / "fig_controlled_timescale_comparison.png"
    fig.savefig(out_diag, dpi=300)
    plt.close(fig)
    print(f"✓ Saved Timescale Diagnostic Figure: {out_diag.name}")

    import shutil
    shutil.copy(out_diag, ARTIFACT_DIR / out_diag.name)


def main():
    print("=" * 80)
    print("STARTING CONTROLLED UPPER-TIMESCALE EXPERIMENT (M=4, BUG-FIXED, 30 MIN)")
    print("=" * 80)

    sessions, _ = load_30min_cohort(max_duration_sec=1800.0)
    print(f"Loaded {len(sessions)} sessions (total bouts: {sum(s['total_bouts'] for s in sessions)}).")

    # Step 1: Run 5-fold CV sweep across timescale conditions
    df_res = run_controlled_timescale_sweep(sessions)
    plot_timescale_diagnostic_figure(df_res)

    # Step 2: Fit master M=4 model with free A_uu on full cohort
    print("\nFitting Master Model with M=4 Upper States on Full Cohort...")
    master_model = ControlledHierarchicalModel(M=4, K=6, S=3, n_features=2, random_seed=42)
    master_model.fit_multi_sequence_joint(sessions, max_iters=25)

    print("\nLearned Upper Transition Matrix A_macro (4 x 4):")
    print(np.round(master_model.A_macro, 4))

    print("\nLearned Routine Selection Matrix B_routine (4 x 6):")
    print(np.round(master_model.B_routine, 4))

    print("\nBug-Fixed p_advance (Learned Expected Bouts per Routine):")
    for k in range(6):
        p = master_model.p_advance[k]
        exp_bouts = 3.0 / p
        print(f"  R{k+1}: p_advance = {p:.4f} -> Expected Dwell = {exp_bouts:.1f} bouts")

    # Step 3: Extract exemplar tracks and render 4-state figures
    exemplar_indices = [1, 2, 4, 0]
    print(f"\nExtracting tracks for exemplars: {exemplar_indices}...")
    sessions_data = []
    for idx in exemplar_indices:
        s_data = extract_tracks_with_posteriors(sessions[idx], master_model)
        sessions_data.append(s_data)

    print("\nRendering 4-State Visualizations...")
    render_4state_visualizations(sessions_data, master_model)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTAL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
