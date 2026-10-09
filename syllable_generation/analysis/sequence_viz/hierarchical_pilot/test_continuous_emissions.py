#!/usr/bin/env python3
"""
test_continuous_emissions.py
Tests ChatGPT's suggestion: Joint Categorical + Continuous Kinematic Emissions
P(y_t, f_t | k, s) = P(y_t | k, s) * Normal(f_t | mu_{k,s}, sigma_{k,s}^2)
where f_t = [log(duration_sec), log(mean_vel + 1.0)]
"""

import sys
import time
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.hierarchical_pilot.hierarchical_subroutine_model import (
    HierarchicalSubroutineModel, jax_forward_backward
)

DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"


def load_cohort_with_kinematics(max_bouts_per_session=1500):
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

        sub_df = df.head(max_bouts_per_session)
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
            "total_bouts": len(sylls)
        })

    # Standardization parameters across entire cohort
    dur_mean = np.mean(all_log_dur)
    dur_std = np.std(all_log_dur)
    vel_mean = np.mean(all_log_vel)
    vel_std = np.std(all_log_vel)

    for sess in session_list:
        norm_dur = (sess["log_dur"] - dur_mean) / dur_std
        norm_vel = (sess["log_vel"] - vel_mean) / vel_std
        sess["features"] = np.column_stack([norm_dur, norm_vel])  # (T, 2)

    return session_list, (dur_mean, dur_std, vel_mean, vel_std)


class ContinuousHierarchicalModel(HierarchicalSubroutineModel):
    """
    Extends HierarchicalSubroutineModel with Gaussian emissions for continuous kinematics:
    f_t = [z_log_duration, z_log_velocity]
    """
    def __init__(self, n_features=2, min_variance=0.05, **kwargs):
        self.n_features = n_features
        self.min_variance = min_variance
        self.mu_feat = None
        self.var_feat = None
        super().__init__(**kwargs)

    def _initialize_parameters(self):
        super()._initialize_parameters()
        rng = np.random.default_rng(self.random_seed)
        self.mu_feat = rng.normal(0.0, 0.2, size=(self.K, self.S, self.n_features)).astype(np.float32)
        self.var_feat = np.ones((self.K, self.S, self.n_features), dtype=np.float32)

    def _build_transition_matrix(self):
        from syllable_generation.analysis.sequence_viz.hierarchical_pilot.hierarchical_subroutine_model import build_transition_matrix_np
        return build_transition_matrix_np(self.M, self.K, self.S, self.p_advance, self.A_macro, self.B_routine)

    def get_log_emission_seq_joint(self, syllables, features):
        T = len(syllables)
        # 1. Categorical emission: (K, S, T) -> transpose to (T, K, S)
        log_theta = np.log(np.clip(self.theta, 1e-12, 1.0))
        cat_log_emiss = log_theta[:, :, syllables].transpose(2, 0, 1)  # (T, K, S)

        # 2. Gaussian emission: (T, K, S)
        f_exp = features[:, None, None, :]
        mu_exp = self.mu_feat[None, :, :, :]
        var_exp = self.var_feat[None, :, :, :]

        diff_sq = (f_exp - mu_exp) ** 2
        gauss_log = -0.5 * np.sum(np.log(2.0 * np.pi * var_exp) + diff_sq / var_exp, axis=-1)  # (T, K, S)

        # Joint log emission
        joint_ks = cat_log_emiss + gauss_log  # (T, K, S)

        # Tile across M macro-states: (T, M, K, S) -> flatten to (T, N_states)
        joint_all = np.repeat(joint_ks[:, None, :, :], self.M, axis=1)
        return joint_all.reshape(T, self.N_states)

    def fit_multi_sequence_joint(self, session_list, max_iters=25, tol=1e-4):
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

            for sess in session_list:
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
                gamma_ks = np.sum(gamma_mks, axis=1)  # (T, K, S)

                for t_idx, y in enumerate(sylls):
                    accum_theta_counts[:, :, y] += gamma_ks[t_idx]

                for k in range(self.K):
                    for s in range(self.S):
                        w = gamma_ks[:, k, s]  # (T,)
                        accum_feat_w[k, s] += np.sum(w)
                        accum_feat_sum[k, s] += np.sum(w[:, None] * feats, axis=0)
                        accum_feat_sq[k, s] += np.sum(w[:, None] * (feats ** 2), axis=0)

            log_history.append(total_cohort_log_lik)

            # M-step
            for k in range(self.K):
                n_self = 0.0
                n_adv = 0.0
                for s in range(self.S - 1):
                    for m in range(self.M):
                        curr_idx = self.tuple_to_state(m, k, s)
                        self_idx = curr_idx
                        adv_idx = self.tuple_to_state(m, k, s + 1)
                        n_self += accum_xi[curr_idx, self_idx]
                        n_adv += accum_xi[curr_idx, adv_idx]

                a = self.alpha_p + n_adv
                b = self.beta_p + n_self
                self.p_advance[k] = float(np.clip(a / (a + b), 0.05, 0.95))

            pseudocount = 0.1
            numer_theta = accum_theta_counts + pseudocount
            denom_theta = np.sum(numer_theta, axis=-1, keepdims=True)
            self.theta = numer_theta / denom_theta

            denom_w = np.maximum(accum_feat_w[:, :, None], 1e-6)
            new_mu = accum_feat_sum / denom_w
            new_var = (accum_feat_sq / denom_w) - (new_mu ** 2)
            self.mu_feat = new_mu
            self.var_feat = np.maximum(new_var, self.min_variance)

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

            prior_A = 2.0
            num_A = trans_exit + prior_A
            self.A_macro = num_A / np.sum(num_A, axis=1, keepdims=True)

            prior_B = 1.0
            num_B = routine_sel + prior_B
            self.B_routine = num_B / np.sum(num_B, axis=1, keepdims=True)

            self.T_mat = self._build_transition_matrix()

            if it > 0 and abs(log_history[-1] - log_history[-2]) < tol:
                break

        return log_history

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


def main():
    print("Loading cohort data with kinematics...")
    sessions, params = load_cohort_with_kinematics(max_bouts_per_session=1500)
    print(f"Loaded {len(sessions)} sessions. Standardization: log_dur mean={params[0]:.2f}, std={params[1]:.2f}; log_vel mean={params[2]:.2f}, std={params[3]:.2f}")

    print("\nFitting Joint Categorical + Continuous Kinematics Model (25 iterations)...")
    model_cont = ContinuousHierarchicalModel(M=3, K=6, S=3, n_features=2, random_seed=42)
    history = model_cont.fit_multi_sequence_joint(sessions, max_iters=25)

    print("\n" + "=" * 60)
    print("CONTINUOUS KINEMATICS MODEL CONVERGENCE")
    print("=" * 60)
    print(f"Initial LL: {history[0]:.1f} -> Final LL: {history[-1]:.1f} (delta = {history[-1] - history[0]:+.1f})")

    print("\nUpper Transition Matrix A_macro (with Continuous Kinematics):")
    print(np.round(model_cont.A_macro, 4))

    print("\nRoutine Selection Matrix B_routine (M x K):")
    print(np.round(model_cont.B_routine, 4))

    print("\nLearned Kinematics (Z-scores of [log_dur, log_vel] by routine stage):")
    for k in range(6):
        durs = model_cont.mu_feat[k, :, 0]
        vels = model_cont.mu_feat[k, :, 1]
        print(f"  R{k+1}: Duration Z=[{durs[0]:+.2f}, {durs[1]:+.2f}, {durs[2]:+.2f}] | Velocity Z=[{vels[0]:+.2f}, {vels[1]:+.2f}, {vels[2]:+.2f}]")

    print("\nViterbi Macro-State Occupancy Across Entire Cohort (30 Sessions):")
    m_counts = {0: 0, 1: 0, 2: 0}
    total_b = 0
    for sess in sessions:
        m_seq, _, _ = model_cont.decode_session_viterbi_joint(sess["syllables"], sess["features"])
        for m in [0, 1, 2]:
            m_counts[m] += int(np.sum(m_seq == m))
        total_b += len(m_seq)

    for m in range(3):
        pct = (m_counts[m] / total_b) * 100
        print(f"  U{m+1}: {m_counts[m]:6d} bouts ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
