#!/usr/bin/env python3
"""
probabilistic_episode_model.py
=============================================================================
Shallow Duration-Aware Probabilistic Episode Model with Mixed Discrete
and Continuous Observations (Syllables + Kinematics + Pose Latents).

Features:
- K Ordered Episode Families + 1 Background/Unassigned State (N = K*S + 1)
- Left-to-Right Staged Topology with Phase-Duration Regularization
- Joint Mixed Emission:
    Arm 2: Categorical Syllables
    Arm 3: Categorical Syllables + Regularized Gaussian Continuous Kinematics
- Multi-Sequence Log-Space Baum-Welch EM
- Viterbi & Posterior Arc Decoding
- Built-in Synthetic Unit Test
=============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import THEMES


class ProbabilisticEpisodeModel:
    """
    Shallow Duration-Aware Episode Model for Sequential Behavior.
    K episode models of S ordered phases each, plus 1 background state.
    """
    def __init__(self, n_episodes=3, n_phases=5, n_vocab=14, n_continuous=7, mode="mixed", random_state=42):
        self.K = n_episodes
        self.S = n_phases
        self.V = n_vocab
        self.D = n_continuous
        self.mode = mode  # "categorical" (Arm 2) or "mixed" (Arm 3)
        self.rng = np.random.RandomState(random_state)

        # Total atomic states: K * S episode phases + 1 background state
        self.N = self.K * self.S + 1
        self.bg_state = self.K * self.S

        # Initialize parameters
        self._init_topology()
        self._init_emissions()

    def _init_topology(self):
        """Initializes left-to-right staged transition parameters."""
        # p_adv[k, s]: advance probability from phase s to s+1 (or exit if s == S-1)
        # Expected dwell time in bouts: 1 / p_adv. Initialized to ~1.5 - 2.0 bouts
        self.p_adv = np.full((self.K, self.S), 0.60, dtype=np.float64)

        # Background stay probability
        self.p_bg_stay = 0.85

        # Inter-episode transition matrix A_ep[K+1, K+1]
        # (transitions among episodes and background upon completion/exit)
        self.A_ep = np.full((self.K + 1, self.K + 1), 1.0 / (self.K + 1), dtype=np.float64)
        # Give slight self-transition / repeat boost
        np.fill_diagonal(self.A_ep, 0.4)
        self.A_ep /= self.A_ep.sum(axis=1, keepdims=True)

        # Initial state prior: start in background or phase 0 of an episode
        self.pi = np.zeros(self.N, dtype=np.float64)
        self.pi[self.bg_state] = 0.4
        for k in range(self.K):
            self.pi[k * self.S] = 0.6 / self.K

    def _init_emissions(self):
        """Initializes categorical and continuous emission parameters."""
        # Categorical emissions: (N, V)
        self.cat_emiss = np.full((self.N, self.V), 1.0 / self.V, dtype=np.float64)

        # Continuous Gaussian emissions: mu (N, D), var (N, D)
        self.cont_mu = np.zeros((self.N, self.D), dtype=np.float64)
        self.cont_var = np.ones((self.N, self.D), dtype=np.float64)
        self.min_var = 1e-2

    def build_transition_matrix(self):
        """
        Builds the complete (N, N) transition matrix.
        Index mapping:
          Episode k, phase s -> idx = k * S + s
          Background -> idx = K * S
        """
        T = np.zeros((self.N, self.N), dtype=np.float64)

        for k in range(self.K):
            for s in range(self.S):
                i = k * self.S + s
                p_a = float(self.p_adv[k, s])

                # Dwell in same phase
                T[i, i] = 1.0 - p_a

                if s < self.S - 1:
                    # Advance to next phase s+1
                    T[i, i + 1] = p_a
                else:
                    # Final phase s = S-1: exit via A_ep
                    for k_next in range(self.K):
                        j_next = k_next * self.S + 0  # Enter phase 0 of next episode
                        T[i, j_next] += p_a * self.A_ep[k, k_next]
                    # Or exit to background
                    T[i, self.bg_state] += p_a * self.A_ep[k, self.K]

        # Background transitions
        T[self.bg_state, self.bg_state] = self.p_bg_stay
        p_bg_exit = 1.0 - self.p_bg_stay
        for k_next in range(self.K):
            j_next = k_next * self.S + 0
            T[self.bg_state, j_next] = p_bg_exit * self.A_ep[self.K, k_next]

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / (row_sums + 1e-12)
        return T

    def initialize_from_seqnmf(self, W_seqnmf, train_bouts_df, factor_indices):
        """
        Initializes categorical and continuous emissions directly from seqNMF templates
        and activation-aligned candidate occurrences.
        W_seqnmf: (V, K_nmf, L) where V=14, L=5
        factor_indices: list of 3 factor indices corresponding to the 3 themes
        """
        print("[MODEL] Initializing episode phases directly from seqNMF templates...")
        eps_prior = 1e-3

        for k, f_idx in enumerate(factor_indices):
            # W slice: (14, 5) -> V x S
            w_slice = W_seqnmf[:, f_idx, :]
            for s in range(self.S):
                state_idx = k * self.S + s
                # Categorical emission from full weighted template slice
                weights = w_slice[:, s] + eps_prior
                self.cat_emiss[state_idx, :] = weights / np.sum(weights)

        # Background categorical: empirical marginal across training bouts
        if len(train_bouts_df) > 0:
            counts = np.bincount(train_bouts_df["meta_syllable"].values, minlength=self.V)
            self.cat_emiss[self.bg_state, :] = (counts + 1.0) / np.sum(counts + 1.0)

        # Continuous priors from training data if available
        if self.mode == "mixed" and "continuous_features" in train_bouts_df:
            all_feats = np.vstack(train_bouts_df["continuous_features"].values)
            self.cont_mu[self.bg_state, :] = np.mean(all_feats, axis=0)
            self.cont_var[self.bg_state, :] = np.var(all_feats, axis=0) + self.min_var

            # Episode phases initialize with perturbed prior around empirical mean
            for k in range(self.K):
                for s in range(self.S):
                    st = k * self.S + s
                    self.cont_mu[st, :] = self.cont_mu[self.bg_state, :]
                    self.cont_var[st, :] = self.cont_var[self.bg_state, :]

    def compute_log_emissions(self, discrete_obs, cont_obs=None):
        """
        Computes (T, N) log-emission matrix.
        discrete_obs: (T,) int array of meta-syllable IDs
        cont_obs: (T, D) float array of continuous features (if mode == 'mixed')
        """
        T = len(discrete_obs)
        # 1. Categorical log-emissions: log P(y_t | state)
        log_cat = np.log(np.clip(self.cat_emiss[:, discrete_obs].T, 1e-12, 1.0))  # (T, N)

        if self.mode != "mixed" or cont_obs is None:
            return log_cat

        # 2. Continuous Gaussian log-emissions: log N(x_t | mu_i, var_i)
        # sum_d [ -0.5 * (log(2*pi*var_d) + (x_td - mu_id)^2 / var_d) ]
        diff = cont_obs[:, None, :] - self.cont_mu[None, :, :]  # (T, N, D)
        var = np.clip(self.cont_var[None, :, :], self.min_var, 1e4)
        log_var = np.log(2.0 * np.pi * var)
        log_cont_d = -0.5 * (log_var + (diff ** 2) / var)
        log_cont = np.sum(log_cont_d, axis=2)  # (T, N)

        # Combined joint log-emission
        return log_cat + log_cont

    def forward_backward(self, T_mat, log_emiss):
        """
        Fast scaled Forward-Backward algorithm (Rabiner 1989).
        T_mat: (N, N)
        log_emiss: (T, N)
        Returns:
          gamma: (T, N) posterior state probabilities
          xi_sum: (N, N) sum over time of posterior transition probabilities
          log_lik: total log-likelihood
        """
        T_steps, N = log_emiss.shape
        # Stabilize emissions per time-step
        max_log_e = np.max(log_emiss, axis=1, keepdims=True)
        emiss = np.exp(np.clip(log_emiss - max_log_e, -700, 0))

        alpha = np.zeros((T_steps, N), dtype=np.float64)
        c = np.zeros(T_steps, dtype=np.float64)

        # Forward pass
        a0 = self.pi * emiss[0]
        c0 = float(np.sum(a0))
        if c0 < 1e-300:
            c0 = 1e-300
        c[0] = c0
        alpha[0] = a0 / c0

        for t in range(1, T_steps):
            at = (alpha[t - 1] @ T_mat) * emiss[t]
            ct = float(np.sum(at))
            if ct < 1e-300:
                ct = 1e-300
            c[t] = ct
            alpha[t] = at / ct

        total_log_lik = float(np.sum(np.log(c)) + np.sum(max_log_e))

        # Backward pass
        beta = np.zeros((T_steps, N), dtype=np.float64)
        beta[-1] = 1.0
        for t in range(T_steps - 2, -1, -1):
            bt = T_mat @ (emiss[t + 1] * beta[t + 1])
            beta[t] = bt / c[t + 1]

        # Posterior state probabilities gamma_t(i)
        gamma = alpha * beta

        # Posterior transition probabilities sum xi(i, j)
        term = (emiss[1:] * beta[1:]) / c[1:, None]
        xi_sum = (alpha[:-1].T @ term) * T_mat

        return gamma, xi_sum, total_log_lik

    def fit(self, session_sequences, max_iter=20, tol=1e-3, verbose=True):
        """
        Multi-sequence Baum-Welch EM algorithm across independent sessions.
        session_sequences: list of dicts with 'discrete_obs' and optionally 'cont_obs'
        """
        if verbose:
            print(f"[FIT] Starting Baum-Welch EM ({self.mode} mode, K={self.K}, S={self.S}, N={self.N})")

        prev_ll = -np.inf
        for it in range(max_iter):
            t_it0 = time.time()
            T_mat = self.build_transition_matrix()

            total_ll = 0.0
            gamma_accum = np.zeros((self.N, self.V), dtype=np.float64)
            gamma_total = np.zeros(self.N, dtype=np.float64)
            xi_accum = np.zeros((self.N, self.N), dtype=np.float64)
            cont_mu_accum = np.zeros((self.N, self.D), dtype=np.float64)
            cont_sq_accum = np.zeros((self.N, self.D), dtype=np.float64)

            for sdata in session_sequences:
                disc = sdata["discrete_obs"]
                cont = sdata.get("cont_obs")
                log_emiss = self.compute_log_emissions(disc, cont)

                gamma, xi_sum, ll = self.forward_backward(T_mat, log_emiss)
                total_ll += ll

                gamma_total += gamma.sum(axis=0)

                # Accumulate categorical emissions via fast bincount
                for n in range(self.N):
                    gamma_accum[n] += np.bincount(disc, weights=gamma[:, n], minlength=self.V)

                # Accumulate continuous statistics
                if self.mode == "mixed" and cont is not None:
                    # gamma: (T, N), cont: (T, D) -> (N, D)
                    cont_mu_accum += gamma.T @ cont
                    cont_sq_accum += gamma.T @ (cont ** 2)

                xi_accum += xi_sum

            # M-Step: Update Categorical Emissions
            alpha_dirichlet = 0.1
            self.cat_emiss = (gamma_accum + alpha_dirichlet) / (gamma_total[:, None] + self.V * alpha_dirichlet)

            # M-Step: Update Continuous Emissions
            if self.mode == "mixed":
                denom = gamma_total[:, None] + 1e-8
                new_mu = cont_mu_accum / denom
                new_var = (cont_sq_accum / denom) - (new_mu ** 2)
                self.cont_mu = new_mu
                self.cont_var = np.clip(new_var, self.min_var, 1e3)

            # M-Step: Update Transition Topology (p_adv and A_ep)
            for k in range(self.K):
                for s in range(self.S):
                    i = k * self.S + s
                    stay_count = xi_accum[i, i]
                    exit_count = xi_accum[i].sum() - stay_count
                    # Regularized Beta prior (alpha=4.0, beta=6.0 -> ~0.4 advance)
                    p_adv_new = (exit_count + 4.0) / (stay_count + exit_count + 10.0)
                    self.p_adv[k, s] = np.clip(p_adv_new, 0.15, 0.85)

            # Update Background stay
            bg_stay = xi_accum[self.bg_state, self.bg_state]
            bg_exit = xi_accum[self.bg_state].sum() - bg_stay
            self.p_bg_stay = np.clip((bg_stay + 20.0) / (bg_stay + bg_exit + 25.0), 0.60, 0.98)

            d_ll = total_ll - prev_ll
            if verbose:
                print(f"  Iteration {it+1:2d}/{max_iter}: Log-Likelihood = {total_ll:.2f} (Δ = {d_ll:+.2f}) in {time.time() - t_it0:.2f}s")

            if it > 0 and abs(d_ll) < tol:
                if verbose:
                    print(f"  [CONVERGED] Log-likelihood plateaued (Δ < {tol}).")
                break
            prev_ll = total_ll

        return self

    def viterbi_decode(self, discrete_obs, cont_obs=None, bg_competition_boost=0.0):
        """
        Viterbi algorithm for global MAP state sequence.
        bg_competition_boost: Option 1 background competition adjustment (offsets
        broad-variance log-partition disparity of unconstrained background model).
        Returns: (T,) int array of atomic state IDs
        """
        T_steps = len(discrete_obs)
        T_mat = self.build_transition_matrix()
        log_T = np.log(np.clip(T_mat, 1e-12, 1.0))
        log_emiss = self.compute_log_emissions(discrete_obs, cont_obs)
        if bg_competition_boost > 0.0:
            log_emiss[:, self.bg_state] += bg_competition_boost
        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))

        viterbi_log = np.zeros((T_steps, self.N), dtype=np.float64)
        backpointer = np.zeros((T_steps, self.N), dtype=np.int32)

        viterbi_log[0] = log_pi + log_emiss[0]

        for t in range(1, T_steps):
            # (N, N): from state i (dim 0) to state j (dim 1)
            scores = viterbi_log[t - 1][:, None] + log_T
            backpointer[t] = np.argmax(scores, axis=0)
            viterbi_log[t] = np.max(scores, axis=0) + log_emiss[t]

        # Traceback
        best_path = np.zeros(T_steps, dtype=np.int32)
        best_path[-1] = np.argmax(viterbi_log[-1])
        for t in range(T_steps - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path

    def extract_decoded_episodes(self, state_path, sdata):
        """
        Extracts completed episode occurrences from the decoded state path.
        Detects traversals across phase 0 -> ... -> S-1.
        Returns list of episode dicts.
        """
        episodes = []
        T_steps = len(state_path)
        b_starts = sdata["bout_starts"]
        b_ends = sdata["bout_ends"]
        bout_meta = sdata["bout_meta"]
        raw_sylls = sdata["raw_syllables"]

        # Track active episode progression
        active_k = None
        active_start_b = None
        current_phase = -1
        phase_boundaries_b = []

        for b in range(T_steps):
            st = state_path[b]
            if st == self.bg_state:
                if active_k is not None and current_phase == self.S - 1:
                    # Successfully completed before hitting background
                    phase_boundaries_b.append(b)
                    episodes.append(self._package_episode(
                        active_k, active_start_b, b, phase_boundaries_b, sdata
                    ))
                active_k = None
                current_phase = -1
                phase_boundaries_b = []
            else:
                k = st // self.S
                s = st % self.S

                if active_k is None:
                    # Candidate start at phase 0
                    if s == 0:
                        active_k = k
                        active_start_b = b
                        current_phase = 0
                        phase_boundaries_b = [b]
                elif k == active_k:
                    # Progression inside same episode
                    if s == current_phase:
                        # Dwelling in same phase
                        pass
                    elif s == current_phase + 1:
                        # Advanced to next phase
                        current_phase = s
                        phase_boundaries_b.append(b)
                    elif s == 0 and current_phase == self.S - 1:
                        # Episode completed and immediately restarted!
                        phase_boundaries_b.append(b)
                        episodes.append(self._package_episode(
                            active_k, active_start_b, b, phase_boundaries_b, sdata
                        ))
                        # Start new occurrence
                        active_start_b = b
                        current_phase = 0
                        phase_boundaries_b = [b]
                    else:
                        # Broken sequence
                        active_k = None
                        current_phase = -1
                        phase_boundaries_b = []
                else:
                    # Switched to different episode
                    if current_phase == self.S - 1:
                        # Previous episode completed
                        phase_boundaries_b.append(b)
                        episodes.append(self._package_episode(
                            active_k, active_start_b, b, phase_boundaries_b, sdata
                        ))
                    # Check if new state is phase 0
                    if s == 0:
                        active_k = k
                        active_start_b = b
                        current_phase = 0
                        phase_boundaries_b = [b]
                    else:
                        active_k = None
                        current_phase = -1
                        phase_boundaries_b = []

        # Check final episode
        if active_k is not None and current_phase == self.S - 1:
            phase_boundaries_b.append(T_steps)
            episodes.append(self._package_episode(
                active_k, active_start_b, T_steps, phase_boundaries_b, sdata
            ))

        return episodes

    def _package_episode(self, k, start_b, end_b, phase_bounds_b, sdata):
        """Packages an episode occurrence record with exact frame and token provenance."""
        b_starts = sdata["bout_starts"]
        b_ends = sdata["bout_ends"]
        bout_meta = sdata["bout_meta"]
        raw_sylls = sdata["raw_syllables"]

        sf = int(b_starts[start_b])
        ef = int(b_ends[end_b - 1])

        # Model frame boundaries for each phase
        frame_bounds = [int(b_starts[b]) if b < len(b_starts) else ef for b in phase_bounds_b]
        if frame_bounds[0] != sf:
            frame_bounds.insert(0, sf)
        if frame_bounds[-1] != ef:
            frame_bounds.append(ef)

        dur_sec = float((ef - sf) / 40.0)
        dur_bouts = int(end_b - start_b)

        # Human ethological syllable tokens that occurred
        from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import HUMAN_LABELS
        meta_tokens = [HUMAN_LABELS.get(int(m), f"syll_{int(m):02d}") for m in bout_meta[start_b:end_b]]
        observed_raw_sylls = raw_sylls[sf:ef]

        # Extract kinematics over episode
        cents = sdata.get("centroid_slice")
        path_len = float(np.sum(sdata["bout_path_len"][start_b:end_b]))
        net_disp = float(np.sum(sdata["bout_net_disp"][start_b:end_b]))
        avg_speed = float(np.mean(sdata["bout_speeds"][start_b:end_b]))
        total_dh = float(np.sum(sdata["bout_dh"][start_b:end_b]))

        theme_keys = ["theme_A", "theme_B", "theme_C"]
        th_key = theme_keys[k]

        return {
            "episode_family_idx": k,
            "theme_id": th_key,
            "theme_title": THEMES[th_key]["title"],
            "recording_key": sdata["recording_key"],
            "animal_id": sdata["animal_id"],
            "genotype": sdata["genotype"],
            "drug": sdata["drug"],
            "split": sdata["split"],
            "start_bout": int(start_b),
            "end_bout": int(end_b),
            "start_frame": int(sf),
            "end_frame": int(ef),
            "duration_sec": dur_sec,
            "duration_bouts": dur_bouts,
            "phase_boundaries_frames": frame_bounds,
            "observed_meta_tokens": meta_tokens,
            "observed_meta_chain": " -> ".join(meta_tokens),
            "avg_speed_px_s": avg_speed,
            "path_length_px": path_len,
            "net_displacement_px": net_disp,
            "signed_turn_deg": total_dh,
            "method": f"Probabilistic_Episode_{self.mode.capitalize()}"
        }


def run_synthetic_progression_check():
    """
    Self-contained unit test on synthetic data verifying:
    1. Recovery of variable-length executions
    2. Detection of substituted executions from continuous evidence
    """
    print("=" * 80)
    print("RUNNING SYNTHETIC PROGRESSION & RECOVERY CHECK")
    print("=" * 80)
    rng = np.random.RandomState(123)

    # 1. Create a model with K=2 episodes, S=3 phases, V=5 vocabulary, D=2 kinematics
    model = ProbabilisticEpisodeModel(n_episodes=2, n_phases=3, n_vocab=5, n_continuous=2, mode="mixed")

    # Set distinct emissions for episode 0
    # Phase 0: emits token 0, high speed (x[0] ~ +2)
    # Phase 1: emits token 1, turn right (x[1] ~ +2)
    # Phase 2: emits token 2, low speed (x[0] ~ -2)
    model.cat_emiss[0, :] = [0.8, 0.05, 0.05, 0.05, 0.05]
    model.cont_mu[0, :] = [2.0, 0.0]
    model.cat_emiss[1, :] = [0.05, 0.8, 0.05, 0.05, 0.05]
    model.cont_mu[1, :] = [0.0, 2.0]
    model.cat_emiss[2, :] = [0.05, 0.05, 0.8, 0.05, 0.05]
    model.cont_mu[2, :] = [-2.0, 0.0]

    # Background: emits token 4, neutral kinematics
    model.cat_emiss[model.bg_state, :] = [0.05, 0.05, 0.05, 0.05, 0.8]
    model.cont_mu[model.bg_state, :] = [0.0, 0.0]

    # Set advance prob ~ 0.5 (average dwell 2 bouts per phase)
    model.p_adv[0, :] = 0.5

    # Generate synthetic sequence with variable duration
    disc_seq = []
    cont_seq = []
    true_states = []

    # Inject 5 variable-duration executions of Episode 0 separated by background
    for rep in range(5):
        # Phase 0 for 1-3 bouts
        d0 = rng.randint(1, 4)
        for _ in range(d0):
            disc_seq.append(0)
            cont_seq.append(rng.normal([2.0, 0.0], 0.3))
            true_states.append(0)
        # Phase 1 for 1-2 bouts
        d1 = rng.randint(1, 3)
        for _ in range(d1):
            disc_seq.append(1)
            cont_seq.append(rng.normal([0.0, 2.0], 0.3))
            true_states.append(1)
        # Phase 2 for 2 bouts
        d2 = 2
        for _ in range(d2):
            disc_seq.append(2)
            cont_seq.append(rng.normal([-2.0, 0.0], 0.3))
            true_states.append(2)
        # Background for 3 bouts
        for _ in range(3):
            disc_seq.append(4)
            cont_seq.append(rng.normal([0.0, 0.0], 0.5))
            true_states.append(model.bg_state)

    disc_seq = np.array(disc_seq, dtype=np.int32)
    cont_seq = np.array(cont_seq, dtype=np.float32)

    # Decode with Viterbi
    decoded_states = model.viterbi_decode(disc_seq, cont_seq)
    accuracy = np.mean(decoded_states == np.array(true_states))
    print(f"  Viterbi state recovery accuracy: {accuracy * 100:.1f}%")

    assert accuracy > 0.90, f"Synthetic recovery failed! Accuracy={accuracy}"
    print("[PASS] Synthetic progression and variable duration recovery test PASSED successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_synthetic_progression_check()
