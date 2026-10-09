#!/usr/bin/env python3
"""
hierarchical_subroutine_model.py
=============================================================================
Hierarchical Shared-Subroutine Behavioral Model (Multi-Sequence & Tied/Free EM).

Architecture:
- M = 3 Upper States (Macro-Regimes)
- K = 6 Shared Routines, each with S = 3 internal stages
- Total atomic states: N = M * K * S = 54
- Transition rules:
  * Stage s in {0, 1}: Stay (1 - p_k) or Advance (p_k) to s+1 within same (u, k).
  * Stage s = 2: Stay (1 - p_k) or Exit (p_k).
  * Upon exit: Upper state transitions u -> u' via A_macro, new routine k'
    is selected via B_routine_{u', k'}, entering stage s' = 0.
- Duration prior: p_k ~ Beta(alpha_p, beta_p)
  D_k = 3 + NegBin(3, p_k), E[D_k | p_k] = 3 / p_k.
- Syllable emission theta_{k, s} in Delta^{99}:
  * Free stage emissions: theta_{k, s} independent across stages (chains/progressions).
  * Tied stage emissions: theta_{k, 0} = theta_{k, 1} = theta_{k, 2} = theta_k (duration clusters).
- Multi-Sequence Baum-Welch: Independent forward-backward per recording session,
  accumulating sufficient statistics without artificial cross-session boundary joins.
- JAX GPU acceleration with log-space forward-backward.
=============================================================================
"""

import time

import numpy as np

# JAX GPU
HAS_JAX = False
try:
    import jax
    import jax.numpy as jnp
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) > 0:
        HAS_JAX = True
except Exception:
    HAS_JAX = False


def build_transition_matrix_np(M, K, S, p_advance, A_macro, B_routine):
    """
    Builds the 54x54 transition matrix T.
    Index mapping: state i = u * (K * S) + k * S + s
    """
    N = M * K * S
    T_mat = np.zeros((N, N), dtype=np.float32)

    for u in range(M):
        for k in range(K):
            p_k = float(p_advance[k])
            for s in range(S):
                i = u * (K * S) + k * S + s
                if s < S - 1:
                    T_mat[i, i] = 1.0 - p_k
                    j_adv = u * (K * S) + k * S + (s + 1)
                    T_mat[i, j_adv] = p_k
                else:
                    # Final stage s = S - 1 (s = 2)
                    T_mat[i, i] = 1.0 - p_k
                    for u_next in range(M):
                        p_u = A_macro[u, u_next]
                        for k_next in range(K):
                            p_rout = B_routine[u_next, k_next]
                            j_exit = u_next * (K * S) + k_next * S + 0
                            T_mat[i, j_exit] += p_k * p_u * p_rout

    row_sums = T_mat.sum(axis=1, keepdims=True)
    T_mat = T_mat / (row_sums + 1e-12)
    return T_mat


if HAS_JAX:
    @jax.jit
    def jax_forward_backward(log_trans, log_emiss_seq, init_log_pi):
        """
        Log-space Forward-Backward algorithm using JAX.
        log_trans: (N, N)
        log_emiss_seq: (T, N)
        init_log_pi: (N,)
        Returns:
          log_gamma: (T, N) state log-posteriors
          log_xi_sum: (N, N) sum over t of log transition posteriors
          total_log_lik: scalar
        """
        T, N = log_emiss_seq.shape

        # Forward pass
        def fwd_step(log_alpha_prev, log_emiss_t):
            m = log_alpha_prev[:, None] + log_trans
            log_alpha_pred = jax.nn.logsumexp(m, axis=0)
            log_alpha_t = log_alpha_pred + log_emiss_t
            return log_alpha_t, log_alpha_t

        init_alpha = init_log_pi + log_emiss_seq[0]
        _, log_alpha_rest = jax.lax.scan(fwd_step, init_alpha, log_emiss_seq[1:])
        log_alpha = jnp.vstack([init_alpha[None, :], log_alpha_rest])

        total_log_lik = jax.nn.logsumexp(log_alpha[-1])

        # Backward pass
        def bwd_step(log_beta_next, log_emiss_next):
            m = log_trans + (log_emiss_next + log_beta_next)[None, :]
            log_beta_t = jax.nn.logsumexp(m, axis=1)
            return log_beta_t, log_beta_t

        init_beta = jnp.zeros(N, dtype=jnp.float32)
        _, log_beta_rev = jax.lax.scan(
            lambda b, e: bwd_step(b, e),
            init_beta,
            log_emiss_seq[1:],
            reverse=True
        )
        log_beta = jnp.vstack([log_beta_rev, init_beta[None, :]])

        log_gamma = log_alpha + log_beta - total_log_lik

        # Posterior transitions xi: sum_{t=0}^{T-2} alpha_t(i) * T_{ij} * b_j(y_{t+1}) * beta_{t+1}(j) / L
        alpha_slice = log_alpha[:-1]
        beta_slice = log_beta[1:]
        emiss_slice = log_emiss_seq[1:]

        term_i = alpha_slice
        term_j = (emiss_slice + beta_slice)
        log_xi_all = term_i[:, :, None] + log_trans[None, :, :] + term_j[:, None, :] - total_log_lik
        log_xi_sum = jax.nn.logsumexp(log_xi_all, axis=0)

        return log_gamma, log_xi_sum, total_log_lik

    @jax.jit
    def jax_forward_loglik_only(log_trans, log_emiss_seq, init_log_pi):
        """Computes log-likelihood of observation sequence without backward pass."""
        def fwd_step(log_alpha_prev, log_emiss_t):
            m = log_alpha_prev[:, None] + log_trans
            log_alpha_pred = jax.nn.logsumexp(m, axis=0)
            log_alpha_t = log_alpha_pred + log_emiss_t
            return log_alpha_t, log_alpha_t

        init_alpha = init_log_pi + log_emiss_seq[0]
        final_alpha, _ = jax.lax.scan(fwd_step, init_alpha, log_emiss_seq[1:])
        return jax.nn.logsumexp(final_alpha)


class HierarchicalSubroutineModel:
    """
    Two-layer hierarchical model for behavioral syllables:
    M macro-states, K shared subroutines of S ordered stages.
    Supports Multi-Sequence EM and Tied vs. Free stage emissions.
    """
    def __init__(
        self,
        M=3,
        K=6,
        S=3,
        n_syllables=100,
        alpha_p=2.0,
        beta_p=2.0,
        init_p=0.5,
        tied_stage_emissions=False,
        random_seed=42
    ):
        self.M = int(M)
        self.K = int(K)
        self.S = int(S)
        self.n_syllables = int(n_syllables)
        self.alpha_p = float(alpha_p)
        self.beta_p = float(beta_p)
        self.init_p = float(init_p)
        self.tied_stage_emissions = bool(tied_stage_emissions)
        self.random_seed = int(random_seed)

        self.N_states = self.M * self.K * self.S

        self.p_advance = None  # (K,)
        self.A_macro = None    # (M, M)
        self.B_routine = None  # (M, K)
        self.theta = None      # (K, S, n_syllables)
        self.pi_init = None    # (N_states,)
        self.T_mat = None      # (N_states, N_states)

        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(self.random_seed)
        M, K, S, V = self.M, self.K, self.S, self.n_syllables

        self.p_advance = np.full(K, self.init_p, dtype=np.float32)

        # Macro transitions (sticky upper layer prior)
        A = np.eye(M, dtype=np.float32) * 5.0 + np.ones((M, M), dtype=np.float32)
        A /= A.sum(axis=1, keepdims=True)
        self.A_macro = A

        # Routine choice per macro state
        B = np.random.dirichlet(np.ones(K) * 1.5, size=M).astype(np.float32)
        self.B_routine = B

        # Syllable emissions
        if self.tied_stage_emissions:
            theta_k = np.random.dirichlet(np.ones(V) * 0.1, size=K).astype(np.float32) # (K, V)
            self.theta = np.repeat(theta_k[:, None, :], S, axis=1) # (K, S, V)
        else:
            self.theta = np.random.dirichlet(np.ones(V) * 0.1, size=(K, S)).astype(np.float32)

        # Initial state prior: uniform across routines at stage 0
        pi = np.zeros(self.N_states, dtype=np.float32)
        for u in range(M):
            for k in range(K):
                idx = u * (K * S) + k * S + 0
                pi[idx] = 1.0 / (M * K)
        self.pi_init = pi

        self.T_mat = build_transition_matrix_np(M, K, S, self.p_advance, self.A_macro, self.B_routine)

    def state_to_tuple(self, idx):
        u = idx // (self.K * self.S)
        rem = idx % (self.K * self.S)
        k = rem // self.S
        s = rem % self.S
        return int(u), int(k), int(s)

    def tuple_to_state(self, u, k, s):
        return int(u * (self.K * self.S) + k * self.S + s)

    def get_log_emission_seq(self, syllables):
        T = len(syllables)
        log_emiss = np.zeros((T, self.N_states), dtype=np.float32)
        log_theta = np.log(np.maximum(self.theta, 1e-12))

        for k in range(self.K):
            for s in range(self.S):
                log_p_sylls = log_theta[k, s, :]
                token_logs = log_p_sylls[syllables]
                for u in range(self.M):
                    idx = self.tuple_to_state(u, k, s)
                    log_emiss[:, idx] = token_logs

        return log_emiss

    def fit_multi_sequence(self, sequence_list, max_iter=25, tol=1e-3, verbose=False):
        """
        Fits model using Multi-Sequence Baum-Welch EM.
        Each element of sequence_list is an independent 1D int array of syllable IDs.
        Accumulates sufficient statistics across all sequences before M-step.
        """
        assert len(sequence_list) > 0, "Sequence list must not be empty"
        history = []
        prev_ll = -np.inf

        log_trans = np.log(np.maximum(self.T_mat, 1e-12))
        log_pi = np.log(np.maximum(self.pi_init, 1e-12))

        total_bouts = sum(len(seq) for seq in sequence_list)

        for it in range(max_iter):
            t0 = time.time()
            total_log_lik = 0.0

            # E-step accumulator across all sequences
            total_xi_sum = np.zeros((self.N_states, self.N_states), dtype=np.float32)
            # Accumulator for emissions: (K, S, V)
            total_emission_counts = np.zeros((self.K, self.S, self.n_syllables), dtype=np.float32)

            for seq in sequence_list:
                seq = np.asarray(seq, dtype=int)
                log_emiss = self.get_log_emission_seq(seq)

                if HAS_JAX:
                    j_log_trans = jnp.array(log_trans)
                    j_log_emiss = jnp.array(log_emiss)
                    j_log_pi = jnp.array(log_pi)

                    log_gamma_j, log_xi_sum_j, ll_j = jax_forward_backward(
                        j_log_trans, j_log_emiss, j_log_pi
                    )
                    log_gamma = np.array(log_gamma_j)
                    xi_sum = np.exp(np.array(log_xi_sum_j))
                    total_log_lik += float(ll_j)
                else:
                    raise NotImplementedError("JAX GPU required")

                total_xi_sum += xi_sum
                gamma = np.exp(log_gamma) # (T_seq, N_states)

                # Accumulate emissions
                for k in range(self.K):
                    for s in range(self.S):
                        u_indices = [self.tuple_to_state(u, k, s) for u in range(self.M)]
                        gamma_ks = gamma[:, u_indices].sum(axis=1) # (T_seq,)
                        counts = np.bincount(seq, weights=gamma_ks, minlength=self.n_syllables).astype(np.float32)
                        total_emission_counts[k, s, :] += counts

            history.append(total_log_lik)

            # M-STEP
            # 1. Update advance probabilities p_k with Beta(alpha_p, beta_p)
            for k in range(self.K):
                n_adv = 0.0
                n_stay = 0.0
                for u in range(self.M):
                    i0 = self.tuple_to_state(u, k, 0)
                    n_stay += total_xi_sum[i0, i0]
                    n_adv += total_xi_sum[i0, self.tuple_to_state(u, k, 1)]

                    i1 = self.tuple_to_state(u, k, 1)
                    n_stay += total_xi_sum[i1, i1]
                    n_adv += total_xi_sum[i1, self.tuple_to_state(u, k, 2)]

                    i2 = self.tuple_to_state(u, k, 2)
                    n_stay += total_xi_sum[i2, i2]
                    for u_next in range(self.M):
                        for k_next in range(self.K):
                            n_adv += total_xi_sum[i2, self.tuple_to_state(u_next, k_next, 0)]

                p_k_new = (n_adv + self.alpha_p - 1.0) / (n_adv + n_stay + self.alpha_p + self.beta_p - 2.0 + 1e-12)
                self.p_advance[k] = float(np.clip(p_k_new, 0.05, 0.95))

            # 2. Update A_macro and B_routine from exits
            C_macro = np.zeros((self.M, self.M), dtype=np.float32)
            C_rout = np.zeros((self.M, self.K), dtype=np.float32)

            for u in range(self.M):
                for k in range(self.K):
                    i2 = self.tuple_to_state(u, k, 2)
                    for u_next in range(self.M):
                        for k_next in range(self.K):
                            trans_count = total_xi_sum[i2, self.tuple_to_state(u_next, k_next, 0)]
                            C_macro[u, u_next] += trans_count
                            C_rout[u_next, k_next] += trans_count

            # Dirichlet smoothing
            self.A_macro = (C_macro + 0.5) / ((C_macro + 0.5).sum(axis=1, keepdims=True) + 1e-12)
            self.B_routine = (C_rout + 0.5) / ((C_rout + 0.5).sum(axis=1, keepdims=True) + 1e-12)

            # 3. Update Emissions theta_{k, s}
            if self.tied_stage_emissions:
                # Pool emissions across all 3 stages for each routine k
                new_theta = np.zeros_like(self.theta)
                for k in range(self.K):
                    pooled_counts = total_emission_counts[k].sum(axis=0) + 0.05 # (V,)
                    theta_k = pooled_counts / pooled_counts.sum()
                    for s in range(self.S):
                        new_theta[k, s, :] = theta_k
                self.theta = new_theta
            else:
                new_theta = np.zeros_like(self.theta)
                for k in range(self.K):
                    for s in range(self.S):
                        counts = total_emission_counts[k, s, :] + 0.05
                        new_theta[k, s, :] = counts / counts.sum()
                self.theta = new_theta

            # 4. Update T_mat
            self.T_mat = build_transition_matrix_np(self.M, self.K, self.S, self.p_advance, self.A_macro, self.B_routine)
            log_trans = np.log(np.maximum(self.T_mat, 1e-12))

            elapsed = time.time() - t0
            d_ll = total_log_lik - prev_ll
            ll_per_bout = total_log_lik / total_bouts

            if verbose:
                exp_durs = [round(3.0 / p, 2) for p in self.p_advance]
                print(f"Iter {it+1:2d} | LogLik: {total_log_lik:12.2f} (Δ={d_ll:8.2f}) | LL/bout: {ll_per_bout:.4f} | E[D_k]: {exp_durs} | {elapsed*1000:.1f}ms")

            if abs(d_ll) < tol and it > 3:
                break
            prev_ll = total_log_lik

        return history

    def score(self, sequence_list):
        """
        Computes total log-likelihood and log-likelihood per bout on test sequences.
        """
        log_trans = np.log(np.maximum(self.T_mat, 1e-12))
        log_pi = np.log(np.maximum(self.pi_init, 1e-12))

        total_ll = 0.0
        total_bouts = 0

        j_log_trans = jnp.array(log_trans)
        j_log_pi = jnp.array(log_pi)

        for seq in sequence_list:
            seq = np.asarray(seq, dtype=int)
            T_seq = len(seq)
            total_bouts += T_seq
            log_emiss = self.get_log_emission_seq(seq)
            j_log_emiss = jnp.array(log_emiss)
            ll = float(jax_forward_loglik_only(j_log_trans, j_log_emiss, j_log_pi))
            total_ll += ll

        ll_per_bout = total_ll / (total_bouts + 1e-12)
        return total_ll, ll_per_bout

    def decode_session_viterbi(self, syllables):
        """
        Decodes a single session sequence into (macro_seq, routine_seq, stage_seq).
        """
        syllables = np.asarray(syllables, dtype=int)
        T = len(syllables)
        log_trans = np.log(np.maximum(self.T_mat, 1e-12))
        log_pi = np.log(np.maximum(self.pi_init, 1e-12))
        log_emiss = self.get_log_emission_seq(syllables)

        viterbi = np.zeros((T, self.N_states), dtype=np.float32)
        backpointers = np.zeros((T, self.N_states), dtype=np.int32)

        viterbi[0] = log_pi + log_emiss[0]

        for t in range(1, T):
            m = viterbi[t-1][:, None] + log_trans
            backpointers[t] = np.argmax(m, axis=0)
            viterbi[t] = np.max(m, axis=0) + log_emiss[t]

        path = np.zeros(T, dtype=np.int32)
        path[-1] = int(np.argmax(viterbi[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backpointers[t + 1, path[t + 1]]

        macro_seq = np.zeros(T, dtype=np.int32)
        routine_seq = np.zeros(T, dtype=np.int32)
        stage_seq = np.zeros(T, dtype=np.int32)

        for t in range(T):
            u, k, s = self.state_to_tuple(path[t])
            macro_seq[t] = u
            routine_seq[t] = k
            stage_seq[t] = s

        return macro_seq, routine_seq, stage_seq
