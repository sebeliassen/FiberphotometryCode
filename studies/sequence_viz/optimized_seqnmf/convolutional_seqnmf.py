#!/usr/bin/env python3
"""
convolutional_seqnmf.py
=============================================================================
Modular, High-Performance GPU-Accelerated Convolutional seqNMF Solver.
Utilizes JAX on NVIDIA RTX 4090 with graceful NumPy CPU fallback.
Implements the mathematical formulation from Mackevicius et al. (2019) & FeeLab:
- Input: One-hot bout matrix X in {0, 1}^{S x T}
- Templates: W in R_+^{S x K x L} (S syllables, K factors, L lags)
- Activations: H in R_+^{K x T}
- Convolutional Reconstruction: X_hat = sum_{l=0}^{L-1} W_{:, :, l} (ShiftRight_l H)
- Objective: min 0.5 * ||X - X_hat||_F^2 + lambda * R(W, H)
- Quantitative Metrics:
  * Held-out cross-animal validation error
  * Factor redundancy / cross-orthogonality
  * Multi-seed template stability (Hungarian alignment)
  * Episode retrieval with whole-template error check:
    E_{k, t} = min_{a >= 0} ||Y_t - a W_k||_F^2 / ||Y_t||_F^2
=============================================================================
"""

import time

import numpy as np

# Detect JAX GPU availability
HAS_JAX_GPU = False
try:
    import jax
    import jax.numpy as jnp
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) > 0:
        HAS_JAX_GPU = True
except Exception:
    HAS_JAX_GPU = False


def shift_right_np(mat, l):
    """Shifts matrix columns right by l positions (zero-padded on left)."""
    if l == 0:
        return mat
    res = np.zeros_like(mat)
    res[:, l:] = mat[:, :-l]
    return res


def shift_left_np(mat, l):
    """Shifts matrix columns left by l positions (zero-padded on right)."""
    if l == 0:
        return mat
    res = np.zeros_like(mat)
    res[:, :-l] = mat[:, l:]
    return res


if HAS_JAX_GPU:
    @jax.jit
    def jax_seqnmf_step(W, H, X, lambda_ortho):
        """Single JIT-compiled GPU multiplicative update step for W and H."""
        S, K, L = W.shape
        _, T = H.shape

        # 1. Forward reconstruction
        X_hat = jnp.zeros((S, T), dtype=jnp.float32)
        for l in range(L):
            H_sh = jnp.pad(H, ((0, 0), (l, 0)), mode="constant")[:, :T]
            X_hat = X_hat + W[:, :, l] @ H_sh
        X_hat = jnp.maximum(X_hat, 1e-7)
        ratio = X / X_hat

        # 2. Update H
        num_H = jnp.zeros((K, T), dtype=jnp.float32)
        den_H = jnp.zeros((K, T), dtype=jnp.float32)
        ones_S = jnp.ones((S, T), dtype=jnp.float32)

        for l in range(L):
            term_num = W[:, :, l].T @ ratio
            num_H = num_H + jnp.pad(term_num, ((0, 0), (0, l)), mode="constant")[:, l:]

            term_den = W[:, :, l].T @ ones_S
            den_H = den_H + jnp.pad(term_den, ((0, 0), (0, l)), mode="constant")[:, l:]

        # Competition penalty: cross-orthogonality between factor activations
        sum_H = jnp.sum(H, axis=0, keepdims=True)
        den_H = den_H + lambda_ortho * (sum_H - H)

        den_H = jnp.maximum(den_H, 1e-7)
        H_new = H * (num_H / den_H)
        H_new = jnp.maximum(H_new, 1e-7)

        # 3. Update W
        X_hat2 = jnp.zeros((S, T), dtype=jnp.float32)
        for l in range(L):
            H_sh = jnp.pad(H_new, ((0, 0), (l, 0)), mode="constant")[:, :T]
            X_hat2 = X_hat2 + W[:, :, l] @ H_sh
        X_hat2 = jnp.maximum(X_hat2, 1e-7)
        ratio2 = X / X_hat2

        ones_X = jnp.ones((S, T), dtype=jnp.float32)
        W_new_list = []
        for l in range(L):
            H_sh = jnp.pad(H_new, ((0, 0), (l, 0)), mode="constant")[:, :T]
            num_W = ratio2 @ H_sh.T
            den_W = jnp.maximum(ones_X @ H_sh.T, 1e-7)
            W_l = W[:, :, l] * (num_W / den_W)
            col_norms = jnp.linalg.norm(W_l, axis=0, keepdims=True) + 1e-7
            W_l = W_l / col_norms
            W_new_list.append(W_l)

        W_new = jnp.stack(W_new_list, axis=2)
        return W_new, H_new

    @jax.jit
    def jax_infer_h_step(W, H, X, lambda_ortho):
        """Single JIT-compiled GPU update for H with W frozen."""
        S, K, L = W.shape
        _, T = H.shape

        X_hat = jnp.zeros((S, T), dtype=jnp.float32)
        for l in range(L):
            H_sh = jnp.pad(H, ((0, 0), (l, 0)), mode="constant")[:, :T]
            X_hat = X_hat + W[:, :, l] @ H_sh
        X_hat = jnp.maximum(X_hat, 1e-7)
        ratio = X / X_hat

        num_H = jnp.zeros((K, T), dtype=jnp.float32)
        den_H = jnp.zeros((K, T), dtype=jnp.float32)
        ones_S = jnp.ones((S, T), dtype=jnp.float32)

        for l in range(L):
            term_num = W[:, :, l].T @ ratio
            num_H = num_H + jnp.pad(term_num, ((0, 0), (0, l)), mode="constant")[:, l:]

            term_den = W[:, :, l].T @ ones_S
            den_H = den_H + jnp.pad(term_den, ((0, 0), (0, l)), mode="constant")[:, l:]

        sum_H = jnp.sum(H, axis=0, keepdims=True)
        den_H = den_H + lambda_ortho * (sum_H - H)

        den_H = jnp.maximum(den_H, 1e-7)
        H_new = H * (num_H / den_H)
        H_new = jnp.maximum(H_new, 1e-7)
        return H_new


class ConvolutionalSeqNMF:
    def __init__(
        self,
        K=6,
        L=5,
        lambda_ortho=0.5,
        max_iter=60,
        tol=1e-5,
        random_seed=42,
        use_gpu=True,
        verbose=False
    ):
        self.K = int(K)
        self.L = int(L)
        self.lambda_ortho = float(lambda_ortho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_seed = int(random_seed)
        self.use_gpu = bool(use_gpu and HAS_JAX_GPU)
        self.verbose = verbose

        self.W = None  # (S, K, L)
        self.H = None  # (K, T)
        self.training_history = []
        self.fit_time_sec = 0.0

    def fit(self, X):
        """Fits W and H on one-hot bout matrix X: (S, T)."""
        S, T = X.shape
        K, L = self.K, self.L
        t0 = time.time()

        np.random.seed(self.random_seed)
        W_init = np.random.uniform(0.1, 1.0, (S, K, L)).astype(np.float32)
        H_init = np.random.uniform(0.1, 1.0, (K, T)).astype(np.float32)

        for l in range(L):
            col_norms = np.linalg.norm(W_init[:, :, l], axis=0, keepdims=True) + 1e-7
            W_init[:, :, l] /= col_norms

        if self.use_gpu:
            W_jax = jnp.array(W_init)
            H_jax = jnp.array(H_init)
            X_jax = jnp.array(X, dtype=jnp.float32)
            lam = float(self.lambda_ortho)

            for it in range(self.max_iter):
                W_jax, H_jax = jax_seqnmf_step(W_jax, H_jax, X_jax, lam)

            _ = W_jax.block_until_ready()
            self.W = np.array(W_jax)
            self.H = np.array(H_jax)
        else:
            # NumPy CPU fallback
            W = W_init.copy()
            H = H_init.copy()
            ones_S = np.ones((S, T), dtype=np.float32)
            ones_X = np.ones_like(X, dtype=np.float32)

            for it in range(self.max_iter):
                X_hat = np.zeros((S, T), dtype=np.float32)
                for l in range(L):
                    X_hat += W[:, :, l] @ shift_right_np(H, l)
                X_hat = np.maximum(X_hat, 1e-7)
                ratio = X / X_hat

                num_H = np.zeros_like(H)
                den_H = np.zeros_like(H)
                for l in range(L):
                    num_H += shift_left_np(W[:, :, l].T @ ratio, l)
                    den_H += shift_left_np(W[:, :, l].T @ ones_S, l)

                if self.lambda_ortho > 0:
                    sum_H = np.sum(H, axis=0, keepdims=True)
                    den_H += self.lambda_ortho * (sum_H - H)

                den_H = np.maximum(den_H, 1e-7)
                H = H * (num_H / den_H)
                H = np.maximum(H, 1e-7)

                X_hat = np.zeros((S, T), dtype=np.float32)
                for l in range(L):
                    X_hat += W[:, :, l] @ shift_right_np(H, l)
                X_hat = np.maximum(X_hat, 1e-7)
                ratio = X / X_hat

                for l in range(L):
                    H_shift = shift_right_np(H, l)
                    num_W = ratio @ H_shift.T
                    den_W = np.maximum(ones_X @ H_shift.T, 1e-7)
                    W[:, :, l] *= (num_W / den_W)
                    col_norms = np.linalg.norm(W[:, :, l], axis=0, keepdims=True) + 1e-7
                    W[:, :, l] /= col_norms

            self.W = W
            self.H = H

        self.fit_time_sec = time.time() - t0
        return self

    def infer_H(self, X_val, max_iter=30):
        """Infers activations H_val on held-out data X_val while keeping W fixed."""
        assert self.W is not None, "Model must be fitted before inferring H"
        S, T_val = X_val.shape
        K, L = self.K, self.L

        np.random.seed(self.random_seed)
        H_init = np.random.uniform(0.1, 1.0, (K, T_val)).astype(np.float32)

        if self.use_gpu:
            W_jax = jnp.array(self.W)
            H_jax = jnp.array(H_init)
            X_jax = jnp.array(X_val, dtype=jnp.float32)
            lam = float(self.lambda_ortho)

            for _ in range(max_iter):
                H_jax = jax_infer_h_step(W_jax, H_jax, X_jax, lam)

            _ = H_jax.block_until_ready()
            return np.array(H_jax)
        else:
            H_val = H_init.copy()
            ones_S = np.ones((S, T_val), dtype=np.float32)

            for _ in range(max_iter):
                X_hat = np.zeros((S, T_val), dtype=np.float32)
                for l in range(L):
                    X_hat += self.W[:, :, l] @ shift_right_np(H_val, l)
                X_hat = np.maximum(X_hat, 1e-7)
                ratio = X_val / X_hat

                num_H = np.zeros_like(H_val)
                den_H = np.zeros_like(H_val)
                for l in range(L):
                    num_H += shift_left_np(self.W[:, :, l].T @ ratio, l)
                    den_H += shift_left_np(self.W[:, :, l].T @ ones_S, l)

                if self.lambda_ortho > 0:
                    sum_H = np.sum(H_val, axis=0, keepdims=True)
                    den_H += self.lambda_ortho * (sum_H - H_val)

                den_H = np.maximum(den_H, 1e-7)
                H_val *= (num_H / den_H)
                H_val = np.maximum(H_val, 1e-7)

            return H_val

    def reconstruct(self, H=None):
        """Reconstructs X_hat from W and H."""
        if H is None:
            H = self.H
        assert self.W is not None and H is not None
        S, K, L = self.W.shape
        _, T = H.shape
        X_hat = np.zeros((S, T), dtype=np.float32)
        for l in range(L):
            X_hat += self.W[:, :, l] @ shift_right_np(H, l)
        return X_hat

    def relative_reconstruction_error(self, X, H=None):
        """Computes relative Frobenius reconstruction error ||X - X_hat||_F / ||X||_F."""
        X_hat = self.reconstruct(H)
        return float(np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-8))

    def compute_factor_redundancy(self, H=None):
        """Pairwise cross-factor activation correlation (competition efficiency)."""
        if H is None:
            H = self.H
        K, T = H.shape
        if K < 2:
            return 0.0

        norms = np.linalg.norm(H, axis=1) + 1e-8
        H_normed = H / norms[:, None]
        corr_matrix = H_normed @ H_normed.T

        triu_indices = np.triu_indices(K, k=1)
        mean_overlap = float(np.mean(corr_matrix[triu_indices]))
        return mean_overlap

    def get_dominant_token_progression(self, k):
        """Returns sequence of dominant syllables and their weights for factor k."""
        assert self.W is not None
        L = self.L
        tokens = []
        weights = []
        for l in range(L):
            best_s = int(np.argmax(self.W[:, k, l]))
            tokens.append(best_s)
            weights.append(float(self.W[best_s, k, l]))
        return tokens, weights

    def retrieve_episodes(
        self,
        X,
        H=None,
        factor_idx=0,
        tau_percentile=92.0,
        nms_window=None,
        max_whole_template_err=0.70,
        min_stage_support=0.50
    ):
        """
        Extracts verified candidate episodes for factor k:
        1. Local activation peak detection (H_{k, t} > tau_k)
        2. Non-maximum suppression across nearby frames
        3. Whole-template normalized reconstruction error check:
           E_{k, t} = min_{a >= 0} ||Y_t - a W_k||_F^2 / ||Y_t||_F^2
        """
        if H is None:
            H = self.H
        assert self.W is not None and H is not None

        k = factor_idx
        L = self.L
        S, T = X.shape
        h_k = H[k]
        W_k = self.W[:, k, :]
        W_k_norm_sq = float(np.sum(W_k ** 2)) + 1e-8

        if nms_window is None:
            nms_window = max(2, L // 2)

        non_zero_acts = h_k[h_k > 1e-4]
        if len(non_zero_acts) == 0:
            return []
        tau_k = float(np.percentile(non_zero_acts, tau_percentile))

        peaks = []
        for t in range(1, T - L - 1):
            if h_k[t] > tau_k and h_k[t] >= h_k[t - 1] and h_k[t] >= h_k[t + 1]:
                peaks.append((t, float(h_k[t])))

        if not peaks:
            return []

        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
        suppressed = np.zeros(T, dtype=bool)
        selected_peaks = []

        for t, act in peaks:
            if not suppressed[t]:
                selected_peaks.append((t, act))
                start_w = max(0, t - nms_window)
                end_w = min(T, t + nms_window + 1)
                suppressed[start_w:end_w] = True

        selected_peaks = sorted(selected_peaks, key=lambda p: p[0])

        episodes = []
        dom_tokens, _ = self.get_dominant_token_progression(k)

        for t, act in selected_peaks:
            if t + L > T:
                continue
            Y_t = X[:, t:t + L]
            Y_t_norm_sq = float(np.sum(Y_t ** 2)) + 1e-8

            inner_prod = float(np.sum(Y_t * W_k))
            a_star = max(0.0, inner_prod / W_k_norm_sq)

            recon_diff = Y_t - a_star * W_k
            E_kt = float(np.sum(recon_diff ** 2) / Y_t_norm_sq)

            stage_matches = 0
            observed_tokens = []
            for l in range(L):
                obs_s = int(np.argmax(Y_t[:, l]))
                observed_tokens.append(obs_s)
                if W_k[obs_s, l] > 0.15:
                    stage_matches += 1
            stage_support_frac = stage_matches / L

            if E_kt <= max_whole_template_err and stage_support_frac >= min_stage_support:
                episodes.append({
                    "start_bout": int(t),
                    "end_bout": int(t + L),
                    "peak_activation": float(act),
                    "template_error": float(E_kt),
                    "stage_support": float(stage_support_frac),
                    "observed_tokens": observed_tokens,
                    "dominant_template": dom_tokens
                })

        return episodes
