from typing import Dict, List, Optional, Tuple  # noqa: F401 - retain historical module attributes

import numpy as np
from sklearn.linear_model import Ridge


class StrategyConditionedPoseModel:
    """
    Autoregressive continuous bout predictor operating on 30-D bout vectors.
    Evaluates:
      1. Unconditioned Baseline: Predicts z_{t+1} given z_t.
      2. Strategy-Conditioned Model: Predicts z_{t+1} given z_t AND Strategy ID S_t.
      3. Shuffled Control: Predicts z_{t+1} given z_t and randomly permuted strategy IDs.

    Computes Relative Information Gain = 1 - (MSE_conditioned / MSE_unconditioned).
    """
    def __init__(self, feature_dim: int = 30, alpha: float = 1.0):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.unconditioned_model: Optional[Ridge] = None
        self.conditioned_models: Dict[int, Ridge] = {}
        self.K_strategies: int = 0

    def fit(self, X_bouts: np.ndarray, strategy_ids: np.ndarray):
        """
        Fits baseline and strategy-specific autoregressive models.
        X_bouts: Array of shape (N_bouts, feature_dim)
        strategy_ids: Array of shape (N_bouts,) containing discrete strategy assignments
        """
        N = len(X_bouts)
        if N < 2:
            return self

        X_t = X_bouts[:-1]       # Current bout
        y_next = X_bouts[1:]     # Next bout
        strat_t = strategy_ids[:-1]

        # 1. Fit generic unconditioned model
        self.unconditioned_model = Ridge(alpha=self.alpha)
        self.unconditioned_model.fit(X_t, y_next)

        # 2. Fit strategy-conditioned models
        unique_strats = sorted(list(set(strat_t)))
        self.K_strategies = len(unique_strats)
        self.conditioned_models = {}

        for s in unique_strats:
            mask = (strat_t == s)
            if np.sum(mask) >= 5:  # Sufficient samples to fit
                m = Ridge(alpha=self.alpha)
                m.fit(X_t[mask], y_next[mask])
                self.conditioned_models[s] = m
            else:
                # Fallback to unconditioned model if cluster is tiny
                self.conditioned_models[s] = self.unconditioned_model

        return self

    def evaluate(self, X_bouts_test: np.ndarray, strategy_ids_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates forecasting error on test sequence.
        Returns:
          - mse_unconditioned: MSE without strategy context
          - mse_conditioned: MSE with true strategy context
          - mse_shuffled: MSE with time-permuted strategy context
          - relative_gain: Fraction of variance error reduced by strategy conditioning
          - cosine_sim_conditioned: Mean cosine similarity between predicted and true next bout
        """
        N = len(X_bouts_test)
        if N < 2 or self.unconditioned_model is None:
            return {
                "mse_unconditioned": 0.0,
                "mse_conditioned": 0.0,
                "mse_shuffled": 0.0,
                "relative_gain": 0.0,
                "cosine_sim_conditioned": 0.0
            }

        X_t = X_bouts_test[:-1]
        y_true = X_bouts_test[1:]
        strat_t = strategy_ids_test[:-1]

        # 1. Unconditioned prediction
        y_pred_uncond = self.unconditioned_model.predict(X_t)
        mse_uncond = float(np.mean((y_true - y_pred_uncond) ** 2))

        # 2. Strategy-conditioned prediction
        y_pred_cond = np.zeros_like(y_true)
        for i in range(len(X_t)):
            s = strat_t[i]
            if s in self.conditioned_models:
                y_pred_cond[i] = self.conditioned_models[s].predict(X_t[i:i+1])[0]
            else:
                y_pred_cond[i] = self.unconditioned_model.predict(X_t[i:i+1])[0]

        mse_cond = float(np.mean((y_true - y_pred_cond) ** 2))

        # 3. Shuffled strategy control
        shuffled_strats = np.random.permutation(strat_t)
        y_pred_shuff = np.zeros_like(y_true)
        for i in range(len(X_t)):
            s = shuffled_strats[i]
            if s in self.conditioned_models:
                y_pred_shuff[i] = self.conditioned_models[s].predict(X_t[i:i+1])[0]
            else:
                y_pred_shuff[i] = self.unconditioned_model.predict(X_t[i:i+1])[0]

        mse_shuff = float(np.mean((y_true - y_pred_shuff) ** 2))

        # 4. Cosine Similarity
        norm_true = np.linalg.norm(y_true, axis=1) + 1e-12
        norm_pred = np.linalg.norm(y_pred_cond, axis=1) + 1e-12
        cosine_sim = float(np.mean(np.sum(y_true * y_pred_cond, axis=1) / (norm_true * norm_pred)))

        # Relative Information Gain
        gain = float(1.0 - (mse_cond / (mse_uncond + 1e-12)))

        return {
            "mse_unconditioned": mse_uncond,
            "mse_conditioned": mse_cond,
            "mse_shuffled": mse_shuff,
            "relative_gain": gain,
            "cosine_sim_conditioned": cosine_sim
        }
