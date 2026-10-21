# Phase 1 Analysis Methods: A Primer

## 1. Convergent Cross Mapping (CCM)
**"The Dynamical Systems Lens"**

*   **Theory**: Based on Takens' Theorem (State Space Reconstruction).
*   **Core Idea**: If variable $X$ causally influences $Y$, then the state of $X$ is encoded in the history of $Y$ (because $X$ is part of the system driving $Y$).
*   **Mechanism**: We reconstruct the "shadow manifold" of $Y$ ($M_Y$) from its time-delayed history. If $X \to Y$, we should be able to predict the state of $X$ using only the neighbors in $M_Y$.
    *   *Note*: The logic is "Counter-intuitive": We use **Effect ($Y$) to predict Cause ($X$)**.
*   **Key Metric**: Prediction Skill ($\rho$). The correlation between the reconstructed $X$ (from $Y$'s manifold) and the actual $X$.
*   **Interpretation**:
    *   High $\rho(X|Y)$ (predicting X from Y) $\implies$ **$X \to Y$**.
    *   If $\rho$ increases with library length (convergence), it's a sign of true causation, not just correlation.
*   **Best For**: Non-linear, deterministic systems (like neural dynamics) where variables are coupled.

## 2. Transfer Entropy (TE)
**"The Information Theory Lens"**

*   **Theory**: Based on Shannon Entropy and conditional probability.
*   **Core Idea**: Does knowing the past of $X$ reduce our uncertainty about the future of $Y$, *beyond* what we already know from $Y$'s own past?
*   **Formula**: $TE_{X \to Y} = I(Y_{future} ; X_{past} | Y_{past})$.
    *   (In our code: Normalized Gaussian Approximation).
*   **Interpretation**:
    *   $TE > 0$: Information is flowing from X to Y.
    *   Asymmetry ($TE_{X \to Y} \gg TE_{Y \to X}$) indicates direction.
*   **Best For**: Detecting "surprise" or novel information transfer. It is more general than correlation but "model-free" (mostly).

## 3. VarLiNGAM (Vector Autoregressive LiNGAM)
**"The Structural Causal Model Lens"**

*   **Theory**: Combines Vector Autoregression (VAR) with Linear Non-Gaussian Acyclic Models (LiNGAM).
*   **Core Idea**:
    1.  **Time-Lagged**: Use VAR to find delayed effects ($X_{t-1} \to Y_t$).
    2.  **Instantaneous**: Analyze the *residuals* (errors) of the VAR model. If the residuals are non-Gaussian (which they usually are in real data), we can mathematically distinguish $X_t \to Y_t$ from $Y_t \to X_t$.
*   **Mechanism**: It constructs a Directed Acyclic Graph (DAG) of instantaneous interactions.
*   **Interpretation**:
    *   Adjacency Matrix: Non-zero entry $B_{ji}$ means $X_i \to X_j$.
*   **Best For**: Disentangling complex networks where variables influence each other effectively "instantly" (relative to sampling rate) and finding the specific structure of the network.

## 4. Multi-Analysis (Group) LiNGAM
**"The Population Lens"**

*   **Core Idea**: Instead of analyzing one animal at a time, we assume all animals share a common causal graph structure (but maybe different connection strengths).
*   **Mechanism**: We pool DataFrames from all sessions and run LiNGAM on the aggregate (or use algorithm specific for multiple groups).
*   **Benefit**: Increases statistical power and finds the "robust" network that exists across the population, ignoring individual quirks.

---

## Synthesis: How to Interpret Your Results

| Scenario | Correlation | TE | CCM | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Independent** | Low | Low | Low | No interaction. |
| **Common Driver** | **High** | Low | Low | Both driven by Z (e.g. movement, arousal). No direct link. |
| **Feedback Loop** | High | **High (Both)** | **High (Both)** | Tight, bidirectional coupling. |
| **Directed Drive** | High | **High ($X \to Y$)** | **High ($X|Y$)** | $X$ drives $Y$. |
| **State Coupling** | High | Low | **High** | Systems are "synchronized" manifolds (like locked oscillators) but maybe little *new* information is being transferred (steady state). |

In your preliminary results:
*   **NAc_5HT $\to$ DRN_NE**: Shows High TE (asymmetry) and Moderate CCM. Suggests **Directed Drive**.
*   **Bilateral NAc**: Shows High Correlation but Low TE/CCM. Suggests **Common Driver** (arousal/locomotion).
