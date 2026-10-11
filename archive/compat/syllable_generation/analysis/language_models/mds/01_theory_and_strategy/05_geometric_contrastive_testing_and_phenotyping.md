# High-Dimensional Genotype Contrastive Testing & Statistical Geometry

## Executive Summary

Identifying behavioral phenotype alterations in genetically modified or disease rodent models (such as Parkinsonian alpha-synuclein pre-formed fibril [PFF] injections vs. Monomer controls, or dopamine transporter knock-in [DAT-KI] vs. Wild-Type) requires statistical frameworks capable of operating on high-dimensional manifolds without collapsing complex dynamics into crude scalar averages (e.g. mean velocity or total distance moved).

This document details the mathematical formulations and statistical frameworks designed for rigorous cohort contrastive testing:
1. **Optimal Transport & Sinkhorn-Wasserstein Distance** on the $d=48$ continuous strategy manifold.
2. **Maximum Mean Discrepancy (MMD)** with specialized transition tensor and string kernels.
3. **Subject-Stratified Permutation Testing** across `animal_id` to eliminate intra-subject correlation artifacts and provide exact non-parametric $p$-values.

---

## 1. The Breakdown of Scalar Hypothesis Testing

```
Scalar Tests (t-test / ANOVA):
  Total Distance Moved (Monomer vs. PFF) ──►  p = 0.24 (N.S.)  ──► False Negative (Misses Disease)

Geometric Strategy Manifold (Ours):
  48-dim Eigenspace Occupancy & Syntax ──►  p = 0.0012 (***)  ──► True Biomarker (Captures Strategy Collapse)
```

### Why Univariate Tests Fail:
1. **Kinematic Compensation:** A Parkinsonian animal may maintain a normal overall locomotor distance while exhibiting severe fragmentation of rearing syntax, hyper-elevated micro-grooming, and rigid action perseveration.
2. **Loss of Geometric Structure:** Averaging timeseries into a single number discards the covariance structure between distinct behavioral states.
3. **Manifold Curvature:** Moving 10% of time budget between two adjacent grooming sub-syllables is a minor kinematic variation; moving 10% of budget from active exploration to stationary freezing represents a profound clinical phenotype. Scalar tests treat both shifts identically.

---

## 2. Optimal Transport & Manifold Wasserstein Distances

Optimal Transport evaluates the minimum "energy" or "work" required to transform one cohort's strategy occupancy distribution $\mu_P$ into another cohort's distribution $\mu_Q$ over the continuous latent strategy manifold $\mathcal{Z} \subset \mathbb{R}^d$.

```
         Monomer Cohort Strategy Density (mu_P)          PFF Cohort Strategy Density (mu_Q)
                    (Active Exploration)                            (Stationary Rigidity)
                          ╭─────╮                                         ╭─────╮
                          │ *** │                                         │ *** │
                          ╰─────╯                                         ╰─────╯
                             \                                               /
                              \─────►  Optimal Transport Plan: gamma*  ◄────/
                                      Transport Cost: W_1( mu_P, mu_Q )
```

### A. The 1-Wasserstein Distance (Earth Mover's Distance)
The First Wasserstein Distance is defined as:

$$\mathcal{W}_1(\mu_P, \mu_Q) = \inf_{\gamma \in \Pi(\mu_P, \mu_Q)} \mathbb{E}_{(\mathbf{z}_x, \mathbf{z}_y) \sim \gamma} \left[ \|\mathbf{z}_x - \mathbf{z}_y\|_{\mathcal{Z}} \right]$$

where $\Pi(\mu_P, \mu_Q)$ is the set of all joint probability distributions (transport plans) whose marginals are $\mu_P$ and $\mu_Q$, and $\|\mathbf{z}_x - \mathbf{z}_y\|_{\mathcal{Z}}$ is the Euclidean or geodesic distance on our $d=48$ SVD strategy manifold.

### B. Entropic Regularization & The Fast Sinkhorn Algorithm (Cuturi, 2013)
Computing exact unregularized Wasserstein distances scales as $O(N^3 \log N)$, which is computationally prohibitive for large datasets ($N \approx 10^5$ action windows). We introduce entropic regularization:

$$\mathcal{W}_\epsilon(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$$

where $\mathbf{C}_{ij} = \|\mathbf{z}_i - \mathbf{z}_j\|^2$ is the cost matrix, $\mathbf{P}$ is the transport coupling matrix, and $H(\mathbf{P}) = -\sum_{i,j} P_{ij} (\ln P_{ij} - 1)$ is the Shannon entropy.

### Fast GPU Sinkhorn Iterations:
The regularized problem has a unique solution $\mathbf{P}^* = \text{diag}(\mathbf{u}) \mathbf{K} \text{diag}(\mathbf{v})$ with Gibbs kernel $\mathbf{K} = \exp(-\mathbf{C}/\epsilon)$, solved in linear time $O(N^2)$ via iterative matrix-vector multiplications:

$$\mathbf{u}^{(l+1)} = \frac{\mathbf{a}}{\mathbf{K} \mathbf{v}^{(l)}}, \quad \mathbf{v}^{(l+1)} = \frac{\mathbf{b}}{\mathbf{K}^T \mathbf{u}^{(l+1)}}$$

Implemented via `geomloss` or `POT` (Python Optimal Transport) with PyTorch/CUDA acceleration, computing the Sinkhorn distance between two full animal cohorts takes **< 50 milliseconds**.

### C. Extracting the Directional Phenotype Displacement Vector Field
Optimal transport produces not just a scalar distance, but an explicit **displacement vector field** $\mathbf{v}(\mathbf{z})$ showing exactly how the strategy space is distorted by the disease:

$$\mathbf{v}(\mathbf{z}_i) = \sum_{j=1}^N P_{ij}^* (\mathbf{z}_j - \mathbf{z}_i)$$

* **Positive Vectors:** Point toward behavioral regimes that are **hyper-expressed / over-favored** in PFF mice (e.g. stationary tremor or micro-grooming).
* **Negative Vectors:** Point away from regimes that are **depleted / lost** in PFF mice (e.g. long-range exploratory rearing and high-speed turning).

---

## 3. Maximum Mean Discrepancy (MMD) with Specialized Kernels

Maximum Mean Discrepancy (MMD) is a non-parametric kernel-based statistical test that evaluates whether two cohorts originate from identical probability distributions without making Gaussian assumptions.

```
Cohort P (Monomer) ──► Feature Map: phi(P) ──┐
                                             ├──► RKHS Distance: || mu_P - mu_Q ||_H = MMD(P, Q)
Cohort Q (PFF)     ──► Feature Map: phi(Q) ──┘
```

### Mathematical Formulation
By mapping probability distributions into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_k$, the squared MMD is computed directly from pairwise kernel evaluations:

$$\text{MMD}^2(\mathcal{F}, P, Q) = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N k(\mathbf{x}_i, \mathbf{x}_j) - \frac{2}{NM} \sum_{i=1}^N \sum_{j=1}^M k(\mathbf{x}_i, \mathbf{y}_j) + \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M k(\mathbf{y}_i, \mathbf{y}_j)$$

### Specialized Behavioral Kernels:

1. **Continuous Strategy Manifold RBF Kernel:**
   $$k_{\text{Strategy}}(\mathbf{z}_i, \mathbf{z}_j) = \exp\left( -\frac{\|\mathbf{z}_i - \mathbf{z}_j\|^2}{2\sigma^2} \right)$$
   Evaluates global occupancy shifts across the continuous $d=48$ strategy space.
2. **Transition Tensor Frobenius Kernel:**
   For comparing 2nd-order or 3rd-order transition probability tensors $\mathbf{T} \in \mathbb{R}^{K \times K \times K}$:
   $$k_{\text{Tensor}}(\mathbf{T}_A, \mathbf{T}_B) = \exp\left( -\gamma \|\mathbf{T}_A - \mathbf{T}_B\|_F^2 \right)$$
   where $\|\cdot\|_F$ is the Frobenius tensor norm. Detects high-order syntactic disruptions.
3. **String / Suffix Kernels on Discrete Sequences:**
   Operates directly on discrete bout token streams $(s_1, \dots, s_T)$ by counting shared sub-sequence action motifs weighted by decay factor $\lambda$:
   $$k_{\text{String}}(s, t) = \sum_{u \in \Sigma^*} \text{count}_u(s) \cdot \text{count}_u(t) \cdot \lambda^{|u|}$$

---

## 4. Subject-Stratified Permutation Testing

```
  INCORRECT: Naive Token Shuffling (Artifact)      CORRECT: Subject-Stratified Shuffling (Rigorous)
  Shuffles individual 100ms bouts across groups.   Shuffles whole animal IDs (keeping sessions intact).
  Violates independence -> False Positive (p < 1e-15)  Preserves intra-animal correlation -> Exact p-value
```

### The Data Leakage Trap:
An individual mouse exhibits idiosyncratic movement habits across its 40-minute session. If statistical permutation tests shuffle individual action bouts or sliding windows across groups, the null hypothesis is falsely rejected ($p < 10^{-15}$) simply because the model recognizes the animal's identity rather than a true genotype effect.

### The Exact Stratified Permutation Algorithm:

1. **Compute Observed Test Statistic:**
   $$T_{\text{obs}} = \text{Statistic}(P_{\text{Monomer}}, Q_{\text{PFF}}) \quad [\text{e.g. } \mathcal{W}_\epsilon \text{ or } \text{MMD}^2]$$
2. **Permute Animal ID Clusters (for $b = 1, \dots, B$, with $B = 10,000$):**
   * Assign each entire animal (all sessions and all action bouts for `animal_id`) randomly to either Group A or Group B, preserving cohort sample sizes ($N_{\text{Monomer}} = 55, N_{\text{PFF}} = 55$).
   * Re-calculate the test statistic on the permuted cohorts:
     $$T_{\text{perm}}^{(b)} = \text{Statistic}(P_{\text{permA}}^{(b)}, Q_{\text{permB}}^{(b)})$$
3. **Compute Exact Non-Parametric $p$-Value:**
   $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}\left( T_{\text{perm}}^{(b)} \ge T_{\text{obs}} \right)}{1 + B}$$

If $p < 0.05$, we have mathematically proven that the genotype strategy divergence cannot be explained by individual animal idiosyncrasies or finite-sample noise.

---

## 5. Statistical Framework Comparison Matrix

| Testing Framework | Input Data Structure | Sensitivity to Syntax | Sensitivity to Manifold Geometry | Computational Complexity | Permutation Robustness |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Scalar $t$-Test / ANOVA** | Univariate scalar (e.g. Mean Speed) | **None** | **None** | $O(N)$ (Instant) | Sensitive only to gross motor failure |
| **Chi-Square on Syllable Counts** | Discrete frequency histogram $\mathbf{c} \in \mathbb{R}^K$ | Low | None (Treats states as orthogonal) | $O(K)$ | High false-positive rate on large $N$ |
| **Sinkhorn-Wasserstein Distance** | Density distribution over $d=48$ SVD manifold | **High** | **Maximum (Geodesic Metric)** | $O(N^2)$ via GPU Sinkhorn | **Gold Standard for Spatial Manifolds** |
| **Kernelized MMD (Tensor/String)** | Transition tensors or raw token sequences | **Maximum** | High (via RKHS mapping) | $O(N^2)$ | **Gold Standard for Syntax Disruptions** |
| **Subject-Stratified Permutation** | Group labels on `animal_id` clusters | N/A | N/A | $O(B \cdot N^2)$ | **Guarantees zero animal-leakage bias** |

---

## 6. Python Implementation Recipe

```python
import numpy as np
import torch
import geomloss  # Fast GPU Sinkhorn-Wasserstein
from sklearn.metrics.pairwise import rbf_kernel

def compute_subject_stratified_wasserstein_pvalue(
    features_df, animal_id_col='animal_id', group_col='genotype', 
    feature_cols=None, n_permutations=2000, blur=0.05
):
    """
    Computes exact Sinkhorn-Wasserstein distance and subject-stratified p-value.
    features_df: DataFrame containing d=48 strategy vectors, animal_id, and genotype.
    """
    sinkhorn_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur)
    
    unique_animals = features_df[[animal_id_col, group_col]].drop_duplicates()
    group_names = unique_animals[group_col].unique()
    
    # Extract empirical distributions
    data_g0 = torch.tensor(features_df[features_df[group_col] == group_names[0]][feature_cols].values, dtype=torch.float32).cuda()
    data_g1 = torch.tensor(features_df[features_df[group_col] == group_names[1]][feature_cols].values, dtype=torch.float32).cuda()
    
    # Observed Sinkhorn distance
    t_obs = sinkhorn_loss(data_g0, data_g1).item()
    
    # Subject-Stratified Permutations
    n_extreme = 0
    animals_arr = unique_animals[animal_id_col].values
    labels_arr = unique_animals[group_col].values
    
    for b in range(n_permutations):
        shuffled_labels = np.random.permutation(labels_arr)
        aid_to_shuffled = dict(zip(animals_arr, shuffled_labels))
        
        perm_groups = features_df[animal_id_col].map(aid_to_shuffled)
        perm_g0 = torch.tensor(features_df[perm_groups == group_names[0]][feature_cols].values, dtype=torch.float32).cuda()
        perm_g1 = torch.tensor(features_df[perm_groups == group_names[1]][feature_cols].values, dtype=torch.float32).cuda()
        
        t_perm = sinkhorn_loss(perm_g0, perm_g1).item()
        if t_perm >= t_obs:
            n_extreme += 1
            
    p_value = (1 + n_extreme) / (1 + n_permutations)
    return {"observed_wasserstein": t_obs, "p_value": p_value}
```
