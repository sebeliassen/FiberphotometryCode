# SKILL DOC: Dimensionality Reduction & Clustering (UMAP)

This document explains the architecture and mathematics behind the topological module (`lib/clustering.py`) in the `ulrik_kinematics` pipeline. It is designed to act as an AI reference guide when modifying or extending the dimensionality spatial mapping.

## 1. Goal & Execution
The goal of this module is to synthesize independent, distinct features—like absolute physical velocity, structural behavioral entropy, and discrete sequence motif frequencies—into a singular high-dimensional mathematical vector representing an entire 30-minute session. We then utilize **UMAP** (Uniform Manifold Approximation and Projection) to squash this matrix down into a 2-Dimensional topology to visually diagnose Genotype separation.

### Endpoints
The pipeline is natively hooked into the `run_analysis.py` orchestrator:
- `python run_analysis.py --verify-clustering` (Executes the deterministic feature extraction checks and tests topological resistance to sequence corruption).
- `python run_analysis.py --run-clustering-discovery` (Generates spatial UMAP scatter plots showing behavioral isolation of groups).

## 2. Core Mining Logic (`extract_features`)
In order for UMAP to find a non-linear shape in the data, the input logic must construct massive vectors of consistent length for each session.
1. The script first scans the entire array of input sessions, runs the Motif extraction (`motifs.py`) across all of them, and isolates the **Top 30 globally occurring Motifs**.
2. For each specific session, it invokes:
   - `metrics.calculate_kinematics()` to append mean Speed and Acceleration parameters.
   - `metrics.calculate_entropy_metrics()` to append the structural density parameter.
   - It iterates through the global Top 30 list and appends the specific probability of execution for each motif.
3. Because probabilities exist tightly bound between `0.0` and `1.0`, but Physical Velocity exists dynamically up to `100.0+`, all vectors are **Standard Scaled (Z-Scored)** (`sklearn.preprocessing.StandardScaler`). Failure to Z-Score mathematically causes UMAP to exclusively map clusters based solely on mouse sprinting speed, completely overwriting the Motif grammar dimensions.

## 3. The 2-Step Defense Math (`verify_clustering_module`)
Manifold dimensionality reductions like UMAP and t-SNE are notoriously "squishy" algorithms that can easily hallucinate geographic clusters if fed mathematically noisy vectors or unstable hyperparameters (like `min_dist`). To prevent hallucination, the module is gated by two strict constraints:

1. **Deterministic Coordinate Math (Invariance Check)**
   - **Mechanism**: We generate a bit-for-bit `IDENTICAL CLONE` of a baseline session and feed both into the pipeline.
   - **Math**: The algorithm calculates the absolute Euclidean Distance of the normalized feature-extracted Z-scored array between the source and the clone. **It must equal exactly $0.000000$**. This proves that the feature dictionary is 100% stable and fully deterministic (no stochastic dictionary ordering bugs).

2. **Topological Space Ejection (Perturbation Crash)**
   - **Mechanism**: We create a `FRANKENSTEIN_PERT` session by randomly shuffling the exact order of every single behavioral syllable in the session, destroying all Sequence Motifs and Sequence Entropy, but fully maintaining the exact same Physical Velocity coordinates.
   - **Math**: UMAP mechanically attempts to cluster similarities. The pipeline enforces that this Frankenstein session must be geographically ejected off the map (away from the biological Wild-type cluster center). It successfully proves that the UMAP algorithm is aggressively prioritizing Motif Strategy dynamics, rather than just clustering mice by identical speed traits.
