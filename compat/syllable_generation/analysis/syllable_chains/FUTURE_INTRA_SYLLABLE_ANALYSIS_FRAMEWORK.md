# Future Research Frontier: Intra-Syllable Micro-Movement Dynamics

> **Document Type:** Research Proposal, Architectural Blueprint, & LLM Prompt  
> **Status:** Roadmap Feature for Future Implementation  
> **Scope:** Recursive Fractal Analysis — Applying Syllable Chain Dynamics Within Individual Syllables (Frame-Level Micro-Movement Analysis)

---

## 1. Conceptual Framing: The Fractal Nature of Behavior

The **Project B Syllable Chain Dynamics** framework operates on two hierarchical tiers:
1. **Syllable Level (Primitives):** Discrete categorical states $s_t \in \{1, \dots, K\}$ emitted by Keypoint-MoSeq (duration $\sim 200$–$500\text{ ms}$).
2. **Chain Level (Routines):** Recurring ordered sequences of syllables $C = [s_1, s_2, \dots, s_L]$ (duration $\sim 1$–$5\text{ s}$).

However, natural biological movement exhibits **fractal / self-similar organization**. An individual syllable is *not* an indivisible atomic point; it is a continuous spatio-temporal trajectory across $T$ video frames:
$$\mathbf{X}_{S} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times D}$$
where $\mathbf{x}_t$ represents the $D$-dimensional pose vector (e.g., 21 keypoint coordinates, PCA posture projections, or velocity/acceleration vectors).

```
HIERARCHY LEVEL 3: Macro-States / Behavioral States (Quiescence, Exploration, Foraging)
       │
       ▼
HIERARCHY LEVEL 2: Syllable Chains / Routines [S1 -> S2 -> S3]  <── (IMPLEMENTED IN PROJECT B)
       │           • Inter-chain transitions, repeats, continuation hazard
       │           • Intra-chain completion, DTW consistency, stage pacing
       ▼
HIERARCHY LEVEL 1: Individual Syllables (KPMS Discrete States)
       │
       ▼
HIERARCHY LEVEL 0: Intra-Syllable Micro-Movements [f1 -> f2 -> ... -> fT] <── (THIS PROPOSAL)
                   • Micro-repeats, stutters, tremors
                   • Micro-completion & early frame truncation hazard
                   • Frame-level temporal warping & pacing dilation
```

**The Core Hypothesis:**  
The exact mathematical and statistical machinery developed for **Chain Dynamics** can be applied **recursively inside individual syllables** by treating each syllable as a sequence of micro-movements or pose frames.

---

## 2. Research Questions & Proposed Functionality

### 2.1 Micro-Consistency: Biomechanical Stereotypy Across Executions
- **The Question:** When a mouse executes Syllable 14 (Rearing) 500 times, how similarly does it execute the trajectory from frame 1 to frame $T$? Does motor variability increase in Parkinson's disease (PFF) or under dopamine agonist challenge (Amphetamine)?
- **Proposed Method:**
  - Dynamic Time Warping (DTW) on high-dimensional keypoint trajectories $\mathbf{X}_i(t)$ and $\mathbf{X}_j(t)$.
  - Multi-dimensional Procrustes analysis and Frechet distance across 3D posture trajectories.
  - Pairwise trajectory correlation matrix $R_{ij}$ within animal and within genotype.

### 2.2 Micro-Completion & Early Frame Abort Hazard
- **The Question:** Does a mouse complete the full movement trajectory of a syllable, or does it stop midway? If an action ends prematurely, at which kinematic milestone does it truncate?
- **Examples in Disease:**
  - In *Rearing* ($\sim 30$ frames): Does a PFF mouse abort at frame 10 (hesitation during hindlimb extension) rather than reaching peak height?
  - In *Forward Dash* ($\sim 18$ frames): Does an animal abort before foot-plant, switching to an emergency orient?
- **Proposed Method:**
  - Identify canonical micro-landmarks (e.g., peak vertical velocity $v_z$, peak spine curvature $\kappa_\text{max}$, foot-strike).
  - Compute a **Frame Hazard Function**:
    $$h_\text{micro}(t) = P(\text{Syllable terminates at frame } t \mid \text{Syllable reached frame } t)$$
  - Compare hazard curves between Wildtype vs. DAT-Het KI and Monomer vs. PFF.

### 2.3 Micro-Variations: Frame Insertions, Omissions, and Stutters
- **The Question:** When a syllable varies from its canonical execution, what kind of structural variation occurs?
  - **Micro-Stutters / Tremors:** Are specific frames or sub-phases repeated (e.g., repeated head-bobbing oscillation within a single sniff syllable)?
  - **Micro-Omissions / Skips:** Does the animal omit the preparation or deceleration phase of an action?
  - **Micro-Insertions:** Are extraneous micro-movements (e.g., a lateral head twitch) injected midway through an otherwise stereotyped stroke?
- **Proposed Method:**
  - Discretize pose space into micro-states (e.g., via Vector Quantization / VQ-VAE or GMM on keypoint velocities).
  - Apply string alignment (Needleman-Wunsch / Levenshtein) on micro-token sequences to quantify insertion, omission, and stutter rates.

### 2.4 Micro-Pacing & Temporal Dilation Profiles
- **The Question:** Which specific temporal phases of a syllable take longer in affected groups?
  - In bradykinesia (Parkinson's), is the whole syllable uniformly slowed down, or is there an asymmetric delay during **initiation** (frames $0$–$25\%$) versus **peak ballistic execution** (frames $25$–$75\%$)?
- **Proposed Method:**
  - Non-linear temporal warping functions $w(t)$ computed via DTW against the group median template:
    $$w: [0, 1] \to [0, 1]$$
  - The derivative $w'(t)$ represents instantaneous temporal elasticity:
    - $w'(t) > 1$: Local acceleration / compression.
    - $w'(t) < 1$: Local deceleration / prolonged dwell time (hesitation).

---

## 3. Proposed Software Architecture: `IntraSyllableAnalyzer`

To integrate seamlessly with the existing `syllable_chains` package, future implementations should introduce an `IntraSyllableAnalyzer` class following the established design pattern:

```python
class IntraSyllableAnalyzer:
    """
    Quantifies micro-movement dynamics, execution consistency, frame-level
    hazard, and temporal dilation within individual behavioral syllables.
    """

    def __init__(
        self,
        target_syllables: List[int],
        keypoints_dict: Dict[str, np.ndarray],   # Shape: (N_frames, N_keypoints, 2 or 3)
        syllables_dict: Dict[str, np.ndarray],   # Shape: (N_frames,)
        fps: float = 60.0
    ):
        self.target_syllables = target_syllables
        self.kp = keypoints_dict
        self.sylls = syllables_dict
        self.fps = fps

    def extract_syllable_trajectories(self, syll_id: int) -> List[SyllableInstance]:
        """Extracts continuous keypoint, velocity, and curvature slices for all occurrences."""
        ...

    def compute_micro_consistency(self, syll_id: int) -> pd.DataFrame:
        """Computes pairwise multi-dimensional DTW distance and Frechet distance across instances."""
        ...

    def compute_frame_hazard_curves(self, syll_id: int, max_frames: int = 50) -> pd.DataFrame:
        """Computes conditional hazard of early termination across normalized frame time."""
        ...

    def compute_micro_pacing_curves(self, syll_id: int) -> pd.DataFrame:
        """Extracts warping derivative w'(t) to identify localized deceleration / hesitation phases."""
        ...

    def quantify_micro_stutters_and_omissions(self, syll_id: int) -> pd.DataFrame:
        """Quantifies oscillatory repetitions (tremors) and skipped sub-phases."""
        ...
```

---

## 4. Prompt & Tasks for AI Research Models

When handing this research package to an external AI model (e.g., Claude 3.5 Sonnet, GPT-4o, Gemini Deep Research), you can prompt the model with the following directives:

```text
You are provided with the complete Syllable Chain Dynamics framework (Project B), which 
quantifies higher-order behavioral organization across syllables.

Your task is to design and implement the next frontier: INTRA-SYLLABLE MICRO-MOVEMENT DYNAMICS.

Specifically:
1. Review `FUTURE_INTRA_SYLLABLE_ANALYSIS_FRAMEWORK.md` and the existing `intra_chain.py` 
   and `core.py` architectures.
2. Formalize the mathematical representation of intra-syllable micro-states:
   - Should we discretize continuous keypoint trajectories into Vector-Quantized (VQ) tokens,
     or operate directly on continuous Riemannian manifold curves?
3. Define the exact algorithm for:
   - Frame-level completion and hazard rates h(t).
   - Micro-movement stutters (e.g., oscillatory tremors during quiescent syllables).
   - Local temporal elasticity w'(t) to detect phase-specific bradykinesia.
4. Provide a modular Python implementation of `IntraSyllableAnalyzer` that plugs directly 
   into `syllable_chains/core.py` and produces unified summary plots matching `visualizer.py`.
```
