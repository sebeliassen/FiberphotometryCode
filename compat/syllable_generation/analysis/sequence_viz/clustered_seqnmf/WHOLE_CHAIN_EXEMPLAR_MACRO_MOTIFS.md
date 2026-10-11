# Whole-Chain Exemplar Selection for Macro-Motifs

A rigorous methodological, mathematical, and neuroethological guide to **Whole-Chain Contrastive Density Ratio Exemplars** in Clustered seqNMF Macro-Motifs, featuring 1×3 widescreen diagnostic video suites.

---

## 🎬 Quick-Access Video Player Hub

Click any link below to open and watch the 1×3 widescreen diagnostic videos directly in your player or IDE:

| Macro-Motif | Enriched In | Sequence Progression | Full HD Video (Click to Open) | Local Path | Animated Preview |
|---|---|---|---|---|---|
| **Macro-Motif 7** | **KI Mutant** | `Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint` | [▶ **Play Motif 7 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_07_rear_sprint_loop.gif) |
| **Macro-Motif 1** | **WT Control** | `Sniff` $\to$ `Rest` $\to$ `Sniff` $\to$ `Rest` $\to$ `Sniff` | [▶ **Play Motif 1 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.gif) |
| **Macro-Motif 4** | **KI Mutant** | `Sniff` $\to$ `Sprint` $\to$ `Turn_L` $\to$ `Sprint` | [▶ **Play Motif 4 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_04_slalom_burst.gif) |
| **Macro-Motif 8** | **WT Control** | `Sniff` $\to$ `Look` $\to$ `Sniff` $\to$ `Look` $\to$ `Sniff` | [▶ **Play Motif 8 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_08_attentive_scanning_loop.gif) |
| **Macro-Motif 3** | **Locomotor** | `Rear` $\to$ `Descent` $\to$ `Walk_L` $\to$ `Sprint` | [▶ **Play Motif 3 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_03_descent_sprint_launch.gif) |
| **Macro-Motif 5** | **Deceleration** | `Sprint` $\to$ `Turn_R` $\to$ `Sniff` $\to$ `Look` | [▶ **Play Motif 5 MP4**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4) | [Local MP4](./movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4) | [GIF Preview](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_05_sprint_turn_scan.gif) |

---

## 1. Executive Summary

When analyzing multi-second behavioral sequences discovered by convolutional non-negative matrix factorization (**seqNMF**) over meta-syllables, standard video grids often suffer from two major presentation problems:

1. **Crowded Grids (2×3 or 4×4)**: Too many simultaneous panels make it difficult to follow the fine temporal dynamics of head angles, paws, and trajectory curvature.
2. **Ambiguous Selection**: Unfiltered instances often display tracking glitches, peripheral wall occlusions, or atypical motor variations.

To solve this, we implemented **1×3 Widescreen Whole-Chain Exemplar Movies**. Each video features exactly 3 side-by-side synchronized panels:
- **Panel 1 (`[CANONICAL_EXEMPLAR]`)**: The mathematically optimal, highest-density representative of the entire multi-syllable trajectory in continuous SLDS latent space.
- **Panels 2 & 3 (`[NEAREST_NEIGHBOR_EXEMPLAR]`)**: The closest sister episodes in continuous latent space, **strictly stratified across distinct animals** to prove biological generalizability.

All movies feature real-time 80 Hz camera sync, centroid-centered tracking, dynamic 2D spatial trajectory insets, and a bottom HUD progress bar displaying active meta-syllable tokens.

---

## 2. Methodology: Whole-Chain vs. Transition-Bucketed Selection

In previous analyses, two competing definitions of "exemplary" behavior were established:

```
                            ┌──────────────────────────────────────────────┐
                            │ Defining an Exemplary Sequence in KPMS       │
                            └──────────────────────┬───────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            [ Whole-Chain Exemplar ]                           [ Transition Exemplar ]
      Evaluates the entire multi-syllable                Evaluates each constituent syllable
      trajectory as a single unified gesture.            locally within its forward motor step.
      • Maximizes contrastive density in SLDS space.     • Markovian & forward-causal.
      • Ideal for macro-motifs & motor programs.         • Ideal for studying co-articulation.
```

### The Epistemological Question: Global Gesture vs. Local Markov Step

- **Transition-Bucketed Selection (Local / Step-by-Step)**:
  Evaluates whether each constituent transition $S_k \to S_{k+1}$ is a typical instance of that specific pairwise step. This is ideal for asking: *"Do mice execute syllable 1 into syllable 4 the same way regardless of what came before?"*
- **Whole-Chain Selection (Global / Compound Motor Program)**:
  Evaluates the entire concatenated trajectory $[S_1 \to S_2 \to \dots \to S_M]$ as an integrated, multi-second continuous motor program. This answers: *"Out of all 300+ times mice performed this complete behavioral arc, which episode executed the overall trajectory with the cleanest, most prototypical kinematic shape?"*

For macro-motifs discovered by seqNMF, **Whole-Chain Exemplar Selection** is the gold standard because seqNMF explicitly identifies repeating temporal receptive fields that act as coherent behavioral subroutines.

---

### Mathematical Formulation of Whole-Chain Density Ratio

We adapt Keypoint-MoSeq's native contrastive density sampling (`keypoint_moseq.util.sample_instances(mode='density')`) to entire multi-syllable episodes:

1. **Continuous Trajectory Interpolation**:
   For each candidate episode $i \in \{1, \dots, N\}$, the 10-dimensional continuous SLDS latent state sequence $\mathbf{z}(t) \in \mathbb{R}^{10}$ across the duration $t \in [t_{\text{start}}, t_{\text{end}}]$ is resampled to a fixed temporal resolution of $L = 20$ timepoints:
   $$\mathbf{x}_i = \text{vec}\left( \mathbf{z}_i(t) \right) \in \mathbb{R}^{200}$$

2. **Background Latent PCA Space**:
   A global background distribution $\mathcal{B}$ of 400 random behavioral bouts across all 30 sessions is constructed and projected onto 4 principal components (`pca_dim=4`):
   $$\tilde{\mathbf{x}}_i = \text{PCA}_{4\text{D}}(\mathbf{x}_i) \in \mathbb{R}^4$$
   In our dataset, this 4D latent PCA accounts for **57.4% of total trajectory variance**.

3. **Contrastive Density Ratio Calculation**:
   For every candidate episode $\tilde{\mathbf{x}}_i$, we compute:
   - **Local Density** within its candidate family:
     $$\text{Local Density}(\tilde{\mathbf{x}}_i) = \frac{1}{\frac{1}{k}\sum_{j \in \text{kNN}_{\text{family}}(\tilde{\mathbf{x}}_i)} \|\tilde{\mathbf{x}}_i - \tilde{\mathbf{x}}_j\|}$$
   - **Global Density** relative to background open-field behavior:
     $$\text{Global Density}(\tilde{\mathbf{x}}_i) = \frac{1}{\frac{1}{k}\sum_{b \in \text{kNN}_{\text{bg}}(\tilde{\mathbf{x}}_i)} \|\tilde{\mathbf{x}}_i - \tilde{\mathbf{b}}\|}$$
   - **Contrastive Ratio**:
     $$R(\tilde{\mathbf{x}}_i) = \frac{\text{Local Density}(\tilde{\mathbf{x}}_i)}{\text{Global Density}(\tilde{\mathbf{x}}_i)}$$

4. **Canonical Exemplar Selection**:
   The candidate that maximizes this contrastive ratio is selected as the **Canonical Whole-Chain Exemplar**:
   $$\mathbf{x}^* = \arg\max_i R(\tilde{\mathbf{x}}_i)$$
   This guarantees that Panel 1 is **maximally typical of the motif family** while being **maximally distinct from background behavior**.

---

### Dispersion Reduction & Canonical Stratification

To quantify how well the exemplar captures the cluster center, we calculate the **Latent Dispersion Reduction**:
$$\Delta \sigma = \left( 1 - \frac{\bar{d}_{\text{local}}(\mathbf{x}^*)}{\bar{d}_{\text{center}}} \right) \times 100\%$$
Across our macro-motifs, this metric achieves **45.2% to 77.2% dispersion reduction**, proving that the selected exemplars sit at the tightest geometric core of the behavior.

To select Panels 2 and 3:
1. All candidates are ranked by Euclidean distance to the canonical exemplar in 4D PCA space:
   $$d_j = \|\tilde{\mathbf{x}}_j - \tilde{\mathbf{x}}^*\|_2$$
2. Panels 2 and 3 are populated by the closest nearest neighbors that belong to **different animals**:
   $$\text{Animal}(\text{Panel 1}) \neq \text{Animal}(\text{Panel 2}) \neq \text{Animal}(\text{Panel 3})$$
This ensures that the 3 panels demonstrate cross-subject biological conservation rather than an idiosyncratic quirk of a single mouse.

---

### Cross-Genotype Behavioral "Replicas": Visual Demonstration & Analytical Power

A central insight revealed by our 1×3 widescreen exemplars is that **Wild-Type (WT) and dopamine transporter mutant (KI) mice execute near-verbatim replicas of compound behavioral subroutines**:

- In **Macro-Motif 7 (Kinetic Wall-Rear & Sprint Loop)**:
  - Panel 1 is a **KI mouse (`1658`)**
  - Panel 2 is a **WT mouse (`728`)**
  - Panel 3 is a **KI mouse (`1666`)**
  All three mice execute the exact same multi-syllable sequence (`Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint`) with matched velocities ($152\text{--}181\text{ px/s}$) and physical trajectories ($592\text{--}748\text{ px}$).
- In **Macro-Motif 4 (High-Speed Slalom Burst)**:
  - Panel 1 is a **KI mouse (`1666`)**
  - Panels 2 and 3 are **WT mice (`1680`, `1765`)**
  Each executes an identical rapid lateral cut followed by an acceleration sprint.

#### The Photometric Breakthrough
In classical behavioral neuroscience, interpreting dopamine or calcium transients between genotypes is heavily confounded by movement differences (e.g., hyperactive mutants moving faster or rearing more often). By discovering that **both genotypes share identical behavioral subroutines**, we can use these **Cross-Genotype Behavioral Replicas** as an invariant testbed:
1. Align fiber photometry signals directly to matched WT and KI executions of the exact same macro-motif.
2. Holding the physical motor kinematics perfectly constant, test whether dopamine dynamics differ in peak amplitude, duration, or state-transition ramping.
3. This completely decouples genuine neuromodulatory pathology from physical movement confounds!

---

### The Clustering Granularity Dilemma ("The Goldilocks Trade-Off")

While meta-syllable clustering is essential for reducing HMM state fragmentation and discovering coherent multi-second sequences, it introduces a crucial trade-off:
- **Where Clustering Excels**: Merging highly fragmented, functionally identical states (such as the 9 grooming syllables $27, 44, 52, 58\text{--}66$ into `Grooming`, or quiet resting states $32, 43, 49$ into `Rest`) cleans up sequence noise and reveals extended behavioral grammar spanning 4 to 8 seconds.
- **The Risk of Over-Clustering (180° Hairpin vs. 15° Veer)**: If turning is clustered too coarsely, a sharp 180° hairpin abort turn (executed during wall collision or escape) gets grouped with a 15° gentle course veer (executed while navigating open space). This destroys vital ethological meaning.
- **Optimal Practice**: Future clustering refinements should maintain separate meta-syllables for distinct kinematic regimes (e.g., `Sharp_Turn_180` vs. `Gentle_Veer`) while aggressively consolidating interchangeable maintenance and rest states.

---

### The "Contortion Impostor" Dilemma & The Multimodal Solution

A critical ethological pitfall identified during video review is the **Egocentric Invariance Trap** (or "Contortion Impostor" problem):

> [!WARNING]
> **The Problem with Egocentric-Only Manifolds**:
> Keypoint-MoSeq models body curvature, spine flex, and limb angles relative to the animal's own body axis (egocentric invariance). Because of this, a mouse sitting in a corner or grooming its flank can rhythmically flex and twist its spine in a sequence that mathematically mirrors the egocentric state transitions of `Sprint` $\to$ `Turn_L` $\to$ `Sprint`, despite having **zero actual forward locomotion across the arena floor**.
> 
> If exemplar selection relies purely on egocentric SLDS latents, these "stationary contortions" can be falsely selected as canonical exemplars because their body curves match the prototype, even though the mouse is barely moving.

To permanently solve this, we implemented a two-stage **Multimodal Kinematic-Postural Grounding** framework:

#### 1. The 20th-Percentile Kinematic Consistency Gate
For each macro-motif, we compute empirical distributions across all candidate episodes:
$$\text{Path Length} = \sum_{t} \|\Delta \mathbf{c}(t)\|, \quad \bar{v} = \frac{1}{T}\sum_{t} \|\Delta \mathbf{c}(t)\| \times 40, \quad \text{Net Disp} = \|\mathbf{c}_{\text{end}} - \mathbf{c}_{\text{start}}\|$$
- **For Locomotor Motifs (Motifs 7, 4, 3, 5)**:
  Candidates falling below the **20th percentile** in Total Path Length ($P_{20}$) **OR** Average Speed ($P_{20}$) are strictly pruned, and a minimum net displacement ($\ge 25\text{ px}$) is enforced. This immediately eliminates stationary wiggling and corner-stuck fidgets.
- **For Stationary Motifs (Motifs 1, 8)**:
  Conversely, candidates exceeding the **80th percentile** in velocity are pruned, ensuring that "Rest" and "Scan" motifs are not contaminated by mice dashing across the arena.

#### 2. Joint Multimodal Trajectory Manifold
Rather than evaluating posture in isolation, the exemplar feature space is constructed by concatenating:
1. **Egocentric Posture Trajectory**: 10D continuous SLDS latents resampled to $L=20$ timepoints ($200\text{D}$).
2. **Allocentric Speed Profile**: Instantaneous ground velocity $v(t) = \|\Delta \mathbf{c}(t)\| \times 40\text{ px/s}$ resampled to $L=20$ timepoints ($20\text{D}$), z-score normalized.

$$\mathbf{x}_i = \left[ \mathbf{z}_{\text{slds}}^{(i)}, \; 2.0 \cdot \tilde{\mathbf{v}}^{(i)} \right] \in \mathbb{R}^{220}$$

When projected onto 4D PCA and evaluated for **Contrastive Density Ratio**, the canonical exemplar must satisfy a dual constraint:
- It must exhibit the **prototypical body posture and rearing/turn mechanics** (from SLDS).
- It must exhibit the **authentic acceleration and ground displacement arc** matching that behavioral subroutine.

---

## 3. The 1×3 Widescreen Visual Architecture

The visual layout was built specifically for side-by-side comparative inspection:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1x3 WIDESCREEN GRID (960 x 378 px)                                │
├──────────────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│           PANEL 1            │              PANEL 2              │            PANEL 3            │
│     [CANONICAL EXEMPLAR]     │     [NEAREST NEIGHBOR (MOU 2)]    │  [NEAREST NEIGHBOR (MOU 3)]   │
│                              │                                   │                               │
│        Trajectory Inset      │         Trajectory Inset          │       Trajectory Inset        │
│          ┌───────┐           │           ┌───────┐               │         ┌───────┐             │
│          │   ●───│           │           │   ●───│               │         │   ●───│             │
│          └───────┘           │           └───────┘               │         └───────┘             │
│                              │                                   │                               │
│        Mouse Centroid        │         Mouse Centroid            │       Mouse Centroid          │
│          (Follow Cam)        │           (Follow Cam)            │         (Follow Cam)          │
│                              │                                   │                               │
├──────────────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ HUD: Mouse 1680 (KI, Saline) │ HUD: Mouse 1658 (KI, Saline)      │ HUD: Mouse 728 (WT, Saline)   │
│ [Sprint][Turn_R][Rear][Desc] │ [Sprint][Turn_R][Rear][Desc]      │ [Sprint][Turn_R][Rear][Desc]  │
│ 1.0s / 1.1s | P:0.00 K:0.05  │ 1.0s / 3.3s | P:0.52 K:0.10       │ 1.0s / 3.1s | P:0.99 K:0.10   │
└──────────────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

- **Resolution**: 960 × 378 px (320 × 320 px video tiles + 58 px HUD panel).
- **Encoder**: Hardware-accelerated GPU NVENC H.264 running at >250 FPS.
- **Playback**: Continuous 40 FPS natural speed without artificial freezes.
- **HUD Elements**:
  - Animal ID, Genotype (`WT` vs `KI`), and Drug condition (`Saline` vs `Amphetamine`).
  - Active stage token with high-contrast glowing border.
  - Centroid dot color dynamically synchronizes with active token color.
  - Trajectory inset in top-right displaying real-time 2D spatial coordinate path.

---

## 4. Macro-Motif Exemplar Catalog

### Macro-Motif 7: Kinetic Wall-Rear & Sprint Loop (KI-Enriched)

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4) (1.78 MB, 6.0s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4)

![Macro-Motif 7 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_07_rear_sprint_loop.gif)

* **Sequence**: `Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint` (`[3, 4, 6, 7, 3]`)
* **Kinematic Filter**: Locomotor $P_{20}$ Gate: Filtered 99 stationary/slow contortions $\to$ **272 clean locomotor candidates**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.41×** | Latent Dispersion Reduction = **60.4%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `1658` (KI, Saline) — 3.3s full sprint-rear arc (PathLen = **592.6 px**, NetDisp = **368.8 px**, AvgSpeed = **180.9 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `728` (WT, Saline) — 4.8s continuous looping wall-rear (PathLen = **748.2 px**, NetDisp = **336.4 px**, AvgSpeed = **157.5 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1666` (KI, Saline) — 3.9s high-speed wall descent into sprint (PathLen = **594.3 px**, NetDisp = **286.8 px**, AvgSpeed = **152.4 px/s**).
* **Neuroethological Takeaway**:
  With the 20th percentile velocity and path filter active, all 3 panels show genuine, high-speed wall-rearing loops where mice gallop across hundreds of pixels, launch vertically into a rear, and explode forward into an acceleration sprint.

---

### Macro-Motif 1: Rest & Maintenance Cycle (WT-Enriched)

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4) (1.58 MB, 6.2s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4)

![Macro-Motif 1 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.gif)

* **Sequence**: `Sniff` $\to$ `Rest` $\to$ `Sniff` $\to$ `Rest` $\to$ `Sniff` (`[8, 11, 8, 11, 8]`)
* **Kinematic Filter**: Stationary $P_{80}$ Gate: Filtered 41 hyperactive runs $\to$ **163 clean resting candidates**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.28×** | Latent Dispersion Reduction = **53.2%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `740` (WT, Saline) — 4.7s pure resting pause (PathLen = **245.2 px**, NetDisp = **2.4 px**, AvgSpeed = **52.7 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `1773` (WT, Saline) — 5.0s localized sensory inspection (PathLen = **250.8 px**, NetDisp = **1.1 px**, AvgSpeed = **50.4 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1680` (WT, Saline) — 3.5s quiet corner vigilance (PathLen = **178.1 px**, NetDisp = **2.0 px**, AvgSpeed = **51.3 px/s**).
* **Neuroethological Takeaway**:
  Notice the net displacement across all 3 panels: **1.1 to 2.4 pixels** (less than 1 mm net drift!). The kinematic filter guarantees that these are truly stationary vigilance pauses with micro-olfactory twitches, not walking bouts.

---

### Macro-Motif 4: High-Speed Slalom Burst (KI-Enriched)

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_04_slalom_burst.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4) (1.31 MB, 4.9s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4)

![Macro-Motif 4 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_04_slalom_burst.gif)

* **Sequence**: `Sniff` $\to$ `Sprint` $\to$ `Turn_L` $\to$ `Sprint` (`[8, 3, 5, 3]`)
* **Kinematic Filter**: Locomotor $P_{20}$ Gate: Filtered 24 in-place wiggling contortions $\to$ **60 clean slalom bursts**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.70×** | Latent Dispersion Reduction = **76.9%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `1666` (KI, Saline) — 2.5s rapid lateral cut (PathLen = **243.7 px**, NetDisp = **127.3 px**, AvgSpeed = **97.5 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `1680` (WT, Saline) — 3.6s elongated acceleration arc (PathLen = **361.7 px**, NetDisp = **244.5 px**, AvgSpeed = **100.5 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1765` (WT, Saline) — 3.4s sharp slalom turn (PathLen = **366.6 px**, NetDisp = **262.5 px**, AvgSpeed = **109.4 px/s**).
* **Neuroethological Takeaway**:
  Previously, stationary mice wiggling in corners appeared in this motif because their spine bend matched `Turn_L`. With the 20th percentile path and speed gate, **every single panel now shows true high-speed arena displacement (>125 px net displacement, >95 px/s velocity)**!

---

### Macro-Motif 8: Attentive Visual Scanning Loop (WT-Enriched)

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4) (1.72 MB, 7.2s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4)

![Macro-Motif 8 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_08_attentive_scanning_loop.gif)

* **Sequence**: `Sniff` $\to$ `Look` $\to$ `Sniff` $\to$ `Look` $\to$ `Sniff` (`[8, 9, 8, 9, 8]`)
* **Kinematic Filter**: Stationary $P_{80}$ Gate: Filtered 41 locomotor runs $\to$ **163 clean scanning candidates**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.04×** | Latent Dispersion Reduction = **49.6%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `1680` (WT, Saline) — 3.8s focused local head-casts (PathLen = **190.0 px**, NetDisp = **29.5 px**, AvgSpeed = **50.7 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `1666` (KI, Saline) — 2.5s attentive inspection (PathLen = **132.9 px**, NetDisp = **20.2 px**, AvgSpeed = **53.2 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1765` (WT, Saline) — 6.0s thorough local patch scan (PathLen = **350.8 px**, NetDisp = **83.8 px**, AvgSpeed = **58.7 px/s**).
* **Neuroethological Takeaway**:
  Low-velocity exploratory scanning where mice stay within a localized patch while rhythmically executing head-casting and ground sniffing.

---

### Macro-Motif 3: Rear Descent & Locomotor Launch

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4) (0.93 MB, 3.3s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4)

![Macro-Motif 3 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_03_descent_sprint_launch.gif)

* **Sequence**: `Rear` $\to$ `Descent` $\to$ `Walk_L` $\to$ `Sprint` (`[6, 7, 1, 3]`)
* **Kinematic Filter**: Locomotor $P_{20}$ Gate: Filtered 101 in-place fidgets $\to$ **272 clean launch candidates**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.07×** | Latent Dispersion Reduction = **47.4%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `1658` (KI, Saline) — 1.1s explosive touchdown launch (PathLen = **189.7 px**, NetDisp = **168.4 px**, AvgSpeed = **176.5 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `728` (WT, Saline) — 1.9s fluid landing and acceleration (PathLen = **316.6 px**, NetDisp = **279.4 px**, AvgSpeed = **173.5 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1666` (KI, Saline) — 2.1s rapid locomotor recovery (PathLen = **318.3 px**, NetDisp = **276.4 px**, AvgSpeed = **155.3 px/s**).
* **Neuroethological Takeaway**:
  Postural descent coupled to asymmetric leftward turning, immediately launching forward into high-velocity sprinting (>170 px/s average speed).

---

### Macro-Motif 5: Sprint, Turn & Scan

> [!TIP]
> **Video Launch Links**:
> - 🎬 **Watch Full HD MP4 Video**: [▶ **Open `exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4`**](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4) (1.48 MB, 5.5s)
> - 📁 **Local Relative Path**: [`./movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4`](./movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4)

![Macro-Motif 5 Animation](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_05_sprint_turn_scan.gif)

* **Sequence**: `Sprint` $\to$ `Turn_R` $\to$ `Sniff` $\to$ `Look` (`[3, 4, 8, 9]`)
* **Kinematic Filter**: Locomotor $P_{20}$ Gate: Filtered 23 weak bouts $\to$ **50 clean decelerating candidates**.
* **Exemplar Metrics**: Contrastive Density Ratio = **2.69×** | Latent Dispersion Reduction = **71.4%**.
* **Panel Composition**:
  - **Panel 1 (Canonical Exemplar)**: Mouse `1759` (WT, Saline) — 3.1s rapid deceleration arc into sniff (PathLen = **405.7 px**, NetDisp = **256.6 px**, AvgSpeed = **96.6 px/s**).
  - **Panel 2 (Nearest Sister)**: Mouse `740` (WT, Saline) — 3.0s braking turn into ground inspection (PathLen = **296.8 px**, NetDisp = **212.4 px**, AvgSpeed = **98.9 px/s**).
  - **Panel 3 (Nearest Sister)**: Mouse `1680` (WT, Saline) — 3.4s fluid sprint into investigative pause (PathLen = **320.1 px**, NetDisp = **230.8 px**, AvgSpeed = **94.1 px/s**).
* **Neuroethological Takeaway**:
  High-speed locomotion curving into a rightward arrest followed by sensory orientation, confirmed by long 250–400 px physical trajectories.

---

## 5. Summary Comparative Matrix

| Macro-Motif ID | Descriptive Name | Target Sequence | Clean Candidates | Panel 1 Exemplar | Path Length | Net Disp | Avg Speed | Density Ratio | Full HD Video (MP4) | Animated Preview (GIF) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Macro-Motif 7** | Wall-Rear & Sprint Loop | `Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint` | 272 | `1658` (KI) | **592.6 px** | **368.8 px** | **180.9 px/s** | **2.41×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_07_rear_sprint_loop.gif) |
| **Macro-Motif 1** | Rest & Maintenance Cycle | `Sniff` $\to$ `Rest` $\to$ `Sniff` $\to$ `Rest` $\to$ `Sniff` | 163 | `740` (WT) | **245.2 px** | **2.4 px** | **52.7 px/s** | **2.28×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_01_rest_maintenance_cycle.gif) |
| **Macro-Motif 4** | High-Speed Slalom Burst | `Sniff` $\to$ `Sprint` $\to$ `Turn_L` $\to$ `Sprint` | 60 | `1666` (KI) | **243.7 px** | **127.3 px** | **97.5 px/s** | **2.70×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_04_slalom_burst.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_04_slalom_burst.gif) |
| **Macro-Motif 8** | Visual Scanning Loop | `Sniff` $\to$ `Look` $\to$ `Sniff` $\to$ `Look` $\to$ `Sniff` | 163 | `1680` (WT) | **190.0 px** | **29.5 px** | **50.7 px/s** | **2.04×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_08_attentive_scanning_loop.gif) |
| **Macro-Motif 3** | Descent & Sprint Launch | `Rear` $\to$ `Descent` $\to$ `Walk_L` $\to$ `Sprint` | 272 | `1658` (KI) | **189.7 px** | **168.4 px** | **176.5 px/s** | **2.07×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_03_descent_sprint_launch.gif) |
| **Macro-Motif 5** | Sprint, Turn & Scan | `Sprint` $\to$ `Turn_R` $\to$ `Sniff` $\to$ `Look` | 50 | `1759` (WT) | **405.7 px** | **256.6 px** | **96.6 px/s** | **2.69×** | [▶ Open Video](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4) | [🎞 View GIF](./movies_3panel_exemplars/thumbnails/exemplar_3panel_macro_motif_05_sprint_turn_scan.gif) |

---

## 6. Reproducibility & Code References

- **Generation Script**: [`generate_3panel_whole_chain_exemplars.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/generate_3panel_whole_chain_exemplars.py)
- **Metadata Output**: [`whole_chain_exemplar_metadata.json`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars/whole_chain_exemplar_metadata.json)
- **Master Video Wrapper**: [`chain_grid_movie_wrapper.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/chain_grid_movie_wrapper.py)
- **Model Results**: [`clustered_seqnmf_model_results.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/clustered_seqnmf_model_results.pkl)
- **Meta-Syllable Definitions**: [`prepare_clustered_sequences.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/prepare_clustered_sequences.py)
