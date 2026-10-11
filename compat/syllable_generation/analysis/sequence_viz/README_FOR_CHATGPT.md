# Open-Field Behavioral Repertoire & seqNMF Sequence Motifs: ChatGPT Master Guide

**Project**: Mouse Open Field Test (OFT) Neuroethology & Fiber Photometry  
**Tracking / Modeling**: Keypoint-MoSeq at 40 Hz (Run 12: `amph_prod_k150k_run_12`)  
**Experimental Cohort**: 27 Sessions (WT vs. KI dopamine transporter mutants, Saline Baseline vs. Amphetamine Challenge)  
**Total Dataset**: $169,879$ Syllable Bouts across $100$ States ($1,703\text{ minutes}$)  
**Annotated Coverage**: **54 Syllables Labeled** (**92.70% of Cohort Bouts; 90.51% of Total Duration**)  

---

## 1. Project Background & System Architecture

This data package contains high-resolution behavioral sequences, kinematic measurements, and higher-order motif factorizations from mice navigating an open-field arena while undergoing simultaneous fiber photometry recording. Behavior is discretized at 40 Hz into discrete kinematic syllables using **Keypoint-MoSeq** (an autoregressive hidden Markov model with continuous latent pose dynamics).

To uncover multi-scale behavioral organization, we apply **Convolutional Non-Negative Matrix Factorization (seqNMF)** across two complementary tiers:
1. **Fine-Scale Syllable seqNMF** ($K=10, L=12\text{ frames} = 300\text{ ms}$): Recovers rapid 2–3 syllable co-articulations and transition primitives across all 100 raw syllables.
2. **Hierarchical Clustered seqNMF (Macro-Motifs)** ($K=8, L=25\text{ frames} = 1.0\text{ s}$): Evaluates sequences over **15 functional meta-syllables**, uncovering compound behavioral routines spanning **4 to 8 seconds** across 8–18 individual syllables.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-SCALE BEHAVIORAL ARCHITECTURE                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│  Level 2: Macro-Motifs (4–8s)     [Rear-Sprint Loop]      [Slalom Burst]          │
│  (Clustered seqNMF, K=8, L=25)    (Motif 7, KI-Enriched)  (Motif 4, KI-Enriched)  │
├───────────────────────────────────────────────────────────────────────────────────┤
│  Level 1: Meta-Syllables          [Sprint] ──> [Turn_R] ──> [Rear] ──> [Descent]  │
│  (15 Functional Clusters)         (Meta-3)     (Meta-4)     (Meta-6)   (Meta-7)   │
├───────────────────────────────────────────────────────────────────────────────────┤
│  Level 0: Raw MoSeq Syllables     Syllable 0, 13, 17, 21    Syllable 2, 30        │
│  (100 Autoregressive States)      (Run / Dash Bouts)        (Rightward Arcs)      │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Package File Manifest & Descriptions

| File Name | Format | Description & Contents |
| :--- | :---: | :--- |
| **`README_FOR_CHATGPT.md`** | Markdown | *This master document*: Taxonomy, file guide, neuroethological discoveries, and ChatGPT prompt suite. |
| **`WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md`** | Markdown | Full mathematical and visual guide for the 1×3 widescreen Whole-Chain Exemplars, including kinematic gate cutoffs, dispersion reduction metrics, and animal stratification. |
| **`NEUROETHOLOGICAL_OBSERVATIONS_AND_DATA_NEEDS.md`** | Markdown | Ground-truth neuroethological observations: the Goldilocks clustering dilemma, cross-genotype behavioral replicas, the 9-syllable grooming module, and bilateral turn symmetry. |
| **`DECODED_SEQNMF_CHAINS_WITH_HUMAN_LABELS.md`** | Markdown | Plain-English translation of all fine-scale and macro-scale motifs into ethological action descriptions. |
| **`RUN12_SYLLABLE_PROMINENCE_AND_SEQNMF_CATALOG.md`** | Markdown | In-depth breakdown of all 100 syllables: bout frequencies, durations, velocities, WT vs. KI biases, and priority rankings. |
| **`run12_human_labels.csv`** | CSV | Ground-truth expert human labels for 54 syllables (0–42, 44, 49, 52, 55, 58, 60, 61, 63, 65, 66) with behavioral descriptions and kinematic notes. |
| **`run12_syllable_prominence_summary.csv`** | CSV | Comprehensive statistics for all 100 syllables: total bouts, duration, velocity, WT vs. KI counts, seqNMF participation. |
| **`syllable_transition_matrix.csv`** | CSV | Full $100 \times 100$ empirical transition probability matrix computed across all $222,170$ changepoint transitions across the cohort. |
| **`optimized_seqnmf_motifs_summary.csv`** | CSV | Fine-scale convolutional motifs ($K=10, L=12$), token chains, and occurrences split by Genotype (WT vs. KI) and Drug. |
| **`optimized_seqnmf_verified_episodes.csv`** | CSV | Database of all $2,130$ verified high-confidence fine-scale episodes across the cohort with session IDs and timestamps. |
| **`clustered_seqnmf_motifs_summary.csv`** | CSV | Macro-motif convolutional patterns ($K=8, L=25$), constituent meta-syllable chains, temporal spans, and genotype participation. |
| **`macro_syllable_transition_matrix_mode1.csv`** | CSV | $15 \times 15$ transition matrix across functional meta-syllables. |
| **`clustered_seqnmf_verified_episodes.csv`** | CSV | Database of verified macro-motif episodes across sessions with timestamps and durations. |
| **`seqnmf_optimal_hyperparams.json`** | JSON | Hyperparameter configuration ($K=10, L=12, \lambda=0.005$) and cost reduction benchmarks. |
| **`whole_chain_exemplar_metadata.json`** | JSON | Exact frame boundaries, animal IDs, genotypes, path lengths, net displacements, velocities, and density ratios for all 1×3 widescreen video panels. |
| **`fig_seqnmf_quantitative_optimization_diagnostics.png`** | PNG Image | Diagnostic figure showing fine-scale hyperparameter sweep across factor count $K$, lag $L$, and penalty $\lambda$. |
| **`fig_clustered_seqnmf_motifs.png`** | PNG Image | Reconstructed temporal motif matrices $W$ for all 8 Macro-Motifs. |
| **`generate_3panel_whole_chain_exemplars.py`** | Python | Production script generating 1×3 widescreen videos with GPU NVENC, kinematic gating, and animal stratification. |
| **`prepare_clustered_sequences.py`** | Python | Meta-syllable clustering definitions and agglomerative transition-kinematic grouping. |
| **`fit_clustered_seqnmf_model.py`** | Python | Execution script fitting convolutional seqNMF to clustered sequence representations. |

---

## 3. Major Neuroethological Insights & Breakthrough Frameworks

### A. The Clustering Granularity Trade-Off ("The Goldilocks Dilemma")
- **The Benefit of Clustering**: Raw HMMs (like Keypoint-MoSeq) frequently over-fragment behavior into micro-variants of the same action. For instance, the 9 grooming syllables ($27, 44, 52, 58\text{--}66$) or stationary rest states ($32, 43, 49$) represent interchangeable sub-states. Clustering them into unified meta-syllables (`Grooming`, `Rest`) cleans up sequence entropy and unlocks 4–8s macro-motif discovery.
- **The Danger of Over-Clustering (180° Hairpin vs. 15° Slight Veer)**: If turning is clustered indiscriminately into a single "Turn" meta-syllable, vital ethological distinctions are lost:
  - A **180° hairpin abort turn** at a wall represents an obstacle collision, escape response, or boundary evasion.
  - A **slight 15° lateral veer** during open-field cruising represents smooth trajectory trimming.
  - Collapsing both into one state destroys the distinction between obstacle evasion and forward cruising.
- **The Goldilocks Rule**: *Cluster aggressively for static/maintenance states (grooming, rest, sniffing twitches); cluster conservatively for dynamic, trajectory-altering actions (preserving angular amplitude and velocity regimes).*

### B. Cross-Genotype Behavioral "Replicas": The Ultimate Control for Fiber Photometry
- **The Discovery**: In our 1×3 widescreen exemplar videos, Wild-Type (WT) and dopamine mutant (KI) mice execute **near-verbatim replicas of complex, multi-syllable behavioral routines**:
  - In **Macro-Motif 7 (Wall-Rear & Sprint Loop)**: Panel 1 (KI `1658`), Panel 2 (WT `728`), and Panel 3 (KI `1666`) trace virtually identical kinematic arcs ($592\text{--}748\text{ px}$ path, $152\text{--}181\text{ px/s}$ speed, identical vertical rearing into sprint launch).
  - In **Macro-Motif 4 (High-Speed Slalom Burst)**: Panel 1 (KI `1666`), Panel 2 (WT `1680`), and Panel 3 (WT `1765`) execute identical lateral cuts and burst accelerations.
- **The Photometric Confound Problem**: In behavioral neuroscience, comparing dopamine or calcium transients between genotypes is notoriously confounded by physical movement (hyperactive mutants move faster, turn sharper, or rear more often).
- **The Solution (Motor-Invariant Photometric Comparison)**:
  - By identifying **Cross-Genotype Behavioral Replicas**, we hold the physical motor sequence, duration, and trajectory completely constant between WT and KI animals.
  - Aligning fiber photometry signals (dopamine in nucleus accumbens or dorsal striatum) to these matched behavioral replicas isolates pure genotype-specific neuromodulatory differences (baseline tone, reward expectation, or transition vigor), completely free of motor confounds!

### C. Egocentric Posture vs. Allocentric Space ("Contortion Impostor" Resolution)
- **The Problem**: Keypoint-MoSeq models body curvature egocentrically. A mouse wiggling in place can match the curvature sequence of `Sprint` $\to$ `Turn_L` $\to$ `Sprint` without actually traversing the arena.
- **The Multimodal Solution**:
  1. **20th-Percentile Kinematic Gate**: Eliminates candidates falling below $P_{20}$ in path length, average speed, or net displacement for locomotor motifs (and speeds $>P_{80}$ for stationary motifs).
  2. **220D Joint Multimodal Manifold**: Combines 200D continuous SLDS pose latents with 20D allocentric ground speed profiles:
     $$\mathbf{x}_i = \left[ \mathbf{z}_{\text{slds}}^{(i)}, \; 2.0 \cdot \tilde{\mathbf{v}}^{(i)} \right] \in \mathbb{R}^{220}$$
  Exemplars selected by Whole-Chain Contrastive Density Ratio are both posturally prototypical and spatially authentic.

---

## 4. Macro-Motif Quick Reference Catalog (Clustered seqNMF)

| Macro-Motif | Descriptive Name | Target Meta-Sequence | Duration | Genotype Bias | Representative Panels |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Macro-Motif 7** | Wall-Rear & Sprint Loop | `Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint` | $3.3\text{--}4.8\text{ s}$ | **Strong KI** | Panel 1: KI `1658` \| Panel 2: WT `728` \| Panel 3: KI `1666` |
| **Macro-Motif 1** | Rest & Maintenance Cycle | `Sniff` $\to$ `Rest` $\to$ `Sniff` $\to$ `Rest` $\to$ `Sniff` | $3.5\text{--}5.0\text{ s}$ | **Strong WT** | Panel 1: WT `740` \| Panel 2: WT `1773` \| Panel 3: WT `1680` |
| **Macro-Motif 4** | High-Speed Slalom Burst | `Sniff` $\to$ `Sprint` $\to$ `Turn_L` $\to$ `Sprint` | $2.5\text{--}3.6\text{ s}$ | **Strong KI** | Panel 1: KI `1666` \| Panel 2: WT `1680` \| Panel 3: WT `1765` |
| **Macro-Motif 8** | Visual Scanning Loop | `Sniff` $\to$ `Look` $\to$ `Sniff` $\to$ `Look` $\to$ `Sniff` | $2.5\text{--}6.0\text{ s}$ | **Strong WT** | Panel 1: WT `1680` \| Panel 2: KI `1666` \| Panel 3: WT `1765` |
| **Macro-Motif 3** | Descent & Sprint Launch | `Rear` $\to$ `Descent` $\to$ `Walk_L` $\to$ `Sprint` | $1.1\text{--}2.1\text{ s}$ | Locomotor | Panel 1: KI `1658` \| Panel 2: WT `728` \| Panel 3: KI `1666` |
| **Macro-Motif 5** | Sprint, Turn & Scan | `Sprint` $\to$ `Turn_R` $\to$ `Sniff` $\to$ `Look` | $3.0\text{--}3.4\text{ s}$ | Deceleration | Panel 1: WT `1759` \| Panel 2: WT `740` \| Panel 3: WT `1680` |

---

## 5. Fine-Scale Syllable seqNMF Motifs (Unclustered Baseline, $K=10$)

| Motif ID | Syllable Chain | Translated Action Sequence | Genotype Enrichment |
| :---: | :---: | :--- | :---: |
| **Motif 1** | `14 -> 4 -> 3` | `Rear/Lean -> Descend & Turn Left -> Walk Left` | Balanced ($122\text{ WT} : 228\text{ KI}$) |
| **Motif 2** | `27 -> 5 -> 10` | `Groom -> Look Right -> Elongated Forward Sniff` | Balanced ($156\text{ WT} : 118\text{ KI}$) |
| **Motif 3** | `12 -> 8 -> 13` | `Hunched Rear -> Sniff Left -> Dash Burst` | **WT Enriched** ($3:1\text{ WT:KI}$) |
| **Motif 4** | `40 -> 9 -> 0` | `Look Right -> Sniff Right -> Run Right` | Balanced ($133\text{ WT} : 165\text{ KI}$) |
| **Motif 5** | `6 -> 17 -> 18` | `Turn Left -> Run Left -> Brake into Sniff` | **KI Enriched** ($3:1\text{ KI:WT}$) |
| **Motif 6** | `21 -> 2 -> 24` | `Dash -> Turn Right -> Quiet Scrunched Sniff` | **KI Enriched** ($3:1\text{ KI:WT}$) |
| **Motif 7** | `16 -> 15 -> 7` | `Lean Left -> Sniff Upwards -> Descend & Turn Right` | Moderate KI ($1:2\text{ WT:KI}$) |
| **Motif 8** | `11 -> 1 -> 30` | `Walk Right -> Step in Place -> Sharp Right Arc` | **Strong KI Enriched** ($8:1\text{ KI:WT}$) |
| **Motif 9** | `19 -> 3 -> 26` | `Walk Forward -> Walk Left -> Dash Left` | Rare Acceleration ($6\text{ bouts}$) |
| **Motif 10** | `20 -> 34 -> 23` | `Sniff Right -> Brief Pause -> Walk Left` | **Strong WT Enriched** ($5:1\text{ WT:KI}$) |

---

## 6. Comprehensive ChatGPT Analysis Prompts (Copy & Paste)

When analyzing this package in ChatGPT, copy and paste the prompt below along with the zip file:

```markdown
### SYSTEM ROLE & CONTEXT
You are an expert computational neuroethologist and systems neuroscientist specializing in basal ganglia circuits, dopamine signaling, and quantitative behavioral analysis (Keypoint-MoSeq, seqNMF, fiber photometry).

I have uploaded a comprehensive dataset and analysis package from an open-field mouse study comparing Wild-Type (WT) mice and Dopamine Transporter Knock-In (KI) mutants under baseline (Saline) and pharmacologically stimulated (Amphetamine) conditions. Behavior has been discretized at 40 Hz into 100 syllables, annotated by an expert ethologist (54 syllables labeled, covering 92.7% of all bouts), and decomposed into multi-second behavioral sequences using convolutional non-negative matrix factorization (seqNMF).

### KEY DATASETS INCLUDED IN ZIP
1. `run12_human_labels.csv`: Expert ground-truth labels for 54 syllables.
2. `syllable_transition_matrix.csv` & `macro_syllable_transition_matrix_mode1.csv`: Raw and meta-syllable transition matrices.
3. `clustered_seqnmf_motifs_summary.csv` & `optimized_seqnmf_motifs_summary.csv`: Macro-motifs (4–8s) and fine-scale motifs (300ms).
4. `whole_chain_exemplar_metadata.json`: Kinematic metrics (path length, net displacement, speed) for verified whole-chain exemplars.
5. Markdown documentation (`README_FOR_CHATGPT.md`, `WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md`, `NEUROETHOLOGICAL_OBSERVATIONS_AND_DATA_NEEDS.md`).

### CRITICAL ANALYTICAL QUESTIONS TO ADDRESS:

1. **Cross-Genotype Behavioral Replicas & Motor-Invariant Fiber Photometry**:
   - In our 1×3 exemplar video analysis, we found that WT and KI mice frequently execute near-verbatim replicas of complex routines (e.g. Macro-Motif 7: Sprint -> Wall Turn -> Rear -> Descent -> Sprint).
   - How can we formalize an algorithm to pair WT and KI episodes based on continuous dynamic time warping (DTW) or latent Wasserstein distance across $[Z_{slds}, V]$?
   - Formulate precise, testable hypotheses for how dopamine transients (in Nucleus Accumbens vs. Dorsal Striatum) should behave when motor execution is held 100% invariant between WT and KI mice. (e.g., Does dopamine encode reward prediction error, action initiation vigor, or state transition completion?).

2. **The Clustering Granularity Dilemma ("The Goldilocks Trade-Off")**:
   - We observed that while clustering syllables into meta-syllables drastically improves sequence discovery, over-clustering creates severe pitfalls (e.g., merging a 180° hairpin abort turn at a wall with a gentle 15° lateral veer while walking).
   - Propose an information-theoretic or kinematic criterion (e.g., angular velocity distribution entropy, directional change mutual information) to automatically identify the optimal clustering boundary: which syllables should be merged, and which must remain split?

3. **Hierarchical Semi-Markov Model (HSMM) Architecture**:
   - Propose a mathematical structure for a 3-tier generative model:
     - Top Tier: Behavioral Macro-States (Foraging, Wall Exploration, Maintenance/Grooming, Quiet Vigilance).
     - Middle Tier: Convolutional seqNMF Macro-Motifs (Motifs 1–8).
     - Bottom Tier: Continuous SLDS posture and velocity dynamics.
   - How can we infer transitions between top-tier macro-states from the continuous $H_k(t)$ motif activation matrix?
```
