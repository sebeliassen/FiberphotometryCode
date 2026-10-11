# Decoded seqNMF Behavioral Motif Families: Empirical Episodes & Phrase Syntax

**Dataset / Model**: `amph_prod_k150k_run_12`  
**Ground-Truth Vocabulary**: 54 Human-Annotated Behavioral Syllables (92.70% Cohort Bouts, 90.51% Duration)  
**Hardware & Pipeline**: RTX 4090 GPU (JAX), seqNMF with Cross-Orthogonality Regularization ($K=10, L=3, \lambda=1.0$)  
**Episode Catalogue**: 2,136 Verified High-Confidence Episodes across 10 Baseline Sessions (5 WT, 5 KI Animals)  

---

## 1. Core Principle: Factor Templates vs. Empirical Phrase Families

In convolutional non-negative matrix factorization (seqNMF), each factor $W_k$ represents a soft weight distribution across syllables over temporal lags. A common shortcut is to take the single $\arg\max$ syllable at each lag position to form a 'token chain' (e.g. `6 -> 17 -> 18`). 

However, examining the **actual retrieved episodes** reveals a crucial neuroethological reality:
1. **The Argmax Phantom Trap**: Taking the $\arg\max$ token at each position can construct a synthetic sequence that rarely or never occurs in reality (e.g. Motif 5's template `6 -> 17 -> 18` has zero exact instances, because the mouse almost always counter-steers into `2` [Turn Right] rather than braking into `18` [Sniff]).
2. **Behavioral Phrase Families**: Instead of rigid deterministic chains, seqNMF identifies **phrase families** anchored on a shared core movement root (e.g. $14 \to 4$, $6 \to 17$, $11 \to 1$) that branch into functionally related continuations (e.g. walk vs. run vs. turn).

Below is the ground-truth empirical breakdown of each motif family derived directly from the verified episode records.

---

## 2. Empirical Motif Family Catalog

### Motif 1: The Rear–Descent Locomotor Family (Flagship Presentation Example)
- **Dominant Template**: `[14, 4, 3]` (Total Verified Episodes: $n=350$; $122\text{ WT}, 228\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[14, 4, 3]`: **$95$ episodes ($27.1\%$, 10 animals)** $\to$ `Rear/Lean (14) -> Descend & Turn Left (4) -> Walk Left (3)`
  - `[14, 4, 17]`: **$79$ episodes ($22.6\%$, 10 animals)** $\to$ `Rear/Lean (14) -> Descend & Turn Left (4) -> Run Left (17)`
  - `[14, 4, 6]`: **$37$ episodes ($10.6\%$, 9 animals)** $\to$ `Rear/Lean (14) -> Descend & Turn Left (4) -> Turn Left (6)`
- **Neuroethological Insight**: A shared postural transition ($14 \to 4$) branches into three distinct locomotion intensities. The animal completes an elevated wall exploration or rear (14), touches down with an orienting pivot (4), and immediately commits to walking, sprinting, or turning left.
- **Photometry Alignment Strategy**: Align calcium/dopamine transients to the touchdown moment (Syllable 4) and compare the subsequent branching decision (Walk vs. Run vs. Turn) while controlling for baseline speed.

---

### Motif 5: The High-Speed Counter-Steer / Slalom Family
- **Dominant Template**: `[6, 17, 18]` (Total Verified Episodes: $n=330$; $79\text{ WT}, 251\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[6, 17, 2]`: **$165$ episodes ($50.0\%$, 10 animals)** $\to$ `Turn Left (6) -> Run Left (17) -> Turn Right (2)`
  - `[6, 17, 6]`: **$44$ episodes ($13.3\%$, 9 animals)** $\to$ `Turn Left (6) -> Run Left (17) -> Turn Left (6)`
  - `[6, 17, 21]`: **$32$ episodes ($9.7\%$, 8 animals)** $\to$ `Turn Left (6) -> Run Left (17) -> Dash (21)`
  - `[6, 17, 18]`: **$0$ episodes ($0.0\%$)** *(The argmax template was a phantom!)*
- **Neuroethological Insight**: Rather than decelerating into a sniff, the mouse whips into a high-speed sprint left ($77.2\text{ cm/s}$, 17) and abruptly counter-steers back to the right (2). This rapid slalom trajectory is enriched $>3\times$ in KI mutants ($251\text{ KI} : 79\text{ WT}$).

---

### Motif 8: The Rightward Locomotor Family
- **Dominant Template**: `[11, 1, 30]` (Total Verified Episodes: $n=296$; $33\text{ WT}, 263\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[11, 1, 30]`: **$129$ episodes ($43.6\%$, 9 animals)** $\to$ `Walk Right (11) -> Walk Right (1) -> Sharp Right Turn (30)`
  - `[11, 1, 0]`: **$103$ episodes ($34.8\%$, 9 animals)** $\to$ `Walk Right (11) -> Walk Right (1) -> Run Right (0)`
  - `[18, 1, 30]`: **$15$ episodes ($5.1\%$, 5 animals)** $\to$ `Sniff (18) -> Walk Right (1) -> Sharp Right Turn (30)`
- **Neuroethological Insight**: A rightward locomotion sequence that branches between an acute angular deflection (30, $71.3\text{ cm/s}$) and forward acceleration sprint (0, $76.1\text{ cm/s}$). Calling this 'stereotypic circling' is an over-interpretation without signed heading tracking; it is best described as a **rightward locomotor family with sharp-turn vs. sprint continuations**. Strongly elevated in KI ($263\text{ KI} : 33\text{ WT}$, $8:1$ ratio in saline).

---

### Motif 10: The Investigative Scanning & Rest Family
- **Dominant Template**: `[20, 34, 23]` (Total Verified Episodes: $n=192$; $159\text{ WT}, 33\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[20, 34, 49]`: **$72$ episodes ($37.5\%$, 8 animals)** $\to$ `Sniff Right (20) -> Look Left Subtle (34) -> Stationary Pause (49)`
  - `[20, 34, 8]`: **$49$ episodes ($25.5\%$, 7 animals)** $\to$ `Sniff Right (20) -> Look Left Subtle (34) -> Sniff Left (8)`
  - `[20, 34, 20]`: **$14$ episodes ($7.3\%$, 4 animals)** $\to$ `Sniff Right (20) -> Look Left Subtle (34) -> Sniff Right (20)`
  - `[20, 34, 23]`: **$0$ episodes ($0.0\%$)** *(The argmax template was a phantom!)*
- **Neuroethological Insight**: Now that Syllable 34 is labeled `look left (subtle)` and Syllable 49 is labeled `stationary`, this motif is revealed as a structured **stationary investigative scan**: sniffing, taking a subtle leftward glance, and settling into stationary arrest or re-sniffing. This flexible scanning motif is **$5\times$ enriched in Wild-Type animals** ($159\text{ WT} : 33\text{ KI}$).

---

### Motif 6: The Dash-and-Redirect Family
- **Dominant Template**: `[21, 2, 24]` (Total Verified Episodes: $n=203$; $52\text{ WT}, 151\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[21, 2, 0]`: **$94$ episodes ($46.3\%$)** $\to$ `Dash (21) -> Turn Right (2) -> Run Right (0)`
  - `[21, 2, 14]`: **$27$ episodes ($13.3\%$)** $\to$ `Dash (21) -> Turn Right (2) -> Rear/Lean (14)`
  - `[55, 2, 0]`: **$22$ episodes ($10.8\%$)** $\to$ `Walk Left (55) -> Turn Right (2) -> Run Right (0)`
- **Neuroethological Insight**: High-speed redirection. A dash burst redirects into a rightward turn and launches into a run or terminates in a rear. Enriched $3:1$ in KI mutants.

---

### Motif 7: Supported Wall Exploration & Descent
- **Dominant Template**: `[16, 15, 7]` (Total Verified Episodes: $n=159$; $50\text{ WT}, 109\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[16, 14, 7]`: **$66$ episodes ($41.5\%$)** $\to$ `Lean Left Wall (16) -> Rear/Lean (14) -> Descend & Turn Right (7)`
  - `[5, 15, 7]`: **$30$ episodes ($18.9\%$)** $\to$ `Look Right (5) -> Sniff Up (15) -> Descend & Turn Right (7)`
- **Neuroethological Insight**: Perimeter elevation followed by rightward postural touchdown.

---

### Motif 2: Locomotive / Exploratory Orienting
- **Dominant Template**: `[27, 5, 10]` (Total Verified Episodes: $n=274$; $156\text{ WT}, 118\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[1, 5, 10]`: **$60$ episodes ($21.9\%$)** $\to$ `Walk Right (1) -> Look Right (5) -> Elongated Forward Sniff (10)`
  - `[3, 5, 10]`: **$42$ episodes ($15.3\%$)** $\to$ `Walk Left (3) -> Look Right (5) -> Elongated Forward Sniff (10)`
  - `[10, 5, 10]`: **$39$ episodes ($14.2\%$)** $\to$ `Sniff Ahead (10) -> Look Right (5) -> Sniff Ahead (10)`
  - `[27, 5, 10]`: **$2$ episodes ($0.7\%$)**
- **Neuroethological Insight**: Moving into a head glance right (5) followed by an elongated forward sniff (10).

---

### Motif 3: Low-Kinetic Sniffing Sequence
- **Dominant Template**: `[12, 8, 13]` (Total Verified Episodes: $n=28$; $21\text{ WT}, 7\text{ KI}$)
- **Empirical Retrieved Variants**:
  - `[32, 8, 36]`: **$18$ episodes ($64.3\%$)** $\to$ `Stationary (32) -> Sniff Left (8) -> Stationary Sniff (36)`
  - `[32, 8, 25]`: **$5$ episodes ($17.9\%$)** $\to$ `Stationary (32) -> Sniff Left (8) -> Active Sniff (25)`

---

## 3. Summary of Presentation Takeaways

1. **Focus on Phrase Families**: Present motifs as branching trees (e.g. $14 \to 4 \to \{3, 17, 6\}$) rather than rigid deterministic chains.
2. **Clear Genotype Contrasts**:
   - **KI Mutants**: Enriched in high-speed locomotor families (Motif 8 rightward walk-to-turn/sprint, $8\times$; Motif 5 left sprint to right counter-steer, $3\times$).
   - **WT Animals**: Enriched in stationary investigative scanning (Motif 10 sniff-look-pause, $5\times$).
3. **Pristine Photometry Alignment Question**: Align striatal dopamine signals to Syllable 4 (postural descent) and ask whether phasic DA predicts continuation into walking ($3$) vs. sprinting ($17$) vs. turning ($6$).
---

## 4. Transition to Hierarchical Clustered seqNMF (Macro-Motifs)

While fine-scale seqNMF reveals 300 ms motor phrases (e.g., $14 \to 4 \to 3$), clustering syllables into 15 functional meta-syllables reveals compound **Macro-Motifs spanning 4 to 8 seconds**:
- **Macro-Motif 7**: `Sprint` $\to$ `Turn_R` $\to$ `Rear` $\to$ `Descent` $\to$ `Sprint` (Wall-Rear Sprint Loop, enriched in KI mutants).
- **Macro-Motif 1**: `Sniff` $\to$ `Rest` $\to$ `Sniff` $\to$ `Rest` $\to$ `Sniff` (Rest & Maintenance Cycle, enriched in WT controls).
- **Macro-Motif 4**: `Sniff` $\to$ `Sprint` $\to$ `Turn_L` $\to$ `Sprint` (High-Speed Slalom Burst).

Most importantly, the macro-motif scale reveals **Cross-Genotype Behavioral Replicas**—exact matches where WT and KI mice execute the identical 4-second sequence—providing an ideal motor-invariant control for fiber photometry. See [WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md) for full visual and kinematic details.
