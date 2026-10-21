# Feasibility Assessment: Proposed Computational Neuroethology Modules

This report evaluates five advanced analysis modules proposed to dissect the neural grammar of behavior. It combines technical feasibility assessments with specific experimental rationales for the IFDN and AMPH datasets.

---

## 1. Action Prediction Error (PPE Analysis)
**The Core Idea**: Treat DLS dopamine not as a reward signal, but as a "Performance Prediction Error" (PPE) — a real-time teaching signal for movement fidelity.
*   **Metric**: Variance of the DLS trace slope ($\frac{dF}{dt}$) at syllable onset (Initiation).

### Experimental Rationale (Why?)
*   **IFDN (WT vs KI)**:
    *   **Hypothesis**: The KI mutation introduces "gain noise" into the VTA-to-Striatum loop.
    *   **Prediction**: KI mice will show **higher slope variance** for identical syllables compared to WT. This noisy teaching signal leads to the "sloppy" or inconsistent habits characteristic of ADHD.
*   **AMPH (Saline vs Amph)**:
    *   **Hypothesis**: Amphetamine clamps the dopamine transporter, saturating the signal.
    *   **Prediction**: The slope variance will collapse (stereotypy) or the signal will ceiling out, losing the dynamic range required to encode Performance Error.

### Technical Feasibility (How?)
*   **Feasibility**: ✅ **High**. We already extract syllable-aligned traces.
*   **Implementation**:
    1.  Calculate 1st derivative of traces.
    2.  Extract mean/variance of slope in the [0, 200ms] window.
*   **Novelty**: ⭐ **High**.
*   **Effort**: Medium.
*   **Recommendation**: **Strong Yes**. This is the mechanistic "smoking gun" for the IFDN model.

---

## 2. Syllable Transition Entropy (State Rigidity)
**The Core Idea**: Quantify the "richness" vs "rigidity" of the behavioral dictionary using Information Theory.
*   **Metric**: $H_{sys}$ (Shannon Entropy of the Syllable Transition Matrix).

### Experimental Rationale (Why?)
*   **IFDN (WT vs KI)**:
    *   **Hypothesis**: KI mice suffer from "State Instability" or "Attentional Fragmentation."
    *   **Prediction**: KI mice may show **higher entropy** (random switching) or specific "attractor loops" (compulsions) that differ from WT.
*   **AMPH (Saline vs Amph)**:
    *   **Hypothesis**: Amphetamine induces "State Collapse" (Stereotypy).
    *   **Prediction**: AMPH should drastically **reduce entropy** compared to Saline.
*   **Global Insight**: Entropy provides a single scalar value that places all animals on a spectrum from "Chaotic" (High H) to "Rigid" (Low H).

### Technical Feasibility (How?)
*   **Feasibility**: ✅ **High**. Purely computational on existing MoSeq labels.
*   **Implementation**:
    1.  Build $N \times N$ transition matrix.
    2.  Compute Row Entropy $\rightarrow$ System Weighted Entropy.
*   **Novelty**: ⭐⭐ **Medium**.
*   **Effort**: Low.
*   **Recommendation**: **Yes**. The most robust "State Fingerprint" we can generate.

---

## 3. DLS-VS Spatiotemporal Coordination (The "Handshake")
**The Core Idea**: Motivation (VS) must precede Action (DLS).
*   **Metric**: Peak Lag ($\tau_{peak}$) of the Cross-Correlation between VS and DLS traces.

### Experimental Rationale (Why?)
*   **IFDN (WT vs KI)**:
    *   **Hypothesis**: The IF Nucleus projects to the VS to "gate" transitions.
    *   **Prediction**: KI mice show a **loose coupling** (high lag variance). Motivation does not reliably trigger Action (Impulsivity/Procrastination).
*   **AMPH (Saline vs Amph)**:
    *   **Hypothesis**: Pharmacological flooding bypasses the circuit.
    *   **Prediction**: **Hypersynchrony** (Lag $\rightarrow$ 0). The distinct timescales of VS and DLS merge.

### Technical Feasibility (How?)
*   **Feasibility**: ⚠️ **Medium**. Requires valid DLS+VS signals in the *same* animal.
*   **Implementation**:
    1.  Filter for dual-site animals.
    2.  Bandpass filter (0.5-5Hz).
    3.  Sliding window Cross-Correlation.
*   **Novelty**: ⭐⭐⭐ **Very High**.
*   **Effort**: Medium.
*   **Recommendation**: **High Priority**. sophisticated circuit-level analysis.

---

## 4. Bilateral Synchrony (Lateralization)
**The Core Idea**: Dexterity requires asymmetry (independence between hemispheres).
*   **Metric**: Correlation ($r$) between Left-DLS and Right-DLS during "Turn" syllables.

### Experimental Rationale (Why?)
*   **IFDN (WT vs KI)**:
    *   **Hypothesis**: "Leaky" suppression.
    *   **Prediction**: KI mice fail to suppress the ipsilateral hemisphere during turns ($\uparrow$ correlation), leading to clumsy motor control.
*   **AMPH (Saline vs Amph)**:
    *   **Hypothesis**: Global Flooding.
    *   **Prediction**: **Massive Hypersynchrony** ($\uparrow \uparrow$ correlation) regardless of action.

### Technical Feasibility (How?)
*   **Feasibility**: ⚠️ **Conditional**. Requires bilateral fibers.
*   **Recommendation**: Only if sample size > 3.

---

## Strategic Summary: Implementation Roadmap

| Priority | Module | Context | Why do it? |
| :--- | :--- | :--- | :--- |
| **1** | **Syllable Entropy** | IFDN & AMPH | Best **Global Readout** of "Brain State". Low effort, high robustness. |
| **2** | **Prediction Error** | IFDN Focus | Best **Mechanistic Readout** of "Noisy Dopamine". Directly tests the IFDN hypothesis. |
| **3** | **DLS-VS Handshake** | Circuit Focus | Best **Network Readout**. Reveals the breakdown in "Motivation-to-Action" coupling. |

---

## References & Key Literature

1. **Markowitz et al. (2018)**. [Spontaneous behaviour is structured by reinforcement without explicit reward](https://pubmed.ncbi.nlm.nih.gov/36653449/). *Nature/PubMed*. (Foundational for PPE and Syllable Reinforcement).
2. **Jørgensen et al. (2023)**. [Behavioral encoding across timescales by region-specific dopamine dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9963838/). *PNAS/PMC*. (Essential for DLS-VS Handshake and Lag analysis).
3. **Wiltschko et al. (2020)**. [Revealing the structure of pharmacobehavioral space through Motion Sequencing](https://pmc.ncbi.nlm.nih.gov/articles/PMC7606807/). *Nature Communications/PMC*. (Foundation for Syllable Entropy and AMPH Stereotypy).
4. **Cai et al. (2020)**. [Dopamine-Glutamate Neuron Projections to the Nucleus Accumbens Medial Shell and Behavioral Switching](https://pmc.ncbi.nlm.nih.gov/articles/PMC6855309/). *Cell Reports/PMC*. (Key for IF Nucleus and Switching hypothesis).
5. **Markowitz et al. (2023)**. [Understanding Action - Datta Lab](http://datta.hms.harvard.edu/research/behavioral-analysis/). (General Framework for Computational Neuroethology).
6. **Fox et al. (2016)**. [Cross-hemispheric dopamine projections have functional significance](https://pmc.ncbi.nlm.nih.gov/articles/PMC4922155/). (Relevant for Bilateral Synchrony).
7. **Pan et al. (2014)**. [ADHD, Altered Dopamine Neurotransmission, and Disrupted Reinforcement Processes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4004668/). (Support for Action Prediction Error in ADHD models).
8. **Mohebi et al. (2019)**. [Dissociable dopamine dynamics for learning and performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC8387116/). (Neural basis for DLS-VS differences).
