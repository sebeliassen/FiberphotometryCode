# Presentation Blueprint: "What Ethologists Can Do With Hierarchical Behavioral Modeling"

**Target Audience**: Behavioral Ethologists, Neuropharmacologists, and Systems Neuroscientists  
**Framing**: A pragmatic, capability-driven pitch: *"Here is what you, the ethologist, can discover and measure with this technology."*  
**Companion Research Prompt**: Formatted for deep literature search tools (Consensus, Elicit, Perplexity, GPT/Gemini Deep Research).

---

## 1. The 6-Step Narrative Arc

```
1. Trajectory ──► Pose ──► Syllables  (Increasing behavioral resolution)
2. Are Syllables Enough?              (Weinreb lagged MI: Real vs. Markov memory peak at ~6s)
3. Applying shMoSeq to the Cohort    (What states capture: videos + syllable compositions)
4. Making it Biological               (DAT-KI & Amphetamine: state occupancy, persistence, syntax)
5. What Added Capacity Buys Us       (Hierarchical state splitting from K=4 to K=8 to K=12)
6. Cross-Species Context & Photometry (Fly hierarchy precedents + coupling to striatal dopamine)
```

---

### Slide 1: The Progression of Behavioral Detail — Trajectories $\to$ Pose $\to$ Syllables
* **The Ethologist's Problem**: Macroscopic metrics (distance traveled, mean velocity, center/periphery occupancy) are blind to action structure. A mouse running in stereotypic circular loops and a mouse actively exploring objects can have identical distance and speed curves.
* **The Technological Staircase**:
  1. *Trajectories (Centroid)*: Coarse macroscopic motion.
  2. *Pose Estimation (DeepLabCut / 3D)*: High-dimensional skeletal kinematics across body parts.
  3. *Action Syllables (Keypoint-MoSeq)*: Unsupervised, sub-second discrete primitives (~300–400 ms: *rear-up*, *sniff*, *dart*, *paw-lick*).
* **The Payoff for Ethologists**: Turns hours of continuous video into an interpretable, quantitative "alphabet" of physical postures.

---

### Slide 2: Are Syllables Enough? Proving Multi-Second Behavioral Memory
* **The Hidden Assumption in Syllable Syntax**: Standard models treat syllable sequences as a 1st-order Markov chain ($P(S_{t+1} \mid S_t)$)—assuming the animal resets its memory every 300 milliseconds.
* **The Real vs. Markov Diagnostic (Weinreb et al. Benchmark)**:
  * We compute Lagged Mutual Information $I(S_t; S_{t+\tau})$ out to hundreds of seconds.
  * *1st-Order Markov Null (Black Dashed)*: Memory collapses toward zero in $<15\text{ seconds}$.
  * *Real Mouse Data (Orange Solid)*: Retains significant excess memory peaking at **$\tau \approx 6.3\text{ seconds}$** ($+0.11\text{ bits}$).
* **Takeaway for Ethologists**: Syllables are not chosen in a vacuum; they belong to **~6-second behavioral bouts** driven by internal latent states.

---

### Slide 3: What Do Macro-States Actually Capture? (Videos & Syllable Fingerprints)
* **Demystifying shMoSeq**: Showing what a macro-state physically looks like in the arena:
  * Synchronized video grid clips demonstrating that a single macro-state (e.g. State 6) reliably groups exploratory rearing, directed locomotion, and sniffing.
  * Another state (State 1) groups stationary grooming, pawing, and face washing.
* **Syllable Composition Charts**: Stacked bar charts and probability vectors showing each state’s characteristic syllable repertoire.

---

### Slide 4: Making It Biological — Genotype & Psychostimulant Phenotyping
* **Wild-Type vs. DAT-KI Baseline**:
  * Normative exploration (State 6) occupies **$48.3\%$** of time in Wild-Type mice.
  * In DAT-KI mutants, State 6 collapses **11-fold to just $4.4\%$** ($p = 0.015$), with mutants trapped in repetitive sub-regimes.
* **Amphetamine Challenge**:
  * WT mice develop persistent stereotypic locomotive bouts (elevated memory past 60s).
  * DAT-KI mice collapse into ultra-fast, high-rigidity alternating loops ($A \leftrightarrow B$ circling in place).
* **Takeaway for Ethologists**: The model uncovers disease and drug phenotypes as **disruptions in behavioral grammar and bout persistence**, invisible to standard open-field software.

---

### Slide 5: What Added Resolution Buys Us — The Capacity Ladder ($K=4 \to 8 \to 12$)
* **The Question**: *"How do I choose the number of macro-states $K$, and what does higher $K$ give me?"*
* **Recognizable State Splitting**:
  * At $K=4$, we observe broad behavioral buckets: *Active Locomotion*, *Stationary Grooming*, *Wall Interaction*, *Freezing*.
  * At $K=8$, *Active Locomotion* cleanly bifurcates into *Directed Exploration* vs. *Stereotypic Edge Pacing*.
  * At $K=12$, fine-grained phenotypic differences between subtly altered genotypes emerge without sacrificing reproducibility.

---

### Slide 6: Cross-Species Precedents (The Fly Hierarchy) & The Dopamine Bridge
* **Invertebrate Inspiration (The Fly Literature)**:
  * In *Drosophila*, multi-timescale behavioral representations prove that animals naturally move across nested behavioral manifolds (Berman et al., Calhoun/Murthy, etc.).
* **The Striatal Dopamine Bridge**:
  * How these multi-second macro-states map onto nucleus accumbens and striatal dopamine transients measured with fiber photometry.
  * Does dopamine release trigger transitions between macro-states, or sustain ongoing state persistence?

---

## 2. Deep Research Prompt for Your Research Tool

Copy and paste this prompt into your research tool to identify the target fly paper and gather exact supporting literature:

```markdown
You are an expert computational neuroethologist and systems neuroscientist. 

I need an exhaustive literature search and synthesis focused on hierarchical behavioral modeling, non-Markovian syntax, and striatal neuromodulation in computational ethology.

### Immediate Priority: Identify the "2025/2024 Fly Paper"
Please search for and provide the full title, authors, journal, and DOI for recent (2024–2025) influential papers on *Drosophila melanogaster* (fruit fly) behavioral hierarchy, internal states, or multi-timescale behavioral manifold modeling. Specifically consider:
1. Work from Gordon Berman's lab (or collaborators like Bialek, Shaevitz) extending the 2016 PNAS predictive hierarchy framework.
2. Work from Mala Murthy, Jonathan Pillow, or Talmo Pereira on internal states, social behavior, or multi-timescale sequence modeling in flies.
3. Studies demonstrating how fruit fly behavior organizes into nested, multi-timescale hierarchies or context-dependent state transitions, and how this compares to mammalian behavioral syllables (e.g., MoSeq / shMoSeq).

### Thematic Synthesis Requirements

#### Topic 1: From Pose Primitives to Multi-Second Behavioral Bouts
- How do modern ethology pipelines (e.g., Keypoint-MoSeq, VAME, B-SOiD) transition from sub-second action primitives (~300–400 ms) to multi-second task programs (~5–10 s)?
- Cite evidence (specifically Caleb Weinreb / Sandeep Robert Datta, *Neuron* 2026; and Gordon Berman, *PNAS* 2016) proving that animal behavior is non-Markovian with lagged mutual information decaying over seconds.

#### Topic 2: Phenotypic Quantification of Dopaminergic Dysfunction
- How do dopamine transporter mutations (DAT knockdown / knock-in) and psychostimulants (amphetamine) alter behavioral repertoire flexibility, state entropy, and stereotypic perseveration?
- What metrics best quantify the collapse from flexible exploration to rigid alternating or repetitive loops?

#### Topic 3: Hierarchical State Capacity and Granularity
- When increasing the number of discrete latent states (K = 4 -> 8 -> 12) in semi-hidden Markov models (shMoSeq) or hierarchical Dirichlet process HMMs, how do ethologists evaluate whether states split into genuine ethological sub-behaviors versus overfitted noise?

#### Topic 4: Coupling Behavioral Hierarchy to Fiber Photometry
- How do sub-second action syllables and multi-second macro-states correlate with striatal dopamine (dDA/NAc) fluctuations? Does dopamine signaling track state boundaries (switching) or state maintenance (persistence)?

### Deliverable Format:
1. Exact citation and summary of the target 2024/2025 Drosophila hierarchy paper(s).
2. Structured summary of findings across the 4 topics with direct citations (Author, Year, Journal).
3. "Pitch to Ethologists" bullet points translating mathematical concepts (lagged mutual information, Jensen-Shannon divergence, HMM transition matrices) into practical behavioral insights.
```
