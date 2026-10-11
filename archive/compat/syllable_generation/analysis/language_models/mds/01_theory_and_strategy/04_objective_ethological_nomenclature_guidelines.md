# Strict Ethological Nomenclature & Objective Descriptor Guidelines

**Status:** Mandatory Repository Protocol  
**Location:** `syllable_generation/analysis/language_models/mds/01_theory_and_strategy/04_objective_ethological_nomenclature_guidelines.md`  
**Applies to:** All code, analyses, visualizations, slide deck prep, tables, and AI assistant outputs.

---

## 1. Core Principle: No Unsubstantiated Behavioral Labeling

> [!CAUTION]
> **Only certified ethologists may assign functional, semantic, or cognitive behavioral labels to unsupervised clusters, syllables, or NMF components.**  
> Ground-truth validation will be conducted by trained human observers performing blinded video scoring at a later project phase.

Until formal ethological validation is performed, **never** label components, syllables, or latent modes using anthropomorphic, functional, or subjective behavioral terms.

---

## 2. Terminology Conversion Matrix

| Strictly Prohibited (Subjective / Functional) | Required Objective Physical / Statistical Replacement |
| :--- | :--- |
| ❌ "Exploration mode" | ✅ "High velocity, spatially dispersed arena coverage" |
| ❌ "Grooming mode" / "Resting mode" | ✅ "Low centroid velocity, localized thigmotaxic cluster" |
| ❌ "Foraging" / "Investigation" | ✅ "Corner-localized low speed with repeated vertical displacement" |
| ❌ "Stereotypy" (as a descriptive label for a motif) | ✅ "High-frequency repetitive single-syllable sequence" / "Low-entropy transition loop" |
| ❌ "Cognitive strategy" / "Planning" | ✅ "Multi-second temporal dependence" / "Non-Markovian history gain $\Delta_{\text{history}}$" |
| ❌ "Novel behaviors" | ✅ "Condition-specific low-occupancy syllable subsets" |
| ❌ "Component 7 = Head-bobbing" | ✅ "Component 7 (characterized by high loading on Syllables 14, 22, 31 with mean elevation $z$)" |

---

## 3. Approved Objective Dimensions

When describing a syllable, NMF component, or behavioral state, restrict descriptions strictly to measurable physical and kinematic coordinates:

1. **Centroid Kinematics:**
   * Mean / variance of translation velocity (speed in cm/s or px/frame)
   * Angular velocity / heading change rate (rad/s)
   * Elevation / vertical displacement ($z$-coordinate relative to arena floor)
2. **Spatial Distribution & Arena Occupancy:**
   * Radial distance from arena center ($r$)
   * Center vs. wall/corner occupancy ratio (thigmotaxis percentage)
   * Spatial dispersion / coverage area ($cm^2$)
3. **Statistical & Sequence Properties:**
   * Syllable duration (mean bout length in frames or ms)
   * Syllable transition entropy (transition matrix sparsity)
   * Repetition frequency / self-transition probability $P(S_t = k \mid S_{t-1} = k)$
   * Component identity: refer objectively to **Component $k$**, **Mode $m$**, or **Cluster $c$**.

---

## 4. Preservation Rule for Automated Models

When reporting mathematical results:
* Distinguish between **model predictability** ($% \Delta D_{\text{JS}}$) and **intrinsic complexity** (Participation Ratio rank, entropy rate).
* High predictability must be reported neutrally as **"constrained behavioral transition dynamics"** or **"reduced state entropy"**, not as "strategic behavior."
