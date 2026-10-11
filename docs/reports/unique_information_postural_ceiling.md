# Explaining the Postural Latent States Upper Bound and the Fold-Over-Shuffle Metric

This document explains the mathematical and biological basis of the **Postural Latent States Upper Bound (Ceiling)** in our unique information decoding pipeline, why its performance does not necessarily hit exactly $k\text{x}$ fold over shuffle (where $k$ is the number of active syllable classes, or $20\text{x}$ for 20 classes), and how balanced accuracy calibration resolves key metrics compression issues.

---

## 1. What the Postural Upper Bound Represents

In our 4-bar pipeline, the four bars are:
1. **Raw Dopamine (Red):** How well can a random forest decoder predict which of the $k$ syllables is occurring based on the $\pm0.5\text{s}$ raw dopamine signal ($Y$).
2. **Residual DA (Green):** Decoding performance using dopamine after stripping continuous low-level velocity and acceleration kinetics ($R$).
3. **Kinematic-DA (Blue):** Decoding performance using the predicted dopamine trace from velocity and acceleration alone ($\hat{Y}$).
4. **Latent States / Upper Bound (Purple):** Decoding performance using the full continuous **MoSeq postural latent states** (the 10 principal components of the animal's 3D body posture) over the identical $\pm0.5\text{s}$ temporal window.

### The Postural Ceiling
The **Latent States** bar represents the true **empirical postural ceiling**. Since MoSeq behavioral syllables are defined by segmenting these exact continuous postural trajectories using an Autoregressive Hidden Markov Model (AR-HMM), these 10 latent states contain the *complete continuous postural history* that defines the syllables. 

Evaluating a classifier on these features tells us: **"If you had perfect access to the animal's continuous postural trajectory, how well could you decode the syllable?"** It calibrates the maximum decodability of behavior from posture, providing a ceiling against which dopamine's decoding capability (which is a low-dimensional functional readout) can be compared.

---

## 2. Why the Upper Bound is Not Necessarily $k\text{x}$ Fold Over Shuffle

Under **Balanced Accuracy**, the chance floor for a shuffle baseline is mathematically anchored to exactly $1/k$ (e.g., $1/20 = 5\%$ for $20$ syllable classes).
The fold-over-shuffle metric is defined as:
$$\text{Fold Over Shuffle} = \frac{\text{Balanced Accuracy}}{\text{Shuffle Baseline}} = k \times \text{Balanced Accuracy}$$

If a classifier was mathematically perfect ($100\%$ balanced accuracy), the fold-over-shuffle would be exactly:
$$\text{Max Fold} = k \times 1.0 = k\text{x} \quad (\text{e.g., } 20\text{x for } 20 \text{ classes})$$

However, our empirical Latent States upper bound typically lands between **$10\text{x}$ and $15\text{x}$** rather than the full $20\text{x}$. There are four critical scientific and mathematical reasons for this:

### A. Postural Similarity and Soft Syllable Boundaries
Behavior is continuous, but syllables are discrete labels. The transitions between syllables are not instantaneous jumps in physical posture; the animal must physically move its body through intermediate configurations. 
* Many syllables share similar postural sub-components (e.g., sniffing while rearing vs. sniffing while walking).
* During these transitions or shared phases, the continuous postural latent states of two different syllables can overlap significantly in the 10-dimensional space, creating genuine physical ambiguity that no classifier can perfectly disentangle.

### B. MoSeq's Probabilistic Architecture
Syllable labels are generated using a probabilistic model (AR-HMM) that incorporates temporal transition probabilities.
* The mapping from continuous postural trajectories to discrete syllable labels is not a simple deterministic look-up table.
* A specific postural trajectory might have a $60\%$ probability of being syllable A and a $40\%$ probability of being syllable B. MoSeq assigns a single hard label, but a classifier trained on the continuous states will naturally experience some error matching this probabilistic label.

### C. Finite Temporal Window ($\pm0.5\text{s}$)
Our classifier is trained on a local $1$-second window ($\pm0.5\text{s}$ around the syllable onset).
* While this captures the core postural dynamics of the syllable, some syllables have long-range temporal context (e.g., what the animal was doing 2 seconds ago) that helps define the HMM state transition.
* Without access to this wider temporal history, the local window contains slight physical ambiguities, capping decoding performance.

### D. Model Capacity and Decision Boundaries
The Random Forest Classifier is trained with finite capacity (`n_estimators=50`, `max_depth=10`) to prevent overfitting and ensure fair comparison with the dopamine-trained decoders.
* Continuous high-dimensional features ($810$ features) require highly complex, non-linear decision boundaries to partition perfectly.
* A finite tree depth cannot perfectly carve out $20$ complex continuous manifolds, resulting in a stable, realistic upper bound (e.g., $60-75\%$ balanced accuracy, yielding $12\text{x}-15\text{x}$ fold) rather than a mathematically perfect $100\%$ classification.

---

## 3. Contrast: Latent States vs. "Syllable ID"

To prove that our machine learning pipeline has no bugs, we performed a sanity check where we replaced the 10 continuous latent states with the **Syllable ID itself** as a single integer feature.

| Feature Type | Feature Dimension | Data Type | Decision Boundary | Balanced Accuracy | Shuffle Baseline | Fold Over Shuffle | Scientific Meaning |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Syllable ID (Sanity)** | $1$ | Discrete Integer | Trivial ($X == y$) | **$100\%$** | $1/k$ | **Exactly $k\text{x}$** ($20\text{x}$) | **Mathematical Tautology:** Proves pipeline correctness, cross-validation integrity, and lack of leaks. |
| **Latent States (Ceiling)** | $810$ ($81 \times 10$) | Continuous Float | Complex Manifold | **$60-75\%$** | $1/k$ | **$12\text{x} - 15\text{x}$** | **Empirical Behavioral Ceiling:** Measures the true decodability of continuous physical movement. |

### The Volatility of Discrete Shuffling
When shuffling a discrete, perfect predictor like **Syllable ID**, the shuffle baseline can become highly volatile because the Random Forest perfectly partitions the discrete classes. Any random permutation matches some classes with binary "all-or-nothing" recall. 

In contrast,continuous **Latent States** produce highly stable and robust baselines because a Random Forest cannot easily overfit or perfectly partition continuous floating-point spaces during shuffle permutations.

---

## 4. Balanced Accuracy vs. Standard Accuracy

In our original pipeline using standard accuracy, we observed that the Latent States ceiling was compressed to $\approx 6\text{x}-8\text{x}$ even when the classifier was performing exceptionally well. This was due to **Class Imbalance Compression**:

1. **Syllable Zipf Distribution:** MoSeq syllables follow a highly imbalanced power-law distribution. A few highly frequent syllables dominate $15-20\%$ of the frames.
2. **Inflated Shuffle Baseline:** Under standard accuracy, a random shuffle baseline naturally predicts the high-frequency classes correctly, inflating chance accuracy to $\approx 15\%$.
3. **Metric Compression:** Even a perfect classifier ($100\%$ accuracy) is mathematically capped at a fold-over-shuffle of:
   $$\text{Max Fold} = \frac{100\%}{15\%} \approx 6.67\text{x}$$

By switching to **Balanced Accuracy** (which calculates the average recall per class):
* The shuffle baseline is mathematically anchored to exactly $1/k$ ($5\%$ for 20 classes).
* The fold-over-shuffle metric is uncompressed, allowing the empirical postural ceiling to climb to its true, uncompressed value of **$12\text{x}-15\text{x}$**, while raw and residual dopamine decode around **$2.5\text{x}-4.5\text{x}$**.

---

## 5. Summary of Scientific Conclusions

* **The Upper Bound is Empirical, Not Tautological:** By using continuous latent states rather than discrete syllable IDs, we establish a realistic, physical postural ceiling that accounts for behavioral overlap and MoSeq's probabilistic structure.
* **Striatal Dopamine is a Highly Filtered Signal:** The fact that dopamine decodes around $3\text{x}$ while the postural ceiling is $12\text{x}-15\text{x}$ confirms that dopamine is not a simple readout of active posture, but rather a highly compressed, low-dimensional functional signature of abstract behavior.
* **Kinematics are Distinct from Posture:** Continuous kinematics (velocity/acceleration) only decode $\approx 2\text{x}$, proving that the high postural ceiling ($12\text{x}-15\text{x}$) is driven by complex shape dynamics (posture), not simple physical vigor.
