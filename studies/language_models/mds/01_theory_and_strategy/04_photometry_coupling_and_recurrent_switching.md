# Continuous-Discrete Multi-Scale Coupling: Fiber Photometry & Macro-Strategy Dynamics

## Executive Summary

Behavioral macro-strategies do not switch in isolation; they are driven, shaped, and reinforced by continuous subcortical neuromodulatory dynamics, prominently **dopamine ($\text{dLight1}$)** and **acetylcholine ($\text{ACh3.0}$)**. 

In our experimental paradigm, continuous 2-channel fiber photometry is acquired at 20 Hz (80 Hz optical multiplexing) synchronously with 80 Hz behavioral video. This document details the mathematical formulations and statistical frameworks designed to bridge continuous neuromodulatory timeseries with discrete symbolic action syllables and macro-strategy transitions.

We formulate:
1. **Point Process Generalized Linear Models (Point Process GLMs)** to estimate continuous-to-discrete temporal response kernels.
2. **Recurrent Switching Linear Dynamical Systems (rSLDS)** (Linderman et al.) to model neural hyperplane attractor crossings that trigger strategic switches.
3. **Marked Point Processes & Neural Jump ODEs** to capture event-driven phasic neuromodulator bursts at syllable boundaries.

---

## 1. The Multi-Scale Biophysical Challenge

```
Continuous Photometry (20Hz/80Hz):  dLight1: ───/\──_/\_/\────/\/\─────── (Dopamine Tone)
                                    ACh3.0:  ───────\____/────────────── (Acetylcholine Dip)
                                                   │
                                                   ▼ [ COUPLING FRAMEWORK ]
                                                   │
Discrete Bouts (Level 1: ~400ms):   Syllables:  [  14  ][ 4 ][ 18 ][ 18 ][ 31 ] (Walk->Rear->Sniff)
                                    ────────────┴───────────────────────┴──────────────────────
Macro-Strategies (Level 3: ~30s):   Strategies: [    Exploration Mode    ][    Grooming State    ]
```

### The Rate & Modality Mismatch:
1. **Photometry ($Y(t) \in \mathbb{R}^2$):** Continuous, smooth, non-negative fluorescent timeseries with baseline decay, artifact noise, and sub-second phasic transients.
2. **Behavioral Syllables ($s_m \in \{1, \dots, K\}$):** Discrete, asynchronous, non-Markovian point events with variable physical durations $d_m$.
3. **Macro-Strategies ($\mathcal{S}_w \in \{1, \dots, M\}$):** Low-frequency categorical regime switches occurring every 5 to 60 seconds.

---

## 2. Point Process Generalized Linear Models (GLMs)

Point Process GLMs formulate the instantaneous transition hazard rate $\lambda_k(t)$ of entering behavioral syllable $k$ as a conditional intensity function of continuous neuromodulatory histories.

```
Neuromodulator History: Y(t - tau)  ───►  [ Temporal Kernel w_k(tau) ]  ───►  Linear Predictor eta_k(t)
                                                                                       │
Past Action History: I_j(t - gamma) ───►  [ Autoregressive Kernel h_kj ] ───►          ▼
                                                                              Exp Non-Linearity
                                                                                       │
                                                                                       ▼
                                                                           Hazard Rate: lambda_k(t)
```

### Mathematical Formulation
The conditional intensity $\lambda_k(t)$ of entering discrete syllable $k$ at time $t$ is:

$$\lambda_k(t) = \exp\left( \mu_k + \sum_{m=1}^2 \int_0^{\tau_{\max}} w_{k,m}(\tau) \, y_m(t - \tau) \, d\tau + \sum_{j=1}^K \int_0^{\gamma_{\max}} h_{k,j}(\gamma) \, I_j(t - \gamma) \, d\gamma \right)$$

where:
* $\mu_k$: Baseline log-hazard rate of syllable $k$.
* $y_m(t - \tau)$: Continuous photometry signal for channel $m \in \{\text{dLight1}, \text{ACh3.0}\}$ at lag $\tau$.
* $w_{k,m}(\tau)$: **Neuromodulatory impulse response kernel** linking dopamine/acetylcholine fluctuations to syllable initiation.
* $h_{k,j}(\gamma)$: **Autoregressive behavioral feedback kernel** capturing intrinsic transition syntax.
* $I_j(t) = \sum_i \delta(t - t_{j,i})$: Discrete spike train of syllable $j$ onset times.

### Temporal Basis Function Expansion
To estimate continuous smooth kernels $w_{k,m}(\tau)$ without overfitting, we project onto a set of $B$ raised cosine or cubic B-spline basis functions $\{\phi_b(\tau)\}_{b=1}^B$:

$$w_{k,m}(\tau) = \sum_{b=1}^B \theta_{k,m,b} \, \phi_b(\tau)$$

$$\phi_b(\tau) = \frac{1}{2} \left[ 1 + \cos\left( \frac{\log(\tau + c) - \phi_b^0}{\Delta \phi} \pi \right) \right]$$

* **Causal Kernels ($\tau > 0$):** Measures whether dopamine surges **precede and drive** the initiation of syllable $k$.
* **Anti-Causal Kernels ($\tau < 0$):** Measures whether executing syllable $k$ **triggers subsequent dopamine reward/reinforcement release**.

---

## 3. Recurrent Switching Linear Dynamical Systems (rSLDS)

While standard SLDS assumes that discrete state switches follow an autonomous Markov chain independent of continuous dynamics, **Recurrent SLDS (Linderman et al., 2017)** couples continuous latent neural/neuromodulator states directly to discrete strategy transitions.

```
                    Continuous Photometry / Neural Trajectory (x_t)
                                        │
                         [ Hyperplane Boundary: R_{j,k} x_t + r_{j,k} ]
                                        │
                                        ▼
                  Discrete Strategy Transition: P( z_t = k | z_{t-1} = j, x_t )
                                        │
                                        ▼
                  Switches Dynamical Regime:  x_{t+1} = A_k x_t + b_k + noise
```

### Mathematical Formulation
1. **Continuous State Evolution:** Continuous latent neural/photometry vector $\mathbf{x}_t \in \mathbb{R}^n$ evolves according to linear dynamics governed by active macro-state $z_t \in \{1, \dots, M\}$:
   $$\mathbf{x}_t = \mathbf{A}_{z_t} \mathbf{x}_{t-1} + \mathbf{B}_{z_t} \mathbf{u}_t + \mathbf{v}_{z_t} + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{Q}_{z_t})$$
2. **Recurrent Discrete State Transitions:** Transitions between discrete macro-strategies depend explicitly on the animal's continuous position in neural state-space $\mathbf{x}_{t-1}$:
   $$P(z_t = k \mid z_{t-1} = j, \mathbf{x}_{t-1}) = \frac{\exp\left( \mathbf{R}_{j,k}^T \mathbf{x}_{t-1} + r_{j,k} \right)}{\sum_{l=1}^M \exp\left( \mathbf{R}_{j,l}^T \mathbf{x}_{t-1} + r_{j,l} \right)}$$

### Biological Interpretation: Line Attractors & Strategy Thresholds
* **Mechanism:** Slowly ramping dopamine tone or cholinergic levels act as a 1D continuous integrator (line attractor). 
* **The Threshold Crossing:** As dopamine gradually accumulates during active exploration, $\mathbf{x}_t$ drifts across the hyperplane defined by $\mathbf{R}_{j,k}^T \mathbf{x}_t + r_{j,k} = 0$. 
* **The Strategic Jump:** Crossing this threshold instantly triggers a discrete macro-switch from "Exploratory Foraging" ($z=1$) to "Consummatory / Grooming" ($z=2$), dynamically altering the animal's subsequent behavioral policy.

---

## 4. Marked Point Processes & Neural Jump ODEs

For modeling fast, event-triggered phasic dopamine bursts occurring precisely at syllable transition boundaries, **Neural Jump Ordinary Differential Equations (Jump ODEs)** provide a continuous-time formulation.

```
Continuous Drift (ODE):   d h(t) / dt = f_theta( h(t) )
                                        │
                                        ▼ (Syllable Transition Event at t_i)
Instantaneous Jump:       h(t_i^+) = g_phi( h(t_i^-), s_i, y_photometry(t_i) )
```

### Mathematical Formulation
A continuous latent state $\mathbf{h}(t) \in \mathbb{R}^H$ represents the animal's internal cognitive/neuromodulatory state:

1. **Continuous Deterministic Drift (Between Syllables):** During the execution of a syllable ($t \in (t_i, t_{i+1})$), $\mathbf{h}(t)$ evolves smoothly according to a continuous neural ODE:
   $$\frac{d\mathbf{h}(t)}{dt} = f_{\boldsymbol{\theta}}(\mathbf{h}(t), \mathbf{y}_{\text{photometry}}(t))$$
2. **Discrete Jump Updates (At Syllable Onset $t_i$):** When the animal transitions into new syllable $s_i$ with instantaneous photometry mark $\mathbf{m}_i = \mathbf{y}(t_i)$, the latent state updates discontinuously:
   $$\mathbf{h}(t_i^+) = g_{\boldsymbol{\phi}}(\mathbf{h}(t_i^-), \mathbf{e}_{s_i}, \mathbf{m}_i)$$
   where $\mathbf{e}_{s_i}$ is the learned syllable embedding vector.

---

## 5. Comparative Architecture Summary

| Coupling Framework | Input Modalities | Mathematical Nature | Primary Computational Output | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Point Process GLM** | Continuous Photometry + Syllable Onset Pulses | Convex Poisson / Bernoulli Intensity | Causal & Anti-Causal Temporal Response Kernels $w_k(\tau)$ | Testing if dopamine bursts precede vs. follow specific syllables |
| **Recurrent SLDS (rSLDS)** | Continuous Photometry / Latents + Discrete Strategies | Non-linear Hybrid State-Space Model | Hyperplane Transition Boundaries $\mathbf{R}_{j,k}$ in Neural Space | Modeling threshold-driven macro-strategy switching |
| **Neural Jump ODE** | Continuous Timeseries + Discrete Event Marks | Hybrid Continuous-Discrete Neural ODE | Continuous Latent Cognitive Trajectory $\mathbf{h}(t)$ | Real-time multi-scale state tracking and asynchronous forecasting |

---

## 6. Experimental Validation Hypotheses in Parkinsonian Models

Applying these multi-scale coupling architectures to our **Parkinsonian PFF vs. Monomer** and **DAT-KI vs. WT** cohorts yields three testable biophysical hypotheses:

1. **Dopamine-Syllable Coupling Decoupling in PFF:**
   * *Hypothesis:* In healthy Monomer mice, Point Process GLM kernels $w_{k,\text{DA}}(\tau)$ show strong causal pre-initiation peaks ($\tau \in [-300\text{ms}, 0]$) preceding high-effort syllables (rearing, fast running). In PFF mice, these causal kernels will be blunted or absent, indicating a failure of nigrostriatal dopamine to invigorate action initiation.
2. **Cholinergic Dip Distortion at Strategy Boundaries:**
   * *Hypothesis:* In wild-type animals, discrete transitions between macro-strategies ($\mathcal{S}_A \to \mathcal{S}_B$) are accompanied by sharp transient dips in striatal acetylcholine ($\text{ACh3.0}$). In dopamine-depleted models, these cholinergic switching transients will be disorganized, leading to behavioral perseveration.
3. **Attractor Threshold Flattening in rSLDS:**
   * *Hypothesis:* The transition hyperplane slopes $\mathbf{R}_{j,k}$ in rSLDS will show shallow, noisy boundaries in PFF mice, explaining why Parkinsonian models exhibit erratic, fragmented strategic switching.
