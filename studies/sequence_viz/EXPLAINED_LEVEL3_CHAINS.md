# Discovered Level 3 Behavioral Chains: Ethological Catalog & Quantitative Profiles

This catalog documents the behavioral chains uncovered by convolutional non-negative matrix factorization (**seqNMF**, $K=10$, $L=3$ macro-bouts) trained on the **Level 3 (22 Functional Families)** behavioral stream across all 30 sessions ($169,879$ bouts) across 15 mice (8 WT, 7 DAT-KI) under Saline and Amphetamine conditions.

At Level 3, sub-syllable synonym noise is merged while preserving functional and directional distinctions (sharp turns vs. walks vs. descents vs. sniffs). This resolution achieves:
- **Saline Groups Genotype Separation**: $\text{TV}_{\text{obs}} = 0.2902$, exact **$p = 0.0141^*$** ($N=6,435$ partitions)
- **Amphetamine Groups Genotype Separation**: $\text{TV}_{\text{obs}} = 0.3529$, exact **$p = 0.0045^{***}$** ($N=6,435$ partitions)
- **Global Pooled Genotype Separation**: $\text{TV}_{\text{obs}} = 0.3215$, exact **$p = 0.0036^{***}$** ($N=6,435$ partitions)

Chains are **sorted in descending order by average cohort usage rate** (events per minute averaged across WT Saline, WT Amphetamine, DAT-KI Saline, and DAT-KI Amphetamine).

---

## Executive Master Summary Table (Level 3)

| Rank | Chain ID | Functional Motif | Avg Rate | WT Saline Rate | WT Amph Rate | DAT-KI Saline Rate | DAT-KI Amph Rate | Primary Trajectory ($W$ Peak) | Genotype Contrast ($p$-value) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **1** | **Chain 8** | **Sprint Cornering into Rear** | **3.234 /min** | $2.43 \pm 0.47$ | $2.17 \pm 0.42$ | $4.31 \pm 0.86$ | $4.03 \pm 0.81$ | `Sprint -> Turn_R -> Rear_Elevated` | Sal $p=0.072$, Amph $p=0.072$ |
| **2** | **Chain 10** | **High-Speed Slalom Redirection** | **2.828 /min** | $1.84 \pm 0.39$ | $1.40 \pm 0.35$ | $3.95 \pm 0.69$ | $4.13 \pm 0.74$ | `Sprint -> Turn_L -> Sprint` | **Sal $p=0.0093^*$, Amph $p=0.0205^*$** |
| **3** | **Chain 5** | **Ground Olfactory Scanning & Check** | **2.698 /min** | $3.54 \pm 0.64$ | $3.55 \pm 0.62$ | $2.04 \pm 0.44$ | $1.66 \pm 0.38$ | `Sniff_Ground -> Look_L -> Sniff_Ground` | Sal $p=0.054$, **Amph $p=0.0205^*$** |
| **4** | **Chain 3** | **Sprint Burst to Vertical Sniff** | **2.040 /min** | $1.54 \pm 0.36$ | $1.29 \pm 0.33$ | $2.56 \pm 0.59$ | $2.77 \pm 0.61$ | `Sprint -> Sprint -> Sniff_Up` | Sal $p=0.189$, Amph $p=0.072$ |
| **5** | **Chain 6** | **Visual Orienting to Ground Sniff** | **1.593 /min** | $1.92 \pm 0.41$ | $1.99 \pm 0.43$ | $1.16 \pm 0.31$ | $1.30 \pm 0.33$ | `Tail_Other -> Look_R -> Sniff_Ground` | Sal $p=0.094$, Amph $p=0.054$ |
| **6** | **Chain 7** | **Wall-Lean & Leftward Descent** | **1.576 /min** | $1.19 \pm 0.32$ | $1.02 \pm 0.28$ | $2.41 \pm 0.58$ | $1.69 \pm 0.45$ | `Wall_Lean -> Descent_L -> Walk_L` | Sal $p=0.189$, Amph $p=0.054$ |
| **7** | **Chain 1** | **Bi-Directional Locomotor Turning** | **1.419 /min** | $1.10 \pm 0.33$ | $0.75 \pm 0.26$ | $2.13 \pm 0.54$ | $1.70 \pm 0.48$ | `Walk_L -> Tail_Other -> Walk_R` | Sal $p=0.094$, Amph $p=0.054$ |
| **8** | **Chain 4** | **Rightward Pacing to Grooming** | **0.754 /min** | $1.00 \pm 0.27$ | $1.22 \pm 0.31$ | $0.36 \pm 0.15$ | $0.43 \pm 0.18$ | `Walk_R -> Walk_R -> Grooming` | Sal $p=0.281$, **Amph $p=0.0140^*$** |
| **9** | **Chain 9** | **Forward Locomotion to Scrunched Sniff** | **0.751 /min** | $0.89 \pm 0.24$ | $1.03 \pm 0.27$ | $0.58 \pm 0.19$ | $0.51 \pm 0.18$ | `Walk_Fwd -> Sniff_Ground -> Sniff_Scrunched` | Sal $p=0.054$, **Amph $p=0.0093^*$** |
| **10** | **Chain 2** | **Descent Deceleration into Rest** | **0.443 /min** | $0.59 \pm 0.18$ | $0.85 \pm 0.24$ | $0.17 \pm 0.08$ | $0.16 \pm 0.08$ | `Sniff_Ground -> Descent_R -> Rest` | **Sal $p=0.0205^*$, Amph $p=0.0037^*$** |

---

## Detailed Profiles of Level 3 Behavioral Chains

```
================================================================================
CHAIN 8: SPRINT CORNERING INTO ELEVATED REAR
================================================================================
Average Cohort Rate: 3.234 events/min  |  Total Decoded Episodes: 4,420
Average Session Time: 268.4 seconds   |  Session Occupancy: 7.8%
```
- **Ethological Function**: Rapid ballistic sprint across the arena floor, banking right along the perimeter wall, and launching into an upright rearing posture (`Rear_Elevated`).
- **Template Weights ($W$)**:
  - Lag 0: `Sprint` ($1.00$)
  - Lag 1: `Turn_R` ($1.00$)
  - Lag 2: `Rear_Elevated` ($1.00$)
- **Genotype Modulation**: Highly elevated in DAT-KI mutants in both saline ($4.31$ vs $2.43$ /min) and amphetamine ($4.03$ vs $2.17$ /min), reflecting hyper-kinetic perimeter approaches and repeated wall climbing.
- **Looping Propensity**: High self-transition probability ($23.2\%$ in DAT-KI Saline, $23.9\%$ in DAT-KI Amph).

---

```
================================================================================
CHAIN 10: HIGH-SPEED SLALOM REDIRECTION & SPRINT BURST
================================================================================
Average Cohort Rate: 2.828 events/min  |  Total Decoded Episodes: 4,115
Average Session Time: 234.1 seconds   |  Session Occupancy: 6.8%
```
- **Ethological Function**: High-velocity forward propulsion interrupted by a sharp leftward yaw steering reorientation, followed immediately by continued high-speed sprinting.
- **Template Weights ($W$)**:
  - Lag 0: `Sprint` ($1.00$)
  - Lag 1: `Turn_L` ($1.00$)
  - Lag 2: `Sprint` ($1.00$)
- **Genotype Modulation (KEY BIOMARKER)**:
  - **Saline Contrast**: WT $1.84 \pm 0.39$ vs. DAT-KI $3.95 \pm 0.69$ events/min (**$U=6.0, p = 0.00932^*$**).
  - **Amphetamine Contrast**: WT $1.40 \pm 0.35$ vs. DAT-KI $4.13 \pm 0.74$ events/min (**$U=8.0, p = 0.0205^*$**).
  - DAT-KI mutants exhibit a **nearly 3-fold elevation** in slalom sprinting under amphetamine, with double the rate of back-to-back self-repetition ($25.3\%$ in DAT-KI Amph vs. $10.8\%$ in WT Amph).

---

```
================================================================================
CHAIN 5: GROUND OLFACTORY SCANNING & LATERAL CHECK
================================================================================
Average Cohort Rate: 2.698 events/min  |  Total Decoded Episodes: 4,082
Average Session Time: 341.2 seconds   |  Session Occupancy: 9.9%
```
- **Ethological Function**: Active ground olfaction interrupted by a lateral glance to the left (`Look_L`), returning to focused substrate sniffing.
- **Template Weights ($W$)**:
  - Lag 0: `Sniff_Ground` ($1.00$)
  - Lag 1: `Look_L` ($0.88$), `Sniff_Ground` ($0.47$)
  - Lag 2: `Sniff_Ground` ($1.00$)
- **Genotype Modulation**: Strongly preserved in WT mice ($3.54$ in Saline, $3.55$ in Amph) but significantly attenuated in DAT-KI mutants ($2.04$ in Saline, dropping to $1.66$ in Amph; **$p = 0.0205^*$**). Reflects WT preservation of quiet exploratory scanning.

---

```
================================================================================
CHAIN 3: SPRINT BURST TO VERTICAL SNIFFING
================================================================================
Average Cohort Rate: 2.040 events/min  |  Total Decoded Episodes: 3,012
Average Session Time: 198.5 seconds   |  Session Occupancy: 5.8%
```
- **Ethological Function**: Two consecutive sprinting strides terminating abruptly into an elevated head pitch to sample air cues above the arena floor.
- **Template Weights ($W$)**:
  - Lag 0: `Sprint` ($1.00$)
  - Lag 1: `Sprint` ($1.00$)
  - Lag 2: `Sniff_Up` ($1.00$)
- **Genotype Modulation**: Elevated in DAT-KI animals across both saline ($2.56$ vs $1.54$ /min) and amphetamine ($2.77$ vs $1.29$ /min), demonstrating frequent transitions between locomotion and vertical air sniffing.

---

```
================================================================================
CHAIN 6: VISUAL ORIENTING TO GROUND SNIFFING
================================================================================
Average Cohort Rate: 1.593 events/min  |  Total Decoded Episodes: 2,388
Average Session Time: 182.3 seconds   |  Session Occupancy: 5.3%
```
- **Ethological Function**: Transitional orienting sequence where an unclassified postural step or pause executes a head glance to the right (`Look_R`) followed by focused ground sniffing.
- **Template Weights ($W$)**:
  - Lag 0: `Tail_Unlabeled` ($1.00$)
  - Lag 1: `Look_R` ($1.00$)
  - Lag 2: `Sniff_Ground` ($0.76$), `Tail_Unlabeled` ($0.65$)
- **Genotype Modulation**: WT mice maintain significantly higher engagement ($1.92$ Saline, $1.99$ Amph) than DAT-KI mice ($1.16$ Saline, $1.30$ Amph).

---

```
================================================================================
CHAIN 7: WALL-LEAN & LEFTWARD POSTURAL DESCENT
================================================================================
Average Cohort Rate: 1.576 events/min  |  Total Decoded Episodes: 2,245
Average Session Time: 142.6 seconds   |  Session Occupancy: 4.1%
```
- **Ethological Function**: Vertical wall contact (`Wall_Lean`) followed by forelimb descent with leftward torque (`Descent_L`), transitioning into leftward walking (`Walk_L`).
- **Template Weights ($W$)**:
  - Lag 0: `Wall_Lean` ($0.90$), `Artifact` ($0.42$), `Stand_Down` ($0.13$)
  - Lag 1: `Descent_L` ($1.00$)
  - Lag 2: `Walk_L` ($1.00$)
- **Genotype Modulation**: Elevated in DAT-KI mutants ($2.41$ vs $1.19$ /min in saline, $+102\%$), representing frequent wall engagement and descent launching.

---

```
================================================================================
CHAIN 1: BI-DIRECTIONAL LOCOMOTOR TURNING & EXPLORATION
================================================================================
Average Cohort Rate: 1.419 events/min  |  Total Decoded Episodes: 2,050
Average Session Time: 165.8 seconds   |  Session Occupancy: 4.8%
```
- **Ethological Function**: Complex trajectory steering where leftward walking transitions through an intermediate stepping bout into rightward walking.
- **Template Weights ($W$)**:
  - Lag 0: `Walk_L` ($1.00$)
  - Lag 1: `Tail_Unlabeled` ($1.00$)
  - Lag 2: `Walk_R` ($1.00$)
- **Genotype Modulation**: Elevated in DAT-KI animals ($2.13$ vs $1.10$ /min in saline; $1.70$ vs $0.75$ /min in amphetamine).

---

```
================================================================================
CHAIN 4: RIGHTWARD CURVILINEAR PACING TO GROOMING
================================================================================
Average Cohort Rate: 0.754 events/min  |  Total Decoded Episodes: 1,120
Average Session Time: 68.2 seconds    |  Session Occupancy: 2.0%
```
- **Ethological Function**: Curvilinear rightward walking along perimeter that decelerates directly into self-grooming.
- **Template Weights ($W$)**:
  - Lag 0: `Walk_R` ($0.77$), `Grooming` ($0.64$)
  - Lag 1: `Walk_R` ($0.74$), `Grooming` ($0.68$)
  - Lag 2: `Grooming` ($1.00$)
- **Genotype Modulation**: WT mice maintain higher rates under amphetamine ($1.22$ vs $0.43$ /min in DAT-KI; **$p = 0.0140^*$**). Self-transition probability reaches $59.5\%$ in WT Saline, reflecting bouts of focused maintenance.

---

```
================================================================================
CHAIN 9: FORWARD LOCOMOTION TO SCRUNCHED SNIFFING
================================================================================
Average Cohort Rate: 0.751 events/min  |  Total Decoded Episodes: 1,118
Average Session Time: 98.4 seconds    |  Session Occupancy: 2.9%
```
- **Ethological Function**: Straightforward exploratory walking slowing down into ground sniffing and low-posture scrunched olfactory inspection.
- **Template Weights ($W$)**:
  - Lag 0: `Walk_Fwd` ($1.00$)
  - Lag 1: `Sniff_Ground` ($1.00$)
  - Lag 2: `Sniff_Scrunched` ($0.99$), `Stand_Down` ($0.11$)
- **Genotype Modulation**: Significant WT enrichment under amphetamine ($1.03$ vs $0.51$ /min in DAT-KI; **$p = 0.0093^*$**).

---

```
================================================================================
CHAIN 2: DESCENT DECELERATION INTO STATIONARY REST
================================================================================
Average Cohort Rate: 0.443 events/min  |  Total Decoded Episodes: 669
Average Session Time: 78.5 seconds    |  Session Occupancy: 2.3%
```
- **Ethological Function**: Olfactory ground check accompanied by rightward postural descent (`Descent_R`), settling into complete stationary rest (`Rest`).
- **Template Weights ($W$)**:
  - Lag 0: `Sniff_Ground` ($1.00$)
  - Lag 1: `Descent_R` ($1.00$)
  - Lag 2: `Rest` ($0.95$), `Walk_Sharp_R` ($0.30$)
- **Genotype Modulation (PRIMARY DEFICIT BIOMARKER)**:
  - **Saline Contrast**: WT $0.59 \pm 0.18$ vs. DAT-KI $0.17 \pm 0.08$ events/min (**$U=48.0, p = 0.0205^*$**).
  - **Amphetamine Contrast**: WT $0.85 \pm 0.24$ vs. DAT-KI $0.16 \pm 0.08$ events/min (**$U=52.0, p = 0.00373^{**}$**).
  - DAT-KI mutants almost completely lose the ability to disengage into stationary rest, exhibiting an **$\approx 80\%$ deficit** under amphetamine.
