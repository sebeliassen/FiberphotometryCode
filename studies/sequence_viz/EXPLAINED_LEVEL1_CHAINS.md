# Discovered Level 1 Behavioral Chains: Ethological Catalog & Quantitative Profiles

This catalog documents the behavioral chains uncovered by convolutional non-negative matrix factorization (**seqNMF**, $K=10$, $L=3$ macro-bouts) trained across all 30 open-field sessions ($169,879$ bouts) across 15 mice (8 WT, 7 DAT-KI) under Saline baseline and Amphetamine challenge.

Chains are **sorted in descending order by average cohort usage rate** (events per minute averaged across WT Saline, WT Amphetamine, DAT-KI Saline, and DAT-KI Amphetamine). Low-occurrence / "ghost" chains ($<0.05$ events/min) are filtered from the primary catalog and documented in the appendix.

---

## Executive Master Summary Table

| Rank | Chain ID | Functional Theme | Avg Cohort Rate | Mean Session Occupancy | WT vs. DAT-KI Modulation | Primary Backbone ($W$ Peak) | Top Empirical Sequence ($N_{\text{obs}}$) | Univariate Contrast |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| **1** | **Chain 1** | **Stand-Down & Leftward Descent** | **1.513 /min** | $112.9\text{ s}$ ($3.3\%$) | **Elevated in DAT-KI** ($1.5\times$ Sal, $1.6\times$ Amph) | $S_{14} \to S_4 \to S_3$ | $S_{14} \to S_4 \to S_3$ ($n=278$) | Sal $p=0.189$, Amph $p=0.094$ |
| **2** | **Chain 8** | **Rightward Steering & Locomotor Pacing** | **1.432 /min** | $151.2\text{ s}$ ($4.3\%$) | **$2.1\times$ surge in DAT-KI Amph** | $S_{11} \to S_1 \to S_{30}$ | $S_{11} \to S_1 \to S_{30}$ ($n=426$) | Sal $p=0.152$, **Amph $p=0.0289^*$** |
| **3** | **Chain 5** | **Turn-Run-Counterturn Slalom** | **1.183 /min** | $116.9\text{ s}$ ($3.3\%$) | **Elevated in DAT-KI** ($1.7\times$ Sal, $1.9\times$ Amph) | $S_6 \to S_{17} \to S_{18}$ | $S_6 \to S_{17} \to S_2$ ($n=673$) | Sal $p=0.152$, Amph $p=0.152$ |
| **4** | **Chain 7** | **Wall-Lean, Sniff-Up & Rightward Descent** | **0.781 /min** | $61.6\text{ s}$ ($1.8\%$) | **Elevated in DAT-KI** ($1.9\times$ Sal, $2.1\times$ Amph) | $S_{16} \to S_{15} \to S_7$ | $S_{16} \to S_{14} \to S_7$ ($n=269$) | Sal $p=0.152$, Amph $p=0.152$ |
| **5** | **Chain 2** | **Deceleration, Look Right & Sniff** | **0.713 /min** | $92.6\text{ s}$ ($2.7\%$) | **Suppressed in DAT-KI** ($0.7\times$ Sal, $0.7\times$ Amph) | $S_{27} \to S_5 \to S_{10}$ | $S_3 \to S_5 \to S_{10}$ ($n=181$) | Sal $p=0.152$, Amph $p=0.397$ |
| **6** | **Chain 4** | **Sensory Alignment & Dash Right** | **0.606 /min** | $53.0\text{ s}$ ($1.5\%$) | **Elevated in DAT-KI Amph** ($1.6\times$) | $S_{40} \to S_9 \to S_0$ | $S_1 \to S_9 \to S_0$ ($n=124$) | Sal $p=0.613$, Amph $p=0.189$ |
| **7** | **Chain 10** | **Sensory Scanning & Rest / Pause** | **0.412 /min** | $52.9\text{ s}$ ($1.6\%$) | **Severe Collapse in DAT-KI** ($0.15\times$ Amph) | $S_{20} \to S_{34} \to S_{23}$ | $S_{20} \to S_{34} \to S_{49}$ ($n=215$) | **Sal $p=0.0289^*$, Amph $p=0.00062^{***}$ (FDR $p=0.0124^*$)** |
| **8** | **Chain 6** | **Dash Redirection & Re-Engagement** | **0.406 /min** | $35.3\text{ s}$ ($1.0\%$) | **Elevated in DAT-KI Saline** ($2.1\times$) | $S_{21} \to S_2 \to S_{24}$ | $S_{21} \to S_2 \to S_0$ ($n=229$) | Sal $p=0.397$, Amph $p=0.152$ |
| *Omitted* | *Chain 3* | *Hunched Rear & Sniff Pause* | *0.034 /min* | $5.6\text{ s}$ ($0.2\%$) | *Low-occurrence ghost component* | $S_{12} \to S_8 \to S_{13}$ | $S_{32} \to S_8 \to S_{36}$ ($n=30$) | *Rare ($N=58$ total episodes)* |
| *Omitted* | *Chain 9* | *Walk-to-Dash Transition* | *0.013 /min* | $1.7\text{ s}$ ($0.05\%$) | *Low-occurrence ghost component* | $S_{19} \to S_3 \to S_{26}$ | $S_{19} \to S_3 \to S_{18}$ ($n=4$) | *Near-zero ($N=22$ total episodes)* |

---

## Detailed Profiles of Primary Behavioral Chains

```
================================================================================
CHAIN 1: STAND-DOWN & LEFTWARD DESCENT LOCOMOTION
================================================================================
Average Cohort Rate: 1.513 events/min  |  Total Decoded Episodes: 2,580
Average Session Time: 112.9 seconds   |  Session Occupancy: 3.3%
```

### 1. Ethological Description & Function
Chain 1 captures the stereotyped motor program where an upright animal terminates vertical exploration along an arena wall, rotates its body downward and to the left, and transitions seamlessly into leftward forward walking. This is one of the most fundamental postural transition motifs in murine open-field behavior, bridging vertical wall rearing and horizontal exploratory locomotion.

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Initiation / Rearing Posture)**:
  - $S_{14}$ (**rear/lean**, weight $0.97$): Vertical rearing posture, forepaws elevated against the arena wall or leaning upward.
  - $S_{37}$ (**stand down**, weight $0.23$): Early initiation of forelimb lowering from elevated posture.
- **Lag 1 (Descent & Counter-Rotation)**:
  - $S_4$ (**back down + turn left**, weight $0.97$): Rapid forelimb placement back onto the floor accompanied by an angular torque rotating the torso to the left.
  - $S_{42}$ (**look right**, weight $0.22$): Transient visual orienting before full descent.
- **Lag 2 (Locomotor Launch)**:
  - $S_3$ (**walk left**, weight $1.00$): Sustained forward leftward quadrupedal walking along the perimeter wall.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=2,580$)
While the template peak is $S_{14} \to S_4 \to S_3$, individual mice exhibit clear functional branching at Lag 0 (initial elevation type) and Lag 2 (launch speed):
1. **$10.8\%$ ($n=278$)**: $S_{14}$ (rear/lean) $\to S_4$ (back down + turn left) $\to S_3$ (walk left) — *Canonical descent to walk*.
2. **$9.8\%$ ($n=252$)**: $S_{14}$ (rear/lean) $\to S_4$ (back down + turn left) $\to S_{17}$ (run left) — *High-velocity descent launching directly into sprint*.
3. **$8.1\%$ ($n=210$)**: $S_{16}$ (lean left) $\to S_4$ (back down + turn left) $\to S_3$ (walk left) — *Descent from shallow lateral wall-lean*.
4. **$7.2\%$ ($n=186$)**: $S_{12}$ (hunched rear) $\to S_4$ (back down + turn left) $\to S_3$ (walk left) — *Descent from low hunched posture*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $1.309 \pm 0.220$ events/min ($94.2\text{ s}$/session)
- **WT Amphetamine**: $1.049 \pm 0.200$ events/min ($72.9\text{ s}$/session)
- **DAT-KI Saline**: $2.002 \pm 0.362$ events/min ($160.2\text{ s}$/session)
- **DAT-KI Amphetamine**: $1.692 \pm 0.349$ events/min ($124.2\text{ s}$/session)
- **Phenotypic Summary**: DAT-KI animals show an elevated baseline rate of Chain 1 execution ($+53\%$ over WT in Saline, $+61\%$ in Amph). Under amphetamine, both genotypes moderately decrease rearing-descent cycles in favor of continuous pacing, but DAT-KI animals maintain elevated execution rates.
- **Stereotypy / Looping**: High self-transition probability ($P(1 \to 1) = 22.9\% \text{ in KI Saline}$, $21.9\% \text{ in KI Amph}$). Mice frequently perform a wall-rear, descend, take several steps, and immediately rear again.

---

```
================================================================================
CHAIN 8: RIGHTWARD STEERING & LOCOMOTOR PACING
================================================================================
Average Cohort Rate: 1.432 events/min  |  Total Decoded Episodes: 2,456
Average Session Time: 151.2 seconds   |  Session Occupancy: 4.3%
```

### 1. Ethological Description & Function
Chain 8 represents an elongated, high-momentum locomotor program characterized by sustained clockwise / rightward steering along the arena perimeter. Mice lock into a persistent rightward curve, chaining together multiple specialized right-stepping bouts.

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Pacing Initiation)**:
  - $S_{11}$ (**walk right**, weight $1.00$): Medium-velocity rightward walking bout initiating the curve.
- **Lag 1 (Curvilinear Pacing Continuation)**:
  - $S_1$ (**walk right in-place**, weight $1.00$): Tighter, rhythmic rightward locomotor steering maintaining angular momentum.
- **Lag 2 (Acceleration / Branching)**:
  - $S_{30}$ (**walk right**, weight $0.77$): Sharp forward-right locomotor step completing the cornering trajectory.
  - $S_0$ (**run right**, weight $0.57$): Acceleration variant where pacing transitions into an explosive sprint.
  - $S_{61}$ (**groom**, weight $0.27$): Low-weight deceleration termination.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=2,456$)
1. **$17.3\%$ ($n=426$)**: $S_{11}$ (walk right) $\to S_1$ (walk right) $\to S_{30}$ (walk right) — *Pure 3-step rightward perimeter pacing*.
2. **$12.9\%$ ($n=316$)**: $S_{11}$ (walk right) $\to S_1$ (walk right) $\to S_{11}$ (walk right) — *Iterative loop maintaining constant curvature radius*.
3. **$12.1\%$ ($n=296$)**: $S_{11}$ (walk right) $\to S_1$ (walk right) $\to S_0$ (run right) — *Cornering acceleration into full right sprint*.
4. **$8.0\%$ ($n=196$)**: $S_{11}$ (walk right) $\to S_1$ (walk right) $\to S_5$ (look right) — *Deceleration with lateral head orienting*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $1.277 \pm 0.584$ events/min ($140.1\text{ s}$/session)
- **WT Amphetamine**: $0.854 \pm 0.329$ events/min ($86.2\text{ s}$/session)
- **DAT-KI Saline**: $1.799 \pm 0.329$ events/min ($192.4\text{ s}$/session)
- **DAT-KI Amphetamine**: $1.800 \pm 0.369$ events/min ($185.9\text{ s}$/session)
- **Phenotypic Summary**: **Significant Genotype Contrast Under Amphetamine ($p = 0.0289^*$)**. While WT mice suppress this specific repetitive rightward chain under amphetamine (dropping from $1.28$ to $0.85$ /min), DAT-KI mice sustain high pacing rates ($1.80$ /min, more than **$2.1\times$ higher than WT**). In DAT-KI mutants, this chain forms the backbone of hyperdopaminergic perimeter-bound stereotypic pacing.

---

```
================================================================================
CHAIN 5: TURN-RUN-COUNTERTURN / SLALOM REDIRECTION
================================================================================
Average Cohort Rate: 1.183 events/min  |  Total Decoded Episodes: 2,022
Average Session Time: 116.9 seconds   |  Session Occupancy: 3.3%
```

### 1. Ethological Description & Function
Chain 5 captures high-speed slalom navigation and trajectory switching. A mouse executes a sharp leftward reorientation ($S_6$), rapidly accelerates forward along the new heading ($S_{17}$), and then counter-steers or checks its forward momentum. This motif represents agile obstacle avoidance and rapid redirection in open-field exploration.

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Sharp Turn Initiation)**:
  - $S_6$ (**turn left**, weight $1.00$): Angular yaw shift reorienting the snout and body axis to the left.
- **Lag 1 (High-Velocity Sprint Vector)**:
  - $S_{17}$ (**run left**, weight $0.82$): Explosive forward-left sprint accelerating down the cleared path.
  - $S_{28}$ (**look left**, weight $0.55$): Slower visual checking branch.
- **Lag 2 (Counter-Steer / Stabilization)**:
  - $S_{18}$ (**sniff / steer**, weight $0.93$): Ground inspection while decelerating.
  - $S_{44}$ (**groom / turn**, weight $0.35$): Abrupt cessation of forward drive.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=2,022$)
Empirical decoding reveals a striking structural feature: while $S_{18}$ is the template maximum in $W$, the single most common real-world continuation is a **sharp counter-turn to the right ($S_2$)**:
1. **$33.3\%$ ($n=673$)**: $S_6$ (turn left) $\to S_{17}$ (run left) $\to S_2$ (turn right) — *Classic Left-Sprint-Right S-curve slalom!*
2. **$17.0\%$ ($n=343$)**: $S_6$ (turn left) $\to S_{17}$ (run left) $\to S_3$ (walk left) — *Turn, sprint, and settle into leftward walk*.
3. **$10.6\%$ ($n=214$)**: $S_6$ (turn left) $\to S_{17}$ (run left) $\to S_{21}$ (dash) — *Extended double-sprint burst*.
4. **$8.9\%$ ($n=179$)**: $S_6$ (turn left) $\to S_{17}$ (run left) $\to S_6$ (turn left) — *Tight 180-degree loop*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $0.934 \pm 0.280$ events/min ($89.0\text{ s}$/session)
- **WT Amphetamine**: $0.759 \pm 0.221$ events/min ($74.7\text{ s}$/session)
- **DAT-KI Saline**: $1.631 \pm 0.323$ events/min ($165.1\text{ s}$/session)
- **DAT-KI Amphetamine**: $1.408 \pm 0.258$ events/min ($138.9\text{ s}$/session)
- **Phenotypic Summary**: DAT-KI animals execute this high-speed redirection motif at nearly **double the rate of WT mice** in both baseline ($1.63$ vs $0.93$ /min) and amphetamine ($1.41$ vs $0.76$ /min), reflecting hyper-kinetic steering and elevated rapid redirection.

---

```
================================================================================
CHAIN 7: WALL-LEAN, SNIFF-UP & RIGHTWARD DESCENT
================================================================================
Average Cohort Rate: 0.781 events/min  |  Total Decoded Episodes: 1,327
Average Session Time: 61.6 seconds    |  Session Occupancy: 1.8%
```

### 1. Ethological Description & Function
Chain 7 represents a vertical inspection motif with rightward descent. An animal engages with the vertical arena boundary via a wall lean or sniff upward, samples airborne olfactory cues, and terminates the inspection by dropping back to all four limbs with an angular rotation to the right ($S_7$).

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Wall Contact / Lean)**:
  - $S_{16}$ (**lean left**, weight $0.90$): Body pitched upward with lateral wall contact on the left.
  - $S_{31}$ (**rear/lean**, weight $0.43$): Elevated bilateral wall lean.
- **Lag 1 (Vertical Sniffing)**:
  - $S_{15}$ (**sniff upwards**, weight $0.88$): Extended upward snout extension sampling air cues above the arena rim.
  - $S_4$ (**back down + turn left**, weight $0.47$): Alternative leftward descent path.
- **Lag 2 (Rightward Descent)**:
  - $S_7$ (**back down + turn right**, weight $0.97$): Controlled drop of forelimbs with clockwise torso reorientation.
  - $S_{56}$ (**Syll_56**, weight $0.25$): Micro-stepping adjustment.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=1,327$)
1. **$20.3\%$ ($n=269$)**: $S_{16}$ (lean left) $\to S_{14}$ (rear/lean) $\to S_7$ (back down + turn right) — *Wall contact, elevation, and right descent*.
2. **$15.1\%$ ($n=201$)**: $S_5$ (look right) $\to S_{15}$ (sniff upwards) $\to S_7$ (back down + turn right) — *Glance, air sniff, and descent*.
3. **$13.2\%$ ($n=175$)**: $S_{21}$ (dash) $\to S_{15}$ (sniff upwards) $\to S_7$ (back down + turn right) — *High-speed wall approach, quick upward sniff, and descent*.
4. **$10.2\%$ ($n=136$)**: $S_{26}$ (dash left) $\to S_{15}$ (sniff upwards) $\to S_7$ (back down + turn right) — *Curvilinear dash checking vertical surface*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $0.494 \pm 0.095$ events/min ($38.0\text{ s}$/session)
- **WT Amphetamine**: $0.561 \pm 0.143$ events/min ($40.7\text{ s}$/session)
- **DAT-KI Saline**: $0.919 \pm 0.199$ events/min ($72.3\text{ s}$/session)
- **DAT-KI Amphetamine**: $1.152 \pm 0.318$ events/min ($95.3\text{ s}$/session)
- **Phenotypic Summary**: DAT-KI animals show an approximate **$2\times$ elevation in Chain 7** across both conditions ($0.92$ vs $0.49$ /min in Saline; $1.15$ vs $0.56$ /min in Amph), reflecting frequent vertical surface checking interspersed with active pacing.

---

```
================================================================================
CHAIN 2: DECELERATION, LOOK RIGHT & GROUND SNIFFING
================================================================================
Average Cohort Rate: 0.713 events/min  |  Total Decoded Episodes: 1,239
Average Session Time: 92.6 seconds    |  Session Occupancy: 2.7%
```

### 1. Ethological Description & Function
Chain 2 captures an investigatory deceleration sequence where active locomotion or low-level grooming pauses for an orienting head glance to the right ($S_5$), followed immediately by elongated, focused ground sniffing ($S_{10}$).

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Deceleration / Pause)**:
  - $S_{27}$ (**groom**, weight $0.95$): Low-mobility grooming or scratching bout.
  - $S_{54}$ (**Syll_54**, weight $0.31$): Low-mobility postural adjustment.
- **Lag 1 (Visual Orienting)**:
  - $S_5$ (**look right**, weight $0.98$): Clear head turn to the right without substantial locomotion.
  - $S_{52}$ (**groom**, weight $0.18$): Maintenance branch.
- **Lag 2 (Focused Ground Olfaction)**:
  - $S_{10}$ (**sniff**, weight $1.00$): Elongated middle ground sniff with snout low to the arena floor.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=1,239$)
In real behavior, Lag 0 is dominated by slow walking or ground sniffing settling into the look-right orienting response:
1. **$14.6\%$ ($n=181$)**: $S_3$ (walk left) $\to S_5$ (look right) $\to S_{10}$ (sniff) — *Locomotor deceleration into inspection*.
2. **$13.8\%$ ($n=171$)**: $S_9$ (sniff) $\to S_5$ (look right) $\to S_{10}$ (sniff) — *Olfactory scanning reorienting across substrate*.
3. **$11.6\%$ ($n=144$)**: $S_1$ (walk right) $\to S_5$ (look right) $\to S_{10}$ (sniff) — *Perimeter check during rightward walk*.
4. **$10.3\%$ ($n=127$)**: $S_{10}$ (sniff) $\to S_5$ (look right) $\to S_{10}$ (sniff) — *Iterative scanning loop*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $0.843 \pm 0.156$ events/min ($112.2\text{ s}$/session)
- **WT Amphetamine**: $0.850 \pm 0.186$ events/min ($110.7\text{ s}$/session)
- **DAT-KI Saline**: $0.556 \pm 0.142$ events/min ($68.5\text{ s}$/session)
- **DAT-KI Amphetamine**: $0.602 \pm 0.255$ events/min ($78.9\text{ s}$/session)
- **Phenotypic Summary**: Unlike locomotor chains, Chain 2 is **moderately suppressed in DAT-KI animals** ($\approx 30\%$ lower rate and occupancy), as hyperdopaminergic mutants spend less time in focused stationary sniffing and orienting pauses.

---

```
================================================================================
CHAIN 4: SENSORY ALIGNMENT & SPRINT INITIATION (DASH RIGHT)
================================================================================
Average Cohort Rate: 0.606 events/min  |  Total Decoded Episodes: 1,035
Average Session Time: 53.0 seconds    |  Session Occupancy: 1.5%
```

### 1. Ethological Description & Function
Chain 4 represents a transitional ignition sequence: the animal pauses or checks an olfactory patch ($S_9$), aligns its heading ($S_{40}$ / look right), and explosively accelerates forward into a rightward sprint ($S_0$).

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Postural Setup)**:
  - $S_{40}$ (**look right**, weight $0.65$): Head cast to the right.
  - $S_{50}$, $S_{59}$ (weights $\approx 0.47$): Stationary postural adjustments.
- **Lag 1 (Olfactory Verification)**:
  - $S_9$ (**sniff**, weight $0.96$): Rapid, focused substrate sniffing.
  - $S_{39}$ (**sniff**, weight $0.29$): Low-sniff variant.
- **Lag 2 (Explosive Run Launch)**:
  - $S_0$ (**run right**, weight $1.00$): Rapid forward-right gallop / sprint initiation.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=1,035$)
1. **$12.0\%$ ($n=124$)**: $S_1$ (walk right) $\to S_9$ (sniff) $\to S_0$ (run right) — *Walk, ground check, explosive acceleration*.
2. **$11.7\%$ ($n=121$)**: $S_{10}$ (sniff) $\to S_9$ (sniff) $\to S_0$ (run right) — *Continuous sniffing terminating in sprint*.
3. **$10.8\%$ ($n=112$)**: $S_{26}$ (dash left) $\to S_9$ (sniff) $\to S_0$ (run right) — *Rapid reversal: dash left, brake into sniff, dash right*.
4. **$8.3\%$ ($n=86$)**: $S_0$ (run right) $\to S_9$ (sniff) $\to S_0$ (run right) — *Sprint-brake-sprint burst*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $0.544 \pm 0.074$ events/min ($43.2\text{ s}$/session)
- **WT Amphetamine**: $0.477 \pm 0.089$ events/min ($41.8\text{ s}$/session)
- **DAT-KI Saline**: $0.657 \pm 0.110$ events/min ($58.8\text{ s}$/session)
- **DAT-KI Amphetamine**: $0.746 \pm 0.145$ events/min ($68.1\text{ s}$/session)
- **Phenotypic Summary**: DAT-KI animals under amphetamine show elevated initiation rates ($0.75$ vs $0.48$ /min in WT, $+56\%$), reflecting heightened locomotor excitability and rapid transitioning from sniffing into running.

---

```
================================================================================
CHAIN 10: ACTIVE SENSORY SCANNING & REST / PAUSE
================================================================================
Average Cohort Rate: 0.412 events/min  |  Total Decoded Episodes: 715
Average Session Time: 52.9 seconds    |  Session Occupancy: 1.6%
```

### 1. Ethological Description & Function
Chain 10 captures the classic **exploratory-scanning and rest routine**. An active sniffing animal ($S_{20}$) executes a wide lateral head sweep from right to left ($S_{34}$), surveying the local environment, and then settles into an extended stationary pause ($S_{49}$) or quiet sniffing.

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (Olfactory Ground Check)**:
  - $S_{20}$ (**sniff**, weight $0.99$): Active sniffing with slight right head bias.
- **Lag 1 (Wide Lateral Scanning Sweep)**:
  - $S_{34}$ (**look left from right**, weight $1.00$): Pronounced, wide head sweep scanning across the horizon.
- **Lag 2 (Stationary Settle / Rest)**:
  - $S_{23}$ (**walk left**, weight $0.60$): Slow creeping continuation.
  - $S_{35}$ (**hand in frame / edge pause**, weight $0.49$): Corner settle.
  - $S_{49}$ (**stationary pause / freeze**, weight $0.30$): Complete immobility / quiet wakefulness.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=715$)
1. **$30.1\%$ ($n=215$)**: $S_{20}$ (sniff) $\to S_{34}$ (look left from right) $\to S_{49}$ (stationary) — *Ground sniff, lateral scan, full rest pause*.
2. **$24.8\%$ ($n=177$)**: $S_{20}$ (sniff) $\to S_{34}$ (look left from right) $\to S_8$ (sniff) — *Sniff, lateral scan, resumption of quiet sniffing*.
3. **$7.4\%$ ($n=53$)**: $S_{20}$ (sniff) $\to S_{34}$ (look left from right) $\to S_{39}$ (sniff) — *Scan into scrunched sniffing*.
4. **$6.4\%$ ($n=46$)**: $S_{20}$ (sniff) $\to S_{34}$ (look left from right) $\to S_{22}$ (look right) — *Bilateral visual scanning check (left-then-right)*.

### 4. Genotype & Drug Modulation: THE STRONGEST GENOTYPE BIOMARKER
- **WT Saline**: $0.556 \pm 0.168$ events/min ($79.8\text{ s}$/session)
- **WT Amphetamine**: $0.816 \pm 0.286$ events/min ($103.1\text{ s}$/session)
- **DAT-KI Saline**: $0.153 \pm 0.053$ events/min ($16.2\text{ s}$/session)
- **DAT-KI Amphetamine**: $0.122 \pm 0.038$ events/min ($12.4\text{ s}$/session)
- **Statistical Significance**:
  - **Amphetamine Contrast**: Mann-Whitney $U = 55.0$, **raw $p = 0.00062$**, **FDR-corrected $p = 0.0124^*$**.
  - **Saline Contrast**: Mann-Whitney $U = 47.0$, **raw $p = 0.0289^*$**, FDR $p = 0.1156$.
- **Neuroethological Interpretation**: Chain 10 reveals a fundamental deficit in dopamine homeostasis. Normal wild-type mice, even when stimulated by amphetamine, periodically interrupt forward locomotion to survey their surroundings and pause ($0.82$ events/min, $>100$ seconds per session). In contrast, DAT-KI mutants suffer an almost **complete collapse of the sensory scanning and rest routine** ($0.12$ events/min, occupying only 12 seconds in a 45-minute session). DAT-KI mice are locked into perpetual locomotion and are unable to disengage into quiet scanning.

---

```
================================================================================
CHAIN 6: DASH REDIRECTION & RE-ENGAGEMENT
================================================================================
Average Cohort Rate: 0.406 events/min  |  Total Decoded Episodes: 693
Average Session Time: 35.3 seconds    |  Session Occupancy: 1.0%
```

### 1. Ethological Description & Function
Chain 6 captures a rapid deceleration and steering transition following a high-velocity dash. The animal dashes forward ($S_{21}$), executes a sharp rightward brake and turn ($S_2$), and transitions into either ground investigation ($S_{24}$) or a secondary sprint ($S_0$).

### 2. Learned seqNMF Syllable Template ($W$)
- **Lag 0 (High-Speed Approach)**:
  - $S_{21}$ (**dash**, weight $0.97$): Fast forward dash along open floor.
  - $S_{43}$ (**stationary**, weight $0.20$): Abrupt deceleration.
- **Lag 1 (Sharp Right Torque)**:
  - $S_2$ (**turn right**, weight $0.99$): Rapid lateral head and torso turn right.
- **Lag 2 (Investigative Deceleration or Re-Sprint)**:
  - $S_{24}$ (**scrunched sniffing**, weight $0.69$): Sudden deceleration into low scrunched posture.
  - $S_0$ (**run right**, weight $0.58$): Continuation into rightward sprint.
  - $S_{43}$ (**stationary pause**, weight $0.34$): Complete stop.

### 3. Empirical Branching Structure (Top Realized Sequences, $N=693$)
1. **$33.0\%$ ($n=229$)**: $S_{21}$ (dash) $\to S_2$ (turn right) $\to S_0$ (run right) — *Dash, bank right, sprint right*.
2. **$12.6\%$ ($n=87$)**: $S_{55}$ (walk left) $\to S_2$ (turn right) $\to S_0$ (run right) — *Directional reversal from left walk into right sprint*.
3. **$10.8\%$ ($n=75$)**: $S_{21}$ (dash) $\to S_2$ (turn right) $\to S_{14}$ (rear/lean) — *Dash, turn, and rear up against corner*.
4. **$8.9\%$ ($n=62$)**: $S_{21}$ (dash) $\to S_2$ (turn right) $\to S_{16}$ (lean left) — *Dash into wall lean*.

### 4. Genotype & Drug Modulation
- **WT Saline**: $0.301 \pm 0.077$ events/min ($26.0\text{ s}$/session)
- **WT Amphetamine**: $0.312 \pm 0.109$ events/min ($27.7\text{ s}$/session)
- **DAT-KI Saline**: $0.625 \pm 0.215$ events/min ($56.6\text{ s}$/session)
- **DAT-KI Amphetamine**: $0.385 \pm 0.049$ events/min ($30.8\text{ s}$/session)
- **Phenotypic Summary**: DAT-KI animals show elevated baseline execution ($+108\%$ over WT in Saline), matching their general tendency toward frequent ballistic bursts and wall re-orientations.

---

## Appendix: Omitted Low-Occurrence / "Ghost" Chains

The two factors below were discovered by the mathematical seqNMF optimization but exhibit near-zero empirical occurrences across the 30 cohorts ($<0.05$ events/min). They were filtered out to avoid over-interpreting low-support noise:

### Chain 3: Hunched Rear & Olfactory Pause ($N=58$ Total Episodes)
- **Average Cohort Rate**: $0.034$ events/min (only $5.6$ seconds per 45-min session; $0.2\%$ occupancy).
- **Template Peak**: $S_{12}$ (hunched rear, $0.89$) $\to S_8$ (sniff, $1.00$) $\to S_{13}$ (dash, $0.71$).
- **Top Empirical Sequence**: $S_{32}$ (stationary) $\to S_8$ (sniff) $\to S_{36}$ (sniff) ($n=30$, $51.7\%$).
- **Rationale for Omission**: With fewer than 2 episodes per session across all cohorts, Chain 3 lacks statistical power for genotype contrasts and represents an infrequent micro-variant of resting rather than an established behavioral theme.

### Chain 9: Walk-to-Dash Transition ($N=22$ Total Episodes)
- **Average Cohort Rate**: $0.013$ events/min (only $1.7$ seconds per 45-min session; $0.05\%$ occupancy).
- **Template Peak**: $S_{19}$ (walk, $0.86$) $\to S_3$ (walk left, $0.84$) $\to S_{26}$ (dash left, $0.80$).
- **Top Empirical Sequence**: $S_{19}$ (walk) $\to S_3$ (walk left) $\to S_{18}$ (sniff) ($n=4$, $18.2\%$).
- **Rationale for Omission**: Totaling only 22 instances across all 169,879 bouts (representing $\approx 0.01\%$ of all behavioral frames), Chain 9 is a mathematical residual factor with negligible ethological presence.

---

## Technical Summary of Decoding Parameters
- **seqNMF Training Hyperparameters**: $K=10$, $L=3$, $\lambda = 0.005$, $W_{\text{flat}}=0$, $W_{\text{up}}=0$, $\text{max\_iter}=400$.
- **Decoding Thresholds**: Session-specific adaptive peak threshold $\tau_k = P_{85}(H_k > 0.1)$, non-maximum suppression radius $W_{\text{NMS}} = 2$ bouts, minimum stage support $\ge 50\%$, maximum normalized reconstruction error $E_{kt} \le 0.68$.
- **Total Valid Dataset**: 30 sessions, 15 mice (8 WT, 7 DAT-KI), $169,879$ bouts, $12,147$ decoded chain episodes.
