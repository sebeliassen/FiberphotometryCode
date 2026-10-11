# Thursday Meeting Notes — 04 Sep 2026
## Language Models: 4-Cohort Multimodal Timescale Analysis

---

## 🔥 This is Really Interesting

The multimodal memory capacity curves (Slide 4 feature set applied per cohort, post-injection only) reveal a **striking pattern of genotype-driven differences in the timescale at which past behavioral history becomes informative for future predictions**.

The key finding: **all four cohorts show a clear predictive peak followed by a monotonic taper**, but the *location* of that peak differs meaningfully across groups:

| Cohort | Optimal Past History (W*) | Peak Gain |
| :--- | :---: | :---: |
| **KI + Amphetamine** | **2.0s** | +25.27% |
| **KI + Saline** | **6.0s** | +24.50% |
| **WT + Amphetamine** | **10.0s** | +18.02% |
| **WT + Saline** | **20.0s** | +14.07% |

---

## 📌 To Explore: Peak Differences as a Signature of Behavioural Timescales

The shift in W* across groups is biologically interpretable as the **effective timescale of behavioural stationarity** — how long the animal's current behavioural state remains predictive before past history becomes noise:

- **KI mice peak at 2–6 seconds**, suggesting their behavioural states are extremely transient and fragmented. Knowing what they did 10 seconds ago already hurts prediction relative to knowing only the last 2 seconds — their repertoire is rapidly switching.
- **WT mice peak at 10–20 seconds**, indicating more sustained engagement in individual behaviours (exploration bouts, grooming sequences, pausing) before transitioning. Their past history remains informative over a longer window.
- **Amphetamine shortens W* within each genotype** (KI: 2s vs. 6s; WT: 10s vs. 20s), consistent with drug-induced hyperlocomotion and increased fragmentation of behavioural sequences.
- The **amplitude difference** (KI: +25% vs. WT: +14% to +18%) suggests KI mice have a more stereotyped and locally predictable short-timescale repertoire — possibly hyperrepetitive micro-movement sequences that a window of 2s captures perfectly.

**Follow-up question for Thursday:** Is the shorter W* in KI mice primarily driven by higher *syllable entropy* at longer timescales, or is it a genuine reflection of faster behavioural state transitions? The NMF trajectory data (Fig 1d) may already show this directly.

---

## ⚠️ Validation Outstanding: Methodological Consistency Check

Before drawing firm conclusions from the cohort comparison, we need to verify that the evaluation is using the same methodology as all prior "simple" experiments:

### 1. Cross-Validation Consistency
- **Cohort curves (new):** WT Saline evaluated via Leave-One-Session-Out (LOSO CV); transfer cohorts trained on full WT Saline and evaluated on held-out cohort populations.
- **Original Slide 4:** All 30 sessions pooled under GroupKFold(5), mixing cohorts together.
- **Action:** Confirm these are comparing apples to apples and that the transfer evaluation is a principled alternative — not a weaker one.

### 2. Feature Set Parity
- **Cohort curves (new):** Post-injection Time Clocks + Spatial Kinematics + 24D Bigram Grammar + BoW(W), confined to post-injection only.
- **Original Slide 4:** Same feature set but pre- *and* post-injection concat across the full 80-minute session.
- **Action:** Run the Slide 4 methodology (all 30 sessions pooled, full session time) with the 4 cohorts labelled separately and compare. This would give a direct baseline-comparable breakdown.

### 3. Injection Frame Consistency
- Double-check that `injection_frame` metadata is correctly populated for *all* 30 sessions (both WT and KI groups), and that the post-injection clock normalization is consistent across groups.
- If any sessions have `injection_frame = 0` (default fallback), those sessions may include pre-injection frames contaminating the post-injection analysis.

### 4. Bout Filtering Consistency
- Current analysis: filters bouts < 3 frames.
- Original Slide 4 analysis: no explicit bout frame filter.
- **Action:** Rerun with and without the 3-frame filter to confirm it doesn't selectively affect KI mice (who have more fragmented bouts by nature of the genotype).

---

## 📅 For Thursday

- [ ] Show the **peak comparison table** (4 cohorts x W* x Peak Gain) — this is the headline finding.
- [ ] Prepare a brief narrative: *"The timescale at which past history is maximally informative is a genotype signature"*.
- [ ] Clarify the validation checklist above and note which checks are still pending.
- [ ] Ask the professor whether the W* shift interpretation aligns with what's known about KI phenotype behavioural fragmentation from the literature.

---

*Generated: 2026-09-02 | Analysis: render_4_cohort_multimodal_figures.py | Feature Set: Multimodal (P_max Tier) | Data: Post-Injection Only, 30 Sessions*
