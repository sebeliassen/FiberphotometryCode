# SKILL DOC: Temporal Dynamics Pipeline

This document explains the architecture and mathematics behind the Temporal Dynamics module (`lib/temporal.py`) in the `ulrik_kinematics` pipeline. It is designed to act as an AI reference guide when modifying or extending the time-series analysis.

## 1. Goal & Execution
The goal of this module is to move beyond static, 30-minute global session averages and instead map how behavioral entropy, velocity, and duration evolve over time, particularly under the influence of delayed-onset drugs (e.g., Amphetamine).

### Endpoints
The pipeline is invoked via the `run_analysis.py` orchestrator:
- `python run_analysis.py --verify-temporal` (Executes Identity Checks and Hardware Desync Perturbations).
- `python run_analysis.py --run-temporal-discovery` (Generates sliding temporal line-plots across groups).

## 2. Core Mining Logic (`get_rolling_metrics`)
The engine leverages the pre-verified physics logic in `metrics.py` but applies it via a sliding window vectorizer.
1. **Window Parameters**: Controlled by `config.TEMPORAL_WINDOW_SEC` (default: 120s) and `config.TEMPORAL_STEP_SEC` (default: 60s).
2. **Execution**: The dataframe is dynamically chunked by the `time` vector (derived implicitly from `frame_index` array natively exported by MoSeq at 30FPS). Each 2-minute chunk is temporarily packaged as a mock Session and passed into the global calculators.
3. **Entropy Fragmentation**: Because it calculates transition entropy locally per 2-minute window, it captures highly-rigid behavioral loops that normally get "smoothed out" if mathematically averaged across the entire 30-minute session.

## 3. Biological Discovery Phase
The pipeline groups all `get_rolling_metrics` outputs by Genotype and Drug. It plots the progression of Velocity, Entropy, and Sustained Duration using `seaborn`, automatically calculating and overlaying 95% Confidence Intervals. If divergence occurs (e.g., KI entropy plummets at 5 minutes while WT remains stable), the confidence intervals physically split on the chart, establishing visual mathematical significance of drug-onset dynamics across the cohort over time.

## 4. Verification Check Constraints (`verify_temporal_module`)

To guarantee mathematical integrity, the Temporal module runs three rigid invariants before accepting results:
1. **Velocity Identity (Should Pass)**: Proves that the average of the 2-minute rolling velocities equals the exact absolute 30-minute global average. This ensures the sliding window code is not mathematically warping the coordinate tracking data.
2. **Entropy Identity (Should Fail)**: Proves the *Entropy Paradox*. Rolling 2-minute entropy *must* be mathematically lower than 30-minute global entropy, establishing that behavioral complexity is locally contained in time.
3. **Hardware Desync Crash**: We purposefully mock a hardware camera failure by forcefully injecting `-1` frame gaps randomly into 15% of the contiguous tracking stream. We prove that the sliding window's `Mean_Duration` metric crashes by > 90%, thereby guaranteeing the pipeline retains extreme sensitivity to behavioral boundaries and tracking drop-outs.
