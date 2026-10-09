# Multiview Top-Bottom Behavioral Experiments

**Location**: `studies/behavior/multiview_experiments`  
**Status**: `FROZEN`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Historical experiment drivers, report generators, and figure scripts for dual-camera (TopCam + BotCam) posture modeling.

## 2. Inputs
TopCam & BotCam h5 pose estimates and calibration matrices

## 3. Outputs
Multiview model rankings, EML evaluation tables, presentation figures

## 4. How to Run
```bash
python 03_evaluation_and_eml/evaluate_pilot_static_comparison.py
```
