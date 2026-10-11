# Multiview KPMS Independent Review & Verification Audit

## Core Findings:
1. Registration Discrepancy:
   - Priority 3 audit showed held-out errors of 11.76 / 9.85 / 10.01 px across Arenas 1-3.
   - Deep-dive across all 118 sessions revealed that while Dec 2025 - Mar 2026 had median residuals of 3.7 - 5.0 px, all April and May 2026 sessions systematically shifted by dy = +20.3 px and dx = +3.5 px across all arenas.
   - Re-fitting on April data reduced median residuals from 21.02 px to 3.51 px, confirming a physical/crop setup shift between March and April that corrupted the fused paw coordinates for late sessions.

2. Static Fusion Invariants:
   - Earlier Priority 2 audit never computed max_paw_diff and assigned bot_anchor_independent rather than measuring it.
   - Re-audit with real projection verification confirmed that current math produces 0.00e+00 px error relative to formula, and negative controls (paw swapping = 10-16 px error, 1-frame shift = 12-18 px error) are reliably caught.

3. Camera Clock Phase Vulnerability:
   - Synthetic reproduction proved that a 1-frame USB arrival latency in Cam_Bonsai_Timestamp causes the calibrator to latch a 1-frame offset (-1 frame error) while returning VALID_CRYSTAL with 0.0 ms residual.
   - Cross-camera timestamp comparison against FP3002 master clock revealed that 4.0% of frames have a 1-frame discrepancy.

4. Model Inputs and Cache Provenance:
   - Training runner reuses preprocessed_data.pkl, pca.p, and results.h5 by filename existence alone without hashing inputs or configs.
   - Timestamp CSVs are omitted from manifest input tracking.

5. Visualizations & Trajectories:
   - Grid movie density sampling was sorting duration differences abs(len - med_len) rather than trajectory density.
   - Trajectory plots display model-reconstructed coordinates rather than observed DLC coordinates.
   - Heading median filtering applied to scalar angles flips headings by 180 deg near +-pi wrap-around.
   - Grid movies and trajectories select completely different sets of bouts.

6. Training Runner Schedule Truncation:
   - The production runner hardcoded exactly 5 chunks of 50 iterations (0-250), ignoring --num-slds-iters when >200 SLDS iters are requested.
   - Telemetry calculations included 1.85 million padded dummy frames in ARI convergence and flip rate summaries.
