#!/usr/bin/env python3
import os
import shutil
import zipfile
from pathlib import Path

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
STAGING = EXP / "scratch_expert_package/multiview_kpms_expert_review_package"

if STAGING.exists():
    shutil.rmtree(STAGING)
STAGING.mkdir(parents=True, exist_ok=True)

# Define directories and mapped destination filenames to avoid basename collisions
mappings = {
    "01_core_architecture": [
        (EXP / "core/camera_correspondence.py", "camera_correspondence.py"),
        (EXP / "core/ttl_sync.py", "ttl_sync.py"),
        (EXP / "core/static_fusion_engine.py", "static_fusion_engine.py"),
        (EXP / "core/fusion_engine.py", "fusion_engine.py"),
        (EXP / "core/pair_matching.py", "pair_matching.py"),
        (EXP / "core/cache_manifest.py", "cache_manifest.py"),
        (EXP / "core/eval_artifacts.py", "eval_artifacts.py"),
        (EXP / "core/evaluation_metrics.py", "evaluation_metrics.py"),
        (EXP / "core/io_utils.py", "io_utils.py"),
        (EXP / "core/logging_utils.py", "logging_utils.py"),
        (EXP / "core/schemas.py", "schemas.py"),
        (EXP / "core/static_arena_transforms.json", "static_arena_transforms.json"),
        (EXP / "core/ttl_cohort_offsets.json", "ttl_cohort_offsets.json"),
    ],
    "02_data_and_fusion_pipelines": [
        (EXP / "01_data_and_fusion/01_match_pairs.py", "01_match_pairs.py"),
        (EXP / "01_data_and_fusion/02_fuse_dorsal_ventral.py", "02_fuse_dorsal_ventral.py"),
        (EXP / "01_data_and_fusion/03_setup_multiview_kpms.py", "03_setup_multiview_kpms.py"),
        (EXP / "01_data_and_fusion/04_test_kpms_shapes_cpu.py", "04_test_kpms_shapes_cpu.py"),
        (EXP / "01_data_and_fusion/fuse_cohort_static.py", "fuse_cohort_static.py"),
        (EXP / "01_data_and_fusion/fuse_pilot_static.py", "fuse_pilot_static.py"),
        (EXP / "01_data_and_fusion/fuse_all_continuous_calibrated_cohort.py", "fuse_all_continuous_calibrated_cohort.py"),
        (EXP / "01_data_and_fusion/fuse_all_tandem_cohort.py", "fuse_all_tandem_cohort.py"),
        (EXP / "01_data_and_fusion/fuse_multiview_cohort.py", "fuse_multiview_cohort.py"),
        (EXP / "01_data_and_fusion/fuse_reflection_corrected_sessions.py", "fuse_reflection_corrected_sessions.py"),
        (EXP / "01_data_and_fusion/audit_cohort_continuous_sync.py", "audit_cohort_continuous_sync.py"),
        (EXP / "01_data_and_fusion/audit_cohort_basler_calibrated.py", "audit_cohort_basler_calibrated.py"),
        (EXP / "01_data_and_fusion/audit_cohort_correlations.py", "audit_cohort_correlations.py"),
        (EXP / "01_data_and_fusion/audit_calibration_validity.py", "audit_calibration_validity.py"),
        (EXP / "01_data_and_fusion/clock_sync.py", "clock_sync.py"),
        (EXP / "01_data_and_fusion/enrich_sessions_with_botcam.py", "enrich_sessions_with_botcam.py"),
        (EXP / "01_data_and_fusion/input_preflight_audit.py", "input_preflight_audit.py"),
        (EXP / "01_data_and_fusion/test_4_golden_rules_fusion.py", "test_4_golden_rules_fusion.py"),
        (EXP / "01_data_and_fusion/test_ttl_aligned_fusion.py", "test_ttl_aligned_fusion.py"),
        (EXP / "01_data_and_fusion/verify_fused_in_kpms_pipeline.py", "verify_fused_in_kpms_pipeline.py"),
    ],
    "03_training_optimization_and_configs": [
        (EXP / "02_training_runners/run_cohort_static_21kp_production.py", "run_cohort_static_21kp_production.py"),
        (EXP / "02_training_runners/launch_cohort_static_21kp_production.sh", "launch_cohort_static_21kp_production.sh"),
        (EXP / "02_training_runners/run_cohort_calibrated_retrain_seed101.py", "run_cohort_calibrated_retrain_seed101.py"),
        (EXP / "02_training_runners/launch_cohort_calibrated_retrain_seed101.sh", "launch_cohort_calibrated_retrain_seed101.sh"),
        (EXP / "02_training_runners/run_pilot_static_comparison.py", "run_pilot_static_comparison.py"),
        (EXP / "02_training_runners/launch_pilot_static_comparison.sh", "launch_pilot_static_comparison.sh"),
        (EXP / "02_training_runners/run_pilot_17kp_vs_21kp_comparison.py", "run_pilot_17kp_vs_21kp_comparison.py"),
        (EXP / "02_training_runners/launch_pilot_17kp_vs_21kp_comparison.sh", "launch_pilot_17kp_vs_21kp_comparison.sh"),
        (EXP / "02_training_runners/run_10session_static_calibration.py", "run_10session_static_calibration.py"),
        (EXP / "02_training_runners/launch_10session_static_calibration.sh", "launch_10session_static_calibration.sh"),
        (EXP / "02_training_runners/run_kappa_scan_and_production_pipeline.py", "run_kappa_scan_and_production_pipeline.py"),
        (EXP / "02_training_runners/launch_kappa_scan_and_production.sh", "launch_kappa_scan_and_production.sh"),
        (EXP / "02_training_runners/run_tandem_cohort_production_50ar_200slds.py", "run_tandem_cohort_production_50ar_200slds.py"),
        (EXP / "02_training_runners/run_tandem_production_50ar_200slds.py", "run_tandem_production_50ar_200slds.py"),
        # Fixed configs without collision:
        (EXP / "cohort_static_21kp/config.yml", "config_cohort_static_21kp.yml"),
        (EXP / "kpms_tandem_cohort_continuous_calibrated/config.yml", "config_legacy_tandem_23kp.yml"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/find_kappa.py", "find_kappa.py"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/kpms_pipeline.py", "kpms_pipeline.py"),
        # Custom training kernels & GPU batching:
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/utils/custom_kernels.py", "custom_kernels.py"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/utils/batch_init.py", "batch_init.py"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py", "resample_batching.py"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/utils/compact_checkpoint.py", "compact_checkpoint.py"),
    ],
    "04_session_and_photometry_weaving": [
        (EXP / "03_evaluation_and_eml/export_tandem_cohort_production_results.py", "export_tandem_cohort_production_results.py"),
        (REPO / "fiberphotometry/data/session_synchronization.py", "session_synchronization.py"),
        (REPO / "syllable_generation/analysis/replications/datta_utils.py", "datta_utils.py"),
        (EXP / "03_evaluation_and_eml/enrich_video_refs_and_analyze_sequences.py", "enrich_video_refs_and_analyze_sequences.py"),
    ],
    "05_visualization_and_grid_movies": [
        (EXP / "04_reports_and_figures/render_all_cohort_static_21kp_seed101_grid_movies.py", "render_all_cohort_static_21kp_seed101_grid_movies.py"),
        (EXP / "04_reports_and_figures/render_all_cohort_static_21kp_seed101_trajectories.py", "render_all_cohort_static_21kp_seed101_trajectories.py"),
        (EXP / "04_reports_and_figures/build_cohort_static_21kp_seed101_labeling_gallery.py", "build_cohort_static_21kp_seed101_labeling_gallery.py"),
        (EXP / "04_reports_and_figures/render_all_seed404_grid_movies.py", "render_all_seed404_grid_movies.py"),
        (EXP / "04_reports_and_figures/render_all_seed404_trajectories.py", "render_all_seed404_trajectories.py"),
        (EXP / "04_reports_and_figures/build_seed404_labeling_gallery.py", "build_seed404_labeling_gallery.py"),
        (EXP / "04_reports_and_figures/render_all_seed101_grid_movies.py", "render_all_seed101_grid_movies.py"),
        (EXP / "04_reports_and_figures/render_all_seed101_trajectories.py", "render_all_seed101_trajectories.py"),
        (EXP / "02_training_runners/generate_dorsal_grid_movies.py", "generate_dorsal_grid_movies.py"),
        (EXP / "04_reports_and_figures/render_tandem_and_superimposed_trajectories.py", "render_tandem_and_superimposed_trajectories.py"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/kpms_grid_movies_source.py", "kpms_grid_movies_source.py"),
        (EXP / "04_reports_and_figures/run_all_cohort_static_21kp_seed101_visualizations.sh", "run_all_cohort_static_21kp_seed101_visualizations.sh"),
        (EXP / "04_reports_and_figures/run_all_seed404_visualizations.sh", "run_all_seed404_visualizations.sh"),
    ],
    "06_independent_audits_and_diagnostics": [
        # New independent verification scripts & jsons:
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_1_heldout_registration_deepdive.py", "audit_1_heldout_registration_deepdive.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/all_sessions_registration_residuals.csv", "all_sessions_registration_residuals.csv"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/registration_by_arena_month_summary.json", "registration_by_arena_month_summary.json"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_2_real_static_fusion_invariants.py", "audit_2_real_static_fusion_invariants.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/real_static_fusion_invariants_and_negative_controls.json", "real_static_fusion_invariants_and_negative_controls.json"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_3_synthetic_clock_phase_and_real_lag.py", "audit_3_synthetic_clock_phase_and_real_lag.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/clock_phase_and_real_correspondence_audit.json", "clock_phase_and_real_correspondence_audit.json"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_4_training_inputs_and_cache_provenance.py", "audit_4_training_inputs_and_cache_provenance.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/training_inputs_and_cache_provenance.json", "training_inputs_and_cache_provenance.json"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_5_heading_and_bouts_discrepancy.py", "audit_5_heading_and_bouts_discrepancy.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/heading_and_bouts_audit.json", "heading_and_bouts_audit.json"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/audit_6_schedule_and_masking.py", "audit_6_schedule_and_masking.py"),
        (EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification/schedule_and_masking_audit.json", "schedule_and_masking_audit.json"),
        # Earlier audits:
        (EXP / "04_reports_and_figures/audits/diagnose_tandem_tile_heterogeneity.py", "diagnose_tandem_tile_heterogeneity.py"),
        (EXP / "04_reports_and_figures/audits/evaluate_pilot_static_comparison.py", "evaluate_pilot_static_comparison.py"),
        (EXP / "04_reports_and_figures/audits/audit_priority4_metric_definitions.py", "audit_priority4_metric_definitions.py"),
        (EXP / "04_reports_and_figures/audits/tile_heterogeneity_diagnosis.json", "tile_heterogeneity_diagnosis.json"),
        (EXP / "04_reports_and_figures/audits/priority4_metric_definitions_audit.json", "priority4_metric_definitions_audit.json"),
        (EXP / "04_reports_and_figures/audits/pilot_static_vs_legacy_comparison.json", "pilot_static_vs_legacy_comparison.json"),
        (REPO / "tests/test_camera_correspondence_synthetic.py", "test_camera_correspondence_synthetic.py"),
        (REPO / "tests/test_fusion_provenance_and_cache_synthetic.py", "test_fusion_provenance_and_cache_synthetic.py"),
        (REPO / "tests/test_multiview_sync_and_renderer_regression.py", "test_multiview_sync_and_renderer_regression.py"),
    ],
    "07_documentation_and_reports": [
        (EXP / "docs/TANDEM_TILE_HETEROGENEITY_DIAGNOSIS_AND_STATIC_REGISTRATION_PROPOSAL.md", "TANDEM_TILE_HETEROGENEITY_DIAGNOSIS_AND_STATIC_REGISTRATION_PROPOSAL.md"),
        (EXP / "04_reports_and_figures/PILOT_17KP_VS_21KP_FULL_COMPARISON_REPORT.md", "PILOT_17KP_VS_21KP_FULL_COMPARISON_REPORT.md"),
        (EXP / "docs/CONFIDENCE_GATING_AND_CONTINUOUS_FUSION_ADDENDUM.md", "CONFIDENCE_GATING_AND_CONTINUOUS_FUSION_ADDENDUM.md"),
        (EXP / "docs/DUAL_VIDEO_PIPELINE_ARCHITECTURE.md", "DUAL_VIDEO_PIPELINE_ARCHITECTURE.md"),
        (EXP / "docs/EXPERIMENT_PLAN.md", "EXPERIMENT_PLAN.md"),
        (EXP / "docs/FIBERPHOTOMETRY_REINTRODUCTION_GUIDE.md", "FIBERPHOTOMETRY_REINTRODUCTION_GUIDE.md"),
        (EXP / "docs/MULTIVIEW_SYNCHRONIZATION_AND_VISUALIZATION_AUDIT_REPORT.md", "MULTIVIEW_SYNCHRONIZATION_AND_VISUALIZATION_AUDIT_REPORT.md"),
        (EXP / "docs/MULTI_VIEW_SYNCHRONIZATION_AND_VFI_ARCHITECTURE.md", "MULTI_VIEW_SYNCHRONIZATION_AND_VFI_ARCHITECTURE.md"),
        (EXP / "docs/POSTMORTEM_MULTIVIEW_TIMING_AND_VISUALIZATION_DEBACLE.md", "POSTMORTEM_MULTIVIEW_TIMING_AND_VISUALIZATION_DEBACLE.md"),
        (EXP / "docs/TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md", "TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md"),
        (EXP / "docs/kpms_tandem_cohort_investigation.md", "kpms_tandem_cohort_investigation.md"),
        (EXP / "docs/tandem_viz_guide.md", "tandem_viz_guide.md"),
        (EXP / "04_reports_and_figures/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md", "KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md"),
        (EXP / "04_reports_and_figures/cohort_multiview_synchronization_deep_dive.md", "cohort_multiview_synchronization_deep_dive.md"),
        (EXP / "04_reports_and_figures/visualizations_cohort_static_21kp_seed101/SEED101_STATIC21KP_SYLLABLE_CATALOG_FOR_LABELING.md", "SEED101_STATIC21KP_SYLLABLE_CATALOG_FOR_LABELING.md"),
        (EXP / "REFACTORING_AND_SWE_ARCHITECTURE_PLAN.md", "REFACTORING_AND_SWE_ARCHITECTURE_PLAN.md"),
        (EXP / "SWE_HARDENING_STAGES_1_TO_3_REPORT.md", "SWE_HARDENING_STAGES_1_TO_3_REPORT.md"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/BUG_REPORT_AND_PARITY_RESOLUTION.md", "BUG_REPORT_AND_PARITY_RESOLUTION.md"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/OPTIMIZATION_EXPLANATION.md", "OPTIMIZATION_EXPLANATION.md"),
        (REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/EML_MODEL_SELECTION_REPORT.md", "EML_MODEL_SELECTION_REPORT.md"),
    ]
}

total_files = 0
for sub_dir, file_list in mappings.items():
    target_d = STAGING / sub_dir
    target_d.mkdir(parents=True, exist_ok=True)
    for src, dst_name in file_list:
        if src.exists():
            dest = target_d / dst_name
            shutil.copy2(src, dest)
            total_files += 1
        else:
            print(f"WARNING: File not found: {src}")

# Copy README_FOR_EXTERNAL_EXPERTS.md
shutil.copy2(EXP / "scratch_expert_package/README_FOR_EXTERNAL_EXPERTS.md", STAGING / "README_FOR_EXTERNAL_EXPERTS.md") if (EXP / "scratch_expert_package/README_FOR_EXTERNAL_EXPERTS.md").exists() else None

# Save multiview_kpms_independent_review.md into STAGING root
review_md = STAGING / "multiview_kpms_independent_review.md"
review_md.write_text("""# Multiview KPMS Independent Review & Verification Audit

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
""")

print(f"Staged {total_files} files into clean expert package.")

# Zip targets
zip_target1 = EXP / "multiview_kpms_expert_review_package.zip"
zip_target2 = REPO / "multiview_kpms_expert_review_package.zip"

for zip_target in [zip_target1, zip_target2]:
    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(STAGING):
            for file in files:
                p = Path(root) / file
                arcname = p.relative_to(STAGING.parent)
                zf.write(p, arcname)
    size_mb = zip_target.stat().st_size / (1024 * 1024)
    print(f"Created {zip_target}: {size_mb:.2f} MB")
