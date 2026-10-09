#!/usr/bin/env python3
import shutil
from pathlib import Path

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
STAGING = EXP / "scratch_expert_package/multiview_kpms_expert_review_package"

if STAGING.exists():
    shutil.rmtree(STAGING)
STAGING.mkdir(parents=True, exist_ok=True)

dirs = {
    "01_core_architecture": [
        EXP / "core/camera_correspondence.py",
        EXP / "core/ttl_sync.py",
        EXP / "core/static_fusion_engine.py",
        EXP / "core/fusion_engine.py",
        EXP / "core/pair_matching.py",
        EXP / "core/cache_manifest.py",
        EXP / "core/eval_artifacts.py",
        EXP / "core/evaluation_metrics.py",
        EXP / "core/io_utils.py",
        EXP / "core/logging_utils.py",
        EXP / "core/schemas.py",
        EXP / "core/static_arena_transforms.json",
        EXP / "core/ttl_cohort_offsets.json",
    ],
    "02_data_and_fusion_pipelines": [
        EXP / "01_data_and_fusion/01_match_pairs.py",
        EXP / "01_data_and_fusion/02_fuse_dorsal_ventral.py",
        EXP / "01_data_and_fusion/03_setup_multiview_kpms.py",
        EXP / "01_data_and_fusion/04_test_kpms_shapes_cpu.py",
        EXP / "01_data_and_fusion/fuse_cohort_static.py",
        EXP / "01_data_and_fusion/fuse_pilot_static.py",
        EXP / "01_data_and_fusion/fuse_all_continuous_calibrated_cohort.py",
        EXP / "01_data_and_fusion/fuse_all_tandem_cohort.py",
        EXP / "01_data_and_fusion/fuse_multiview_cohort.py",
        EXP / "01_data_and_fusion/fuse_reflection_corrected_sessions.py",
        EXP / "01_data_and_fusion/audit_cohort_continuous_sync.py",
        EXP / "01_data_and_fusion/audit_cohort_basler_calibrated.py",
        EXP / "01_data_and_fusion/audit_cohort_correlations.py",
        EXP / "01_data_and_fusion/audit_calibration_validity.py",
        EXP / "01_data_and_fusion/clock_sync.py",
        EXP / "01_data_and_fusion/enrich_sessions_with_botcam.py",
        EXP / "01_data_and_fusion/input_preflight_audit.py",
        EXP / "01_data_and_fusion/test_4_golden_rules_fusion.py",
        EXP / "01_data_and_fusion/test_ttl_aligned_fusion.py",
        EXP / "01_data_and_fusion/verify_fused_in_kpms_pipeline.py",
    ],
    "03_training_and_session_building": [
        EXP / "02_training_runners/run_cohort_static_21kp_production.py",
        EXP / "02_training_runners/launch_cohort_static_21kp_production.sh",
        EXP / "02_training_runners/run_cohort_calibrated_retrain_seed101.py",
        EXP / "02_training_runners/launch_cohort_calibrated_retrain_seed101.sh",
        EXP / "02_training_runners/run_pilot_static_comparison.py",
        EXP / "02_training_runners/launch_pilot_static_comparison.sh",
        EXP / "02_training_runners/run_pilot_17kp_vs_21kp_comparison.py",
        EXP / "02_training_runners/launch_pilot_17kp_vs_21kp_comparison.sh",
        EXP / "02_training_runners/run_10session_static_calibration.py",
        EXP / "02_training_runners/launch_10session_static_calibration.sh",
        EXP / "02_training_runners/run_kappa_scan_and_production_pipeline.py",
        EXP / "02_training_runners/launch_kappa_scan_and_production.sh",
        EXP / "02_training_runners/run_tandem_cohort_production_50ar_200slds.py",
        EXP / "02_training_runners/run_tandem_production_50ar_200slds.py",
        EXP / "02_training_runners/launch_tandem_cohort_production.sh",
        EXP / "02_training_runners/launch_tandem_cohort_seed_queue.sh",
        EXP / "cohort_static_21kp/config.yml",
        EXP / "kpms_tandem_cohort_continuous_calibrated/config.yml",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/find_kappa.py",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/kpms_pipeline.py",
    ],
    "04_visualization_and_grid_movies": [
        EXP / "04_reports_and_figures/render_all_cohort_static_21kp_seed101_grid_movies.py",
        EXP / "04_reports_and_figures/render_all_cohort_static_21kp_seed101_trajectories.py",
        EXP / "04_reports_and_figures/build_cohort_static_21kp_seed101_labeling_gallery.py",
        EXP / "04_reports_and_figures/render_all_seed404_grid_movies.py",
        EXP / "04_reports_and_figures/render_all_seed404_trajectories.py",
        EXP / "04_reports_and_figures/build_seed404_labeling_gallery.py",
        EXP / "04_reports_and_figures/render_all_seed101_grid_movies.py",
        EXP / "04_reports_and_figures/render_all_seed101_trajectories.py",
        EXP / "02_training_runners/generate_dorsal_grid_movies.py",
        EXP / "04_reports_and_figures/render_tandem_and_superimposed_trajectories.py",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/kpms_grid_movies_source.py",
        EXP / "04_reports_and_figures/run_all_cohort_static_21kp_seed101_visualizations.sh",
        EXP / "04_reports_and_figures/run_all_seed404_visualizations.sh",
    ],
    "05_audits_and_diagnostics": [
        EXP / "04_reports_and_figures/audits/diagnose_tandem_tile_heterogeneity.py",
        EXP / "04_reports_and_figures/audits/evaluate_pilot_static_comparison.py",
        EXP / "04_reports_and_figures/audits/audit_priority1_frame_label_correspondence.py",
        EXP / "04_reports_and_figures/audits/audit_priority2_static_fusion_invariants.py",
        EXP / "04_reports_and_figures/audits/audit_priority3_heldout_registration_accuracy.py",
        EXP / "04_reports_and_figures/audits/audit_priority4_metric_definitions.py",
        EXP / "04_reports_and_figures/audits/audit_current_seed404_gallery_tiles.py",
        EXP / "04_reports_and_figures/audits/compare_pilot_static_diagnostics.py",
        EXP / "04_reports_and_figures/audits/tile_heterogeneity_diagnosis.json",
        EXP / "04_reports_and_figures/audits/priority4_metric_definitions_audit.json",
        EXP / "04_reports_and_figures/audits/priority1_frame_label_correspondence_audit.json",
        EXP / "04_reports_and_figures/audits/priority2_static_fusion_invariants_audit.json",
        EXP / "04_reports_and_figures/audits/priority3_heldout_registration_accuracy_audit.json",
        EXP / "04_reports_and_figures/audits/pilot_static_comparison_results.json",
        EXP / "04_reports_and_figures/audits/pilot_static_vs_legacy_comparison.json",
        EXP / "04_reports_and_figures/audits/seed404_grid_movies_tile_by_tile_audit.md",
        REPO / "tests/test_camera_correspondence_synthetic.py",
        REPO / "tests/test_fusion_provenance_and_cache_synthetic.py",
        REPO / "tests/test_multiview_sync_and_renderer_regression.py",
        REPO / "tests/test_multiview_north_stars_4_1_to_4_3.py",
        REPO / "tests/test_cache_manifest_synthetic.py",
    ],
    "06_documentation_and_reports": [
        EXP / "docs/TANDEM_TILE_HETEROGENEITY_DIAGNOSIS_AND_STATIC_REGISTRATION_PROPOSAL.md",
        EXP / "04_reports_and_figures/PILOT_17KP_VS_21KP_FULL_COMPARISON_REPORT.md",
        EXP / "docs/CONFIDENCE_GATING_AND_CONTINUOUS_FUSION_ADDENDUM.md",
        EXP / "docs/DUAL_VIDEO_PIPELINE_ARCHITECTURE.md",
        EXP / "docs/EXPERIMENT_PLAN.md",
        EXP / "docs/FIBERPHOTOMETRY_REINTRODUCTION_GUIDE.md",
        EXP / "docs/MULTIVIEW_SYNCHRONIZATION_AND_VISUALIZATION_AUDIT_REPORT.md",
        EXP / "docs/MULTI_VIEW_SYNCHRONIZATION_AND_VFI_ARCHITECTURE.md",
        EXP / "docs/POSTMORTEM_MULTIVIEW_TIMING_AND_VISUALIZATION_DEBACLE.md",
        EXP / "docs/TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md",
        EXP / "docs/kpms_tandem_cohort_investigation.md",
        EXP / "docs/tandem_viz_guide.md",
        EXP / "04_reports_and_figures/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md",
        EXP / "04_reports_and_figures/cohort_multiview_synchronization_deep_dive.md",
        EXP / "04_reports_and_figures/visualizations_cohort_static_21kp_seed101/SEED101_STATIC21KP_SYLLABLE_CATALOG_FOR_LABELING.md",
        EXP / "REFACTORING_AND_SWE_ARCHITECTURE_PLAN.md",
        EXP / "SWE_HARDENING_STAGES_1_TO_3_REPORT.md",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/BUG_REPORT_AND_PARITY_RESOLUTION.md",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/OPTIMIZATION_EXPLANATION.md",
        REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/EML_MODEL_SELECTION_REPORT.md",
    ]
}

total_files = 0
for sub_dir, file_list in dirs.items():
    target_d = STAGING / sub_dir
    target_d.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        if src.exists():
            dest = target_d / src.name
            shutil.copy2(src, dest)
            total_files += 1
        else:
            print(f"WARNING: File not found: {src}")

print(f"Staged {total_files} files across {len(dirs)} categories.")
