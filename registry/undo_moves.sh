#!/usr/bin/env bash
# Auto-generated reversal script for Phase 3 scaffolding moves
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Reversing Phase 3 scaffolding moves..."

if [ -e "archive/logs/archive_checkpoints.log" ]; then
    mkdir -p "."
    mv "archive/logs/archive_checkpoints.log" "archive_checkpoints.log"
    echo "Restored: archive_checkpoints.log"
fi

if [ -e "archive/logs/model_fitting.log" ]; then
    mkdir -p "."
    mv "archive/logs/model_fitting.log" "model_fitting.log"
    echo "Restored: model_fitting.log"
fi

if [ -e "archive/logs/nohup.out" ]; then
    mkdir -p "."
    mv "archive/logs/nohup.out" "nohup.out"
    echo "Restored: nohup.out"
fi

if [ -e "archive/logs/run_dlc_analysis.log" ]; then
    mkdir -p "."
    mv "archive/logs/run_dlc_analysis.log" "run_dlc_analysis.log"
    echo "Restored: run_dlc_analysis.log"
fi

if [ -e "archive/logs/sweep_debug.log" ]; then
    mkdir -p "."
    mv "archive/logs/sweep_debug.log" "sweep_debug.log"
    echo "Restored: sweep_debug.log"
fi

if [ -e "archive/logs/sweep.log" ]; then
    mkdir -p "."
    mv "archive/logs/sweep.log" "sweep.log"
    echo "Restored: sweep.log"
fi

if [ -e "archive/text_dumps/tandem_multiview_kpms_expert_review_package_concat.txt" ]; then
    mkdir -p "."
    mv "archive/text_dumps/tandem_multiview_kpms_expert_review_package_concat.txt" "tandem_multiview_kpms_expert_review_package_concat.txt"
    echo "Restored: tandem_multiview_kpms_expert_review_package_concat.txt"
fi

if [ -e "archive/debug_images/debug_bot_raw_52299.png" ]; then
    mkdir -p "."
    mv "archive/debug_images/debug_bot_raw_52299.png" "debug_bot_raw_52299.png"
    echo "Restored: debug_bot_raw_52299.png"
fi

if [ -e "archive/debug_images/debug_syll30_r1_c2.png" ]; then
    mkdir -p "."
    mv "archive/debug_images/debug_syll30_r1_c2.png" "debug_syll30_r1_c2.png"
    echo "Restored: debug_syll30_r1_c2.png"
fi

if [ -e "outputs/test_runs" ]; then
    mkdir -p "."
    mv "outputs/test_runs" "tests/runs"
    echo "Restored: tests/runs"
fi

if [ -e "archive/packages/chain_grid_generation_code.zip" ]; then
    mkdir -p "."
    mv "archive/packages/chain_grid_generation_code.zip" "chain_grid_generation_code.zip"
    echo "Restored: chain_grid_generation_code.zip"
fi

if [ -e "archive/packages/chatgpt_ethological_seqnmf_package.zip" ]; then
    mkdir -p "."
    mv "archive/packages/chatgpt_ethological_seqnmf_package.zip" "chatgpt_ethological_seqnmf_package.zip"
    echo "Restored: chatgpt_ethological_seqnmf_package.zip"
fi

if [ -e "archive/packages/code_backup_2026-04-10.zip" ]; then
    mkdir -p "."
    mv "archive/packages/code_backup_2026-04-10.zip" "code_backup_2026-04-10.zip"
    echo "Restored: code_backup_2026-04-10.zip"
fi

if [ -e "archive/packages/density_sampling_v2_audit.zip" ]; then
    mkdir -p "."
    mv "archive/packages/density_sampling_v2_audit.zip" "density_sampling_v2_audit.zip"
    echo "Restored: density_sampling_v2_audit.zip"
fi

if [ -e "archive/packages/fiberphotometry_code_review.zip" ]; then
    mkdir -p "."
    mv "archive/packages/fiberphotometry_code_review.zip" "fiberphotometry_code_review.zip"
    echo "Restored: fiberphotometry_code_review.zip"
fi

if [ -e "archive/packages/multiview_kpms_expert_review_package.zip" ]; then
    mkdir -p "."
    mv "archive/packages/multiview_kpms_expert_review_package.zip" "multiview_kpms_expert_review_package.zip"
    echo "Restored: multiview_kpms_expert_review_package.zip"
fi

if [ -e "archive/packages/multiview_static_21kp_v2_engineering_dossier.zip" ]; then
    mkdir -p "."
    mv "archive/packages/multiview_static_21kp_v2_engineering_dossier.zip" "multiview_static_21kp_v2_engineering_dossier.zip"
    echo "Restored: multiview_static_21kp_v2_engineering_dossier.zip"
fi

if [ -e "archive/packages/multiview_tandem_expert_review_package.zip" ]; then
    mkdir -p "."
    mv "archive/packages/multiview_tandem_expert_review_package.zip" "multiview_tandem_expert_review_package.zip"
    echo "Restored: multiview_tandem_expert_review_package.zip"
fi

if [ -e "archive/packages/syllable_chains_research_package.zip" ]; then
    mkdir -p "."
    mv "archive/packages/syllable_chains_research_package.zip" "syllable_chains_research_package.zip"
    echo "Restored: syllable_chains_research_package.zip"
fi

if [ -e "archive/packages/syllable_generation_source_backup_20260216_161040.zip" ]; then
    mkdir -p "."
    mv "archive/packages/syllable_generation_source_backup_20260216_161040.zip" "syllable_generation_source_backup_20260216_161040.zip"
    echo "Restored: syllable_generation_source_backup_20260216_161040.zip"
fi

if [ -e "archive/packages/tandem_multiview_kpms_expert_review_package_7days.zip" ]; then
    mkdir -p "."
    mv "archive/packages/tandem_multiview_kpms_expert_review_package_7days.zip" "tandem_multiview_kpms_expert_review_package_7days.zip"
    echo "Restored: tandem_multiview_kpms_expert_review_package_7days.zip"
fi

if [ -e "archive/packages/v2_grid_movies_and_trajectories_engineering_rfc.zip" ]; then
    mkdir -p "."
    mv "archive/packages/v2_grid_movies_and_trajectories_engineering_rfc.zip" "v2_grid_movies_and_trajectories_engineering_rfc.zip"
    echo "Restored: v2_grid_movies_and_trajectories_engineering_rfc.zip"
fi

if [ -e "archive/packages/v2_visualization_fixes_and_gemini_plan (1).zip" ]; then
    mkdir -p "."
    mv "archive/packages/v2_visualization_fixes_and_gemini_plan (1).zip" "v2_visualization_fixes_and_gemini_plan (1).zip"
    echo "Restored: v2_visualization_fixes_and_gemini_plan (1).zip"
fi

if [ -e "archive/packages/v2_visualization_fixes_and_gemini_plan.zip" ]; then
    mkdir -p "."
    mv "archive/packages/v2_visualization_fixes_and_gemini_plan.zip" "v2_visualization_fixes_and_gemini_plan.zip"
    echo "Restored: v2_visualization_fixes_and_gemini_plan.zip"
fi

if [ -e "archive/packages/v4_kpms_integrity_and_labeling_deliverable.zip" ]; then
    mkdir -p "."
    mv "archive/packages/v4_kpms_integrity_and_labeling_deliverable.zip" "v4_kpms_integrity_and_labeling_deliverable.zip"
    echo "Restored: v4_kpms_integrity_and_labeling_deliverable.zip"
fi

if [ -e "archive/backups/syllable_generation_backup_20260204_133746" ]; then
    mkdir -p "."
    mv "archive/backups/syllable_generation_backup_20260204_133746" "syllable_generation_backup_20260204_133746"
    echo "Restored: syllable_generation_backup_20260204_133746"
fi

if [ -e "archive/backups/multiview_top_bot_experiments_code_backup_20260930" ]; then
    mkdir -p "."
    mv "archive/backups/multiview_top_bot_experiments_code_backup_20260930" "syllable_generation/multiview_top_bot_experiments_code_backup_20260930"
    echo "Restored: syllable_generation/multiview_top_bot_experiments_code_backup_20260930"
fi

if [ -e "archive/backups/multiview_top_bot_experiments_code_backup_20261005" ]; then
    mkdir -p "."
    mv "archive/backups/multiview_top_bot_experiments_code_backup_20261005" "syllable_generation/multiview_top_bot_experiments_code_backup_20261005"
    echo "Restored: syllable_generation/multiview_top_bot_experiments_code_backup_20261005"
fi

echo "All Phase 3 moves successfully reverted."

# Round 2 Moves
if [ -L "KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md" ]; then rm "KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md"; fi
if [ -e "docs/reports/KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md" ]; then mv "docs/reports/KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md" "KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md"; echo "Restored: KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md"; fi
if [ -L "KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md" ]; then rm "KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md"; fi
if [ -e "docs/reports/KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md" ]; then mv "docs/reports/KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md" "KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md"; echo "Restored: KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md"; fi
if [ -L "KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md" ]; then rm "KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md"; fi
if [ -e "docs/reports/KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md" ]; then mv "docs/reports/KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md" "KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md"; echo "Restored: KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md"; fi
if [ -L "KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md" ]; then rm "KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md"; fi
if [ -e "docs/reports/KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md" ]; then mv "docs/reports/KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md" "KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md"; echo "Restored: KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md"; fi
if [ -L "KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md" ]; then rm "KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md"; fi
if [ -e "docs/reports/KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md" ]; then mv "docs/reports/KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md" "KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md"; echo "Restored: KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md"; fi
if [ -L "KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md" ]; then rm "KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md"; fi
if [ -e "docs/reports/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md" ]; then mv "docs/reports/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md" "KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md"; echo "Restored: KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md"; fi
if [ -L "KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md" ]; then rm "KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md"; fi
if [ -e "docs/reports/KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md" ]; then mv "docs/reports/KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md" "KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md"; echo "Restored: KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md"; fi
if [ -L "KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md" ]; then rm "KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md"; fi
if [ -e "docs/reports/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md" ]; then mv "docs/reports/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md" "KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md"; echo "Restored: KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md"; fi
if [ -L "monday_14_7_2026_presentation.md" ]; then rm "monday_14_7_2026_presentation.md"; fi
if [ -e "docs/reports/monday_14_7_2026_presentation.md" ]; then mv "docs/reports/monday_14_7_2026_presentation.md" "monday_14_7_2026_presentation.md"; echo "Restored: monday_14_7_2026_presentation.md"; fi
if [ -L "TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md" ]; then rm "TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md"; fi
if [ -e "docs/reports/TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md" ]; then mv "docs/reports/TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md" "TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md"; echo "Restored: TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md"; fi
if [ -L "unique_information_postural_ceiling.md" ]; then rm "unique_information_postural_ceiling.md"; fi
if [ -e "docs/reports/unique_information_postural_ceiling.md" ]; then mv "docs/reports/unique_information_postural_ceiling.md" "unique_information_postural_ceiling.md"; echo "Restored: unique_information_postural_ceiling.md"; fi
if [ -L "SUMMARY_FOR_TOMORROW.md" ]; then rm "SUMMARY_FOR_TOMORROW.md"; fi
if [ -e "docs/reports/SUMMARY_FOR_TOMORROW.md" ]; then mv "docs/reports/SUMMARY_FOR_TOMORROW.md" "SUMMARY_FOR_TOMORROW.md"; echo "Restored: SUMMARY_FOR_TOMORROW.md"; fi
if [ -L "VALIDATION_REPORT.md" ]; then rm "VALIDATION_REPORT.md"; fi
if [ -e "docs/reports/VALIDATION_REPORT.md" ]; then mv "docs/reports/VALIDATION_REPORT.md" "VALIDATION_REPORT.md"; echo "Restored: VALIDATION_REPORT.md"; fi
if [ -L "BACKUP_REPORT.md" ]; then rm "BACKUP_REPORT.md"; fi
if [ -e "docs/reports/BACKUP_REPORT.md" ]; then mv "docs/reports/BACKUP_REPORT.md" "BACKUP_REPORT.md"; echo "Restored: BACKUP_REPORT.md"; fi
if [ -L "SCAFFOLDING_REPORT.md" ]; then rm "SCAFFOLDING_REPORT.md"; fi
if [ -e "docs/reports/SCAFFOLDING_REPORT.md" ]; then mv "docs/reports/SCAFFOLDING_REPORT.md" "SCAFFOLDING_REPORT.md"; echo "Restored: SCAFFOLDING_REPORT.md"; fi
if [ -L "REFACTORING_CHANGE_REPORT.md" ]; then rm "REFACTORING_CHANGE_REPORT.md"; fi
if [ -e "docs/reports/REFACTORING_CHANGE_REPORT.md" ]; then mv "docs/reports/REFACTORING_CHANGE_REPORT.md" "REFACTORING_CHANGE_REPORT.md"; echo "Restored: REFACTORING_CHANGE_REPORT.md"; fi
if [ -L "SWE_ORGANIZATION_BOT_PROMPT.md" ]; then rm "SWE_ORGANIZATION_BOT_PROMPT.md"; fi
if [ -e "docs/reports/SWE_ORGANIZATION_BOT_PROMPT.md" ]; then mv "docs/reports/SWE_ORGANIZATION_BOT_PROMPT.md" "SWE_ORGANIZATION_BOT_PROMPT.md"; echo "Restored: SWE_ORGANIZATION_BOT_PROMPT.md"; fi
if [ -L "registry.tsv" ]; then rm "registry.tsv"; fi
if [ -e "registry/registry.tsv" ]; then mv "registry/registry.tsv" "registry.tsv"; echo "Restored: registry.tsv"; fi
if [ -L "ARTIFACT_INVENTORY.tsv" ]; then rm "ARTIFACT_INVENTORY.tsv"; fi
if [ -e "registry/ARTIFACT_INVENTORY.tsv" ]; then mv "registry/ARTIFACT_INVENTORY.tsv" "ARTIFACT_INVENTORY.tsv"; echo "Restored: ARTIFACT_INVENTORY.tsv"; fi
if [ -L "DUPLICATES.tsv" ]; then rm "DUPLICATES.tsv"; fi
if [ -e "registry/DUPLICATES.tsv" ]; then mv "registry/DUPLICATES.tsv" "DUPLICATES.tsv"; echo "Restored: DUPLICATES.tsv"; fi
if [ -L "PERMISSIONS_MODIFIED_TIER_A.tsv" ]; then rm "PERMISSIONS_MODIFIED_TIER_A.tsv"; fi
if [ -e "registry/PERMISSIONS_MODIFIED_TIER_A.tsv" ]; then mv "registry/PERMISSIONS_MODIFIED_TIER_A.tsv" "PERMISSIONS_MODIFIED_TIER_A.tsv"; echo "Restored: PERMISSIONS_MODIFIED_TIER_A.tsv"; fi
if [ -L "restore_tier_a_permissions.sh" ]; then rm "restore_tier_a_permissions.sh"; fi
if [ -e "registry/restore_tier_a_permissions.sh" ]; then mv "registry/restore_tier_a_permissions.sh" "restore_tier_a_permissions.sh"; echo "Restored: restore_tier_a_permissions.sh"; fi
if [ -L "cohort_static_21kp_v2_expert_review_package" ]; then rm "cohort_static_21kp_v2_expert_review_package"; fi
if [ -e "archive/bundles/cohort_static_21kp_v2_expert_review_package" ]; then mv "archive/bundles/cohort_static_21kp_v2_expert_review_package" "cohort_static_21kp_v2_expert_review_package"; echo "Restored: cohort_static_21kp_v2_expert_review_package"; fi
if [ -L "karen_project_research_bundle" ]; then rm "karen_project_research_bundle"; fi
if [ -e "archive/bundles/karen_project_research_bundle" ]; then mv "archive/bundles/karen_project_research_bundle" "karen_project_research_bundle"; echo "Restored: karen_project_research_bundle"; fi
if [ -L "kpms_acceleration_research_bundle" ]; then rm "kpms_acceleration_research_bundle"; fi
if [ -e "archive/bundles/kpms_acceleration_research_bundle" ]; then mv "archive/bundles/kpms_acceleration_research_bundle" "kpms_acceleration_research_bundle"; echo "Restored: kpms_acceleration_research_bundle"; fi
if [ -L "kpms_engineering_handoff_v2" ]; then rm "kpms_engineering_handoff_v2"; fi
if [ -e "archive/bundles/kpms_engineering_handoff_v2" ]; then mv "archive/bundles/kpms_engineering_handoff_v2" "kpms_engineering_handoff_v2"; echo "Restored: kpms_engineering_handoff_v2"; fi
if [ -L "kpms_parity_additions" ]; then rm "kpms_parity_additions"; fi
if [ -e "archive/bundles/kpms_parity_additions" ]; then mv "archive/bundles/kpms_parity_additions" "kpms_parity_additions"; echo "Restored: kpms_parity_additions"; fi
if [ -L "kpms_scaling_bundle" ]; then rm "kpms_scaling_bundle"; fi
if [ -e "archive/bundles/kpms_scaling_bundle" ]; then mv "archive/bundles/kpms_scaling_bundle" "kpms_scaling_bundle"; echo "Restored: kpms_scaling_bundle"; fi
if [ -L "multiview_tandem_expert_review_package" ]; then rm "multiview_tandem_expert_review_package"; fi
if [ -e "archive/bundles/multiview_tandem_expert_review_package" ]; then mv "archive/bundles/multiview_tandem_expert_review_package" "multiview_tandem_expert_review_package"; echo "Restored: multiview_tandem_expert_review_package"; fi
if [ -L "syllable_chains_research_package" ]; then rm "syllable_chains_research_package"; fi
if [ -e "archive/bundles/syllable_chains_research_package" ]; then mv "archive/bundles/syllable_chains_research_package" "syllable_chains_research_package"; echo "Restored: syllable_chains_research_package"; fi
if [ -L "diagnostics" ]; then rm "diagnostics"; fi
if [ -e "archive/bundles/diagnostics" ]; then mv "archive/bundles/diagnostics" "diagnostics"; echo "Restored: diagnostics"; fi
if [ -L "QA" ]; then rm "QA"; fi
if [ -e "archive/bundles/QA" ]; then mv "archive/bundles/QA" "QA"; echo "Restored: QA"; fi
if [ -L "wandb" ]; then rm "wandb"; fi
if [ -e "archive/bundles/wandb" ]; then mv "archive/bundles/wandb" "wandb"; echo "Restored: wandb"; fi
if [ -L "karen_project" ]; then rm "karen_project"; fi
if [ -e "studies/karen_pff_vs_monomer" ]; then mv "studies/karen_pff_vs_monomer" "karen_project"; echo "Restored: karen_project"; fi
if [ -L "src_oft" ]; then rm "src_oft"; fi
if [ -e "studies/oft_baseline" ]; then mv "studies/oft_baseline" "src_oft"; echo "Restored: src_oft"; fi
if [ -L "src_mono_inter" ]; then rm "src_mono_inter"; fi
if [ -e "studies/mono_inter_juvenile_intruder" ]; then mv "studies/mono_inter_juvenile_intruder" "src_mono_inter"; echo "Restored: src_mono_inter"; fi
if [ -L "monoamine_interactions" ]; then rm "monoamine_interactions"; fi
if [ -e "studies/monoamine_interactions" ]; then mv "studies/monoamine_interactions" "monoamine_interactions"; echo "Restored: monoamine_interactions"; fi
if [ -L "scripts/backup_code.sh" ]; then rm "scripts/backup_code.sh"; fi
if [ -e "ops/backup/backup_code.sh" ]; then mv "ops/backup/backup_code.sh" "scripts/backup_code.sh"; echo "Restored: scripts/backup_code.sh"; fi
if [ -L "scripts/verify_tier_a_backup.py" ]; then rm "scripts/verify_tier_a_backup.py"; fi
if [ -e "ops/backup/verify_tier_a_backup.py" ]; then mv "ops/backup/verify_tier_a_backup.py" "scripts/verify_tier_a_backup.py"; echo "Restored: scripts/verify_tier_a_backup.py"; fi
if [ -L "scripts/batch_compact_all_checkpoints.py" ]; then rm "scripts/batch_compact_all_checkpoints.py"; fi
if [ -e "ops/compaction/batch_compact_all_checkpoints.py" ]; then mv "ops/compaction/batch_compact_all_checkpoints.py" "scripts/batch_compact_all_checkpoints.py"; echo "Restored: scripts/batch_compact_all_checkpoints.py"; fi
if [ -L "scripts/launch_batch_compaction.sh" ]; then rm "scripts/launch_batch_compaction.sh"; fi
if [ -e "ops/compaction/launch_batch_compaction.sh" ]; then mv "ops/compaction/launch_batch_compaction.sh" "scripts/launch_batch_compaction.sh"; echo "Restored: scripts/launch_batch_compaction.sh"; fi
if [ -L "scripts/ram_watchdog.py" ]; then rm "scripts/ram_watchdog.py"; fi
if [ -e "ops/monitoring/ram_watchdog.py" ]; then mv "ops/monitoring/ram_watchdog.py" "scripts/ram_watchdog.py"; echo "Restored: scripts/ram_watchdog.py"; fi
if [ -L "scripts/ntfy_monitor.py" ]; then rm "scripts/ntfy_monitor.py"; fi
if [ -e "ops/monitoring/ntfy_monitor.py" ]; then mv "ops/monitoring/ntfy_monitor.py" "scripts/ntfy_monitor.py"; echo "Restored: scripts/ntfy_monitor.py"; fi
if [ -L "scripts/notify_kpms_iterations.py" ]; then rm "scripts/notify_kpms_iterations.py"; fi
if [ -e "ops/monitoring/notify_kpms_iterations.py" ]; then mv "ops/monitoring/notify_kpms_iterations.py" "scripts/notify_kpms_iterations.py"; echo "Restored: scripts/notify_kpms_iterations.py"; fi
if [ -L "scripts/launch_extended_when_limited_done.py" ]; then rm "scripts/launch_extended_when_limited_done.py"; fi
if [ -e "ops/monitoring/launch_extended_when_limited_done.py" ]; then mv "ops/monitoring/launch_extended_when_limited_done.py" "scripts/launch_extended_when_limited_done.py"; echo "Restored: scripts/launch_extended_when_limited_done.py"; fi
if [ -L "run_dlc_analysis.py" ]; then rm "run_dlc_analysis.py"; fi
if [ -e "ops/dlc/run_dlc_analysis.py" ]; then mv "ops/dlc/run_dlc_analysis.py" "run_dlc_analysis.py"; echo "Restored: run_dlc_analysis.py"; fi
if [ -L "scripts/karen_rsync_all.sh" ]; then rm "scripts/karen_rsync_all.sh"; fi
if [ -e "ops/dlc/karen_rsync_all.sh" ]; then mv "ops/dlc/karen_rsync_all.sh" "scripts/karen_rsync_all.sh"; echo "Restored: scripts/karen_rsync_all.sh"; fi
if [ -L "scripts/karen_split_all.sh" ]; then rm "scripts/karen_split_all.sh"; fi
if [ -e "ops/dlc/karen_split_all.sh" ]; then mv "ops/dlc/karen_split_all.sh" "scripts/karen_split_all.sh"; echo "Restored: scripts/karen_split_all.sh"; fi
if [ -L "scripts/split_videos.sh" ]; then rm "scripts/split_videos.sh"; fi
if [ -e "ops/dlc/split_videos.sh" ]; then mv "ops/dlc/split_videos.sh" "scripts/split_videos.sh"; echo "Restored: scripts/split_videos.sh"; fi
if [ -L "scripts/link_karen_botcam.py" ]; then rm "scripts/link_karen_botcam.py"; fi
if [ -e "ops/dlc/link_karen_botcam.py" ]; then mv "ops/dlc/link_karen_botcam.py" "scripts/link_karen_botcam.py"; echo "Restored: scripts/link_karen_botcam.py"; fi
if [ -L "scripts/relink_split_videos.py" ]; then rm "scripts/relink_split_videos.py"; fi
if [ -e "ops/dlc/relink_split_videos.py" ]; then mv "ops/dlc/relink_split_videos.py" "scripts/relink_split_videos.py"; echo "Restored: scripts/relink_split_videos.py"; fi
if [ -L "scripts/rsync_karen_botcam.py" ]; then rm "scripts/rsync_karen_botcam.py"; fi
if [ -e "ops/dlc/rsync_karen_botcam.py" ]; then mv "ops/dlc/rsync_karen_botcam.py" "scripts/rsync_karen_botcam.py"; echo "Restored: scripts/rsync_karen_botcam.py"; fi
if [ -L "scripts/verify_botcam_orientation.py" ]; then rm "scripts/verify_botcam_orientation.py"; fi
if [ -e "ops/dlc/verify_botcam_orientation.py" ]; then mv "ops/dlc/verify_botcam_orientation.py" "scripts/verify_botcam_orientation.py"; echo "Restored: scripts/verify_botcam_orientation.py"; fi
if [ -L "scripts/build_swe_archive.py" ]; then rm "scripts/build_swe_archive.py"; fi
if [ -e "ops/repo/build_swe_archive.py" ]; then mv "ops/repo/build_swe_archive.py" "scripts/build_swe_archive.py"; echo "Restored: scripts/build_swe_archive.py"; fi
if [ -L "scripts/execute_phase3_moves.py" ]; then rm "scripts/execute_phase3_moves.py"; fi
if [ -e "ops/repo/execute_phase3_moves.py" ]; then mv "ops/repo/execute_phase3_moves.py" "scripts/execute_phase3_moves.py"; echo "Restored: scripts/execute_phase3_moves.py"; fi
