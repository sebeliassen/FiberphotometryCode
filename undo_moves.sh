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
