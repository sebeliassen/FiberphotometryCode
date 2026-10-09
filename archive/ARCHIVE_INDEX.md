# Archive Index

This document registers all historical snapshots, legacy packages, execution logs,
and temporary files proposed for relocation to the `archive/` hierarchy as part of the scaffolding refactoring.

**Reversibility Guarantee**: Every item moved has an entry in [`MOVES.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/MOVES.tsv) and can be reverted
instantly using [`undo_moves.sh`](file:///home/fsp585/Desktop/FiberphotometryCode/undo_moves.sh). Backward-compatible symlinks are maintained at all original paths.

## Cataloged Relocations

| Original Path | New Archive Path | Category | Description | Symlink Maintained |
|---|---|---|---|---|
| `archive_checkpoints.log` | `archive/logs/archive_checkpoints.log` | SAFE_CLEANLINESS | Execution log file from checkpoint maintenance | YES |
| `model_fitting.log` | `archive/logs/model_fitting.log` | SAFE_CLEANLINESS | Execution log file from model fitting | YES |
| `nohup.out` | `archive/logs/nohup.out` | SAFE_CLEANLINESS | Background terminal stdout/stderr dump | YES |
| `run_dlc_analysis.log` | `archive/logs/run_dlc_analysis.log` | SAFE_CLEANLINESS | DeepLabCut batch inference execution log | YES |
| `sweep_debug.log` | `archive/logs/sweep_debug.log` | SAFE_CLEANLINESS | Hyperparameter sweep debug execution log | YES |
| `sweep.log` | `archive/logs/sweep.log` | SAFE_CLEANLINESS | Hyperparameter sweep main execution log | YES |
| `tandem_multiview_kpms_expert_review_package_concat.txt` | `archive/text_dumps/tandem_multiview_kpms_expert_review_package_concat.txt` | SAFE_CLEANLINESS | Concatenated review bundle text dump | YES |
| `debug_bot_raw_52299.png` | `archive/debug_images/debug_bot_raw_52299.png` | SAFE_CLEANLINESS | Ad-hoc ventral camera frame render | YES |
| `debug_syll30_r1_c2.png` | `archive/debug_images/debug_syll30_r1_c2.png` | SAFE_CLEANLINESS | Ad-hoc syllable debug render | YES |
| `tests/runs` | `outputs/test_runs` | SAFE_CLEANLINESS | Heavy historical test execution artifacts directory | YES |
| `chain_grid_generation_code.zip` | `archive/packages/chain_grid_generation_code.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `chatgpt_ethological_seqnmf_package.zip` | `archive/packages/chatgpt_ethological_seqnmf_package.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `code_backup_2026-04-10.zip` | `archive/packages/code_backup_2026-04-10.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `density_sampling_v2_audit.zip` | `archive/packages/density_sampling_v2_audit.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `fiberphotometry_code_review.zip` | `archive/packages/fiberphotometry_code_review.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `multiview_kpms_expert_review_package.zip` | `archive/packages/multiview_kpms_expert_review_package.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `multiview_static_21kp_v2_engineering_dossier.zip` | `archive/packages/multiview_static_21kp_v2_engineering_dossier.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `multiview_tandem_expert_review_package.zip` | `archive/packages/multiview_tandem_expert_review_package.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `syllable_chains_research_package.zip` | `archive/packages/syllable_chains_research_package.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `syllable_generation_source_backup_20260216_161040.zip` | `archive/packages/syllable_generation_source_backup_20260216_161040.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `tandem_multiview_kpms_expert_review_package_7days.zip` | `archive/packages/tandem_multiview_kpms_expert_review_package_7days.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `v2_grid_movies_and_trajectories_engineering_rfc.zip` | `archive/packages/v2_grid_movies_and_trajectories_engineering_rfc.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `v2_visualization_fixes_and_gemini_plan (1).zip` | `archive/packages/v2_visualization_fixes_and_gemini_plan (1).zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `v2_visualization_fixes_and_gemini_plan.zip` | `archive/packages/v2_visualization_fixes_and_gemini_plan.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `v4_kpms_integrity_and_labeling_deliverable.zip` | `archive/packages/v4_kpms_integrity_and_labeling_deliverable.zip` | SAFE_CLEANLINESS | Root zip distribution / export package snapshot | YES |
| `syllable_generation_backup_20260204_133746` | `archive/backups/syllable_generation_backup_20260204_133746` | DATED_BACKUP | Dated code snapshot from Feb 2026 | YES |
| `syllable_generation/multiview_top_bot_experiments_code_backup_20260930` | `archive/backups/multiview_top_bot_experiments_code_backup_20260930` | DATED_BACKUP | Dated code snapshot from Sep 30 2026 | YES |
| `syllable_generation/multiview_top_bot_experiments_code_backup_20261005` | `archive/backups/multiview_top_bot_experiments_code_backup_20261005` | DATED_BACKUP | Dated code snapshot from Oct 05 2026 | YES |
