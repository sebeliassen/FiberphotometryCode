# Latest supplied Python source index — auxiliary revision

All 84 files below are unchanged copies of the auxiliary uploaded archive. Four archive paths changed relative to the previous snapshot; see `AUXILIARY_DIFF.patch`. Descriptions are original module-docstring excerpts, not independently validated claims. Definitions list top-level functions/classes; details and nested definitions remain in each file. No file was imported or executed for this index.

Use the repository-path mapping in `KPMS_ENGINEERING_DOSSIER.md` or the per-file `intended_repository_path` in `SOURCE_MANIFEST.json`. This review layout is not a runnable replacement for the original repository.

## core_patches (7 Python files)

| File | Supplied description | Top-level functions/classes |
|---|---|---|
| [batch_init.py](auxiliary_bundle/core_patches/batch_init.py) | No module docstring supplied. | batched_init_model_by_video |
| [custom_kernels.py](auxiliary_bundle/core_patches/custom_kernels.py) | No module docstring supplied. | _resample_scales_kernel, batched_resample_scales, apply_optimizations |
| [flipping.py](auxiliary_bundle/core_patches/flipping.py) | No module docstring supplied. | get_symmetry_indices, flip_data |
| [kpms_pipeline.py](auxiliary_bundle/core_patches/kpms_pipeline.py) | No module docstring supplied. | run_pipeline |
| [paths.py](auxiliary_bundle/core_patches/paths.py) | No module docstring supplied. | _collect_cam_paths_from_dfs, _subroutine_jonathan, _subroutine_soeren, ref_pairs_from_sessions, get_key_to_aid_mapping |
| [resample_batching.py](auxiliary_bundle/core_patches/resample_batching.py) | No module docstring supplied. | BatchConfig, precompute_optimal_batch_config, set_batch_sizes, KPMSStateCache, _jit_compute_discrete_emissions, _numba_sample_categorical, _numba_ffbs_single_sequence, _numba_ffbs_all_parallel, _jit_resample_discrete_batch, fast_resample_discrete_stateseqs; plus 22 more |
| [trimming.py](auxiliary_bundle/core_patches/trimming.py) | No module docstring supplied. | _clean_short_runs, presence_mask_from_conf, _init_worker, _process_outliers_single, filter_with_masks, clamp_conf_nonneg, subset_spread_blocks, subset_after_filter, subset_by_activity |

## production_and_test_runners (8 Python files)

| File | Supplied description | Top-level functions/classes |
|---|---|---|
| [find_kappa.py](auxiliary_bundle/production_and_test_runners/find_kappa.py) | No module docstring supplied. | get_sessions, _next_kappa, _write_json, main |
| [run_amph_benchmark.py](auxiliary_bundle/production_and_test_runners/run_amph_benchmark.py) | run_amph_benchmark.py | get_sessions, NpEncoder, _write_json |
| [run_amph_production.py](auxiliary_bundle/production_and_test_runners/run_amph_production.py) | run_amph_production.py | get_sessions, NpEncoder, _write_json |
| [run_cohort_2ar_2slds_test.py](auxiliary_bundle/production_and_test_runners/run_cohort_2ar_2slds_test.py) | run_cohort_2ar_2slds_test.py | main |
| [run_multiview_kappa_search.py](auxiliary_bundle/production_and_test_runners/run_multiview_kappa_search.py) | run_multiview_kappa_search.py | send_ntfy, create_snippet_subset, _next_kappa, main |
| [run_tandem_cohort_production_50ar_200slds.py](auxiliary_bundle/production_and_test_runners/run_tandem_cohort_production_50ar_200slds.py) | run_tandem_cohort_production_50ar_200slds.py | send_ntfy, handle_shutdown_signal, main |
| [run_tandem_production_50ar_200slds.py](auxiliary_bundle/production_and_test_runners/run_tandem_production_50ar_200slds.py) | run_tandem_production_50ar_200slds.py | send_ntfy, handle_shutdown_signal, fuse_full_session, main |
| [test_tandem_pipeline_2sessions.py](auxiliary_bundle/production_and_test_runners/test_tandem_pipeline_2sessions.py) | test_tandem_pipeline_2sessions.py | rot2d, smooth_heading_circular, smooth_centroid_median, fuse_full_session, main |

## multiview_and_fusion (8 Python files)

| File | Supplied description | Top-level functions/classes |
|---|---|---|
| [01_match_pairs.py](auxiliary_bundle/multiview_and_fusion/01_match_pairs.py) | 01_match_pairs.py | extract_botcam_tokens, main |
| [02_fuse_dorsal_ventral.py](auxiliary_bundle/multiview_and_fusion/02_fuse_dorsal_ventral.py) | 02_fuse_dorsal_ventral.py | compute_heading_and_centroid, fuse_session_pair |
| [03_setup_multiview_kpms.py](auxiliary_bundle/multiview_and_fusion/03_setup_multiview_kpms.py) | 03_setup_multiview_kpms.py | create_multiview_config, main |
| [04_test_kpms_shapes_cpu.py](auxiliary_bundle/multiview_and_fusion/04_test_kpms_shapes_cpu.py) | 04_test_kpms_shapes_cpu.py | TeeLogger, create_test_config, generate_fused_sliver, run_cpu_shape_test |
| [audit_cohort_correlations.py](auxiliary_bundle/multiview_and_fusion/audit_cohort_correlations.py) | audit_cohort_correlations.py | circular_correlation, audit_single_session, find_matched_pairs, main |
| [clock_sync.py](auxiliary_bundle/multiview_and_fusion/clock_sync.py) | clock_sync.py | load_timestamps_from_csv, align_camera_timestamps, align_tracking_dataframes |
| [fuse_all_tandem_cohort.py](auxiliary_bundle/multiview_and_fusion/fuse_all_tandem_cohort.py) | fuse_all_tandem_cohort.py | rot2d, smooth_heading_circular, smooth_centroid_median, fuse_single_pair, find_matched_pairs, main |
| [render_tandem_and_superimposed_trajectories.py](auxiliary_bundle/multiview_and_fusion/render_tandem_and_superimposed_trajectories.py) | render_tandem_and_superimposed_trajectories.py | modern_rasterize_figure, main |

## diagnostics_and_benchmarks (61 Python files)

| File | Supplied description | Top-level functions/classes |
|---|---|---|
| [audit_training_mask_distribution.py](auxiliary_bundle/diagnostics_and_benchmarks/audit_training_mask_distribution.py) | No module docstring supplied. | audit_dataset_mask |
| [benchmark_fused_transform.py](auxiliary_bundle/diagnostics_and_benchmarks/benchmark_fused_transform.py) | diagnostics/benchmark_fused_transform.py | current_numpy_transform, fused_numba_transform, main |
| [benchmark_lapack_controlled.py](auxiliary_bundle/diagnostics_and_benchmarks/benchmark_lapack_controlled.py) | diagnostics/benchmark_lapack_controlled.py | run_numba_parallel, run_benchmark |
| [benchmark_numba_hmm_sampler.py](auxiliary_bundle/diagnostics_and_benchmarks/benchmark_numba_hmm_sampler.py) | No module docstring supplied. | log, _numba_sample_categorical, _numba_ffbs_single_sequence, _numba_ffbs_all_parallel, main |
| [benchmark_steady_state_speed.py](auxiliary_bundle/diagnostics_and_benchmarks/benchmark_steady_state_speed.py) | No module docstring supplied. | benchmark_iterations |
| [cache_karen_init.py](auxiliary_bundle/diagnostics_and_benchmarks/cache_karen_init.py) | No module docstring supplied. | cache_init |
| [compare_ar_checkpoint_slds_iters.py](auxiliary_bundle/diagnostics_and_benchmarks/compare_ar_checkpoint_slds_iters.py) | No module docstring supplied. | calculate_duration_stats, run_comparative_experiment |
| [cpu_ram_watchdog.py](auxiliary_bundle/diagnostics_and_benchmarks/cpu_ram_watchdog.py) | No module docstring supplied. | send_ntfy, main |
| [debug_ar_iteration_speed.py](auxiliary_bundle/diagnostics_and_benchmarks/debug_ar_iteration_speed.py) | diagnostics/debug_ar_iteration_speed.py | sync, main |
| [diagnostic_jitter_investigation.py](auxiliary_bundle/diagnostics_and_benchmarks/diagnostic_jitter_investigation.py) | No module docstring supplied. | run_jitter_diagnostic |
| [diagnostic_lag_noise_sweep.py](auxiliary_bundle/diagnostics_and_benchmarks/diagnostic_lag_noise_sweep.py) | No module docstring supplied. | build_diagnostic_kalman_sampler, run_diagnostic |
| [diagnostic_prior_conditioning_and_lag_sweep.py](auxiliary_bundle/diagnostics_and_benchmarks/diagnostic_prior_conditioning_and_lag_sweep.py) | No module docstring supplied. | build_kalman_sampler, run_comprehensive_diagnostic |
| [fast_state_cache.py](auxiliary_bundle/diagnostics_and_benchmarks/fast_state_cache.py) | No module docstring supplied. | save_init_state, load_init_state |
| [find_optimal_batch_size.py](auxiliary_bundle/diagnostics_and_benchmarks/find_optimal_batch_size.py) | No module docstring supplied. | probe_batch_size, binary_search_optimal_bs |
| [fitting_reference.py](auxiliary_bundle/diagnostics_and_benchmarks/fitting_reference.py) | No module docstring supplied. | StopResampling, _wrapped_resample, _set_parallel_flag, init_model, fit_model, apply_model, estimate_syllable_marginals, update_hypparams, expected_marginal_likelihoods |
| [orchestrator.py](auxiliary_bundle/diagnostics_and_benchmarks/orchestrator.py) | No module docstring supplied. | get_vram_usage_mb, check_tensor_health, _vectorized_gpu_ar_kernel, custom_resample_ar_params_gpu, baseline_resample_ar_params_cpu, fast_resample_discrete_stateseqs, fast_joint_resample_pose, _fast_numba_assemble_banded_system, _solve_single_banded_seq, fast_compiled_banded_gmrf; plus 4 more |
| [preseed_cache.py](auxiliary_bundle/diagnostics_and_benchmarks/preseed_cache.py) | No module docstring supplied. | Script statements only |
| [profile_fit_model_gap.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_fit_model_gap.py) | diagnostics/profile_fit_model_gap.py | sync, main |
| [profile_full_dataset_granular.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_full_dataset_granular.py) | diagnostics/profile_full_dataset_granular.py | get_process_ram_gb, get_gpu_vram_gb, sync, reset_sub_timings, instrumented_resample_discrete, instrumented_resample_ar, instrumented_resample_continuous, instrumented_resample_location, instrumented_resample_heading, instrumented_resample_scales; plus 2 more |
| [profile_granular_ar_and_slds.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_granular_ar_and_slds.py) | diagnostics/profile_granular_ar_and_slds.py | sync, run_profiler |
| [profile_granular_subops.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_granular_subops.py) | No module docstring supplied. | profile_granular_subops |
| [profile_jitted_full_loop.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_jitted_full_loop.py) | No module docstring supplied. | jit_resample_heading, jit_pose_prep, jit_resample_scales, main |
| [profile_kpms_breakdown.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_kpms_breakdown.py) | No module docstring supplied. | device_put_as_scalar, get_gpu_memory_mb, instrumented_resample_arhmm, instrumented_resample_slds, main |
| [profile_line_by_line.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_line_by_line.py) | No module docstring supplied. | main |
| [profile_production_loop_gap.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_production_loop_gap.py) | diagnostics/profile_production_loop_gap.py | sync, main |
| [profile_scaling_memory.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_scaling_memory.py) | No module docstring supplied. | get_ram_gb, main |
| [profile_timing_breakdown.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_timing_breakdown.py) | No module docstring supplied. | sync, main |
| [profile_z_breakdown.py](auxiliary_bundle/diagnostics_and_benchmarks/profile_z_breakdown.py) | No module docstring supplied. | log, sync, main |
| [run_3session_frontier_benchmark.py](auxiliary_bundle/diagnostics_and_benchmarks/run_3session_frontier_benchmark.py) | No module docstring supplied. | get_vram_usage_mb, hybrid_vectorized_ar_sample, pure_fp64_vectorized_ar_sample, pure_fp64_fast_z_sample, build_and_sample_banded_gmrf, _fast_numba_assemble_banded_system, _solve_single_banded_seq, fast_compiled_banded_gmrf, _numba_cpu_kalman_single_seq, numba_cpu_kalman_sampler; plus 1 more |
| [run_final_3session_baseline_vs_optimized.py](auxiliary_bundle/diagnostics_and_benchmarks/run_final_3session_baseline_vs_optimized.py) | No module docstring supplied. | send_ntfy, get_gpu_vram_mb, get_cpu_ram_mb, run_gibbs_eval, main |
| [run_full_karen_single_iter.py](auxiliary_bundle/diagnostics_and_benchmarks/run_full_karen_single_iter.py) | No module docstring supplied. | send_ntfy, get_vram_gb, get_ram_gb, main |
| [run_remaining_frontiers_benchmark.py](auxiliary_bundle/diagnostics_and_benchmarks/run_remaining_frontiers_benchmark.py) | No module docstring supplied. | _numba_assemble_and_solve_tridiagonal_v, fused_resample_pose, _fast_numba_ffbs_z, main |
| [test_2_sessions_real.py](auxiliary_bundle/diagnostics_and_benchmarks/test_2_sessions_real.py) | No module docstring supplied. | main |
| [test_advanced_kalman_frontiers.py](auxiliary_bundle/diagnostics_and_benchmarks/test_advanced_kalman_frontiers.py) | No module docstring supplied. | run_advanced_frontiers_test |
| [test_amph_production_acceptance.py](auxiliary_bundle/diagnostics_and_benchmarks/test_amph_production_acceptance.py) | diagnostics/test_amph_production_acceptance.py | get_process_ram_gb, get_gpu_vram_gb, sync, run_production_acceptance |
| [test_amph_production_warmed.py](auxiliary_bundle/diagnostics_and_benchmarks/test_amph_production_warmed.py) | diagnostics/test_amph_production_warmed.py | get_process_ram_gb, get_gpu_vram_gb, load_filtered_sessions, run_acceptance_benchmark |
| [test_cached_slds_iter.py](auxiliary_bundle/diagnostics_and_benchmarks/test_cached_slds_iter.py) | No module docstring supplied. | main |
| [test_checkpoint_resume_and_fused_transform.py](auxiliary_bundle/diagnostics_and_benchmarks/test_checkpoint_resume_and_fused_transform.py) | diagnostics/test_checkpoint_resume_and_fused_transform.py | numpy_transform_reference, test_fused_transform_reused_and_partial, test_checkpoint_resume_continuity, main |
| [test_consolidated_nan_checker.py](auxiliary_bundle/diagnostics_and_benchmarks/test_consolidated_nan_checker.py) | diagnostics/test_consolidated_nan_checker.py | create_mock_pytree, test_nan_checker |
| [test_continuous_sampler_equivalence.py](auxiliary_bundle/diagnostics_and_benchmarks/test_continuous_sampler_equivalence.py) | No module docstring supplied. | log, main |
| [test_cpu_cache_validity.py](auxiliary_bundle/diagnostics_and_benchmarks/test_cpu_cache_validity.py) | diagnostics/test_cpu_cache_validity.py | test_cache_validity |
| [test_deep_optimizations.py](auxiliary_bundle/diagnostics_and_benchmarks/test_deep_optimizations.py) | No module docstring supplied. | benchmark_hmm, benchmark_kalman |
| [test_double_buffer_ffbs.py](auxiliary_bundle/diagnostics_and_benchmarks/test_double_buffer_ffbs.py) | diagnostics/test_double_buffer_ffbs.py | run_serial_ffbs_reference, test_double_buffer |
| [test_fast_ar_accum.py](auxiliary_bundle/diagnostics_and_benchmarks/test_fast_ar_accum.py) | No module docstring supplied. | numba_accumulate_ar_sufficient_stats, main |
| [test_fast_pipeline_2sessions.py](auxiliary_bundle/diagnostics_and_benchmarks/test_fast_pipeline_2sessions.py) | No module docstring supplied. | get_2_sessions, main |
| [test_gpu_pipeline_real_small.py](auxiliary_bundle/diagnostics_and_benchmarks/test_gpu_pipeline_real_small.py) | No module docstring supplied. | main |
| [test_gpu_pipeline_small.py](auxiliary_bundle/diagnostics_and_benchmarks/test_gpu_pipeline_small.py) | No module docstring supplied. | _jit_resample_discrete_batch, batched_resample_discrete_stateseqs, _jit_resample_x_batch, batched_resample_continuous_stateseqs, _jit_resample_h_batch, batched_resample_heading, _jit_resample_s_batch, batched_resample_scales, _jit_resample_v_batch, batched_resample_location |
| [test_hmm_sequential_posterior.py](auxiliary_bundle/diagnostics_and_benchmarks/test_hmm_sequential_posterior.py) | diagnostics/test_hmm_sequential_posterior.py | compute_exact_forward_backward, run_temporal_posterior_test |
| [test_kernel_semantic_parity.py](auxiliary_bundle/diagnostics_and_benchmarks/test_kernel_semantic_parity.py) | No module docstring supplied. | test_get_lags_parity, test_negative_control_old_buggy_get_lags_fails, test_ar_params_parity, test_discrete_resampling_parity, test_scales_and_location_parity, test_continuous_stateseqs_distribution_parity |
| [test_masked_sampler_diagnostics.py](auxiliary_bundle/diagnostics_and_benchmarks/test_masked_sampler_diagnostics.py) | No module docstring supplied. | _numba_assemble_and_solve_with_pseudo_obs, numba_resample_continuous_with_pseudo_obs, run_diagnostics |
| [test_mixed_precision_boundaries.py](auxiliary_bundle/diagnostics_and_benchmarks/test_mixed_precision_boundaries.py) | No module docstring supplied. | fast_gumbel_z_sample, evaluate_mixed_precision_drift |
| [test_multi_iter_stability.py](auxiliary_bundle/diagnostics_and_benchmarks/test_multi_iter_stability.py) | No module docstring supplied. | custom_ar_sample, custom_z_sample, test_multi_iter_drift |
| [test_numba_ar_stats.py](auxiliary_bundle/diagnostics_and_benchmarks/test_numba_ar_stats.py) | No module docstring supplied. | numba_accumulate_ar_sufficient_stats, test |
| [test_optimized_kernels_standalone.py](auxiliary_bundle/diagnostics_and_benchmarks/test_optimized_kernels_standalone.py) | Test script for verified optimization of: | test_ffbs_parity, test_lapack_banded_parity |
| [test_paper_frontiers_pcg_and_assoc_scan.py](auxiliary_bundle/diagnostics_and_benchmarks/test_paper_frontiers_pcg_and_assoc_scan.py) | No module docstring supplied. | associative_scan_hmm_filter, gpu_papandreou_yuille_pcg_resample_x, main |
| [test_par_hmm.py](auxiliary_bundle/diagnostics_and_benchmarks/test_par_hmm.py) | No module docstring supplied. | test_parallel_hmm |
| [test_tier1_v_and_z.py](auxiliary_bundle/diagnostics_and_benchmarks/test_tier1_v_and_z.py) | No module docstring supplied. | _assemble_and_solve_tridiagonal_v, _numba_cpu_ffbs, main |
| [test_unmonkeypatch_routine.py](auxiliary_bundle/diagnostics_and_benchmarks/test_unmonkeypatch_routine.py) | diagnostics/test_unmonkeypatch_routine.py | compute_syllable_stats, generate_synthetic_session_data, run_experiment |
| [verify_big_scope_equivalence.py](auxiliary_bundle/diagnostics_and_benchmarks/verify_big_scope_equivalence.py) | No module docstring supplied. | get_all_durations, compute_model_macro_metrics, main |
| [verify_bounded_variance_ratio.py](auxiliary_bundle/diagnostics_and_benchmarks/verify_bounded_variance_ratio.py) | No module docstring supplied. | run_bounded_experiment |
| [verify_full_drift_and_stability.py](auxiliary_bundle/diagnostics_and_benchmarks/verify_full_drift_and_stability.py) | No module docstring supplied. | _fast_numba_assemble_banded_system, _assemble_and_solve_tridiagonal_v, main |

## Shell launchers

- [launch_2ar_2slds_test.sh](auxiliary_bundle/production_and_test_runners/launch_2ar_2slds_test.sh)
- [launch_tandem_cohort_production.sh](auxiliary_bundle/production_and_test_runners/launch_tandem_cohort_production.sh)

These assume the original machine and repository layout. They were syntax-checked with `bash -n`, not launched. See the dossier for background-job, notification and missing-watchdog considerations.

## Historical documents

- [KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md](auxiliary_bundle/KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md)
- [BUG_REPORT_AND_PARITY_RESOLUTION.md](auxiliary_bundle/docs/BUG_REPORT_AND_PARITY_RESOLUTION.md)
- [BUG_REPORT_KPMS_PERFORMANCE.md](auxiliary_bundle/docs/BUG_REPORT_KPMS_PERFORMANCE.md)
- [CONTINUOUS_SAMPLER_DISCREPANCY_ANALYSIS.md](auxiliary_bundle/docs/CONTINUOUS_SAMPLER_DISCREPANCY_ANALYSIS.md)
- [DUAL_VIDEO_PIPELINE_ARCHITECTURE.md](auxiliary_bundle/docs/DUAL_VIDEO_PIPELINE_ARCHITECTURE.md)
- [GRANULAR_PROFILE_AR_SLDS.md](auxiliary_bundle/docs/GRANULAR_PROFILE_AR_SLDS.md)
- [KPMS_KALMAN_FRONTIERS_AND_OPTIMIZATIONS.md](auxiliary_bundle/docs/KPMS_KALMAN_FRONTIERS_AND_OPTIMIZATIONS.md)
- [KPMS_OPTIMIZATION_REPORT.md](auxiliary_bundle/docs/KPMS_OPTIMIZATION_REPORT.md)
- [MEASURED_GRANULAR_PROFILE_FULL_DATASET.md](auxiliary_bundle/docs/MEASURED_GRANULAR_PROFILE_FULL_DATASET.md)
- [OPTIMIZATION_EXPLANATION.md](auxiliary_bundle/docs/OPTIMIZATION_EXPLANATION.md)
- [PREEMPTIVE_TENSOR_SIZING_AND_OOM_PREVENTION.md](auxiliary_bundle/docs/PREEMPTIVE_TENSOR_SIZING_AND_OOM_PREVENTION.md)
- [REAL_DATASET_SCALING_AND_REALIZATIONS.md](auxiliary_bundle/docs/REAL_DATASET_SCALING_AND_REALIZATIONS.md)
- [TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md](auxiliary_bundle/docs/TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md)
- [kpms_tandem_cohort_investigation.md](auxiliary_bundle/docs/kpms_tandem_cohort_investigation.md)

- [KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md](auxiliary_bundle/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md)
