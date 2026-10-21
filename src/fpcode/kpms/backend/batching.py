"""KPMS sizing; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from typing import Optional

from fpcode.kpms.backend.configuration import CONFIG, BatchConfig


def precompute_optimal_batch_config(
    data: dict,
    model: Optional[dict] = None,
    gpu_budget_gb: Optional[float] = None,
    cpu_budget_gb: Optional[float] = None,
    safety_margin: float = 0.70,
    verbose: bool = True
) -> BatchConfig:
    """
    Preemptively computes optimal batch sizes from closed-form tensor memory formulae.

    Eliminates trial-and-error OOM crashes and kernel recompilations by analyzing
    sequence length T, keypoint count K_bps, latent dimension D, and syllable states K_states
    against the hardware's available VRAM and RAM budgets.
    """
    Y = data.get("Y")
    if Y is None:
        return CONFIG

    N = len(Y)
    T = Y.shape[1] if (hasattr(Y, "shape") and len(Y.shape) >= 2) else max(len(y) for y in Y)
    K_bps = Y.shape[2] if (hasattr(Y, "shape") and len(Y.shape) >= 3) else 23

    D = 10
    K_states = 100
    nlags = 3
    if model is not None:
        if "params" in model:
            p = model["params"]
            if "pi" in p:
                K_states = p["pi"].shape[0]
            if "Ab" in p:
                D = p["Ab"].shape[1]
                nlags = (p["Ab"].shape[2] - 1) // D
        elif "states" in model and "x" in model["states"]:
            D = model["states"]["x"].shape[-1]


    total_vram_gb = None
    free_vram_gb = None
    if gpu_budget_gb is None:
        # Query Hardware Memory
        total_vram_gb = None
        free_vram_gb = None
        try:
            import jax
            dev = jax.devices()[0]
            if hasattr(dev, "memory_stats"):
                stats = dev.memory_stats()
                if "bytes_limit" in stats and stats["bytes_limit"] > 0:
                    limit = stats["bytes_limit"]
                    in_use = stats.get("bytes_in_use", 0)
                    free_vram_gb = (limit - in_use) / 1e9
                    total_vram_gb = limit / 1e9
        except Exception:
            pass

        if free_vram_gb is None:
            try:
                import torch
                if torch.cuda.is_available():
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    total_vram_gb = total_bytes / 1e9
                    free_vram_gb = free_bytes / 1e9
            except Exception:
                pass

    if gpu_budget_gb is None:
        if free_vram_gb is not None:
            gpu_budget_gb = free_vram_gb * safety_margin
        elif total_vram_gb is not None:
            gpu_budget_gb = total_vram_gb * 0.50
        else:
            gpu_budget_gb = 10.0  # safe default

    if cpu_budget_gb is None:
        try:
            import psutil
            cpu_budget_gb = (psutil.virtual_memory().available / 1e9) * 0.60
        except Exception:
            cpu_budget_gb = 25.0

    # Closed-form Analytical Tensor Bytes per Sequence (in FP64)
    # 1. Discrete state emissions: intermediate mu_all (N,T,K,D) + diff (N,T,K,D) + einsum product requires 4x buffer
    bytes_per_seq_z = max(1, (T - nlags)) * K_states * D * 8 * 4
    # 2. Heading resampling (rotated keypoints + inner products)
    bytes_per_seq_h = T * K_bps * 8 * 8
    # 3. Location / Centroid
    bytes_per_seq_v = T * K_bps * 6 * 8
    # 4. Noise scales (compute_squared_error intermediate rotations + residuals)
    bytes_per_seq_s = T * K_bps * 11 * 8
    # 5. Continuous states Banded Cholesky on CPU: (bw + 4) * D + 2 * K_bps (workspace + solves)
    bw = (nlags + 1) * D - 1
    bytes_per_seq_x_cpu = ((bw + 4) * D + 2 * K_bps) * T * 8

    gpu_budget_bytes = gpu_budget_gb * 1e9
    cpu_budget_bytes = cpu_budget_gb * 1e9

    optimal_discrete_bs = max(1, min(N, 10, int(gpu_budget_bytes // bytes_per_seq_z)))
    optimal_heading_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_h)))
    optimal_gpu_loc_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_v)))
    optimal_scales_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_s)))
    optimal_chunk_x = max(1, min(N, 50, int(cpu_budget_bytes // bytes_per_seq_x_cpu)))

    CONFIG.discrete_bs = optimal_discrete_bs
    CONFIG.heading_bs = optimal_heading_bs
    CONFIG.gpu_loc_bs = optimal_gpu_loc_bs
    CONFIG.scales_bs = optimal_scales_bs
    CONFIG.continuous_chunk_size = optimal_chunk_x

    if verbose:
        print(f"[Preemptive Batching] Analytical Tensor Sizing (N={N}, T={T:,}, K_bps={K_bps}, D={D}, K_states={K_states}):")
        print(f"  • GPU VRAM Budget: {gpu_budget_gb:.2f} GB | Host RAM Budget: {cpu_budget_gb:.2f} GB")
        print(f"  • Discrete emissions per seq: {bytes_per_seq_z / 1e6:7.1f} MB  ==> discrete_bs = {optimal_discrete_bs}")
        print(f"  • Heading rotation per seq:   {bytes_per_seq_h / 1e6:7.1f} MB  ==> heading_bs  = {optimal_heading_bs}")
        print(f"  • Centroid / Location per seq:{bytes_per_seq_v / 1e6:7.1f} MB  ==> gpu_loc_bs  = {optimal_gpu_loc_bs}")
        print(f"  • Noise scales per seq:       {bytes_per_seq_s / 1e6:7.1f} MB  ==> scales_bs   = {optimal_scales_bs}")
        print(f"  • CPU Banded Cholesky per seq:{bytes_per_seq_x_cpu / 1e6:7.1f} MB  ==> cont_chunk  = {optimal_chunk_x}")

    return CONFIG
