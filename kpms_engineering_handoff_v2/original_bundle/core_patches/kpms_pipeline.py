import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", "")

import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import keypoint_moseq as kpms
from keypoint_moseq.viz import get_durations
import jax

# --- CUSTOM IMPORTS ---
from utils.paths import ref_pairs_from_sessions, get_key_to_aid_mapping
from utils.trimming import presence_mask_from_conf, filter_with_masks, clamp_conf_nonneg, subset_after_filter
from utils.batch_init import batched_init_model_by_video
# Ensure this filename matches exactly what you have on disk
from utils.resample_batching import apply_optimizations, restore_original_kernels 
from utils.flipping import flip_data 

def run_pipeline(
    sessions,
    project_dir,
    detector="deeplabcut",
    cam="center_cam",
    subroutine="jonathan",
    latent_dim=10,
    num_ar_iters=1,
    total_iters=1,
    override_video_extension=None,
    pca_mode="load",
    model_name=None,
    init_kappa=None,
    decrease_kappa_factor=None,
    subset_fraction=1.0,
    subset_blocks=5,
    subset_block_len=None,
    subset_seed=None,
    mixed_map_iters=24,
    seed=None,
    downsample_rate=1,  
    new_fps=None,
    use_batched_init=False,
    use_memory_patch=True,
    flipping_mode="none",
    subset_method="random",
    save_every_n_iters=25,
    location_aware=False,
    **kwargs
):
    # 0) Conditionally Apply the Memory Patch
    if use_memory_patch:
        print("[setup] Enabling memory optimization patch (Batched Resampling)...")
        apply_optimizations()
    else:
        print("[setup] Memory optimization patch disabled (restoring upstream kernels)...")
        restore_original_kernels()

    # 1) Derive file lists
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine=subroutine, cam=cam)
    vid_dir = str(Path(video_paths[0]).parent)
    ext = (override_video_extension or Path(video_paths[0]).suffix.lstrip(".").lower())

    # 2) Load keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, detector)

    # 2b) Data Augmentation
    # We flip BEFORE masking/filtering so that we generate masks for the new flipped data too.
    if flipping_mode is not None and flipping_mode != "none":
        # Build Key to AID mapping for correct flipping logic
        final_key_to_aid = get_key_to_aid_mapping(sessions, coordinates)
        
        # Hardcoded width 404 based on ffprobe of source videos
        coordinates, confidences = flip_data(
            coordinates, confidences, bodyparts, image_width=404, mode=flipping_mode, key_to_aid=final_key_to_aid
        )

    # 3) Downsampling Config Update

    if downsample_rate > 1 and new_fps is not None:
        kpms.update_config(project_dir, fps=new_fps)

    # 4) Presence masks
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}

    # 5) Compress + outlier removal
    filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
        coordinates, confidences, all_masks, project_dir,
        return_video_frame_indexes=True, downsample_rate=downsample_rate, 
        **kpms.load_config(project_dir)
    )
    # Immediately free the uncompressed raw dicts to reclaim ~15 GB RAM
    del coordinates, confidences, all_masks
    import gc
    gc.collect()

    clamp_conf_nonneg(filtered_confs)

    if subset_fraction < 1.0:
        if subset_method == "activity":
             from utils.trimming import subset_by_activity
             filtered_coords, filtered_confs, video_frame_indexes = subset_by_activity(
                filtered_coords, filtered_confs, video_frame_indexes,
                frac=subset_fraction
            )
        else:
            # Default "random" spread
            filtered_coords, filtered_confs, video_frame_indexes = subset_after_filter(
                filtered_coords, filtered_confs, video_frame_indexes,
                frac=subset_fraction, n_blocks=subset_blocks,
                block_len=subset_block_len, seed=subset_seed,
            )

    # Filter out any sequences with fewer than 5 frames (MoSeq requires >= 5)
    keys_to_remove = [k for k, v in filtered_coords.items() if len(v) < 5]
    for k in keys_to_remove:
        print(f"[warning] Removing sequence '{k}' because it has only {len(filtered_coords[k])} frames (MoSeq requires >= 5)")
        filtered_coords.pop(k)
        filtered_confs.pop(k)
        if k in video_frame_indexes:
            video_frame_indexes.pop(k)

    # 6) Config
    kpms.update_config(project_dir, video_dir=vid_dir, video_extension=ext, latent_dim=latent_dim)

    # 7) Format data
    data, metadata = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(project_dir))
    # Retain observations in host DRAM to eliminate 12 synchronous PCIe downloads per SLDS iteration
    data["Y"] = np.asarray(data["Y"])

    # 8) PCA
    pca = None
    if pca_mode == "fit":
        print("Forced PCA fit enabled. Fitting new PCA model...")
    elif pca_mode in ["load", "load_or_fit"]:
        try:
            pca = kpms.load_pca(project_dir)
            print(f"Loaded PCA from {project_dir}")
        except Exception:
            if pca_mode == "load":
                print(f"FAILED to load PCA from {project_dir}. Mode is strict 'load'.")
                raise
            print("PCA load failed or file missing. Fitting new PCA model...")

    def _safe_fit_pca(d_dict, p_dir, max_seqs=30):
        if "Y" in d_dict and len(d_dict["Y"]) > max_seqs:
            print(f"[PCA] Subsampling {max_seqs} representative sequences from {len(d_dict['Y'])} to avoid GPU VRAM exhaustion...")
            sample_idx = np.linspace(0, len(d_dict["Y"]) - 1, max_seqs, dtype=int)
            sub_d = {
                k: v[sample_idx] if (hasattr(v, "__len__") and len(v) == len(d_dict["Y"])) else v
                for k, v in d_dict.items()
            }
            return kpms.fit_pca(**sub_d, **kpms.load_config(p_dir))
        return kpms.fit_pca(**d_dict, **kpms.load_config(p_dir))

    if pca is None:
        pca = _safe_fit_pca(data, project_dir)
        kpms.save_pca(pca, project_dir)
        print(f"Saved fresh PCA to {project_dir}")
    else:
        # Default fallback
        try:
            pca = kpms.load_pca(project_dir)
        except:
            pca = _safe_fit_pca(data, project_dir)
    
    # === Batching settings ===
    from jax_moseq.utils import set_mixed_map_iters
    import math
    eff_mixed_map_iters = mixed_map_iters
    if (subset_fraction is not None) and (subset_fraction < 1.0):
        eff_mixed_map_iters = max(1, math.ceil(mixed_map_iters * subset_fraction))
    set_mixed_map_iters(eff_mixed_map_iters)

    import utils.resample_batching as rb
    rb.KPMSStateCache.clear()
    if use_memory_patch:
        rb.precompute_optimal_batch_config(data, verbose=True)
    import gc

    while True:
        try:
            # === 9) Initialization & Checkpoint Loading ===
            requested_name = model_name
            model_name_used = requested_name 
            ckpt_path = Path(project_dir) / (requested_name or "NON_EXISTENT") / "checkpoint.h5"

            # [STEP 1] GET STATE (Load or Init)
            if requested_name and ckpt_path.exists():
                print(f"[resume] Checkpoint found. Loading latest state for '{requested_name}'...")
                # Load latest (do not specify iteration, let it find the max)
                model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, requested_name)
                data["Y"] = np.asarray(data["Y"])
                rb.KPMSStateCache.clear()
                if use_memory_patch:
                    rb.precompute_optimal_batch_config(data, model=model, verbose=True)
                print(f"[resume] Loaded at iteration {current_iter}.")
            else:
                print(f"[init] Fresh start. Initializing model...")
                rb.KPMSStateCache.clear()
                rng_key = jax.random.PRNGKey(int(seed)) if seed is not None else jax.random.PRNGKey(0)
                init_config = kpms.load_config(project_dir)

                if use_batched_init:
                    init_bs = min(rb.CONFIG.discrete_bs, 20)
                    print(f"[init] Using BATCHED initialization (batch_size={init_bs})...")
                    model = batched_init_model_by_video(
                        data, pca, init_config, 
                        params_subset_size=5, batch_size=init_bs, seed=rng_key
                    )
                else:
                    print("[init] Using STANDARD initialization...")
                    model = kpms.init_model(data, pca=pca, seed=rng_key, location_aware=location_aware, **init_config)
                    
                if init_kappa is not None:
                     model = kpms.update_hypparams(model, kappa=float(init_kappa))
                     
                current_iter = 0

            # Store iteration at the beginning of the run for progress logging and start-over reference
            run_start_iter = current_iter

            # [STEP 2] PHASE 1: AR-ONLY FIT (Reach the AR Milestone)
            # We only run this if we haven't reached the AR iteration count yet.
            if current_iter < num_ar_iters:
                print(f"[phase 1] Current iter ({current_iter}) < AR target ({num_ar_iters}). Running AR fit in chunks of 10...")
                
                import time
                from datetime import datetime, timedelta
                ar_start_time = time.time()
                
                chunk_size = 10
                while current_iter < num_ar_iters:
                    next_target = min(current_iter + chunk_size, num_ar_iters)
                    start_i = 0 if current_iter == 0 else current_iter + 1
                    is_final_chunk = (next_target == num_ar_iters)
                    should_save = (save_every_n_iters > 0 and next_target % save_every_n_iters == 0) or is_final_chunk
                    chunk_save_every = save_every_n_iters if should_save else None
                    model, model_name_used = kpms.fit_model(
                        model, data, metadata, project_dir,
                        model_name=requested_name,
                        ar_only=True,
                        start_iter=start_i,
                        num_iters=next_target,
                        parallel_message_passing=False,
                        save_every_n_iters=chunk_save_every,
                        generate_progress_plots=is_final_chunk,
                        location_aware=location_aware,
                    )
                    requested_name = model_name_used
                    current_iter = next_target
                    
                    if should_save:
                        savedir = os.path.join(project_dir, model_name_used)
                        checkpoint_path = os.path.join(savedir, "checkpoint.h5")
                        kpms.save_hdf5(
                            checkpoint_path,
                            model,
                            f"model_snapshots/{next_target}",
                            exist_ok=True,
                            overwrite=True,
                        )
                    
                    # Calculate ETA
                    elapsed = time.time() - ar_start_time
                    sec_per_iter = elapsed / (current_iter - run_start_iter) if (current_iter - run_start_iter) > 0 else 0
                    remaining_iters = num_ar_iters - current_iter
                    eta_sec = remaining_iters * sec_per_iter
                    
                    eta_time_str = str(timedelta(seconds=int(eta_sec)))
                    est_completion = (datetime.now() + timedelta(seconds=eta_sec)).strftime('%H:%M:%S')
                    
                    # Send ntfy progress notification
                    if not os.environ.get("DISABLE_NTFY"):
                        try:
                            import urllib.request
                            msg = (
                                f"Model '{model_name_used}' progress: AR {current_iter}/{num_ar_iters} complete.\n"
                                f"AR ETA: {eta_time_str} remaining (Est. AR complete at {est_completion})."
                            )
                            req = urllib.request.Request(
                                "https://ntfy.sh/woody_videos_seb",
                                data=msg.encode("utf-8"),
                                headers={"Title": "MoSeq AR Training Progress"}
                            )
                            with urllib.request.urlopen(req) as response:
                                pass
                        except Exception as e:
                            print(f"Failed to send progress notification: {e}")
                        
                print(f"[phase 1] AR Milestone reached: {current_iter}/{num_ar_iters} AR iterations complete.")
            else:
                print(f"[phase 1] Skipping AR fit (Already at iter {current_iter} >= {num_ar_iters}).")
                model_name_used = requested_name

            # [STEP 3] PHASE 2: SLDS FIT (Only if current_iter < final_target_iter)
            final_target_iter = num_ar_iters + total_iters
            if current_iter < final_target_iter:
                # Update Kappa transition if transitioning from AR to SLDS
                # Kappa is scaled for SLDS transitions: kappa = init_kappa / decrease_kappa_factor
                if (init_kappa is not None) and (decrease_kappa_factor is not None):
                    kappa = float(init_kappa) / float(decrease_kappa_factor)
                    print(f"[transition] Updating Kappa for SLDS: {kappa}")
                    model = kpms.update_hypparams(model, kappa=kappa)

                print(f"[phase 2] Starting SLDS fit from {current_iter} to {final_target_iter} in chunks of 10...")
                
                import time
                from datetime import datetime, timedelta
                slds_start_time = time.time()
                
                # DIAGNOSTICS: Comprehensive Object Inspection
                print("\n=======================================================================")
                print("           MOSEQ DIAGNOSTICS: PRE-SLDS DATA & MODEL MEMORY AUDIT")
                print("=======================================================================")
                def print_dict_diagnostics(tag, obj_dict):
                    total_bytes = 0
                    print(f"\n--- {tag} ---")
                    if isinstance(obj_dict, dict):
                        for k, v in obj_dict.items():
                            try:
                                if hasattr(v, "shape") and hasattr(v, "dtype"):
                                    nbytes = v.nbytes if hasattr(v, "nbytes") else (v.size * v.dtype.itemsize)
                                    total_bytes += nbytes
                                    dev = str(getattr(v, "device", "CPU/NumPy"))
                                    print(f"  * [{k:20s}]: shape={str(v.shape):26s} dtype={str(v.dtype):10s} size={nbytes/1e6:8.2f} MB  dev={dev}")
                                elif isinstance(v, dict):
                                    print(f"  * [{k:20s}] (sub-dict with {len(v)} keys)")
                                else:
                                    print(f"  * [{k:20s}]: type={type(v).__name__}")
                            except Exception as ex:
                                print(f"  * [{k:20s}]: error={ex}")
                    print(f"Total Array Footprint for {tag}: {total_bytes/1e6:.2f} MB ({total_bytes/1e9:.3f} GB)")
                
                print_dict_diagnostics("MODEL DICT ARRAYS", model)
                print_dict_diagnostics("DATA DICT ARRAYS", data)
                print("=======================================================================\n", flush=True)

                chunk_size = 10
                while current_iter < final_target_iter:
                    next_target = min(current_iter + chunk_size, final_target_iter)
                    start_i = 0 if current_iter == 0 else current_iter + 1
                    is_final_chunk = (next_target == final_target_iter)
                    should_save = (save_every_n_iters > 0 and next_target % save_every_n_iters == 0) or is_final_chunk
                    chunk_save_every = save_every_n_iters if should_save else None
                    model = kpms.fit_model(
                        model, data, metadata, project_dir, model_name_used,
                        ar_only=False,
                        start_iter=start_i,
                        num_iters=next_target,
                        parallel_message_passing=False,
                        save_every_n_iters=chunk_save_every,
                        generate_progress_plots=is_final_chunk,
                        location_aware=location_aware,
                    )[0]
                    current_iter = next_target
                    
                    if should_save:
                        savedir = os.path.join(project_dir, model_name_used)
                        checkpoint_path = os.path.join(savedir, "checkpoint.h5")
                        kpms.save_hdf5(
                            checkpoint_path,
                            model,
                            f"model_snapshots/{next_target}",
                            exist_ok=True,
                            overwrite=True,
                        )
                    
                    # Calculate ETA
                    slds_iters_done = current_iter - num_ar_iters
                    elapsed = time.time() - slds_start_time
                    sec_per_iter = elapsed / slds_iters_done if slds_iters_done > 0 else 0
                    remaining_iters = final_target_iter - current_iter
                    eta_sec = remaining_iters * sec_per_iter
                    
                    eta_time_str = str(timedelta(seconds=int(eta_sec)))
                    est_completion = (datetime.now() + timedelta(seconds=eta_sec)).strftime('%H:%M:%S')
                    
                    # Send ntfy progress notification
                    if not os.environ.get("DISABLE_NTFY"):
                        try:
                            import urllib.request
                            msg = (
                                f"Model '{model_name_used}' progress: SLDS {slds_iters_done}/{total_iters} complete.\n"
                                f"SLDS ETA: {eta_time_str} remaining (Est. complete at {est_completion})."
                            )
                            req = urllib.request.Request(
                                "https://ntfy.sh/woody_videos_seb",
                                data=msg.encode("utf-8"),
                                headers={"Title": "MoSeq SLDS Training Progress"}
                            )
                            with urllib.request.urlopen(req) as response:
                                pass
                        except Exception as e:
                            print(f"Failed to send progress notification: {e}")
            else:
                print(f"[phase 2] Training already complete (Current {current_iter} >= Target {final_target_iter}).")
            
            # Succeeded without OOM, break out of while True loop
            break

        except Exception as e:
            err_msg = str(e)
            is_oom = any(keyword in err_msg.lower() for keyword in ["resource_exhausted", "out of memory", "allocation failed", "oom"])
            if is_oom:
                old_bs = rb.CONFIG.discrete_bs
                if old_bs > 1:
                    rb.set_batch_sizes(scale_factor=0.5)
                    new_bs = rb.CONFIG.discrete_bs
                    oom_title = "MoSeq OOM - Scaling Down Batch Sizes"
                    oom_msg = (
                        f"WARNING: Out-of-memory detected at iteration {current_iter} with discrete_bs {old_bs}.\n"
                        f"Scaled down batch sizes (discrete_bs={new_bs}, chunk_size={rb.CONFIG.continuous_chunk_size}) and restarting from iteration {run_start_iter}."
                    )
                    print(f"\n[OOM DETECTED] {oom_msg}\n")
                    
                    # Send ntfy notification
                    try:
                        import urllib.request
                        req = urllib.request.Request(
                            "https://ntfy.sh/woody_videos_seb",
                            data=oom_msg.encode("utf-8"),
                            headers={"Title": oom_title}
                        )
                        with urllib.request.urlopen(req) as response:
                            pass
                    except Exception as ntfy_err:
                        print(f"Failed to send ntfy notification: {ntfy_err}")
                    
                    # Force Python and JAX garbage collection to release VRAM
                    if 'model' in locals():
                        del model
                    gc.collect()
                    
                    # Let the loop start over (it will reload the checkpoint or re-init)
                    continue
            raise e

    # --- Summary metrics ---
    z = np.array(model["states"]["z"])
    mask = np.array(data["mask"])
    median_duration = float(np.median(get_durations(z, mask))) if mask.size else float("nan")
    frames_used = int(mask.sum())

    model_dir = os.path.join(project_dir, model_name_used)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_dir, "video_frame_indexes.npz"),
        **{k: np.asarray(v, dtype=np.int64) for k, v in video_frame_indexes.items()}
    )

    kappa_val = model.get("hypparams", {}).get("kappa", None) if isinstance(model, dict) else None

    return {
        "project_dir": project_dir,
        "video_dir": vid_dir,
        "frames_used": frames_used,
        "median_duration_frames": median_duration,
        "model_name": model_name_used,
        "iters_done": current_iter,
        "kappa_slds": kappa_val,
        "detector": detector,
        "cam": cam,
        "subroutine": subroutine,
        "latent_dim": latent_dim,
        "num_ar_iters": num_ar_iters,
        "total_iters": total_iters,
        "override_video_extension": override_video_extension,
        "pca_mode": pca_mode,
        "init_kappa": init_kappa,
        "decrease_kappa_factor": decrease_kappa_factor,
        "subset_fraction": subset_fraction,
        "subset_blocks": subset_blocks,
        "subset_block_len": subset_block_len,
        "subset_seed": subset_seed,
        "mixed_map_iters": mixed_map_iters,
        "seed": seed,
        "downsample_rate": downsample_rate,
        "new_fps": new_fps,
        "use_batched_init": use_batched_init,
        "use_memory_patch": use_memory_patch,
        "flipping_mode": flipping_mode,
        "subset_method": subset_method
    }