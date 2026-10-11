# ------------------------------------------------------------
# load_and_analyze_session_state.py
# ------------------------------------------------------------

def load_state(pkl_path, *, sanity_check=True):
    import pickle
    from pathlib import Path
    pkl_path = Path(pkl_path)
    if sanity_check:
        print(f"Loading state: {pkl_path}")
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)
    if sanity_check:
        print(f"Loaded {len(state.get('sessions', []))} session(s)")
    return state  # dict: sessions, model_names, video_frame_indexes, original_lengths, project_dir_full, kappa


def merge_syllables_into_sync_df(sessions, *, sanity_check=True, sync_key="synchronized_data_tracking_0"):
    updated = 0
    for i, s in enumerate(sessions):
        try:
            if not hasattr(s, 'all_results'):
                raise KeyError("missing 'all_results'")
            if sync_key not in s.dfs.data:
                raise KeyError(f"missing '{sync_key}'")
            df = s.dfs.data[sync_key]
            syll = s.all_results['syllable']
            if len(df) != len(syll):
                raise ValueError(f"length mismatch: df={len(df)} vs syll={len(syll)}")
            df['syllable'] = syll
            updated += 1
            if sanity_check:
                print(f"OK {i+1}: {s.trial_id} -> 'syllable' attached")
        except Exception as e:
            if sanity_check:
                print(f"SKIP {i+1}: {getattr(s, 'trial_id', '?')} -> {e}")
    if sanity_check:
        print(f"Updated {updated}/{len(sessions)} sessions")
    return updated


def select_model(project_dir, model_names, *, sanity_check=True, strategy="first"):
    # strategies: "first" | "eml_best"
    import keypoint_moseq as kpms
    import numpy as np
    if strategy == "eml_best":
        eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(str(project_dir), model_names)
        idx = int(np.argmax(eml_scores))
        chosen = model_names[idx]
        if sanity_check:
            print("EML scores:", dict(zip(model_names, map(float, eml_scores))))
            print(f"Chosen model by EML: {chosen}")
        return chosen
    chosen = model_names[0]
    if sanity_check:
        print(f"Chosen model (first): {chosen}")
    return chosen


def load_model_and_extract(project_dir, model_name, *, sanity_check=True):
    import keypoint_moseq as kpms
    model, data, metadata, current_iter = kpms.load_checkpoint(str(project_dir), model_name)
    # 🔧 persist results so compute_moseq_df can read results.h5 later
    results = kpms.extract_results(model, metadata, str(project_dir), model_name, save_results=True)
    config = kpms.load_config(str(project_dir))

    # ensure bodyparts/use_bodyparts consistent
    if 'use_bodyparts' in config:
        config = dict(config)
        config['bodyparts'] = config['use_bodyparts']

    if sanity_check:
        print(f"Loaded checkpoint '{model_name}' (iter={current_iter})")
    return model, data, metadata, results, config



def estimate_coordinates_from_checkpoint(model, metadata):
    import jax.numpy as jnp
    from jax_moseq.models.keypoint_slds import estimate_coordinates
    from jax_moseq.utils import unbatch
    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )
    coordinates_est = unbatch(Y_est, *metadata)
    return coordinates_est


def compute_centroids_headings(coordinates_est, config):
    import keypoint_moseq as kpms
    # ensure bodyparts key is what kpms expects
    if 'use_bodyparts' in config:
        config = dict(config)
        config['bodyparts'] = config['use_bodyparts']
    centroids, headings = kpms.get_centroids_headings(coordinates_est, **config)
    return centroids, headings


def map_syllable_parts_to_videos(video_frame_indexes: dict, video_dir: str, *, sanity_check=True):
    import os
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_exts)]
    m = {}
    misses = 0
    for part_key in video_frame_indexes.keys():
        base = part_key.split('_part')[0]
        try:
            video_root = base.split('DLC_')[0]
            match = next((vf for vf in video_files if vf.startswith(video_root)), None)
            if match:
                m[part_key] = os.path.join(video_dir, match)
            else:
                misses += 1
                if sanity_check:
                    print(f"No matching video for key: {part_key}")
        except Exception:
            misses += 1
            if sanity_check:
                print(f"Failed to parse key: {part_key}")
    if sanity_check:
        print(f"Mapped {len(m)} keys to videos; {misses} missing")
    return m


def align_common_keys(results, video_frame_indexes, mapping, coordinates_est, centroids, headings, *, sanity_check=True):
    kr = set(results.keys())
    kf = set(video_frame_indexes.keys())
    common = kr & kf
    if sanity_check and (len(common) != len(kr) or len(common) != len(kf)):
        print(f"results-only: {sorted(kr - common)[:5]} ...")
        print(f"frames-only:  {sorted(kf - common)[:5]} ...")
    filtered = (
        {k: results[k] for k in common},
        {k: video_frame_indexes[k] for k in common},
        {k: mapping[k] for k in common if k in mapping},
        {k: coordinates_est[k] for k in common if k in coordinates_est},
        {k: centroids[k] for k in common if k in centroids},
        {k: headings[k] for k in common if k in headings},
    )
    return filtered


def generate_grid_movies_aligned(project_dir, model_name, results_common, video_frame_indexes_common,
                                 mapping_common, coordinates_common, centroids_common, headings_common,
                                 *, fps=60, overlay_keypoints=False, config=None, sanity_check=True):
    import keypoint_moseq as kpms
    cfg = dict(config or {})

    # Avoid duplicate kw
    cfg.pop('fps', None)

    # 🔧 ensure bodyparts/use_bodyparts consistency
    if 'use_bodyparts' in cfg:
        cfg['bodyparts'] = cfg['use_bodyparts']
    elif 'bodyparts' in cfg:
        cfg['use_bodyparts'] = cfg['bodyparts']

    # 🔍 Optional sanity check right here
    if sanity_check:
        try:
            sample = next(iter(coordinates_common.values()))
            K = sample.shape[1]
            if len(cfg.get('bodyparts', [])) != K:
                print(f"⚠️ Warning: bodyparts length {len(cfg.get('bodyparts', []))} "
                      f"!= coordinates keypoints {K}")
        except StopIteration:
            print("⚠️ No coordinates available for sanity check.")

    kpms.generate_grid_movies(
        results_common,
        str(project_dir),
        model_name,
        video_paths=mapping_common,
        coordinates=coordinates_common,
        centroids=centroids_common,
        headings=headings_common,
        fps=fps,
        video_frame_indexes=video_frame_indexes_common,
        overlay_keypoints=overlay_keypoints,
        **cfg
    )
    if sanity_check:
        print("Grid movies generation dispatched.")


def assign_groups_and_write_index_csv(sessions, metadata, project_dir, *, sanity_check=True):
    import os
    import re

    import pandas as pd
    names = metadata[0]  # name per recording
    for sess in sessions:
        sess.result_names = []
        h5_path = getattr(sess, 'kpms_h5_path', '')
        for name in names:
            base = re.sub(r'_part\d+$', '', name)
            if base in h5_path and name not in sess.result_names:
                sess.result_names.append(name)
    rows = []
    for sess in sessions:
        for name in getattr(sess, 'result_names', []):
            rows.append({"name": name, "group": sess.genotype})
    idx_df = pd.DataFrame(rows)
    out_csv = os.path.join(str(project_dir), "index.csv")
    idx_df.to_csv(out_csv, index=False)
    if sanity_check:
        print(f"Wrote index.csv with {len(idx_df)} rows -> {out_csv}")
    return idx_df


def compute_moseq_tables(project_dir, model_name, *, fps=60, min_frequency=0.005, sanity_check=True):
    import os

    import keypoint_moseq as kpms
    results_h5 = os.path.join(str(project_dir), model_name, "results.h5")
    if not os.path.exists(results_h5):
        # create it on the fly
        model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
        kpms.extract_results(model, metadata, str(project_dir), model_name, save_results=True)
        if sanity_check:
            print(f"Created missing results.h5 at {results_h5}")

    moseq_df = kpms.compute_moseq_df(str(project_dir), model_name, smooth_heading=True)
    stats_df = kpms.compute_stats_df(
        str(project_dir),
        model_name,
        moseq_df,
        min_frequency=min_frequency,
        groupby=["group", "name"],
        fps=fps,
    )
    if sanity_check:
        print(f"moseq_df rows: {len(moseq_df)}, stats_df rows: {len(stats_df)}")
    return moseq_df, stats_df


def apply_plotting_setup_to_sessions(sessions, *, sanity_check=True):
    from studies.photometry.exploratory.config import PLOTTING_CONFIG
    from studies.photometry.exploratory.processing.plotting_setup import PlottingSetup
    PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)
    if sanity_check:
        print("Applied plotting setup to sessions.")


def plot_data_availability(sessions, project_dir, *, sanity_check=True):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    save_dir = Path(project_dir) / "availability_plots"
    save_dir.mkdir(exist_ok=True)

    plots = 0
    for s in sessions:
        if hasattr(s, 'all_results') and 'syllable' in s.all_results:
            mask = ~np.isnan(s.all_results['syllable'])
            plt.figure(figsize=(18, 3))
            plt.plot(mask, linewidth=1.5)
            plt.title(f"Data Availability: {s.trial_id}")
            plt.xlabel("Frame"); plt.ylabel("Valid (1) / NaN (0)")
            plt.yticks([0,1], ['NaN','Valid']); plt.ylim(-0.1, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            out_path = save_dir / f"{s.trial_id}_availability.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

            plots += 1
            if sanity_check:
                print(f"Saved availability plot -> {out_path}")

    if sanity_check:
        print(f"Generated {plots} availability plot(s). Saved to {save_dir}")
    return plots

def run_load_and_analyze(
    state_pkl,
    *,
    sanity_check=True,
    model_select="first",        # or "eml_best"
    explicit_model_name=None,    # override selection
    video_dir=None,              # required for grid movies
    fps=60,
    min_freq_for_stats=0.005,
    overlay_keypoints=False,
):
    # 1) Load saved state
    state = load_state(state_pkl, sanity_check=sanity_check)
    sessions = state["sessions"]
    model_names = state["model_names"]
    video_frame_indexes = state["video_frame_indexes"]
    project_dir = state["project_dir_full"]

    # 2) Attach syllables to sync df
    merge_syllables_into_sync_df(sessions, sanity_check=sanity_check)

    # 3) Choose model
    model_name = explicit_model_name or select_model(project_dir, model_names, sanity_check=sanity_check, strategy=model_select)

    # 4) Load model + results + config
    model, data, metadata, results, config = load_model_and_extract(project_dir, model_name, sanity_check=sanity_check)

    # 5) Coordinates, centroids, headings
    coordinates_est = estimate_coordinates_from_checkpoint(model, metadata)
    centroids, headings = compute_centroids_headings(coordinates_est, config)

    # 6) Optional grid movies
    if video_dir is not None:
        mapping = map_syllable_parts_to_videos(video_frame_indexes, video_dir, sanity_check=sanity_check)
        res_c, vfi_c, map_c, coord_c, cent_c, head_c = align_common_keys(
            results, video_frame_indexes, mapping, coordinates_est, centroids, headings, sanity_check=sanity_check
        )
        generate_grid_movies_aligned(project_dir, model_name, res_c, vfi_c, map_c, coord_c, cent_c, head_c,
                                     fps=fps, overlay_keypoints=overlay_keypoints, config=config, sanity_check=sanity_check)

    # 7) Group assignment + tables
    index_df = assign_groups_and_write_index_csv(sessions, metadata, project_dir, sanity_check=sanity_check)
    moseq_df, stats_df = compute_moseq_tables(project_dir, model_name, fps=fps, min_frequency=min_freq_for_stats, sanity_check=sanity_check)

    # 8) Always apply plotting setup (like in creation step)
    apply_plotting_setup_to_sessions(sessions, sanity_check=sanity_check)

    # 9) Optional: quick availability plots
    plot_data_availability(sessions, project_dir, sanity_check=sanity_check)

    return {
        "project_dir": str(project_dir),
        "model_name": model_name,
        "index_df_rows": len(index_df),
        "moseq_df_rows": len(moseq_df),
        "stats_df_rows": len(stats_df),
    }


# Example CLI-style entry (adjust paths)
if __name__ == "__main__":
    from pathlib import Path
    out = run_load_and_analyze(
        state_pkl=Path("/home/fsp585/Desktop/FiberphotometryCode/kpms_projects/kpms_all_sessions_frac_1.0_ld9_jn24/create_session_state_test.pkl"),
        sanity_check=True,
        model_select="first",             # or "eml_best"
        explicit_model_name=None,         # or 'final_model_1.20e+06_idx1'
        video_dir="/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/video_files",
        fps=60,
        min_freq_for_stats=0.005,
        overlay_keypoints=False,
    )
    print(out)
