import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import PLOTTING_CONFIG

# fiberphotometry package
# inside fiberphotometry/kpms/create_session_state_2.py
from ..data.data_loading import load_all_sessions
from ..data.oft_session_loading import populate_oft_containers
from ..data.session_loading import populate_containers
from ..data.session_synchronization import synchronize_session_data
from ..processing.plotting_setup import PlottingSetup
from .syllable_appender import extract_and_append_results
from .syllable_generator import generate_syllable_models


# ----------------------------
# Filename details simplifier
# ----------------------------
def simplify_filename_details(filename: str) -> str:
    """
    Extracts spatial keywords from a filename and joins them.
    Includes 'top' only if the exact lowercase 'top' appears.
    """
    keywords = ['left', 'right', 'bottom', 'top', 'upper', 'lower']
    pattern = r'|'.join(keywords)
    matches = re.finditer(pattern, filename, flags=re.IGNORECASE)

    final_words = []
    for match in matches:
        word = match.group(0)
        if word.lower() == 'top':
            if word == 'top':
                final_words.append(word)
        else:
            final_words.append(word)
    return '_'.join([w.capitalize() for w in final_words]) if final_words else "N/A"

# ----------------------------
# Step 1: load + populate
# ----------------------------
def load_and_populate_sessions(baseline_dir, h5_dir, video_dir, *, sanity_check=False):
    sessions = load_all_sessions(
        baseline_dir=str(baseline_dir),
        session_type="oft",
        remove_bad_signal_sessions=True,
    )
    populate_containers(sessions)
    populate_oft_containers(sessions, str(h5_dir), str(video_dir))

    # Always run plotting setup (NOT gated by sanity_check)
    PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)

    if sanity_check:
        print(f"Loaded {len(sessions)} sessions")
    return sessions

# ----------------------------
# Step 2: summary dataframe
# ----------------------------
def build_summary_df(sessions, *, sanity_check=False):
    records = []
    for s in sessions:
        session_id = f"{s.trial_id}/{s.chamber_id}"
        trial_dir = Path(s.trial_dir)

        for key, df in s.dfs.data.items():
            data_type = "unknown"
            details = "N/A"

            if key.startswith('phot_'):
                data_type = "photometry"
                details = "Photometry Data"

            elif key.startswith('tracking_'):
                data_type = "tracking_h5"
                if not df.empty:
                    path_col = df.columns[0]
                    try:
                        filename = Path(df[path_col].iloc[0]).name
                        details = simplify_filename_details(filename)
                    except Exception:
                        details = "Info not available"

            elif key.startswith('video_'):
                if not df.empty:
                    path_col = df.columns[0]
                    try:
                        filename = Path(df[path_col].iloc[0]).name
                        extension = Path(filename).suffix.lstrip('.')
                        data_type = f"video_{extension or 'unknown'}"
                        details = simplify_filename_details(filename)
                    except Exception:
                        data_type = "video_unknown"
                        details = "Info not available"

            elif key.startswith('sync_'):
                data_type = "sync_csv"
                try:
                    cam_name = df['camera_name'].iloc[0]
                    path = next(Path(trial_dir).glob(f'{cam_name}_AllTimestampsCombined*.csv'))
                    details = simplify_filename_details(path.name)
                except Exception:
                    details = "Info not available"
            else:
                continue

            records.append((session_id, data_type, key, details))

    summary_df = pd.DataFrame(
        records,
        columns=['Session ID', 'Data Type', 'Source Key', 'Simplified Details']
    ).sort_values(by=['Session ID', 'Data Type', 'Source Key']).reset_index(drop=True)

    if sanity_check:
        print(summary_df.head(20))
    return summary_df

# ----------------------------
# Step 3: synchronize
# ----------------------------
def synchronize_all_sessions(sessions, *, sanity_check=False,
                             downsample_video_by=1, tolerance=0.02):
    failures = []
    for i, s in enumerate(sessions):
        if sanity_check:
            print(f"Processing {i+1}/{len(sessions)}: {s.trial_id}")
        try:
            synchronize_session_data(s,
                downsample_video_by=downsample_video_by,
                tolerance=tolerance,
            )
            if sanity_check:
                print(f"✅ Synced {s.trial_id}")
        except Exception as e:
            failures.append((s.trial_id, str(e)))
            if sanity_check:
                print(f"❌ ERROR {s.trial_id}: {e}")
    return failures

def verify_synchronization(sessions, *, sanity_check=False):
    all_ok = True
    for i, s in enumerate(sessions):
        sync_keys = [k for k in s.dfs.data if k.startswith('synchronized_')]
        if not sync_keys:
            if sanity_check:
                print(f"❌ Session {i} ({s.trial_id}): no synchronized DataFrames found.")
            all_ok = False
        else:
            if sanity_check:
                print(f"✅ Session {i} ({s.trial_id}): {len(sync_keys)} synchronized DataFrame(s).")

    if sanity_check and sessions:
        s0 = sessions[0]
        sync_keys = sorted([k for k in s0.dfs.data if k.startswith('synchronized_')])
        if sync_keys:
            key = sync_keys[0]
            df0 = s0.dfs.data[key]
            print(f"\nFirst session '{s0.trial_id}': {key} -> rows={len(df0)}, cols={list(df0.columns)}")
    return all_ok

# ----------------------------
# Step 4: KPMS prep + run
# ----------------------------
def prepare_kpms_inputs(sessions, *, sanity_check=False):
    all_h5_paths = []
    for session in sessions:
        session.kpms_h5_path = None
        for df_name, df in session.dfs.data.items():
            if df_name.startswith('tracking_') and 'h5_path' in df.columns and not df['h5_path'].empty:
                h5p = df['h5_path'].iloc[0]
                if pd.notna(h5p) and 'Center' in str(h5p):
                    session.kpms_h5_path = h5p
                    all_h5_paths.append(h5p)
                    break
    if sanity_check:
        print(f"Selected {len(all_h5_paths)} H5 paths for KPMS")
    return all_h5_paths

def generate_models_and_append_results(
    sessions,
    project_dir,
    kappa,
    subset_fraction,
    num_models,
    ar_iters,
    fitting_iters,
    n_jax_batches,
    latent_dim,
    *,
    sanity_check=False,
):
    if sanity_check:
        print("Generating syllable models…")
    model_names, original_lengths, video_frame_indexes, final_metadata = generate_syllable_models(
        sessions=sessions,
        project_dir=project_dir,
        kappa=kappa,
        subset_fraction=subset_fraction,
        num_models=num_models,
        ar_iters=ar_iters,
        fitting_iters=fitting_iters,
        n_jax_batches=n_jax_batches,
        latent_dim=latent_dim,
        test_mode=False,
    )
    if sanity_check:
        print(f"Created {len(model_names)} model(s)")

    for model_name in model_names:
        if sanity_check:
            print(f"📎 Appending results for model: {model_name}")
        extract_and_append_results(
            sessions=sessions,
            project_dir=project_dir,
            model_name=model_name,
            video_frame_indexes=video_frame_indexes,
            original_lengths=original_lengths,
            metadata=final_metadata,
        )
    return model_names, original_lengths, video_frame_indexes, final_metadata

# ----------------------------
# Step 5: verify + visualize
# ----------------------------
def verify_results(sessions, *, sanity_check=False, data_key='all_results'):
    ok = True
    for session in sessions:
        if not hasattr(session, data_key):
            if sanity_check:
                print(f"❌ {session.trial_id}: missing '{data_key}'.")
            ok = False
            continue

        results = getattr(session, data_key)
        if 'syllable' not in results:
            if sanity_check:
                print(f"⚠️ {session.trial_id}: '{data_key}' present but no 'syllable'.")
            ok = False
            continue

        if sanity_check:
            try:
                shape = results['syllable'].shape
            except Exception:
                shape = (len(results['syllable']),)
            print(f"✅ {session.trial_id}: syllable shape {shape}")
    return ok

def plot_data_availability(sessions, *, sanity_check=False):
    plots = 0
    for s in sessions:
        if hasattr(s, 'all_results') and 'syllable' in s.all_results:
            mask = ~np.isnan(s.all_results['syllable'])
            plt.figure(figsize=(18, 4))
            plt.plot(mask, linewidth=1.5)
            plt.title(f"Data Availability: {s.trial_id}")
            plt.xlabel("Frame")
            plt.ylabel("Data Status")
            plt.yticks([0, 1], ['NaN', 'Valid'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(-0.1, 1.1)
            if sanity_check:
                plt.show()
            plots += 1
    if sanity_check:
        print(f"Generated {plots} plots")
    return plots

# ----------------------------
# Step 6: save compact state
# ----------------------------
def save_state(
    out_path,
    sessions,
    model_names,
    video_frame_indexes,
    original_lengths,
    project_dir_full,
    kappa,
    *,
    sanity_check=False,
):
    state_to_save = {
        'sessions': sessions,
        'model_names': model_names,
        'video_frame_indexes': video_frame_indexes,
        'original_lengths': original_lengths,
        'project_dir_full': project_dir_full,
        'kappa': kappa,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    if sanity_check:
        print(f"Saved state to {out_path}")
    return out_path

# ----------------------------
# Orchestrator (single entry)
# ----------------------------
def run_create_session_state(
    baseline_dir: Path,
    h5_dir: Path,
    video_dir: Path,
    *,
    sanity_check: bool = True,
    subset_fraction: float = 1.0,
    latent_dim: int = 9,
    n_jax_batches: int = 24,
    kappa: float = 1.205e6,
    num_models: int = 3,
    ar_iters: int = 1,
    fitting_iters: int = 1,
    sync_downsample: int = 1,
    sync_tolerance: float = 0.02,
    out_pkl: Path = Path("create_session_state_test.pkl"),
):
    # 1) Load + populate
    sessions = load_and_populate_sessions(
        baseline_dir, h5_dir, video_dir, sanity_check=sanity_check
    )

    # 2) Optional summary
    _ = build_summary_df(sessions, sanity_check=sanity_check)

    # 3) Sync
    failures = synchronize_all_sessions(
        sessions,
        sanity_check=sanity_check,
        downsample_video_by=sync_downsample,
        tolerance=sync_tolerance,
    )
    _ = verify_synchronization(sessions, sanity_check=sanity_check)

    # 4) KPMS
    _ = prepare_kpms_inputs(sessions, sanity_check=sanity_check)
    project_dir_full = Path("kpms_projects") / f"kpms_all_sessions_frac_{subset_fraction}_ld{latent_dim}_jn{n_jax_batches}"

    if sanity_check:
        print(f"Project dir: {project_dir_full}")

    model_names, original_lengths, video_frame_indexes, final_metadata = (
        generate_models_and_append_results(
            sessions=sessions,
            project_dir=project_dir_full,
            kappa=kappa,
            subset_fraction=subset_fraction,
            num_models=num_models,
            ar_iters=ar_iters,
            fitting_iters=fitting_iters,
            n_jax_batches=n_jax_batches,
            latent_dim=latent_dim,
            sanity_check=sanity_check,
        )
    )

    # 5) Verify + plot
    _ = verify_results(sessions, sanity_check=sanity_check)
    _ = plot_data_availability(sessions, sanity_check=sanity_check)

    # 6) Save
    out_filename = Path(out_pkl).name if out_pkl else f"state_ld{latent_dim}_jn{n_jax_batches}_fit{fitting_iters}.pkl"
    save_path = project_dir_full / out_filename

    saved = save_state(
        out_path=save_path,
        sessions=sessions,
        model_names=model_names,
        video_frame_indexes=video_frame_indexes,
        original_lengths=original_lengths,
        project_dir_full=project_dir_full,
        kappa=kappa,
        sanity_check=sanity_check,
    )

    return {
        "saved": str(saved),
        "n_sessions": len(sessions),
        "models": model_names,
        "sync_failures": failures,
        "project_dir": project_dir_full,
    }

# --------------------------------------------
# Example call (leave commented if not needed)
# --------------------------------------------
result = run_create_session_state(
    baseline_dir=Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/Jonathan_IFxDN_csvs"),
    h5_dir=Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/h5_files"),
    video_dir=Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/video_files"),
)
print(result)
