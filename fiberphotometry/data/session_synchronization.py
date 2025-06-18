# fiberphotometry/data/session_synchronization.py

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

def synchronize_session_data(session, downsample_video_by: int = 2, tolerance: float = 0.02):
    """
    Creates clean, synchronized DataFrames for ALL tracking/sync pairs in a session.

    This function iterates through each 'tracking_n' file, finds its corresponding
    sync file by matching camera names, and generates a dedicated synchronized
    DataFrame for each pair (e.g., 'synchronized_data_tracking_0', 
    'synchronized_data_tracking_1', etc.).

    Args:
        session: The session object to process.
        downsample_video_by (int): Factor used ONLY to calculate the 'curr_video_frame' column.
        tolerance (float): The maximum allowed gap in seconds for matching timestamps.
    """
    print(f"--- Synchronizing all pairs for session: {session.trial_id}/{session.chamber_id} ---")

    # --- THIS IS THE KEY CHANGE: We now loop over all tracking keys ---
    tracking_keys = sorted([k for k in session.dfs.data if k.startswith('tracking_')])

    if not tracking_keys:
        warnings.warn("No tracking data found in session to synchronize.")
        return

    # The main loop now iterates through each available tracking file
    for t_key in tracking_keys:
        # Load a fresh copy of photometry data for each pair to prevent modification issues
        try:
            phot_470_df = session.dfs.data['phot_470'].copy()
            phot_415_df = session.dfs.data['phot_415'].copy()
        except KeyError as e:
            warnings.warn(f"Cannot start sync for {t_key}: missing photometry data key {e}.")
            continue

        # Find the camera name and H5 path for the current tracking file
        tracking_info_df = session.dfs.data[t_key]
        if 'camera_name' not in tracking_info_df.columns:
            warnings.warn(f"Cannot process {t_key}: 'camera_name' column is missing.")
            continue
        
        tracking_cam_name = tracking_info_df['camera_name'].iloc[0]
        h5_path = tracking_info_df['h5_path'].iloc[0]
        print(f"\nProcessing pair for camera: '{tracking_cam_name}' (from {t_key})")

        # Find the corresponding sync file by matching the camera name
        matching_sync_df = None
        sync_keys = [k for k in session.dfs.data if k.startswith('sync_')]
        for s_key in sync_keys:
            sync_info_df = session.dfs.data[s_key]
            if sync_info_df['camera_name'].iloc[0] == tracking_cam_name:
                matching_sync_df = sync_info_df.copy()
                break
        
        if matching_sync_df is None:
            warnings.warn(f"Could not find a matching sync file for {t_key}.")
            continue

        # 1. ESTABLISH THE FULL-RESOLUTION MASTER TIMELINE from the matched sync file
        master_timeline_df = matching_sync_df.copy()
        print(f"Master timeline established with {len(master_timeline_df)} rows.")

        # 2. MAP PHOTOMETRY DATA ONTO THE MASTER TIMELINE
        master_timeline_df.sort_values('FP3002_System_Timestamp', inplace=True)
        phot_470_df.sort_values('SystemTimestamp', inplace=True)
        phot_415_df.sort_values('SystemTimestamp', inplace=True)

        phot_470_df.rename(columns={'SystemTimestamp': 'SystemTimestamp_470', 'signal_0': 'signal_0_470'}, inplace=True)
        phot_415_df.rename(columns={'SystemTimestamp': 'SystemTimestamp_415', 'signal_0': 'signal_0_415'}, inplace=True)
        
        phot_470_subset = phot_470_df[['SystemTimestamp_470', 'signal_0_470']]
        phot_415_subset = phot_415_df[['SystemTimestamp_415', 'signal_0_415']]

        merged_df = pd.merge_asof(
            left=master_timeline_df, right=phot_470_subset,
            left_on='FP3002_System_Timestamp', right_on='SystemTimestamp_470',
            direction='nearest', tolerance=tolerance
        )
        final_merged_df = pd.merge_asof(
            left=merged_df, right=phot_415_subset,
            left_on='FP3002_System_Timestamp', right_on='SystemTimestamp_415',
            direction='nearest', tolerance=tolerance
        )
        
        # 3. CLEAN UP AND ADD HELPER COLUMNS
        essential_cols = [
            'FP3002_System_Timestamp', 'signal_0_470', 'signal_0_415'
        ]
        final_df = final_merged_df[[col for col in essential_cols if col in final_merged_df.columns]].copy()
        
        n_rows = len(final_df)
        if downsample_video_by >= 1:
            final_df['curr_video_frame'] = (np.floor(np.arange(n_rows) / downsample_video_by)).astype(int)
        
        final_df['current_h5_row'] = np.arange(n_rows)
        
        # 4. SAVE THE FINAL CLEANED DATAFRAME
        new_key = f'synchronized_data_{t_key}' 
        session.dfs.add_data(new_key, final_df)
        print(f"✅ Successfully created '{new_key}' with {len(final_df)} rows and {len(final_df.columns)} columns.")
    
    print("\n--- Synchronization complete for all pairs ---")