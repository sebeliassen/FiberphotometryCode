import pandas as pd
import numpy as np
import warnings
import re
from pathlib import Path

def synchronize_session_data(session, downsample_video_by: int = 1, tolerance: float = 0.02):
    print(f"--- Synchronizing all pairs for session: {session.trial_id}/{session.chamber_id} ---")

    tracking_keys = sorted(k for k in session.dfs.data if k.startswith('tracking_'))
    if not tracking_keys:
        warnings.warn("No tracking data found in session to synchronize.")
        return

    # helper to rename all signal_<n> columns to include the freq suffix
    def _add_freq_suffix(df, freq):
        mapper = {}
        for col in df.columns:
            m = re.match(r'^(signal_\d+)(.*)$', col)
            if m:
                stem, rest = m.groups()
                mapper[col] = f"{stem}_{freq}{rest}"
        return df.rename(columns=mapper)

    for t_key in tracking_keys:
        try:
            phot_470_df = session.dfs.data['phot_470'].copy()
            phot_415_df = session.dfs.data['phot_415'].copy()
        except KeyError as e:
            warnings.warn(f"Cannot start sync for {t_key}: missing photometry data key {e}.")
            continue

        # rename *all* signal_<n>... columns
        phot_470_df = _add_freq_suffix(phot_470_df, '470')
        phot_415_df = _add_freq_suffix(phot_415_df, '415')

        # now SystemTimestamp → SystemTimestamp_<freq>
        phot_470_df = phot_470_df.rename(columns={'SystemTimestamp': 'SystemTimestamp_470'})
        phot_415_df = phot_415_df.rename(columns={'SystemTimestamp': 'SystemTimestamp_415'})

        # subset both timestamp+any signal_*_470* columns
        sig470 = phot_470_df.filter(regex=r'^(SystemTimestamp_470|signal_\d+_470)')
        sig415 = phot_415_df.filter(regex=r'^(SystemTimestamp_415|signal_\d+_415)')

        # proceed as before
        master_timeline_df = session.dfs.data[
            next(s for s in session.dfs.data if s.startswith('sync_') and
                 session.dfs.data[s]['camera_name'].iloc[0] ==
                 session.dfs.data[t_key]['camera_name'].iloc[0])
        ].copy()
        master_timeline_df.sort_values('FP3002_System_Timestamp', inplace=True)

        sig470 = sig470.sort_values('SystemTimestamp_470')
        sig415 = sig415.sort_values('SystemTimestamp_415')

        merged = pd.merge_asof(
            master_timeline_df, sig470,
            left_on='FP3002_System_Timestamp', right_on='SystemTimestamp_470',
            direction='nearest', tolerance=tolerance
        )
        merged = pd.merge_asof(
            merged, sig415,
            left_on='FP3002_System_Timestamp', right_on='SystemTimestamp_415',
            direction='nearest', tolerance=tolerance
        )

        # now pick timestamp + *all* signal_* columns in the merged df
        cols = ['FP3002_System_Timestamp'] + [
            c for c in merged.columns if c.startswith('signal_')
        ]
        final_df = merged[cols].copy()

        # helper columns
        n = len(final_df)
        if downsample_video_by >= 1:
            final_df['curr_video_frame'] = (np.floor(np.arange(n) / downsample_video_by)).astype(int)
        final_df['current_h5_row'] = np.arange(n)

        new_key = f'synchronized_data_{t_key}'
        session.dfs.add_data(new_key, final_df)
        print(f"✅ Created '{new_key}' with {len(final_df)} rows × {len(final_df.columns)} cols")

    print("\n--- Synchronization complete ---")
