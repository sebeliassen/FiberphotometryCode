import pandas as pd
from fiberphotometry import config


def merge_phot_bonsai_cam(phot_df, bonsai_df, freq_label, cam_df):
    # Infer the TimestampBonsai column directly from bonsai_df
    phot_df[f'TimestampBonsai_{freq_label}'] = bonsai_df[f'TimestampBonsai_{freq_label}']
    
    # Merge with the cam dataframe to add BonsaiTrackingTimestamp and cam_frame_num
    merged_phot_final = pd.merge_asof(
        left=phot_df,
        right=cam_df[['cam_frame_num', 'BonsaiTrackingTimestamp']],
        left_on=f'TimestampBonsai_{freq_label}',  # Use the inferred TimestampBonsai column
        right_on='BonsaiTrackingTimestamp',
        direction='nearest'
    )
    
    # Drop the TimestampBonsai column after merging
    merged_phot_final = merged_phot_final.drop(columns=[f'TimestampBonsai_{freq_label}'])
    
    return merged_phot_final


def update_phot_dfs_with_timestamps(session):
    # Prepare the cam dataframe by resetting the index for cam_frame_num
    cam = session.dfs.data.get('cam')
    if cam is None:
        raise ValueError("Camera data ('cam') is missing from the session.")
        
    cam_df = cam.reset_index().rename(columns={'index': 'cam_frame_num'})
    
    # List to track auxiliary dataframes for removal
    aux_keys_to_remove = ['cam']
    
    # Loop through config.PHOT_DF_PATTERNS to process each photometry frequency
    for phot_key in config.PHOT_DF_PATTERNS.keys():
        # Extract the frequency label (e.g., '415' from 'phot_415')
        freq_label = phot_key.split('_')[-1]
        
        # Get corresponding photometry and bonsai dataframes
        phot_df = session.dfs.data.get(phot_key)
        bonsai_key = f'bonsai_{freq_label}'
        bonsai_df = session.dfs.data.get(bonsai_key)
        
        if phot_df is not None and bonsai_df is not None:
            # Perform the merging and update session data in-place
            session.dfs.data[phot_key] = merge_phot_bonsai_cam(phot_df, bonsai_df, freq_label, cam_df)
            # Track bonsai dataframe for removal
            aux_keys_to_remove.append(bonsai_key)
    
    # Remove auxiliary dataframes
    for key in aux_keys_to_remove:
        session.dfs.data.pop(key, None)


def add_phot_timestamps_phot_df(sessions):
    for session in sessions:
        update_phot_dfs_with_timestamps(session)