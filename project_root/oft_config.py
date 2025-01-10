# config.py

# Plotting configuration parameters
PLOTTING_CONFIG = {
    'baseline_duration': 20,  # in minutes
    'trial_length': 30,       # in minutes
    'fps': 20,                # frames per second
    'fit_window_start': 11,   # in minutes before trial start
    'fit_window_end': 1       # in minutes before trial start
}

RENAME_PATTERNS = [
    ('bonsai', {"pattern": "(CombiTimestamps)?_\\d+_?[AB].?", "replacement": ""}),
    ('bonsai', {"pattern": "FP3002_Timestamp", "replacement": "Timestamp_FP3002"}),
    ('bonsai', {"pattern": "TimestampBonsai", "replacement": "Timestamp_Bonsai"})
]
LETTER_TO_FREQS = {'iso': '415', 'G': '470', 'R': '560'}

all_brain_regions = ['VS', 'DLS']

# self.dfs.add_data('phot', phot_df)
#         self.dfs.add_data('tracking', tracking_df)
# phot_df = self.load_data('photometry_data*.csv', use_cols=self.filter_columns)
#         tracking_df = self.load_data('CenterTopCam_TrackData*.csv')

# PHOT_DF_PATTERNS = {
#     'phot': 'photometry_data*.csv',
# }

# tracking_df_pattern = 'CenterTopCam_TrackData*.csv'

PHOT_DF_PATTERNS = {
    'phot_415': 'channel415*.csv',
    'phot_470': 'channel470*.csv',
}

TIMESTAMP_DF_PATTERNS = {
    'tracking': 'TopCamTracking*.csv',
    'bonsai_415': 'c415_bonsaiTS*.csv',
    'bonsai_470': 'c470_bonsaiTS*.csv'
}

