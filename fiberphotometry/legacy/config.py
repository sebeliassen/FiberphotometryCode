# config.py

# how to pick off your TTL columns
RENAME_RULES = {
    'ttl': [
        (['FP3002.*Value'],                  'DigitalIOState'),
        (['FP3002.*Seconds', 'FP3002_Timestamp'], 'SystemTimestamp'),
    ]
}

# the canonical photometry schema
STANDARD_BASE_COLS = [
    'FrameCounter',
    'SystemTimestamp',
    'LedState',
    'ComputerTimestamp',  # may not always be present
]

# match legacy Region<N><G/R>
REGION_COL_RE = r'^Region(\d+)([GR])$'

# glob pattern for raw photometry fragments
PHOTOWRIT_GLOB = 'channel*photwrit*.csv'

# Regex to pull out an optional “Setup<id>” and then the timestamp.
#  - Group “setup_id” will be the letter(s) after “_Setup” (if present)
#  - Group “timestamp” is your ISO‐style TS
PHOTOWRIT_SETUP_RE = (
    r'channel\d+photwrit'                 # prefix
    r'(?:_Setup(?P<setup_id>[^_]+))?'     # optional “_SetupX”
    r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})'  # the TS
)

# regex to catch RAW_{chamber}_*.csv files (case-insensitive, chamber A–D)
RAW_CSV_RE = r'(?i)^raw_([A-D])_.+\.csv$'