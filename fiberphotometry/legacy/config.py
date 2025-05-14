# config.py

# how to pick off your TTL columns
RENAME_RULES = {
    'ttl': [
        (['FP3002.*Seconds', 'FP3002_Timestamp'], 'SystemTimestamp'),
        (['FP3002.*Value'],                  'DigitalIOState'),
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

# glob to find your per-channel photometry files
PHOTOWRIT_GLOB = 'channel*photwrit*.csv'
