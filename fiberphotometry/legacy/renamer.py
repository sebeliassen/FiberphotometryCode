#!/usr/bin/env python3
import os
import glob
import re
import argparse
import pandas as pd

from config import (
    RENAME_RULES,
    PHOTOWRIT_GLOB,
    PHOTOWRIT_SETUP_RE,
    REGION_COL_RE,
)

# ----------------------------------------------------------------------
def rename_ttl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename any TTL columns according to RENAME_RULES['ttl'].
    Returns a new DataFrame; does not write or print.
    """
    mapping = {}
    for col in df.columns:
        for patterns, std_name in RENAME_RULES['ttl']:
            if any(re.match(p, col) for p in patterns):
                mapping[col] = std_name
                break
    return df.rename(columns=mapping)


# ----------------------------------------------------------------------
def rename_photometry(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) If there’s a “Timestamp” but no “SystemTimestamp”, rename it.
    2) Canonicalize Region* / G*/R* → G#/R#.
    3) Keep only standard cols + any of our region channels.
    Returns a new DataFrame; does not write or print.
    """
    # 1) Timestamp → SystemTimestamp
    if 'Timestamp' in df.columns and 'SystemTimestamp' not in df.columns:
        df = df.rename(columns={'Timestamp': 'SystemTimestamp'})

    # 2) unify Region<> / G#/R#
    def _unify(col: str) -> str:
        # a) new‐style already in G<N>/R<N> → leave it
        if re.fullmatch(r'[GR]\d+', col):
            return col
        # b) old‐style Region<N><G/R> → G/R + idx
        m = re.fullmatch(REGION_COL_RE, col)
        if m:
            idx, chan = m.groups()
            return f"{chan}{idx}"
        # c) leave everything else untouched
        return col

    df = df.rename(columns=_unify)

    # 3) pick standard + region columns
    base = ['FrameCounter', 'SystemTimestamp', 'LedState', 'ComputerTimestamp']
    regions = sorted(c for c in df.columns if re.fullmatch(r'[GR]\d+', c))
    keep = [c for c in base if c in df.columns] + regions
    return df.loc[:, keep]


# ----------------------------------------------------------------------
_setup_ts_re = re.compile(PHOTOWRIT_SETUP_RE)

def combine_photometry(
    input_folder: str,
    head: int = None
) -> list[tuple[str, pd.DataFrame]]:
    """
    Looks in `input_folder` for either:
      • existing photometry_data_combined*.csv’s, or
      • raw channel*photwrit*.csv fragments.

    Returns a list of (out_filename, df) to be written by caller.
    """
    outputs: list[tuple[str,pd.DataFrame]] = []

    # 1) pick up any pre-combined file(s)
    existing = sorted(glob.glob(os.path.join(input_folder,
                                              'photometry_data_combined*.csv')))
    if existing:
        # warn if more than one, but keep logic outside
        chosen = existing[0]
        df = pd.read_csv(chosen, nrows=head) if head else pd.read_csv(chosen)
        norm = rename_photometry(df)

        # ensure it has a Setup suffix
        basename = os.path.basename(chosen)
        if '_Setup' not in basename:
            ts_match = re.search(r'combined_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', basename)
            ts = ts_match.group(1) if ts_match else 'UNKNOWN'
            new_name = f"photometry_data_combined_SetupA_{ts}.csv"
        else:
            new_name = basename

        outputs.append((new_name, norm))
        return outputs

    # 2) bucket raw fragments by (setup, timestamp)
    files = sorted(glob.glob(os.path.join(input_folder, PHOTOWRIT_GLOB)))
    if not files:
        raise RuntimeError(f"No photometry files in {input_folder!r}")

    by_setup: dict[tuple[str,str], list[str]] = {}
    for f in files:
        fn = os.path.basename(f)
        m = _setup_ts_re.search(fn)
        if m:
            setup = m.group('setup_id') or 'A'
            ts    = m.group('timestamp')
        else:
            # legacy: no Setup→ default 'A' + grab timestamp
            setup = 'A'
            tsm = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', fn)
            ts = tsm.group(1) if tsm else 'UNKNOWN'
        by_setup.setdefault((setup, ts), []).append(f)

    # 3) for each bucket, concat/drop-dupes/sort/rename
    for (setup, ts), flist in by_setup.items():
        dfs = []
        for f in flist:
            df = pd.read_csv(f, nrows=head) if head else pd.read_csv(f)
            if len(df) >= 10:      # skip tiny fragments
                dfs.append(df)

        if not dfs:
            continue

        big = pd.concat(dfs, ignore_index=True)
        big = big.drop_duplicates(subset='FrameCounter')
        big = big.sort_values('FrameCounter')
        norm = rename_photometry(big)

        out_name = f"photometry_data_combined_Setup{setup}_{ts}.csv"
        outputs.append((out_name, norm))

    if not outputs:
        raise RuntimeError(f"No valid photometry fragments in {input_folder!r}")

    return outputs


# ----------------------------------------------------------------------
def process_trial_folder(
    input_folder: str,
    output_folder: str,
    head: int = None
):
    os.makedirs(output_folder, exist_ok=True)

    # 1) TTL files
    for ttl_path in sorted(glob.glob(os.path.join(input_folder, 'DigInput_*.csv'))):
        df_ttl = pd.read_csv(ttl_path)
        df_ttl = rename_ttl(df_ttl)
        if head:
            df_ttl = df_ttl.head(head)

        out_name = os.path.basename(ttl_path)
        out_path = os.path.join(output_folder, out_name)
        df_ttl.to_csv(out_path, index=False)
        print("wrote", out_path)

    # 2) Photometry files (may be 0, 1 or many per-setup)
    try:
        phot_outputs = combine_photometry(input_folder, head=head)
    except RuntimeError:
        return

    for name, df in phot_outputs:
        out_path = os.path.join(output_folder, name)
        df.to_csv(out_path, index=False)
        print("wrote", out_path)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input_folder')
    p.add_argument('output_folder')
    p.add_argument('-n','--head', type=int,
                   help='keep only first N rows of each CSV')
    args = p.parse_args()

    process_trial_folder(args.input_folder,
                         args.output_folder,
                         head=args.head)
