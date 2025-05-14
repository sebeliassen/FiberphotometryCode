#!/usr/bin/env python3
import os, glob, re, argparse
import pandas as pd
from config import (
    RENAME_RULES,
    STANDARD_BASE_COLS,
    REGION_COL_RE,
    PHOTOWRIT_GLOB,
)

def rename_photometry(df: pd.DataFrame) -> pd.DataFrame:
    # 1) rename any “Timestamp” → “SystemTimestamp”
    if 'Timestamp' in df.columns and 'SystemTimestamp' not in df.columns:
        df = df.rename(columns={'Timestamp': 'SystemTimestamp'})

    # 2) normalize new‐style (e.g. “G0” / “R1”) → old‐style “Region0G” etc.
    def _norm(col):
        m = re.fullmatch(r'([GR])(\d+)', col)
        return f"Region{m.group(2)}{m.group(1)}" if m else col

    df = df.rename(columns=_norm)

    # 3) build a map Region<N><G/R> → G/R<N>
    region_map = {}
    for col in df.columns:
        m = re.fullmatch(REGION_COL_RE, col)
        if m:
            idx, chan = m.groups()
            region_map[col] = f"{chan}{idx}"

    # 4) apply it
    df = df.rename(columns=region_map)

    # 5) select only the standard base cols + whatever region_map produced
    keep = [c for c in STANDARD_BASE_COLS if c in df.columns]
    keep += [v for v in region_map.values() if v in df.columns]
    return df.loc[:, keep]

def combine_photometry(folder: str, head: int = None) -> pd.DataFrame:
    # if they already gave us a combined file, just use that
    combined_files = glob.glob(os.path.join(folder, 'photometry_data_combined*.csv'))
    if combined_files:
        path = combined_files[0]
        df = pd.read_csv(path, nrows=head) if head else pd.read_csv(path)
        return rename_photometry(df)

    # otherwise glob all channel*photwrit*.csv
    files = sorted(glob.glob(os.path.join(folder, PHOTOWRIT_GLOB)))
    if not files:
        raise RuntimeError(f"No photometry files in {folder}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, nrows=head) if head else pd.read_csv(f)
        dfs.append(df)

    # concatenate, drop duplicate FrameCounters, sort
    big = pd.concat(dfs, ignore_index=True)
    big = big.drop_duplicates(subset='FrameCounter').sort_values('FrameCounter')
    return rename_photometry(big)

def rename_ttl(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rules = RENAME_RULES['ttl']
    mapping = {}
    for col in df.columns:
        for pats, new in rules:
            if any(re.match(p, col) for p in pats):
                mapping[col] = new
                break
    return df.rename(columns=mapping)

def process_trial_folder(input_folder: str, output_folder: str, head: int = None):
    os.makedirs(output_folder, exist_ok=True)

    # 1) TTLs
    for ttl_path in sorted(glob.glob(os.path.join(input_folder, 'DigInput_*.csv'))):
        df_ttl = rename_ttl(ttl_path)
        if head:
            df_ttl = df_ttl.head(head)
        out = os.path.join(output_folder, os.path.basename(ttl_path))
        df_ttl.to_csv(out, index=False)
        print("wrote", out)

    # 2) photometry
    try:
        df_photo = combine_photometry(input_folder, head=head)
    except RuntimeError:
        return

    # write out as a single CSV
    name = os.path.basename(input_folder.rstrip(os.sep)) + '_photometry.csv'
    out = os.path.join(output_folder, name)
    df_photo.to_csv(out, index=False)
    print("wrote", out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input_folder')
    p.add_argument('output_folder')
    p.add_argument('-n', '--head', type=int,
                   help='keep only first N rows of each CSV')
    args = p.parse_args()
    process_trial_folder(args.input_folder, args.output_folder, head=args.head)
