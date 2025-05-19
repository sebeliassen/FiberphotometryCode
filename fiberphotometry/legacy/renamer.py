#!/usr/bin/env python3
import os, glob, re, argparse, shutil
import pandas as pd

from config import RENAME_RULES, REGION_COL_RE, PHOTOWRIT_GLOB, PHOTOWRIT_SETUP_RE, RAW_CSV_RE

# ----------------------------------------------------------------------------
def rename_ttl(path: str) -> pd.DataFrame:
    """Read CSV at `path`, apply TTL‐rename rules, return new DataFrame."""
    df = pd.read_csv(path)
    mapping = {}
    for col in df.columns:
        for patterns, std_name in RENAME_RULES['ttl']:
            if any(re.match(p, col) for p in patterns):
                mapping[col] = std_name
                break
    return df.rename(columns=mapping)


# ----------------------------------------------------------------------------
def rename_photometry(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps + Region*/G*/R* → G#/R# → pick only standard + region cols."""
    if 'Timestamp' in df.columns and 'SystemTimestamp' not in df.columns:
        df = df.rename(columns={'Timestamp': 'SystemTimestamp'})

    def _unify(col: str) -> str:
        if re.fullmatch(r'[GR]\d+', col):
            return col
        m = re.fullmatch(REGION_COL_RE, col)
        if m:
            idx, chan = m.groups()
            return f"{chan}{idx}"
        return col

    df = df.rename(columns=_unify)
    base    = ['FrameCounter','SystemTimestamp','LedState','ComputerTimestamp']
    regions = sorted(c for c in df.columns if re.fullmatch(r'[GR]\d+', c))
    keep    = [c for c in base if c in df.columns] + regions
    return df.loc[:, keep]


# ----------------------------------------------------------------------------
_setup_ts_re = re.compile(PHOTOWRIT_SETUP_RE)

def combine_photometry(input_folder: str, head: int = None) -> list[tuple[str, pd.DataFrame]]:
    """
    1) If existing photometry_data_combined*.csv → normalize & return [(name,df)]
    2) else glob raw fragments, bucket by setup/timestamp,
       concat-drop-dupes-sort, normalize, return list of (name,df).
    """
    outputs = []

    # 1) pre‐combined?
    existing = sorted(glob.glob(os.path.join(input_folder,
                                              'photometry_data_combined*.csv')))
    if existing:
        chosen = existing[0]
        df     = pd.read_csv(chosen, nrows=head) if head else pd.read_csv(chosen)
        norm   = rename_photometry(df)

        bn = os.path.basename(chosen)
        if '_Setup' not in bn:
            ts_m = re.search(r'combined_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', bn)
            ts   = ts_m.group(1) if ts_m else 'UNKNOWN'
            out_name = f"photometry_data_combined_SetupA_{ts}.csv"
        else:
            out_name = bn

        outputs.append((out_name, norm))
        return outputs

    # 2) raw fragments
    files = sorted(glob.glob(os.path.join(input_folder, PHOTOWRIT_GLOB)))
    if not files:
        raise RuntimeError(f"No photometry files in {input_folder!r}")

    by_setup = {}
    for f in files:
        fn = os.path.basename(f)
        m  = _setup_ts_re.search(fn)
        if m:
            setup = m.group('setup_id') or 'A'
            ts    = m.group('timestamp')
        else:
            setup = 'A'
            ts_m  = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', fn)
            ts    = ts_m.group(1) if ts_m else 'UNKNOWN'
        by_setup.setdefault((setup, ts), []).append(f)

    for (setup, ts), flist in by_setup.items():
        dfs = []
        for f in flist:
            df = pd.read_csv(f, nrows=head) if head else pd.read_csv(f)
            if len(df) >= 10:
                dfs.append(df)
        if not dfs:
            continue

        big    = pd.concat(dfs, ignore_index=True)
        big    = big.drop_duplicates(subset='FrameCounter')
        big    = big.sort_values('FrameCounter')
        norm   = rename_photometry(big)
        out_nm = f"photometry_data_combined_Setup{setup}_{ts}.csv"
        outputs.append((out_nm, norm))

    if not outputs:
        raise RuntimeError(f"No valid photometry fragments in {input_folder!r}")
    return outputs


# ----------------------------------------------------------------------------
def process_trial_folder(input_folder: str, output_folder: str, head: int = None):
    os.makedirs(output_folder, exist_ok=True)

    # 0) copy RAW_{chamber}_*.csv → RAW_{chamber}.csv
    for path in sorted(glob.glob(os.path.join(input_folder, '*.csv'))):
        fn = os.path.basename(path)
        m  = re.match(RAW_CSV_RE, fn)
        if m:
            chamber = m.group(1).upper()
            dest    = os.path.join(output_folder, f'RAW_{chamber}.csv')
            shutil.copy(path, dest)
            print(f"wrote {dest}")

    # 0.5) copy trial‐guide files (T<digits>_trial_guide.csv or .xlsx)
    for ext in ('csv', 'xlsx'):
        for path in glob.glob(os.path.join(input_folder, 'T[0-9]*_trial_guide.' + ext)):
            fn   = os.path.basename(path)
            dest = os.path.join(output_folder, fn)
            shutil.copy(path, dest)
            print(f"wrote {dest}")

    # 1) TTLs
    for ttl_path in sorted(glob.glob(os.path.join(input_folder, 'DigInput_*.csv'))):
        df_ttl = rename_ttl(ttl_path)
        if head:
            df_ttl = df_ttl.head(head)
        out = os.path.join(output_folder, os.path.basename(ttl_path))
        df_ttl.to_csv(out, index=False)
        print(f"wrote {out}")

    # 2) Photometry
    try:
        photolist = combine_photometry(input_folder, head=head)
    except RuntimeError:
        return

    for fname, df in photolist:
        out = os.path.join(output_folder, fname)
        df.to_csv(out, index=False)
        print(f"wrote {out}")


# ----------------------------------------------------------------------------
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
