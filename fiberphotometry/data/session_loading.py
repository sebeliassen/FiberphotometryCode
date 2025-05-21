from pathlib import Path
from io import StringIO
import re
import pandas as pd

from fiberphotometry.data.data_loading import DataContainer
from fiberphotometry.config import DATA_PATTERNS, COMBINED_SPLIT

FIBER_RE = re.compile(r'^([A-Za-z])(\d+)$')

def _process_raw(fpath: Path, kwargs: dict):
    """
    Parse the raw file into header attributes and a DataFrame.
    Returns (raw_attrs: dict, df: DataFrame).
    """
    lines = fpath.read_text().splitlines()
    # find separator
    idx = next((i for i, L in enumerate(lines) if L.strip().startswith('---')), None)
    if idx is None:
        idx = -1

    # header attributes
    raw_attrs = {}
    for L in lines[:idx]:
        if ',' in L:
            k, v = L.split(',', 1)
            raw_attrs[k.strip()] = v.strip()

    # csv data block
    data_block = "\n".join(lines[idx+1:])
    df = pd.read_csv(StringIO(data_block), **kwargs)
    return raw_attrs, df


def _process_phot(fpath: Path, kwargs: dict, session):
    # discover columns
    all_cols = pd.read_csv(fpath, nrows=0, **kwargs).columns

    keep_cols = [
        c for c in all_cols
        if not (m := FIBER_RE.match(c))
        or m.group(2) in session.fiber_to_region
    ]

    # now read only those cols
    df_all = pd.read_csv(fpath, usecols=keep_cols, **kwargs)

    # … rest of your splitting + renaming logic unchanged …
    split_map = COMBINED_SPLIT[session.session_type]
    phot_dfs, phot_meta = {}, {}

    for led_state, freq_label in split_map.items():
        sub = df_all[df_all["LedState"] == led_state].copy().reset_index(drop=True)

        fiber_cols = [
            (c, m)
            for c in sub.columns
            if (m := FIBER_RE.match(c))
        ]

        rename_map = {}
        for i, (orig_col, m) in enumerate(fiber_cols):
            num = m.group(2)
            region, side, color = session.fiber_to_region[num]
            new_name = f"signal_{i}"
            rename_map[orig_col] = new_name
            phot_meta[new_name] = (region, side, color, orig_col)

        if rename_map:
            sub = sub.rename(columns=rename_map)

        phot_dfs[f"phot_{freq_label}"] = sub

    return phot_dfs, phot_meta


def populate_session(session):
    """
    Load raw, ttl, and split photometry for a single Session.
    """
    session.dfs = DataContainer()
    specs = DATA_PATTERNS[session.session_type]

    for name, spec in specs.items():
        patt = spec['pattern'].format(
            chamber_id=session.chamber_id,
            setup_id=session.setup_id
        )
        files = list(Path(session.trial_dir).glob(patt))
        if not files:
            continue

        fpath = files[0]
        kwargs = spec.get('kwargs', {})

        if name == 'raw':
            raw_attrs, df = _process_raw(fpath, kwargs)
            session.raw_attributes = raw_attrs
            session.dfs.add_data('raw', df)

        elif name == 'phot':
            phot_dfs, phot_meta = _process_phot(fpath, kwargs, session)
            session.signal_meta = phot_meta
            for key, df in phot_dfs.items():
                session.dfs.add_data(key, df)

        else:
            df = pd.read_csv(fpath, **kwargs)
            session.dfs.add_data(name, df)


def populate_containers(sessions):
    """Apply populate_session to every Session in the list."""
    for session in sessions:
        populate_session(session)