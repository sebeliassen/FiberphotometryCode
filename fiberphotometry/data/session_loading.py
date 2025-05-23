from pathlib import Path
from io import StringIO
import re
from typing import TextIO, Union
import warnings
import pandas as pd

from fiberphotometry.data.data_loading import DataContainer
from fiberphotometry.config import DATA_PATTERNS, COMBINED_SPLIT

FIBER_RE = re.compile(r'^([A-Za-z])(\d+)$')

def _safe_read_csv(fsrc: Union[Path, TextIO], **kwargs) -> pd.DataFrame:
    fname = getattr(fsrc, 'name', repr(fsrc))
    try:
        return pd.read_csv(fsrc, **kwargs)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{fname} is empty.")
    except pd.errors.ParserError as pe:
        raise ValueError(f"Could not parse {fname}: {pe}")

def _process_raw(fpath: Path, kwargs: dict):
    """
    Parse the raw file into header attributes and a DataFrame.
    Returns (raw_attrs: dict, df: DataFrame).
    """

    try:
        text = fpath.read_text()
    except Exception as e:
        raise IOError(f"Could not read raw file {fpath!r}: {e}")
    
    lines = text.splitlines()
    # find separator
    idx = next((i for i, L in enumerate(lines) if L.strip().startswith('---')), None)
    if idx is None:
        warnings.warn(f"No header separator (`---`) found in {fpath.name}; treating entire file as data.")
        idx = -1

    # parsing header
    raw_attrs = {}
    for L in lines[:idx]:
        if ',' not in L:
            warnings.warn(f"Unrecognized header line in {fpath.name}: {L!r}")
            continue
        k, v = L.split(',', 1)
        raw_attrs[k.strip()] = v.strip()

    # csv data block
    data_block = "\n".join(lines[idx+1:])
    df = _safe_read_csv(StringIO(data_block), **kwargs)
    
    return raw_attrs, df


def _process_phot(fpath: Path, kwargs: dict, session):
    # discover columns
    all_cols = _safe_read_csv(fpath, nrows=0, **kwargs).columns
    if "LedState" not in all_cols:
        raise KeyError(f"{fpath.name} missing required column 'LedState'. Found: {list(all_cols)}")

    keep_cols = [
        c for c in all_cols
        if not (m := FIBER_RE.match(c))
        or m.group(2) in session.fiber_to_region
    ]

    # now read only those cols
    #df_all = pd.read_csv(fpath, usecols=keep_cols, **kwargs)
    df_all = _safe_read_csv(fpath, usecols=keep_cols, **kwargs)

    # … rest of your splitting + renaming logic unchanged …
    split_map = COMBINED_SPLIT.get(session.session_type)
    if split_map is None:
        raise KeyError(
            f"No COMBINED_SPLIT entry for session_type {session.session_type!r}; "
            f"available types: {list(COMBINED_SPLIT.keys())}"
        )
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

    if not phot_dfs:
        warnings.warn(f"No photometry streams extracted from {fpath.name} for session_type {session.session_type!r}")

    return phot_dfs, phot_meta


def populate_session(session):
    """
    Load raw, ttl, and split photometry for a single Session.
    """
    session.dfs = DataContainer()
    specs = DATA_PATTERNS[session.session_type]

    def _add_data(name: str, df: pd.DataFrame):
        try:
            session.dfs.add_data(name, df)
        except TypeError as te:
            raise TypeError(
                f"While adding data '{name}' for session "
                f"{session.trial_id}/{session.chamber_id}: {te}"
            )

    for name, spec in specs.items():
        patt = spec['pattern'].format(
            chamber_id=session.chamber_id,
            setup_id=session.setup_id
        )
        files = list(Path(session.trial_dir).glob(patt))
        if not files:
            warnings.warn(
                f"No files matching pattern {patt!r} in {session.trial_dir!r}; skipping '{name}' data."
            )
            continue

        if len(files) > 1:
            warnings.warn(
                f"Multiple files matched pattern {patt!r} in {session.trial_dir!r}; "
                f"using first one ({files[0].name})."
            )
        fpath = files[0]
        kwargs = spec.get('kwargs', {})

        if name == 'raw':
            raw_attrs, df = _process_raw(fpath, kwargs)
            session.raw_attributes = raw_attrs
            _add_data('raw', df)

        elif name == 'phot':
            phot_dfs, phot_meta = _process_phot(fpath, kwargs, session)
            session.signal_meta = phot_meta
            for key, df in phot_dfs.items():
                _add_data(key, df)

        else:
            df = _safe_read_csv(fpath, **kwargs)
            _add_data(name, df)


def populate_containers(sessions):
    """Apply populate_session to every Session in the list."""
    for session in sessions:
        try:
            populate_session(session)
        except Exception as e:
            warnings.warn(f"Failed processing session {session.trial_id}/{session.chamber_id}: {e}")
            continue
