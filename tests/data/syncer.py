import pytest
import pandas as pd
from typing import Dict, Optional
from fiberphotometry.data.syncer import Syncer
from fiberphotometry import config

# Default SYNC config for tests
def default_cfg():
    return {
        "raw_time_col": "Time",
        "ttl_time_cols": ["TTL_Time", "Alt_TTL"],
        "phot_time_cols": ["Time"],
        "frequencies": [10, 20],
        "reference_phot_freq": 10,
        "sec_zero_col": "SecZero",
        "sec_trial_col": "SecTrial",
        "truncate_streams": False,
    }

@pytest.fixture(autouse=True)
def _patch_default_config(monkeypatch):
    # Always use a fresh copy
    monkeypatch.setattr(config, 'SYNC', default_cfg())

# Dummy DFS and Session for synthetic data
DummyData = Dict[str, pd.DataFrame]
class DummyDFS:
    def __init__(self, data: DummyData):
        self.data = data
    def get_data(self, key: str) -> pd.DataFrame | None:
        return self.data.get(key)

class DummySession:
    def __init__(self, data: DummyData):
        self.dfs = DummyDFS(data)
        self.cpt = None
        self.sync_time = None

@pytest.fixture
def synthetic_session():
    # Raw has two "Set Blank Images" to test picking the first
    raw = pd.DataFrame({
        'Item_Name': ['Set Blank Images', 'A', 'Set Blank Images', 'B'],
        'Time': [0.0, 1.0, 2.0, 3.0],
    })
    ttl = pd.DataFrame({ 'TTL_Time': [0.5, 1.5, 2.5] })
    phot10 = pd.DataFrame({ 'Time': [0.2, 1.2, 2.2, 3.2] })
    phot20 = pd.DataFrame({ 'Time': [0.1, 1.1, 2.1] })
    data = {
        'raw': raw.copy(),
        'ttl': ttl.copy(),
        'phot_10': phot10.copy(),
        'phot_20': phot20.copy(),
    }
    return DummySession(data)

# calculate_cpt_index tests
@pytest.mark.parametrize("items, expected", [
    (['A', 'Set Blank Images', 'B'], 1),
    (['A', 'B', 'C'], -1),
    (['Set Blank Images','Set Blank Images'], 0),
])
def test_calculate_cpt_index(items, expected):
    df = pd.DataFrame({ 'Item_Name': items })
    assert Syncer.calculate_cpt_index(df) == expected

# Basic sync_session synthetic check
def test_sync_session_synthetic(synthetic_session):
    session = synthetic_session
    Syncer.sync_session(session)
    assert session.cpt == 0  # first occurrence
    assert session.sync_time == 0.0
    raw = session.dfs.get_data('raw')
    assert 'SecZero' in raw.columns and 'SecTrial' in raw.columns
    for freq in [10, 20]:
        df = session.dfs.get_data(f'phot_{freq}')
        # phot_20 sec columns exist even if reference freq is 10
        assert 'SecZero' in df.columns and 'SecTrial' in df.columns

# Exact timestamp alignment test
def test_exact_timestamp_alignment(monkeypatch):
    cfg = default_cfg()
    cfg.update({'ttl_time_cols': ['TTL_Time'], 'truncate_streams': False})
    monkeypatch.setattr(config, 'SYNC', cfg)
    # Construct simple data
    raw = pd.DataFrame({'Item_Name': ['Start', 'Set Blank Images'], 'Time': [0.0, 2.0]})
    ttl = pd.DataFrame({'TTL_Time': [1.0, 3.0]})
    phot10 = pd.DataFrame({'Time': [0.5, 2.5]})
    data = {'raw': raw, 'ttl': ttl, 'phot_10': phot10}
    session = DummySession(data)
    Syncer.sync_session(session)
    # offset = (1.0 - 0.5) - 2.0 = -1.5
    raw = session.dfs.get_data('raw')
    assert pytest.approx(raw.at[0,'SecZero'], rel=1e-3) == 0.0 + -1.5
    assert pytest.approx(raw.at[1,'SecZero'], rel=1e-3) == 2.0 + -1.5
    assert pytest.approx(raw.at[0,'SecTrial'], rel=1e-3) == 0.0 - 2.0
    ttl = session.dfs.get_data('ttl')
    assert pytest.approx(ttl.at[0,'SecZero'], rel=1e-3) == 1.0 - 1.0
    phot = session.dfs.get_data('phot_10')
    assert pytest.approx(phot.at[0,'SecTrial'], rel=1e-3) == (0.0 - -1.5)

# Non-numeric time entries do not crash
def test_non_numeric_time(monkeypatch, synthetic_session):
    session = synthetic_session
    raw = session.dfs.get_data('raw')
    raw.loc[0, 'Time'] = 'bad'
    # Should coerce to NaN but still run
    Syncer.sync_session(session)
    raw = session.dfs.get_data('raw')
    assert raw['SecZero'].isna().iloc[0]

# Missing photometry time column: stream is skipped
def test_missing_phot_time_column(monkeypatch, synthetic_session):
    cfg = default_cfg()
    cfg['phot_time_cols'] = ['No_Col']
    monkeypatch.setattr(config, 'SYNC', cfg)
    session = synthetic_session
    # Should not raise
    Syncer.sync_session(session)
    # phot_10 still exists but without Sec columns
    phot = session.dfs.get_data('phot_10')
    assert 'SecZero' not in phot.columns

# Empty photometry stream is skipped without error
def test_empty_stream(monkeypatch):
    cfg = default_cfg()
    monkeypatch.setattr(config, 'SYNC', cfg)
    raw = pd.DataFrame({'Item_Name': ['Set Blank Images'], 'Time': [0]})
    ttl = pd.DataFrame({'TTL_Time': [0]})
    phot10 = pd.DataFrame()
    data = {'raw': raw, 'ttl': ttl, 'phot_10': phot10}
    session = DummySession(data)
    # No exception
    Syncer.sync_session(session)

# Direct sync_all_streams without sync_session
def test_direct_sync_all_streams(monkeypatch):
    cfg = default_cfg()
    cfg['truncate_streams'] = False
    monkeypatch.setattr(config, 'SYNC', cfg)
    raw = pd.DataFrame({'Item_Name': [], 'Time': []})
    ttl = pd.DataFrame({'TTL_Time': [0]})
    phot10 = pd.DataFrame({'Time': [0]})
    data = {'raw': raw, 'ttl': ttl, 'phot_10': phot10}
    session = DummySession(data)
    session.sync_time = 0.0
    # Should not raise and add SecZero
    Syncer.sync_all_streams(session)
    assert 'SecZero' in session.dfs.get_data('raw').columns

# Raw time column missing error
def test_raw_time_col_missing(monkeypatch, synthetic_session):
    cfg = default_cfg()
    monkeypatch.setattr(config, 'SYNC', cfg)
    session = synthetic_session
    raw = session.dfs.get_data('raw')
    raw.drop(columns=['Time'], inplace=True)
    with pytest.raises(KeyError):
        Syncer.sync_session(session)

# Truncate streams no-op when only one frequency
def test_truncate_streams_noop(monkeypatch):
    cfg = default_cfg()
    cfg.update({'frequencies': [10], 'truncate_streams': True, 'reference_phot_freq': 10})
    monkeypatch.setattr(config, 'SYNC', cfg)
    raw = pd.DataFrame({'Item_Name': ['Set Blank Images'], 'Time': [0]})
    ttl = pd.DataFrame({'TTL_Time': [0]})
    phot10 = pd.DataFrame({'Time': [0, 1, 2]})
    data = {'raw': raw, 'ttl': ttl, 'phot_10': phot10}
    session = DummySession(data)
    Syncer.sync_session(session)
    # phot10 length should remain unchanged
    assert len(session.dfs.get_data('phot_10')) == 3
