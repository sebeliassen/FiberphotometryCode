import os
import fnmatch
import pandas as pd
import pytest

from pathlib import Path
from fiberphotometry.data.data_loading import (
    DataContainer,
    sort_key_func,
    detect_freqs,
    Session,
)


def test_data_container_basic():
    dc = DataContainer()
    # first add sets the data_type
    dc.add_data('a', 1)
    assert dc.get_data('a') == 1
    assert dc.data_type is int
    # adding wrong type raises
    with pytest.raises(TypeError):
        dc.add_data('b', 'string')
    # remove and clear
    dc.add_data('c', 2)
    assert set(dc.fetch_all_data_names()) == {'a', 'c'}
    dc.remove_data('a')
    assert 'a' not in dc.fetch_all_data_names()
    dc.clear_data()
    assert dc.fetch_all_data_names() == []


def test_sort_key_func():
    assert sort_key_func('T1') == (1,)
    assert sort_key_func('T2_extra') == (2,)
    # multi-digit
    assert sort_key_func('T10') == (10,)
    # sorting order
    dirs = ['T2', 'T10', 'T1']
    assert sorted(dirs, key=sort_key_func) == ['T1', 'T2', 'T10']


def test_detect_freqs_primary(tmp_path):
    d = tmp_path / 'trial'
    d.mkdir()
    # primary photwrit files
    (d / 'channel123photwrit_file.csv').write_text('')
    (d / 'channel045photwrit_file.csv').write_text('')
    freqs = detect_freqs(d, setup=None)
    # leading zeros preserved, sorted lex
    assert freqs == ['045', '123']


def test_detect_freqs_fallback(tmp_path):
    d = tmp_path / 'trial'
    d.mkdir()
    (d / 'c789_bonsaiTS_file.csv').write_text('')
    freqs = detect_freqs(d, setup=None)
    assert freqs == ['789']


def test_create_fiber_to_region_dict_and_filter():
    # Build a session_guide with fiber columns
    series = pd.Series(
        ['LH_L', float('nan'), 'mPFC_R', float('nan')],
        index=['G1', 'dummy', 'G2', 'dummy2'],
    )
    session = Session.__new__(Session)
    session.session_guide = series
    # create mapping
    mapping = Session.create_fiber_to_region_dict(session)
    assert mapping == {'1': ('LH', 'L', 'G'), '2': ('mPFC', 'R', 'G')}
    # filter_columns based on mapping
    session.fiber_to_region = mapping
    assert Session.filter_columns(session, 'G1')
    assert Session.filter_columns(session, 'Region1X')
    assert not Session.filter_columns(session, 'G3')


def test_load_data(tmp_path):
    # Create dummy session object with trial_dir
    class Dummy:
        def __init__(self, trial_dir):
            self.trial_dir = str(trial_dir)
        load_data = Session.load_data

    d = tmp_path / 'trial'
    d.mkdir()
    # simple CSV
    csv = d / 'datafile.csv'
    csv.write_text('col1,col2\n1,2\n3,4')
    ds = Dummy(d)
    df = ds.load_data('datafile.csv')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    # header only
    cols = ds.load_data('datafile.csv', only_header=True)
    assert cols == ['col1', 'col2']
    # use_cols selection
    df2 = ds.load_data('datafile.csv', use_cols=['col1'])
    assert list(df2.columns) == ['col1']
    # pattern matching
    (d / 'test1.csv').write_text('x,y\n5,6')
    (d / 'test2.csv').write_text('x,y\n7,8')
    df3 = ds.load_data('test?.csv')
    assert isinstance(df3, pd.DataFrame)
    # only first match is loaded
    assert 'x' in df3.columns


def test_load_all_data_invalid_type():
    # Unknown session_type should raise ValueError in load_all_data
    session = Session.__new__(Session)
    session.trial_dir = str(tmp_path) if 'tmp_path' in globals() else ''
    session.chamber_id = 'A'
    session.session_type = 'invalid'
    with pytest.raises(ValueError):
        session.load_all_data()