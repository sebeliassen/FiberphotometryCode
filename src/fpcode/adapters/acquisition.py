"""Decode photometry and Bonsai tables without changing scientific values."""

import pandas as pd

from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis


def signal_from_table(table, *, time_column, signal_columns, time_unit, clock, metadata=None):
    columns = tuple(signal_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("signal columns must be nonempty and unique")
    missing = [column for column in (time_column, *columns) if column not in table]
    if missing:
        raise KeyError(f"acquisition table is missing columns: {missing}")
    labels = tuple(metadata) if metadata is not None else tuple(SignalMetadata(str(c)) for c in columns)
    return SignalSeries(TimeAxis(table[time_column].to_numpy(), time_unit, clock),
                        table.loc[:, list(columns)].to_numpy(), labels)


def read_photometry_csv(path, *, time_column, signal_columns, time_unit, clock, metadata=None, **csv_options):
    table = pd.read_csv(path, **csv_options)
    return signal_from_table(table, time_column=time_column, signal_columns=signal_columns,
                             time_unit=time_unit, clock=clock, metadata=metadata)


def read_bonsai_timestamps(path, *, column, unit, clock, **csv_options):
    table = pd.read_csv(path, usecols=[column], **csv_options)
    return TimeAxis(table[column].to_numpy(), unit, clock)
