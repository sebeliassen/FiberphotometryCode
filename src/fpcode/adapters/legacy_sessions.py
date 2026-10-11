"""Read legacy Session and SessionRecord objects without importing their packages."""

import numpy as np

from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis

LEGACY_TANDEM_PHOTOMETRY_COLUMNS = [
    "SystemTimestamp", "signal_0", "signal_1", "signal_2", "signal_3",
    *[f"signal_{channel}_{kind}" for channel in range(4)
      for kind in ("zscore_ransac", "zscore_wallace", "dff", "zscore", "prominence")],
]


def project_legacy_photometry_table(table, index):
    """Preserve the tandem export schema, index reindexing and missing-column NaNs."""
    projected = table.reindex(index).reset_index(drop=True).copy()
    for column in LEGACY_TANDEM_PHOTOMETRY_COLUMNS:
        if column not in projected:
            projected[column] = np.nan
    return projected.loc[:, LEGACY_TANDEM_PHOTOMETRY_COLUMNS]


def attach_export_tables(session, *, recording_key, syllables, all_data=None):
    """Update existing empirical objects without replacing their class or metadata."""
    session.results_dict_key = recording_key
    session.syllables = syllables
    if all_data is not None:
        session.all_data_df = all_data
    return session


def photometry_table(session, frequency):
    """Resolve the existing SessionRecord or DataContainer storage convention."""
    if hasattr(session, "photometry") and frequency in session.photometry:
        return session.photometry[frequency]
    if hasattr(session, "dfs"):
        table = session.dfs.get_data(f"phot_{frequency}")
        if table is not None:
            return table
    raise KeyError(f"session has no photometry channel {frequency!r}")


def signal_metadata(session):
    """Return the existing mapping, including legacy tuple-valued entries."""
    metadata = getattr(session, "metadata", {})
    return metadata.get("signal_meta", getattr(session, "signal_meta", {}))


def region_columns(session):
    """Resolve region labels using each record's historical metadata convention."""
    out = {}
    for column, entry in signal_metadata(session).items():
        region = entry.get("region") if isinstance(entry, dict) else entry[0]
        if region:
            out[column] = region
    return out


def as_signal_series(session, frequency, *, time_column="SystemTimestamp", time_unit, clock):
    """Convert raw columns only; no interpolation, correction, or unit guessing."""
    table = photometry_table(session, frequency)
    columns = [c for c in table if str(c).startswith("signal_")]
    if not columns:
        raise ValueError("session channel contains no signal columns")
    regions = region_columns(session)
    labels = tuple(SignalMetadata(str(c), region=regions.get(c), wavelength=str(frequency)) for c in columns)
    return SignalSeries(TimeAxis(table[time_column].to_numpy(), time_unit, clock), table[columns].to_numpy(), labels)


def iter_signal_matrices(session, *, missing="skip"):
    """Bridge the historical signal_info dictionary into the aggregation API."""
    if missing not in {"skip", "raise"}:
        raise ValueError("missing must be skip or raise")
    if missing == "raise":
        information = session.signal_info
    else:
        information = getattr(session, "signal_info", None) or {}
    for key, item in information.items():
        matrix = item.get("signal_matrix")
        if matrix is not None and getattr(matrix, "size", 0):
            yield key, matrix


__all__ = ["photometry_table", "signal_metadata", "region_columns", "as_signal_series", "iter_signal_matrices",
           "LEGACY_TANDEM_PHOTOMETRY_COLUMNS", "project_legacy_photometry_table", "attach_export_tables"]
