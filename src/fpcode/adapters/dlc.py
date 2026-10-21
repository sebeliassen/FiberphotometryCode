"""Decode DLC multi-index tables into labelled, immutable pose arrays."""

import numpy as np
import pandas as pd

from fpcode.contracts import PoseSeries, TimeAxis


def pose_from_dlc_table(table, *, time: TimeAxis, unit, coordinate_frame, scorer=None, bodyparts=None):
    if not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels != 3:
        raise ValueError("DLC columns must contain scorer, bodypart and coordinate levels")
    scorers = table.columns.get_level_values(0).unique().tolist()
    if scorer is None:
        if len(scorers) != 1:
            raise ValueError("choose a scorer explicitly for tables containing several models")
        scorer = scorers[0]
    if scorer not in scorers:
        raise KeyError(f"DLC scorer {scorer!r} is absent")
    parts = tuple(bodyparts) if bodyparts is not None else tuple(table[scorer].columns.get_level_values(0).unique())
    if not parts:
        raise ValueError("DLC table contains no bodyparts")
    coordinates = np.stack([table[(scorer, part)].loc[:, ["x", "y"]].to_numpy() for part in parts], axis=1)
    confidence = np.stack([table[(scorer, part, "likelihood")].to_numpy() for part in parts], axis=1)
    return PoseSeries(time, coordinates, confidence, parts, unit, coordinate_frame)


def read_dlc_hdf(path, *, time, unit, coordinate_frame, scorer=None, bodyparts=None, key=None):
    return pose_from_dlc_table(pd.read_hdf(path, key=key), time=time, unit=unit,
                               coordinate_frame=coordinate_frame, scorer=scorer, bodyparts=bodyparts)
