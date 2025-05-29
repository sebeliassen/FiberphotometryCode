from collections import defaultdict
import warnings
from fiberphotometry.analysis.response_metrics import calculate_signal_response_metrics_matrix
from fiberphotometry.utils import find_start_end_idxs
from fiberphotometry import config
import numpy as np
from fiberphotometry.config import SYNC


class EventSignalExtractor:
    """Extract per-event photometry windows and summary metrics for one session."""

    # ── constructor ──────────────────────────────────────────────────────────
    def __init__(self, session):
        self.session = session
        self._validate_session()
        self.trial_time_col = SYNC["sec_trial_col"]
        self.interval_start = config.peak_interval_config["interval_start"]
        self.interval_end   = config.peak_interval_config["interval_end"]
        self.window_len     = self.interval_start + self.interval_end

        if not hasattr(self.session, 'response_metrics'):
            self.session.response_metrics = defaultdict(list)

    # ── public API ───────────────────────────────────────────────────────────
    def extract(self, event_type, brain_region):
        """
        Return dict with signal_matrix, idx ranges, metrics, phot_pointer,
        mirroring the old get_brain_region_event_signal_info().
        """
        phot_df, raw_df, signal_col = self._get_dataframes(brain_region)
        fps             = config.PLOTTING_CONFIG[self.session.session_type]["fps"]
        response_interval    = find_start_end_idxs(event_type, fps)

        event_idxs = self.session.event_idxs_container.get_data(event_type)
        if not event_idxs:
            raise ValueError(f"No event indices for {event_type!r}")

        matrix, ranges = self._build_signal_matrix(
            event_idxs,
            phot_df,
            raw_df[self.trial_time_col].values,
            phot_df[self.trial_time_col].values,
            signal_col
        )

        metrics = calculate_signal_response_metrics_matrix(matrix, response_interval, fps)
        return {
            "signal_matrix":     matrix,
            "signal_idx_ranges": ranges,
            "response_metrics":  metrics,
            "phot_pointer":      phot_df[signal_col],
        }

    # ── private helpers ──────────────────────────────────────────────────────
    def _validate_session(self):
        s = self.session
        if not hasattr(s, "dfs") or not hasattr(s, "event_idxs_container"):
            raise AttributeError("session must have .dfs and .event_idxs_container")
        if not isinstance(config.peak_interval_config["interval_start"], int):
            raise TypeError("peak_interval_config must contain integers")

    def _get_dataframes(self, brain_region):
        # 1) raw DataFrame as before
        raw_df = self.session.dfs.get_data("raw")
        if raw_df is None:
            raise ValueError("raw DataFrame missing")

        # 2) find which photometry stream (freq) to load
        freq_label = config.LETTER_TO_FREQS.get(brain_region[-1])
        phot_df    = self.session.dfs.get_data(f"phot_" + freq_label)
        if phot_df is None:
            raise ValueError(f"phot_{freq_label} DataFrame missing")

        # 3) confirm we have a signal_meta mapping for this session
        if not hasattr(self.session, "signal_meta"):
            raise AttributeError("session.signal_meta not set; did you run populate_session()?")

        # 4) find the column name in phot_df that matches the requested brain_region
        #    signal_meta maps new_name → (region, side, color, orig_col)
        target_key = None
        for new_name, (region, side, color, orig_col) in self.session.signal_meta.items():
            if (region, side, color) == brain_region:
                target_key = new_name
                break

        if target_key is None:
            raise KeyError(
                f"No entry in signal_meta matching brain_region={brain_region!r}"
            )

        # 5) now verify the DataFrame actually has that column
        if target_key not in phot_df.columns:
            raise KeyError(
                f"Expected column {target_key!r} for region {brain_region!r} "
                f"not found in photometry DF"
            )

        # 6) confirm time‐column is present & sorted
        col = self.trial_time_col
        for name, df in (("raw", raw_df), ("phot", phot_df)):
            if col not in df.columns:
                raise KeyError(f"{col!r} missing from {name} DataFrame")

        if not np.all(np.diff(phot_df[col].values) >= 0):
            raise ValueError("phot_times not monotonically increasing")

        return phot_df, raw_df, target_key

    def _build_signal_matrix(self, raw_event_idxs, phot_df, raw_times,
                             phot_times, brain_region):

        n_events   = len(raw_event_idxs)
        matrix     = np.zeros((n_events, self.window_len), float)
        idx_ranges = []

        for i, ridx in enumerate(raw_event_idxs):
            ev_time  = raw_times[ridx]
            pidx     = int(np.searchsorted(phot_times, ev_time, side="right"))

            # bounds check
            if pidx < self.interval_start or pidx + self.interval_end > len(phot_times):
                warnings.warn(
                    f"Skip event at raw idx {ridx}: window would overflow photometry data"
                )
                continue

            s = pidx - self.interval_start
            e = pidx + self.interval_end
            idx_ranges.append((s, e))

            trace    = phot_df[brain_region].iloc[s:e].to_numpy()
            pre_mean = trace[self.interval_start-7 : self.interval_start+7].mean()
            matrix[i] = trace - pre_mean

        return matrix, idx_ranges


def get_session_signal_info(session):
    """
    Build the full signal_info dict for a session by
    extracting every (event_type, brain_region) pair.
    """
    extractor   = EventSignalExtractor(session)
    signal_info = {}
    import re

    # same regex you had for suffix logic
    suffix_re = re.compile(r'^(?P<base>.+)_(?P<suffix>[a-z]{2}\d+)$')

    for br_region in session.brain_regions:
        # unpack region as before
        try:
            br, side, color = (br_region.split("_") 
                               if isinstance(br_region, str) 
                               else br_region)
        except ValueError:
            raise ValueError(
                f"brain_region {br_region!r} must split/iter to exactly 3 parts"
            )

        for event_key, idxs in session.event_idxs_container.data.items():
            if not idxs:
                continue

            m = suffix_re.match(event_key)
            # full key
            info_full = extractor.extract(event_key, br_region)
            signal_info[(br, color, event_key)] = info_full

            # base key if present
            if m:
                base_key = m.group("base")
                if session.event_idxs_container.data.get(base_key):
                    info_base = extractor.extract(base_key, br_region)
                    signal_info[(br, color, base_key)] = info_base

    return signal_info


def assign_sessions_signal_info(sessions):
    """
    Iterate sessions and attach the full signal_info dict.
    """
    for session in sessions:
        session.signal_info = get_session_signal_info(session)