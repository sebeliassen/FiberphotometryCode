# fiberphotometry/data/syncer.py

import pandas as pd
from fiberphotometry import config

class Syncer:
    @staticmethod
    def calculate_cpt_index(raw_df: pd.DataFrame) -> int:
        """Find the row index of the ‘Set Blank Images’ event."""
        hits = raw_df.index[raw_df["Item_Name"] == "Set Blank Images"]
        return int(hits[0]) if len(hits) > 0 else -1

    @staticmethod
    def sync_session(session) -> None:
        """Calculate sync time and align raw, TTL, and all photometry streams."""
        raw_df = session.dfs.get_data("raw")
        cpt_idx = Syncer.calculate_cpt_index(raw_df)
        if cpt_idx < 0:
            return  # no sync event found

        session.cpt = cpt_idx
        session.sync_time = float(raw_df.at[cpt_idx, config.SYNC["raw_time_col"]])
        Syncer.sync_all_streams(session)

    @staticmethod
    def sync_all_streams(session) -> None:
        """Perform the single‐logic synchronization across TTL, raw, and photometry."""
        cfg               = config.SYNC
        raw_time_col      = cfg["raw_time_col"]
        ttl_df            = session.dfs.get_data("ttl")
        raw_df            = session.dfs.get_data("raw")
        frequencies       = cfg["frequencies"]
        sec_zero_name     = cfg["sec_zero_col"]
        sec_trial_name    = cfg["sec_trial_col"]
        reference_freq    = cfg["reference_phot_freq"]
        reference_key     = f"phot_{reference_freq}"

        # 1) Pick TTL column and compute TTL zero
        for candidate in cfg["ttl_time_cols"]:
            if candidate in ttl_df.columns:
                ttl_time_col = candidate
                break
        else:
            raise KeyError(f"No TTL column in {cfg['ttl_time_cols']}")

        ttl_start_time = float(ttl_df[ttl_time_col].iloc[0])
        ttl_df[sec_zero_name] = pd.to_numeric(ttl_df[ttl_time_col], errors="coerce") - ttl_start_time

        # 2) Pick reference photometry column and compute its zero
        ref_photometry_df = session.dfs.get_data(reference_key)
        for candidate in cfg["phot_time_cols"]:
            if candidate in ref_photometry_df.columns:
                phot_time_col = candidate
                break
        else:
            raise KeyError(f"No photometry time col in {cfg['phot_time_cols']} for {reference_key}")

        phot_start_time = float(ref_photometry_df[phot_time_col].iloc[0])

        # 3) Compute offset: align TTL clock to phot clock, adjusted by the raw sync event
        offset = (ttl_start_time - phot_start_time) - session.sync_time
        # 4) Stamp Raw data with SecFromZero
        raw_df[sec_zero_name] = pd.to_numeric(raw_df[raw_time_col], errors="coerce") + offset
        raw_df[sec_trial_name] = raw_df[raw_time_col] - session.sync_time

        # 5) Stamp every photometry stream
        for freq in frequencies:
            key = f"phot_{freq}"
            phot_df = session.dfs.get_data(key)
            if phot_df is None or phot_df.empty:
                continue

            # pick that stream’s time column
            for candidate in cfg["phot_time_cols"]:
                if candidate in phot_df.columns:
                    stream_time_col = candidate
                    break
            else:
                continue  # no time column → skip

            first_time = float(phot_df[stream_time_col].iloc[0])
            phot_df[sec_zero_name]   = pd.to_numeric(phot_df[stream_time_col], errors="coerce") - first_time
            phot_df[sec_trial_name]  = phot_df[sec_zero_name] - offset

        # 6) Optionally truncate all photometry series to the shortest
        if cfg.get("truncate_streams", False):
            lengths = []
            for freq in cfg["frequencies"]:
                key = f"phot_{freq}"
                df = session.dfs.get_data(key)
                if df is not None:
                    lengths.append(len(df))
            
            if not lengths:
                return
            
            n_min = min(lengths)
            for freq in cfg["frequencies"]:
                key = f"phot_{freq}"
                df = session.dfs.get_data(key)
                if df is not None:
                    session.dfs.data[key] = df.iloc[:n_min]

