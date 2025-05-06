# fiberphotometry/data/renamer.py
import re
from typing import List, Tuple, Dict
import warnings
import pandas as pd
from fiberphotometry import config

class Renamer:
    @staticmethod
    def rename_df_columns(df: pd.DataFrame, pattern: str, replacement: str) -> None:
        # Use a function with re.sub for more robust replacement
        def replacer(col):
            return re.sub(pattern, replacement, str(col)) # Ensure col is string
        df.rename(columns=replacer, inplace=True)

    @staticmethod
    def rename_sessions_data(
        sessions: List,
        patterns: List[Tuple[str, dict]]
    ) -> None:
        """
        Applies a series of renaming patterns to specified dataframe types within sessions.
        patterns: list of (df_type_prefix, {"pattern": str, "replacement": str})
        e.g. [("ttl", {"pattern":"_DigInput[012]","replacement":""}), ...]
        """
        for session in sessions:
            keys = session.dfs.fetch_all_data_names()
            for df_type, pat_info in patterns:
                # Ensure pattern and replacement exist
                if "pattern" not in pat_info or "replacement" not in pat_info:
                    warnings.warn(f"Skipping invalid pattern entry: {pat_info}")
                    continue

                pattern = pat_info["pattern"]
                replacement = pat_info["replacement"]

                to_rename = [k for k in keys if k.startswith(df_type)]
                for key in to_rename:
                    df = session.dfs.get_data(key)
                    if df is not None and not df.empty:
                         # Check if the pattern actually exists before trying to rename
                         # This avoids unnecessary operations and potential issues
                         # Note: This simple check might not catch complex regex patterns perfectly
                         # if any(re.search(pattern, str(col)) for col in df.columns):
                         Renamer.rename_df_columns(df, pattern, replacement)
                         # else:
                         #    pass # Pattern not found in columns, skip rename for this df/pattern

    @staticmethod
    def finalize_ttl_for_session(session):
        """
        1) apply cleanup
        2) tag columns by priority
        3) pick the highest‐priority tag → TTL_ts
        4) drop all other temporary tags
        """
        df = session.dfs.get_data('ttl')
        if df is None or df.empty:
            return

        # 1) cleanup legacy bits
        Renamer.rename_sessions_data([session], config.TTL_STRIPPING_PATTERNS)

        # 2) tag by priority
        Renamer.rename_sessions_data([session], config.TTL_PRIORITY_PATTERNS)

        # 3) pick the top‐priority tag and rename it to the final column
        for _, pat_info in config.TTL_PRIORITY_PATTERNS:
            tag = pat_info['replacement']
            if tag in df.columns:
                df.rename(columns={tag: config.TTL_FINAL_NAME}, inplace=True)
                break

        # 4) drop any other __TTL_P*_ columns
        to_drop = [c for c in df.columns
                   if c.startswith("__TTL_P") and c != config.TTL_FINAL_NAME]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)

    @staticmethod
    def extract_region_number(column_name: str, letter: str) -> str:
        if letter == 'iso':
            letter = ''
        m = re.search(r'Region(\d+)'+letter, column_name)
        return m.group(1) if m else None

    @staticmethod
    def rename_sessions_fiber_to_brain_region(
        sessions: List,
        frequencies: Dict[str,str]
    ) -> None:
        """
        frequencies: e.g. {"G":"470","R":"560","iso":"415"}
        """
        for session in sessions:
            for letter, freq in frequencies.items():
                key = f'phot_{freq}'
                df = session.dfs.get_data(key)
                if df is None:
                    continue
                df.rename(
                    columns=lambda col: session
                        .fiber_to_region
                        .get(Renamer.extract_region_number(col, letter), col),
                    inplace=True
                )
