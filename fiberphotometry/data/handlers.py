# handlers.py

from abc import ABC, abstractmethod
import re
from typing import Any, Callable, Optional, Tuple
import pandas as pd
from pandas import DataFrame
import warnings

# Matches names like "Current_Reward_Delay_RD4" or "Current_ITI_VI300"
# so we can split off the "_RD4" or "_VI300" suffix.
EVENT_RE = re.compile(r'^(.*?)(_([A-Z]{2})(\d+(?:\.\d+)?))$')

def split_event_suffix(name: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    If `name` ends with _XX123 (two uppercase letters + digits), split it off.
    Returns:
      - base:     the part before the suffix (or the whole name if no suffix)
      - code:     the two-letter code (e.g. "RD") or None
      - number:   the numeric string (e.g. "4", "300.0") or None
    """
    # Attempt a full match against our strict pattern
    m = EVENT_RE.fullmatch(name)
    if not m:
        # No valid suffix found → return the original name, with no code/number
        return name, None, None

    # Unpack the captured groups
    base, _, code, num = m.groups()
    return base, code, num


def base_event_name(name: str) -> str:
    """
    Shortcut to get just the base name without any suffix.
    """
    return split_event_suffix(name)[0]


def _ensure_columns(df, require_arg1: bool = False):
    """
    Verify that `df` is a pandas DataFrame and contains the
    required columns before any processing occurs.

    Args:
      df:             The DataFrame to check.
      require_arg1:   If True, also require an 'Arg1_Value' column.

    Raises:
      TypeError:  if `df` is not a DataFrame.
      ValueError: if any required column is missing.
    """
    # 1) Type check
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    # 2) Collect any missing column names
    missing = []
    if 'Item_Name' not in df.columns:
        missing.append('Item_Name')
    if require_arg1 and 'Arg1_Value' not in df.columns:
        missing.append('Arg1_Value')

    # 3) If any are missing, raise with a clear message
    if missing:
        cols = ', '.join(missing)
        raise ValueError(f"DataFrame is missing required columns: {cols}")


class SpecialEventProcessor(ABC):
    trigger: str
    converter: Callable[[Any], Any]
    suffix_prefix: str

    def __init__(
        self,
        trigger: str,
        converter: Callable[[Any], Any],
        suffix_prefix: str
    ) -> None:
        self.trigger = trigger
        self.converter = converter
        self.suffix_prefix = suffix_prefix

    def _safe_convert(self, value: Any) -> Any:
        """
        Run the converter on the raw Arg1_Value, raising a
        contextual error if it fails.
        """
        try:
            return self.converter(value)
        except Exception as e:
            raise ValueError(
                f"[{self.__class__.__name__}] failed to convert "
                f"Arg1_Value={value!r} for trigger {self.trigger!r}"
            ) from e
        
    def _rename_trigger_rows(self, df: DataFrame, mask: pd.Series) -> None:
        """
        Rename all rows where mask is True by appending
        '_<converted Arg1_Value>' to the trigger.
        """
        def _format_row(row: pd.Series) -> str:
            suffix = self._safe_convert(row['Arg1_Value'])
            return f"{self.trigger}_{suffix}"

        # Apply the formatting only to the matching rows
        df.loc[mask, 'Item_Name'] = df.loc[mask].apply(_format_row, axis=1)

    def apply(self, df: DataFrame) -> DataFrame:
        """
        1) Validate that df has 'Item_Name' and 'Arg1_Value'.
        2) Find rows where Item_Name == trigger.
        3) If any, rename them via _rename_trigger_rows; otherwise warn.
        4) Delegate to subclass-specific process().
        """
        # 1) Check required columns
        _ensure_columns(df, require_arg1=True)

        # 2) Identify trigger rows
        mask = df['Item_Name'] == self.trigger

        # 3) Rename or warn
        if not mask.any():
            warnings.warn(
                f"No rows found for trigger {self.trigger!r}; suffix logic skipped",
                UserWarning,
                stacklevel=2
            )
        else:
            self._rename_trigger_rows(df, mask)

        # 4) Hand off to the subclass’s additional logic
        return self.process(df)

    @abstractmethod
    def process(self, df: DataFrame) -> DataFrame:
        """
        Subclasses implement their specific suffix-painting logic here.
        """
        ...


class AdjacentProcessor(SpecialEventProcessor):
    """
    For each trigger row, “paint” its suffix onto the immediately
    preceding and following rows, then remove the trigger rows.
    """
    def process(self, df: DataFrame) -> DataFrame:
        # Only 'Item_Name' is needed here
        _ensure_columns(df, require_arg1=False)

        # 1) If the very first row is a trigger, drop it so it has no prior neighbor
        first = df.iloc[0]['Item_Name']
        if first.startswith(f"{self.trigger}_"):
            df = df.iloc[1:].reset_index(drop=True)

        # 2) Find all remaining trigger rows
        mask = df['Item_Name'].str.startswith(f"{self.trigger}_")

        # 3) For each trigger row, extract the suffix value and
        #    append '_<prefix><value>' to its neighbors
        for idx in df.index[mask]:
            # everything after the last '_' is our raw suffix
            value = df.at[idx, 'Item_Name'].split('_')[-1]
            suffix = f"_{self.suffix_prefix}{value}"

            # paint onto the previous row, if it exists
            if idx > 0:
                df.at[idx-1, 'Item_Name'] += suffix

            # paint onto the next row, if it exists
            if idx < len(df) - 1:
                df.at[idx+1, 'Item_Name'] += suffix

        # 4) Remove the original trigger rows and reset the index
        return df.drop(index=df.index[mask]).reset_index(drop=True)


class RunningProcessor(SpecialEventProcessor):
    """
    Carry the most recent trigger’s suffix forward onto all subsequent
    rows (except 'Display Image') until the next trigger, then drop triggers.
    """
    def process(self, df: DataFrame) -> DataFrame:
        # Only 'Item_Name' is needed here
        _ensure_columns(df, require_arg1=False)

        current_suffix = None
        to_drop = []

        # Walk through each row in order
        for i in range(len(df)):
            name = df.loc[i, 'Item_Name']

            if name.startswith(f"{self.trigger}_"):
                # This is a new trigger: capture its suffix and mark for removal
                raw = name.rsplit('_', 1)[-1]
                current_suffix = f"_{self.suffix_prefix}{raw}"
                to_drop.append(i)
            elif current_suffix and name != 'Display Image':
                # Apply the running suffix to all other rows (except Display Image)
                df.at[i, 'Item_Name'] += current_suffix

        # Drop all trigger rows in one go, then reset index
        return df.drop(index=to_drop).reset_index(drop=True)
    

class RenameOnlyProcessor(SpecialEventProcessor):
    """
    Simply rename trigger rows to include their Arg1_Value suffix,
    but leave all other rows untouched.
    """
    def apply(self, df: DataFrame) -> DataFrame:
        # Ensure both columns are present
        _ensure_columns(df, require_arg1=True)

        mask = df['Item_Name'] == self.trigger
        if mask.any():
            # Use _safe_convert for consistent error messaging
            def _format_row(r: pd.Series) -> str:
                val = self._safe_convert(r['Arg1_Value'])
                return f"{self.trigger}_{self.suffix_prefix}{val}"

            df.loc[mask, 'Item_Name'] = df.loc[mask].apply(_format_row, axis=1)
        else:
            # No trigger rows to rename
            warnings.warn(
                f"No rows found for trigger {self.trigger!r}; nothing renamed",
                UserWarning, stacklevel=2
            )

        # No additional processing needed
        return df

    def process(self, df: DataFrame) -> DataFrame:
        return df


class CompositeProcessor:
    """
    Chains multiple SpecialEventProcessor instances in sequence.

    After running each processor’s apply(), it drops the shared 'Arg1_Value'
    column so downstream code only sees fully suffixed Item_Name values.
    """
    def __init__(self, procs: list[SpecialEventProcessor]) -> None:
        # Validate that we got a proper list of processors
        if not isinstance(procs, list) or any(
            not isinstance(p, SpecialEventProcessor) for p in procs
        ):
            raise TypeError(
                "CompositeProcessor requires a list of SpecialEventProcessor instances"
            )
        self.processors = procs

    def apply(self, df: DataFrame) -> DataFrame:
        """
        1) (Optionally) validate basic DataFrame shape
        2) Apply each sub-processor in order
        3) Drop 'Arg1_Value' once all renaming is done
        """
        # 1) Just ensure we have a DataFrame; Arg1_Value may or may not still be present
        _ensure_columns(df, require_arg1=False)

        # 2) Sequentially apply each contained processor
        for proc in self.processors:
            df = proc.apply(df)

        # 3) Clean up the Arg1_Value column once everyone has had their turn
        if 'Arg1_Value' in df.columns:
            df = df.drop(columns=['Arg1_Value'])

        return df

    @property
    def triggers(self) -> list[str]:
        """
        The list of raw trigger names this composite handles.
        """
        return [p.trigger for p in self.processors]

    @property
    def suffix_prefix(self) -> None:
        """
        CompositeProcessor itself does not define a single suffix_prefix.
        """
        return None
    

# your special‐case map
SPECIAL_PROCESSORS = {
    'rewardDelay':             AdjacentProcessor('Current_Reward_Delay', int,  'RD'),
    'probabilisticReward_50%': AdjacentProcessor('Feeder #1',           str,  'PR'),
    'varITI':                  RunningProcessor(  'Current_ITI',        int,  'VI'),
    'varStimDur':              RunningProcessor(  'stimulus_duration',  str,  'SD'),
}
SPECIAL_PROCESSORS["Fixed_Ratio_baseline"] = CompositeProcessor([
    RenameOnlyProcessor("Correct_Counter",     int, "CC"),
    RenameOnlyProcessor("FIXED_RATIO_COUNTER", int, "FR"),
])

# ─────── import-time validation of triggers ─────────
for key, proc in SPECIAL_PROCESSORS.items():
    base, code, num = split_event_suffix(proc.trigger)
    if code is not None:
        raise ValueError(
            f"Configuration error in SPECIAL_PROCESSORS[{key!r}]: "
            f"trigger {proc.trigger!r} was parsed as base={base!r}, "
            f"code={code!r}, num={num!r}.  "
            "Handler triggers must be pure base names (no _XX123 suffix)."
        )

def get_event_key(event_name: str, mapping: dict) -> Optional[str]:
    """
    Try to map an event_name (with optional suffix) into your attr key.
    Warns if no mapping is found.
    """
    base, code, num = split_event_suffix(event_name)

    # 1) If there's a suffix code and base is known, use that
    if code and base in mapping:
        return mapping[base] + f"_{code.lower()}{num}"

    # 2) Fallback to direct mapping
    key = mapping.get(event_name)
    if key is not None:
        return key

    # 3) Nothing matched → warn the user so they know something’s off
    warnings.warn(
        f"[get_event_key] no mapping for event_name={event_name!r} "
        f"(parsed base={base!r}, code={code!r}, num={num!r})",
        UserWarning,
        stacklevel=2
    )
    return None


def is_reward_event(name: str, reward_dict: dict, suffix_prefix=None) -> bool:
    if suffix_prefix:
        return any(name == k or name.startswith(f"{k}_{suffix_prefix}") for k in reward_dict)
    return name in reward_dict
