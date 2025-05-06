# parsers.py
from abc import ABC, abstractmethod
from pathlib import Path
import warnings, pandas as pd

# A global registry mapping short names → Parser classes
PARSER_REGISTRY: dict[str, type] = {}

def register_parser(name: str):
    """Class decorator to register a parser under a given name."""
    def decorator(cls):
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator


class FileParser(ABC):
    def __init__(self, name: str, spec: dict):
        self.name     = name
        self.glob_pat = spec['pattern']
        self.kwargs   = spec.get('kwargs', {})

    @abstractmethod
    def find_and_parse(self, trial_path: Path, session):
        """Return { key: DataFrame | None }."""
        ...

    def _warn(self, pattern):
        warnings.warn(f"No files match '{pattern}' for '{self.name}'")

    def _read(self, fp: Path, session, phot: bool=False):
        if phot:
            df = pd.read_csv(fp, **self.kwargs)
            df = session._normalise_photometry_columns(df)
            cols = [c for c in df.columns if session.filter_columns(c)]
            return df.loc[:, cols]
        else:
            return pd.read_csv(fp, usecols=session.filter_columns, **self.kwargs)


@register_parser('raw')
class RawParser(FileParser):
    def find_and_parse(self, trial_path, session):
        pat = self.glob_pat.format(setup=session.chamber_id)
        files = list(trial_path.glob(pat))
        # print(f"[DEBUG] RawParser matched {len(files)} file(s) for pattern {pat}: {files}")
        if not files:
            self._warn(pat)
            return {self.name: None}
        return {self.name: self._read(files[0], session, phot=False)}


@register_parser('ttl')
class TTLParser(FileParser):
    def find_and_parse(self, trial_path, session):
        pat = self.glob_pat.format(setup=session.chamber_id)
        files = list(trial_path.glob(pat))
        # print(f"[DEBUG] TTLParser matched {len(files)} TTL file(s) for pattern {pat}: {files}")
        if not files:
            # fallback for legacy TTL
            files += list(trial_path.glob("DigInput*.csv"))
            files += list(trial_path.glob("DigitalIO*.csv"))
        if not files:
            self._warn(pat)
            return {self.name: None}
        return {self.name: self._read(files[0], session, phot=False)}


@register_parser('phot')
class PhotometryParser(FileParser):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.combined_glob = spec.get('combined_glob')
        self.split_map     = spec.get('split_map', {})

    def find_and_parse(self, trial_path, session):
        out = {}
        # 1) Combined CSV takes priority
        if self.combined_glob:
            combo = list(trial_path.glob(self.combined_glob))
            if combo:
                df_all = pd.read_csv(combo[0], **self.kwargs)
                df_all = session._normalise_photometry_columns(df_all)
                for led, freq in self.split_map.items():
                    df = df_all[df_all['LedState']==led].copy()
                    df = df.reset_index(drop=True)
                    cols = [c for c in df.columns if session.filter_columns(c)]
                    out[f"phot_{freq}"] = df.loc[:, cols]
                return out

        # 2) Otherwise fall back on per‐freq files
        for freq in session.freqs:
            pat  = self.glob_pat.format(setup=session.setup_id, freq=freq)
            files = list(trial_path.glob(pat))
            key   = f"phot_{freq}"
            if not files:
                self._warn(pat)
                out[key] = None
            else:
                out[key] = self._read(files[0], session, phot=True)
        return out
