from pathlib import Path
import pandas as pd
from io import StringIO
from fiberphotometry.data.data_loading import DataContainer
from fiberphotometry.config import DATA_PATTERNS

def populate_containers(sessions):
    """
    Given a list of Session objects (already created via load_all_sessions),
    fill each session.dfs with raw, ttl, and photometry DataFrames if present,
    and for raw also capture the leading key/value pairs into session.raw_attributes.
    """
    for session in sessions:
        specs = DATA_PATTERNS[session.session_type]
        session.dfs = DataContainer()

        for name, spec in specs.items():
            patt = spec['pattern'].format(
                chamber_id=session.chamber_id,
                setup_id=session.setup_id
            )
            files = list(Path(session.trial_dir).glob(patt))
            if not files:
                continue

            f = files[0]
            kwargs = spec.get('kwargs', {})

            # special‐case "raw": split off the header block before "----------"
            if name == 'raw':
                text = f.read_text().splitlines()
                # find the first line of dashes
                sep_idx = next((i for i,l in enumerate(text) if l.startswith('---')), None)
                if sep_idx is None:
                    # no separator found; treat entire file as data
                    sep_idx = -1

                # parse Name,Value pairs into a dict
                raw_attrs = {}
                for line in text[:sep_idx]:
                    if ',' in line:
                        k, v = line.split(',', 1)
                        raw_attrs[k.strip()] = v.strip()
                session.raw_attributes = raw_attrs

                # and now read the CSV after the separator
                data_txt = "\n".join(text[sep_idx+1:])
                df = pd.read_csv(StringIO(data_txt), **kwargs)

            else:
                # ttl & phot use plain read_csv
                df = pd.read_csv(f, **kwargs)

            session.dfs.add_data(name, df)
