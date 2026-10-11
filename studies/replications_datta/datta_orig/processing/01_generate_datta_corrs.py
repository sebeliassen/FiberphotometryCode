import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fpcode.common.paths import get_repo_root

# Try to discover the repo root
start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
for p in (start, *start.parents):
    if p.name == "syllable_generation":
        REPO_ROOT = p
        break
else:
    REPO_ROOT = Path(str((get_repo_root() / 'compat/syllable_generation')))

# Make package importable
PKG_PARENT = REPO_ROOT.parent
for pth in (str(PKG_PARENT), str(REPO_ROOT)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

# --- I/O CONFIG ---
OUT_DIR = REPO_ROOT / "analysis/replications/datta/processing/dlight_scalar_corr"
CACHE   = OUT_DIR / "datta_cache"
for d in [OUT_DIR, CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# --- ANALYSIS CONFIG ---
BIN_GRID = list(range(1, 61))  # 1..60 s bins
SEED     = 123
NBOOT    = 500
NSHUFF   = 300        # 300 shuffles per mode (Total 600/bin)
NJOBS    = -1

# OPTIMIZATION: Mean Only
SCALAR_MODES = ["mean"]
SCALAR_ABS_MEAN_KEYS = {}

# REGEX: Capture Standard (_zscore) AND Datta (_zscore_datta)
SIGNAL_COL_REGEX = r"^signal_\d+_zscore.*$"

RATE_METHOD = "cross"
CORR_METHOD = "pearson"
TIME_BIN_MAX_S  = None

rng = np.random.RandomState(SEED)

# ==============================================================================
# 1. LOAD SESSIONS
# ==============================================================================
PKL = REPO_ROOT / "loaders" / "jonathan_ifdn_sessions_w_sylls.pkl"

if not PKL.exists():
    loaders_dir = REPO_ROOT / "loaders"
    candidates = list(loaders_dir.glob("*sessions*all_data*.pkl")) + list(loaders_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"Could not find sessions pickle in {loaders_dir}")
    PKL = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Using fallback pickle: {PKL}")

print(f"Loading sessions from: {PKL} ...")
with PKL.open("rb") as f:
    sessions = pickle.load(f)

# Filter usable sessions
sessions = [
    s for s in sessions
    if hasattr(s, "all_data_df") and isinstance(s.all_data_df, pd.DataFrame) and not s.all_data_df.empty
]
print(f"Loaded {len(sessions)} usable sessions.")

# ==============================================================================
# 2. HELPERS
# ==============================================================================
def _zscore_groupwise(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    def _z(x):
        v = x.values.astype(float)
        m = np.nanmean(v)
        s = np.nanstd(v)
        if not np.isfinite(s) or s == 0:
            return (v - m)
        return (v - m) / s
    out[cols] = out.groupby(by, observed=True)[cols].transform(_z)
    return out

def _grad(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    t = t.astype(float)
    if len(y) < 2: return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, t)

def peak_rate_cross(x: pd.Series, threshold: float = 1.96) -> float:
    v = x.to_numpy()
    if v.size <= 1: return np.nan

    # Identify valid transitions:
    # 1. Start is below threshold
    # 2. End is above threshold
    # 3. BOTH Start and End must be finite (not NaN)

    # Shifted arrays
    v_curr = v[:-1]
    v_next = v[1:]

    # Logic: (Below -> Above) AND (Not NaN -> Not NaN)
    crossing = (v_curr < threshold) & (v_next > threshold) & np.isfinite(v_curr) & np.isfinite(v_next)

    # We normalize by the number of VALID transitions, not total frames
    valid_transitions = np.isfinite(v_curr) & np.isfinite(v_next)
    n_valid = valid_transitions.sum()

    if n_valid == 0:
        return np.nan

    return crossing.sum() / n_valid

def _shuffle_signals_within_groups(df, group_cols, cols_to_shuffle, seed, mode):
    out = df.copy()
    rs = np.random.RandomState(seed)
    if mode == "permute":
        for c in cols_to_shuffle:
            out[c] = (
                out.groupby(group_cols, observed=True)[c]
                .transform(lambda x: pd.Series(rs.permutation(x.values), index=x.index))
            )
        return out
    elif mode == "roll":
        grouped = out.groupby(group_cols, observed=True, sort=False)
        group_ids = grouped.ngroup().to_numpy()
        n_groups = group_ids.max() + 1
        arrays = {c: out[c].to_numpy() for c in cols_to_shuffle}
        for g in range(n_groups):
            idx = np.nonzero(group_ids == g)[0]
            if idx.size <= 1: continue
            shift = rs.randint(0, idx.size)
            if shift == 0: continue
            for c in cols_to_shuffle:
                arr = arrays[c]
                arr[idx] = np.roll(arr[idx], shift)
        for c, arr in arrays.items():
            out[c] = arr
        return out
    return df

# ==============================================================================
# 3. BUILD LONG TABLE
# ==============================================================================
rows = []
sig_annot_rows = []

print("Building long table...")
for si, s in enumerate(sessions):
    sid = s.session_id
    meta = s.metadata or {}
    animal = meta.get("animal_id", "UNK")
    geno   = meta.get("genotype", "UNK")
    adf = s.all_data_df.copy()

    # Labels
    labels = adf["syllable"].replace({-5: pd.NA, -1: pd.NA}).astype("Int64")
    adf["labels"] = labels
    adf = adf.dropna(subset=["labels"]).copy()
    if adf.empty: continue

    # Transitions
    change = adf["labels"].diff().ne(0)
    adf["transition"] = change.fillna(False).astype(float)
    adf["syllable_number"] = adf["transition"].cumsum().astype("uint32")

    # Kinematics
    adf["timestamp"] = adf["SystemTimestamp"].astype(float)
    vel = adf["velocity_px_s"].to_numpy()
    t   = adf["timestamp"].to_numpy()

    # Calculate Jerk/Acc
    acc = _grad(vel, t)
    jerk = _grad(acc, t)
    adf["acceleration"] = acc
    adf["jerk"] = jerk

    # Duration
    syl_size = adf.groupby(["syllable_number"], observed=True).size().rename("duration")
    adf = adf.set_index(["syllable_number"]).join(syl_size).reset_index()

    # Identify Signals
    all_sig_cols = sorted([c for c in adf.columns if re.match(SIGNAL_COL_REGEX, c)])

    # FILTER: Only keep signals that have a valid Brain Region (Not UNK)
    keep_sig_cols = []
    smeta = meta.get("signal_meta", {}) or {}

    for c in all_sig_cols:
        k = c.split("_")[1]
        sig_key = f"signal_{k}"
        m = smeta.get(sig_key, {})

        brain_reg = m.get("brain_reg", "UNK")

        # SKIP UNK REGIONS (Speed optimization)
        if brain_reg == "UNK":
            continue

        sig_annot_rows.append({
            "session_id": sid, "signal": c, "base_signal": sig_key,
            "brain_reg": brain_reg,
            "hemisphere": m.get("hemisphere", "UNK"),
            "fiber_num": m.get("fiber_num", "UNK"),
            "animal_id": animal, "genotype": geno,
        })
        keep_sig_cols.append(c)

    if not keep_sig_cols:
        continue

    adf["session_id"] = sid
    adf["animal_id"]  = animal
    adf["genotype"]   = geno

    keep_cols = ([
        "session_id","animal_id","genotype","timestamp",
        "labels","transition","syllable_number",
        "velocity_px_s","angular_velocity",
        "acceleration","jerk","duration"
    ] + keep_sig_cols)

    rows.append(adf[keep_cols])

if not rows:
    raise RuntimeError("No data found! Check regex or animals.csv mapping.")

use_data = pd.concat(rows, ignore_index=True)

# ==============================================================================
# [FIX] ALIGNMENT CLEANUP
# ==============================================================================
# 1. Drop rows where Timestamp is NaN (Alignment Gaps / Dropped Packets).
#    These cannot be binned and cause IntCastingNaNError.
print(f"Rows before cleanup: {len(use_data)}")
use_data = use_data.dropna(subset=["timestamp"])
print(f"Rows after cleanup:  {len(use_data)}")

# 2. Ensure all signal columns are strictly numeric (float), converting any
#    leftover objects (from manual exclusions) to NaN to avoid aggregation errors.
signal_cols = [c for c in use_data.columns if re.match(SIGNAL_COL_REGEX, c)]
for col in signal_cols:
    use_data[col] = pd.to_numeric(use_data[col], errors='coerce')
# ==============================================================================

print(f"Final use_data shape: {use_data.shape}")

scalar_cols = ["velocity_px_s","angular_velocity","acceleration","jerk","transition","duration"]

# Z-Score Scalars & Signals within session
use_data = _zscore_groupwise(use_data, by="session_id", cols=signal_cols + scalar_cols)

# ==============================================================================
# 4. AGGREGATION & CORRELATION
# ==============================================================================
def _time_bin_index(ts: pd.Series, sid: pd.Series, width_s: float) -> pd.Series:
    t0 = use_data.groupby(sid, observed=True)["timestamp"].transform("min")
    return np.floor((ts - t0) / float(width_s)).astype("int32")

def bin_data(timescale: float | str, neural_agg) -> pd.DataFrame:
    data_keys = scalar_cols + ["labels"]
    data = use_data
    if timescale == "syllable":
        time_bin = data["syllable_number"].astype("int32")
    else:
        time_bin = _time_bin_index(data["timestamp"], data["session_id"], timescale)

    grp_keys = [data["session_id"], data["animal_id"], time_bin]
    y = data.groupby(grp_keys, observed=True)[signal_cols].agg(neural_agg)
    neural_agg_name = getattr(neural_agg, "__name__", str(neural_agg))
    x = data.groupby(grp_keys, observed=True)[data_keys].mean()

    agg_matrix = x.join(y)
    if timescale != "syllable":
        sz = data.groupby(grp_keys, observed=True).size()
        if not sz.empty:
            modal_sz = sz.mode().iat[0]
            agg_matrix = agg_matrix.loc[sz.loc[sz == modal_sz].index]

    agg_matrix["timescale"]   = -5 if timescale == "syllable" else int(timescale)
    agg_matrix["neural_agg"]  = neural_agg_name
    agg_matrix["scalar_agg"]  = "mean"
    return agg_matrix

def run_bin_block(neural_agg, timescales: list[int]) -> pd.DataFrame:
    mats = []
    for ts in timescales:
        mats.append(bin_data(ts, neural_agg))
    mats.append(bin_data("syllable", neural_agg))
    return pd.concat(mats, axis=0)

# --- Run Aggregation ---
delays = []
delays.append(delayed(run_bin_block)(peak_rate_cross, BIN_GRID))
delays.append(delayed(run_bin_block)("mean", BIN_GRID))

print("Running aggregation...")
agg_mats = Parallel(n_jobs=NJOBS, verbose=5, backend="loky")(delays)
agg_data = pd.concat(agg_mats).reset_index().rename(columns={"level_0": "session_id", "level_1": "animal_id", "level_2": "bin"})
agg_data["timescale"] = agg_data["timescale"].astype(int)

# --- Run Correlation (Observed) ---
print("Calculating Observed correlations...")
corr_group = ["timescale", "session_id", "neural_agg", "scalar_agg"]
corr_keys  = scalar_cols + ["labels"] + signal_cols
use_agg = agg_data.dropna(subset=corr_keys).copy()
use_agg = use_agg.sort_values(["timescale", "session_id", "animal_id", "bin"]).reset_index(drop=True)

blocks = []
for grp_vals, sub in use_agg.groupby(corr_group, observed=True):
    C = sub[corr_keys].corr(method=CORR_METHOD)
    tall_rows = []
    for sig in signal_cols:
        if sig not in C.index: continue
        for sk in scalar_cols + ["labels", "duration"]:
            if sk in C.columns:
                r_val = C.loc[sk, sig]
                if pd.notna(r_val):
                    tall_rows.append({"signal": sig, "feature": sk, "r": r_val})
    blk = pd.DataFrame(tall_rows)
    if blk.empty: continue
    for k, v in zip(corr_group, grp_vals): blk[k] = v
    blocks.append(blk)

corrs = pd.concat(blocks, ignore_index=True)

# ==============================================================================
# 5. SHUFFLE (ROLL + PERMUTE)
# ==============================================================================
print("Running Shuffles...")
shuffle_group = ["timescale", "session_id", "neural_agg", "scalar_agg"]

def _one_shuffle(seed: int, mode_name: str) -> pd.DataFrame:
    df_perm = _shuffle_signals_within_groups(
        use_agg, group_cols=shuffle_group, cols_to_shuffle=signal_cols, seed=seed, mode=mode_name
    )
    blocks = []
    for grp_vals, sub in df_perm.groupby(shuffle_group, observed=True):
        C = sub[corr_keys].corr(method=CORR_METHOD)
        tall_rows = []
        for sig in signal_cols:
            if sig not in C.index: continue
            for sk in scalar_cols + ["labels", "duration"]:
                if sk in C.columns:
                    r_val = C.loc[sk, sig]
                    if pd.notna(r_val):
                        tall_rows.append({"signal": sig, "feature": sk, "r": r_val})
        blk = pd.DataFrame(tall_rows)
        if blk.empty: continue
        for k, v in zip(shuffle_group, grp_vals): blk[k] = v
        blk["shuffle_index"] = seed
        blk["shuffle_mode"]  = mode_name
        blocks.append(blk)
    if blocks: return pd.concat(blocks, ignore_index=True)
    return pd.DataFrame()

shuffle_tasks = []
for i in range(NSHUFF):
    shuffle_tasks.append(delayed(_one_shuffle)(i, "roll"))
    shuffle_tasks.append(delayed(_one_shuffle)(i, "permute"))

shuffle_df_list = Parallel(n_jobs=NJOBS, verbose=5, backend="loky")(shuffle_tasks)
shuffle_df = pd.concat([d for d in shuffle_df_list if not d.empty], ignore_index=True)

# ==============================================================================
# 6. SAVE TO PARQUET
# ==============================================================================
sig_annot = pd.DataFrame(sig_annot_rows).drop_duplicates()
corrs = corrs.merge(sig_annot, how="left", on=["session_id","signal"])
shuffle_df = shuffle_df.merge(sig_annot, how="left", on=["session_id","signal"])

def clean_for_parquet(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

ts_str = str(int(time.time()))
agg_out   = OUT_DIR / f"datta_corr_AGG_data_raw_{ts_str}.parquet"
corrs_out = OUT_DIR / f"datta_corr_OBSERVED_raw_{ts_str}.parquet"
shuff_out = OUT_DIR / f"datta_corr_SHUFFLE_raw_combined_{ts_str}.parquet"
annot_out = OUT_DIR / f"datta_corr_ANNOTATIONS_{ts_str}.parquet"

print("\nSaving outputs (Parquet format)...")
clean_for_parquet(agg_data).to_parquet(agg_out, index=False)
clean_for_parquet(corrs).to_parquet(corrs_out, index=False)
clean_for_parquet(shuffle_df).to_parquet(shuff_out, index=False)
clean_for_parquet(sig_annot).to_parquet(annot_out, index=False)
print("Done.")
