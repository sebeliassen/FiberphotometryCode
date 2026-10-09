def detector_percentile_sanity(df, method='omega', bins=(50,60,70,80,85,90,92,95,97,99),
                               speed_col='velocity_px_s', omega_col='angular_velocity',
                               kappa_col='curvature', speed_floor_omega=2.0, speed_floor_kappa=10.0,
                               sessions_to_show=5):
    import numpy as np
    import pandas as pd
    sids = df['session_idx'].unique().tolist()
    print(f"Sessions: {len(sids)}  | method={method}  | bins={bins}")
    shown = 0
    rows = []
    for sid in sids:
        x = df[df['session_idx']==sid]
        spd = x[speed_col].to_numpy()
        mag = np.abs(x[omega_col].to_numpy()) if method=='omega' else np.abs(x[kappa_col].to_numpy())
        speed_floor = speed_floor_omega if method=='omega' else speed_floor_kappa
        ref = mag[spd >= speed_floor]
        if ref.size == 0:
            ref = mag
        thr = np.percentile(ref, bins).astype(float)
        # monotonic?
        mono = np.all(np.diff(thr) > 0)
        rows.append([sid, mono, *thr])
        if shown < sessions_to_show:
            shown += 1
            print(f"sid {sid}: monotonic={mono}  " + "  ".join([f"P{b}={t:.4g}" for b,t in zip(bins,thr)]))
    cols = ['session_idx','monotonic']+[f'P{b}' for b in bins]
    return pd.DataFrame(rows, columns=cols)

def label_bin_counts(df, label_col='turn_syllable', onset_col='turn_onset'):
    lab = df.loc[df[onset_col].astype(bool), label_col].dropna().astype(str)
    # extract trailing number after underscore
    bins = lab.str.extract(r'_(\d+)$')[0]
    vc = bins.value_counts().sort_index()
    print("Onsets per bin:\n", vc.to_string())
    return vc
