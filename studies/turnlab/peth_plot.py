import os

import matplotlib.pyplot as plt
import numpy as np

from .config import *
from .utils import _collect_regions_hemis, _interp_then_smooth, _prefix_for, _region_hemi_for


def compute_turn_peths_from_merged(
    merged_all,
    desired_labels=None,
    label_col='turn_syllable',
    onset_col='turn_onset',
    signal_cols=None,
    sampling_rate=None,
    window_seconds=None,
    smoothing_window=None,
    smoothing_poly_order=None,
    confidence=None,
    min_traces_per_syllable=None,
    save=True,
    out_dir=None,
    combine_signals=True,
    lateralization='absolute',         # 'absolute' or 'relative'
    link_y_by_region=True,
    collapse_relative=False,            # False | True/'by_bin' | 'all'
    verbose=False
):
    signal_cols           = signal_cols or SIGNAL_COLS
    sampling_rate         = sampling_rate or SAMPLING_RATE
    window_seconds        = window_seconds or WINDOW_SECONDS
    smoothing_window      = smoothing_window or SMOOTHING_WINDOW
    smoothing_poly_order  = smoothing_poly_order or SMOOTHING_POLY_ORDER
    confidence            = confidence if confidence is not None else CONFIDENCE_LEVEL
    min_traces_per_syllable = (MIN_TRACES_PER_SYLLABLE
                               if min_traces_per_syllable is None else min_traces_per_syllable)
    out_dir = out_dir or os.path.join(PROJECT_DIR_FULL or '.', 'peth_turns')

    window_samples = int(sampling_rate * window_seconds)
    total_samples  = 2 * window_samples
    time_vector    = np.linspace(-window_seconds, window_seconds, total_samples)
    regions, hemis_order_abs = _collect_regions_hemis(signal_cols)
    lat_order = ['contra','ipsi'] if lateralization == 'relative' else hemis_order_abs

    try: os.makedirs(out_dir, exist_ok=True)
    except: pass

    df = merged_all.sort_values(['session_idx','frame_index']).reset_index(drop=True)
    sy = df[label_col]
    on = df[onset_col].astype(bool)

    if verbose:
        print(f"[compute_turn_peths] desired_labels={desired_labels}")
        print(f"[compute_turn_peths] lateralization={lateralization}, collapse={collapse_relative}")
        print(f"[compute_turn_peths] total onsets in df: {int(on.sum())}")

    if desired_labels is not None:
        desired_labels = set(desired_labels)
    else:
        desired_labels = set(sy.dropna().astype(str).unique().tolist())

    def _dir_and_bin(lbl: str):
        if not isinstance(lbl, str): return None, None
        L = lbl.lower().strip()
        turn_dir = 'left' if L.startswith('left') else ('right' if L.startswith('right') else None)
        # pull trailing number after underscore, e.g. left_90 -> '90'
        m = None
        try:
            parts = L.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                m = parts[-1]
        except:
            pass
        return turn_dir, m

    # ----- decide aggregation labels -----
    collapse_mode = 'none'
    if lateralization == 'relative':
        if collapse_relative is True or (isinstance(collapse_relative, str) and collapse_relative.lower()=='by_bin'):
            collapse_mode = 'by_bin'
        elif isinstance(collapse_relative, str) and collapse_relative.lower()=='all':
            collapse_mode = 'all'

    if collapse_mode == 'by_bin':
        # keep one figure per percentile bin present in desired_labels
        bins = sorted({ _dir_and_bin(lbl)[1] for lbl in desired_labels if _dir_and_bin(lbl)[1] is not None })
        labels_for_agg = [f'__REL_BIN_{b}' for b in bins]
    elif collapse_mode == 'all':
        labels_for_agg = ['__REL_ALL__']
    else:
        labels_for_agg = sorted(desired_labels)

    # traces[label_key][(region,lat,group)][signal] = [trace,...]
    syll_traces = {lab:{} for lab in labels_for_agg}
    onset_idx = on[on].index.to_numpy()

    # track why onsets get skipped
    skipped_bad_label = skipped_boundary = skipped_region = 0

    for i in onset_idx:
        raw_label = sy.iloc[i]
        if not isinstance(raw_label, str) or raw_label == '':
            skipped_bad_label += 1
            continue
        if desired_labels is not None and raw_label not in desired_labels:
            skipped_bad_label += 1
            continue

        ses   = int(df.loc[i, 'session_idx'])
        start = i - window_samples
        end   = i + window_samples
        if start < 0 or end > len(df) or df.loc[start:end-1, 'session_idx'].nunique() != 1:
            skipped_boundary += 1
            continue

        turn_dir, bin_key = _dir_and_bin(raw_label)
        group = str(df.loc[i, 'group']) if 'group' in df.columns else 'UNK'

        if collapse_mode == 'by_bin':
            if bin_key is None:
                skipped_bad_label += 1
                continue
            label_key = f'__REL_BIN_{bin_key}'
        elif collapse_mode == 'all':
            label_key = '__REL_ALL__'
        else:
            label_key = raw_label

        bucket = syll_traces.get(label_key)
        if bucket is None:
            continue

        for col in signal_cols:
            prefix = _prefix_for(col)
            region, hemi = _region_hemi_for(ses, prefix)
            if region is None or hemi is None:
                skipped_region += 1
                continue

            lat = hemi
            if lateralization == 'relative':
                if turn_dir is None:
                    skipped_bad_label += 1
                    continue
                if turn_dir == 'left':
                    lat = 'contra' if hemi == 'right' else ('ipsi' if hemi == 'left' else None)
                else:
                    lat = 'contra' if hemi == 'left' else ('ipsi' if hemi == 'right' else None)
                if lat is None:
                    skipped_region += 1
                    continue

            key = (region, lat, group)
            if key not in bucket:
                bucket[key] = {c: [] for c in signal_cols}

            trace = df.loc[start:end-1, col].to_numpy()
            if trace.shape[0] != total_samples:
                skipped_boundary += 1
                continue
            smoothed = _interp_then_smooth(trace, smoothing_window, smoothing_poly_order)
            baseline = smoothed[window_samples]
            z = (smoothed - baseline).astype('float32', copy=False)
            bucket[key][col].append(z)

    # counts before filtering
    raw_counts = {lbl: sum(len(vv) for dd in bucket.values() for vv in dd.values())
                  for lbl, bucket in syll_traces.items()}

    if verbose:
        print(f"[compute_turn_peths] built trace buckets for {len(syll_traces)} figure keys.")
        print("  raw counts per key:", raw_counts)
        print(f"  skipped: bad_label={skipped_bad_label}, boundary/cross-session={skipped_boundary}, no_region/hemi={skipped_region}")

    # ----- choose which figures to render (by count) -----
    syll_counts = {}
    for label_key, bucket in syll_traces.items():
        total = 0
        for data_by_signal in bucket.values():
            total += sum(len(v) for v in data_by_signal.values())
        syll_counts[label_key] = total
    plot_labels = sorted([lbl for lbl, c in syll_counts.items() if c >= min_traces_per_syllable])

    if verbose:
        print(f"[compute_turn_peths] min_traces_per_syllable={min_traces_per_syllable}")
        print(f"[compute_turn_peths] plot_labels (passed filter): {plot_labels}")

    # ----- plotting -----
    from scipy.stats import sem
    totals_plotted = {}

    def _nice_title_and_fname(label_key):
        if label_key.startswith('__REL_BIN_'):
            b = label_key[len('__REL_BIN_'):]
            title = f"Turn onset: contra/ipsi (collapsed across left/right) | bin {b}"
            fname = f"rel_collapsed_{b}.png"
        elif label_key == '__REL_ALL__':
            title = "Turn onset: contra/ipsi (collapsed across all selected labels)"
            fname = "rel_collapsed_all.png"
        else:
            title = f"Turn onset: {label_key} ({'contra/ipsi' if lateralization=='relative' else 'left/right'})"
            safe = str(label_key).replace('/', '_').replace(' ', '')
            fname = f"{safe}.png"
        return title, fname

    for label_key in plot_labels:
        n_rows, n_cols = max(1, len(lat_order)), max(1, len(regions))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 2, 3.8 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        title_txt, file_name = _nice_title_and_fname(label_key)
        try: fig.suptitle(title_txt, fontsize=16)
        except: pass

        y_extrema = {reg: [np.inf, -np.inf] for reg in regions}
        label_total = 0
        bucket = syll_traces[label_key]

        for r_idx, region in enumerate(regions):
            for h_idx, lat in enumerate(lat_order):
                ax = axes[h_idx, r_idx]
                traces_by_group = {'WT': [], 'KI': []}

                for (reg, lat_key, grp), data_by_signal in bucket.items():
                    if reg == region and lat_key == lat and grp in traces_by_group:
                        if combine_signals:
                            for c in signal_cols:
                                traces_by_group[grp].extend([t for t in data_by_signal.get(c, []) if t is not None])
                        else:
                            for c in signal_cols:
                                traces_by_group.setdefault(f"{grp}:{c}", [])
                                traces_by_group[f"{grp}:{c}"].extend([t for t in data_by_signal.get(c, []) if t is not None])

                for grp, traces in traces_by_group.items():
                    if not traces: continue
                    valid = [t for t in traces if t is not None and t.shape[0]==total_samples and not np.isnan(t).any()]
                    if not valid: continue
                    arr = np.stack(valid)
                    m = arr.mean(axis=0)
                    e = sem(arr, axis=0)
                    lo, hi = m - confidence*e, m + confidence*e
                    color = 'darkblue' if ('KI' in grp) else 'green'
                    ax.plot(time_vector, m, label=f"{grp} (n={len(valid)})", color=color)
                    ax.fill_between(time_vector, lo, hi, color=color, alpha=0.2)
                    label_total += len(valid)

                    ymin = np.nanmin(lo); ymax = np.nanmax(hi)
                    if np.isfinite(ymin) and np.isfinite(ymax):
                        y_extrema[region][0] = min(y_extrema[region][0], ymin)
                        y_extrema[region][1] = max(y_extrema[region][1], ymax)

                ax.set_title(f"{region} | {lat}", fontsize=12)
                ax.axvline(0, color='crimson', linestyle='--', lw=1.8)
                ax.axhline(0, color='grey', linestyle=':', lw=1)
                if h_idx == len(lat_order) - 1: ax.set_xlabel("Time from turn onset (s)")
                if r_idx == 0: ax.set_ylabel("ΔF/F (z), baseline @ onset")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(fontsize='small', frameon=False, loc='upper right')

        if link_y_by_region:
            for r_idx, region in enumerate(regions):
                lo, hi = y_extrema[region]
                if not (np.isfinite(lo) and np.isfinite(hi)): continue
                span = hi - lo
                lo2, hi2 = lo - 0.05*span, hi + 0.05*span
                for h_idx in range(len(lat_order)):
                    axes[h_idx, r_idx].set_ylim(lo2, hi2)

        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        out_path = os.path.join(out_dir, file_name)
        if save:
            try:
                plt.savefig(out_path, dpi=150); print(f"💾 Saved: {out_path}")
            except Exception as e:
                print(f"⚠️ Save failed for {label_key}: {e}")
            plt.close()
        else:
            plt.show(); plt.close()

        totals_plotted[label_key] = label_total

    return {
        'syll_traces': syll_traces,
        'plot_labels': plot_labels,
        'regions': regions,
        'lat_order': lat_order,
        'time_vector': time_vector,
        'totals_plotted': totals_plotted,
        'out_dir': out_dir,
        'lateralization': lateralization,
        'link_y_by_region': link_y_by_region,
        'collapse_mode': collapse_mode
    }
