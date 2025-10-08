# turnlab/experiments.py

from typing import Dict, List, Optional, Set, Tuple
import os
import numpy as np
import pandas as pd

from .detect import make_turn_syllable_series
from .peth_plot import compute_turn_peths_from_merged
from .cache_index import build_or_load_turn_index, slice_by_percentile, onset_mask_from_slice

def exp(name, method, bins, start_pct,
        min_bout_ms, merge_gap_ms,
        smooth_win, speed_floor,
        stop_ratio=0.6, desired_labels=None, dataset_tag=None):
    """
    Return a dict describing one detection+plot configuration.
    dataset_tag: optional string used to name the cache file (e.g., 'merged_all', 'merged_wt')
    """
    return {
        'name': name,
        'dataset_tag': dataset_tag,  # optional; runner will infer if None
        'detector_kwargs': dict(
            session_idx_col='session_idx',
            frame_col='frame_index',
            omega_col='angular_velocity',
            speed_col='velocity_px_s',
            method=method,                           # 'omega' or 'kappa' (also selects cache metric)
            granularity_percentiles=bins,            # thresholds like [90] or [90,97]
            fps=80,
            turn_start_pct=start_pct,
            stop_ratio=stop_ratio,
            min_bout_ms=min_bout_ms,
            merge_gap_ms=merge_gap_ms,
            smooth_win=smooth_win,
            clip_hi_pct=99.5,
            speed_floor=speed_floor
        ),
        'desired_labels': desired_labels,  # None -> inferred from bins
        'save': True
    }

def _bins_to_ranges(sorted_bins: List[int]) -> List[Tuple[int, int]]:
    """
    Convert thresholds [b1, b2, ..., bk] into half-open percentile ranges:
    [(b1, b2), (b2, b3), ..., (bk, 100)]
    """
    b = sorted(set(int(x) for x in sorted_bins))
    if not b:
        return []
    ranges = [(b[i], b[i+1]) for i in range(len(b)-1)]
    ranges.append((b[-1], 100))
    return ranges

def _labels_from_bins(bins: List[int]) -> Set[str]:
    """Return {'left_X','right_X', ...} where X are the *starts* of each bin."""
    return {f"{d}_{int(p)}" for d in ('left','right') for p in sorted(set(bins))}

def _infer_dataset_tag(df, fallback='df'):
    # best-effort tag based on columns present
    if 'group' in df.columns:
        # common split pattern in your notebooks
        if str(getattr(df['group'], 'cat', df['group'])).find('WT') >= 0 and df['group'].nunique() == 1:
            return 'merged_wt'
    return fallback

def run_turn_experiment(
    merged_all: pd.DataFrame,
    detector_kwargs: Dict,
    desired_labels: Optional[Set[str]] = None,
    label_col: str = 'turn_syllable',
    canonize: bool = False,               # kept for API compat; cache path ignores it
    save: bool = False,
    out_dir: Optional[str] = None,
    plot_kwargs: Optional[Dict] = None,
    tag: Optional[str] = None,
    use_cache: bool = True,
    dataset_tag: Optional[str] = None,
    verbose=False,
):
    """
    Run one experiment.
    - If use_cache=True (default): build/load turn-index cache and create onsets/labels from ranks.
    - Else: call detector directly (legacy path).
    """
    plot_kwargs = dict(plot_kwargs or {})

    # Compose output directory suffix for clarity
    if out_dir is not None:
        base = out_dir
    else:
        base = plot_kwargs.get('out_dir')  # peth_plot will default if not provided
    if tag:
        base = (base or '').rstrip('/\\')
        base = f"{base}_{tag}"
    if base:
        plot_kwargs['out_dir'] = base

    # ---- DEDUPE: pop keys that we also pass positionally
    if 'save' in plot_kwargs:
        save = plot_kwargs.pop('save')
    if 'label_col' in plot_kwargs:
        plot_kwargs.pop('label_col')
    if 'onset_col' in plot_kwargs:
        plot_kwargs.pop('onset_col')

    # Decide labels if not provided (mirrors original behavior)
    bins = detector_kwargs.get('granularity_percentiles', [90])
    if desired_labels is None:
        desired_labels = _labels_from_bins(bins)

    if not use_cache:
        # ----- Legacy path: recompute turns now -----
        turn_syll, turn_on = make_turn_syllable_series(merged_all, **detector_kwargs)
        df = merged_all.copy()
        df['turn_syllable'] = turn_syll
        df['turn_onset']    = turn_on
        use_label_col = label_col
        res = compute_turn_peths_from_merged(
            df,
            desired_labels=desired_labels,
            label_col=use_label_col,
            save=save,
            **plot_kwargs
        )
        onset_counts = {lbl: int(((df['turn_onset']) & (df[use_label_col] == lbl)).sum()) for lbl in desired_labels}
        return {
            'onset_counts': onset_counts,
            'plot_result': res,
            'label_col': use_label_col,
            'desired_labels': desired_labels,
            'out_dir': res['out_dir'],
            'detector_kwargs': detector_kwargs,
            'used_cache': False
        }

    # ----- Cache path: build/load index and slice bins from ranks -----
    metric = 'omega' if detector_kwargs.get('method','omega') == 'omega' else 'kappa'
    ds_tag = dataset_tag or _infer_dataset_tag(merged_all, 'df')

    cache_obj = build_or_load_turn_index(merged_all, dataset_tag=ds_tag, detector_kwargs=detector_kwargs)
    turn_ix = cache_obj['turn_index']

    # Build target percentile ranges that mirror "choose the highest bin exceeded"
    # e.g. [90,97] -> bins: (90,97], (97,100]
    ranges = _bins_to_ranges(sorted(detector_kwargs.get('granularity_percentiles', [90])))

    # If user passed explicit desired_labels (e.g., {'left_90','right_97'}), limit to those bin starts
    keep_starts = sorted({int(lbl.split('_')[-1]) for lbl in desired_labels})
    ranges = [r for r in ranges if r[0] in keep_starts]

    if verbose:
        total_bouts = len(turn_ix)
        print(f"[run_turn_experiment] metric={metric} | cache bouts={total_bouts} | ranges={ranges}")

    if not ranges:
        return {
            'onset_counts': {},
            'plot_result': {'out_dir': plot_kwargs.get('out_dir')},
            'label_col': 'turn_syllable_bin',
            'desired_labels': desired_labels,
            'out_dir': plot_kwargs.get('out_dir'),
            'detector_kwargs': detector_kwargs,
            'used_cache': True,
            'note': 'No percentile ranges matched the requested labels.'
        }

    # Build one combined onset mask + per-onset labels from cached direction
    df = merged_all.copy()
    df['turn_onset_sel'] = False
    df['turn_syllable_bin'] = None

    dirmap = dict(zip(turn_ix['onset_df_idx'], turn_ix['dir']))

    total_counts = {}
    selected_idx_all = []
    for (lo, hi) in ranges:
        sl = slice_by_percentile(turn_ix, metric=metric, p_lo=lo, p_hi=hi)
        idxs = sl['onset_df_idx'].to_numpy()
        total_counts[f'bin_{lo}'] = int(len(idxs))
        selected_idx_all.extend(idxs.tolist())
        if verbose:
            dirs = [dirmap.get(i, None) for i in idxs]
            n_left  = sum(1 for d in dirs if d == 'left')
            n_right = sum(1 for d in dirs if d == 'right')
            n_none  = sum(1 for d in dirs if d not in ('left','right'))
            print(f"  bin [{lo},{hi}): n={len(idxs)} | L={n_left}, R={n_right}, dir=None={n_none}")

        df.loc[idxs, 'turn_onset_sel'] = True
        labels = [f"{dirmap.get(i)}_{int(lo)}" if dirmap.get(i) in ('left','right') else None for i in idxs]
        df.loc[idxs, 'turn_syllable_bin'] = labels

    # Final pre-plot summary
    if verbose:
        sel_n = int(df['turn_onset_sel'].sum())
        vc = df.loc[df['turn_onset_sel'], 'turn_syllable_bin'].value_counts(dropna=False)
        print(f"  total selected onsets: {sel_n}")
        print("  labels among selected:\n", vc.to_string())

    # Decide which labels to plot
    desired_labels = {f'left_{int(lo)}' for (lo, _) in ranges} | {f'right_{int(lo)}' for (lo, _) in ranges}

    # Plot (relative/absolute handled inside)
    res = compute_turn_peths_from_merged(
        df,
        desired_labels=desired_labels,
        label_col='turn_syllable_bin',
        onset_col='turn_onset_sel',
        save=save,
        **plot_kwargs
    )

    # Counts for sanity
    onset_counts = {lbl: int(((df['turn_onset_sel']) & (df['turn_syllable_bin'] == lbl)).sum()) for lbl in desired_labels}
    if verbose:
        print("  onset_counts (actually passed to plotter):", onset_counts)

    return {
        'onset_counts': onset_counts,
        'bin_counts': total_counts,
        'plot_result': res,
        'label_col': 'turn_syllable_bin',
        'desired_labels': desired_labels,
        'out_dir': res['out_dir'],
        'detector_kwargs': detector_kwargs,
        'used_cache': True,
        'cache_meta': cache_obj.get('meta', {})
    }

def run_turn_experiments(
    merged_all: pd.DataFrame,
    experiments: List[Dict],
    common_plot: Optional[Dict] = None,
    use_cache: bool = True,
    dataset_tag: Optional[str] = None,
    verbose: bool = True
):
    """
    Run a list of experiments; by default uses the cached turn index.
    You can pass dataset_tag to label the cache files ('merged_all', 'merged_wt', ...).
    """
    from tqdm.notebook import tqdm

    results = {}
    for exp_cfg in tqdm(experiments, desc="Experiments", leave=False):
        name = exp_cfg.get('name', 'exp')
        dk   = exp_cfg['detector_kwargs']
        dl   = exp_cfg.get('desired_labels')
        sv   = exp_cfg.get('save', False)
        od   = exp_cfg.get('out_dir')
        pk   = dict(common_plot or {})
        pk.update(exp_cfg.get('plot_kwargs', {}))
        ds_tag = exp_cfg.get('dataset_tag', dataset_tag)

        # derive a default tag if none provided
        bins = dk.get('granularity_percentiles', [90])
        method = dk.get('method', 'omega')
        default_tag = f"{name}_{method}_{'-'.join(str(int(b)) for b in bins)}"
        tag = exp_cfg.get('tag', default_tag)

        # single progress update per experiment
        from tqdm.notebook import tqdm as _tqdm
        with _tqdm(total=1, desc=f"{name}", leave=False) as pbar:
            res = run_turn_experiment(
                merged_all,
                detector_kwargs=dk,
                desired_labels=dl,
                save=sv,
                out_dir=od,
                plot_kwargs=pk,
                tag=tag,
                use_cache=use_cache,
                dataset_tag=ds_tag,
                verbose=verbose
            )
            results[name] = res
            pbar.update(1)

        print(f"✅ Finished: {name} (cache={'on' if use_cache else 'off'}) → {res['out_dir']}")
        print("   onsets per label:", res.get('onset_counts', {}))
    return results
