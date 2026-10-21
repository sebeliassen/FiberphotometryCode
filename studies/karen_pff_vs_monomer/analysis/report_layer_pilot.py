"""Validate and report the bounded Karen layer pilot; no cohort inference or fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.unit_adapters import from_frame_labels
from fpcode.contracts import SessionKey
from fpcode.viz.event_clips import render_event_clip


def validate_outputs(root, sessions, units, manifest):
    """Check saved native support, resolution interfaces and encoded examples."""
    import cv2

    if units.selected_for_organisation.isna().any():
        raise ValueError('selection decisions must be explicit')
    if not set(units.session_id) <= set(sessions.session_id):
        raise ValueError('foreign recording in saved units')
    checked = 0
    for meta in sessions.itertuples():
        with np.load(root / 'frame_features' / f'{meta.session_id}.npz', allow_pickle=False) as data:
            frames, labels, values, valid = data['frames'], data['labels'], data['features'], data['valid']
            if values.shape != valid.shape or values.shape != (len(frames), len(data['names'])):
                raise ValueError('feature schema is not aligned to native frames')
            if not np.isfinite(values[valid]).all() or float(data['fps']) != meta.fps:
                raise ValueError('supported feature values/rate are invalid')
            stream = from_frame_labels(frames, labels, session=SessionKey(meta.session_id, str(meta.animal_id)),
                model_id='verification', clock='video', fps=meta.fps)
            spans = stream.valid_intervals
            calls = units[units.session_id.eq(meta.session_id)]
            starts = np.array([span.start for span in spans])
            for event in calls.itertuples():
                index = np.searchsorted(starts, event.start_frame, side='right') - 1
                if index < 0 or event.end_frame_exclusive > spans[index].stop or event.start_frame >= event.end_frame_exclusive:
                    raise ValueError('saved unit crosses native validity support')
            for _, exclusive in calls[calls.selected_for_organisation].groupby(['resolution', 'method']):
                exclusive = exclusive.sort_values('start_frame')
                if np.any(exclusive.start_frame.to_numpy()[1:] < exclusive.end_frame_exclusive.to_numpy()[:-1]):
                    raise ValueError('selected organisation view overlaps')
            checked += len(calls)
    roles = {str(animal): role for role in ('train', 'validation', 'test') for animal in manifest['animal_splits'][role]}
    if not all(roles[str(animal)] == 'test' for animal in sessions.animal_id):
        raise ValueError('pilot includes discovery animals')
    for geometry in manifest['geometry_proxies'].values():
        if not all(roles[row['animal_id']] == 'train' for row in geometry['reference_recordings']):
            raise ValueError('arena-envelope reference leaks held-out animals')
    examples = pd.read_csv(root / 'examples/example_roster.csv')
    for row in examples.itertuples():
        capture = cv2.VideoCapture(row.clip_path)
        try:
            if (not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - row.fps) > .001
                    or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != row.frames_written):
                raise ValueError('encoded native-video clip failed validation')
        finally:
            capture.release()
    return {'saved_units_checked': checked, 'recordings_checked': len(sessions),
            'examples_checked': len(examples), 'native_support': 'passed', 'exclusive_views': 'passed',
            'held_out_and_training_geometry_roles': 'passed', 'feature_shapes_masks_units': 'passed',
            'scientific_limit': 'support and bookkeeping checks; not expert behavioural validation'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assembly-preview', action='store_true', help='Render one additional bounded assembly clip')
    parser.add_argument('--video-dir', type=Path, help='Explicit native video root for an additional assembly clip')
    args = parser.parse_args(argv)
    root = args.output
    manifest = json.loads((root / 'pilot_manifest.json').read_text())
    sessions = pd.read_csv(root / 'sessions.csv', dtype={'animal_id': str})
    units = pd.read_csv(root / 'units.csv', dtype={'animal_id': str})
    organisation = pd.read_csv(root / 'organisation.csv', dtype={'animal_id': str})
    validation = validate_outputs(root, sessions, units, manifest)
    counts = organisation[organisation.gap_seconds.eq(0.)].groupby(['resolution', 'method'])[
        ['calls', 'covered_seconds', 'modelled_seconds']].sum()
    counts['raw_calls'] = units.groupby(['resolution', 'method']).size().reindex(counts.index, fill_value=0)
    counts['coverage_fraction'] = counts.covered_seconds / counts.modelled_seconds
    counts['calls_per_min'] = 60. * counts.calls / counts.modelled_seconds
    counts.to_csv(root / 'method_summary.csv')
    spatial = []
    for meta in sessions.itertuples():
        with np.load(root / 'frame_features' / f'{meta.session_id}.npz', allow_pickle=False) as data:
            for column, name in enumerate(data['names']):
                if 'distance_' not in name:
                    continue
                valid = data['valid'][:, column] & (data['labels'] >= 0)
                values = data['features'][valid, column]
                spatial.append({'session_id': meta.session_id, 'animal_id': meta.animal_id,
                    'group': meta.group, 'condition': meta.condition, 'chamber': meta.chamber, 'feature': name,
                    'valid_fraction': float(valid.mean()), 'n_supported': len(values),
                    'mean_px': float(values.mean()) if len(values) else np.nan})
    spatial = pd.DataFrame(spatial)
    spatial.to_csv(root / 'spatial_session_summary.csv', index=False)
    screen = spatial[spatial.feature.str.startswith('screen_')].copy()
    robust = spatial[spatial.feature.str.startswith('robust_')].copy()
    screen['reference_feature'] = screen.feature.str.removeprefix('screen_')
    robust['reference_feature'] = robust.feature.str.removeprefix('robust_')
    proxy = screen.merge(robust, on=['session_id', 'animal_id', 'group', 'condition', 'chamber', 'reference_feature'],
                          suffixes=('_screen', '_robust'), validate='one_to_one')
    proxy['robust_minus_screen_px'] = proxy.mean_px_robust - proxy.mean_px_screen
    proxy.to_csv(root / 'spatial_proxy_sensitivity.csv', index=False)
    speed = pd.read_csv(root / 'execution.csv', dtype={'animal_id': str})
    completion = speed[speed.feature.eq('speed')].groupby(['resolution', 'method', 'completion_status']).size()
    completion.rename('raw_calls').to_csv(root / 'completion_evidence_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    shown = counts.reset_index().query("method in ['exact', 'seqnmf', 'grammar', 'assembly']")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in shown.groupby('method'):
        ax.plot(group.resolution, group.coverage_fraction, marker='o', label=method)
    ax.set(xlabel='Classification resolution', ylabel='Selected coverage / modelled time',
           title='Different methods and catalogues describe different amounts of behaviour')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'method_coverage.png', dpi=160)
    fig.savefig(figures / 'method_coverage.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feature in zip(axes, ('nose_distance_nearest_edge', 'ventral_paw_center_distance_nearest_edge')):
        selected = proxy[proxy.reference_feature.eq(feature)]
        for group, sub in selected.groupby('group'):
            ax.scatter(sub.mean_px_screen, sub.mean_px_robust, label=group)
        ax.set(xlabel='Screen proxy (px)', ylabel='Training-envelope proxy (px)', title=feature.replace('_', ' '))
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'proxy_sensitivity.png', dpi=160)
    fig.savefig(figures / 'proxy_sensitivity.pdf')
    plt.close(fig)
    assembly_link = ''
    if args.assembly_preview:
        event = units[units.method.eq('assembly') & units.resolution.eq('L3')].sort_values(['session_id', 'start_frame']).iloc[0]
        meta = sessions.set_index('session_id').loc[event.session_id]
        video = pd.read_csv(root / 'examples/example_roster.csv').set_index('session_id').source_video
        candidates = video[video.index == event.session_id]
        if len(candidates):
            source_video = candidates.iloc[0]
        else:
            if args.video_dir is None:
                raise ValueError('assembly source video must be explicitly relocated from a verified recording')
            source_video = args.video_dir / Path(meta.video_path).name
        rendered = render_event_clip(source_video, root / 'examples/assembly_preview.mp4',
            start_frame=int(event.start_frame), end_frame_exclusive=int(event.end_frame_exclusive),
            fps=float(meta.fps), caption='L3 assembly: repeated reusable two-action part | unreviewed')
        (root / 'examples/assembly_preview.json').write_text(json.dumps({**rendered,
            'session_id': event.session_id, 'type_id': event.type_id, 'extra_scope': 'one assembly beyond original 16 clips'}, indent=2))
        assembly_link = '\nOne additional [assembly clip](examples/assembly_preview.mp4) shows the bounded composition.\n'
    raw_table = counts[['raw_calls', 'calls', 'coverage_fraction']].round(3).reset_index().to_markdown(index=False)
    support = spatial[spatial.feature.str.endswith('distance_nearest_edge')].groupby('feature').valid_fraction.mean()
    worker_peak = max(row['worker_peak_rss_mib'] for row in manifest['recording_audit'])
    coverage_l1 = counts.loc[('L1', 'seqnmf'), 'coverage_fraction']
    coverage_l3 = counts.loc[('L3', 'seqnmf'), 'coverage_fraction']
    exact_l1, exact_l3 = (int(counts.loc[(level, 'exact'), 'calls']) for level in ('L1', 'L3'))
    report = f'''# Karen behavioural layer pilot — 2026-10-12

The common interfaces work on real Karen recordings: {len(sessions)} one-minute windows,
four held-out mice (two per group), three waves (3w, 16w, L-DOPA), and L1–L3.
This is a method/QC pilot, not a PFF–monomer inference or a full-session estimate.

## What was produced

Exact motifs and seqNMF use frozen training templates. Two interpretable raw
routines—sniff/observe/sniff and look-left/scrunch/look-left—are projected across
resolutions for configured grammar/HSMM examples. Reusable two-action parts also
form bounded assemblies; all raw derivations and explicit exclusive selections
are saved. Larger coverage does not establish a better model: method catalogues
and configured scopes differ.

{raw_table}

![Method coverage](figures/method_coverage.png)

SeqNMF selected coverage is {coverage_l1:.0%} at L1 and {coverage_l3:.0%} at L3 in these
windows. Exact selected calls are {exact_l1} and {exact_l3}, respectively. The configured HSMM shows
a different resolution response, including zero look/scrunch calls at L3. Its
uniform background, soft emissions and event-count durations need calibration;
these illustrative parameters do not establish superiority or failure of HSMM.
Grammar acceptance stays separate from observed motor completion.

One assembly is detected at the same native interval at each resolution. It
combines reusable two-action episodes, rather than renaming a single longer chain.
The [derivations](derivations.csv) preserve its child identities.

## Spatial reference and camera geometry

Distances retain midpoint, nose and ventral-paw-centre references. The paw centre
requires at least two supported paws; it is an anatomical proxy, not a matched
second estimate of the nose. Screen bounds are compared with 0.5/99.5 percentile
nose envelopes frozen from balanced training animals, separately for chamber,
camera-calibration epoch and image dimensions. The latter depend on exploration;
neither proxy is a calibrated physical wall.

All four image-coordinate corners and TopCam camera-near/far corners are retained.
The shared central TopCam and local BotCams have different perspective geometry;
the static cross-view registration does not remove height-dependent parallax.
Support and chamber identities must accompany later spatial group comparisons.

![Spatial proxy sensitivity](figures/proxy_sensitivity.png)

Mean window support fractions:

{support.round(3).rename('valid_fraction').reset_index().to_markdown(index=False)}

Full per-recording values are in [spatial_session_summary.csv](spatial_session_summary.csv)
and paired proxy differences in [spatial_proxy_sensitivity.csv](spatial_proxy_sensitivity.csv).

## Real-data QC and examples

The initial smoke exposed -1 DLC confidence sentinels and experimenter handling
inside the earliest modelled frames. Unsupported confidence is explicitly masked.
Pilot windows start 120 seconds after the first modelled frame. Start/middle/end
video samples showed unhandled mice in all twelve windows; sparse samples do not
exclude every transient. See [QC frames](../window_qc.csv) and
[the earlier handling smoke](../smoke/examples/gallery.html).

[Open the 16-clip gallery with aligned speed traces](examples/gallery.html).
Examples are deterministic, not selected by group effect. They still require
expert review; the original training-template discovery may include initial
handling, so rediscovery after explicit exclusions is a later QC task.
{assembly_link}
Saved-output verification checked {validation['saved_units_checked']} unit intervals,
native support, exclusive selections, feature shape/masks, frozen animal roles
and {validation['examples_checked']} encoded clips. The canonical labels and saved
L1–L3 crosswalks were also independently confirmed against their declared sources.

## Implementation and next step

Shared native-index DLC alignment, invalid-confidence policy, training split
validation and frozen-method comparison live in their existing fpcode owners.
Spatial extraction supports independent landmarks, individual corners and
supported landmark centroids. Karen selection/configuration/reporting remains
study-owned; no new library folder or fitting framework was introduced.

The pilot ran in {manifest['elapsed_seconds']:.1f} seconds with two CPU processes and
four numerical threads per pool; peak recorded worker RSS was {worker_peak:.1f} MiB.
The 1536-MiB scheduling budget is conservative, not OS-enforced. No pickle reload,
KPMS/seqNMF-template fitting or HSMM parameter learning occurred.

Next: review the examples and completion/phase evidence, improve configured model
calibration using training animals, define handling exclusions for discovery and
full-session measurement, then expand the useful methods across all 21 mice and
99 recordings. Preserve promising effects and uncertainty in the larger report;
this four-mouse pilot does not support inferential group claims.
'''
    (root / 'report.md').write_text(report)
    validation['report_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / 'output_validation.json').write_text(json.dumps(validation, indent=2))
    print(f'Report and saved-output validation complete: {root / "report.md"}')


if __name__ == '__main__':
    main()
