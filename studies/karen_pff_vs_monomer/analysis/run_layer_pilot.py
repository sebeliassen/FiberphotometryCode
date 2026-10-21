"""Bounded held-out Karen pilot through common behavioural-unit interfaces.

Study policy only: fixed existing discovery splits/templates, recording selection,
configured routine examples and report/video choices. Reusable inference/alignment
remain in fpcode. No pickle, model fitting or backend initialization is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from fpcode.adapters.behavior_sessions import pose_from_dlc_table, verify_result_projection
from fpcode.behavior.analysis.execution import measure_occurrence_execution, measure_phase_profiles
from fpcode.behavior.analysis.organisation import metric_values
from fpcode.behavior.analysis.unit_views import measure_unit_organisation, organisation_view
from fpcode.behavior.assemblies.grammar import GrammarAssemblyComposer
from fpcode.behavior.episodes.grammar import GrammarEpisodeDetector
from fpcode.behavior.episodes.staged import StagedEpisodeModel
from fpcode.behavior.observations.representations import (
    append_landmark_centroid,
    extract_associated_trajectory_features,
    robust_position_bounds,
)
from fpcode.behavior.primitives.resolutions import remap_states
from fpcode.behavior.unit_adapters import from_frame_labels, occurrence_table
from fpcode.contracts import SessionKey, SignalMetadata, SignalSeries, UnitType
from fpcode.contracts.behavior_results import FeatureAssociation
from fpcode.contracts.composition import GrammarRule, GrammarSpecification, RepeatRule, SequenceRule, TerminalMatch
from fpcode.viz.event_clips import render_event_clip
from fpcode.workflows.behavior_layers import compare_frozen_episode_methods
from fpcode.workflows.sequence_atlas import validate_animal_splits

if __package__:
    from .karen_hierarchy import build_hierarchy
else:
    from karen_hierarchy import build_hierarchy

CONDITIONS = ('3w_PI', '16w_PI', 'L_dopa')
LEVELS = ('L1', 'L2', 'L3')
GROUPS = ('Monomer', 'PFF')
VENTRAL_PAWS = ('left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw')
CAMERA_CORNERS = {'top_left': ('bottom_right', 'top_left'), 'bottom_left': ('top_right', 'bottom_left'),
                  'bottom_right': ('top_left', 'bottom_right'), 'top_right': ('bottom_left', 'top_right')}


def select_pilot_sessions(sessions, splits, *, animals_per_group=2, conditions=CONDITIONS):
    """Choose held-out mice by wave coverage, then ID; never by behaviour/effect."""
    roles = validate_animal_splits(sessions, splits)
    if isinstance(animals_per_group, bool) or not isinstance(animals_per_group, int) or animals_per_group < 1:
        raise ValueError('animals_per_group must be a positive integer')
    selected = []
    for group in GROUPS:
        eligible = sessions[(sessions.group == group) & sessions.animal_id.astype(str).map(roles).eq('test')]
        coverage = eligible[eligible.condition.isin(conditions)].groupby('animal_id').condition.nunique()
        candidates = sorted(coverage.index, key=lambda animal: (-coverage.loc[animal], str(animal)))
        if len(candidates) < animals_per_group:
            raise ValueError(f'not enough frozen held-out animals in {group}')
        for animal in candidates[:animals_per_group]:
            for condition in conditions:
                choices = eligible[(eligible.animal_id == animal) & (eligible.condition == condition)]
                if choices.empty:
                    raise ValueError(f'selected animal {animal} lacks pilot wave {condition}')
                selected.append(choices.sort_values('session_id').iloc[0])
    result = pd.DataFrame(selected).reset_index(drop=True)
    result['split'] = result.animal_id.astype(str).map(roles)
    return result


def configured_routines(catalog, annotations, *, limit=2):
    """First interpretable frozen exact motifs, excluding declared unresolved types."""
    labels = annotations.set_index('syllable_id')
    motifs, definitions, inventories = {}, [], set()
    for row in catalog[catalog.method.eq('exact')].itertuples():
        tokens = tuple(str(int(token)) for token in row.tokens.split())
        categories = [str(labels.loc[int(token), 'category']) for token in tokens]
        unresolved = ('Unassigned', 'Dump', 'Other')
        flags = [bool(labels.loc[int(token), 'unresolved']) for token in tokens] if 'unresolved' in labels else []
        if any(flags) or any(category.startswith(unresolved) for category in categories):
            continue
        inventory = frozenset(tokens)
        if inventory in inventories:
            continue
        inventories.add(inventory)
        motifs[row.chain_id] = tokens
        definitions.append({'routine_id': row.chain_id, 'tokens': tokens,
                            'descriptions': [str(labels.loc[int(token), 'description']) for token in tokens],
                            'selection': 'frozen training-catalogue order; exclude unresolved components'})
        if len(motifs) >= limit:
            break
    if not motifs:
        raise ValueError('no interpretable frozen motif for configured routine examples')
    return motifs, definitions


def staged_for_motif(catalog, name, tokens, *, duration_support):
    """Declared illustrative HSMM; no parameter learning or effect-based tuning."""
    phases = tuple(f'phase_{i}' for i in range(len(tokens)))
    vocabulary = tuple(unit.type_id for unit in catalog)
    phase_emissions = {
        phase: {symbol: .03 / len(vocabulary) + (.97 if symbol == token else 0.) for symbol in vocabulary}
        for phase, token in zip(phases, tokens)
    }
    duration = .05 * .95 ** np.arange(duration_support)
    duration /= duration.sum()
    return StagedEpisodeModel.create_linear_routine(
        model_id=f'configured_hsmm_{name}', routine_name=name, phase_names=phases,
        primitive_catalog=catalog, phase_emissions=phase_emissions,
        background_emissions={symbol: 1. / len(vocabulary) for symbol in vocabulary},
        phase_durations={phase: np.array([.8, .15, .05]) for phase in phases},
        background_duration=duration,
    )


def _fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}


def _geometry_descriptor(meta, args):
    import cv2

    path = args.fused_dir / f'{meta.session_id}_tandem_21kp.h5'
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if manifest['extra']['sid'] != meta.session_id or manifest['extra']['calibration_status'] != 'VALID_CRYSTAL':
        raise ValueError('foreign or invalid fused-pose manifest')
    video = args.video_dir / Path(meta.video_path).name
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - float(meta.fps)) > .001:
            raise ValueError('selected native video cannot be verified against recording FPS')
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 2 or height < 2:
            raise ValueError('invalid selected video dimensions')
    finally:
        capture.release()
    key = f'{meta.chamber}/{manifest["config"]["epoch"]}/{width}x{height}'
    return key, (0., float(width - 1), 0., float(height - 1))


def training_geometry_proxies(sessions, selected, splits, args):
    """Freeze robust envelopes from balanced training animals only, by camera setup."""
    roles = validate_animal_splits(sessions, splits)
    geometry, selected_keys = {}, {}
    for meta in selected.itertuples(index=False):
        key, screen = _geometry_descriptor(meta, args)
        selected_keys[meta.session_id] = key
        geometry[key] = {'screen_bounds': screen, 'reference_recordings': []}
    references = {key: {group: [] for group in GROUPS} for key in geometry}
    for meta in sessions.sort_values('session_id').itertuples(index=False):
        if roles[str(meta.animal_id)] != 'train' or meta.condition not in CONDITIONS:
            continue
        key, _ = _geometry_descriptor(meta, args)
        if key in references:
            references[key][meta.group].append(meta)
    for key, groups in references.items():
        eligible = {group: list({row.animal_id: row for row in reversed(rows)}.values()) for group, rows in groups.items()}
        n = min(2, *(len(rows) for rows in eligible.values()))
        if not n:
            raise ValueError(f'no balanced training-animal support for geometry proxy {key}')
        samples = []
        for group in GROUPS:
            for meta in sorted(eligible[group], key=lambda row: row.session_id)[:n]:
                if roles[str(meta.animal_id)] != 'train':
                    raise ValueError('geometry source must be a training animal')
                path = args.fused_dir / f'{meta.session_id}_tandem_21kp.h5'
                raw = pd.read_hdf(path)
                with np.load(args.atlas / 'frame_cache' / f'{meta.session_id}.npz', allow_pickle=False) as cache:
                    frames, labels = cache['frames'], cache['labels']
                pose = pose_from_dlc_table(raw, frames=frames, bodyparts=('nose',),
                    clock='video', unit='px', coordinate_frame='cropped_topcam', invalid_confidence_policy='mask')
                positions = pose.coordinates[:, 0]
                xmin, xmax, ymin, ymax = geometry[key]['screen_bounds']
                supported = ((labels >= 0) & pose.valid[:, 0] & (pose.confidence[:, 0] >= .95)
                    & (positions[:, 0] >= xmin) & (positions[:, 0] <= xmax)
                    & (positions[:, 1] >= ymin) & (positions[:, 1] <= ymax))
                indices = np.flatnonzero(supported)
                if len(indices) < 100:
                    raise ValueError('insufficient supported training nose positions for geometry proxy')
                indices = indices[np.linspace(0, len(indices) - 1, min(4000, len(indices)), dtype=int)]
                samples.append(positions[indices])
                geometry[key]['reference_recordings'].append({'session_id': meta.session_id,
                    'animal_id': str(meta.animal_id), 'group': group, 'split': roles[str(meta.animal_id)],
                    'samples': len(indices), 'pose': _fingerprint(path)})
        geometry[key]['robust_bounds'] = robust_position_bounds(np.concatenate(samples), quantiles=(.005, .995))
        geometry[key]['policy'] = 'pooled training nose positions; equal recordings per group; 0.5/99.5 percentiles'
    return geometry, selected_keys


def _read_recording(meta, args, geometry):
    cache = args.atlas / 'frame_cache' / f'{meta.session_id}.npz'
    with np.load(cache, allow_pickle=False) as raw:
        frames, labels, fps = raw['frames'], raw['labels'], float(raw['fps'])
        if fps != float(meta.fps) or len(frames) != int(meta.n_frames):
            raise ValueError('cache rate/length disagrees with selected recording manifest')
        if np.any(frames[1:] <= frames[:-1]) or np.any(frames < 0):
            raise ValueError('cache native frames must be nonnegative and strictly increasing')
        if 'exported_static_21kp_v2_syllable' in raw:
            exported = raw['exported_static_21kp_v2_syllable']
            if not np.array_equal(np.where(np.isfinite(exported), exported, -1), labels):
                raise ValueError('cached selected-model label exports disagree')
        supported = np.flatnonzero(labels >= 0)
        if not len(supported):
            raise ValueError('selected recording contains no modelled frames')
        first = int(frames[supported[0]]) + round(args.start_offset_seconds * fps)
        chosen = (frames >= first) & (frames < first + round(args.window_seconds * fps))
        frames, labels = frames[chosen].copy(), labels[chosen].copy()
    paths = json.loads((args.atlas / 'input_paths.json').read_text())
    with h5py.File(paths['results'], 'r') as result, np.load(paths['vfi'], allow_pickle=False) as mappings:
        keys = [key for key in result if key == meta.session_id or key.startswith(meta.session_id + '_seg')]
        if not keys:
            raise ValueError('selected recording has no named-model result')
        expected, native = {}, {}
        for key in keys:
            vfi = mappings[key]
            use = (vfi >= frames[0]) & (vfi <= frames[-1])
            native[key] = vfi[use]
            expected[key] = {'syllable': result[key]['syllable'][:][use]}
        verification = verify_result_projection({'frames': frames, 'labels': labels}, expected, native)
    pose_path = args.fused_dir / f'{meta.session_id}_tandem_21kp.h5'
    manifest_path = Path(str(pose_path) + '.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['extra']['sid'] != meta.session_id or manifest['kind'] != 'fused_static_21kp'
            or manifest['extra']['keypoint_mode'] != '21kp'
            or manifest['extra']['calibration_status'] != 'VALID_CRYSTAL'):
        raise ValueError('fused pose manifest has foreign recording/method or invalid camera calibration')
    if pose_path.stat().st_size != manifest['artifact']['size']:
        raise ValueError('fused pose file size differs from its manifest')
    # Fixed-format DLC HDF must be loaded as a table; bounded recording workers
    # cap the resulting RAM. Original integer index is preserved by the adapter.
    raw_pose = pd.read_hdf(pose_path)
    pose = pose_from_dlc_table(raw_pose, frames=frames, bodyparts=('nose', 'tail_base', *VENTRAL_PAWS),
                               clock='video', unit='px', coordinate_frame='cropped_topcam',
                               invalid_confidence_policy='mask')
    pose = append_landmark_centroid(pose, bodyparts=VENTRAL_PAWS, name='ventral_paw_center',
                                    confidence_threshold=.8, minimum_count=2)
    arena_bounds = None
    if args.arena_geometry is not None:
        geometry = json.loads(args.arena_geometry.read_text())[meta.session_id]
        if geometry['unit'] != pose.unit or geometry['coordinate_frame'] != pose.coordinate_frame:
            raise ValueError('arena calibration coordinates/units disagree with pose')
        arena_bounds = tuple(geometry['bounds'])
    session = SessionKey(meta.session_id, str(meta.animal_id), 'karen_pff_vs_monomer')
    association = extract_associated_trajectory_features(
        pose, session=SessionKey(meta.session_id, str(meta.animal_id), 'karen_pff_vs_monomer'),
        fps=fps, confidence_threshold=.8, smooth_window=5, arena_bounds=arena_bounds,
        arena_landmarks=('nose', 'ventral_paw_center') if arena_bounds is not None else (),
        arena_corner_features=arena_bounds is not None,
    )
    if arena_bounds is None:
        variants = []
        for proxy in ('screen', 'robust'):
            features = extract_associated_trajectory_features(pose, session=session,
                fps=fps, confidence_threshold=.8, smooth_window=5,
                arena_bounds=geometry[f'{proxy}_bounds'], arena_landmarks=('nose', 'ventral_paw_center'),
                arena_corner_features=True).series
            variants.append((proxy, features))
        values = np.column_stack([association.series.values, *(features.values[:, 5:] for _, features in variants)])
        valid = np.column_stack([association.series.valid, *(features.valid[:, 5:] for _, features in variants)])
        signals = association.series.signals + tuple(SignalMetadata(f'{proxy}_{signal.name}', signal.unit)
            for proxy, features in variants for signal in features.signals[5:])
        association = FeatureAssociation(SignalSeries(pose.time, values, signals, valid), session,
            pose.coordinate_frame, provenance=json.dumps({'operation': 'trajectory_features_with_arena_proxies',
                'screen_bounds': geometry['screen_bounds'], 'robust_bounds': geometry['robust_bounds'],
                'references': geometry['reference_recordings'], 'confidence_threshold': .8,
                'smooth_window': 5, 'distance_references': ['body_midpoint', 'nose', 'ventral_paw_center'],
                'ventral_paw_center_minimum_count': 2}))
    # Camera-relative names are study policy. Keep all four original corner
    # channels as well; no parallax correction is inferred from these distances.
    near, far = CAMERA_CORNERS[meta.chamber]
    extra_values, extra_valid, extra_signals = [], [], []
    for i, signal in enumerate(association.series.signals):
        for camera_label, corner in (('camera_near', near), ('camera_far', far)):
            suffix = f'distance_{corner}_corner'
            if signal.name.endswith(suffix):
                extra_values.append(association.series.values[:, i])
                extra_valid.append(association.series.valid[:, i])
                extra_signals.append(SignalMetadata(signal.name[:-len(suffix)] + f'distance_{camera_label}_corner', 'px'))
    ventral_count = (pose.valid[:, 2:6] & (pose.confidence[:, 2:6] >= .8)).sum(axis=1)
    extra_values.append(ventral_count)
    extra_valid.append(np.ones(len(frames), dtype=bool))
    extra_signals.append(SignalMetadata('ventral_paw_support_count', 'count'))
    series = association.series
    association = replace(association, series=SignalSeries(series.time,
        np.column_stack([series.values, *extra_values]), series.signals + tuple(extra_signals),
        np.column_stack([series.valid, *extra_valid])))
    audit = {**vars(meta), **verification, 'window_start_frame': int(frames[0]),
             'window_stop_frame': int(frames[-1]) + 1, 'window_rows': len(frames),
             'supported_pose_frames': int(pose.valid.all(axis=1).sum()),
             'camera_near_corner': near, 'camera_far_corner': far,
             'ventral_at_least_two_paws_fraction': float((ventral_count >= 2).mean()),
             'spatial_distance_status': 'calibrated' if arena_bounds is not None else 'screen_and_training_envelope_proxies',
             'spatial_support_fraction': {signal.name: float(association.series.valid[:, i].mean())
                for i, signal in enumerate(association.series.signals) if 'distance_' in signal.name},
             'cache': _fingerprint(cache), 'pose': _fingerprint(pose_path),
             'pose_manifest': _fingerprint(manifest_path)}
    return frames, labels, association, audit, pose


def analyse_recording(meta, args, choices, geometry):
    frames, labels, association, audit, pose = _read_recording(meta, args, geometry)
    identity = {key: getattr(meta, key) for key in ('session_id', 'animal_id', 'group', 'condition', 'split', 'chamber')}
    tables = {name: [] for name in ('units', 'execution', 'organisation', 'runs', 'derivations', 'phases', 'phase_profiles')}
    for level, choice in choices.items():
        remapped = remap_states(labels, choice['mapping'])
        primitives = from_frame_labels(frames, remapped, session=association.session,
                                       model_id=f'kpms_static21kp/{level}', clock='video', fps=float(meta.fps),
                                       catalog=choice['catalog'])
        staged = tuple(staged_for_motif(primitives.catalog, name, tokens, duration_support=args.max_bouts)
                       for name, tokens in choice['routines'].items())
        result = compare_frozen_episode_methods(
            primitives, association, exact_motifs=choice['motifs'],
            seqnmf_templates=choice['w'], seqnmf_type_ids=tuple(u.type_id for u in primitives.catalog),
            seqnmf_thresholds=choice['thresholds'], seqnmf_penalty=choice['penalty'],
            grammar=choice['grammar'], staged_models=staged,
            max_bouts=args.max_bouts, memory_budget_bytes=args.worker_memory_mb * 1024 * 1024,
        )
        grammar_stream = result['results']['grammar'].stream
        # Reusable two-action parts are separate episode definitions. Compose
        # those parts, rather than hoping two A-B-A windows can touch after RLE.
        parts = replace(grammar_stream, occurrences=tuple(o for o in grammar_stream.occurrences
                                                         if o.type_id.startswith('part_')))
        _, part_audit = organisation_view(parts, overlap_policy='score_ranked')
        selected_ids = set(part_audit.loc[part_audit.selected, 'occurrence_id'])
        exclusive = replace(parts, occurrences=tuple(o for o in parts.occurrences if o.occurrence_id in selected_ids))
        assembly = choice['assembly'].compose(exclusive, memory_budget_bytes=args.worker_memory_mb * 1024 * 1024)
        result['results']['assembly'] = replace(assembly, features=association)
        for method, payload in result['results'].items():
            ids = {**identity, 'resolution': level, 'method': method}
            overlap = 'reject' if method == 'primitives' or method.startswith('staged/') else 'score_ranked'
            base = measure_unit_organisation(payload.stream, gap_seconds=0., overlap_policy=overlap)
            calls = occurrence_table(payload.stream)
            audit_view = base['selection_audit'].set_index('occurrence_id')
            calls['selected_for_organisation'] = calls.occurrence_id.map(audit_view.selected).astype(bool)
            tables['units'].append(calls.assign(**ids))
            for gap in (0., .5, 2., 5.):
                org = base if gap == 0. else measure_unit_organisation(payload.stream, gap_seconds=gap, overlap_policy=overlap)
                tables['organisation'].append(pd.DataFrame([{**ids, 'gap_seconds': gap,
                                                            **org['counts'], **metric_values(org['counts'])}]))
                tables['runs'].append(org['runs'].assign(**ids, gap_seconds=gap))
            features = [signal.name for signal in association.series.signals
                        if signal.name in ('speed', 'angular_velocity') or any(tag in signal.name for tag in
                            ('distance_nearest_', 'distance_center', 'distance_camera_'))]
            for feature in features:
                execution = measure_occurrence_execution(payload, feature_name=feature)
                execution['selected_for_organisation'] = execution.occurrence_id.map(audit_view.selected).astype(bool)
                tables['execution'].append(execution.assign(**ids, feature=feature))
            for derivation in payload.derivations or ():
                tables['derivations'].append(pd.DataFrame([{**ids, 'parent_occurrence_id': derivation.parent_occurrence_id,
                    'rule_id': derivation.rule_id, 'start_frame': derivation.interval.start,
                    'end_frame_exclusive': derivation.interval.stop, 'truncated': derivation.truncated,
                    'children': json.dumps([{'model_id': ref.model_id, 'occurrence_id': ref.occurrence_id}
                                            for ref in derivation.child_references]), 'trace': json.dumps(derivation.trace)}]))
            if payload.phases:
                tables['phases'].append(pd.DataFrame([{**ids, 'parent_occurrence_id': phase.parent_occurrence_id,
                    'phase_id': phase.phase_id, 'order': phase.order, 'start_frame': phase.interval.start,
                    'end_frame_exclusive': phase.interval.stop} for phase in payload.phases]))
                tables['phase_profiles'].append(measure_phase_profiles(payload, feature_name='speed', n_bins=5).assign(**ids))
    target = args.output / 'frame_features' / f'{meta.session_id}.npz'
    np.savez_compressed(target, frames=frames, labels=labels, features=association.series.values,
                        valid=association.series.valid, names=[s.name for s in association.series.signals],
                        units=[s.unit for s in association.series.signals], fps=float(meta.fps),
                        pose=pose.coordinates, confidence=pose.confidence, bodyparts=pose.bodyparts,
                        ventral_paw_support_count=(pose.valid[:, 2:6] & (pose.confidence[:, 2:6] >= .8)).sum(axis=1))
    import resource
    audit['worker_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    audit['worker_pid'] = os.getpid()
    return tables, audit


def _render_examples(args, sessions, units):
    """Small deterministic native-frame clip/trace gallery; no behavioural approval claim."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination = args.output / 'examples'
    destination.mkdir(exist_ok=True)
    meta = sessions.set_index('session_id')
    rows, cards = [], []
    eligible = units[units.selected_for_organisation.eq(True) & ~units.method.eq('primitives')].copy()
    eligible['duration_s'] = (eligible.end_frame_exclusive - eligible.start_frame) / eligible.session_id.map(meta.fps)
    eligible = eligible[eligible.duration_s.between(.1, 8.)]
    # One early occurrence per primary method/resolution/group, plus configured
    # model examples at L3. No effect-size-based or score-based cherry-picking.
    wanted = eligible[eligible.method.isin(('exact', 'seqnmf'))]
    configured = eligible[(eligible.resolution == 'L3') & ~eligible.method.isin(('exact', 'seqnmf'))]
    selected = pd.concat([wanted, configured]).sort_values(['session_id', 'start_frame']).groupby(
        ['resolution', 'method', 'group'], sort=True).head(1).head(args.max_clips)
    for index, event in enumerate(selected.itertuples()):
        recording = meta.loc[event.session_id]
        video = args.video_dir / Path(recording.video_path).name
        stem = f'{index:02d}_{event.resolution}_{event.method.replace("/", "_")}_{event.group}'
        caption = f'{event.resolution} {event.method} {event.type_id} | {event.group} mouse {event.animal_id} | unreviewed'
        rendered = render_event_clip(video, destination / f'{stem}.mp4', start_frame=int(event.start_frame),
            end_frame_exclusive=int(event.end_frame_exclusive), fps=float(recording.fps), caption=caption)
        with np.load(args.output / 'frame_features' / f'{event.session_id}.npz', allow_pickle=False) as f:
            times = (f['frames'] - event.start_frame) / float(recording.fps)
            display = (times >= -.5) & (times <= event.duration_s + .5)
            fig, ax = plt.subplots(figsize=(7, 2.5))
            speed_column = list(f['names']).index('speed')
            speed = np.where(f['valid'][:, speed_column], f['features'][:, speed_column], np.nan)
            ax.plot(times[display], speed[display], color='black')
            ax.axvspan(0., event.duration_s, color='steelblue', alpha=.2)
            ax.set(xlabel='Seconds relative to detected onset', ylabel='Midpoint speed (px/s)', title=caption)
            fig.tight_layout()
            fig.savefig(destination / f'{stem}.png', dpi=140)
            plt.close(fig)
        rows.append({**event._asdict(), **rendered, 'source_video': str(video.resolve()),
                     'trace_path': str(destination / f'{stem}.png'), 'review_status': 'unreviewed'})
        cards.append(f'<article><p>{html.escape(caption)}</p><video controls preload="none" src="{stem}.mp4"></video>'
                     f'<img src="{stem}.png" alt="Aligned speed trace"></article>')
    pd.DataFrame(rows).to_csv(destination / 'example_roster.csv', index=False)
    (destination / 'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Karen layer pilot</title>'
        '<style>body{font:16px system-ui;margin:24px}article{max-width:850px;margin:24px 0}video,img{width:100%}</style>'
        '<h1>Karen held-out layer pilot</h1><p>Deterministic examples; behavioural interpretation requires review. '
        'All clips and traces use verified native video frames.</p>' + '\n'.join(cards))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--clustering', type=Path, required=True)
    parser.add_argument('--fused-dir', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arena-geometry', type=Path)
    parser.add_argument('--window-seconds', type=float, default=60.)
    parser.add_argument('--start-offset-seconds', type=float, default=120.,
                        help='Offset from first modelled frame; earliest labels can include handling')
    parser.add_argument('--max-bouts', type=int, default=1200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker-memory-mb', type=int, default=256)
    parser.add_argument('--ram-cap-mb', type=int, default=1536)
    parser.add_argument('--max-clips', type=int, default=16)
    parser.add_argument('--max-recordings', type=int, default=12, help='Bound smoke scope after balanced selection')
    parser.add_argument('--execute', action='store_true', help='Run the explicitly bounded empirical pilot; default is read-only plan')
    args = parser.parse_args(argv)
    if (not np.isfinite(args.window_seconds) or args.window_seconds <= 0
            or not np.isfinite(args.start_offset_seconds) or args.start_offset_seconds < 0
            or min(args.workers, args.worker_memory_mb, args.ram_cap_mb, args.max_bouts, args.max_recordings) < 1 or args.max_clips < 0):
        parser.error('scope and resource bounds must be positive; max-clips may be zero')
    started = time.monotonic()
    sessions = pd.read_csv(args.atlas / 'sessions.csv', dtype={'animal_id': str})
    splits = json.loads((args.atlas / 'chains/animal_splits.json').read_text())
    selected = select_pilot_sessions(sessions, splits).head(args.max_recordings)
    raw_annotations = pd.read_csv(args.atlas / 'syllable_annotations.csv')
    declared_inputs = json.loads((args.atlas / 'input_paths.json').read_text())
    if not raw_annotations.equals(pd.read_csv(declared_inputs['labels'])):
        raise ValueError('cached annotations differ from their current declared source')
    maps, annotations, _ = build_hierarchy(raw_annotations)
    raw_catalog = pd.read_csv(args.atlas / 'chains/chain_catalog.csv')
    raw_routines, raw_definitions = configured_routines(raw_catalog, annotations['L1'])
    choices, definitions = {}, {}
    inputs = [args.atlas / name for name in ('sessions.csv', 'syllable_annotations.csv', 'input_paths.json', 'chains/animal_splits.json')]
    for level in LEVELS:
        frozen_map_path = args.clustering / level / 'mapping.json'
        if not np.array_equal(maps[level], json.loads(frozen_map_path.read_text())):
            raise ValueError('annotation crosswalk differs from the frozen template resolution')
        inputs.append(frozen_map_path)
        directory = args.atlas / 'chains' if level == 'L1' else args.clustering / level / 'chains'
        actual_split = json.loads((directory / 'animal_splits.json').read_text())
        if actual_split != splits:
            raise ValueError('resolution templates use different discovery splits')
        catalog = pd.read_csv(directory / 'chain_catalog.csv')
        motifs = {row.chain_id: tuple(str(int(t)) for t in row.tokens.split())
                  for row in catalog[catalog.method.eq('exact')].itertuples()}
        routines, definitions[level] = {}, []
        for definition in raw_definitions:
            name = definition['routine_id']
            mapped = remap_states(np.array([int(t) for t in raw_routines[name]]), maps[level])
            tokens = tuple(str(t) for t in mapped[np.r_[True, mapped[1:] != mapped[:-1]]])
            if len(tokens) < 2:
                raise ValueError('configured routine collapses to one unit at this resolution')
            anchored = f'anchor_{name}'
            routines[anchored] = tokens
            motifs[anchored] = tokens
            definitions[level].append({**definition, 'routine_id': anchored, 'raw_tokens': definition['tokens'],
                'tokens': tokens, 'descriptions': [str(annotations[level].set_index('syllable_id').loc[int(t), 'description']) for t in tokens],
                'selection': 'same interpretable frozen L1 routine projected and RLE-collapsed; no new discovery'})
        specification = GrammarSpecification(f'{level}_configured_routines', tuple(
            rule for name, tokens in routines.items() for rule in (
                GrammarRule(name, SequenceRule(tuple(TerminalMatch(t) for t in tokens))),
                GrammarRule(f'part_{name}', SequenceRule(tuple(TerminalMatch(t) for t in tokens[:2]))),
            )))
        assembly_spec = GrammarSpecification(f'{level}_bounded_repetition', tuple(
            GrammarRule(f'repeated_{name}', RepeatRule(TerminalMatch(f'part_{name}'), 2, 3)) for name in routines))
        with np.load(directory / 'seqnmf_templates.npz', allow_pickle=False) as model:
            choices[level] = {'mapping': maps[level],
                'catalog': tuple(UnitType(str(row.syllable_id), str(row.description)) for row in annotations[level].itertuples()),
                'motifs': motifs, 'routines': routines, 'w': model['w'].copy(), 'thresholds': model['thresholds'].copy(),
                'penalty': float(model['penalty']), 'grammar': GrammarEpisodeDetector(specification),
                'assembly': GrammarAssemblyComposer(assembly_spec)}
        inputs.extend([directory / name for name in ('animal_splits.json', 'chain_catalog.csv', 'seqnmf_templates.npz')])
    # Allow for the loaded fixed HDF table and copied pose/feature arrays in each
    # worker in addition to its guarded inference arrays. This is conservative
    # scheduling, not an OS-enforced hard RSS cap.
    largest_pose = max((args.fused_dir / f'{sid}_tandem_21kp.h5').stat().st_size for sid in selected.session_id)
    allowance_mb = args.worker_memory_mb + 4 * largest_pose / 1024**2 + 128
    workers = min(args.workers, max(0, int(args.ram_cap_mb / allowance_mb)))
    if workers < 1:
        raise ValueError('RAM cap is too small for one bounded recording worker')
    plan = {'recordings': len(selected), 'animals': selected.animal_id.nunique(), 'held_out_only': True,
            'conditions': CONDITIONS, 'resolutions': LEVELS, 'window_seconds': args.window_seconds,
            'start_offset_seconds': args.start_offset_seconds,
            'effective_workers': workers, 'worker_allowance_mb': allowance_mb,
            'wall_calibration': 'supplied' if args.arena_geometry else 'screen and frozen training-envelope proxies',
            'frozen_templates': True, 'configured_routines': definitions}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('choose an empty output directory; previous products are preserved')
    (args.output / 'frame_features').mkdir(parents=True, exist_ok=True)
    if args.arena_geometry is None:
        geometry, selected_keys = training_geometry_proxies(sessions, selected, splits, args)
    else:
        geometry, selected_keys = {}, {}
    (args.output / 'geometry_proxies.json').write_text(json.dumps(geometry, indent=2))
    selected.to_csv(args.output / 'sessions.csv', index=False)
    totals = {name: [] for name in ('units', 'execution', 'organisation', 'runs', 'derivations', 'phases', 'phase_profiles')}
    audit = []
    jobs_metadata = [SimpleNamespace(**row) for row in selected.to_dict('records')]
    worker_geometry = [geometry[selected_keys[meta.session_id]] if selected_keys else {} for meta in jobs_metadata]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = pool.map(analyse_recording, jobs_metadata, repeat(args), repeat(choices), worker_geometry)
        for index, (tables, record) in enumerate(jobs):
            for name in totals:
                totals[name].extend(tables[name])
            audit.append(record)
            print(f'Analysed {index + 1}/{len(selected)}: {record["session_id"]}', flush=True)
    merged = {}
    for name, tables in totals.items():
        nonempty = [table for table in tables if len(table)]
        merged[name] = pd.concat(nonempty, ignore_index=True) if nonempty else pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        merged[name].to_csv(args.output / f'{name}.csv', index=False)
    clips = _render_examples(args, selected, merged['units']) if args.max_clips else []
    from threadpoolctl import threadpool_info
    code_root = Path(__file__).resolve().parents[3]
    manifest = {**plan, 'elapsed_seconds': time.monotonic() - started, 'animal_splits': splits,
                'input_files': [_fingerprint(path) for path in inputs], 'recording_audit': audit,
                'geometry_proxies': geometry, 'examples': len(clips), 'threads': {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                'observed_numerical_pools': threadpool_info(),
                'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__).resolve(), *(code_root / name for name in (
                        'src/fpcode/workflows/behavior_layers.py', 'src/fpcode/adapters/behavior_sessions.py',
                        'src/fpcode/behavior/observations/representations.py', 'src/fpcode/workflows/sequence_atlas.py')))},
                'limitations': ['configured HSMM parameters are illustrative, not learned',
                    'HSMM dwell support is in primitive-event counts, not seconds',
                    'short first-modelled windows are not full-session estimates',
                    'declared 120-second start offset avoids observed initial handling; sparse video QC is not exhaustive annotation',
                    'no inferential PFF/monomer test on this small selected pilot',
                    'camera registration and arena proxies are not physical wall calibration',
                    'training-envelope bounds depend on training exploration; screen bounds include visible walls',
                    'chamber-specific camera-near/far corners are retained; distances do not correct height-dependent parallax',
                    'tandem21 supplies dorsal body points and ventral paws, not independent dual-camera nose estimates',
                    'raw calls preserve ambiguity; organisation uses explicit score-ranked exclusive views',
                    'grammar acceptance is not observed physical completion',
                    'video examples have not received expert behavioural review']}
    (args.output / 'pilot_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Pilot complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
