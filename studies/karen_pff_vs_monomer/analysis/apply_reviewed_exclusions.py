"""Build a separate atlas-cache view from explicitly reviewed native intervals.

Unchanged recordings are linked to the selected source cache; reviewed recordings
are copied and masked. This never installs a production mask or changes a model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.workflows.behavior_atlas import mask_frame_cache_intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--reviewed-exclusions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-visual-review', action='store_true',
                        help='Also accept individually inspected agent visual intervals with saved evidence')
    args = parser.parse_args(argv)
    policy = pd.read_csv(args.reviewed_exclusions)
    required = {'session_id', 'start_frame', 'end_frame_exclusive', 'review_status', 'reason'}
    accepted = {'user_confirmed_interval'}
    if args.allow_visual_review:
        accepted.add('agent_visual_review_interval')
    if not required.issubset(policy.columns) or not policy.review_status.isin(accepted).all():
        raise ValueError('only explicitly confirmed native intervals may be applied')
    visual = policy[policy.review_status.eq('agent_visual_review_interval')]
    if len(visual) and ('evidence_path' not in visual or
                        not visual.evidence_path.map(lambda value: Path(str(value)).is_file()).all()):
        raise ValueError('visual intervals require saved evidence paths')
    sessions = pd.read_csv(args.atlas / 'sessions.csv')
    if not set(policy.session_id).issubset(set(sessions.session_id)):
        raise ValueError('policy recording missing from selected atlas')
    for name in ('start_frame', 'end_frame_exclusive'):
        if policy[name].dtype.kind not in 'iu':
            raise ValueError('native interval boundaries must be integers')
    if (policy.start_frame < 0).any() or (policy.end_frame_exclusive <= policy.start_frame).any():
        raise ValueError('native intervals must be nonnegative half-open pairs')
    print(json.dumps({'sessions': len(sessions), 'reviewed_intervals': len(policy),
                      'status': 'execute' if args.execute else 'plan'}), flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('choose a new analysis-view directory')
    destination = args.output / 'frame_cache'
    destination.mkdir(parents=True, exist_ok=True)
    coverage = []
    for meta in sessions.itertuples():
        source = args.atlas / 'frame_cache' / f'{meta.session_id}.npz'
        if not source.is_file():
            raise ValueError('missing source frame cache: ' + str(source))
        target = destination / source.name
        intervals = policy[policy.session_id == meta.session_id]
        if intervals.empty:
            target.symlink_to(source.resolve())
            continue
        with np.load(source, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded}
        masked = mask_frame_cache_intervals(arrays, intervals[['start_frame', 'end_frame_exclusive']].to_numpy(),
            observation_fields=('centroid', 'heading', 'latent', 'exported_velocity_px_s'),
            label_fields=('labels', 'exported_static_21kp_v2_syllable'))
        if not np.array_equal(masked['frames'], arrays['frames']):
            raise ValueError('native frame identity changed')
        removed = int(((arrays['labels'] >= 0) & (masked['labels'] < 0)).sum())
        np.savez_compressed(target, **masked)
        coverage.append({'session_id': meta.session_id, 'removed_modelled_frames': removed,
                         'removed_seconds': removed / float(arrays['fps']),
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    sessions.to_csv(args.output / 'sessions.csv', index=False)
    policy.to_csv(args.output / 'reviewed_exclusions.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.output / 'mask_coverage.csv', index=False)
    (args.output / 'manifest.json').write_text(json.dumps({
        'source_atlas': str(args.atlas.resolve()), 'policy': str(args.reviewed_exclusions.resolve()),
        'masked_recordings': len(coverage), 'unmodified_linked_recordings': len(sessions) - len(coverage),
        'removed_modelled_frames': sum(row['removed_modelled_frames'] for row in coverage),
        'scope': 'analysis view only; reviewed intervals; no template cleaning or production changes',
        'clock': 'native frame indices preserved; labels invalidated and observations unsupported at exclusions'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
