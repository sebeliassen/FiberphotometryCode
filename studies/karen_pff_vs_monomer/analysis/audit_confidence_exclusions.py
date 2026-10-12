"""Propose Karen confidence/handling review candidates without changing exclusions.

Compares the older long-low-mean rule, camera-specific confidence and the current
KPMS default presence replay against cached native labels. All images are review
candidates; confidence is not a handling classifier. No pickle or model fit.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fpcode.behavior.observations.kinematics import presence_mask_from_conf_kpms
from fpcode.tracking.quality import confidence_streak_candidates
from fpcode.viz.event_clips import read_native_frame


def scan_recording(metadata, args):
    meta = SimpleNamespace(**metadata)
    path = args.fused_dir / f'{meta.session_id}_tandem_21kp.h5'
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if manifest['extra']['sid'] != meta.session_id or manifest['extra']['keypoint_mode'] != '21kp':
        raise ValueError('pose source identity/keypoint mode mismatch')
    raw = pd.read_hdf(path)
    if not raw.index.is_unique or raw.index.dtype.kind not in 'iu':
        raise ValueError('tracking requires unique native frame keys')
    frames = raw.index.to_numpy(dtype=np.int64)
    if np.any(frames < 0) or np.any(frames[1:] <= frames[:-1]):
        raise ValueError('native tracking frame order is invalid')
    scorer = raw.columns.get_level_values(0).unique()
    if len(scorer) != 1:
        raise ValueError('one explicitly identified pose scorer required')
    parts = raw.columns.get_level_values(1).unique().tolist()
    dorsal = parts[:17]
    ventral = ['left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw']
    if len(parts) != 21 or parts[0] != 'nose' or set(dorsal) & set(ventral):
        raise ValueError('expected declared static tandem21 body/paw schema')
    confidence = raw.loc[:, [(scorer[0], part, 'likelihood') for part in parts]].to_numpy(np.float32)
    clean = np.where(np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1), confidence, 0.)
    columns = {part: i for i, part in enumerate(parts)}
    scores = {'all21_mean': clean.mean(axis=1), 'dorsal17_mean': clean[:, [columns[p] for p in dorsal]].mean(axis=1),
              'ventral4_mean': clean[:, [columns[p] for p in ventral]].mean(axis=1),
              'dorsal_core_mean': clean[:, [columns[p] for p in ('nose', 'neck', 'mid_back', 'mouse_center', 'tail_base')]].mean(axis=1)}
    with np.load(args.atlas / 'frame_cache' / f'{meta.session_id}.npz', allow_pickle=False) as cache:
        cache_frames, labels, fps = cache['frames'], cache['labels'], float(cache['fps'])
        if fps != float(meta.fps) or np.any(cache_frames[1:] <= cache_frames[:-1]):
            raise ValueError('cache native clock/rate mismatch')
        rows = np.searchsorted(cache_frames, frames)
        inside = rows < len(cache_frames)
        inside[inside] &= cache_frames[rows[inside]] == frames[inside]
        modelled = np.zeros(len(frames), dtype=bool)
        modelled[inside] = labels[rows[inside]] >= 0
        first = int(cache_frames[np.flatnonzero(labels >= 0)[0]])
    # Replay a declared current policy; provenance does not establish that these
    # exact defaults produced the historical V2 preprocessing artifact.
    presence = presence_mask_from_conf_kpms(confidence)
    proposals = []
    for metric, score in scores.items():
        for bridge in (0., .5):
            calls = confidence_streak_candidates(score, frames, fps=fps, threshold=.8,
                minimum_seconds=15., bridge_seconds=bridge)
            for call in calls:
                a, b = np.searchsorted(frames, [call['start_frame'], call['end_frame_exclusive']])
                proposals.append({**metadata, **call, 'metric': metric, 'threshold': .8,
                    'minimum_seconds': 15., 'bridge_seconds': bridge, 'first_modelled_frame': first,
                    'modelled_frames_in_candidate': int(modelled[a:b].sum()),
                    'modelled_fraction': float(modelled[a:b].mean()),
                    'review_status': 'unreviewed', 'exclusion_status': 'candidate_only'})
    np.savez_compressed(args.output / 'profiles' / f'{meta.session_id}.npz', frames=frames,
        fps=fps, modelled=modelled, default_presence=presence, **scores)
    audit = {**metadata, 'pose_rows': len(frames), 'first_modelled_frame': first,
        'modelled_pose_frames': int(modelled.sum()), 'cache_modelled_frames': int((labels >= 0).sum()),
        'modelled_default_presence_rejected': int((modelled & ~presence).sum()),
        'modelled_all21_below_08': int((modelled & (scores['all21_mean'] < .8)).sum()),
        'modelled_dorsal17_below_08': int((modelled & (scores['dorsal17_mean'] < .8)).sum()),
        'negative_confidence_cells': int((confidence < 0).sum()),
        'pose_path': str(path.resolve()), 'pose_size': path.stat().st_size, 'pose_mtime_ns': path.stat().st_mtime_ns}
    return proposals, audit


def review_examples(args, sessions, candidates, audit):
    """Mix transition, interior-dip and supported controls without group labels."""
    known = '2025-12-03T15_26_16_top_left'
    chosen = []
    def add(sid, frame, reason):
        if len(chosen) < 8 and not any(row['session_id'] == sid and row['frame_index'] == frame for row in chosen):
            chosen.append({'session_id': sid, 'frame_index': int(frame), 'selection_reason': reason})
    if known in set(sessions.session_id):
        first = int(audit.set_index('session_id').loc[known, 'first_modelled_frame'])
        add(known, first + 53, 'previous handling smoke; default presence/cache mismatch witness')
        add(known, first + round(3 * float(sessions.set_index('session_id').loc[known, 'fps'])), 'nearby transition after initial confidence rise')
    body = candidates[(candidates.metric == 'dorsal17_mean') & candidates.bridge_seconds.eq(.5)].copy()
    body['boundary_distance'] = (body.end_frame_exclusive - body.first_modelled_frame).abs()
    for event in body[body.boundary_distance < 30 * body.fps].sort_values('boundary_distance').drop_duplicates('session_id').itertuples():
        if event.session_id == known:
            continue
        add(event.session_id, max(event.start_frame, event.end_frame_exclusive - round(event.fps)), 'one second before long dorsal-confidence dip ends')
        if len(chosen) >= 5:
            break
    # An interior dropout and a ventral-only dropout help expose false positives.
    for metric in ('dorsal17_mean', 'ventral4_mean'):
        options = candidates[(candidates.metric == metric) & candidates.bridge_seconds.eq(.5)
            & (candidates.start_frame > candidates.first_modelled_frame + 600 * candidates.fps)
            & (candidates.modelled_frames_in_candidate > 0)].sort_values('duration_s', ascending=False)
        if len(options):
            event = options.iloc[0]
            add(event.session_id, int((event.start_frame + event.end_frame_exclusive) // 2), f'interior {metric} dip; possible non-handling control')
    if known in set(sessions.session_id):
        first = int(audit.set_index('session_id').loc[known, 'first_modelled_frame'])
        add(known, first + round(120 * float(sessions.set_index('session_id').loc[known, 'fps'])), 'supported later behaviour control')
    import cv2
    destination = args.output / 'review'
    destination.mkdir(exist_ok=True)
    tiles, roster = [], []
    sessions = sessions.set_index('session_id')
    for number, event in enumerate(chosen, 1):
        meta = sessions.loc[event['session_id']]
        video = args.video_dir / Path(meta.video_path).name
        image = read_native_frame(video, frame_index=event['frame_index'], fps=float(meta.fps))
        name = f'H{number}'
        cv2.imwrite(str(destination / f'{name}.png'), image)
        image = cv2.resize(image, (404, 404))
        banner = np.zeros((44, 404, 3), dtype=np.uint8)
        cv2.putText(banner, f'{name} | native frame {event["frame_index"]}', (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.putText(banner, event['session_id'], (8, 36), cv2.FONT_HERSHEY_SIMPLEX, .42, (210,210,210), 1)
        tiles.append(np.concatenate([banner, image]))
        with np.load(args.output / 'profiles' / f'{event["session_id"]}.npz', allow_pickle=False) as profile:
            row = int(np.searchsorted(profile['frames'], event['frame_index']))
            if row >= len(profile['frames']) or profile['frames'][row] != event['frame_index']:
                raise ValueError('review frame has no exact confidence record')
            values = {key: float(profile[key][row]) for key in ('all21_mean', 'dorsal17_mean', 'ventral4_mean', 'dorsal_core_mean')}
            values.update(modelled=bool(profile['modelled'][row]), default_presence=bool(profile['default_presence'][row]))
        roster.append({'candidate_id':name, **event, **values, 'source_video':str(video.resolve()),
                       'fps':float(meta.fps), 'review_label':'pending', 'image_path':str(destination/f'{name}.png')})
    for start in range(0, len(tiles), 4):
        batch = tiles[start:start+4]
        while len(batch)<4:
            batch.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(batch[:2],axis=1),np.concatenate(batch[2:],axis=1)])
        cv2.imwrite(str(destination / f'contact_sheet_{start//4+1}.png'), sheet)
    pd.DataFrame(roster).to_csv(destination / 'review_candidates.csv',index=False)
    return roster


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--fused-dir',type=Path,required=True)
    parser.add_argument('--video-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--ram-cap-mb',type=int,default=1536)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    sessions=pd.read_csv(args.atlas/'sessions.csv',dtype={'animal_id':str})
    if args.workers<1 or args.ram_cap_mb<1:
        parser.error('worker/resource bounds must be positive')
    sizes=[(args.fused_dir/f'{sid}_tandem_21kp.h5').stat().st_size for sid in sessions.session_id]
    allowance=4*max(sizes)/1024**2+128
    workers=min(args.workers,int(args.ram_cap_mb/allowance))
    if workers<1:
        raise ValueError('RAM budget cannot support one recording worker')
    print(f'Confidence audit: {len(sessions)} recordings, {workers} workers, 15-second dips at .8; no exclusions applied',flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('choose a new output directory to preserve previous products')
    (args.output/'profiles').mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    calls,records=[],[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=pool.map(scan_recording,sessions.to_dict('records'),repeat(args))
        for number,(events,record) in enumerate(jobs,1):
            calls.extend(events)
            records.append(record)
            if number%10==0 or number==len(sessions):
                print(f'Scanned {number}/{len(sessions)}',flush=True)
    candidates=pd.DataFrame(calls)
    audit=pd.DataFrame(records)
    candidates.to_csv(args.output/'confidence_candidates.csv',index=False)
    audit.to_csv(args.output/'recording_audit.csv',index=False)
    roster=review_examples(args,sessions,candidates,audit)
    manifest={'scope':'99 Karen recordings; confidence candidates only; no persisted exclusions changed',
        'elapsed_seconds':time.monotonic()-started,'effective_workers':workers,'ram_cap_mb':args.ram_cap_mb,
        'worker_allowance_mb':allowance,'confidence_policy':'invalid/negative likelihood -> zero support; separate cameras',
        'candidate_rule':{'threshold':.8,'minimum_seconds':15.,'bridge_seconds':[0.,.5]},
        'default_presence_replay':'current KPMS defaults; historical preprocessing mask settings not proven by artifact manifest',
        'review_candidates':len(roster),'candidate_intervals':len(candidates),
        'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),Path('src/fpcode/tracking/quality.py'),Path('src/fpcode/viz/event_clips.py')]}}
    (args.output/'audit_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Audit complete: {args.output}',flush=True)


if __name__=='__main__':
    main()
