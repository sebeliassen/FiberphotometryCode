"""Render a bounded, group-blind review of confidence candidate boundaries.

This writes review media only. Midpoint judgements do not validate full intervals;
confidence loss is not an annotation of handling. Original artifacts stay intact.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.viz.event_clips import read_native_frame, render_event_clip


def render_candidate(job):
    """Render start/middle/end media on the recording's verified native clock."""
    import cv2

    destination = Path(job['output'])
    tiles, clips = [], []
    for phase, frame in [('start', job['start_frame']),
                         ('middle', (job['start_frame'] + job['end_frame_exclusive']) // 2),
                         ('end', job['end_frame_exclusive'])]:
        capture = cv2.VideoCapture(job['source_video'])
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        frame = min(count - 1, frame)
        first, last = max(0, frame - round(3 * job['fps'])), min(count, frame + round(3 * job['fps']))
        clip = render_event_clip(job['source_video'], destination / f"{job['candidate_id']}_{phase}.mp4",
                                 start_frame=first, end_frame_exclusive=last, fps=job['fps'],
                                 caption=f"{job['candidate_id']} {phase}: confidence candidate; not a handling label",
                                 pre_seconds=0, post_seconds=0, maximum_width=480,
                                 active_label='REVIEW')
        check = cv2.VideoCapture(clip['clip_path'])
        try:
            if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != last - first:
                raise ValueError('encoded clip frame count mismatch')
            if abs(check.get(cv2.CAP_PROP_FPS) - job['fps']) > .001:
                raise ValueError('encoded clip rate mismatch')
        finally:
            check.release()
        clips.append({'candidate_id': job['candidate_id'], 'phase': phase, 'anchor_frame': frame, **clip})
        snapshots = []
        for delta in (-120, 0, 120):
            native = min(count - 1, max(0, frame + delta))
            image = read_native_frame(job['source_video'], frame_index=native, fps=job['fps'])
            image = cv2.resize(image, (300, 300))
            banner = np.zeros((30, 300, 3), dtype=np.uint8)
            cv2.putText(banner, f"{job['candidate_id']} {phase} | {native}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            snapshots.append(np.concatenate([banner, image]))
        tiles.append(np.concatenate(snapshots, axis=1))
    sheet = destination / f"{job['candidate_id']}_boundaries.png"
    cv2.imwrite(str(sheet), np.concatenate(tiles))
    return clips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 2:
        raise ValueError('review scope supports one or two video workers')
    candidates = pd.read_csv(args.audit / 'proposed_exclusions.csv')
    later = candidates[(candidates.modelled_frames_in_candidate > 0)
                       & (candidates.start_frame > candidates.first_modelled_frame)]
    selected = later.sort_values('modelled_frames_in_candidate', ascending=False).drop_duplicates('session_id').head(6)
    known = '2025-12-03T15_26_16_top_left'
    initial = candidates[(candidates.session_id == known) & (candidates.start_frame <= candidates.first_modelled_frame)]
    selected = pd.concat([selected, initial.head(1)], ignore_index=True)
    sessions = pd.read_csv(args.atlas / 'sessions.csv').set_index('session_id')
    jobs = []
    for number, row in enumerate(selected.itertuples(), 1):
        video = args.video_dir / Path(sessions.loc[row.session_id, 'video_path']).name
        jobs.append({'candidate_id': f'B{number}', 'session_id': row.session_id,
                     'source_video': str(video.resolve()), 'fps': float(row.fps),
                     'start_frame': int(row.start_frame), 'end_frame_exclusive': int(row.end_frame_exclusive),
                     'modelled_overlap': int(row.modelled_frames_in_candidate), 'output': str(args.output.resolve())})
    print(json.dumps({'candidates': len(jobs), 'clips': len(jobs) * 3, 'workers': args.workers,
                      'status': 'execute' if args.execute else 'plan'}, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        clips = [clip for result in pool.map(render_candidate, jobs) for clip in result]
    pd.DataFrame(jobs).to_csv(args.output / 'review_candidates.csv', index=False)
    pd.DataFrame(clips).to_csv(args.output / 'clips.csv', index=False)
    (args.output / 'manifest.json').write_text(json.dumps({
        'scope': 'six largest later-session candidates from distinct recordings plus confirmed initial handling witness',
        'clock': 'native video frames, verified 60 FPS; no index conversion',
        'candidates': len(jobs), 'clips': len(clips), 'workers': args.workers,
        'policy': 'review only; no exclusions applied; genotype omitted from media',
        'sampling': 'start/middle/end clips of six seconds, boundary sheets at -2/0/+2 seconds'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
