"""Propose removal boundaries from large foreground changes for visual review.

This cohort prototype is a review aid, not a handling classifier or production
mask. The user identified wire-lid activity as removal. Each proposed boundary
still needs native-video inspection. Pose confidence is deliberately not the rule.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.viz.event_clips import read_native_frame


def propose(job):
    import cv2

    def gray(frame):
        image = read_native_frame(job['video'], frame_index=int(frame), fps=60.)
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (240, 240)).astype(float)
    last = int(job['last_modelled_frame'])
    baseline_frames = [max(0, last - round(seconds * 60)) for seconds in np.linspace(180, 150, 9)]
    background = np.median(np.stack([gray(frame) for frame in baseline_frames]), axis=0)
    frames = np.arange(max(0, last - 120 * 60), last + 1, 120, dtype=np.int64)
    if frames[-1] != last:
        frames = np.r_[frames, last]
    scores = []
    for frame in frames:
        image = gray(frame)
        difference = image - background
        difference -= np.median(difference)
        foreground = (np.abs(difference) > 25).astype(np.uint8)
        # Median filtering reduces thin moving tethers; large hands/lids persist.
        foreground = cv2.medianBlur(foreground, 3)
        scores.append(float(foreground.mean()))
    large = np.asarray(scores) >= .04
    stable = np.flatnonzero(large[:-2] & large[1:-1] & large[2:])
    cutoff = int(frames[stable[0]]) if len(stable) else None
    # Conservative two-second buffer before the coarse sustained-change onset.
    cutoff = max(0, cutoff - 120) if cutoff is not None else None
    destination = Path(job['output'])
    tiles = []
    anchors = [max(0, cutoff - 120), cutoff, cutoff + 120] if cutoff is not None else [last - 240, last - 120, last]
    for number, frame in enumerate(anchors, 1):
        image = read_native_frame(job['video'], frame_index=int(frame), fps=60.)
        image = cv2.resize(image, (280, 280))
        banner = np.zeros((42, 280, 3), dtype=np.uint8)
        cv2.putText(banner, f"E{job['number']}.{number} | {frame}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(banner, ['proposed end - 2s', 'proposed exclusive end', 'proposed end + 2s'][number - 1],
                    (6, 36), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1)
        tiles.append(np.concatenate([banner, image]))
    sheet = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(destination / f"E{job['number']}_boundary.png"), sheet)
    return {'number': job['number'], 'session_id': job['session_id'], 'video': job['video'],
            'last_modelled_frame': last, 'proposed_end_frame_exclusive': cutoff,
            'status': 'unreviewed_candidate' if cutoff is not None else 'no_sustained_change',
            'maximum_foreground_fraction': max(scores)}, pd.DataFrame({'number': job['number'], 'frame': frames, 'foreground_fraction': scores})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.workers not in (1, 2):
        raise ValueError('choose one or two bounded video workers')
    selected = pd.read_csv(args.final_review / 'review_candidates.csv')
    selected = selected[selected.visual_label.isin(['wire_lid_activity', 'empty_arena'])]
    jobs = [{**row._asdict(), 'output': str(args.output.resolve())} for row in selected.itertuples(index=False)]
    print(json.dumps({'candidate_recordings': len(jobs), 'status': 'execute' if args.execute else 'plan'}), flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('choose a new review output')
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(propose, jobs))
    pd.DataFrame([result[0] for result in results]).to_csv(args.output / 'candidates.csv', index=False)
    pd.concat([result[1] for result in results], ignore_index=True).to_csv(args.output / 'foreground_profiles.csv', index=False)
    import cv2
    for start in range(0, len(jobs), 4):
        images = [cv2.imread(str(args.output / f"E{job['number']}_boundary.png")) for job in jobs[start:start + 4]]
        cv2.imwrite(str(args.output / f'page_{start // 4 + 1:02}.png'), np.concatenate(images))
    (args.output / 'manifest.json').write_text(json.dumps({
        'scope': 'review aid for visually identified removal/empty endpoints only; no exclusions applied',
        'background': 'median of nine grayscale samples 180 to 150 seconds before last modelled frame',
        'foreground': '240x240 pixels; median offset removed; absolute difference >25; median filter 3',
        'rule': 'fraction >=0.04 on three samples two seconds apart; cutoff two seconds before onset',
        'lookback_seconds': 120, 'workers': args.workers,
        'limits': 'Lighting/camera/tether movement can cause false positives; visually inspect every boundary'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
