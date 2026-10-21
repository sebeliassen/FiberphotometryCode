"""Render unreviewed Karen chain examples using explicitly relocated video paths."""

import argparse
import html
from pathlib import Path

import pandas as pd

from fpcode.viz.event_clips import render_event_clip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sessions = pd.read_csv(args.output / "sessions.csv", dtype={"animal_id": str}).set_index("session_id")
    catalog = pd.read_csv(args.output / "chains/chain_catalog.csv")
    events = pd.read_csv(args.output / "chains/chain_occurrences.csv", dtype={"animal_id": str})
    chosen = catalog[catalog.tokens.isin(["15 25 15", "56 58 56", "42 71 42", "1 16 71"])].chain_id
    clips = args.output / "examples"
    clips.mkdir(exist_ok=True)
    roster, cards = [], []
    for (chain, group, condition), sub in events[events.chain_id.isin(chosen)].groupby(["chain_id", "group", "condition"]):
        target = sub.duration_s.median()
        row = sub.iloc[(sub.duration_s - target).abs().argmin()]
        meta = sessions.loc[row.session_id]
        video = args.video_dir / Path(meta.video_path).name
        path = clips / f"{chain}_{group}_{condition}.mp4"
        rendered = render_event_clip(video, path, start_frame=int(row.start_frame),
                                     end_frame_exclusive=int(row.end_frame_exclusive), fps=float(meta.fps),
                                     caption=f"{chain} | {group} | {condition} | mouse {row.animal_id} | UNREVIEWED")
        roster.append({**row.to_dict(), "source_video": str(video.resolve()), **rendered, "review_status": "unreviewed"})
        cards.append(f'<div><p>{html.escape(path.stem)}</p><video controls preload="none" src="{html.escape(path.name)}"></video></div>')
        print("Rendered", path.name, flush=True)
    pd.DataFrame(roster).to_csv(clips / "clip_roster.csv", index=False)
    (clips / "gallery.html").write_text("<!doctype html><meta charset='utf-8'><title>Karen chain examples</title>"
                                      "<style>body{font:16px system-ui;margin:25px}main{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}video{width:100%}</style>"
                                      "<h1>Unreviewed chain examples</h1><p>One occurrence closest to the median duration in each chain/group/wave cell. "
                                      "These examples illustrate detections and require behavioural review.</p><main>" + "\n".join(cards) + "</main>")


if __name__ == "__main__":
    main()
