"""Run the retained CenterTopCam DLC analysis with explicit project locations."""
import argparse
import datetime
import glob
import os
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    args = parser.parse_args()
    config = str(args.config)
    video_dir = str(args.video_dir)
    import deeplabcut

    # Only select videos with "CenterTopCamVid" in their filename
    all_videos = sorted([v for v in glob.glob(os.path.join(video_dir, '*.avi')) if 'CenterTopCamVid' in v])
    total_videos = len(all_videos)
    print(f"Found {total_videos} total CenterTopCamVid videos.")

    # Filter out already analyzed videos
    videos_to_analyze = []
    for v in all_videos:
        base_name = os.path.splitext(os.path.basename(v))[0]
        h5_pattern = os.path.join(video_dir, f"{base_name}*.h5")
        if glob.glob(h5_pattern):
            print(f"Skipping already analyzed video: {os.path.basename(v)}")
        else:
            videos_to_analyze.append(v)

    # Reorder to run the crash-prone session last and other sessions in reverse order
    crash_video = "T2_5.11.12_16.05.2026_Arena1234_CenterTopCamVid_2026-05-16T12_28_03_top_left.avi"
    crash_paths = [v for v in videos_to_analyze if os.path.basename(v) == crash_video]
    other_paths = [v for v in videos_to_analyze if os.path.basename(v) != crash_video]
    videos_to_analyze = other_paths[::-1] + crash_paths

    total_to_analyze = len(videos_to_analyze)
    print(f"\nDetermined {total_to_analyze} videos need analysis.")

    # Send initial notification
    try:
        init_msg = f"Starting DLC analysis for {total_to_analyze} new videos (out of {total_videos} total)."
        urllib.request.urlopen("https://ntfy.sh/woody_videos_seb", data=init_msg.encode('utf-8'))
    except Exception as ntfy_err:
        print(f"Notification error: {ntfy_err}")

    start_time = time.time()
    analyzed_count = 0

    try:
        for i, video in enumerate(videos_to_analyze):
            video_name = os.path.basename(video)
            print(f"\n--- Analyzing video {i+1}/{total_to_analyze}: {video_name} ---")

            # Run analysis for this single video
            deeplabcut.analyze_videos(config, [video], shuffle=10, save_as_csv=True)

            analyzed_count += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / analyzed_count
            remaining_count = total_to_analyze - analyzed_count
            eta_seconds = avg_time * remaining_count
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            # Send progress notification after each video
            try:
                msg = f"Progress: {analyzed_count}/{total_to_analyze} analyzed. Completed: {video_name}. ETA: {eta_str}"
                urllib.request.urlopen("https://ntfy.sh/woody_videos_seb", data=msg.encode('utf-8'))
            except Exception as ntfy_err:
                print(f"Notification error: {ntfy_err}")

        # Send final success notification
        try:
            urllib.request.urlopen("https://ntfy.sh/woody_videos_seb", data=b"DeepLabCut analysis finished successfully for all new videos!")
        except Exception:
            pass

    except Exception as e:
        # Send error notification if it crashes
        try:
            error_msg = f"DeepLabCut analysis failed with error: {str(e)}"
            urllib.request.urlopen("https://ntfy.sh/woody_videos_seb", data=error_msg.encode('utf-8'))
        except Exception:
            pass
        raise e



if __name__ == "__main__":
    main()
