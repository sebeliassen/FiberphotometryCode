"""Native-video-frame clip rendering for explicitly selected behavioural events."""

from pathlib import Path


def read_native_frame(video_path, *, frame_index, fps):
    """Read one explicitly indexed native frame after verifying rate and seek.

    Returns OpenCV's BGR image. No clock origin, rate conversion or recording
    relocation is inferred. Optional video dependencies load only on invocation.
    """
    import math
    from numbers import Integral, Real

    import cv2

    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral) or frame_index < 0:
        raise ValueError('frame_index must be a nonnegative native integer')
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f'cannot open video: {video_path}')
        if abs(capture.get(cv2.CAP_PROP_FPS) - fps) > .001:
            raise ValueError('selected recording and native video rates disagree')
        if frame_index >= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError('native frame lies outside selected video')
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
            raise ValueError('native video seek failed')
        ok, frame = capture.read()
        if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index + 1:
            raise ValueError('native video read/seek position failed verification')
        return frame
    finally:
        capture.release()


def render_event_clip(video_path, output_path, *, start_frame, end_frame_exclusive, fps,
                      caption, pre_seconds=.5, post_seconds=.5, maximum_width=720,
                      active_label="CHAIN"):
    """Render a half-open interval after checking the selected video's own clock."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(native_fps - fps) > .001:
            raise ValueError("selected recording and event frame rates disagree")
        if not 0 <= start_frame < end_frame_exclusive <= n_frames:
            raise ValueError("event interval outside native recording")
        first = max(0, start_frame - round(pre_seconds * fps))
        last = min(n_frames, end_frame_exclusive + round(post_seconds * fps))
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError("native video seek failed")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        for frame_index in range(first, last):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"video ended before requested frame {frame_index}")
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index + 1:
                raise ValueError("native video read/seek position failed verification")
            ratio = min(1., maximum_width / image.shape[1])
            width, height = int(image.shape[1] * ratio) // 2 * 2, int(image.shape[0] * ratio) // 2 * 2
            image = cv2.resize(image, (width, height))
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise ValueError("cannot create output clip")
            active = start_frame <= frame_index < end_frame_exclusive
            cv2.rectangle(image, (0, 0), (width, 70), (20, 20, 20), -1)
            cv2.putText(image, caption[:85], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1)
            label = f"native frame {frame_index} | {(frame_index-start_frame)/fps:+.3f}s | {active_label if active else 'context'}"
            cv2.putText(image, label, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 220, 120) if active else (200, 200, 200), 1)
            writer.write(image)
            written += 1
        return {"clip_start_frame": first, "clip_end_frame_exclusive": last, "frames_written": written,
                "fps": fps, "clip_path": str(output_path)}
    finally:
        capture.release()
        if writer is not None:
            writer.release()
