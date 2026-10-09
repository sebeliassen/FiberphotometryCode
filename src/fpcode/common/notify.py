"""
fpcode.common.notify
====================
Unified ntfy.sh notification helper for FiberphotometryCode.
Replaces 94 copy-pasted implementations across the repository.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
DEFAULT_NTFY_TOPIC = "woody_videos_seb"


def send_ntfy(
    title_or_msg: str = "",
    msg: Optional[str] = None,
    priority: str = "default",
    tags: str = "robot",
    topic: Optional[str] = None,
    message: Optional[str] = None,
    title: Optional[str] = None,
) -> bool:
    """
    Sends notification to user's ntfy.sh endpoint.
    Configurable via NTFY_SERVER and NTFY_TOPIC environment variables.
    Fails gracefully without interrupting pipeline execution.
    Supports all legacy keyword and positional signatures.
    """
    if os.environ.get("NTFY_DISABLE", "0") in ("1", "true", "True"):
        return False

    # Resolve message body and title across positional and keyword permutations
    if msg is not None:
        resolved_msg = str(msg)
        resolved_title = str(title or title_or_msg or "Notification")
    elif message is not None:
        resolved_msg = str(message)
        resolved_title = str(title or title_or_msg or "Notification")
    elif title is not None:
        resolved_title = str(title)
        resolved_msg = str(title_or_msg)
    else:
        # Single positional argument passed: treat as message
        resolved_msg = str(title_or_msg)
        resolved_title = "Notification"

    server = os.environ.get("NTFY_SERVER", DEFAULT_NTFY_SERVER).rstrip("/")
    resolved_topic = topic or os.environ.get("NTFY_TOPIC", DEFAULT_NTFY_TOPIC).strip("/")
    url = f"{server}/{resolved_topic}"

    try:
        req = urllib.request.Request(
            url,
            data=resolved_msg.encode("utf-8"),
            headers={
                "Title": resolved_title.encode("ascii", "replace").decode("ascii"),
                "Priority": str(priority),
                "Tags": str(tags),
            },
        )
        with urllib.request.urlopen(req, timeout=8):
            pass
        return True
    except Exception as e:
        sys.stderr.write(f"[ntfy warning] Notification '{resolved_title}' failed ({e})\n")
        sys.stderr.flush()
        return False
