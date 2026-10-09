"""
fpcode.common.notify
====================
Unified ntfy.sh notification helper for FiberphotometryCode.
Replaces 94 copy-pasted implementations across the repository.
"""

import os
import sys
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
DEFAULT_NTFY_TOPIC = "woody_videos_seb"


def send_ntfy(
    title: str,
    msg: str,
    priority: str = "default",
    tags: str = "robot",
    topic: Optional[str] = None,
) -> bool:
    """
    Sends notification to user's ntfy.sh endpoint.
    Configurable via NTFY_SERVER and NTFY_TOPIC environment variables.
    Fails gracefully without interrupting pipeline execution.
    """
    if os.environ.get("NTFY_DISABLE", "0") in ("1", "true", "True"):
        return False

    server = os.environ.get("NTFY_SERVER", DEFAULT_NTFY_SERVER).rstrip("/")
    resolved_topic = topic or os.environ.get("NTFY_TOPIC", DEFAULT_NTFY_TOPIC).strip("/")
    url = f"{server}/{resolved_topic}"

    try:
        req = urllib.request.Request(
            url,
            data=msg.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": str(priority),
                "Tags": str(tags),
            },
        )
        with urllib.request.urlopen(req, timeout=8):
            pass
        return True
    except Exception as e:
        sys.stderr.write(f"[ntfy warning] Notification '{title}' failed ({e})\n")
        sys.stderr.flush()
        return False
