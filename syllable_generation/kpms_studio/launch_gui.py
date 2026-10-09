#!/usr/bin/env python3
"""
KPMS Studio Launcher
Launches the FastAPI server in kpms_gpu_cuml and opens either a native desktop window (pywebview)
or the system web browser in app mode.
"""

import os
import sys
import time
import socket
import threading
import webbrowser
from pathlib import Path

# Ensure numpy compatibility
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Prevent JAX from preallocating 90% GPU VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"

import uvicorn

APP_DIR = Path(__file__).parent.resolve()
HOST = "127.0.0.1"
PORT = 8000

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0

def run_server():
    uvicorn.run("server:app", host=HOST, port=PORT, app_dir=str(APP_DIR), log_level="warning")

def main():
    use_window = "--browser" not in sys.argv
    
    # 1. Start backend server in a daemon thread if not already running
    if not is_port_in_use(PORT):
        print(f"[KPMS Studio] Starting server on http://{HOST}:{PORT} using kpms_gpu_cuml...")
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to bind
        for _ in range(30):
            if is_port_in_use(PORT):
                break
            time.sleep(0.1)
    else:
        print(f"[KPMS Studio] Server already running on http://{HOST}:{PORT}.")

    url = f"http://{HOST}:{PORT}"

    # 2. Try native window with pywebview
    if use_window and os.environ.get("DISPLAY"):
        try:
            import webview
            print("[KPMS Studio] Opening native application window...")
            window = webview.create_window(
                title="KPMS Studio - Keypoint-MoSeq Calibration & Ethological Labeler",
                url=url,
                width=1480,
                height=920,
                min_size=(1024, 680),
                background_color='#080c14'
            )
            webview.start()
            return
        except Exception as e:
            print(f"[KPMS Studio] pywebview native window failed ({e}), falling back to browser...")

    # 3. Fallback to default browser
    print(f"[KPMS Studio] Opening browser at {url}...")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[KPMS Studio] Shutting down.")

if __name__ == "__main__":
    main()
