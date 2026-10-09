"""
fpcode.common.run_record
========================
Execution provenance tracking emitting atomic run.json manifests.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .paths import resolve_path, to_relative_path


def _get_git_info() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        is_dirty = len(status_out.strip()) > 0
        return {"git_sha": sha, "git_dirty": is_dirty}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": False}


def _get_gpu_info() -> Dict[str, Any]:
    gpu_info = {"available": False, "device_name": "None", "cuda_version": "None"}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return gpu_info


class RunRecord:
    """Manages structured provenance recording for pipelines. Emits run.json atomically."""

    def __init__(
        self,
        run_name: str = "run",
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[str] = None,
        config_path: Optional[str] = None,
        config_hash: Optional[str] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.run_name = pipeline or run_name
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.config = config or {}
        if config_path:
            self.config["config_path"] = config_path
        if config_hash:
            self.config["config_hash"] = config_hash
        if seed is not None:
            self.config["seed"] = seed
        self.extra = extra or {}
        self.start_time = time.time()
        self.start_time_iso = datetime.now(timezone.utc).isoformat()
        self.input_fingerprints: Dict[str, Any] = {}
        self.output_fingerprints: Dict[str, Any] = {}
        self.status = "INITIALIZED"
        self.error_message: Optional[str] = None
        self.output_path: Optional[str] = None

    def add_input_file(self, file_path: Union[str, Path]) -> None:
        p = resolve_path(file_path)
        if p.exists():
            st = p.stat()
            self.input_fingerprints[to_relative_path(p)] = {
                "size_bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }

    def add_output_file(self, file_path: Union[str, Path]) -> None:
        p = resolve_path(file_path)
        if p.exists():
            st = p.stat()
            self.output_fingerprints[to_relative_path(p)] = {
                "size_bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }

    def finish(
        self,
        status: str = "SUCCESS",
        error: Optional[str] = None,
        output_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        self.status = status
        self.error_message = error
        if output_path:
            self.output_path = output_path
        if extra:
            self.extra.update(extra)
        end_time = time.time()
        end_time_iso = datetime.now(timezone.utc).isoformat()
        duration = round(end_time - self.start_time, 2)

        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

        record_data = {
            "run_name": self.run_name,
            "status": self.status,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "extra": self.extra,
            "start_time_utc": self.start_time_iso,
            "end_time_utc": end_time_iso,
            "duration_seconds": duration,
            "command": " ".join(sys.argv),
            "argv": sys.argv,
            "git": _get_git_info(),
            "config_hash": config_hash,
            "config": self.config,
            "environment": {
                "python_version": sys.version.split()[0],
                "executable": sys.executable,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "gpu": _get_gpu_info(),
            },
            "inputs": self.input_fingerprints,
            "outputs": self.output_fingerprints,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_json_path = self.output_dir / "run.json"
        temp_path = self.output_dir / f"run.json.tmp.{os.getpid()}"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, run_json_path)
        return run_json_path
