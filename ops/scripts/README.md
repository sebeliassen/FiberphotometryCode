# Operational compatibility entry points

Implementations live in `ops/monitoring`, `ops/dlc`, `ops/compaction` and `ops/repo`.
This directory retains import and direct-execution aliases for older callers.
Each alias executes the target module as a script, including targets whose entry
point is named `run_pipeline` rather than `main`.

Install the package in the intended environment before invoking these tools.
For example, `python ops/monitoring/ntfy_monitor.py --help` shows the log and topic
arguments. Monitoring and data maintenance run only when explicitly invoked.
Logs from new compaction runs live under `outputs/compaction`.

`launch_extended_when_limited_done` requires explicit predecessor log, sweep
script and output log paths; the old sweeps are preserved historical material.
DLC analysis similarly requires `--config` and `--video-dir`.
