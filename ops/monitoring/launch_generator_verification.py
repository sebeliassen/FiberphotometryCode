"""Explicitly launch a finite frozen-model verification round, detached and capped.

Default is a read-only argv plan. --execute freezes source and starts a separate
nohup/user-systemd service with no restart policy; no random search is resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from ops.monitoring.launch_behavior_search import execute_plan
from studies.sequence_viz.run_broad_search import digest


def build_launch(*, root, python, panel, manifest, cache, output, unit, workers=4, threads=2, preflight=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}",unit):
        raise ValueError("simple explicit service unit name required")
    if isinstance(workers,bool) or isinstance(threads,bool) or min(workers,threads)<1:
        raise ValueError("positive worker/thread counts required")
    if workers*threads>len(os.sched_getaffinity(0)):
        raise ValueError("worker/thread product exceeds available CPUs")
    root,python,panel,manifest,cache = [Path(p).resolve() for p in (root,python,panel,manifest,cache)]
    selected = Path(output).absolute()
    if selected.is_symlink() or selected.exists():
        raise ValueError("choose a new explicit output for a new launch")
    output = selected.resolve()
    if not python.is_file() or not os.access(python,os.X_OK):
        raise ValueError("selected Python must be executable")
    if not (root/"studies/sequence_viz/run_generator_verification.py").is_file():
        raise ValueError("verification source missing")
    for path in (panel,manifest):
        if not path.is_file() or path.is_symlink():
            raise ValueError("ordinary explicit panel/manifest files required")
    if not cache.is_dir() or cache.is_symlink():
        raise ValueError("ordinary saved component cache required")
    nohup = shutil.which("nohup")
    if not nohup:
        raise ValueError("nohup missing")
    runtime = output/"runtime"
    frozen_manifest = output/"frozen_input_manifest.json"
    env = dict(PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join((str(runtime/"src"),str(runtime))),FIBERPHOTOMETRY_ROOT=str(runtime),FP_DATA_ROOT=str(root))
    env.update({name:str(threads) for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    log = output/"verification.log"
    argv = ["systemd-run","--user","--unit",unit,"--description","Fixed chain-generator quantitative verification",
        f"--property=WorkingDirectory={runtime}",f"--property=MemoryHigh={(workers*2+2)*1024**3}",
        f"--property=MemoryMax={(workers*2+4)*1024**3}","--property=MemorySwapMax=0",
        "--property=KillMode=control-group","--property=TimeoutStopSec=20s","--property=StandardInput=null",
        "--property=Restart=no",f"--property=StandardOutput=append:{log}",f"--property=StandardError=append:{log}"]
    argv += [f"--setenv={k}={v}" for k,v in env.items()]
    argv += [nohup,str(python),"-B","-m","studies.sequence_viz.run_generator_verification",
        "--panel",str(panel),"--manifest",str(frozen_manifest),"--cache",str(cache),"--output",str(output),
        "--code-root",str(runtime),"--workers",str(workers),"--threads",str(threads),"--execute"]
    if preflight:
        argv.append("--preflight")
    return dict(mode="broad",argv=argv,unit=unit+".service",code_root=str(root),python=str(python),
        panel=str(panel),manifest=str(manifest),manifest_sha256=digest(manifest),runtime=str(runtime),
        frozen_manifest=str(frozen_manifest),output=str(output),log=str(log),environment=env,resume=False,
        launch_record=str(output/"service_launch.json"),preflight=preflight,
        stop_file=str(output/"STOP"),status_argv=["systemctl","--user","status",unit+".service"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","python","panel","manifest","cache","output","unit"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    kwargs = vars(args)
    execute = kwargs.pop("execute")
    plan = build_launch(**kwargs)
    print(json.dumps(execute_plan(plan) if execute else plan,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
