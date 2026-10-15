"""Launcher validation is read-only and does not require a systemd manager."""

import sys
from pathlib import Path

import pytest

from ops.monitoring.launch_behavior_search import build_plan


def test_command_preserves_paths_limits_and_rejects_unsafe_log_or_unit(tmp_path):
    code = tmp_path / "code with spaces"
    script = code / "studies/karen_pff_vs_monomer/analysis/run_overnight_random_search.py"
    script.parent.mkdir(parents=True)
    script.touch()
    atlas = tmp_path / "atlas with spaces"
    atlas.mkdir()
    (atlas / "sessions.csv").touch()
    output = tmp_path / "output with spaces"
    options = dict(code_root=code, python=sys.executable, atlas=atlas, output=output, unit="fixture-search", hours=8)
    plan = build_plan(**options)
    assert not output.exists(), "planning must not create output artifacts"
    assert plan["unit"] == "fixture-search.service"
    command = plan["argv"]
    assert command[command.index("--atlas") + 1] == str(atlas)
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--workers") + 1] == "4"
    assert "--property=RuntimeMaxSec=29700s" in command
    assert f"--property=MemoryMax={24 * 1024 ** 3}" in command
    assert f"--property=StandardOutput=append:{output / 'search.log'}" in command
    assert "--setenv=OPENBLAS_NUM_THREADS=4" in command
    assert f"--setenv=PYTHONPATH={code / 'src'}:{code}" in command
    assert command[command.index("--min-free-disk-gb") + 1] == "80"
    assert command[command.index("--min-free-ram-gb") + 1] == "8"
    assert plan["min_free_disk_gb"] == 80 and plan["min_free_ram_gb"] == 8
    with pytest.raises(ValueError, match="nonnegative"):
        build_plan(**dict(options, min_free_ram_gb=-1))
    assert plan["stop_argv"] == ["systemctl", "--user", "stop", "fixture-search.service"]
    indefinite = build_plan(**dict(options, hours=0))["argv"]
    assert "--property=RuntimeMaxSec=infinity" in indefinite
    assert "--property=RuntimeMaxSec=900s" not in indefinite
    assert "--property=Restart=on-failure" in indefinite
    assert "--property=StandardInput=null" in indefinite
    python_index = indefinite.index(str(Path(sys.executable).resolve()))
    assert Path(indefinite[python_index - 1]).name == "nohup"
    assert indefinite[indefinite.index("--hours") + 1] == "0"
    with pytest.raises(ValueError, match="unit must"):
        build_plan(**dict(options, unit="bad;run-command"))
    output.mkdir()
    target = tmp_path / "unrelated.log"
    target.write_text("preserve me")
    (output / "search.log").symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        build_plan(**options)
    assert target.read_text() == "preserve me"
