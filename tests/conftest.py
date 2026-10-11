"""
Pytest configuration and global fixtures for FiberphotometryCode.
Provides `require_artifact` fixture to gracefully skip tests when running on
trimmed mirrors or placeholder-only environments.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
compat_dir = REPO_ROOT / "archive/compat"
if compat_dir.exists() and str(compat_dir) not in sys.path:
    sys.path.insert(0, str(compat_dir))


def is_placeholder(path: Path) -> bool:
    """Check if a path points to a lightweight metadata placeholder rather than real binary data."""
    if not path.exists():
        # Check if an original.ext.txt placeholder exists
        txt_companion = path.with_name(path.name + ".txt")
        if txt_companion.exists():
            return True
        return False

    # Check if the file itself is a text placeholder
    try:
        if path.is_file() and path.stat().st_size < 8192:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(512)
                if "[PLACEHOLDER FILE" in header or "PLACEHOLDER FILE" in header:
                    return True
    except Exception:
        pass

    return False


@pytest.fixture
def require_artifact():
    """
    Fixture that verifies required data files exist in full binary form.
    Gracefully skips tests when executed on trimmed/placeholder archives.

    Usage:
        def test_heavy_fitting(require_artifact):
            require_artifact('path/to/checkpoint.h5')
            ...
    """
    def _check(path_or_paths):
        if isinstance(path_or_paths, (str, Path)):
            paths = [Path(path_or_paths)]
        else:
            paths = [Path(p) for p in path_or_paths]

        for p in paths:
            # Resolve relative to repo root if not absolute
            resolved = p if p.is_absolute() else (REPO_ROOT / p)
            if not resolved.exists() or is_placeholder(resolved):
                pytest.skip(
                    f"Test skipped: Required artifact '{p.name}' is missing or a trimmed placeholder on this machine."
                )
        return True

    return _check


@pytest.fixture
def repo_root():
    """Return the absolute Path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def golden_dir(repo_root):
    """Return the absolute Path to tests/golden."""
    return repo_root / "tests" / "golden"


# The default testpaths in pyproject.toml are an explicit functional suite.
# CPU is not a runtime guarantee: empirical/model tests require explicit paths.
