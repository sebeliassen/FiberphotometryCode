"""Run functional or empirical-collection checks with modeling imports forbidden."""

import argparse
import importlib.abc
import os
import sys


class NoModelingBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "jax", "jaxlib", "jax_moseq", "keypoint_moseq", "torch",
            "tensorflow", "tensorflow_probability",
        }:
            raise AssertionError(f"CPU check attempted a modeling-backend import: {fullname}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-empirical", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoModelingBackend())
    import pytest

    options = ["-q", "-p", "no:cacheprovider"]
    if args.collect_empirical:
        options += ["tests/empirical", "--empirical", "--collect-only", "-m", "needs_data or gpu or slow"]
    return int(pytest.main(options))


if __name__ == "__main__":
    raise SystemExit(main())
