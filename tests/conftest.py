# tests/conftest.py

import sys, os

# 1) Add project root (where `fiberphotometry/` lives) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
from pathlib import Path
import fiberphotometry.data.session_loading as sl

def pytest_addoption(parser):
    parser.addoption(
        "--num-sources",
        action="store",
        default=3,
        type=int,
        help="How many source folders to sample for testing"
    )

@pytest.fixture(scope="module")
def num_sources(request):
    return request.config.getoption("--num-sources")

