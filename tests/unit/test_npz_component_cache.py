"""Atomic component identities, allocation caps, and process-death recovery."""

import hashlib
import io
import json
import multiprocessing
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fpcode.common.cache_manifest import cached_npz_component


def _concurrent_worker(directory, ready, results):
    ready.wait(5)

    def build():
        with (Path(directory)/"builds.txt").open("a") as handle:
            handle.write("build\n")
        return {"values": np.arange(3)}

    arrays, hit, key = cached_npz_component(directory, {"source": "same"}, build)
    results.put((arrays["values"].tolist(), hit, key))


def _killed_worker(directory, started):
    def build():
        encoded = json.dumps({"source": "killed"}, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(encoded.encode()).hexdigest()
        (Path(directory)/f".{key}.npz.killed.tmp").write_bytes(b"partial")
        started.set()
        multiprocessing.Event().wait(30)
        return {"values": np.array([999])}

    cached_npz_component(directory, {"source": "killed"}, build)


def test_component_identity_roundtrip_and_caps_precede_publication(tmp_path):
    arrays, hit, key = cached_npz_component(tmp_path, {"mask": [1, 0], "split": "train"},
                                           lambda: {"x": np.arange(3)})
    assert not hit
    np.testing.assert_array_equal(arrays["x"], [0, 1, 2])
    same, hit, second_key = cached_npz_component(tmp_path, {"split": "train", "mask": [1, 0]},
                                                lambda: pytest.fail("cache hit rebuilt"))
    assert hit and key == second_key
    np.testing.assert_array_equal(same["x"], arrays["x"])
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            cached_npz_component(tmp_path, {"bad": cap}, lambda: {}, max_uncompressed_bytes=cap)
    with pytest.raises(MemoryError, match="cap"):
        cached_npz_component(tmp_path, {"large": True}, lambda: {"x": np.ones(9)}, max_uncompressed_bytes=8)
    with pytest.raises(ValueError, match="non-object"):
        cached_npz_component(tmp_path, {"object": True}, lambda: {"x": np.array([{}], dtype=object)})
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_component_corruption_and_wrong_identity_are_not_hits(tmp_path):
    identity = {"source": "verified"}
    _, _, key = cached_npz_component(tmp_path, identity, lambda: {"x": np.ones(2)})
    target = tmp_path/f"{key}.npz"
    target.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="corrupt"):
        cached_npz_component(tmp_path, identity, lambda: pytest.fail("corruption erased"))
    np.savez_compressed(target, __identity_json=np.array('{"source":"other"}'), x=np.ones(2))
    with pytest.raises(ValueError, match="identity mismatch"):
        cached_npz_component(tmp_path, identity, lambda: pytest.fail("identity failure erased"))


def test_array_headers_are_checked_before_large_allocations(tmp_path):
    identity = {"source": "header"}
    _, _, key = cached_npz_component(tmp_path, identity, lambda: {"x": np.ones(2)})
    target = tmp_path/f"{key}.npz"
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    metadata = io.BytesIO()
    np.save(metadata, np.array(encoded), allow_pickle=False)
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {'shape': (10**9,), 'fortran_order': False, 'descr': '<f8'})
    with zipfile.ZipFile(target, 'w') as archive:
        archive.writestr('__identity_json.npy', metadata.getvalue())
        archive.writestr('x.npy', payload.getvalue())
    with pytest.raises(ValueError, match='header disagrees'):
        cached_npz_component(tmp_path, identity, lambda: pytest.fail('malformed cache rebuilt'), max_uncompressed_bytes=32)


def test_component_only_one_concurrent_builder(tmp_path):
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()
    processes = [context.Process(target=_concurrent_worker, args=(tmp_path, ready, results)) for _ in range(3)]
    try:
        for process in processes:
            process.start()
        ready.set()
        received = [results.get(timeout=5) for _ in processes]
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        assert (tmp_path/"builds.txt").read_text().splitlines() == ["build"]
        assert sum(not entry[1] for entry in received) == 1
        assert len({entry[2] for entry in received}) == 1
        assert all(entry[0] == [0, 1, 2] for entry in received)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)


def test_killed_builder_lock_and_partial_artifact_are_recoverable(tmp_path):
    context = multiprocessing.get_context("fork")
    started = context.Event()
    process = context.Process(target=_killed_worker, args=(tmp_path, started))
    process.start()
    try:
        assert started.wait(5)
        process.terminate()
        process.join(5)
        assert process.exitcode != 0
        arrays, hit, _ = cached_npz_component(tmp_path, {"source": "killed"}, lambda: {"values": np.array([7])})
        assert not hit
        np.testing.assert_array_equal(arrays["values"], [7])
        assert not list(tmp_path.glob(".*.tmp"))
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
