import json

import pytest

from fpcode.common import cache_manifest as cm


def test_removing_last_input_is_stale(tmp_path):
    artifact, source = tmp_path / "result", tmp_path / "input"
    artifact.write_text("result")
    source.write_text("source")
    cm.write_manifest(artifact, kind="test", inputs=[source])
    assert cm.verify_manifest(artifact)[0] == cm.OK
    assert cm.verify_manifest(artifact, inputs=[])[0] == cm.STALE


@pytest.mark.parametrize("value", [None, [], "string", {"schema_version": 1, "artifact": []}])
def test_valid_json_with_wrong_shape_is_corrupt(tmp_path, value):
    artifact = tmp_path / "result"
    artifact.write_text("result")
    cm.manifest_path_for(artifact).write_text(json.dumps(value))
    assert cm.verify_manifest(artifact)[0] == cm.CORRUPT_MANIFEST


def test_missing_implementation_dependency_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        cm.source_fingerprint([tmp_path / "moved_implementation.py"])
