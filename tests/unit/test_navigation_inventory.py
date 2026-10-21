"""Navigation derives source facts without traversing data or executing modules."""

import subprocess

from ops.repo import generate_folder_maps as maps


def test_inventory_prunes_data_and_external_links_and_resolves_relative_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "ROOT", tmp_path)
    package = tmp_path / "src/fpcode/common"
    package.mkdir(parents=True)
    (package / "example.py").write_text("from .child import value\nraise RuntimeError('must never import')\ndef operation(): pass\n")
    (package / "child.py").write_text("value = 1\n")
    (package / "data").mkdir()
    (package / "data/private.py").write_text("invalid source must not be read")
    (package / "cache").mkdir()
    (package / "cache/private.py").write_text("invalid cached source must not be read")
    external = tmp_path / "external"
    external.mkdir()
    (external / "hidden.py").write_text("invalid external source must not be read")
    (package / "linked").symlink_to(external, target_is_directory=True)
    inventory = maps.collect_directory_tree(["src/fpcode"])
    assert {entry.rel_path for entry in inventory} == {"src/fpcode", "src/fpcode/common"}
    leaf = next(entry for entry in inventory if entry.path == package)
    assert "fpcode.common.child" in leaf.internal_dependencies
    rendered = maps.render_leaf_map(leaf)
    assert "example.py#L3" in rendered
    assert not (external / ".folder-map.md").exists()
    assert not (package / "data/.folder-map.md").exists()
    external_map = external / ".folder-map.md"
    external_map.write_text("external notes must remain untouched\n")
    # A stale Git path can now lie below a directory that was replaced by a
    # symlink. It must not become eligible for obsolete-map deletion.
    monkeypatch.setattr(maps, "repository_files", lambda targets: {package / "linked/.folder-map.md"})
    assert not maps.existing_maps(["src/fpcode"])
    assert external_map.read_text() == "external notes must remain untouched\n"


def test_empty_subdirectories_are_not_linked_as_missing_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "ROOT", tmp_path)
    package = tmp_path / "src/fpcode"
    (package / "empty").mkdir(parents=True)
    (package / "__init__.py").write_text("\"\"\"Example.\"\"\"\n")
    inventory = maps.collect_directory_tree(["src/fpcode"])
    assert len(inventory) == 1 and inventory[0].is_leaf
    assert "empty" not in maps.render_leaf_map(inventory[0])


def test_git_ignored_local_scripts_are_not_navigation_inputs(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    monkeypatch.setattr(maps, "ROOT", tmp_path)
    package = tmp_path / "ops/diagnostics"
    package.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("ignored.py\nlocal_only/\n")
    (package / "maintained.py").write_text("def verify(): pass\n")
    (package / "ignored.py").write_text("not Python source")
    (package / "local_only").mkdir()
    (package / "local_only/experiment.py").write_text("not Python source")
    inventory = maps.collect_directory_tree(["ops"])
    assert {entry.rel_path for entry in inventory} == {"ops", "ops/diagnostics"}
    leaf = next(entry for entry in inventory if entry.path == package)
    assert [entry.rel_path for entry in leaf.files] == ["maintained.py"]


def test_check_reports_obsolete_maps_and_regeneration_preserves_other_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "ROOT", tmp_path)
    package = tmp_path / "ops"
    package.mkdir()
    (package / "current.py").write_text("def validate(): pass\n")
    assert maps.execute(["ops"], quiet=True) == 0
    old = package / "retired"
    old.mkdir()
    stale = old / ".folder-map.md"
    stale.write_text("obsolete generated map\n")
    notes = old / "NOTES.md"
    notes.write_text("preserved historical notes\n")
    assert maps.execute(["ops"], check_only=True, quiet=True) == 1
    assert stale.exists()
    assert maps.execute(["ops"], quiet=True) == 0
    assert not stale.exists() and notes.read_text() == "preserved historical notes\n"


def test_named_source_data_packages_remain_visible_without_empirical_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "ROOT", tmp_path)
    source = tmp_path / "studies/photometry_exploration/data"
    source.mkdir(parents=True)
    (source / "loading.py").write_text("raise RuntimeError('never import')\ndef load(): pass\n")
    empirical = tmp_path / "studies/unrelated/data"
    empirical.mkdir(parents=True)
    (empirical / "private.py").write_text("not Python")
    root_data = tmp_path / "data"
    root_data.mkdir()
    (root_data / "private.py").write_text("not Python")
    inventory = maps.collect_directory_tree(["studies", "data"])
    assert {entry.rel_path for entry in inventory} == {"studies", "studies/photometry_exploration",
                                                              "studies/photometry_exploration/data"}
    assert maps.execute(["studies"], quiet=True) == 0
    assert source / ".folder-map.md" in maps.existing_maps(["studies"])
    assert not (empirical / ".folder-map.md").exists()
    assert not (root_data / ".folder-map.md").exists()
