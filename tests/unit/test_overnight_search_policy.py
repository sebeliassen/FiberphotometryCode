"""Real SQLite and metadata fixtures for Karen search scheduling and resume policy."""

import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis import run_overnight_random_search as search


def _identity():
    return {"protocol": "karen-sequence-search-v3", "source_sha256": {"evaluator.py": "source-a"},
            "input_sha256": {"bouts.csv": "input-a"}, "evaluation": {"iterations": 80},
            "replicates_per_configuration": 2, "threads_per_worker": 4}


def _insert(conn, trial_id, config, replicate, fitness, status="SUCCESS"):
    result = {"fitness": fitness, "details": {"ranking_scope": [config["level"], config["method"]]}}
    conn.execute("INSERT INTO trials (trial_id,config_key,replicate,level,method,status,fitness,order_gain,"
                 "stability,explained,dead_fraction,redundancy,result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (trial_id, search.canonical(config), replicate, config["level"], config["method"], status,
                  fitness, .1, .5, .4, 0., .2, search.canonical(result)))
    conn.commit()


def test_deterministic_schedule_covers_active_knobs_and_two_replicates_in_eight_strata():
    buckets = search.configuration_buckets(101)
    assert buckets == search.configuration_buckets(101)
    assert buckets != search.configuration_buckets(102)
    assert len(buckets) == 8 and {len(bucket) for bucket in buckets} == {320}
    length = sum(map(len, buckets)) * 2
    scheduled = [search.configuration_at(ordinal, buckets, 2) for ordinal in range(length)]
    counts = {}
    for config, replicate in scheduled:
        expected = {"level", "method", "min_bout_ms", "chain_len", "n_units",
                    "min_prevalence" if config["method"] == "exact" else "penalty"}
        assert set(config) == expected
        counts.setdefault(search.canonical(config), []).append(replicate)
    assert len(counts) == 2560
    assert all(replicates == [0, 1] for replicates in counts.values())
    for offset in range(0, length, 8):
        strata = {(config["level"], config["method"]) for config, _ in scheduled[offset:offset + 8]}
        assert strata == {(level, method) for level in search.LEVELS for method in search.METHODS}
    assert search.configuration_at(length, buckets, 2) == (buckets[0][0], 2)


@pytest.mark.parametrize("changed_field", ["source_sha256", "input_sha256", "evaluation", "threads_per_worker",
                                         "replicates_per_configuration", "protocol"])
def test_resume_refuses_changed_identity_without_altering_saved_evidence(tmp_path, changed_field):
    identity = _identity()
    conn = search.init_db(tmp_path, identity)
    _insert(conn, 1, search.configuration_buckets(101)[0][0], 0, .6)
    conn.close()
    changed = copy.deepcopy(identity)
    changed[changed_field] = {"changed": True} if isinstance(changed[changed_field], dict) else "changed"
    with pytest.raises(ValueError, match="identity changed"):
        search.init_db(tmp_path, changed)
    check = sqlite3.connect(tmp_path / "trials.db")
    assert check.execute("SELECT fitness,status FROM trials").fetchall() == [(.6, "SUCCESS")]
    assert check.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0] == search.canonical(identity)
    assert json.loads((tmp_path / "run_manifest.json").read_text()) == identity
    check.close()


def test_resume_interrupts_pending_work_and_preserves_terminal_rows_and_schedule(tmp_path):
    identity, buckets = _identity(), search.configuration_buckets(101)
    conn = search.init_db(tmp_path, identity)
    _insert(conn, 1, buckets[0][0], 0, .6)
    _insert(conn, 2, buckets[1][0], 0, None, "RUNNING")
    _insert(conn, 3, buckets[2][0], 0, None, "PENDING")
    conn.execute("UPDATE metadata SET value='3' WHERE key='next_ordinal'")
    conn.commit()
    conn.close()
    conn = search.init_db(tmp_path, identity)
    assert conn.execute("SELECT status FROM trials ORDER BY trial_id").fetchall() == [
        ("SUCCESS",), ("INTERRUPTED",), ("INTERRUPTED",)]
    assert conn.execute("SELECT value FROM metadata WHERE key='next_ordinal'").fetchone() == ("3",)
    assert conn.execute("SELECT fitness FROM trials WHERE trial_id=1").fetchone() == (.6,)
    conn.close()


def test_legacy_placeholder_database_cannot_be_adopted_as_new_protocol(tmp_path):
    path = tmp_path / "trials.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trials (trial_id INTEGER PRIMARY KEY,status TEXT,fitness_score REAL,error TEXT)")
    conn.execute("INSERT INTO trials VALUES (1,'SUCCESS',.99,'invalid placeholder scorer')")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="identity|protocol|legacy|schema"):
        search.init_db(tmp_path, _identity())
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT * FROM trials").fetchall() == [(1, "SUCCESS", .99, "invalid placeholder scorer")]
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "metadata" in tables:
        assert conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone() is None
    conn.close()


def test_leaderboard_uses_replicate_means_per_stratum_and_removes_stale_rank_files(tmp_path):
    conn = search.init_db(tmp_path, _identity())
    buckets = search.configuration_buckets(101)
    volatile, reliable = buckets[0][:2]
    _insert(conn, 1, volatile, 0, 1.)
    _insert(conn, 2, volatile, 1, 0.)
    _insert(conn, 3, reliable, 0, .6)
    _insert(conn, 4, reliable, 1, .6)
    # Lower scores in another method still have rank one in that stratum.
    _insert(conn, 5, buckets[1][0], 0, .01)
    folder = tmp_path / "top_models/L1_exact"
    folder.mkdir(parents=True)
    stale = folder / "rank_99_old_config.json"
    stale.write_text("obsolete")
    search.write_summaries(conn, tmp_path)
    assert not stale.exists()
    board = pd.read_csv(tmp_path / "leaderboard.csv")
    exact = board[(board.level == "L1") & (board.method == "exact")]
    assert exact.fitness_mean.tolist() == [.6, .5]
    assert exact.successful_replicates.tolist() == [2, 2]
    assert exact.rank_within_stratum.tolist() == [1, 2]
    other = board[(board.level == "L1") & (board.method == "seqnmf")].iloc[0]
    assert other.rank_within_stratum == 1
    top = json.loads((folder / "rank_01.json").read_text())
    assert top["config"] == reliable and top["fitness_mean"] == .6
    assert top["successful_replicates"] == 2 and top["representative_trial"] == 3
    assert top["ranking_scope"] == ["L1", "exact"]
    assert top["evidence"]["fitness"] == .6
    assert "first_order_null" in top["confirmation_pending"]
    assert len(list(folder.glob("rank_*.json"))) == 2
    conn.close()


def _atlas_fixture(path):
    path.mkdir()
    (path / "chains").mkdir()
    animals = [f"a{i}" for i in range(6)] + ["reserved"]
    sessions = pd.DataFrame({"session_id": [f"record_{animal}" for animal in animals], "animal_id": animals,
                             "group": ["PFF"] * 3 + ["Monomer"] * 3 + ["PFF"]})
    sessions.to_csv(path / "sessions.csv", index=False)
    (path / "chains/animal_splits.json").write_text(json.dumps(
        {"train": animals[:4], "validation": animals[4:6], "test": ["reserved"]}))
    pd.DataFrame({"syllable_id": [0, 1, 2]}).to_csv(path / "syllable_annotations.csv", index=False)
    for level in search.LEVELS:
        folder = path / "measurements" if level == "L1" else path / "clustering" / level / "measurements"
        (folder / "bouts_by_session").mkdir(parents=True)
        if level != "L1":
            (folder.parent / "mapping.json").write_text("[0,1,2]")
        for animal in animals[:-1]:
            pd.DataFrame({"syllable_id": [0, 1, 2], "segment": [0, 0, 0], "duration_s": [1., 1., 1.],
                          "start_frame": [0, 10, 20], "end_frame_exclusive": [10, 20, 30]}).to_csv(
                              folder / "bouts_by_session" / f"record_{animal}.csv", index=False)
    return path


def test_search_data_never_loads_reserved_bouts_and_identity_hashes_selected_inputs(tmp_path, monkeypatch):
    atlas = _atlas_fixture(tmp_path / "atlas")
    data = search.SearchData(atlas, 101)
    assert data.test == ["reserved"] and len(data.pool) == 6
    assert set(data.sessions.animal_id) == set(data.pool)
    assert all("record_reserved" not in bouts for bouts in data.raw.values())
    assert len(data.bouts("L1", 150)) == 6
    code = tmp_path / "code"
    code.write_text("fixture source")
    monkeypatch.setattr(search, "source_identity", lambda *_: {str(code): search.digest(code)})
    args = SimpleNamespace(atlas=atlas, seed=101, iterations=80, per_animal=700, null_repeats=5,
                           replicates=2, threads=4)
    identity = search.build_identity(args, Path(__file__).parents[2])
    assert identity["phenotype_in_selection"] is False
    assert identity["history_lag_in_discovery_search"] is False
    assert identity["reserved_animals"] == ["reserved"]
    assert not any("record_reserved" in path for path in identity["input_sha256"])
    changed = atlas / "measurements/bouts_by_session/record_a0.csv"
    old_hash = identity["input_sha256"][str(changed.resolve())]
    changed.write_text(changed.read_text() + "\n")
    next_identity = search.build_identity(args, Path(__file__).parents[2])
    assert old_hash != next_identity["input_sha256"][str(changed.resolve())]
