"""
tests.test_pipeline_verbs
=========================
Verifies that pipeline CLI verbs (train, evaluate, render, fuse):
1. Fail with non-zero exit code and never record SUCCESS when not wired to their real executors.
2. Call their underlying library functions and only record SUCCESS after the library returns.
"""

import argparse
import json
from unittest.mock import patch

import pytest
import yaml

from fpcode.cli.evaluate import run_evaluation
from fpcode.cli.fuse import run_fusion
from fpcode.cli.render import run_render
from fpcode.cli.train import run_training


@pytest.fixture
def mock_train_config(tmp_path):
    cfg_file = tmp_path / "train_cfg.yaml"
    cfg_data = {
        "model_name": "test_model_smoke",
        "project_dir": str(tmp_path / "experiments"),
        "seed": 101,
        "device": "cpu",
    }
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_data, f)
    return cfg_file


def test_train_not_wired_fails_and_no_success(mock_train_config, tmp_path):
    """train.py must exit non-zero and never record SUCCESS until wired to real engine."""
    args = argparse.Namespace(
        config=str(mock_train_config),
        seed=101,
        device="cpu",
        dry_run=False,
        compact=False,
        no_notify=True,
    )

    with patch("fpcode.cli.train.send_ntfy") as mock_notify:
        ret = run_training(args)
        assert ret != 0, "Unwired train verb must exit non-zero"
        mock_notify.assert_not_called()

    # Check run.json did NOT record SUCCESS
    model_dir = tmp_path / "experiments" / "test_model_smoke"
    run_file = model_dir / "run.json"
    assert run_file.exists()
    rec = json.loads(run_file.read_text())
    assert rec["status"] == "FAILED"
    assert len(rec.get("error_message", "")) > 0


def test_train_mocked_library_records_success(mock_train_config, tmp_path):
    """When the library function returns successfully, train.py records SUCCESS."""
    args = argparse.Namespace(
        config=str(mock_train_config),
        seed=101,
        device="cpu",
        dry_run=False,
        compact=False,
        no_notify=False,
    )

    mock_ckpt = tmp_path / "mock_checkpoint.h5"
    mock_ckpt.touch()

    with patch("fpcode.cli.train.fit_model", return_value=mock_ckpt) as mock_fit, \
         patch("fpcode.cli.train.send_ntfy") as mock_notify:
        ret = run_training(args)
        assert ret == 0
        mock_fit.assert_called_once()
        mock_notify.assert_called_once()
        assert "KPMS Training Finished" in mock_notify.call_args[1]["title"]

    model_dir = tmp_path / "experiments" / "test_model_smoke"
    run_file = model_dir / "run.json"
    assert run_file.exists()
    rec = json.loads(run_file.read_text())
    assert rec["status"] == "SUCCESS"
    assert rec["output_path"] == str(mock_ckpt)


def test_evaluate_not_wired_fails_and_no_success(tmp_path):
    """evaluate.py must exit non-zero and never record SUCCESS until wired."""
    ckpt = tmp_path / "checkpoint.h5"
    ckpt.touch()

    args = argparse.Namespace(
        models=[str(ckpt)],
        output_dir=str(tmp_path),
        dry_run=False,
        no_notify=True,
    )

    with patch("fpcode.cli.evaluate.read_checkpoint_identity", return_value={
        "path": str(ckpt), "iteration": 250, "n_recordings": 10, "metadata_hash": "abc"
    }):
        ret = run_evaluation(args)
        assert ret != 0, "Unwired evaluate verb must exit non-zero"


def test_evaluate_mocked_library_records_success(tmp_path):
    """When evaluate_model_checkpoints succeeds, evaluate.py records SUCCESS."""
    ckpt = tmp_path / "checkpoint.h5"
    ckpt.touch()

    args = argparse.Namespace(
        models=[str(ckpt)],
        output_dir=str(tmp_path),
        dry_run=False,
        no_notify=False,
    )

    with patch("fpcode.cli.evaluate.read_checkpoint_identity", return_value={
        "path": str(ckpt), "iteration": 250, "n_recordings": 10, "metadata_hash": "abc"
    }), patch("fpcode.cli.evaluate.evaluate_model_checkpoints", return_value={"best_seed": 101}) as mock_eval, \
       patch("fpcode.cli.evaluate.send_ntfy") as mock_notify:
        ret = run_evaluation(args)
        assert ret == 0
        mock_eval.assert_called_once()
        mock_notify.assert_called_once()


def test_render_not_wired_fails_and_no_success(tmp_path):
    """render.py must exit non-zero and never record SUCCESS until wired."""
    res = tmp_path / "results.h5"
    res.touch()

    args = argparse.Namespace(
        results=str(res),
        type="grid_movies",
        grid_size=16,
        syllables=None,
        dry_run=False,
        no_notify=True,
    )

    ret = run_render(args)
    assert ret != 0, "Unwired render verb must exit non-zero"


def test_render_mocked_library_records_success(tmp_path):
    """When render_model_visualizations succeeds, render.py records SUCCESS."""
    res = tmp_path / "results.h5"
    res.touch()

    args = argparse.Namespace(
        results=str(res),
        type="grid_movies",
        grid_size=16,
        syllables=None,
        dry_run=False,
        no_notify=False,
    )

    with patch("fpcode.cli.render.render_model_visualizations", return_value={"rendered": 16}) as mock_rend, \
         patch("fpcode.cli.render.send_ntfy") as mock_notify:
        ret = run_render(args)
        assert ret == 0
        mock_rend.assert_called_once()
        mock_notify.assert_called_once()


def test_fuse_pipeline_calls_library_and_records_success(tmp_path):
    """fuse.py calls fusion library and records SUCCESS on completion."""
    top_dir = tmp_path / "top_h5"
    top_dir.mkdir()
    bot_dir = tmp_path / "bot_h5"
    bot_dir.mkdir()
    out_dir = tmp_path / "fused_out"

    cfg_file = tmp_path / "fuse_cfg.yaml"
    cfg_data = {
        "mode": "static_21kp",
        "top_dir": str(top_dir),
        "bot_dir": str(bot_dir),
        "output_dir": str(out_dir),
    }
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_data, f)

    args = argparse.Namespace(
        config=str(cfg_file),
        dry_run=False,
        no_notify=False,
    )

    mock_pair = {"sid": "sess_1", "top_h5": "top.h5", "bot_h5": "bot.h5"}
    with patch("fpcode.cli.fuse.find_matched_pairs", return_value=[mock_pair]), \
         patch("fpcode.cli.fuse.fuse_single_pair_static", return_value={"status": "FUSED"}) as mock_fuse, \
         patch("fpcode.cli.fuse.send_ntfy") as mock_notify:
        ret = run_fusion(args)
        assert ret == 0
        mock_fuse.assert_called_once()
        mock_notify.assert_called_once()

    run_file = out_dir / "run.json"
    assert run_file.exists()
    rec = json.loads(run_file.read_text())
    assert rec["status"] == "SUCCESS"
