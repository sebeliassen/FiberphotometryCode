"""Selected PPE input/output routes with only small, test-owned objects/files."""

import importlib
import inspect
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

CANONICAL = "studies.deep_research_experiments.action_prediction_error.run_ppe_analysis"
PLOT_MODULE = "studies.deep_research_experiments.action_prediction_error.lib.plotting"
HISTORICAL_OUTPUT = (
    "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/"
    "deep_research_experiments/action_prediction_error/output"
)
DEFAULTS = dict(test_mode=False, use_history=False, use_kinematics=False,
                use_timewarp=False, use_force=False, suffix="", metric="variance")


class OwnedPath:
    def __init__(self, text):
        self.text = text

    def __fspath__(self):
        return self.text


@pytest.fixture
def owner():
    return importlib.import_module(CANONICAL)


def test_existing_positional_api_and_new_keyword_only_defaults(owner):
    parameters = inspect.signature(owner.run_analysis).parameters
    assert list(parameters) == list(DEFAULTS) + ["pkl_path", "output_dir"]
    for name, default in DEFAULTS.items():
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[name].default == default
    for name in ("pkl_path", "output_dir"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


@pytest.mark.parametrize("sessions, output", [
    ("relative selected input.pkl", None),
    (None, "relative output;$(literal) directory"),
    ("input with spaces;$(literal).pkl", "out with spaces;$(literal)"),
    ("", ""),
])
def test_cli_each_new_option_alone_or_together_is_a_literal_path(owner, monkeypatch, sessions, output):
    calls = []
    monkeypatch.setattr(owner, "run_analysis", lambda **kwargs: calls.append(kwargs) or object())
    argv = ["--history", "--metric", "mean", "--suffix", " unchanged suffix "]
    expected = dict(DEFAULTS, use_history=True, metric="mean", suffix=" unchanged suffix ")
    if sessions is not None:
        argv += ["--sessions-pkl", sessions]
        expected["pkl_path"] = Path(sessions)
    if output is not None:
        argv += ["--output-dir", output]
        expected["output_dir"] = Path(output)
    assert owner.main(argv) is None
    assert calls == [expected]


def test_cli_without_paths_retains_exact_original_keyword_set(owner, monkeypatch):
    calls = []
    monkeypatch.setattr(owner, "run_analysis", lambda **kwargs: calls.append(kwargs))
    assert owner.main([]) is None
    assert calls == [DEFAULTS]


def test_help_advertises_selected_paths_without_loading(owner, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("help attempted analysis or cohort loading")

    monkeypatch.setattr(owner, "run_analysis", forbidden)
    monkeypatch.setattr(owner, "load_sessions", forbidden)
    with pytest.raises(SystemExit) as error:
        owner.main(["--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert "--sessions-pkl" in text and "--output-dir" in text


@pytest.mark.parametrize("kind", ["omitted", "string", "path", "pathlike"])
def test_api_omission_keeps_old_loader_call_and_explicit_inputs_normalize_to_path(owner, monkeypatch, tmp_path, kind):
    calls = []
    selected = "relative input with spaces;$(literal).pkl"
    value = {"omitted": None, "string": selected, "path": Path(selected), "pathlike": OwnedPath(selected)}[kind]

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(owner, "load_sessions", load)
    output = tmp_path / "empty results must not create this"
    arguments = {} if kind == "omitted" else {"pkl_path": value}
    # The seven original positional arguments remain callable.
    assert owner.run_analysis(False, False, False, False, False, "", "variance",
                              output_dir=output, **arguments) is None
    expected = {"apply_filter": True}
    if kind != "omitted":
        expected["pkl_path"] = Path(selected)
    assert calls == [(("amph",), expected)]
    assert not output.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case", ["missing", "empty_string", "empty_pickle"])
def test_explicit_input_never_discovers_or_opens_test_owned_fallback(owner, monkeypatch, tmp_path, case):
    """Only the empty-list pickle created here may be decoded by the real loader."""
    loaders = importlib.import_module("studies.deep_research_experiments.utils.loaders")
    base = loaders.base_loader
    mirror_data = tmp_path / "mirror" / "data"
    mirror_data.mkdir(parents=True)
    fallback = mirror_data / "sessions_all_data.pkl"
    fallback.write_bytes(pickle.dumps([SimpleNamespace(
        session_id="fallback must never load", metadata={},
        all_data_df=pd.DataFrame({"value": [1.]}),
    )]))
    selected = tmp_path / "selected empty.pkl"
    selected.write_bytes(pickle.dumps([]))
    before = {path: path.read_bytes() for path in (fallback, selected)}
    reads = []
    actual_load = pickle.load

    def selected_only(stream, *args, **kwargs):
        assert Path(stream.name) == selected
        reads.append(Path(stream.name))
        return actual_load(stream, *args, **kwargs)

    def forbidden_glob(*args, **kwargs):
        pytest.fail("explicit PPE input attempted fallback discovery")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(base, "resolve_data_path", lambda relative: mirror_data)
    monkeypatch.setattr(Path, "glob", forbidden_glob)
    monkeypatch.setattr(base.pickle, "load", selected_only)
    output = tmp_path / "no results output"
    if case == "missing":
        with pytest.raises(FileNotFoundError, match="missing selected.pkl"):
            owner.run_analysis(pkl_path=tmp_path / "missing selected.pkl", output_dir=output)
    elif case == "empty_string":
        # Path('') is the explicit current directory, not absent input. The real
        # loader must fail opening it instead of discovering a cohort.
        with pytest.raises(IsADirectoryError):
            owner.run_analysis(pkl_path="", output_dir=output)
    else:
        assert owner.run_analysis(pkl_path=selected, output_dir=output) is None
    assert reads == ([selected] if case == "empty_pickle" else [])
    assert {path: path.read_bytes() for path in before} == before
    assert not output.exists()


def synthetic_session():
    """12 independent two-row fits with slopes -3..8, no regression backend.

    fps=10 gives the preserved two-row window. Its native-looking dataframe
    index and a different fs value must not alter the existing row policy.
    Population slope variance is 143/12; mean absolute slope is 3.5.
    """
    slopes = np.arange(-3., 9., dtype=np.float64)
    signal = np.concatenate([100. + slope * np.arange(4) for slope in slopes])
    count = len(signal)
    frame = pd.DataFrame({
        "syllable": np.ones(count, dtype=int),
        "onset": np.tile([1, 0, 0, 0], len(slopes)),
        "signal_2_dff": signal,
        "signal_2": np.full(count, -999.),
        "SystemTimestamp": np.arange(count, dtype=float) / 10.,
    }, index=100 + 3 * np.arange(count))
    return SimpleNamespace(
        session_id="synthetic twelve onsets", project="amph", all_data_df=frame,
        metadata={"genotype": "WT", "drug": "Saline", "fps": 10., "fs": 123.,
                  "signal_meta": {"signal_2": {"brain_reg": "L-DLS", "orig_letter": "G"}}},
    )


def install_analysis_inputs_and_plot_stub(owner, monkeypatch, *, fail_at=None):
    session = synthetic_session()
    original = session.all_data_df.copy(deep=True)
    loads, plots = [], []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return [session]

    def plot(frame, directory, *, filename_suffix, ylabel):
        plots.append((frame.copy(deep=True), directory, filename_suffix, ylabel))
        if fail_at is not None and len(plots) == fail_at:
            raise RuntimeError("test-owned plot failure")

    package = importlib.import_module(PLOT_MODULE.rsplit(".", 1)[0])
    stub = SimpleNamespace(plot_slope_variance=plot)
    monkeypatch.setattr(package, "plotting", stub, raising=False)
    monkeypatch.setitem(sys.modules, PLOT_MODULE, stub)
    monkeypatch.setattr(owner, "load_sessions", load)
    return SimpleNamespace(session=session, original=original, loads=loads, plots=plots)


def assert_independent_result(frame, metric):
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["session"] == "synthetic twelve onsets"
    assert row["genotype"] == "WT" and row["drug"] == "Saline"
    assert row["direction"] == "All" and row["syllable"] == 1 and row["n_trials"] == 12
    assert row["slope_variance_raw"] == pytest.approx(143. / 12., rel=1e-12, abs=1e-10)
    assert row["slope_variance_resid"] == pytest.approx(143. / 12., rel=1e-12, abs=1e-10)
    assert row["slope_mean_abs_raw"] == pytest.approx(3.5, rel=1e-12, abs=1e-10)
    assert row["slope_mean_abs_resid"] == 0 and row["r_squared"] == 0
    assert row["slope_variance"] == pytest.approx(3.5 if metric == "mean" else 143. / 12., abs=1e-10)
    assert row["residual_variance_field"] == pytest.approx(0 if metric == "mean" else 143. / 12., abs=1e-10)


@pytest.mark.parametrize("metric", ["variance", "mean"])
@pytest.mark.parametrize("kind", ["path", "relative_string", "pathlike"])
def test_selected_csv_and_both_plot_calls_share_exact_output_and_numerical_results(owner, monkeypatch, tmp_path, metric, kind):
    fixture = install_analysis_inputs_and_plot_stub(owner, monkeypatch)
    monkeypatch.chdir(tmp_path)
    literal = "selected output with spaces;$(literal)"
    directory = {"path": tmp_path / literal, "relative_string": literal, "pathlike": OwnedPath(literal)}[kind]
    expected_directory = os.fspath(directory)
    selected_input = Path("selected input with spaces.pkl")
    suffix = "_selected [literal]"
    assert owner.run_analysis(pkl_path=selected_input, output_dir=directory, suffix=suffix, metric=metric) is None
    assert fixture.loads == [(("amph",), {"apply_filter": True, "pkl_path": selected_input})]
    csv = Path(expected_directory) / f"ppe_results{suffix}.csv"
    saved = pd.read_csv(csv)
    # The existing CSV includes its dataframe index.
    assert saved.columns[0] == "Unnamed: 0" and saved.iloc[0, 0] == 0
    assert_independent_result(saved, metric)
    assert len(fixture.plots) == 2
    raw, residual = fixture.plots
    assert raw[1:] == (expected_directory, suffix + "_raw",
                       "Mean Absolute Slope (dF/F/s)" if metric == "mean" else "Total Slope Variance")
    assert residual[1:] == (expected_directory, suffix + "_residual",
                            "Mean Absolute Residual (MAE)" if metric == "mean" else "Residual (Error) Variance")
    assert_independent_result(raw[0], metric)
    assert residual[0].iloc[0]["slope_variance"] == pytest.approx(0 if metric == "mean" else 143. / 12., abs=1e-10)
    pd.testing.assert_frame_equal(fixture.session.all_data_df, fixture.original)
    assert list(Path(expected_directory).iterdir()) == [csv]


@pytest.mark.parametrize("fail_at", [1, 2])
def test_plot_failure_preserves_already_written_selected_csv(owner, monkeypatch, tmp_path, capsys, fail_at):
    fixture = install_analysis_inputs_and_plot_stub(owner, monkeypatch, fail_at=fail_at)
    output = tmp_path / "selected failure output"
    assert owner.run_analysis(output_dir=output, suffix="_failure") is None
    csv = output / "ppe_results_failure.csv"
    assert_independent_result(pd.read_csv(csv), "variance")
    assert len(fixture.plots) == fail_at
    assert all(call[1] == str(output) for call in fixture.plots)
    assert "Plotting failed: test-owned plot failure" in capsys.readouterr().out
    assert list(output.iterdir()) == [csv]


def test_omitted_output_retains_exact_historical_directory_without_real_external_write(owner, monkeypatch):
    fixture = install_analysis_inputs_and_plot_stub(owner, monkeypatch)
    directories, serializations = [], []
    monkeypatch.setattr(owner.os, "makedirs", lambda path, **kwargs: directories.append((path, kwargs)))

    def save(frame, path, *args, **kwargs):
        serializations.append((frame.copy(deep=True), path, args, kwargs))

    monkeypatch.setattr(owner.pd.DataFrame, "to_csv", save)
    assert owner.run_analysis() is None
    assert fixture.loads == [(("amph",), {"apply_filter": True})]
    assert directories == [(HISTORICAL_OUTPUT, {"exist_ok": True})]
    assert len(serializations) == 1
    frame, path, args, kwargs = serializations[0]
    assert path == HISTORICAL_OUTPUT + "/ppe_results.csv" and args == () and kwargs == {}
    assert_independent_result(frame, "variance")
    assert [call[1] for call in fixture.plots] == [HISTORICAL_OUTPUT, HISTORICAL_OUTPUT]
