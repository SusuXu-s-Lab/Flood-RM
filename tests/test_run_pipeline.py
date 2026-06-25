from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_pipeline.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rel(paths, location):
    root = REPO_ROOT / "locations" / location
    return tuple(path.relative_to(root).as_posix() for path in paths)


def test_loads_location_pipeline_manifest():
    runner = load_runner()

    module = runner.load_pipeline_module(REPO_ROOT, "marshfield")

    assert module.GRID_NOTEBOOKS[0] == "01_grid/01_base_network.ipynb"
    assert module.FLOOD_NOTEBOOKS[-1] == "02_flood/07_risk_fiat.ipynb"
    assert module.PIPELINE_NOTEBOOKS["all"] == (*module.GRID_NOTEBOOKS, *module.FLOOD_NOTEBOOKS)


def test_resolves_grid_flood_and_all_order():
    runner = load_runner()

    grid = runner.resolve_notebooks(REPO_ROOT, "greensboro", "grid")
    flood = runner.resolve_notebooks(REPO_ROOT, "greensboro", "flood")
    all_notebooks = runner.resolve_notebooks(REPO_ROOT, "greensboro", "all")

    assert rel(grid, "greensboro") == ("01_grid/sds_plot.ipynb",)
    assert rel(flood, "greensboro")[0] == "02_flood/01_region_setup.ipynb"
    assert rel(flood, "greensboro")[-1] == "02_flood/06_evaluate.ipynb"
    assert all_notebooks == (*grid, *flood)


def test_default_output_path_preserves_location_relative_notebook_path():
    runner = load_runner()
    root = REPO_ROOT / "locations" / "marshfield"
    notebook = root / "02_flood" / "04" / "a_build_waves.ipynb"

    output = runner.output_notebook_path(root, notebook, in_place=False)
    log = runner.log_path(root, notebook)

    assert output == root / "data/pipeline/executed_notebooks/02_flood/04/a_build_waves.ipynb"
    assert log == root / "data/pipeline/logs/02_flood/04/a_build_waves.log"


def test_in_place_output_path_is_source_notebook():
    runner = load_runner()
    root = REPO_ROOT / "locations" / "austin"
    notebook = root / "01_grid" / "sds_plot.ipynb"

    assert runner.output_notebook_path(root, notebook, in_place=True) == notebook


def test_unknown_location_and_stage_have_helpful_errors():
    runner = load_runner()

    with pytest.raises(runner.PipelineError, match="Unknown location"):
        runner.resolve_notebooks(REPO_ROOT, "does-not-exist", "all")
    with pytest.raises(runner.PipelineError, match="Unknown stage"):
        runner.resolve_notebooks(REPO_ROOT, "austin", "ops")


def test_missing_manifest_has_helpful_error(tmp_path):
    runner = load_runner()
    (tmp_path / "locations" / "empty").mkdir(parents=True)

    with pytest.raises(runner.PipelineError, match="Missing pipeline manifest"):
        runner.resolve_notebooks(tmp_path, "empty", "all")


def test_dry_run_prints_order_without_creating_outputs(capsys):
    runner = load_runner()
    args = Namespace(
        location="austin",
        stage="grid",
        in_place=False,
        dry_run=True,
        kernel="python3",
        timeout=None,
        stop_on_error=True,
    )

    code = runner.run_pipeline(args, repo_root=REPO_ROOT)

    assert code == 0
    captured = capsys.readouterr()
    assert "austin:grid" in captured.out
    assert "01_grid/sds_plot.ipynb -> data/pipeline/executed_notebooks/01_grid/sds_plot.ipynb" in captured.out
