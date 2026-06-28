import os
import subprocess
import sys
from pathlib import Path

from sfincs_runs.scenarios import scenario_stats


def test_scenario_stats_import_does_not_require_selected_location():
    env = os.environ.copy()
    env.pop("FLOOD_RM_LOCATION", None)
    env.pop("FLOOD_RM_LOCATION_CONFIG", None)
    source_root = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", "import sfincs_runs.scenarios.scenario_stats"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_load_scenario_build_uses_explicit_design_outputs_fallback(tmp_path):
    summary, report = scenario_stats.load_scenario_build(
        tmp_path,
        design_outputs_root=tmp_path / "design_outputs",
    )

    assert summary["design_outputs_root"] == str(tmp_path / "design_outputs")
    assert summary["design_scenario"] == "base"
    assert report == {}


def test_load_design_events_uses_explicit_design_outputs_fallback(tmp_path):
    rows, attrs = scenario_stats.load_design_events(
        {"design_scenario": "future"},
        design_outputs_root=tmp_path / "design_outputs",
    )

    assert rows == {}
    assert attrs == {"scenario_name": "future", "slr_offset_m": None}
