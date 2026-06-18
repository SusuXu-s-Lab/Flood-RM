from pathlib import Path

import xarray as xr

from sfincs_runs.scenarios.scenario_stats import (
    cell_area_m2,
    completed_event_inventory,
    event_dirs,
    event_id,
    parse_args,
)


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_scenario_stats_accepts_location_config_for_defaults():
    args = parse_args(["--config", "locations/marshfield/config.yaml"])

    assert suffix(args.scenarios_dir) == "locations/marshfield/data/sfincs/scenarios"
    assert suffix(args.storage_dir) == "locations/marshfield/data/sfincs/run_outputs"
    assert suffix(args.stats_dir) == "locations/marshfield/data/sfincs/stats"


def test_scenario_stats_selects_design_event_ids(tmp_path):
    scenarios = tmp_path / "scenarios"
    (scenarios / "design_0001").mkdir(parents=True)
    (scenarios / "evt_0001").mkdir()
    (scenarios / ".ipynb_checkpoints").mkdir()

    assert event_id("design_0001") == "design_0001"
    assert [path.name for path in event_dirs(scenarios, ids=["design_0001"])] == ["design_0001"]
    assert [path.name for path in event_dirs(scenarios)] == ["design_0001", "evt_0001"]


def test_completed_event_inventory_falls_back_to_self_contained_run_outputs(tmp_path):
    scenarios = tmp_path / "scenarios"
    storage = tmp_path / "run_outputs"
    for event in ["design_0001", "design_0002"]:
        (scenarios / event).mkdir(parents=True)

    for i in range(1, 6):
        event_dir = storage / f"design_{i:04d}"
        event_dir.mkdir(parents=True)
        for name in ["sfincs_map.nc", "sfincs.inp", "sfincs.bzs", "forcing_manifest.json"]:
            (event_dir / name).write_text("", encoding="utf-8")

    inventory = completed_event_inventory(scenarios, storage)

    assert inventory["use_storage_outputs"] is True
    assert inventory["event_source_root"] == storage
    assert [path.name for path in inventory["scenario_completed"]] == ["design_0001", "design_0002"]
    assert [path.name for path in inventory["completed_events"]] == [
        "design_0001",
        "design_0002",
        "design_0003",
        "design_0004",
        "design_0005",
    ]

    one_event = completed_event_inventory(scenarios, storage, ids=["design_0005"])
    assert one_event["use_storage_outputs"] is True
    assert [path.name for path in one_event["completed_events"]] == ["design_0005"]


def test_cell_area_uses_quadtree_grid_attrs(tmp_path):
    event_dir = tmp_path / "design_0001"
    event_dir.mkdir()
    ds = xr.Dataset(
        {"level": ("mesh2d_nFaces", [1, 1])},
        attrs={"dx": 60.0, "dy": 60.0},
    )
    ds.to_netcdf(event_dir / "sfincs.nc")

    assert cell_area_m2(event_dir, {"qtrfile": "sfincs.nc"}) == 3600.0
