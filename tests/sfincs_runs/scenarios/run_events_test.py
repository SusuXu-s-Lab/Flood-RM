from pathlib import Path
from types import SimpleNamespace
import json
import sys

from sfincs_runs.scenarios.run_events import parse_args, run_one, save_outputs, stage_event


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_run_events_accepts_location_config_for_defaults():
    args = parse_args(["--config", "locations/marshfield/config.yaml"])

    assert suffix(args.scenarios_dir) == "locations/marshfield/data/sfincs/scenarios"
    assert suffix(args.storage_dir) == "locations/marshfield/data/sfincs/run_outputs"
    assert suffix(args.run_root) == "locations/marshfield/data/sfincs/run_stage"
    assert args.sfincs_bin == "/usr/local/bin/sfincs"


def test_save_outputs_keeps_compact_run_receipts(tmp_path):
    stage_dir = tmp_path / "stage" / "evt_0001"
    storage_dir = tmp_path / "storage" / "evt_0001"
    stage_dir.mkdir(parents=True)

    retained = {
        "sfincs_map.nc",
        "sfincs_his.nc",
        "sfincs.log",
        "sfincs_log.txt",
        "sfincs.inp",
        "forcing_manifest.json",
        "sfincs.bnd",
        "sfincs.bzs",
        "snapwave.bnd",
        "snapwave.bhs",
        "snapwave.btp",
        "snapwave.bwd",
        "snapwave.bds",
        "sfincs.rug",
        "sfincs.rug.obs",
        "sfincs.obs",
        "sfincs.weir",
        "sfincs.thd",
    }
    expensive_or_rebuildable = {
        "sfincs_rst.nc",
        "sfincs_netampr.nc",
        "aorc_precip_for_sfincs.nc",
        "snapwave.upw",
        "sfincs.nc",
        "sfincs_subgrid.nc",
        "flood_ocean_animation.mp4",
    }
    for name in retained | expensive_or_rebuildable:
        (stage_dir / name).write_text(name, encoding="utf-8")

    save_outputs(stage_dir, storage_dir, {"event_id": "evt_0001"})

    for name in retained:
        assert (storage_dir / name).exists(), f"missing retained output: {name}"
    for name in expensive_or_rebuildable:
        assert not (storage_dir / name).exists(), f"unexpected bulky output retained: {name}"
    assert (storage_dir / "run_metadata.json").exists()


def test_run_one_expands_stage_dir_placeholder_for_container_bind(tmp_path):
    event_dir = tmp_path / "scenarios" / "evt_0001"
    event_dir.mkdir(parents=True)
    (event_dir / "sfincs.inp").write_text("input", encoding="utf-8")

    args = SimpleNamespace(
        storage_dir=tmp_path / "storage",
        run_root=tmp_path / "stage",
        force_rerun=True,
        dry_run=False,
        keep_stage=False,
    )
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "Path('sfincs_map.nc').write_text('ok'); "
        "Path('stage_seen.txt').write_text('{stage_dir}')",
    ]

    result = run_one(event_dir, args, command)

    metadata = json.loads((args.storage_dir / "evt_0001" / "run_metadata.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert str(args.run_root / "evt_0001") in metadata["runner_command"][2]


def test_stage_event_keeps_required_subdirectories(tmp_path):
    src = tmp_path / "scenarios" / "evt_0001"
    dst = tmp_path / "stage" / "evt_0001"
    (src / "subgrid").mkdir(parents=True)
    (src / "gis").mkdir()
    (src / "sfincs.inp").write_text("input", encoding="utf-8")
    (src / "subgrid" / "dep_subgrid_lev0.tif").write_text("dep", encoding="utf-8")
    (src / "gis" / "region.geojson").write_text("{}", encoding="utf-8")
    (src / "sfincs.log").write_text("old log", encoding="utf-8")

    stage_event(src, dst)

    assert (dst / "sfincs.inp").exists()
    assert (dst / "subgrid" / "dep_subgrid_lev0.tif").exists()
    assert (dst / "gis" / "region.geojson").exists()
    assert not (dst / "sfincs.log").exists()
