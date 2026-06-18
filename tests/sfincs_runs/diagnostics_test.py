import json

import numpy as np
import pandas as pd
import xarray as xr

from sfincs_runs.diagnostics import (
    _resolve_run_start,
    plot_inland_coupled_forcing_qa,
    plot_inland_coupled_postrun_diagnostics,
)


def test_resolve_run_start_prefers_forcing_manifest(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "forcing_manifest.json").write_text(
        json.dumps({"run_start": "2018-01-01 17:00:00"}),
        encoding="utf-8",
    )
    (run_root / "sfincs.inp").write_text("tstart = 20000101 000000\n", encoding="utf-8")

    assert _resolve_run_start(run_root, fallback="1999-01-01") == pd.Timestamp("2018-01-01 17:00:00")


def test_resolve_run_start_falls_back_to_sfincs_inp(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "sfincs.inp").write_text("tstart = 20180101 170000\n", encoding="utf-8")

    assert _resolve_run_start(run_root, fallback="1999-01-01") == pd.Timestamp("2018-01-01 17:00:00")


def test_plot_inland_coupled_forcing_qa_handles_missing_discharge(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    run_root = location_root / "data" / "sfincs" / "scenarios" / "evt_001"
    run_root.mkdir(parents=True)
    (location_root / "config.yaml").write_text("project: {name: greensboro}\n", encoding="utf-8")
    (run_root / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "event_id": "evt_001",
                "forcing_mode": "dual_fluvial_pluvial",
                "wflow_discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                "direct_rainfall_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    out = plot_inland_coupled_forcing_qa(forcing_manifest=run_root / "forcing_manifest.json")

    assert out.exists()
    assert out.name == "evt_001_inland_forcing_qa.png"


def test_plot_inland_coupled_forcing_qa_reads_location_relative_discharge(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    run_root = location_root / "data" / "sfincs" / "scenarios" / "evt_001"
    discharge = location_root / "data" / "wflow" / "events" / "evt_001" / "sfincs_discharge.nc"
    run_root.mkdir(parents=True)
    discharge.parent.mkdir(parents=True)
    (location_root / "config.yaml").write_text("project: {name: greensboro}\n", encoding="utf-8")
    times = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
    ds = xr.Dataset(
        {"discharge": (("time", "source"), np.array([[1.0], [3.0], [2.0]]))},
        coords={"time": times, "source": [0]},
    )
    ds.to_netcdf(discharge)
    (run_root / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "event_id": "evt_001",
                "forcing_mode": "dual_fluvial_pluvial",
                "wflow_discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                "direct_rainfall_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    out = plot_inland_coupled_forcing_qa(forcing_manifest=run_root / "forcing_manifest.json")

    assert out.exists()


def test_plot_inland_coupled_postrun_diagnostics_handles_pending_outputs(tmp_path):
    run_root = tmp_path / "evt_001"
    run_root.mkdir()
    (run_root / "forcing_manifest.json").write_text(json.dumps({"event_id": "evt_001"}), encoding="utf-8")

    out = plot_inland_coupled_postrun_diagnostics(run_root=run_root)

    assert out.exists()
