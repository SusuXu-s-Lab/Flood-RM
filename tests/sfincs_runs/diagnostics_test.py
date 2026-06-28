import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")

from sfincs_runs.diagnostics import FloodResponseDiagnostics, driver_response_diagnostics, plot_flood_response_diagnostics, plot_forcing


def test_plot_forcing_plots_point_discharge_sources(tmp_path):
    run_root = tmp_path / "scenario"
    run_root.mkdir()
    discharge_path = tmp_path / "sfincs_discharge.nc"
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    ds = xr.Dataset(
        {
            "discharge": (
                ("index", "time"),
                np.array([[1.0, 2.0, 3.0], [2.0, 3.5, 4.0]], dtype=float),
            )
        },
        coords={
            "index": [1, 2],
            "time": times,
            "name": ("index", ["inflow_01", "inflow_02"]),
            "x": ("index", [100.0, 200.0]),
            "y": ("index", [1000.0, 1100.0]),
        },
    )
    ds.to_netcdf(discharge_path)
    (run_root / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "event_id": "evt_001",
                "forcing_mode": "dual_fluvial_pluvial",
                "direct_rainfall_enabled": True,
                "wflow_discharge_forcing": str(discharge_path),
            }
        ),
        encoding="utf-8",
    )
    np.array([2.0, 4.0], dtype="<f4").tofile(run_root / "sfincs.smax")
    np.array([1.0, 2.0], dtype="<f4").tofile(run_root / "sfincs.seff")
    np.array([5.0, 10.0], dtype="<f4").tofile(run_root / "sfincs.ks")

    out_path = plot_forcing(
        forcing_manifest=run_root / "forcing_manifest.json",
        event_id="evt_001",
        event_label="evt_001",
    )

    assert Path(out_path).exists()


def test_flood_response_plot_keeps_historical_tail_outlier_off_scale():
    flood = pd.DataFrame(
        {
            "event_id": ["design_001", "design_002", "historical_001"],
            "severity_band": ["common", "extreme", "beyond_design"],
            "storm_type": ["nor_easter", "nor_easter", "other_non_tropical"],
            "sample_rp_years": [5.0, 475.0, 3.9e17],
            "probability_weight": [0.4, 0.001, np.nan],
            "anytime_incremental_flooded_area_km2": [0.2, 3.5, 4.0],
            "peak_incremental_flooded_area_km2": [0.1, 2.8, 3.1],
            "rainfall_metric_mm": [25.0, 90.0, 120.0],
            "coastal_peak_m": [0.5, 1.1, 1.4],
        }
    )
    response = FloodResponseDiagnostics(flood, pd.Series(dtype=object), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    fig = plot_flood_response_diagnostics(response)

    try:
        _, high = fig.axes[0].get_xlim()
        assert high < 1_000.0
    finally:
        plt.close(fig)


def test_driver_response_diagnostics_drops_all_nan_lag_driver(tmp_path):
    outcomes = pd.DataFrame(
        {
            "event_id": [f"event_{i}" for i in range(12)],
            "storm_type": ["nor_easter"] * 12,
            "probability_weight": np.full(12, 1 / 12),
            "coastal_water_level": np.linspace(0.2, 1.4, 12),
            "rainfall": np.linspace(10.0, 80.0, 12),
            "rainfall_pairing_lag_hours": [np.nan] * 12,
            "peak_incremental_land_depth_m": np.linspace(0.0, 1.1, 12),
            "peak_incremental_flooded_area_km2": np.linspace(0.0, 3.0, 12),
            "anytime_incremental_flooded_area_km2": np.linspace(0.0, 4.0, 12),
            "longest_incremental_flood_duration_h": np.linspace(1.0, 24.0, 12),
        }
    )

    response = driver_response_diagnostics(outcomes, outdir=tmp_path, min_rows=8)

    assert "rainfall_pairing_lag_hours" not in response.driver_columns
    assert response.associations_path.exists()
