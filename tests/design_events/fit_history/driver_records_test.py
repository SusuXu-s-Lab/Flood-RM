from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from design_events.runtime import load_runtime
from design_events.fit_history.driver_records import (
    assemble_paired_observations,
    assemble_paired_observations_from_config,
    build_member_libraries,
    load_driver_series,
    member_library_from_records,
    record_specs_from_config,
)

REPO = Path(__file__).resolve().parents[3]
GREENSBORO = REPO / "locations/greensboro"


def _write_synthetic_records(root):
    # rainfall: per-storm basin-mean depths (sparse storm-event series, AORC SST shape)
    storms = pd.date_range("2000-06-01", periods=120, freq="11D")
    rng = np.random.default_rng(0)
    (root / "rain").mkdir(parents=True)
    pd.DataFrame({"storm_date": storms.astype(str), "mean": rng.gamma(2.0, 25.0, len(storms)), "max": 0.0}).to_csv(
        root / "rain/storm-stats.csv", index=False
    )
    # soil moisture: multi-point rows per timestamp (NWM shape) -> aggregated to one/time;
    # span the full rainfall period so co-occurrence exists for both conditioning directions
    times = pd.date_range("2000-01-01", periods=1300, freq="1D")
    rows = []
    for t in times:
        for pid in range(3):
            rows.append({"time": str(t), "SOILSAT_TOP": rng.uniform(0.2, 0.6), "point_id": pid})
    (root / "soil").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "soil/soil_moisture.csv", index=False)
    return {
        "rainfall": {"path": "rain/storm-stats.csv", "time_column": "storm_date", "value_column": "mean"},
        "soil_moisture": {"path": "soil/soil_moisture.csv", "time_column": "time", "value_column": "SOILSAT_TOP", "aggregate": "mean"},
    }


def test_load_driver_series_aligns_and_aggregates(tmp_path):
    specs = _write_synthetic_records(tmp_path)
    series = load_driver_series(specs, location_root=tmp_path)
    assert set(series) == {"rainfall", "soil_moisture"}
    assert isinstance(series["soil_moisture"].index, pd.DatetimeIndex)
    # multi-point soil rows collapsed to one value per timestamp
    assert series["soil_moisture"].index.is_unique
    assert series["rainfall"].index.is_monotonic_increasing


def test_load_driver_series_uses_configured_rainfall_peak_time(tmp_path):
    pd.DataFrame(
        {
            "storm_date": ["2000-01-01T00:00:00"],
            "rainfall_peak_time": ["2000-01-02T06:00:00"],
            "mean": [55.0],
        }
    ).to_csv(tmp_path / "storm-stats.csv", index=False)

    series = load_driver_series(
        {"rainfall": {"path": "storm-stats.csv", "time_column": "rainfall_peak_time", "value_column": "mean"}},
        location_root=tmp_path,
    )

    assert series["rainfall"].index.tolist() == [pd.Timestamp("2000-01-02T06:00:00")]


def test_load_driver_series_fails_when_configured_time_column_is_stale(tmp_path):
    pd.DataFrame({"storm_date": ["2000-01-01T00:00:00"], "mean": [55.0]}).to_csv(
        tmp_path / "storm-stats.csv", index=False
    )

    with pytest.raises(ValueError, match="rainfall_peak_time"):
        load_driver_series(
            {"rainfall": {"path": "storm-stats.csv", "time_column": "rainfall_peak_time", "value_column": "mean"}},
            location_root=tmp_path,
        )


def test_assemble_paired_observations_from_records(tmp_path):
    specs = _write_synthetic_records(tmp_path)
    paired = assemble_paired_observations(
        specs, ["rainfall", "soil_moisture"], location_root=tmp_path,
        threshold_quantiles=0.85, decluster_window_hours=240.0, pairing_window_hours=120.0,
    )
    assert list(paired.columns) == [
        "event_time", "conditioned_on", "rainfall", "soil_moisture", "rainfall_time", "soil_moisture_time"
    ]
    assert paired[["rainfall_time", "soil_moisture_time"]].notna().all().all()
    assert set(paired["conditioned_on"].unique()) == {"rainfall", "soil_moisture"}
    assert len(paired) > 10


def test_missing_record_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_driver_series({"rainfall": {"path": "nope.csv", "time_column": "t", "value_column": "v"}}, location_root=tmp_path)


def _config_with_synthetic(root):
    specs = _write_synthetic_records(root)
    return {
        "event_catalog": {
            "forcing_members": {"rainfall": "rain/storm-stats.csv"},
            "dependence": {
                "driver_vector": ["rainfall", "soil_moisture"],
                "cooccurrence": {"threshold_quantile": 0.85, "decluster_window_hours": 240, "pairing_window_hours": 120},
                "driver_records": specs,
                "member_libraries": {
                    "rainfall": {"from": "member_table"},
                    "soil_moisture": {"from": "records", "index_column": "soil_moisture_mean"},
                },
            },
        }
    }


def test_assemble_paired_observations_from_config_is_modular(tmp_path):
    config = _config_with_synthetic(tmp_path)
    assert set(record_specs_from_config(config)) == {"rainfall", "soil_moisture"}
    paired = assemble_paired_observations_from_config(config, location_root=tmp_path)
    assert set(paired["conditioned_on"].unique()) == {"rainfall", "soil_moisture"}
    assert len(paired) > 10


def test_build_member_libraries_member_table_and_records(tmp_path):
    config = _config_with_synthetic(tmp_path)
    libs = build_member_libraries(config, location_root=tmp_path)
    assert set(libs) == {"rainfall", "soil_moisture"}
    # rainfall came from the curated member table; soil was built per-timestamp from records
    assert "mean" in libs["rainfall"].columns
    for column in ["member_id", "member_file", "time", "soil_moisture_mean"]:
        assert column in libs["soil_moisture"].columns
    assert libs["soil_moisture"]["member_id"].is_unique


def test_build_member_libraries_drops_coastal_members_outside_wave_collection(tmp_path):
    jan_peak = pd.Timestamp("1979-01-01 17:00:00")
    feb_peak = pd.Timestamp("1979-02-10 17:00:00")
    times = pd.date_range("1978-12-25 00:00:00", "1979-02-18 00:00:00", freq="h")
    values = pd.Series(0.0, index=times)
    values.loc[jan_peak] = 10.0
    values.loc[feb_peak] = 9.0
    pd.DataFrame({"time": times.astype(str), "value": values.to_numpy(dtype=float)}).to_csv(
        tmp_path / "waterlevel.csv",
        index=False,
    )
    config = {
        "coastal_waves": True,
        "collection": {"era5_waves": {"start_date": "1979-02-01", "end_date": "1979-02-28"}},
        "design_events": {"tide_resolving_half_window_hours": 72},
        "event_catalog": {
            "dependence": {
                "driver_vector": ["coastal_water_level"],
                "driver_records": {
                    "coastal_water_level": {
                        "path": "waterlevel.csv",
                        "time_column": "time",
                        "value_column": "value",
                    },
                },
                "member_libraries": {
                    "coastal_water_level": {
                        "from": "records",
                        "index_column": "coastal_peak_m",
                        "decluster_window_hours": 120,
                        "threshold_quantile": 0.95,
                    },
                },
            },
        },
    }

    libs = build_member_libraries(config, location_root=tmp_path)

    coastal = libs["coastal_water_level"]
    assert coastal["member_id"].tolist() == ["coastal_water_level_19790210T170000"]
    assert coastal["time"].tolist() == ["1979-02-10T17:00:00"]


def test_member_library_from_records_collapses_per_timestamp():
    times = pd.date_range("2000-01-01", periods=10, freq="1D")
    records = pd.DataFrame({"time": np.repeat(times.astype(str), 3), "v": np.arange(30.0)})
    lib = member_library_from_records(records, value_column="v", time_column="time", index_column="x_mean",
                                      aggregate="mean", id_prefix="x", member_file="/tmp/x.csv")
    assert len(lib) == 10
    assert (lib["member_file"] == "/tmp/x.csv").all()


@pytest.mark.skipif(not (GREENSBORO / "config.yaml").exists(), reason="greensboro workspace not present")
def test_real_greensboro_records_load_and_pair():
    # Integration against the real collected records, driven entirely by the greensboro config.
    config, _ = load_runtime(GREENSBORO / "config.yaml")
    specs = record_specs_from_config(config)
    for driver in ("streamflow", "rainfall"):
        if not (GREENSBORO / specs[driver]["path"]).exists():
            pytest.skip(f"real {driver} record not collected at {specs[driver]['path']}")
    paired = assemble_paired_observations_from_config(config, location_root=GREENSBORO)
    assert not paired.empty
    assert {"streamflow", "rainfall"}.issubset(paired.columns)
    assert set(paired["conditioned_on"].unique()) == {"streamflow"}
