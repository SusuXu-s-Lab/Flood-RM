import pytest
import pandas as pd
from pathlib import Path

import sfincs_runs.scenarios.event_forcing as event_forcing
from sfincs_runs.scenarios.event_forcing import (
    EventForcing,
    _catalog_rainfall_start,
    _precip_source_name,
    _validate_catalog_precip_timing,
)


def test_synthetic_copula_precip_timing_requires_catalog_start_offset():
    catalog = {
        "event_id": "design_0001",
        "event_origin": "synthetic_tail",
        "forcing_pairing_policy": "copula_joint",
        "rainfall_member_id": "rain_001",
        "event_reference_time": "2020-01-01T00:00:00",
    }

    with pytest.raises(RuntimeError, match="requires event_reference_time"):
        _validate_catalog_precip_timing(catalog, _catalog_rainfall_start(catalog))


def test_synthetic_copula_precip_timing_accepts_catalog_start_offset():
    catalog = {
        "event_id": "design_0001",
        "event_origin": "synthetic_tail",
        "forcing_pairing_policy": "copula_joint",
        "rainfall_member_id": "rain_001",
        "event_reference_time": "2020-01-01T00:00:00",
        "rainfall_start_offset_hours": -30.0,
    }

    _validate_catalog_precip_timing(catalog, _catalog_rainfall_start(catalog))


def test_precip_source_name_is_event_specific_and_catalog_safe():
    assert _precip_source_name("design_0015") == "event_precip_design_0015"
    assert _precip_source_name("historical/2006-05-13T06:00") == "event_precip_historical_2006_05_13T06_00"


def test_stage_precip_clears_hydromt_precipitation_component_before_create(monkeypatch, tmp_path):
    calls = []

    class FakeRoot:
        path = tmp_path
        mode = "r+"

        def set(self, path, mode):
            self.path = Path(path)
            self.mode = mode

    class FakeConfig:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            calls.append(("config.set", key, value))
            self.values[key] = value

        def get(self, key, default=None):
            return self.values.get(key, default)

    class FakeDataCatalog:
        def from_dict(self, data):
            calls.append(("data_catalog.from_dict", tuple(data)))

    class FakePrecipitation:
        def clear(self):
            calls.append(("precipitation.clear",))

        def create(self, **kwargs):
            calls.append(("precipitation.create", kwargs["precip"]))

        def write(self):
            calls.append(("precipitation.write",))

    class FakeSfincs:
        root = FakeRoot()
        config = FakeConfig()
        data_catalog = FakeDataCatalog()
        precipitation = FakePrecipitation()

    monkeypatch.setattr(
        event_forcing,
        "hydrology_inputs",
        lambda forcing, paths, config: {
            "rainfall_source_nc": str(tmp_path / "source.nc"),
            "rainfall_member_id": "rain_001",
            "rainfall_member_time": "2000-01-01T00:00:00",
            "rainfall_storm_start": "2000-01-01 00:00:00",
            "soil_moisture_summary": None,
        },
    )
    monkeypatch.setattr(
        event_forcing,
        "prepare_aorc_precip_for_sfincs",
        lambda *args, **kwargs: tmp_path / "aorc_precip_for_sfincs.nc",
    )
    monkeypatch.setattr(event_forcing, "_stage_event_soil_moisture", lambda run_root, hydrology: {})

    forcing = EventForcing(
        event_id="design_0015",
        catalog={
            "event_id": "design_0015",
            "event_origin": "synthetic_tail",
            "forcing_pairing_policy": "copula_joint",
            "rainfall_member_id": "rain_001",
            "event_reference_time": "2000-01-04T00:00:00",
            "rainfall_start_offset_hours": -72.0,
            "rainfall_scale_factor": 1.0,
        },
        h=pd.Series([0.0, 1.0]),
        forcing_variable="coastal_water_level",
        t_start=pd.Timestamp("2000-01-01"),
        t_stop=pd.Timestamp("2000-01-02"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge_event_members.nc",
    )

    event_forcing.stage_precip(FakeSfincs(), tmp_path, forcing, paths={}, config={})

    clear_index = calls.index(("precipitation.clear",))
    create_index = next(i for i, call in enumerate(calls) if call[0] == "precipitation.create")
    assert clear_index < create_index
    assert calls[create_index] == ("precipitation.create", "event_precip_design_0015")
