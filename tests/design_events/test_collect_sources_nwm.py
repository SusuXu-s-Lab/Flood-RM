import json

import pandas as pd
import xarray as xr

from design_events.collect_sources.nwm import collect_nwm, collect_soil_moisture


def test_collect_nwm_writes_empty_configured_artifacts_and_manifest(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "nwm_root": tmp_path / "nwm",
        "nwm_streamflow_csv": tmp_path / "nwm/streamflow.csv",
        "nwm_soil_moisture_csv": tmp_path / "nwm/soil_moisture.csv",
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-02"),
        "nwm": {
            "version": "3.0",
            "bucket": "noaa-nwm-retrospective-3-0-pds",
            "streamflow": {
                "zarr": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                "feature_ids": [],
            },
            "soil_moisture": {
                "zarr": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr",
                "points": [],
            },
        },
    }

    result = collect_nwm(settings)

    assert result["reused"] is False
    assert result["streamflow_rows"] == 0
    assert result["soil_moisture_rows"] == 0
    assert paths["nwm_streamflow_csv"].exists()
    assert paths["nwm_soil_moisture_csv"].exists()
    manifest = json.loads((paths["source_artifacts_root"] / "nwm_retrospective_hydrologic_state.json").read_text())
    assert manifest["source"] == "nwm"
    assert manifest["kind"] == "retrospective_hydrologic_state"
    assert manifest["start"] == "2020-01-01T00:00:00"
    assert manifest["end"] == "2020-01-02T00:00:00"
    assert manifest["metadata"]["version"] == "3.0"
    assert manifest["metadata"]["bucket"] == "noaa-nwm-retrospective-3-0-pds"
    assert manifest["metadata"]["streamflow_zarr"] == "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
    assert manifest["metadata"]["soil_moisture_zarr"] == "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr"
    assert manifest["metadata"]["soil_moisture_variables"] == []


def test_collect_nwm_reuses_existing_only_when_manifest_covers_requested_window(tmp_path, monkeypatch):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "nwm_root": tmp_path / "nwm",
        "nwm_streamflow_csv": tmp_path / "nwm/streamflow.csv",
        "nwm_soil_moisture_csv": tmp_path / "nwm/soil_moisture.csv",
    }
    paths["nwm_root"].mkdir()
    paths["source_artifacts_root"].mkdir()
    pd.DataFrame(columns=["time", "feature_id", "streamflow"]).to_csv(paths["nwm_streamflow_csv"], index=False)
    pd.DataFrame({"time": ["2020-01-01"], "point_id": ["center"], "SOIL_M": [0.2]}).to_csv(paths["nwm_soil_moisture_csv"], index=False)
    (paths["source_artifacts_root"] / "nwm_retrospective_hydrologic_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-31T00:00:00",
                "metadata": {"smoke": False},
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-31"),
        "nwm": {"streamflow": {}, "soil_moisture": {"variables": ["SOIL_M"]}},
    }

    def fail_collect_streamflow(settings):
        raise AssertionError("should reuse existing NWM streamflow")

    monkeypatch.setattr("design_events.collect_sources.nwm.collect_streamflow", fail_collect_streamflow)

    result = collect_nwm(settings, skip_existing=True)

    assert result["reused"] is True
    assert result["soil_moisture_rows"] == 1


def test_collect_nwm_does_not_reuse_when_soil_csv_missing_non_derivable_variables(tmp_path, monkeypatch):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "nwm_root": tmp_path / "nwm",
        "nwm_streamflow_csv": tmp_path / "nwm/streamflow.csv",
        "nwm_soil_moisture_csv": tmp_path / "nwm/soil_moisture.csv",
    }
    paths["nwm_root"].mkdir()
    paths["source_artifacts_root"].mkdir()
    pd.DataFrame(columns=["time", "feature_id", "streamflow"]).to_csv(paths["nwm_streamflow_csv"], index=False)
    pd.DataFrame({"time": ["2020-01-01"], "point_id": ["center"], "soil_moisture": [0.2]}).to_csv(paths["nwm_soil_moisture_csv"], index=False)
    (paths["source_artifacts_root"] / "nwm_retrospective_hydrologic_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-31T00:00:00",
                "metadata": {"smoke": False},
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-31"),
        "nwm": {"streamflow": {}, "soil_moisture": {"variables": ["SOIL_M", "SOILSAT_TOP"]}},
    }

    monkeypatch.setattr(
        "design_events.collect_sources.nwm.collect_streamflow",
        lambda settings: pd.DataFrame(columns=["time", "feature_id", "streamflow"]),
    )
    monkeypatch.setattr(
        "design_events.collect_sources.nwm.collect_soil_moisture",
        lambda settings: pd.DataFrame(
            {
                "time": ["2020-01-01"],
                "point_id": ["center"],
                "SOIL_M": [0.2],
                "SOILSAT_TOP": [0.7],
            }
        ).to_csv(paths["nwm_soil_moisture_csv"], index=False)
        or pd.read_csv(paths["nwm_soil_moisture_csv"]),
    )

    result = collect_nwm(settings, skip_existing=True)

    assert result["reused"] is False
    assert "SOILSAT_TOP" in pd.read_csv(paths["nwm_soil_moisture_csv"], nrows=1).columns


def test_collect_nwm_repairs_derivable_soilsat_top_without_recollecting(tmp_path, monkeypatch):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "nwm_root": tmp_path / "nwm",
        "nwm_streamflow_csv": tmp_path / "nwm/streamflow.csv",
        "nwm_soil_moisture_csv": tmp_path / "nwm/soil_moisture.csv",
    }
    paths["nwm_root"].mkdir()
    paths["source_artifacts_root"].mkdir()
    pd.DataFrame(columns=["time", "feature_id", "streamflow"]).to_csv(paths["nwm_streamflow_csv"], index=False)
    pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "soil_layers_stag": [0, 1, 2],
            "SOIL_M": [0.2, 0.6, 0.9],
            "x": [10.0, 10.0, 10.0],
            "y": [20.0, 20.0, 20.0],
            "point_id": ["center", "center", "center"],
        }
    ).to_csv(paths["nwm_soil_moisture_csv"], index=False)
    (paths["source_artifacts_root"] / "nwm_retrospective_hydrologic_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-31T00:00:00",
                "metadata": {"smoke": False},
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-31"),
        "nwm": {
            "streamflow": {},
            "soil_moisture": {
                "variables": ["SOIL_M", "SOILSAT_TOP"],
                "soilsat_top_layers": [0, 1],
            },
        },
    }

    def fail_collect_streamflow(settings):
        raise AssertionError("should reuse existing NWM streamflow")

    def fail_collect_soil_moisture(settings):
        raise AssertionError("should repair existing soil moisture CSV")

    monkeypatch.setattr("design_events.collect_sources.nwm.collect_streamflow", fail_collect_streamflow)
    monkeypatch.setattr("design_events.collect_sources.nwm.collect_soil_moisture", fail_collect_soil_moisture)

    result = collect_nwm(settings, skip_existing=True)
    repaired = pd.read_csv(paths["nwm_soil_moisture_csv"])

    assert result["reused"] is True
    assert repaired["SOILSAT_TOP"].tolist() == [0.4, 0.4, 0.4]
    assert set(repaired["SOILSAT_TOP_source"]) == {"derived_from_SOIL_M_layers_0_1"}


def test_collect_soil_moisture_derives_soilsat_top_from_soil_m_when_missing(tmp_path):
    paths = {
        "nwm_root": tmp_path / "nwm",
        "nwm_soil_moisture_csv": tmp_path / "nwm/soil_moisture.csv",
    }
    ds = xr.Dataset(
        data_vars={
            "SOIL_M": (
                ("time", "soil_layers_stag", "y", "x"),
                [[[[0.2]], [[0.6]], [[0.9]]]],
            )
        },
        coords={
            "time": [pd.Timestamp("2020-01-01")],
            "soil_layers_stag": [0, 1, 2],
            "x": [10.0],
            "y": [20.0],
        },
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-02"),
        "nwm": {
            "soil_moisture": {
                "zarr": "memory://nwm",
                "variables": ["SOIL_M", "SOILSAT_TOP"],
                "soilsat_top_layers": [0, 1],
                "x": "x",
                "y": "y",
                "points": [{"id": "center", "x": 10.0, "y": 20.0}],
            }
        },
    }

    frame = collect_soil_moisture(settings, open_zarr=lambda *args, **kwargs: ds)

    assert "SOILSAT_TOP" in frame.columns
    assert "SOILSAT_TOP_source" in frame.columns
    assert frame["SOILSAT_TOP"].tolist() == [0.4, 0.4, 0.4]
    assert set(frame["SOILSAT_TOP_source"]) == {"derived_from_SOIL_M_layers_0_1"}
