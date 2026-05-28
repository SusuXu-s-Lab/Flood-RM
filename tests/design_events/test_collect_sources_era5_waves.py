import json

import pandas as pd
import pytest
import xarray as xr

from design_events.collect_sources.all_sources import collect_all_sources
from design_events.collect_sources.era5_waves import collect_era5_waves
from design_events.cli import build_parser


def _wave_dataset(times):
    return xr.Dataset(
        {
            "swh": ("valid_time", [1.0] * len(times)),
            "pp1d": ("valid_time", [8.0] * len(times)),
            "mwd": ("valid_time", [90.0] * len(times)),
            "wdw": ("valid_time", [0.4] * len(times)),
        },
        coords={"valid_time": times},
    )


def test_collect_era5_waves_writes_snapwave_source_artifact(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2018-01-01 23:00"),
        "era5_waves": {
            "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
            "output_path": output.as_posix(),
        },
    }

    def fetcher(bbox_wgs84, time_window, output_path, variables=None, force=False):
        calls.append((bbox_wgs84, time_window, output_path, variables, force))
        output_path.parent.mkdir(parents=True)
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)
        return output_path

    result = collect_era5_waves(settings, fetcher=fetcher)

    manifest = json.loads(result["source_artifact_json"].read_text(encoding="utf-8"))
    assert result["wave_netcdf"] == output
    assert calls[0][0] == (-71.0, 42.0, -70.0, 42.5)
    assert calls[0][3] == [
        "significant_height_of_combined_wind_waves_and_swell",
        "peak_wave_period",
        "mean_wave_direction",
        "wave_spectral_directional_width",
    ]
    assert calls[0][4] is True
    assert manifest["study_location"] == "marshfield"
    assert manifest["source"] == "era5"
    assert manifest["kind"] == "snapwave_boundary_forcing"
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["short_variables"] == ["swh", "pp1d", "mwd", "wdw"]


def test_collect_era5_waves_uses_earthdatahub_provider_short_variables(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2018-01-01 23:00"),
        "era5_waves": {
            "provider": "earthdatahub",
            "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
        },
    }

    def fetcher(bbox_wgs84, time_window, output_path, variables=None, force=False):
        calls.append((bbox_wgs84, time_window, output_path, variables, force))
        output_path.parent.mkdir(parents=True)
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)
        return output_path

    result = collect_era5_waves(settings, fetcher=fetcher)

    manifest = json.loads(result["source_artifact_json"].read_text(encoding="utf-8"))
    assert calls[0][3] == ["swh", "pp1d", "mwd", "wdw"]
    assert manifest["metadata"]["provider"] == "earthdatahub"


def test_collect_era5_waves_passes_earthdatahub_auth_path(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2018-01-01 23:00"),
        "era5_waves": {
            "provider": "earthdatahub",
            "auth_path": "code/api-key.txt",
            "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
        },
    }

    def fetcher(
        bbox_wgs84,
        time_window,
        output_path,
        variables=None,
        force=False,
        auth_path=None,
    ):
        calls.append(auth_path)
        output_path.parent.mkdir(parents=True)
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)
        return output_path

    collect_era5_waves(settings, fetcher=fetcher)

    assert calls == [tmp_path / "code/api-key.txt"]


def test_collect_era5_waves_passes_earthdatahub_chunk_months(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2018-01-01 23:00"),
        "era5_waves": {
            "provider": "earthdatahub",
            "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
            "chunk_months": 3,
        },
    }

    def fetcher(
        bbox_wgs84,
        time_window,
        output_path,
        variables=None,
        force=False,
        chunk_months=None,
    ):
        calls.append(chunk_months)
        output_path.parent.mkdir(parents=True)
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)
        return output_path

    collect_era5_waves(settings, fetcher=fetcher)

    assert calls == [3]


def test_collect_era5_waves_does_not_reuse_smoke_limited_artifact(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    output.parent.mkdir(parents=True)
    _wave_dataset(pd.date_range("2018-01-01", periods=24, freq="h")).to_netcdf(output)
    paths["source_artifacts_root"].mkdir()
    (paths["source_artifacts_root"] / "era5_snapwave_boundary_forcing.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2018-01-01T00:00:00",
                "end": "2018-01-01T23:00:00",
                "metadata": {"smoke": True},
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("1979-02-01"),
        "end": pd.Timestamp("1979-02-02"),
        "era5_waves": {"bbox_wgs84": [-71.0, 42.0, -70.0, 42.5]},
    }

    def fetcher(bbox_wgs84, time_window, output_path, variables=None, force=False):
        calls.append((time_window, force))
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)

    collect_era5_waves(settings, skip_existing=True, fetcher=fetcher)

    assert calls == [((pd.Timestamp("1979-02-01"), pd.Timestamp("1979-02-02")), True)]


def test_collect_era5_waves_does_not_reuse_manifest_when_dataset_does_not_cover_window(tmp_path):
    calls = []
    output = tmp_path / "waves/era5_marshfield.nc"
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": output,
    }
    output.parent.mkdir(parents=True)
    _wave_dataset(pd.date_range("2018-01-01", periods=24, freq="h")).to_netcdf(output)
    paths["source_artifacts_root"].mkdir()
    (paths["source_artifacts_root"] / "era5_snapwave_boundary_forcing.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "1979-02-01T00:00:00",
                "end": "2022-12-31T00:00:00",
                "metadata": {"smoke": False},
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "paths": paths,
        "start": pd.Timestamp("1979-02-01"),
        "end": pd.Timestamp("1979-02-02"),
        "era5_waves": {"bbox_wgs84": [-71.0, 42.0, -70.0, 42.5]},
    }

    def fetcher(bbox_wgs84, time_window, output_path, variables=None, force=False):
        calls.append((time_window, force))
        _wave_dataset(pd.date_range(time_window[0], time_window[1], freq="h")).to_netcdf(output_path)

    result = collect_era5_waves(settings, skip_existing=True, fetcher=fetcher)

    assert calls == [((pd.Timestamp("1979-02-01"), pd.Timestamp("1979-02-02")), True)]
    assert result["time_count"] == 25


def test_collect_era5_waves_requires_bbox(tmp_path):
    settings = {
        "paths": {
            "location_name": "marshfield",
            "repo_root": tmp_path,
            "source_artifacts_root": tmp_path / "source_artifacts",
            "era5_waves_nc": tmp_path / "waves.nc",
        },
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2018-01-01 23:00"),
        "era5_waves": {},
    }

    with pytest.raises(ValueError, match="era5_waves.bbox_wgs84 is required"):
        collect_era5_waves(settings)


def test_collect_all_sources_runs_configured_era5_waves(tmp_path):
    calls = []
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-03",
            "era5_waves": {"bbox_wgs84": [-71.0, 42.0, -70.0, 42.5]},
        }
    }
    paths = {"era5_waves_nc": tmp_path / "era5.nc"}

    def collect_era5(settings, skip_existing=False, smoke=False):
        calls.append(
            (
                settings["start"].date().isoformat(),
                settings["end"].date().isoformat(),
                skip_existing,
                smoke,
            )
        )
        return {"wave_netcdf": paths["era5_waves_nc"]}

    result = collect_all_sources(
        config,
        paths,
        skip_existing=True,
        smoke=True,
        funcs={"collect_era5_waves": collect_era5},
    )

    assert calls == [("2020-01-01", "2020-01-03", True, True)]
    assert result["era5_waves"] == {"wave_netcdf": paths["era5_waves_nc"]}


def test_pipeline_accepts_collect_era5_waves_stage():
    args = build_parser().parse_args(
        [
            "collect_era5_waves",
            "--config",
            "locations/marshfield/config.yaml",
            "--smoke",
            "--skip-existing",
        ]
    )

    assert args.stage == "collect_era5_waves"
    assert args.config == "locations/marshfield/config.yaml"
    assert args.smoke is True
    assert args.skip_existing is True
