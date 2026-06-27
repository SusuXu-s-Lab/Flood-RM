import json

import numpy as np
import pandas as pd
import xarray as xr

from design_events.collect_sources.aorc_sst import (
    collect_aorc_sst_event_windows,
    _aorc_sst_artifact_covers_settings,
    _compute_selected_event_windows,
    _ensure_transposition_targets,
    _event_window_file_has_required_variables,
    _strip_netcdf_endian_encoding,
    write_source_artifact,
)


def _synthetic_aorc_dataset():
    time = pd.date_range("2020-01-01 00:00", periods=1, freq="1h")
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([10.0, 11.0, 12.0])
    precip = np.zeros((1, lat.size, lon.size), dtype="float32")
    precip[0, 2, 2] = 100.0
    return xr.Dataset(
        {"APCP_surface": (("time", "latitude", "longitude"), precip)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_selected_event_window_realizes_sst_field_transposition(tmp_path):
    ranked = pd.DataFrame(
        {
            "storm_date": [pd.Timestamp("2020-01-01 00:00")],
            "mean": [100.0],
            "max": [100.0],
            "min": [100.0],
            "x": [12.0],
            "y": [2.0],
            "historical_footprint_center_lon": [12.0],
            "historical_footprint_center_lat": [2.0],
            "target_footprint_center_lon": [11.0],
            "target_footprint_center_lat": [1.0],
            "potential_method": ["moving_footprint_max_mean"],
            "por_rank": [1],
            "annual_rank": [1],
        }
    )

    def opener(year, spec):
        return _synthetic_aorc_dataset()

    written, with_centroids = _compute_selected_event_windows(
        opener,
        ranked,
        {"variable": "APCP_surface"},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )
    with_targets = _ensure_transposition_targets({}, {}, with_centroids)

    assert len(written) == 1
    assert with_targets.loc[0, "historical_centroid_lon"] == 12.0
    assert with_targets.loc[0, "historical_centroid_lat"] == 2.0
    assert with_targets.loc[0, "transposed_centroid_lon"] == 11.0
    assert with_targets.loc[0, "transposed_centroid_lat"] == 1.0
    assert with_targets.loc[0, "transposition_offset_lon"] == -1.0
    assert with_targets.loc[0, "transposition_offset_lat"] == -1.0

    assert _event_window_file_has_required_variables(
        written[0],
        {"variable": "APCP_surface"},
        require_transposition=True,
    )
    with xr.open_dataset(written[0]) as ds:
        assert ds.attrs["aorc_sst_field_transposition"] == "applied"
        assert ds.attrs["transposition_method"] == "coordinate_shift_to_study_footprint"
        assert ds.attrs["transposition_offset_lon"] == -1.0
        assert ds.attrs["transposition_offset_lat"] == -1.0
        assert float(ds["longitude"].isel(longitude=-1)) == 11.0
        assert float(ds["latitude"].isel(latitude=-1)) == 1.0
        assert float(ds["APCP_surface"].sel(time="2020-01-01", latitude=1.0, longitude=11.0)) == 100.0


def test_selected_event_window_reuses_existing_file_without_reopening_source(tmp_path):
    ranked = pd.DataFrame(
        {
            "storm_date": [pd.Timestamp("2020-01-01 00:00")],
            "mean": [100.0],
            "max": [100.0],
            "min": [100.0],
            "x": [12.0],
            "y": [2.0],
            "historical_footprint_center_lon": [12.0],
            "historical_footprint_center_lat": [2.0],
            "target_footprint_center_lon": [11.0],
            "target_footprint_center_lat": [1.0],
            "potential_method": ["moving_footprint_max_mean"],
            "por_rank": [1],
            "annual_rank": [1],
        }
    )

    def opener(year, spec):
        return _synthetic_aorc_dataset()

    written, _ = _compute_selected_event_windows(
        opener,
        ranked,
        {"variable": "APCP_surface"},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )

    def fail_if_reopened(year, spec):
        raise AssertionError("existing event window should have been reused")

    resumed, with_centroids = _compute_selected_event_windows(
        fail_if_reopened,
        ranked,
        {"variable": "APCP_surface"},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )

    assert resumed == written
    assert with_centroids.loc[0, "historical_centroid_lon"] == 12.0
    assert with_centroids.loc[0, "historical_centroid_lat"] == 2.0


def test_selected_event_window_promotes_complete_tmp_without_reopening_source(tmp_path):
    ranked = pd.DataFrame(
        {
            "storm_date": [pd.Timestamp("2020-01-01 00:00")],
            "mean": [100.0],
            "max": [100.0],
            "min": [100.0],
            "x": [12.0],
            "y": [2.0],
            "historical_footprint_center_lon": [12.0],
            "historical_footprint_center_lat": [2.0],
            "target_footprint_center_lon": [11.0],
            "target_footprint_center_lat": [1.0],
            "potential_method": ["moving_footprint_max_mean"],
            "por_rank": [1],
            "annual_rank": [1],
        }
    )

    def opener(year, spec):
        return _synthetic_aorc_dataset()

    written, _ = _compute_selected_event_windows(
        opener,
        ranked,
        {"variable": "APCP_surface"},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )
    path = written[0]
    tmp = path.with_name(f".{path.name}.tmp")
    path.replace(tmp)

    def fail_if_reopened(year, spec):
        raise AssertionError("complete temp event window should have been promoted")

    resumed, with_centroids = _compute_selected_event_windows(
        fail_if_reopened,
        ranked,
        {"variable": "APCP_surface"},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )

    assert resumed == written
    assert path.exists()
    assert not tmp.exists()
    assert with_centroids.loc[0, "historical_centroid_lon"] == 12.0
    assert with_centroids.loc[0, "historical_centroid_lat"] == 2.0


def test_selected_event_window_trims_open_year_cache_between_storms(tmp_path):
    years = [2020, 2021, 2022, 2023, 2024]
    ranked = pd.DataFrame(
        {
            "storm_date": [pd.Timestamp(f"{year}-01-01 00:00") for year in years],
            "mean": [100.0] * len(years),
            "max": [100.0] * len(years),
            "min": [100.0] * len(years),
            "x": [12.0] * len(years),
            "y": [2.0] * len(years),
            "historical_footprint_center_lon": [12.0] * len(years),
            "historical_footprint_center_lat": [2.0] * len(years),
            "target_footprint_center_lon": [11.0] * len(years),
            "target_footprint_center_lat": [1.0] * len(years),
            "potential_method": ["moving_footprint_max_mean"] * len(years),
            "por_rank": range(1, len(years) + 1),
            "annual_rank": [1] * len(years),
        }
    )
    active_years = set()
    active_counts_before_open = []

    def opener(year, spec):
        active_counts_before_open.append(len(active_years))
        active_years.add(year)
        ds = _synthetic_aorc_dataset().assign_coords(time=pd.date_range(f"{year}-01-01 00:00", periods=1, freq="1h"))
        ds.set_close(lambda year=year: active_years.discard(year))
        return ds

    _compute_selected_event_windows(
        opener,
        ranked,
        {"variable": "APCP_surface", "max_open_year_datasets": 2},
        (10.0, 0.0, 12.0, 2.0),
        tmp_path,
        1,
        "test",
    )

    assert active_counts_before_open == [0, 1, 2, 2, 2]
    assert active_years == set()


def test_strip_netcdf_endian_encoding_removes_inherited_endian_metadata():
    ds = _synthetic_aorc_dataset()
    ds["APCP_surface"].encoding["endian"] = "little"
    ds["latitude"].encoding["endian"] = "little"

    stripped = _strip_netcdf_endian_encoding(ds)

    assert "endian" not in stripped["APCP_surface"].encoding
    assert "endian" not in stripped["latitude"].encoding


def test_collect_aorc_sst_event_windows_completes_deferred_catalog(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "test",
        "aorc_sst_root": tmp_path / "data/sources/aorc_sst",
        "aorc_sst_rainfall_members_csv": tmp_path / "data/sources/aorc_sst/rainfall_members.csv",
        "source_artifacts_root": tmp_path / "data/sources/source_artifacts",
    }
    ranked = pd.DataFrame(
        {
            "storm_date": [pd.Timestamp("2020-01-01 00:00")],
            "mean": [100.0],
            "max": [100.0],
            "min": [100.0],
            "x": [12.0],
            "y": [2.0],
            "historical_footprint_center_lon": [12.0],
            "historical_footprint_center_lat": [2.0],
            "target_footprint_center_lon": [11.0],
            "target_footprint_center_lat": [1.0],
            "potential_method": ["moving_footprint_max_mean"],
            "por_rank": [1],
            "annual_rank": [1],
        }
    )
    collection_dir = paths["aorc_sst_root"] / "test/1hr-events"
    collection_dir.mkdir(parents=True)
    ranked.to_csv(collection_dir / "ranked-storms.csv", index=False)

    def opener(year, spec):
        return _synthetic_aorc_dataset()

    result = collect_aorc_sst_event_windows(
        {
            "paths": paths,
            "start": pd.Timestamp("2020-01-01"),
            "end": pd.Timestamp("2020-01-01"),
            "aorc_sst": {
                "bbox_wgs84": [10.0, 0.0, 12.0, 2.0],
                "variable": "APCP_surface",
                "storm_duration_hours": 1,
            },
        },
        opener=opener,
    )

    manifest = json.loads((paths["source_artifacts_root"] / "aorc_sst_rainfall_catalog.json").read_text())
    assert result["event_window_count"] == 1
    assert result["rainfall_member_rows"] == 1
    assert paths["aorc_sst_rainfall_members_csv"].exists()
    assert manifest["status"] == "complete"


def test_aorc_sst_reuse_gate_rejects_stale_selection_parameters(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "test",
        "source_artifacts_root": tmp_path / "data/sources/source_artifacts",
    }
    write_source_artifact(
        paths,
        source="aorc_sst",
        kind="rainfall_catalog",
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-31"),
        metadata={
            "duration_hours": 72,
            "check_every_n_hours": 6,
            "min_precip_threshold": 2.5,
            "decluster_hours": 72,
            "top_n_events_safety_cap": 440,
        },
    )

    assert not _aorc_sst_artifact_covers_settings(
        paths,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-31"),
        duration_hours=72,
        check_every_n_hours=6,
        min_threshold=60.0,
        decluster_hours=72,
        top_n=None,
    )
