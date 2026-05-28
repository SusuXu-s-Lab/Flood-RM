import json

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import box

from design_events.collect_sources.aorc_sst import collect_aorc_sst


def _toy_aorc_dataset():
    times = pd.date_range("2020-01-01", periods=10, freq="h")
    lat = [42.0, 42.1]
    lon = [-70.8, -70.7]
    precip = xr.DataArray(
        0.1,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
        name="APCP_surface",
    )
    precip.loc[{"time": slice("2020-01-01T07", "2020-01-01T09")}] = 3.0
    precip.loc[{"time": slice("2020-01-01T03", "2020-01-01T05")}] = 2.0
    return xr.Dataset({"APCP_surface": precip})


def _single_cell_basin(path):
    gpd.GeoDataFrame(
        {"id": ["test-basin"]},
        geometry=[box(9.75, -0.25, 10.25, 0.25)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")


def _moving_watershed_dataset():
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    lat = [0.0, 1.0, 2.0]
    lon = [10.0, 11.0, 12.0]
    precip = xr.DataArray(
        0.0,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
        name="APCP_surface",
    )
    precip.loc[{"time": slice("2020-01-01T03", "2020-01-01T05"), "latitude": 2.0, "longitude": 12.0}] = 10.0
    precip.loc[{"time": slice("2020-01-01T03", "2020-01-01T05"), "latitude": 0.0, "longitude": 10.0}] = 1.0
    return xr.Dataset({"APCP_surface": precip})


def test_collect_aorc_sst_writes_ranked_storm_tables_and_selected_event_files(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "aorc_sst_root": tmp_path / "aorc_sst",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-01 09:00:00"),
        "aorc_sst": {
            "zarr_year_pattern": "memory://{year}.zarr",
            "variable": "APCP_surface",
            "bbox_wgs84": [-71.0, 41.9, -70.6, 42.2],
            "storm_duration_hours": 3,
            "check_every_n_hours": 1,
            "top_n_events": 2,
            "min_precip_threshold": 1.0,
            "decluster_hours": 3,
            "transposition_region": {"id": "test-region"},
            "write_event_windows": True,
        },
    }

    result = collect_aorc_sst(
        settings,
        opener=lambda year, spec: _toy_aorc_dataset(),
    )

    collection_dir = tmp_path / "aorc_sst/marshfield/3hr-events"
    ranked = pd.read_csv(collection_dir / "ranked-storms.csv")
    stats = pd.read_csv(collection_dir / "storm-stats.csv")
    members = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
    manifest = json.loads((paths["source_artifacts_root"] / "aorc_sst_rainfall_catalog.json").read_text())

    assert result["ranked_rows"] == 2
    assert result["event_window_count"] == 2
    assert ranked["por_rank"].tolist() == [1, 2]
    assert ranked["mean"].round(2).tolist() == [9.0, 6.0]
    assert set(ranked["storm_date"]).issubset(set(stats["storm_date"]))
    assert len(stats) > len(ranked)
    assert {"x", "y"}.isdisjoint(stats.columns)
    assert ranked["x"].notna().all()
    assert ranked["y"].notna().all()
    assert members["source"].tolist() == ["aorc_sst", "aorc_sst"]
    assert {"mean_precip_mm", "max_precip_mm", "min_precip_mm", "precip_units"}.issubset(members.columns)
    assert "mean_precip_in" not in members.columns
    assert members["precip_units"].eq("mm").all()
    assert members["transposition_region_id"].tolist() == ["test-region", "test-region"]
    assert members["centroid_lon"].notna().all()
    assert members["centroid_lat"].notna().all()
    assert len(list((collection_dir / "event_windows").glob("*.nc"))) == 2
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["backend"] == "direct_aorc_sst"


def test_collect_aorc_sst_computes_selected_centroids_without_event_window_files(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "aorc_sst_root": tmp_path / "aorc_sst",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-01 09:00:00"),
        "aorc_sst": {
            "zarr_year_pattern": "memory://{year}.zarr",
            "variable": "APCP_surface",
            "bbox_wgs84": [-71.0, 41.9, -70.6, 42.2],
            "storm_duration_hours": 3,
            "check_every_n_hours": 1,
            "top_n_events": 1,
            "min_precip_threshold": 1.0,
            "decluster_hours": 3,
            "write_event_windows": False,
        },
    }

    result = collect_aorc_sst(settings, opener=lambda year, spec: _toy_aorc_dataset())

    collection_dir = tmp_path / "aorc_sst/marshfield/3hr-events"
    ranked = pd.read_csv(collection_dir / "ranked-storms.csv")
    members = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])

    assert result["ranked_rows"] == 1
    assert result["event_window_count"] == 0
    assert ranked["x"].notna().all()
    assert ranked["y"].notna().all()
    assert members["centroid_lon"].notna().all()
    assert members["centroid_lat"].notna().all()
    assert not (collection_dir / "event_windows").exists()


def test_collect_aorc_sst_ranks_by_max_potential_precipitation_over_moved_watershed(tmp_path):
    region = tmp_path / "region.geojson"
    basin = tmp_path / "basin.geojson"
    gpd.GeoDataFrame(
        {"id": ["test-region"]},
        geometry=[box(9.5, -0.5, 12.5, 2.5)],
        crs="EPSG:4326",
    ).to_file(region, driver="GeoJSON")
    _single_cell_basin(basin)
    paths = {
        "repo_root": tmp_path,
        "location_root": tmp_path,
        "location_name": "marshfield",
        "aorc_sst_root": tmp_path / "aorc_sst",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    settings = {
        "config": {"grid_footprint": {"source": basin.name}},
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-01 05:00:00"),
        "aorc_sst": {
            "zarr_year_pattern": "memory://{year}.zarr",
            "variable": "APCP_surface",
            "storm_duration_hours": 3,
            "check_every_n_hours": 1,
            "top_n_events": 2,
            "min_precip_threshold": 1.0,
            "decluster_hours": 3,
            "transposition_stride_cells": 1,
            "transposition_region": {"id": "test-region", "geometry_file": region.name},
            "write_event_windows": False,
        },
    }

    collect_aorc_sst(settings, opener=lambda year, spec: _moving_watershed_dataset())

    ranked = pd.read_csv(tmp_path / "aorc_sst/marshfield/3hr-events/ranked-storms.csv")
    members = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
    assert ranked["potential_method"].tolist() == ["moving_footprint_max_mean"]
    assert ranked["mean"].round(3).tolist() == [30.0]
    assert ranked["x"].round(3).tolist() == [12.0]
    assert ranked["y"].round(3).tolist() == [2.0]
    assert members["centroid_lon"].round(3).tolist() == [12.0]
    assert members["centroid_lat"].round(3).tolist() == [2.0]
    assert members["historical_centroid_lon"].notna().all()
    assert members["historical_centroid_lat"].notna().all()


def test_collect_aorc_sst_does_not_reuse_ranked_storms_without_moving_footprint_metadata(tmp_path):
    region = tmp_path / "region.geojson"
    basin = tmp_path / "basin.geojson"
    gpd.GeoDataFrame(
        {"id": ["test-region"]},
        geometry=[box(9.5, -0.5, 12.5, 2.5)],
        crs="EPSG:4326",
    ).to_file(region, driver="GeoJSON")
    _single_cell_basin(basin)
    collection_dir = tmp_path / "aorc_sst/marshfield/3hr-events"
    collection_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "storm_date": ["2020-01-01T03:00:00"],
            "mean": [3.0],
            "max": [30.0],
            "min": [0.0],
            "por_rank": [1],
            "annual_rank": [1],
        }
    ).to_csv(collection_dir / "ranked-storms.csv", index=False)
    pd.DataFrame(
        {
            "storm_date": ["2020-01-01T03:00:00"],
            "mean": [3.0],
            "max": [30.0],
            "min": [0.0],
        }
    ).to_csv(collection_dir / "storm-stats.csv", index=False)
    source_artifacts = tmp_path / "source_artifacts"
    source_artifacts.mkdir()
    (source_artifacts / "aorc_sst_rainfall_catalog.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-01T05:00:00",
                "metadata": {"smoke": False},
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "repo_root": tmp_path,
        "location_root": tmp_path,
        "location_name": "marshfield",
        "aorc_sst_root": tmp_path / "aorc_sst",
        "source_artifacts_root": source_artifacts,
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    settings = {
        "config": {"grid_footprint": {"source": basin.name}},
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-01-01 05:00:00"),
        "aorc_sst": {
            "zarr_year_pattern": "memory://{year}.zarr",
            "variable": "APCP_surface",
            "storm_duration_hours": 3,
            "check_every_n_hours": 1,
            "top_n_events": 1,
            "min_precip_threshold": 1.0,
            "decluster_hours": 3,
            "transposition_stride_cells": 1,
            "transposition_region": {"id": "test-region", "geometry_file": region.name},
            "write_event_windows": False,
        },
    }

    result = collect_aorc_sst(settings, skip_existing=True, opener=lambda year, spec: _moving_watershed_dataset())

    ranked = pd.read_csv(collection_dir / "ranked-storms.csv")
    assert result["ranked_rows"] == 1
    assert ranked["potential_method"].tolist() == ["moving_footprint_max_mean"]
    assert ranked["mean"].round(3).tolist() == [30.0]


def test_collect_aorc_sst_reuses_yearly_checkpoint_stats(tmp_path):
    collection_dir = tmp_path / "aorc_sst/marshfield/3hr-events"
    checkpoints = collection_dir / "yearly-stats"
    checkpoints.mkdir(parents=True)
    pd.DataFrame(
        {
            "storm_date": ["2020-01-01T03:00:00"],
            "mean": [20.0],
            "max": [30.0],
            "min": [1.0],
            "x": [-70.8],
            "y": [42.0],
            "historical_centroid_lon": [-70.8],
            "historical_centroid_lat": [42.0],
            "potential_method": ["moving_footprint_max_mean"],
        }
    ).to_csv(checkpoints / "storm-stats-2020.csv", index=False)
    calls = []
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "aorc_sst_root": tmp_path / "aorc_sst",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2021-01-01 05:00:00"),
        "aorc_sst": {
            "zarr_year_pattern": "memory://{year}.zarr",
            "variable": "APCP_surface",
            "bbox_wgs84": [-71.0, 41.9, -70.6, 42.2],
            "storm_duration_hours": 3,
            "check_every_n_hours": 1,
            "top_n_events": 1,
            "min_precip_threshold": 1.0,
            "decluster_hours": 3,
            "write_event_windows": False,
        },
    }

    def opener(year, spec):
        calls.append(year)
        return _toy_aorc_dataset().assign_coords(time=pd.date_range(f"{year}-01-01", periods=10, freq="h"))

    collect_aorc_sst(settings, skip_existing=True, opener=opener)

    stats = pd.read_csv(collection_dir / "storm-stats.csv")
    assert calls == [2021]
    assert (checkpoints / "storm-stats-2021.csv").exists()
    assert len(stats) > 1
    assert "2020-01-01 03:00:00" in set(stats["storm_date"].astype(str))
