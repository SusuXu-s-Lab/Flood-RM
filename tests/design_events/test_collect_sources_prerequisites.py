import geopandas as gpd
import matplotlib
import pandas as pd
from shapely.geometry import box

import design_events.collect_sources.workflow as workflow
from design_events.collect_sources.workflow import (
    CollectSourcesNotebookRuntime,
    collect_all_sources,
    prepare,
    refresh_wflow_hydrography_basemap,
)


matplotlib.use("Agg")


def test_prepare_writes_review_required_aorc_transposition_region(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    footprint_path = location_root / "data/static/aoi/evaluation_footprint.geojson"
    output_path = location_root / "data/sources/aorc_sst/transposition_regions/transposition_region.geojson"
    footprint_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"id": ["evaluation"]},
        geometry=[box(-80.0, 36.0, -79.9, 36.1)],
        crs="EPSG:4326",
    ).to_file(footprint_path, driver="GeoJSON")
    config = {
        "crs": "EPSG:32617",
        "collection": {
            "usgs_streamgages": {"discovery": {"hydrologic_buffer_km": 50}},
            "aorc_sst": {
                "transposition_region": {
                    "id": "greensboro-inland-review-required",
                    "geometry_file": "data/sources/aorc_sst/transposition_regions/transposition_region.geojson",
                }
            },
        },
        "smart_ds_evaluation_footprint": {
            "output": "data/static/aoi/evaluation_footprint.geojson",
        },
    }
    paths = {
        "location_root": location_root,
        "repo_root": tmp_path,
    }

    result = prepare(config, paths)

    assert output_path.exists()
    assert result.to_dict("records") == [
        {
            "artifact": "aorc_sst transposition region",
            "path": str(output_path),
            "status": "created_review_required",
            "source_geometry": str(footprint_path),
            "buffer_km": 50.0,
            "review_status": "review_required",
        }
    ]
    region = gpd.read_file(output_path)
    assert region.crs.to_epsg() == 4326
    assert region.iloc[0]["region_id"] == "greensboro-inland-review-required"
    assert region.iloc[0]["review_status"] == "review_required"
    assert float(region.to_crs("EPSG:32617").area.iloc[0]) > float(
        gpd.read_file(footprint_path).to_crs("EPSG:32617").area.iloc[0]
    )


def test_plot_sst_region_uses_smart_ds_evaluation_footprint(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    footprint_path = location_root / "data/static/aoi/evaluation_footprint.geojson"
    region_path = location_root / "data/sources/aorc_sst/transposition_regions/transposition_region.geojson"
    footprint_path.parent.mkdir(parents=True)
    region_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"id": ["evaluation"]},
        geometry=[box(-80.0, 36.0, -79.9, 36.1)],
        crs="EPSG:4326",
    ).to_file(footprint_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"region_id": ["greensboro-inland-review-required"]},
        geometry=[box(-80.1, 35.9, -79.8, 36.2)],
        crs="EPSG:4326",
    ).to_file(region_path, driver="GeoJSON")
    config = {
        "collection": {
            "aorc_sst": {
                "transposition_region": {
                    "geometry_file": "data/sources/aorc_sst/transposition_regions/transposition_region.geojson",
                }
            },
        },
        "smart_ds_evaluation_footprint": {
            "output": "data/static/aoi/evaluation_footprint.geojson",
        },
    }

    fig, ax = workflow.plot_sst_region(config, {"location_root": location_root, "location_name": "greensboro"}, basemap=False)

    assert fig is not None
    assert ax.get_title() == "Greensboro stochastic storm transposition region"


def test_prepare_reuses_existing_aorc_transposition_region(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    output_path = location_root / "data/sources/aorc_sst/transposition_regions/transposition_region.geojson"
    output_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"region_id": ["reviewed"], "review_status": ["accepted"]},
        geometry=[box(-80.0, 36.0, -79.9, 36.1)],
        crs="EPSG:4326",
    ).to_file(output_path, driver="GeoJSON")
    config = {
        "collection": {
            "aorc_sst": {
                "transposition_region": {
                    "geometry_file": "data/sources/aorc_sst/transposition_regions/transposition_region.geojson",
                }
            }
        }
    }

    result = prepare(config, {"location_root": location_root, "repo_root": tmp_path})

    assert result.loc[0, "status"] == "reused"
    assert result.loc[0, "path"] == str(output_path)


def test_collect_all_sources_uses_source_collection_plan(tmp_path):
    config = {"collection": {"start": "2020-01-01", "end": "2020-01-02", "cora": {}}}
    paths = {"waterlevel_csv": tmp_path / "waterlevel.csv"}

    def collect_cora(settings, *, skip_existing, smoke):
        assert settings["start"] == pd.Timestamp("2020-01-01")
        assert settings["end"] == pd.Timestamp("2020-01-02")
        assert skip_existing is False
        assert smoke is True
        return pd.DataFrame({"water_level": [1.0, 2.0]})

    result = collect_all_sources(
        config,
        paths,
        skip_existing=False,
        smoke=True,
        funcs={"collect_cora": collect_cora},
    )

    assert result["cora_rows"] == 2
    assert result["waterlevel_csv"] == paths["waterlevel_csv"]


def test_refresh_wflow_hydrography_basemap_uses_plan_settings(monkeypatch, tmp_path):
    class FakePlan:
        def has(self, name):
            return name == "national_hydrography"

        def settings_for(self, name):
            assert name == "national_hydrography"
            return {"paths": {"location_root": tmp_path}, "national_hydrography": {"mode": "test"}}

    captured = {}

    def fake_plan(config, paths):
        captured["config"] = config
        captured["paths"] = paths
        return FakePlan()

    def fake_refresh(settings, *, skip_existing):
        captured["settings"] = settings
        captured["skip_existing"] = skip_existing
        return {"status": "refreshed"}

    monkeypatch.setattr(workflow, "plan", fake_plan)
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.refresh_wflow_hydrography_basemap",
        fake_refresh,
    )
    runtime = CollectSourcesNotebookRuntime(
        location_root=tmp_path,
        location_name="test",
        repo_root=tmp_path,
        runtime_config={"collection": {"national_hydrography": {}}},
        config={},
        grid_config={},
        data_sources={},
        sfincs_config={},
        wflow_config={},
        runtime_paths={"location_root": tmp_path},
        collection={"national_hydrography": {}},
        usgs_streamgages={},
        candidate_path=tmp_path / "candidates.geojson",
        reviewed_network_path=tmp_path / "reviewed.geojson",
        streamflow_records_cfg={},
        streamflow_records_path=tmp_path / "records.csv",
    )

    result = refresh_wflow_hydrography_basemap(runtime, force=True)

    assert result == {"status": "refreshed"}
    assert captured["config"] == runtime.runtime_config
    assert captured["paths"] == runtime.runtime_paths
    assert captured["settings"]["national_hydrography"] == {"mode": "test"}
    assert captured["skip_existing"] is False
