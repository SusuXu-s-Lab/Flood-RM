import geopandas as gpd
from shapely.geometry import LineString, box

import design_events.collect_sources.national_hydrography as national_hydrography
from sfincs_runs.build_base import plan_inland_sfincs_domain_set


def _write_inland_location(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    study_area = location_root / "data/static/aoi/study_area.geojson"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    for path in (study_area, bbox_path, rivers_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:4326").to_file(study_area, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0]},
        geometry=[LineString([(15.0, 5.0), (5.0, 5.0)])],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")
    return location_root


def _inland_config(outlet_source):
    return {
        "project": {"name": "greensboro"},
        "sfincs": {"model_crs": "EPSG:4326"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "sfincs_domain_set": {"allow_multiple_domains": True},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
            }
        },
        "inland_coupling": {"discharge_forcing": {"handoff_location": "stream_boundary_intersection"}},
        "wflow": {"domain_set": {"outlet_source": outlet_source, "crossings": {"min_uparea_km2": 5.0}}},
    }


def test_sfincs_handoff_sources_in_encompassing_huc_mode(tmp_path, monkeypatch):
    # In HUC mode SFINCS sources are still the box crossings; each coverage box has one Wflow basin.
    def fake_fetch(bbox, *, huc_level, **kwargs):
        return gpd.GeoDataFrame(
            {"huc_id": [f"h{huc_level}"], "huc_level": [huc_level]},
            geometry=[box(-5.0, -5.0, 15.0, 15.0)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)
    location_root = _write_inland_location(tmp_path)

    plan = plan_inland_sfincs_domain_set(_inland_config("encompassing_huc"), {"location_root": location_root})

    assert plan.handoff_count == 1
    domain = plan.domains[0]
    assert domain["handoff_source_ids"] == ["greensboro_main_inflow_01"]
    assert domain["wflow_submodel_ids"] == ["greensboro_main"]


def test_sfincs_domain_keeps_only_its_enclosing_huc_submodel(tmp_path, monkeypatch):
    def fake_fetch(bbox_arg, *, huc_level, **kwargs):
        return gpd.GeoDataFrame(
            {"huc_id": [f"h{huc_level}"], "huc_level": [huc_level]},
            geometry=[box(-5.0, -5.0, 20.0, 10.0)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)
    location_root = tmp_path / "locations" / "austin"
    study_area = location_root / "data/static/aoi/study_area.geojson"
    exposure_path = location_root / "data/static/aoi/evaluation_footprint.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    for path in (study_area, exposure_path, rivers_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame(
        {"subregion_id": ["P1R", "P1U"]},
        geometry=[box(0.0, 0.0, 6.0, 5.0), box(5.0, 0.0, 11.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(study_area, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"subregion_id": ["P1R", "P1U"]},
        geometry=[box(0.0, 0.0, 6.0, 5.0), box(5.0, 0.0, 11.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(exposure_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 80.0]},
        geometry=[
            LineString([(-2.0, 2.0), (2.0, 2.0)]),
            LineString([(13.0, 2.0), (8.0, 2.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")

    config = {
        "project": {"name": "austin"},
        "sfincs": {"model_crs": "EPSG:4326"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "sfincs_domain_set": {
            "source": "data/static/aoi/evaluation_footprint.geojson",
            "allow_multiple_domains": True,
            "region_geometry": "bounding_box",
        },
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
            }
        },
        "inland_coupling": {"discharge_forcing": {"handoff_location": "stream_boundary_intersection"}},
        "wflow": {"domain_set": {"outlet_source": "encompassing_huc", "crossings": {"min_uparea_km2": 5.0}}},
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    by_domain = {domain["sfincs_domain_id"]: domain for domain in plan.domains}
    assert by_domain["austin_p1r"]["wflow_submodel_ids"] == ["austin_p1r"]
    assert by_domain["austin_p1u"]["wflow_submodel_ids"] == ["austin_p1u"]


def test_sfincs_domain_handoff_sources_come_from_stream_crossings(tmp_path):
    # Gage-free: SFINCS discharge source points are the same stream/coverage-box
    # crossings the Wflow domain plan delineates from, not reviewed USGS gages.
    location_root = tmp_path / "locations" / "greensboro"
    study_area = location_root / "data/static/aoi/study_area.geojson"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    for path in (study_area, bbox_path, rivers_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:4326").to_file(study_area, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0]},
        geometry=[LineString([(15.0, 5.0), (5.0, 5.0)])],  # inflow -> crossing (10, 5)
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")

    config = {
        "project": {"name": "greensboro"},
        "sfincs": {"model_crs": "EPSG:4326"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "sfincs_domain_set": {"allow_multiple_domains": True},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
            }
        },
        "inland_coupling": {"discharge_forcing": {"handoff_location": "stream_boundary_intersection"}},
        "wflow": {
            "domain_set": {
                "outlet_source": "stream_boundary_crossings",
                "crossings": {"min_uparea_km2": 5.0},
            }
        },
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.handoff_count == 1
    domain = plan.domains[0]
    assert domain["sfincs_domain_id"] == "greensboro_main"
    assert domain["handoff_source_ids"] == ["greensboro_main_inflow_01"]
    assert domain["wflow_submodel_ids"] == ["greensboro_main_inflow_01"]
