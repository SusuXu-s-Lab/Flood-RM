import geopandas as gpd
from shapely.geometry import LineString, Point, box

import design_events.collect_sources.national_hydrography as national_hydrography
from wflow_runs.build_plan import (
    plan_wflow_domain_set,
    plan_wflow_domain_set_from_boundary_handoff_watersheds,
    plan_wflow_domain_set_from_encompassing_huc,
    plan_wflow_domain_set_from_stream_boundary_crossings,
    write_wflow_crossing_gauge_locations,
)


def _write_location(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    rivers_path.parent.mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 1.0]},
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),  # trunk inflow -> crossing (10, 5)
            LineString([(12.0, 8.0), (8.0, 8.0)]),  # trickle, below threshold
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")
    return location_root


def _config():
    return {
        "project": {"name": "greensboro"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
            }
        },
        "wflow": {
            "domain_set": {
                "outlet_source": "stream_boundary_crossings",
                "crossings": {"min_uparea_km2": 5.0},
            }
        },
    }


def test_plan_from_crossings_delineates_one_subbasin_per_inflow(tmp_path):
    location_root = _write_location(tmp_path)

    plan = plan_wflow_domain_set_from_stream_boundary_crossings(
        _config(), {"location_root": location_root}
    )

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.gage_count == 0
    submodel = plan.submodels[0]
    assert submodel["wflow_submodel_id"] == "greensboro_main_inflow_01"
    assert submodel["region"] == {"subbasin": [10.0, 5.0], "uparea": 100.0}
    assert submodel["sfincs_domain_ids"] == ["greensboro_main"]


def test_dispatcher_routes_on_outlet_source(tmp_path):
    location_root = _write_location(tmp_path)

    plan = plan_wflow_domain_set(_config(), {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodels[0]["region_kind"] == "subbasin"


def test_boundary_handoff_watershed_groups_crossings_by_sfincs_domain(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    rivers_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 80.0]},
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),
            LineString([(12.0, 8.0), (8.0, 8.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "boundary_handoff_watershed"

    plan = plan_wflow_domain_set_from_boundary_handoff_watersheds(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.handoff_count == 2
    submodel = plan.submodels[0]
    assert submodel["wflow_submodel_id"] == "greensboro_main"
    assert submodel["region_kind"] == "subbasin"
    assert submodel["region"] == {"subbasin": [[10.0, 10.0], [5.0, 8.0]], "uparea": 5.0}
    assert submodel["subbasin_geometry"] is None
    assert submodel["watershed_source"] == "hydromt_subbasin_boundary_handoff_points"
    assert submodel["sfincs_handoff_ids"] == ["greensboro_main_inflow_01", "greensboro_main_inflow_02"]
    assert [point["uparea_km2"] for point in submodel["handoff_points"]] == [100.0, 80.0]


def test_boundary_handoff_watershed_prefers_native_sfincs_source_artifacts(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    handoff_path = location_root / "data/sfincs/domains/greensboro_main/base/gis/wflow_handoff_sources.geojson"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    rivers_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    network_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 80.0]},
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),
            LineString([(12.0, 8.0), (8.0, 8.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")
    gpd.GeoDataFrame(
        {
            "site_no": ["greensboro_main_inflow_01"],
            "sfincs_handoff_id": ["greensboro_main_inflow_01"],
            "sfincs_domain_id": ["greensboro_main"],
            "wflow_submodel_id": ["greensboro_main"],
            "uparea": [95.0],
            "handoff_placement": ["sfincs_native_river_inflow"],
        },
        geometry=[Point(10.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(handoff_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02095500"],
            "sfincs_domain_id": ["greensboro_main", "greensboro_main"],
            "wflow_submodel_id": ["greensboro_main", "greensboro_main"],
            "status": ["active", "active"],
            "review_status": ["accepted", "accepted_with_warning"],
            "frequency_basis": ["pot", "pot"],
            "roles": ["frequency", "validation"],
            "drainage_area_sqmi": [150.0, 180.0],
        },
        geometry=[Point(5.0, 5.0), Point(6.0, 6.0)],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "boundary_handoff_watershed"
    config["wflow"]["streamgage_network"] = {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"}
    config["sfincs_domain_set"] = {"domains_root": "data/sfincs/domains"}
    config["inland_coupling"] = {"discharge_forcing": {"handoff_location": "sfincs_native_river_inflow"}}

    plan = plan_wflow_domain_set_from_boundary_handoff_watersheds(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.handoff_count == 1
    submodel = plan.submodels[0]
    assert submodel["wflow_submodel_id"] == "greensboro_main"
    assert submodel["region"] == {"subbasin": [10.0, 5.0], "uparea": 5.0}
    assert submodel["watershed_source"] == "hydromt_sfincs_handoff_source_artifacts"
    assert submodel["sfincs_handoff_ids"] == ["greensboro_main_inflow_01"]
    assert submodel["gauge_site_nos"] == ("02095000", "02095500")
    assert submodel["frequency_basis"] == ("pot",)
    assert [point["uparea_km2"] for point in submodel["handoff_points"]] == [95.0]


def _fake_wbd(monkeypatch):
    # HUC8 polygon does not cover the full box(0,0,10,10); HUC6 does.
    def fake_fetch(bbox, *, huc_level, **kwargs):
        if huc_level == 8:
            geom, huc_id = box(-5.0, -5.0, 5.0, 15.0), "03030001"
        else:
            geom, huc_id = box(-5.0, -5.0, 15.0, 15.0), f"huc{huc_level}"
        return gpd.GeoDataFrame({"huc_id": [huc_id], "huc_level": [huc_level]}, geometry=[geom], crs="EPSG:4326")

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)


def test_plan_from_encompassing_huc_is_one_basin_with_crossing_gauges(tmp_path, monkeypatch):
    _fake_wbd(monkeypatch)
    location_root = _write_location(tmp_path)
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "encompassing_huc"

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1  # one HUC basin, not one subbasin per crossing
    submodel = plan.submodels[0]
    assert submodel["region_kind"] == "geom"
    assert "geom" in submodel["region"]
    assert submodel["huc_level"] == 6  # HUC8 didn't encapsulate; HUC6 did
    # the single basin still feeds every crossing as a SFINCS source
    assert submodel["sfincs_handoff_ids"] == ["greensboro_main_inflow_01"]
    assert (location_root / "data/wflow/wflow_domain_huc.geojson").exists()


def test_encompassing_huc_prefers_native_sfincs_source_artifacts(tmp_path, monkeypatch):
    _fake_wbd(monkeypatch)
    location_root = _write_location(tmp_path)
    handoff_path = location_root / "data/sfincs/domains/greensboro_main/base/gis/wflow_handoff_sources.geojson"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["greensboro_main_inflow_01"],
            "sfincs_handoff_id": ["greensboro_main_inflow_01"],
            "sfincs_domain_id": ["greensboro_main"],
            "wflow_submodel_id": ["greensboro_main"],
            "uparea": [95.0],
            "handoff_placement": ["sfincs_native_river_inflow"],
        },
        geometry=[Point(10.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(handoff_path, driver="GeoJSON")
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "encompassing_huc"
    config["sfincs_domain_set"] = {"domains_root": "data/sfincs/domains"}
    config["inland_coupling"] = {"discharge_forcing": {"handoff_location": "sfincs_native_river_inflow"}}

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.handoff_count == 1
    submodel = plan.submodels[0]
    assert submodel["region_kind"] == "geom"
    assert submodel["sfincs_handoff_ids"] == ["greensboro_main_inflow_01"]
    assert [point["uparea_km2"] for point in submodel["handoff_points"]] == [95.0]


def test_encompassing_huc_ignores_generated_placeholder_huc_cache(tmp_path, monkeypatch):
    location_root = _write_location(tmp_path)
    huc_root = location_root / "data/wflow/domain_huc"
    huc_root.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"huc_id": ["h8"], "huc_level": [8], "huc_kind": ["single"]},
        geometry=[box(-0.5, -0.5, 10.5, 10.5)],
        crs="EPSG:4326",
    ).to_file(huc_root / "greensboro_main.geojson", driver="GeoJSON")

    def fake_fetch(bbox_arg, *, huc_level, **kwargs):
        return gpd.GeoDataFrame(
            {"huc_id": ["03030001"], "huc_level": [huc_level]},
            geometry=[box(-1.0, -1.0, 11.0, 11.0)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "encompassing_huc"
    config["wflow"]["domain_set"]["huc"] = {"levels": [8], "root": "data/wflow/domain_huc"}

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodels[0]["huc_id"] == "03030001"
    written = gpd.read_file(huc_root / "greensboro_main.geojson")
    assert written["huc_id"].tolist() == ["03030001"]


def test_encompassing_huc_is_per_box_basin_with_union_fallback(tmp_path, monkeypatch):
    # Two coverage boxes in different basins (the Greensboro divide): each box gets its own
    # HUC basin, and since each straddles HUC8 lines, each resolves to a union of HUC8s.
    location_root = tmp_path / "locations" / "greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    rivers_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main", "greensboro_east"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(20.0, 0.0, 30.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 80.0]},
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),    # -> west box crossing (10, 5)
            LineString([(35.0, 5.0), (25.0, 5.0)]),   # -> east box crossing (30, 5)
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")

    # HUC8 tiles 6 wide: every 10-wide box spans >1 tile, so no single HUC covers a box.
    def fake_fetch(bbox_arg, *, huc_level, **kwargs):
        if huc_level != 8:
            return gpd.GeoDataFrame({"huc_id": []}, geometry=[], crs="EPSG:4326")
        tiles, ids, x, i = [], [], -6.0, 0
        while x < 36.0:
            tiles.append(box(x, -5.0, x + 6.0, 15.0))
            ids.append(f"030300{i:02d}")
            x += 6.0
            i += 1
        return gpd.GeoDataFrame({"huc_id": ids, "huc_level": [8] * len(ids)}, geometry=tiles, crs="EPSG:4326")

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)
    config = {
        "project": {"name": "greensboro"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "collection": {"national_hydrography": {"river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"}},
        "wflow": {"domain_set": {"outlet_source": "encompassing_huc", "crossings": {"min_uparea_km2": 5.0}}},
    }

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 2  # one basin per box, not one merged domain
    by_id = {s["wflow_submodel_id"]: s for s in plan.submodels}
    assert set(by_id) == {"greensboro_main", "greensboro_east"}
    for domain_id, submodel in by_id.items():
        assert submodel["region_kind"] == "geom"
        assert submodel["huc_level"] == 8
        assert submodel["sfincs_handoff_ids"] == [f"{domain_id}_inflow_01"]
        assert (location_root / f"data/wflow/domain_huc/{domain_id}.geojson").exists()
    # Region setup plots the larger Wflow watersheds as separate hydrologic domains,
    # not as one dissolved watershed wrapped around both SFINCS coverage boxes.
    combined = gpd.read_file(location_root / "data/wflow/wflow_domain_huc.geojson")
    assert set(combined["sfincs_domain_id"]) == {"greensboro_main", "greensboro_east"}
    assert combined["huc_kind"].tolist() == ["union", "union"]


def test_encompassing_huc_uses_single_sfincs_outer_box_when_configured(tmp_path, monkeypatch):
    location_root = tmp_path / "locations" / "austin"
    exposure_path = location_root / "data/static/aoi/evaluation_footprint.geojson"
    stale_bbox_path = location_root / "data/static/aoi/bbox.geojson"
    rivers_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    for path in (exposure_path, stale_bbox_path, rivers_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame(
        {"subregion_id": ["P1R", "P1U"]},
        geometry=[box(0.0, 0.0, 5.0, 5.0), box(4.0, 0.0, 9.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(exposure_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"subregion_id": ["austin_p1r", "austin_p1u"]},
        geometry=[box(0.0, 0.0, 5.0, 5.0), box(4.0, 0.0, 9.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(stale_bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0, 80.0]},
        geometry=[
            LineString([(12.0, 2.0), (8.0, 2.0)]),
            LineString([(-2.0, 1.0), (1.0, 1.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(rivers_path, driver="GPKG")

    def fake_fetch(bbox_arg, *, huc_level, **kwargs):
        return gpd.GeoDataFrame(
            {"huc_id": [f"austin_huc{huc_level}"], "huc_level": [huc_level]},
            geometry=[box(-1.0, -1.0, 10.0, 6.0)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)
    config = {
        "project": {"name": "austin"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "sfincs_domain_set": {
            "source": "data/static/aoi/evaluation_footprint.geojson",
            "allow_multiple_domains": False,
            "region_geometry": "bounding_box",
        },
        "collection": {"national_hydrography": {"river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"}},
        "wflow": {"domain_set": {"outlet_source": "encompassing_huc", "crossings": {"min_uparea_km2": 5.0}}},
    }

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.reviewed_network == exposure_path
    assert plan.submodel_count == 1
    assert plan.handoff_count == 2
    submodel = plan.submodels[0]
    assert submodel["wflow_submodel_id"] == "austin_main"
    assert submodel["sfincs_domain_ids"] == ["austin_main"]
    assert submodel["sfincs_handoff_ids"] == ["austin_main_inflow_01", "austin_main_inflow_02"]
    assert (location_root / "data/wflow/domain_huc/austin_main.geojson").exists()


def test_encompassing_huc_attaches_reviewed_streamgages_inside_each_huc(tmp_path, monkeypatch):
    _fake_wbd(monkeypatch)
    location_root = _write_location(tmp_path)
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02094659", "02099000"],
            "status": ["active", "active", "inactive"],
            "review_status": ["accepted", "accepted_with_warning", "accepted"],
            "drainage_area_sqmi": [34.0, 7.33, 99.0],
            "frequency_basis": ["yes", "no", "yes"],
        },
        geometry=[Point(1.0, 1.0), Point(12.0, 1.0), Point(2.0, 2.0)],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "encompassing_huc"
    config["wflow"]["streamgage_network"] = {
        "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
    }

    plan = plan_wflow_domain_set_from_encompassing_huc(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.gage_count == 2
    submodel = plan.submodels[0]
    assert submodel["gauge_site_nos"] == ("02094659", "02095000")
    assert submodel["frequency_basis"] == ("no", "yes")


def test_dispatcher_routes_to_encompassing_huc(tmp_path, monkeypatch):
    _fake_wbd(monkeypatch)
    location_root = _write_location(tmp_path)
    config = _config()
    config["wflow"]["domain_set"]["outlet_source"] = "encompassing_huc"

    plan = plan_wflow_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodels[0]["region_kind"] == "geom"


def test_crossing_gauge_writer_emits_outlet_point_with_uparea(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    location_root.mkdir(parents=True)
    submodel = {
        "wflow_submodel_id": "greensboro_main_inflow_01",
        "region": {"subbasin": [10.0, 5.0], "uparea": 100.0},
        "outlet_region": {"subbasin": [10.0, 5.0]},
        "sfincs_handoff_ids": ["greensboro_main_inflow_01"],
        "sfincs_domain_ids": ["greensboro_main"],
    }

    summary = write_wflow_crossing_gauge_locations({"wflow": {}}, {"location_root": location_root}, submodel)

    assert summary["gauge_count"] == 1
    assert summary["snap_uparea"] is True  # drainage area known -> snap on uparea
    gauges = gpd.read_file(summary["gauges_fn"])
    assert gauges["name"].tolist() == ["greensboro_main_inflow_01"]
    assert float(gauges["uparea"].iloc[0]) == 100.0
    point = gauges.geometry.iloc[0]
    assert (round(point.x, 6), round(point.y, 6)) == (10.0, 5.0)


def test_crossing_gauge_writer_emits_one_gauge_per_handoff_point(tmp_path):
    # An encompassing-HUC submodel is one basin with many crossings -> many gauges.
    location_root = tmp_path / "locations" / "greensboro"
    location_root.mkdir(parents=True)
    submodel = {
        "wflow_submodel_id": "greensboro_huc6_030300",
        "region_kind": "geom",
        "region": {"geom": "/abs/huc.geojson"},
        "sfincs_handoff_ids": ["greensboro_main_inflow_01", "greensboro_east_inflow_01"],
        "handoff_points": [
            {"sfincs_handoff_id": "greensboro_main_inflow_01", "sfincs_domain_id": "greensboro_main", "lon": 10.0, "lat": 5.0, "uparea_km2": 100.0},
            {"sfincs_handoff_id": "greensboro_east_inflow_01", "sfincs_domain_id": "greensboro_east", "lon": 20.0, "lat": 5.0, "uparea_km2": 50.0},
        ],
    }

    summary = write_wflow_crossing_gauge_locations({"wflow": {}}, {"location_root": location_root}, submodel)

    assert summary["gauge_count"] == 2
    gauges = gpd.read_file(summary["gauges_fn"])
    assert sorted(gauges["name"]) == ["greensboro_east_inflow_01", "greensboro_main_inflow_01"]
    assert sorted(gauges["uparea"]) == [50.0, 100.0]


def test_boundary_handoff_watershed_gauges_snap_to_river_without_nhdplus_uparea(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    location_root.mkdir(parents=True)
    submodel = {
        "wflow_submodel_id": "greensboro_main",
        "region_kind": "subbasin",
        "region": {"subbasin": [10.0, 5.0], "uparea": 5.0},
        "sfincs_handoff_ids": ["greensboro_main_inflow_01"],
        "handoff_points": [
            {
                "sfincs_handoff_id": "greensboro_main_inflow_01",
                "sfincs_domain_id": "greensboro_main",
                "lon": 10.0,
                "lat": 5.0,
                "uparea_km2": 100.0,
            },
        ],
    }
    config = {"wflow": {"domain_set": {"outlet_source": "boundary_handoff_watershed"}}}

    summary = write_wflow_crossing_gauge_locations(config, {"location_root": location_root}, submodel)

    assert summary["gauge_count"] == 1
    assert summary["snap_to_river"] is True
    assert summary["snap_uparea"] is False
