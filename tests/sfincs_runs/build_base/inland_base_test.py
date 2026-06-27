from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from shapely.geometry import LineString, Point

from sfincs_runs.build_base import inland_base
from sfincs_runs.build_base.inland_base import InlandSfincsBasePlan


class _FakeComponent:
    def create(self, *args, **kwargs):
        return None

    def create_active(self, *args, **kwargs):
        return None

    def create_from_region(self, *args, **kwargs):
        return None


class _FakeSfincsModel:
    def __init__(self, root, mode, write_gis):
        self.root = Path(root)
        self.mode = mode
        self.write_gis = write_gis
        self.data_catalog = SimpleNamespace(from_dict=lambda *args, **kwargs: None)
        self.grid = _FakeComponent()
        self.elevation = _FakeComponent()
        self.mask = _FakeComponent()
        self.roughness = _FakeComponent()
        self.subgrid = _FakeComponent()
        self.config = {}

    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sfincs.inp").write_text(
            "smaxfile = sfincs.smax\nsefffile = sfincs.seff\nksfile = sfincs.ks\n",
            encoding="utf-8",
        )
        (self.root / "sfincs.dep").write_text("dep\n", encoding="utf-8")


def test_stale_rain_on_grid_sfincs_base_rebuilds_with_native_physics(monkeypatch, tmp_path):
    model_root = tmp_path / "data/sfincs/base"
    model_root.mkdir(parents=True)
    (model_root / "sfincs.inp").write_text("qinf = 0\n", encoding="utf-8")
    (model_root / "sfincs.dep").write_text("dep\n", encoding="utf-8")
    plan = InlandSfincsBasePlan(
        base_model_root=model_root,
        region=tmp_path / "region.geojson",
        dem=tmp_path / "dem.tif",
        landcover=tmp_path / "landcover.tif",
        hsg=tmp_path / "hsg.tif",
        ksat=tmp_path / "ksat.tif",
        handoff_manifest=tmp_path / "handoff.yaml",
        model_crs="EPSG:32617",
        grid_resolution_m=30.0,
        ready_to_build=True,
        missing_inputs=(),
        built=True,
    )
    for path in (plan.region, plan.dem, plan.landcover, plan.hsg, plan.ksat, plan.handoff_manifest):
        path.write_text("placeholder\n", encoding="utf-8")

    calls = {"validate": 0, "infiltration": 0}

    def fake_validate(model_root, config):
        calls["validate"] += 1
        if calls["validate"] == 1:
            raise RuntimeError("SFINCS rain-on-grid model lacks active native infiltration")
        return {"infiltration": {"status": "active"}, "roughness": {"spatially_varying": True}}

    def fake_setup_hydromt_infiltration(*args, **kwargs):
        calls["infiltration"] += 1
        return {"written": True, "method": "cn_with_recovery"}

    monkeypatch.setattr(inland_base, "validate_physics", fake_validate)
    monkeypatch.setattr(inland_base, "setup_hydromt_infiltration", fake_setup_hydromt_infiltration)
    monkeypatch.setattr(inland_base, "add_inland_outflow_boundary", lambda *args, **kwargs: {"outflow_zmax_m": 1.0})

    summary = inland_base._build_inland_sfincs_base_plan(
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "inland_coupling": {
                "infiltration": {
                    "enabled": True,
                    "method": "cn_with_recovery",
                    "hsg": "hsg.tif",
                    "ksat": "ksat.tif",
                    "effective": 0.5,
                }
            },
        },
        {"location_root": tmp_path},
        plan,
        model_cls=_FakeSfincsModel,
        force=False,
    )

    assert summary["status"] == "rebuilt_stale_native_physics"
    assert summary["built"] is True
    assert calls == {"validate": 2, "infiltration": 1}


def test_stream_boundary_selection_falls_back_when_river_direction_blocks_upstream_crossing():
    point = Point(1.0, 0.0)
    rivers = gpd.GeoDataFrame(
        {
            "_boundary_river_uid": [7],
            "river_geometry_source": ["hydromt_wflow_setup_rivers"],
            "river_geometry_source_path": ["rivers.geojson"],
        },
        geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])],
        crs="EPSG:32617",
    )
    candidates = gpd.GeoDataFrame(
        {
            "river_index": [0],
            "_boundary_river_uid": [7],
            "river_id": ["reach-7"],
            "river_geometry_source": ["hydromt_wflow_setup_rivers"],
            "river_geometry_source_path": ["rivers.geojson"],
        },
        geometry=[Point(10.0, 0.0)],
        crs="EPSG:32617",
    )

    selected = inland_base._select_upstream_stream_boundary_intersection(
        point,
        rivers,
        candidates,
        {"sfincs_handoff_id": "greensboro_rural_inflow_02", "wflow_submodel_id": "greensboro_rural"},
    )

    assert selected["geometry"].equals(Point(10.0, 0.0))
    assert selected["handoff_location_review_status"] == "review_required_stream_boundary_direction_fallback"
    assert selected["stream_boundary_upstream_candidate_count"] == 0


def test_stream_boundary_selection_falls_back_when_nearest_reach_does_not_cross_boundary():
    point = Point(1.0, 0.0)
    rivers = gpd.GeoDataFrame(
        {
            "_boundary_river_uid": [1, 2],
            "river_geometry_source": ["hydromt_wflow_setup_rivers", "hydromt_wflow_setup_rivers"],
            "river_geometry_source_path": ["rivers.geojson", "rivers.geojson"],
        },
        geometry=[
            LineString([(0.0, 0.0), (2.0, 0.0)]),
            LineString([(8.0, 0.0), (10.0, 0.0)]),
        ],
        crs="EPSG:32617",
    )
    candidates = gpd.GeoDataFrame(
        {
            "river_index": [1],
            "_boundary_river_uid": [2],
            "river_id": ["reach-2"],
            "river_geometry_source": ["hydromt_wflow_setup_rivers"],
            "river_geometry_source_path": ["rivers.geojson"],
        },
        geometry=[Point(10.0, 0.0)],
        crs="EPSG:32617",
    )

    selected = inland_base._select_upstream_stream_boundary_intersection(
        point,
        rivers,
        candidates,
        {"sfincs_handoff_id": "greensboro_rural_inflow_04", "wflow_submodel_id": "greensboro_rural"},
    )

    assert selected["geometry"].equals(Point(10.0, 0.0))
    assert selected["handoff_location_review_status"] == "review_required_stream_boundary_nearest_reach_fallback"


def test_handoff_writer_persists_wflow_native_rivers_inflow(monkeypatch, tmp_path):
    network = gpd.GeoDataFrame(
        {"site_no": ["placeholder"]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:32617",
    )
    network_path = tmp_path / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True)
    network.to_file(network_path, driver="GeoJSON")

    handoff = gpd.GeoDataFrame(
        {
            "site_no": ["001"],
            "sfincs_handoff_id": ["test_inflow_01"],
            "wflow_submodel_id": ["test_submodel"],
            "sfincs_domain_id": ["test_domain"],
            "handoff_placement": ["stream_boundary_intersection"],
            "handoff_location_review_status": ["accepted"],
        },
        geometry=[Point(1.0, 0.0)],
        crs="EPSG:32617",
    )
    monkeypatch.setattr(inland_base, "_accepted_handoff_gages", lambda config, location_root: handoff.copy())
    monkeypatch.setattr(
        inland_base,
        "_snap_handoff_locations_to_domain_boundary",
        lambda handoff, config, paths, domain_region=None: handoff.copy(),
    )

    rivers_path = tmp_path / "data/wflow/base/test_submodel/staticgeoms/rivers.geojson"
    rivers_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"river_id": ["inside", "outside"]},
        geometry=[
            LineString([(0.0, 0.0), (2.0, 0.0)]),
            LineString([(20.0, 20.0), (21.0, 21.0)]),
        ],
        crs="EPSG:32617",
    ).to_file(rivers_path, driver="GeoJSON")

    region_path = tmp_path / "domain.geojson"
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(-1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]).envelope],
        crs="EPSG:32617",
    ).to_file(region_path, driver="GeoJSON")

    output = tmp_path / "data/sfincs/domains/test_domain/base/gis/wflow_handoff_sources.geojson"
    summary = inland_base.write_inland_sfincs_handoff_locations(
        {
            "project": {"model_crs": "EPSG:32617"},
            "sfincs": {"model_crs": "EPSG:32617"},
            "wflow": {"base_model_root": "data/wflow/base"},
        },
        {"location_root": tmp_path},
        output=output,
        domain_region=region_path,
        sfincs_domain_id="test_domain",
    )

    rivers_inflow = output.parent / "rivers_inflow.geojson"
    assert summary["rivers_inflow_count"] == 1
    assert summary["rivers_inflow"] == rivers_inflow
    assert rivers_inflow.exists()
    persisted = gpd.read_file(rivers_inflow)
    assert len(persisted) == 1


def test_sfincs_rivers_inflow_geoms_resolves_plain_model_root(tmp_path):
    model_root = tmp_path / "base"
    rivers_path = model_root / "gis/rivers_inflow.geojson"
    rivers_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"river_id": ["reach-1"]},
        geometry=[LineString([(0.0, 0.0), (1.0, 0.0)])],
        crs="EPSG:32617",
    ).to_file(rivers_path, driver="GeoJSON")

    model = SimpleNamespace(root=model_root, crs="EPSG:32617", components={})
    rivers = inland_base.sfincs_rivers_inflow_geoms(model)

    assert len(rivers) == 1
    assert str(rivers.crs) == "EPSG:32617"
