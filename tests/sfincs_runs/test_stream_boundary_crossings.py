import geopandas as gpd
from shapely.geometry import LineString, Point, box

from sfincs_runs.build_base.crossings import (
    coverage_box_crossings,
    select_encompassing_huc,
    stream_boundary_inflow_crossings,
    subbasin_submodels_from_crossings,
)


def test_inflow_crossings_rank_by_uparea_and_drop_subthreshold():
    # SFINCS coverage box; rivers approach from outside and cross its edges.
    coverage = box(0.0, 0.0, 10.0, 10.0)
    rivers = gpd.GeoDataFrame(
        {"uparea": [100.0, 1.0]},
        geometry=[
            # trunk river crossing the right edge at (10, 5): a real inflow
            LineString([(15.0, 5.0), (5.0, 5.0)]),
            # trickle crossing the right edge at (10, 8): below the threshold
            LineString([(12.0, 8.0), (8.0, 8.0)]),
        ],
        crs="EPSG:4326",
    )

    crossings = stream_boundary_inflow_crossings(rivers, coverage, min_uparea_km2=5.0)

    # Only the trunk survives the drainage-area filter.
    assert list(crossings["uparea_km2"]) == [100.0]
    point = crossings.geometry.iloc[0]
    assert (round(point.x, 6), round(point.y, 6)) == (10.0, 5.0)


def test_outflow_trunk_is_excluded():
    # Rivers are digitised upstream->downstream. The trunk that LEAVES the box
    # (downstream end outside) is the SFINCS outflow, not a discharge source,
    # even though it carries the largest drainage area.
    coverage = box(0.0, 0.0, 10.0, 10.0)
    rivers = gpd.GeoDataFrame(
        {"uparea": [200.0, 100.0]},
        geometry=[
            # outflow: inside (5,5) -> outside (15,5), downstream end outside
            LineString([(5.0, 5.0), (15.0, 5.0)]),
            # inflow: outside (5,2) <- ... downstream end (5,5) inside the box
            LineString([(15.0, 2.0), (5.0, 2.0)]),
        ],
        crs="EPSG:4326",
    )

    crossings = stream_boundary_inflow_crossings(rivers, coverage, min_uparea_km2=5.0)

    assert list(crossings["uparea_km2"]) == [100.0]
    point = crossings.geometry.iloc[0]
    assert (round(point.x, 6), round(point.y, 6)) == (10.0, 2.0)


def test_crossings_become_one_subbasin_submodel_each():
    # Each inflow crossing is a Wflow subbasin outlet (region = subbasin snapped at
    # the crossing, with the crossing's drainage area) and a SFINCS source point.
    crossings = gpd.GeoDataFrame(
        {"uparea_km2": [100.0, 30.0]},
        geometry=[Point(10.0, 5.0), Point(10.0, 2.0)],
        crs="EPSG:4326",
    )

    submodels = subbasin_submodels_from_crossings(
        crossings, project_name="greensboro", sfincs_domain_id="greensboro_main"
    )

    assert [s["wflow_submodel_id"] for s in submodels] == [
        "greensboro_main_inflow_01",
        "greensboro_main_inflow_02",
    ]
    first = submodels[0]
    assert first["region"] == {"subbasin": [10.0, 5.0], "uparea": 100.0}
    assert first["region_kind"] == "subbasin"
    assert first["sfincs_domain_ids"] == ["greensboro_main"]
    # The handoff id ties this Wflow outlet to its SFINCS discharge source point.
    assert first["sfincs_handoff_ids"] == ["greensboro_main_inflow_01"]
    # No USGS gage backs a crossing-derived outlet.
    assert first["gauge_site_nos"] == []


def test_coverage_box_crossings_ids_each_inflow_per_domain():
    # Two coverage boxes, one inflow each; ids are stable and namespaced by domain.
    boxes = gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main", "greensboro_east"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(20.0, 0.0, 30.0, 10.0)],
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {"uparea": [100.0, 50.0]},
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),   # -> west box crossing (10, 5)
            LineString([(15.0, 5.0), (25.0, 5.0)]),  # -> east box crossing (20, 5)
        ],
        crs="EPSG:4326",
    )

    crossings = coverage_box_crossings(boxes, rivers, project_name="greensboro", min_uparea_km2=5.0)

    assert set(crossings["sfincs_handoff_id"]) == {"greensboro_main_inflow_01", "greensboro_east_inflow_01"}
    main = crossings[crossings["sfincs_domain_id"] == "greensboro_main"].iloc[0]
    assert (round(main.geometry.x, 6), round(main.geometry.y, 6)) == (10.0, 5.0)


def test_select_smallest_single_huc_covering_all_boxes():
    coverage = box(0.0, 0.0, 30.0, 10.0)  # spans both footprints

    def huc_loader(level):
        if level == 8:
            # two HUC8s split the coverage -- neither single polygon covers all
            return gpd.GeoDataFrame(
                {"huc_id": ["03010001", "03010002"]},
                geometry=[box(0.0, 0.0, 15.0, 10.0), box(15.0, 0.0, 30.0, 10.0)],
                crs="EPSG:4326",
            )
        if level == 6:
            return gpd.GeoDataFrame(
                {"huc_id": ["030100"]},
                geometry=[box(-5.0, -5.0, 35.0, 15.0)],
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    selected = select_encompassing_huc(coverage, huc_loader, levels=(8, 6, 4))

    assert selected["level"] == 6
    assert selected["huc_id"] == "030100"
    assert selected["kind"] == "single"
    assert selected["geometry"].covers(coverage)


def test_select_encompassing_huc_unions_when_no_single_polygon_covers():
    # A coverage box straddling two HUCs (the Greensboro divide case): no single HUC
    # covers it, so the union of the HUCs it intersects becomes the domain.
    coverage = box(0.0, 0.0, 30.0, 10.0)

    def huc_loader(level):
        if level == 8:
            return gpd.GeoDataFrame(
                {"huc_id": ["03030002", "03010104"]},
                geometry=[box(0.0, 0.0, 15.0, 10.0), box(15.0, 0.0, 30.0, 10.0)],
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame({"huc_id": []}, geometry=[], crs="EPSG:4326")

    selected = select_encompassing_huc(coverage, huc_loader, levels=(8, 6, 4), allow_union=True)

    assert selected["kind"] == "union"
    assert selected["level"] == 8
    assert set(selected["huc_ids"]) == {"03030002", "03010104"}
    assert selected["geometry"].covers(coverage)
