import json

import geopandas as gpd
from shapely.geometry import box

from design_events.collect_sources.lcra_hydromet import (
    discover_lcra_hydromet_flow_sites,
    lcra_hydromet_flow_site_frame,
    summarize_lcra_hydromet_headwater_coverage,
)
from design_events.collect_sources.workflow import plan


def _paths(tmp_path):
    location_root = tmp_path / "locations/austin"
    source_artifacts_root = location_root / "data/sources/source_artifacts"
    return {
        "repo_root": tmp_path,
        "location_name": "austin",
        "location_root": location_root,
        "source_artifacts_root": source_artifacts_root,
    }


def _write_region(paths):
    region_path = paths["location_root"] / "data/static/aoi/wflow_nhdplus_watersheds.geojson"
    region_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        [{"wflow_submodel_id": "austin_p5u", "geometry": box(-98.4, 30.0, -97.6, 30.8)}],
        crs="EPSG:4326",
    ).to_file(region_path, driver="GeoJSON")
    return "data/static/aoi/wflow_nhdplus_watersheds.geojson"


def test_lcra_hydromet_flow_sites_are_supplemental_and_spatially_filtered(tmp_path):
    paths = _paths(tmp_path)
    search_geometry = _write_region(paths)
    records = [
        {
            "agency": "LCRA",
            "siteNumber": 3018,
            "siteName": "Hamilton Creek near Marble Falls",
            "flow": 8.31,
            "stage": 6.94,
            "dateTime": "2026-06-23T16:55:30Z",
            "latitude": 30.6122,
            "longitude": -98.2339,
        },
        {
            "agency": "LCRA",
            "siteNumber": 3015,
            "siteName": "Burnet 1 WSW",
            "flow": None,
            "latitude": 30.75675,
            "longitude": -98.2353,
        },
        {
            "agency": "LCRA",
            "siteNumber": 9999,
            "siteName": "Outside flow site",
            "flow": 4.0,
            "latitude": 31.1,
            "longitude": -98.2,
        },
    ]

    frame = lcra_hydromet_flow_site_frame(records, {"search_geometry": search_geometry}, paths)

    assert list(frame["site_no"]) == ["LCRA:3018"]
    row = frame.iloc[0]
    assert row["source_network"] == "LCRA Hydromet"
    assert row["review_status"] == "supplemental_candidate"
    assert row["flow_cfs"] == 8.31

    summary = summarize_lcra_hydromet_headwater_coverage(
        {"collection": {"lcra_hydromet": {"search_geometry": search_geometry}}},
        paths,
        site_records=records,
    )
    assert summary["flow_sites_in_region"] == 1
    assert summary["northwest_flow_sites"] == 1


def test_lcra_hydromet_flow_sites_exclude_usgs_and_near_duplicate_usgs_locations(tmp_path):
    paths = _paths(tmp_path)
    search_geometry = _write_region(paths)
    usgs_path = paths["location_root"] / "data/sources/usgs_streamgages/streamgage_network.geojson"
    usgs_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        [{"site_no": "08159000", "site_name": "Onion Ck at US Hwy 183", "geometry": box(-97.689, 30.177, -97.688, 30.178).centroid}],
        crs="EPSG:4326",
    ).to_file(usgs_path, driver="GeoJSON")
    records = [
        {
            "agency": "USGS",
            "siteNumber": "08159000",
            "siteName": "Onion Ck at US Hwy 183, Austin, TX",
            "flow": 4.0,
            "stage": 2.0,
            "latitude": 30.1775,
            "longitude": -97.6885,
        },
        {
            "agency": "LCRA",
            "siteNumber": 4598,
            "siteName": "Onion Creek at Hwy 183, Austin",
            "flow": 7.56,
            "stage": 5.06,
            "latitude": 30.17732,
            "longitude": -97.68896,
        },
        {
            "agency": "LCRA",
            "siteNumber": 3920,
            "siteName": "Cow Creek near Lago Vista",
            "flow": 23.57,
            "stage": 5.18,
            "latitude": 30.50472,
            "longitude": -98.04361,
        },
    ]

    frame = lcra_hydromet_flow_site_frame(
        records,
        {
            "search_geometry": search_geometry,
            "distinct_from_usgs_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
            "distinct_distance_m": 150,
        },
        paths,
    )

    assert list(frame["site_no"]) == ["LCRA:3920"]


def test_discover_lcra_hydromet_flow_sites_writes_geojson_and_manifest(tmp_path):
    paths = _paths(tmp_path)
    search_geometry = _write_region(paths)
    config = {
        "collection": {
            "start": "1979-01-01",
            "end": "2022-12-31",
            "lcra_hydromet": {
                "output": "data/sources/lcra_hydromet/flow_sites.geojson",
                "current_output": "data/sources/lcra_hydromet/flow_sites_current.csv",
                "search_geometry": search_geometry,
            },
        }
    }

    result = discover_lcra_hydromet_flow_sites(
        config,
        paths,
        site_records=[
            {
                "agency": "LCRA",
                "siteNumber": 3920,
                "siteName": "Cow Creek near Lago Vista",
                "flow": 23.57,
                "stage": 5.18,
                "latitude": 30.50472,
                "longitude": -98.04361,
            }
        ],
    )

    assert result["candidate_count"] == 1
    assert result["candidate_geojson"].exists()
    assert result["current_csv"].exists()
    payload = json.loads(result["candidate_geojson"].read_text(encoding="utf-8"))
    assert payload["features"][0]["properties"]["site_no"] == "LCRA:3920"
    artifact = json.loads(
        (paths["source_artifacts_root"] / "lcra_hydromet_flow_sites.json").read_text(encoding="utf-8")
    )
    assert artifact["metadata"]["supplemental_source"] is True


def test_lcra_hydromet_is_collected_only_when_configured(tmp_path):
    base = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "usgs_streamgages": {},
            "stream_geo_nldi": {},
            "national_hydrography": {},
            "nwm": {},
            "aorc_sst": {},
        }
    }
    assert plan(base, {"outputs_root": tmp_path}).source_names == (
        "usgs_streamgages",
        "stream_geo_nldi",
        "national_hydrography",
        "nwm",
        "aorc_sst",
    )

    with_hydromet = {
        "collection": {
            **base["collection"],
            "lcra_hydromet": {},
        }
    }
    assert plan(with_hydromet, {"outputs_root": tmp_path}).source_names == (
        "usgs_streamgages",
        "lcra_hydromet",
        "stream_geo_nldi",
        "national_hydrography",
        "nwm",
        "aorc_sst",
    )
