import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from design_events.collect_sources.workflow import plan
from design_events.collect_sources.usgs_streamgages import (
    active_streamgage_candidate_artifact_ready,
    build_reviewed_streamgage_decisions,
    collect_usgs_streamflow_records,
    discover_active_streamgage_candidates,
    fetch_nwis_discharge_records,
    fetch_nwis_streamgage_site_records,
    write_reviewed_streamgage_network,
)


def test_discover_active_streamgage_candidates_writes_review_schema_geojson(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": tmp_path / "locations/greensboro",
        "source_artifacts_root": tmp_path / "locations/greensboro/data/sources/source_artifacts",
    }
    config = {
        "collection": {
            "start": "2000-01-01",
            "end": "2020-01-01",
            "usgs_streamgages": {
                "candidate_output": "data/sources/usgs_streamgages/streamgage_candidates.geojson",
                "active_records_only": True,
                "discovery": {
                    "parameter_cd": "00060",
                    "site_status": "active",
                },
            },
        }
    }
    site_records = [
        {
            "site_no": "02095000",
            "station_nm": "HAW RIVER NEAR BENAJAH, NC",
            "site_status": "active",
            "dec_lat_va": 36.234,
            "dec_long_va": -79.555,
            "drain_area_va": 168.0,
            "parm_cd": "00060",
            "begin_date": "2000-01-01",
            "end_date": "2020-01-01",
            "count_nu": 7306,
        },
        {
            "site_no": "02095000",
            "station_nm": "HAW RIVER NEAR BENAJAH, NC",
            "site_status": "active",
            "dec_lat_va": 36.234,
            "dec_long_va": -79.555,
            "drain_area_va": 168.0,
            "parm_cd": "00060",
            "begin_date": "2015-01-01",
            "end_date": "2020-01-01",
            "count_nu": 50000,
            "data_type_cd": "iv",
        },
        {
            "site_no": "02095100",
            "station_nm": "INACTIVE CREEK NEAR GREENSBORO, NC",
            "site_status": "inactive",
            "dec_lat_va": 36.09,
            "dec_long_va": -79.79,
            "drain_area_va": 10.0,
            "parm_cd": "00060",
            "begin_date": "1970-01-01",
            "end_date": "1990-01-01",
        },
        {
            "site_no": "02095200",
            "station_nm": "STAGE ONLY CREEK NEAR GREENSBORO, NC",
            "site_status": "active",
            "dec_lat_va": 36.1,
            "dec_long_va": -79.8,
            "drain_area_va": 5.0,
            "parm_cd": "00065",
            "begin_date": "2000-01-01",
            "end_date": "2020-01-01",
        },
    ]

    result = discover_active_streamgage_candidates(config, paths, site_records=site_records)

    assert result["candidate_count"] == 1
    assert result["candidate_geojson"] == paths["location_root"] / "data/sources/usgs_streamgages/streamgage_candidates.geojson"
    geojson = json.loads(result["candidate_geojson"].read_text(encoding="utf-8"))
    assert len(geojson["features"]) == 1
    properties = geojson["features"][0]["properties"]
    assert properties == {
        "site_no": "02095000",
        "site_name": "HAW RIVER NEAR BENAJAH, NC",
        "status": "active",
        "drainage_area_sqmi": 168.0,
        "period_start": "2000-01-01",
        "period_end": "2020-01-01",
        "record_years": 20.0,
        "completeness_score": 1.0,
        "roles": [],
        "frequency_basis": None,
        "wflow_submodel_id": None,
        "sfincs_domain_id": None,
        "sfincs_handoff_id": None,
        "review_status": "candidate",
        "review_notes": "",
    }
    assert geojson["features"][0]["geometry"] == {
        "type": "Point",
        "coordinates": [-79.555, 36.234],
    }
    manifest = json.loads(
        (paths["source_artifacts_root"] / "usgs_streamgages_active_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source"] == "usgs_streamgages"
    assert manifest["kind"] == "active_candidates"
    assert manifest["metadata"]["candidate_count"] == 1
    assert manifest["metadata"]["parameter_cd"] == "00060"
    assert "discovery_signature" in manifest["metadata"]


def test_discover_active_streamgage_candidates_refreshes_when_discovery_signature_changes(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    candidate_path = location_root / "data/sources/usgs_streamgages/streamgage_candidates.geojson"
    source_artifact = location_root / "data/sources/source_artifacts/usgs_streamgages_active_candidates.json"
    candidate_path.parent.mkdir(parents=True)
    source_artifact.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    source_artifact.write_text(
        json.dumps(
            {
                "study_location": "greensboro",
                "source": "usgs_streamgages",
                "kind": "active_candidates",
                "status": "complete",
                "start": "2000-01-01T00:00:00",
                "end": "2020-01-01T00:00:00",
                "metadata": {"candidate_count": 0},
                "artifacts": {"candidate_geojson": str(candidate_path)},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "collection": {
            "start": "2000-01-01",
            "end": "2020-01-01",
            "usgs_streamgages": {
                "candidate_output": "data/sources/usgs_streamgages/streamgage_candidates.geojson",
                "discovery": {"bbox": [-80.0, 36.0, -79.5, 36.3]},
            },
        }
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": location_root,
        "source_artifacts_root": source_artifact.parent,
    }

    assert active_streamgage_candidate_artifact_ready(config, paths) is False

    result = discover_active_streamgage_candidates(
        config,
        paths,
        site_records=[
            {
                "site_no": "02095000",
                "station_nm": "HAW RIVER NEAR BENAJAH, NC",
                "site_status": "active",
                "dec_lat_va": 36.234,
                "dec_long_va": -79.555,
                "drain_area_va": 168.0,
                "parm_cd": "00060",
                "begin_date": "2000-01-01",
                "end_date": "2020-01-01",
                "count_nu": 7306,
            }
        ],
        skip_existing=True,
    )

    assert result["reused"] is False
    assert result["candidate_count"] == 1
    assert active_streamgage_candidate_artifact_ready(config, paths) is True


def test_source_collection_plan_includes_usgs_streamgages_before_hydrologic_state(tmp_path):
    config = {
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

    collection_plan = plan(config, {"outputs_root": tmp_path})

    assert collection_plan.source_names == ("usgs_streamgages", "stream_geo_nldi", "national_hydrography", "nwm", "aorc_sst")
    assert collection_plan.settings_for("usgs_streamgages")["usgs_streamgages"] == {}


def test_fetch_nwis_streamgage_records_rejects_unbounded_discovery(tmp_path):
    with pytest.raises(ValueError, match="requires a bounded search geometry"):
        fetch_nwis_streamgage_site_records({"discovery": {"parameter_cd": "00060"}}, {"location_root": tmp_path})


def test_fetch_nwis_streamgage_records_uses_compatible_site_and_series_requests(monkeypatch, tmp_path):
    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    series_text = "\n".join(
        [
            "agency_cd\tsite_no\tstation_nm\tdec_lat_va\tdec_long_va\tdata_type_cd\tparm_cd\tbegin_date\tend_date\tcount_nu",
            "5s\t15s\t50s\t16s\t16s\t2s\t5s\t20d\t20d\t5n",
            "USGS\t02095000\tHAW RIVER NEAR BENAJAH, NC\t36.234\t-79.555\tdv\t00060\t2000-01-01\t2020-01-01\t7306",
        ]
    )
    expanded_text = "\n".join(
        [
            "agency_cd\tsite_no\tstation_nm\tlat_va\tlong_va\tdec_lat_va\tdec_long_va\tdrain_area_va",
            "5s\t15s\t50s\t16s\t16s\t16s\t16s\t8s",
            "USGS\t02095000\tHAW RIVER NEAR BENAJAH, NC\t361402\t0793318\t36.234\t-79.555\t168.0",
        ]
    )

    def fake_get(url, params, timeout):
        calls.append((url, dict(params), timeout))
        if params.get("seriesCatalogOutput") == "true":
            return Response(series_text)
        return Response(expanded_text)

    monkeypatch.setattr("design_events.collect_sources.usgs_streamgages.requests.get", fake_get)

    records = fetch_nwis_streamgage_site_records(
        {
            "discovery": {
                "parameter_cd": "00060",
                "site_status": "active",
                "bbox": [-80.0, 36.0, -79.5, 36.3],
                "has_data_type_cd": "dv",
            }
        },
        {"location_root": tmp_path},
    )

    assert len(calls) == 2
    assert calls[0][1]["siteOutput"] == "Basic"
    assert calls[0][1]["seriesCatalogOutput"] == "true"
    assert calls[0][1]["hasDataTypeCd"] == "dv"
    assert calls[1][1]["siteOutput"] == "Expanded"
    assert "seriesCatalogOutput" not in calls[1][1]
    assert records == [
        {
            "agency_cd": "USGS",
            "site_no": "02095000",
            "station_nm": "HAW RIVER NEAR BENAJAH, NC",
            "dec_lat_va": "36.234",
            "dec_long_va": "-79.555",
            "data_type_cd": "dv",
            "parm_cd": "00060",
            "begin_date": "2000-01-01",
            "end_date": "2020-01-01",
            "count_nu": "7306",
            "drain_area_va": "168.0",
        }
    ]


def test_fetch_nwis_streamgage_records_buffers_search_geometry_bbox(monkeypatch, tmp_path):
    location_root = tmp_path / "locations/greensboro"
    watershed_path = location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson"
    watershed_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"name": ["wflow"]},
        geometry=[box(-80.0, 36.0, -79.5, 36.3)],
        crs="EPSG:4326",
    ).to_file(watershed_path, driver="GeoJSON")
    calls = []

    class Response:
        text = "\n".join(
            [
                "agency_cd\tsite_no\tstation_nm\tdec_lat_va\tdec_long_va\tdata_type_cd\tparm_cd\tbegin_date\tend_date\tcount_nu",
                "5s\t15s\t50s\t16s\t16s\t2s\t5s\t20d\t20d\t5n",
                "USGS\t02095000\tHAW RIVER NEAR BENAJAH, NC\t36.234\t-79.555\tdv\t00060\t2000-01-01\t2020-01-01\t7306",
            ]
        )

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        calls.append(dict(params))
        return Response()

    monkeypatch.setattr("design_events.collect_sources.usgs_streamgages.requests.get", fake_get)

    fetch_nwis_streamgage_site_records(
        {
            "discovery": {
                "search_geometry": "data/static/aoi/wflow_nhdplus_watersheds.geojson",
                "hydrologic_buffer_km": 50,
            }
        },
        {"location_root": location_root},
    )

    west, south, east, north = [float(value) for value in calls[0]["bBox"].split(",")]
    assert west < -80.0
    assert south < 36.0
    assert east > -79.5
    assert north > 36.3


def test_collect_usgs_streamflow_records_uses_reviewed_active_gages_and_writes_manifest(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    reviewed_network = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    output = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    reviewed_network.parent.mkdir(parents=True, exist_ok=True)
    reviewed_network.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095000",
                            "site_name": "HAW RIVER NEAR BENAJAH, NC",
                            "status": "active",
                            "roles": ["frequency", "calibration"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.55, 36.23]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095100",
                            "site_name": "REJECTED CREEK",
                            "status": "active",
                            "roles": ["frequency"],
                            "review_status": "rejected",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.56, 36.24]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095200",
                            "site_name": "INACTIVE CREEK",
                            "status": "inactive",
                            "roles": ["frequency"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.57, 36.25]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-02",
            "usgs_streamgages": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
                "active_records_only": True,
                "streamflow_records": {
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                    "service": "dv",
                    "stat_cd": "00003",
                },
            },
        }
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": location_root,
        "source_artifacts_root": location_root / "data/sources/source_artifacts",
    }
    response_by_site = {
        "02095000": "\n".join(
            [
                "# comment",
                "agency_cd\tsite_no\tdatetime\t00060_00003\t00060_00003_cd",
                "5s\t15s\t20d\t14n\t10s",
                "USGS\t02095000\t2020-01-01\t1250\tA",
                "USGS\t02095000\t2020-01-02\t1300\tA",
            ]
        )
    }

    result = collect_usgs_streamflow_records(config, paths, response_text_by_site=response_by_site)

    assert result["record_count"] == 2
    assert result["site_count"] == 1
    assert result["streamflow_records_csv"] == output
    records = pd.read_csv(output, dtype={"site_no": str})
    assert records.to_dict("records") == [
        {
            "site_no": "02095000",
            "time": "2020-01-01T00:00:00",
            "discharge_cfs": 1250.0,
            "source": "usgs_dv",
        },
        {
            "site_no": "02095000",
            "time": "2020-01-02T00:00:00",
            "discharge_cfs": 1300.0,
            "source": "usgs_dv",
        },
    ]
    manifest = json.loads(
        (paths["source_artifacts_root"] / "usgs_streamgages_streamflow_records.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["kind"] == "streamflow_records"
    assert manifest["metadata"]["site_count"] == 1
    assert manifest["metadata"]["service"] == "dv"


def test_collect_usgs_streamflow_records_refreshes_when_reviewed_network_grows(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    reviewed_network = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    output = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    source_artifact = location_root / "data/sources/source_artifacts/usgs_streamgages_streamflow_records.json"
    reviewed_network.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    reviewed_network.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095000",
                            "site_name": "SOUTH BUFFALO CREEK",
                            "status": "active",
                            "roles": ["frequency"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.55, 36.23]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095500",
                            "site_name": "NORTH BUFFALO CREEK",
                            "status": "active",
                            "roles": ["frequency"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.56, 36.24]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "site_no": "02095000",
                "time": "2020-01-01T00:00:00",
                "discharge_cfs": 1250.0,
                "source": "usgs_dv",
            }
        ]
    ).to_csv(output, index=False)
    source_artifact.write_text(
        json.dumps(
            {
                "study_location": "greensboro",
                "source": "usgs_streamgages",
                "kind": "streamflow_records",
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-02T00:00:00",
                "metadata": {"site_count": 1, "record_count": 1},
                "artifacts": {"streamflow_records_csv": str(output)},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-02",
            "usgs_streamgages": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
                "streamflow_records": {
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                    "service": "dv",
                    "stat_cd": "00003",
                },
            },
        }
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": location_root,
        "source_artifacts_root": location_root / "data/sources/source_artifacts",
    }
    response_by_site = {
        "02095000": "\n".join(
            [
                "agency_cd\tsite_no\tdatetime\t00060_00003\t00060_00003_cd",
                "5s\t15s\t20d\t14n\t10s",
                "USGS\t02095000\t2020-01-01\t1250\tA",
            ]
        ),
        "02095500": "\n".join(
            [
                "agency_cd\tsite_no\tdatetime\t00060_00003\t00060_00003_cd",
                "5s\t15s\t20d\t14n\t10s",
                "USGS\t02095500\t2020-01-01\t2150\tA",
            ]
        ),
    }

    result = collect_usgs_streamflow_records(
        config,
        paths,
        response_text_by_site=response_by_site,
        skip_existing=True,
    )

    assert result["reused"] is False
    assert result["site_count"] == 2
    assert sorted(pd.read_csv(output, dtype={"site_no": str})["site_no"].unique()) == ["02095000", "02095500"]


def test_collect_usgs_streamflow_records_resumes_missing_sites_from_cache(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    reviewed_network = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    output = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    source_artifact = location_root / "data/sources/source_artifacts/usgs_streamgages_streamflow_records.json"
    reviewed_network.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    reviewed_network.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095000",
                            "status": "active",
                            "roles": ["frequency"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.55, 36.23]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095500",
                            "status": "active",
                            "roles": ["frequency"],
                            "review_status": "accepted",
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.56, 36.24]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "site_no": "02095000",
                "time": "2020-01-01T00:00:00",
                "discharge_cfs": 1250.0,
                "source": "usgs_dv",
            }
        ]
    ).to_csv(output, index=False)
    source_artifact.write_text(
        json.dumps(
            {
                "study_location": "greensboro",
                "source": "usgs_streamgages",
                "kind": "streamflow_records",
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-02T00:00:00",
                "metadata": {"site_count": 2, "record_count": 1},
                "artifacts": {"streamflow_records_csv": str(output)},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-02",
            "usgs_streamgages": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
                "streamflow_records": {
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                    "service": "dv",
                    "stat_cd": "00003",
                },
            },
        }
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": location_root,
        "source_artifacts_root": location_root / "data/sources/source_artifacts",
    }
    fetched_sites = []

    def fake_fetch(spec, site_no, start, end):
        fetched_sites.append(site_no)
        if site_no == "02095000":
            raise AssertionError("cached site should not be refetched")
        return [
            {
                "site_no": "02095500",
                "time": "2020-01-01T00:00:00",
                "discharge_cfs": 2150.0,
                "source": "usgs_dv",
            }
        ]

    monkeypatch.setattr(
        "design_events.collect_sources.usgs_streamgages.fetch_nwis_discharge_records",
        fake_fetch,
    )

    result = collect_usgs_streamflow_records(config, paths, skip_existing=True)

    assert result["reused"] is False
    assert fetched_sites == ["02095500"]
    assert sorted(pd.read_csv(output, dtype={"site_no": str})["site_no"].unique()) == ["02095000", "02095500"]


def test_write_reviewed_streamgage_network_promotes_review_decisions(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    paths = {
        "location_root": location_root,
        "repo_root": tmp_path,
    }
    config = {
        "collection": {
            "usgs_streamgages": {
                "candidate_output": "data/sources/usgs_streamgages/streamgage_candidates.geojson",
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
            }
        }
    }
    candidate_path = location_root / "data/sources/usgs_streamgages/streamgage_candidates.geojson"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095000",
                            "site_name": "SOUTH BUFFALO CR NEAR GREENSBORO, NC",
                            "status": "active",
                            "review_status": "candidate",
                            "roles": [],
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.725833, 36.06]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095500",
                            "site_name": "NORTH BUFFALO CREEK NEAR GREENSBORO, NC",
                            "status": "active",
                            "review_status": "candidate",
                            "roles": [],
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.708056, 36.120556]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = write_reviewed_streamgage_network(
        config,
        paths,
        [
            {
                "site_no": "02095000",
                "review_status": "accepted",
                "roles": ["frequency", "calibration", "sfincs_handoff"],
                "frequency_basis": "buffalo_creek",
                "wflow_submodel_id": "south_buffalo",
                "sfincs_domain_id": "greensboro_main",
                "sfincs_handoff_id": "south_buffalo_02095000",
                "review_notes": "Long active record near Greensboro.",
            },
            {
                "site_no": "02095500",
                "review_status": "rejected",
                "review_notes": "Rejected in this fixture.",
            },
        ],
    )

    assert result["accepted_count"] == 1
    assert result["reviewed_count"] == 2
    payload = json.loads(result["reviewed_network_geojson"].read_text(encoding="utf-8"))
    assert len(payload["features"]) == 2
    accepted = payload["features"][0]["properties"]
    assert accepted["site_no"] == "02095000"
    assert accepted["review_status"] == "accepted"
    assert accepted["roles"] == ["frequency", "calibration", "sfincs_handoff"]
    assert accepted["sfincs_handoff_id"] == "south_buffalo_02095000"


def test_write_reviewed_streamgage_network_requires_roles_for_accepted_gages(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    candidate_path = location_root / "data/sources/usgs_streamgages/streamgage_candidates.geojson"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "site_no": "02095000",
                            "site_name": "SOUTH BUFFALO CR NEAR GREENSBORO, NC",
                            "status": "active",
                            "review_status": "candidate",
                            "roles": [],
                        },
                        "geometry": {"type": "Point", "coordinates": [-79.725833, 36.06]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires at least one role"):
        write_reviewed_streamgage_network(
            {
                "collection": {
                    "usgs_streamgages": {
                        "candidate_output": "data/sources/usgs_streamgages/streamgage_candidates.geojson",
                        "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
                    }
                }
            },
            {"location_root": location_root, "repo_root": tmp_path},
            [{"site_no": "02095000", "review_status": "accepted"}],
        )


def test_build_reviewed_streamgage_decisions_uses_policy_not_notebook_literals(tmp_path):
    config = {
        "collection": {
            "usgs_streamgages": {
                "review_policy": {
                    "default_review_status": "accepted_with_warning",
                    "default_sfincs_domain_id": "greensboro_main",
                    "default_roles": ["calibration", "validation"],
                    "long_record_years": 50,
                    "long_record_roles": ["frequency", "validation"],
                    "handoff_roles": ["frequency", "calibration", "validation", "sfincs_handoff"],
                    "handoff_site_nos": {"02095000": "south_buffalo_02095000"},
                    "basin_rules": [
                        {
                            "contains": "SOUTH BUFFALO",
                            "frequency_basis": "south_buffalo_creek",
                            "wflow_submodel_id": "south_buffalo",
                        },
                        {
                            "contains": "BRUSH CREEK",
                            "frequency_basis": "brush_creek",
                            "wflow_submodel_id": "brush_creek",
                        },
                    ],
                }
            }
        }
    }
    candidate_records = [
        {
            "site_no": "02095000",
            "site_name": "SOUTH BUFFALO CR NEAR GREENSBORO, NC",
            "status": "active",
            "record_years": 97.0,
            "longitude": -79.7,
            "latitude": 36.0,
        },
        {
            "site_no": "02093877",
            "site_name": "BRUSH CREEK AT MUIRFIELD RD AT GREENSBORO, NC",
            "status": "active",
            "record_years": 22.0,
            "longitude": -79.8,
            "latitude": 36.1,
        },
    ]

    decisions = build_reviewed_streamgage_decisions(
        config,
        {"location_root": tmp_path / "locations/greensboro"},
        candidate_records=candidate_records,
    )

    by_site = {decision["site_no"]: decision for decision in decisions}
    assert by_site["02095000"]["sfincs_handoff_id"] == "south_buffalo_02095000"
    assert by_site["02095000"]["roles"] == ["frequency", "calibration", "validation", "sfincs_handoff"]
    assert by_site["02095000"]["wflow_submodel_id"] == "south_buffalo"
    assert by_site["02093877"]["sfincs_handoff_id"] is None
    assert by_site["02093877"]["roles"] == ["calibration", "validation"]
    assert by_site["02093877"]["frequency_basis"] == "brush_creek"


def test_build_reviewed_streamgage_decisions_allows_sfincs_domain_override(tmp_path):
    config = {
        "collection": {
            "usgs_streamgages": {
                "review_policy": {
                    "default_review_status": "candidate",
                    "default_sfincs_domain_id": "review_required",
                    "default_roles": ["calibration", "validation"],
                    "site_overrides": {
                        "08158000": {
                            "review_status": "accepted",
                            "roles": ["frequency", "calibration", "validation", "sfincs_handoff"],
                            "frequency_basis": "austin_basin",
                            "wflow_submodel_id": "austin_basin",
                            "sfincs_domain_id": "austin_p1u",
                            "sfincs_handoff_id": "austin_basin_08158000",
                        }
                    },
                }
            }
        }
    }

    decisions = build_reviewed_streamgage_decisions(
        config,
        {"location_root": tmp_path / "locations/austin"},
        candidate_records=[
            {
                "site_no": "08158000",
                "site_name": "COLORADO RIVER AT AUSTIN, TX",
                "status": "active",
                "record_years": 80.0,
                "longitude": -97.7,
                "latitude": 30.3,
            }
        ],
    )

    assert decisions[0]["review_status"] == "accepted"
    assert decisions[0]["sfincs_domain_id"] == "austin_p1u"
    assert decisions[0]["sfincs_handoff_id"] == "austin_basin_08158000"


def test_build_reviewed_streamgage_decisions_can_use_huc_region_review(tmp_path):
    location_root = tmp_path / "locations/austin"
    huc_path = location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson"
    huc_path.parent.mkdir(parents=True)
    hucs = gpd.GeoDataFrame(
        [
            {
                "wflow_submodel_id": "austin_p4u",
                "sfincs_domain_id": "austin_p4u",
                "huc_id": "12090205",
                "huc_level": 8,
                "geometry": box(-98.0, 30.0, -97.0, 31.0),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    hucs.to_file(huc_path, driver="GeoJSON")
    config = {"collection": {"usgs_streamgages": {}}}

    decisions = build_reviewed_streamgage_decisions(
        config,
        {"location_root": location_root, "repo_root": tmp_path},
        candidate_records=[
            {
                "site_no": "08158000",
                "site_name": "COLORADO RIVER AT AUSTIN, TX",
                "status": "active",
                "record_years": 80.0,
                "longitude": -97.7,
                "latitude": 30.3,
            },
            {
                "site_no": "00000000",
                "site_name": "OUTSIDE CREEK, TX",
                "status": "active",
                "record_years": 20.0,
                "longitude": -96.0,
                "latitude": 30.3,
            },
        ],
    )

    assert [decision["site_no"] for decision in decisions] == ["08158000"]
    assert decisions[0]["review_status"] == "accepted_with_warning"
    assert decisions[0]["roles"] == ["frequency", "calibration", "validation"]
    assert decisions[0]["frequency_basis"] == "austin_p4u"
    assert decisions[0]["wflow_submodel_id"] == "austin_p4u"
    assert decisions[0]["sfincs_domain_id"] == "austin_p4u"
    assert decisions[0]["sfincs_handoff_id"] is None
    assert "HUC-derived" in decisions[0]["review_notes"]


def test_fetch_nwis_discharge_records_builds_daily_values_request(monkeypatch):
    calls = []

    class Response:
        text = "\n".join(
            [
                "agency_cd\tsite_no\tdatetime\t00060_00003\t00060_00003_cd",
                "5s\t15s\t20d\t14n\t10s",
                "USGS\t02095000\t2020-01-01\t1250\tA",
            ]
        )

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return Response()

    monkeypatch.setattr("design_events.collect_sources.usgs_streamgages.requests.get", fake_get)

    records = fetch_nwis_discharge_records(
        {"streamflow_records": {"service": "dv", "stat_cd": "00003"}},
        "02095000",
        "2020-01-01",
        "2020-01-02",
    )

    assert records == [
        {
            "site_no": "02095000",
            "time": "2020-01-01T00:00:00",
            "discharge_cfs": 1250.0,
            "source": "usgs_dv",
        }
    ]
    assert calls[0][0].endswith("/nwis/dv/")
    assert calls[0][1]["sites"] == "02095000"
    assert calls[0][1]["parameterCd"] == "00060"
    assert calls[0][1]["statCd"] == "00003"
