from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
from shapely.geometry import Polygon

from design_events.collect_sources.fetch_ssurgo import (
    build_ssurgo_wfs_url,
    fetch_ssurgo_mapunit_attributes,
    fetch_ssurgo_mapunit_polygons,
    normalize_ssurgo_axis_order,
)


class FakeResponse:
    def __init__(self, content=b"<gml></gml>", payload=None):
        self.content = content
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_build_ssurgo_wfs_url_uses_soil_data_access_bbox():
    url = build_ssurgo_wfs_url(
        (-76.65, 38.95, -76.55, 39.05),
        max_features=25,
    )
    query = parse_qs(urlparse(url).query)

    assert query["service"] == ["WFS"]
    assert query["version"] == ["1.1.0"]
    assert query["request"] == ["GetFeature"]
    assert query["typename"] == ["mapunitpoly"]
    assert query["srsname"] == ["EPSG:4326"]
    assert query["outputformat"] == ["GML2"]
    assert query["bbox"] == ["-76.65,38.95,-76.55,39.05"]
    assert query["maxfeatures"] == ["25"]


def test_fetch_ssurgo_mapunit_polygons_writes_geopackage(tmp_path):
    calls = []

    def fake_get(url, timeout):
        calls.append({"url": url, "timeout": timeout})
        return FakeResponse()

    def fake_read_file(path):
        assert Path(path).suffix == ".gml"
        return gpd.GeoDataFrame(
            {"mukey": ["1"]},
            geometry=[
                Polygon(
                    [
                        (-76.65, 38.95),
                        (-76.55, 38.95),
                        (-76.55, 39.05),
                        (-76.65, 39.05),
                    ]
                )
            ],
        )

    output_path = tmp_path / "ssurgo_mapunitpoly.gpkg"

    soils = fetch_ssurgo_mapunit_polygons(
        bbox_wgs84=(-76.65, 38.95, -76.55, 39.05),
        output_path=output_path,
        timeout_seconds=30,
        session_get=fake_get,
        read_file=fake_read_file,
    )

    assert calls[0]["timeout"] == 30
    assert output_path.exists()
    assert not output_path.with_suffix(".gml").exists()
    assert soils.crs.to_epsg() == 4326

    written = gpd.read_file(output_path)
    assert list(written["mukey"]) == ["1"]


def test_normalize_ssurgo_axis_order_repairs_wfs_lat_lon_coordinates():
    soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[
            Polygon(
                [
                    (38.95, -76.65),
                    (38.95, -76.55),
                    (39.05, -76.55),
                    (39.05, -76.65),
                ]
            )
        ],
        crs="EPSG:4326",
    )

    fixed = normalize_ssurgo_axis_order(soils, (-76.65, 38.95, -76.55, 39.05))

    assert fixed.crs.to_epsg() == 4326
    assert tuple(round(value, 2) for value in fixed.total_bounds) == (-76.65, 38.95, -76.55, 39.05)


def test_fetch_ssurgo_mapunit_attributes_writes_hsg_and_ksat_table(tmp_path):
    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(
            payload={
                "Table": [
                    ["mukey", "hydgrp", "ksat_r", "hzdept_r", "hzdepb_r"],
                    ["1001", "B", 12.5, 0, 20],
                    ["1002", "C/D", 2.0, 0, 15],
                ]
            }
        )

    out = tmp_path / "ssurgo_mapunit_attributes.csv"

    attrs = fetch_ssurgo_mapunit_attributes(
        ["1001", "1002"],
        out,
        session_post=fake_post,
        timeout_seconds=45,
    )

    assert calls[0]["timeout"] == 45
    assert "component" in calls[0]["json"]["query"].lower()
    assert "chorizon" in calls[0]["json"]["query"].lower()
    assert out.exists()
    assert list(attrs.columns) == ["mukey", "hydgrp", "ksat_r", "hzdept_r", "hzdepb_r"]
    assert attrs["hydgrp"].tolist() == ["B", "C/D"]
