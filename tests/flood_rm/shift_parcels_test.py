import geopandas as gpd
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import box

from power.source_inputs import fetch_building_parcels_in_geometry


def test_fetch_building_parcels_preserves_lon_lat_polygon_order(monkeypatch):
    calls = []

    def fake_features_from_polygon(polygon, tags):
        calls.append((polygon, tags))
        west, south, east, north = polygon.bounds
        assert west < -70
        assert east < -70
        assert south > 42
        assert north > 42
        return gpd.GeoDataFrame(
            {"building": ["yes"]},
            geometry=[Point(-70.70, 42.10)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr("power.source_inputs.ox.features_from_polygon", fake_features_from_polygon)

    parcels = fetch_building_parcels_in_geometry(box(-70.72, 42.08, -70.68, 42.12))

    assert len(calls) == 1
    assert len(parcels) == 1
    assert parcels[0].geometry.longitude == -70.70
    assert parcels[0].geometry.latitude == 42.10


def test_fetch_building_parcels_skips_empty_multipolygon_parts(monkeypatch):
    responses = iter(
        [
            InsufficientResponseError("No matching features. Check query location, tags, and log."),
            gpd.GeoDataFrame(
                {"building": ["yes"]},
                geometry=[Point(-70.66, 42.13)],
                crs="EPSG:4326",
            ),
        ]
    )

    def fake_features_from_polygon(polygon, tags):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("power.source_inputs.ox.features_from_polygon", fake_features_from_polygon)
    geometry = MultiPolygon(
        [
            box(-70.72, 42.08, -70.70, 42.10),
            box(-70.67, 42.12, -70.65, 42.14),
        ]
    )

    parcels = fetch_building_parcels_in_geometry(geometry)

    assert len(parcels) == 1
    assert parcels[0].geometry.longitude == -70.66
    assert parcels[0].geometry.latitude == 42.13
