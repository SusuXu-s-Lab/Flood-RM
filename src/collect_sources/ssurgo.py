"""Fetch SSURGO map unit polygons from USDA NRCS Soil Data Access."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from shapely.ops import transform


soil_wfs_url = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs"
soil_tabular_url = "https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest"


def _ranges_overlap(first_min, first_max, second_min, second_max):
    return max(first_min, second_min) <= min(first_max, second_max)


def _bounds_overlap(bounds, bbox_wgs84):
    xmin, ymin, xmax, ymax = bounds
    west, south, east, north = bbox_wgs84
    return _ranges_overlap(xmin, xmax, west, east) and _ranges_overlap(ymin, ymax, south, north)


def _swapped_bounds_overlap(bounds, bbox_wgs84):
    xmin, ymin, xmax, ymax = bounds
    west, south, east, north = bbox_wgs84
    return _ranges_overlap(xmin, xmax, south, north) and _ranges_overlap(ymin, ymax, west, east)


def _swap_xy(geometry):
    return transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geometry)


def normalize_ssurgo_axis_order(soils, bbox_wgs84):
    """Repair WFS 1.1 EPSG:4326 latitude/longitude axis order when needed."""
    if soils.empty:
        return soils
    if _bounds_overlap(soils.total_bounds, bbox_wgs84):
        return soils
    if not _swapped_bounds_overlap(soils.total_bounds, bbox_wgs84):
        return soils
    fixed = soils.copy()
    fixed["geometry"] = fixed.geometry.map(_swap_xy)
    return fixed.set_crs("EPSG:4326", allow_override=True)


def build_ssurgo_wfs_url(
    bbox_wgs84,
    *,
    layer="mapunitpoly",
    version="1.1.0",
    output_format="GML2",
    max_features=None,
):
    west, south, east, north = bbox_wgs84
    params = {
        "service": "WFS",
        "version": version,
        "request": "GetFeature",
        "typename": layer,
        "srsname": "EPSG:4326",
        "outputformat": output_format,
        "bbox": ",".join(str(value) for value in (west, south, east, north)),
    }
    if max_features is not None:
        params["maxfeatures"] = str(int(max_features))
    return f"{soil_wfs_url}?{urlencode(params)}"


def fetch_ssurgo_mapunit_polygons(
    bbox_wgs84,
    output_path,
    *,
    timeout_seconds=120,
    keep_gml=False,
    session_get=None,
    read_file=None,
    fallback_tile_degrees=0.5,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = build_ssurgo_wfs_url(bbox_wgs84)
    get = session_get or requests.get
    response = get(url, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code != 400 or not _bbox_needs_tiling(bbox_wgs84, fallback_tile_degrees):
            raise
        return _fetch_ssurgo_mapunit_polygon_tiles(
            bbox_wgs84,
            output_path,
            timeout_seconds=timeout_seconds,
            keep_gml=keep_gml,
            session_get=get,
            read_file=read_file,
            tile_degrees=fallback_tile_degrees,
        )

    gml_path = output_path.with_suffix(".gml")
    gml_path.write_bytes(response.content)
    reader = read_file or gpd.read_file
    soils = reader(gml_path)
    if soils.crs is None:
        soils = soils.set_crs("EPSG:4326")
    else:
        soils = soils.to_crs("EPSG:4326")
    soils = normalize_ssurgo_axis_order(soils, bbox_wgs84)
    soils.to_file(output_path, driver="GPKG")
    if not keep_gml:
        gml_path.unlink(missing_ok=True)
    return soils


def _bbox_needs_tiling(bbox_wgs84, tile_degrees):
    west, south, east, north = bbox_wgs84
    return east - west > tile_degrees or north - south > tile_degrees


def _split_bbox(bbox_wgs84, tile_degrees):
    west, south, east, north = bbox_wgs84
    y = south
    while y < north:
        tile_north = min(y + tile_degrees, north)
        x = west
        while x < east:
            tile_east = min(x + tile_degrees, east)
            yield (x, y, tile_east, tile_north)
            x = tile_east
        y = tile_north


def _fetch_ssurgo_mapunit_polygon_tiles(
    bbox_wgs84,
    output_path,
    *,
    timeout_seconds,
    keep_gml,
    session_get,
    read_file,
    tile_degrees,
):
    reader = read_file or gpd.read_file
    frames = []
    for index, tile_bbox in enumerate(_split_bbox(bbox_wgs84, tile_degrees)):
        response = session_get(build_ssurgo_wfs_url(tile_bbox), timeout=timeout_seconds)
        response.raise_for_status()
        gml_path = output_path.with_name(f"{output_path.stem}_tile_{index:03d}.gml")
        gml_path.write_bytes(response.content)
        tile = reader(gml_path)
        if tile.crs is None:
            tile = tile.set_crs("EPSG:4326")
        else:
            tile = tile.to_crs("EPSG:4326")
        tile = normalize_ssurgo_axis_order(tile, tile_bbox)
        if not tile.empty:
            frames.append(tile)
        if not keep_gml:
            gml_path.unlink(missing_ok=True)

    soils = (
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        if frames
        else gpd.GeoDataFrame(columns=["mukey", "geometry"], geometry="geometry", crs="EPSG:4326")
    )
    soils = normalize_ssurgo_axis_order(soils, bbox_wgs84)
    soils.to_file(output_path, driver="GPKG")
    return soils


def fetch_ssurgo_mapunit_attributes(
    mukeys,
    output_path,
    *,
    timeout_seconds=120,
    session_post=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(value).strip() for value in mukeys if str(value).strip()})
    if not keys:
        frame = pd.DataFrame(columns=ssurgo_attribute_columns())
        frame.to_csv(output_path, index=False)
        return frame

    response = (session_post or requests.post)(
        soil_tabular_url,
        json={"query": build_ssurgo_attribute_query(keys), "format": "JSON+COLUMNNAME"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    frame = _soil_data_access_table(response.json())
    frame.to_csv(output_path, index=False)
    return frame


def build_ssurgo_attribute_query(mukeys):
    quoted_keys = []
    for key in mukeys:
        clean = str(key).replace("'", "''")
        quoted_keys.append(f"'{clean}'")
    key_list = ", ".join(quoted_keys)
    return f"""
SELECT
  mu.mukey,
  co.hydgrp,
  ch.ksat_r,
  ch.hzdept_r,
  ch.hzdepb_r,
  ch.sandtotal_r,
  ch.silttotal_r,
  ch.claytotal_r,
  ch.dbthirdbar_r,
  ch.om_r,
  ch.ph1to1h2o_r
FROM mapunit AS mu
INNER JOIN component AS co ON mu.mukey = co.mukey
LEFT JOIN chorizon AS ch ON co.cokey = ch.cokey
WHERE mu.mukey IN ({key_list})
  AND co.majcompflag = 'Yes'
  AND (ch.hzdept_r IS NULL OR ch.hzdept_r <= 200)
ORDER BY mu.mukey, co.comppct_r DESC, ch.hzdept_r
""".strip()


def ssurgo_attribute_columns():
    return [
        "mukey",
        "hydgrp",
        "ksat_r",
        "hzdept_r",
        "hzdepb_r",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "dbthirdbar_r",
        "om_r",
        "ph1to1h2o_r",
    ]


def _soil_data_access_table(payload):
    rows = payload.get("Table", [])
    if not rows:
        return pd.DataFrame(columns=ssurgo_attribute_columns())
    columns, values = rows[0], rows[1:]
    return pd.DataFrame(values, columns=columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch SSURGO map unit polygons.")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--keep-gml", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    soils = fetch_ssurgo_mapunit_polygons(
        bbox_wgs84=tuple(args.bbox),
        output_path=args.out,
        timeout_seconds=args.timeout_seconds,
        keep_gml=args.keep_gml,
    )
    print(f"Downloaded {len(soils)} SSURGO map unit polygons")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


# NWM soil-moisture sampling points derived from local soil/footprint evidence
"""Derive NWM soil-moisture sampling points from a location footprint.

Representative NWM cells are never hand-typed in the location YAML. Instead they
are *derived* from the location's footprint geometry the same way the AORC
transposition region is (see ``prerequisites.prepare_aorc_transposition_region``):
sample the footprint centroid, its bounding-box corners, and its edge midpoints,
then let the collector snap each to the nearest NWM grid cell. The derived points
are written to a stakeholder-facing, review-required GeoJSON so a reviewer can
open them in any GIS and confirm them against the Wflow-SFINCS domain set.

Both the collection prerequisite step and the collector itself call into here, so
the derivation lives in exactly one place.
"""

from pathlib import Path

# GeoJSON is WGS84 by the RFC 7946 default, so the artifact opens correctly in any
# GIS while the collector reprojects to the NWM grid CRS for cell selection.
GEOJSON_CRS = "EPSG:4326"

# Footprint candidates, most specific first. Mirrors the AORC transposition-region
# source resolution so soil-moisture sampling tracks the same evaluation AOI; the
# coastal study area is the fallback for locations without an evaluation footprint.
FOOTPRINT_CANDIDATES = (
    "data/static/aoi/evaluation_footprint.geojson",
    "data/static/aoi/study_area.geojson",
)

DEFAULT_POINTS_FILE = "data/static/aoi/nwm_soil_moisture_points.geojson"

_REVIEW_NOTES = (
    "Derived from the location footprint (centroid, bounding-box corners, edge "
    "midpoints) and snapped to the nearest NWM cell at collection time; review "
    "against the Wflow-SFINCS domain set before production use."
)


def _location_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(root) / path
    return Path(paths.get("repo_root", root)) / path


def _footprint_path(spec, paths):
    candidates = [spec.get("points_source"), *FOOTPRINT_CANDIDATES]
    for value in candidates:
        if not value:
            continue
        path = _location_path(paths, value)
        if path.exists():
            return path
    return None


def _output_path(spec, paths):
    configured = paths.get("nwm_soil_moisture_points_geojson")
    if configured:
        return Path(configured)
    return _location_path(paths, spec.get("points_file", DEFAULT_POINTS_FILE))


def has_footprint(spec, paths):
    """True when a footprint geometry is available to derive points from."""
    return _footprint_path(spec, paths) is not None


def _sample_points(geometry):
    """Footprint centroid, four bbox corners, and four edge midpoints, in CRS units."""
    minx, miny, maxx, maxy = geometry.bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    centroid = geometry.centroid
    return [
        ("center", centroid.x, centroid.y),
        ("southwest", minx, miny),
        ("southeast", maxx, miny),
        ("northwest", minx, maxy),
        ("northeast", maxx, maxy),
        ("south_mid", midx, miny),
        ("north_mid", midx, maxy),
        ("west_mid", minx, midy),
        ("east_mid", maxx, midy),
    ]


def build_points_geodataframe(spec, paths):
    """Build the derived sampling points as a WGS84 GeoDataFrame plus its source path."""
    import geopandas as gpd
    from shapely.geometry import Point

    crs = spec.get("crs")
    if not crs:
        raise ValueError(
            "collection.nwm.soil_moisture.crs is required to snap sampling points to NWM cells"
        )
    footprint_path = _footprint_path(spec, paths)
    if footprint_path is None:
        raise FileNotFoundError(
            "could not find a footprint geometry for NWM soil-moisture sampling; set "
            "collection.nwm.soil_moisture.points_source to a footprint GeoJSON"
        )
    footprint = gpd.read_file(footprint_path)
    if footprint.empty:
        raise ValueError(f"NWM soil-moisture footprint geometry is empty: {footprint_path}")
    if footprint.crs is None:
        footprint = footprint.set_crs(GEOJSON_CRS)

    geometry = footprint.to_crs(crs).geometry.union_all()
    samples = _sample_points(geometry)
    count = len(samples)
    points = gpd.GeoDataFrame(
        {
            "id": [role for role, _, _ in samples],
            "role": [role for role, _, _ in samples],
            "x_nwm": [round(x, 4) for _, x, _ in samples],
            "y_nwm": [round(y, 4) for _, _, y in samples],
            "source_geometry": [str(footprint_path)] * count,
            "review_status": ["review_required"] * count,
            "review_notes": [_REVIEW_NOTES] * count,
            "geometry": [Point(x, y) for _, x, y in samples],
        },
        crs=crs,
    ).to_crs(GEOJSON_CRS)
    return points, footprint_path


def ensure_points_geojson(spec, paths, *, overwrite=False):
    """Write the derived sampling-point GeoJSON if absent; return a status dict."""
    import geopandas as gpd

    output_path = _output_path(spec, paths)
    if output_path.exists() and not overwrite:
        return {
            "path": output_path,
            "status": "reused",
            "source_geometry": None,
            "point_count": int(len(gpd.read_file(output_path))),
        }
    points, footprint_path = build_points_geodataframe(spec, paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_file(output_path, driver="GeoJSON")
    return {
        "path": output_path,
        "status": "created_review_required",
        "source_geometry": footprint_path,
        "point_count": int(len(points)),
    }


def load_points(spec, paths):
    """Resolve the soil-moisture sampling points as ``[{id, x, y}, ...]``.

    Explicit ``points`` in the spec win (back-compatible). Otherwise the points are
    derived from the footprint, generating the review-required GeoJSON on demand so
    the collector does not depend on a separate prerequisite step having run first.
    Locations with neither explicit points nor a footprint resolve to ``[]``.
    """
    import geopandas as gpd

    explicit = spec.get("points") or []
    if explicit:
        return [dict(point) for point in explicit]
    if not has_footprint(spec, paths):
        return []

    ensure_points_geojson(spec, paths)
    crs = spec.get("crs")
    x_name = spec.get("x", "x")
    y_name = spec.get("y", "y")
    projected = gpd.read_file(_output_path(spec, paths)).to_crs(crs)
    return [
        {"id": str(point_id), x_name: float(x), y_name: float(y)}
        for point_id, x, y in zip(projected["id"], projected.geometry.x, projected.geometry.y)
    ]
