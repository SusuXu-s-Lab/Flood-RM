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
