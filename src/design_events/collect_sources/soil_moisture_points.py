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

from __future__ import annotations

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
