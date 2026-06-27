"""Path and GeoJSON helpers shared by notebook-facing modules."""

from __future__ import annotations
import json
import os
from pathlib import Path
from shapely.geometry import mapping, shape

def find_repo_root(start=None):
    path = Path(start or Path.cwd()).resolve()
    if path.is_file():
        path = path.parent
    while not (path / "pyproject.toml").exists():
        if path == path.parent:
            raise FileNotFoundError("could not locate repo root")
        path = path.parent
    return path

def default_location_config_path(repo_root=None, environ=None):
    root = Path(repo_root or find_repo_root())
    env = os.environ if environ is None else environ
    if env.get("FLOOD_RM_LOCATION_CONFIG"):
        path = Path(env["FLOOD_RM_LOCATION_CONFIG"]).expanduser()
        return path if path.is_absolute() else (root / path).resolve()
    if env.get("FLOOD_RM_LOCATION"):
        location = env["FLOOD_RM_LOCATION"]
    else:
        location = Path.cwd().resolve().relative_to(root / "locations").parts[0]
    return root / "locations" / location / "config.yaml"

def resolve_repo_path(path, repo_root=None):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (Path(repo_root or find_repo_root()) / path).resolve()

def resolve_under_location(repo_root, location_name, value):
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / "locations" / location_name / path

def resolve_optional_under_root(repo_root, location_root, value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path

def read_geojson_geometry(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    item = data["features"][0] if data.get("type") == "FeatureCollection" else data
    return shape(item.get("geometry", item))

def write_geojson_features(path, features):
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "geometry": mapping(geometry), "properties": properties}
            for geometry, properties in features
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def write_geojson(path, geometry, properties):
    write_geojson_features(path, [(geometry, properties)])

def resolve_location_config_path(config_path):
    path = Path(config_path).expanduser()
    return path if path.is_absolute() else resolve_repo_path(path)