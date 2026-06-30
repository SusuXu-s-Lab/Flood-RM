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

def repo_root_for_location(root, location_name=None, *, fallback="root"):
    root = Path(root)
    if root.parent.name == "locations" and (location_name is None or root.name == location_name):
        return root.parents[1]
    if fallback == "parent":
        return root.parent
    return root

def location_path(location_root, value, *, repo_root=None, location_name=None):
    root = Path(location_root)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    name = location_name or root.name
    if path.parts[:2] == ("locations", name):
        repo = Path(repo_root) if repo_root is not None else repo_root_for_location(root, name)
        return repo / path
    return root / path

def resolve_location_path(location_root, value, *, repo_root=None, location_name=None):
    """Resolve a location path and re-home serialized paths from another checkout."""
    root = Path(location_root)
    path = Path(value).expanduser()
    name = location_name or root.name
    if not path.is_absolute():
        return location_path(root, path, repo_root=repo_root, location_name=name)
    relocated = relocate_absolute_location_path(root, path, location_name=name)
    return relocated if relocated is not None else path

def relocate_absolute_location_path(location_root, path, *, location_name=None):
    """Map ``.../locations/<name>/...`` paths onto the active location root."""
    root = Path(location_root)
    path = Path(path)
    name = location_name or root.name
    parts = path.parts
    marker = ("locations", name)
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == marker:
            suffix = Path(*parts[index + 2 :])
            return root / suffix
    return None

def location_or_repo_path(location_root, value, *, repo_root=None, location_name=None):
    path = Path(value).expanduser()
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(location_root) / path
    repo = Path(repo_root) if repo_root is not None else repo_root_for_location(location_root, location_name)
    return repo / path

def location_root_from_paths(paths):
    """Return the Location Workspace root from a notebook/runtime ``paths`` dict."""
    if paths.get("location_root") is not None:
        return Path(paths["location_root"])
    repo_root = Path(paths.get("repo_root", Path.cwd()))
    location_name = paths.get("location_name")
    if location_name is None:
        raise ValueError("paths must include 'location_root' or 'location_name'")
    return repo_root / "locations" / str(location_name)

def location_or_repo_path_from_paths(paths, value):
    """Resolve a collection/static path from a notebook/runtime ``paths`` dict."""
    root = (
        paths["location_root"]
        if paths.get("location_root") is not None
        else location_root_from_paths(paths)
        if paths.get("location_name") is not None
        else paths.get("repo_root", Path.cwd())
    )
    return location_or_repo_path(
        root,
        value,
        repo_root=paths.get("repo_root", root),
        location_name=paths.get("location_name"),
    )

def configured_path(paths, value, *, base="location"):
    """Resolve a path from a notebook/runtime ``paths`` dict."""
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    location_root = Path(paths.get("location_root") or paths.get("repo_root") or Path.cwd())
    repo_root = Path(paths.get("repo_root") or location_root)
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid", "locations"}:
        return location_root / path
    return (repo_root if base == "repo" else location_root) / path

def relative_to_or_absolute(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path

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
