from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io import resolve_path
from .schema import RuntimePaths


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a location YAML file without mutating import state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def paths_from_config(config: dict[str, Any], *, location_root: str | Path) -> RuntimePaths:
    """Resolve the small set of paths needed by the runtime.

    All paths default to the layout already used by the previous SFINCS runner,
    but no notebook-specific path layer is imported.
    """
    root = Path(location_root).resolve()
    paths = config.get("paths") or {}
    wflow = config.get("wflow") or {}
    coupling = config.get("inland_coupling") or {}
    discharge = coupling.get("discharge_forcing") or {}

    return RuntimePaths(
        location_root=root,
        base_model_root=resolve_path(root, paths.get("base_model_root"), default="data/sfincs/base"),  # type: ignore[arg-type]
        scenarios_root=resolve_path(root, paths.get("scenarios_root"), default="data/sfincs/scenarios"),  # type: ignore[arg-type]
        storage_root=resolve_path(root, paths.get("storage_root"), default="data/sfincs/run_outputs"),  # type: ignore[arg-type]
        run_root=resolve_path(root, paths.get("run_root"), default="data/sfincs/run_stage"),  # type: ignore[arg-type]
        data_catalog=resolve_path(root, paths.get("data_catalog"), default="data/static/data_catalogue.yaml"),
        event_catalog=resolve_path(root, paths.get("event_catalog"), default="data/event_catalog/catalog/event_catalog.csv"),
        wflow_events_root=resolve_path(root, wflow.get("events_root"), default="data/wflow/events"),
        source_contract=resolve_path(
            root,
            discharge.get("source_contract") or discharge.get("handoff_locations"),
            default="data/sfincs/base/gis/wflow_handoff_sources.geojson",
        ),
    )


def native_source_config_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize config keys for :class:`NativeSourceConfig` construction."""
    return {
        "hydrography": data.get("hydrography", "merit_hydro"),
        "river_upa_km2": float(data.get("river_upa_km2", data.get("river_upa", 10.0))),
        "river_len_m": float(data.get("river_len_m", data.get("river_len", 1000.0))),
        "river_width_m": float(data.get("river_width_m", data.get("river_width", 0.0))),
        "buffer_m": float(data.get("river_inflow_buffer_m", data.get("buffer_m", data.get("buffer", 200.0)))),
        "first_index": int(data.get("first_index", 1)),
        "src_type": str(data.get("src_type", "inflow")),
        "keep_rivers_geom": bool(data.get("keep_rivers_geom", True)),
        "reverse_river_geom": bool(data.get("reverse_river_geom", False)),
        "max_source_points": None if data.get("max_source_points") in (None, "") else int(data["max_source_points"]),
    }
