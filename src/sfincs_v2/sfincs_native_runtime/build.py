from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import register_raster_or_dataset


def _sfincs_model_class():
    from hydromt_sfincs import SfincsModel

    return SfincsModel


def open_model(root: str | Path, *, mode: str = "r+", read: bool = True, write_gis: bool = True, data_libs=None):
    """Open a HydroMT-SFINCS model using the native model class."""
    SfincsModel = _sfincs_model_class()
    sf = SfincsModel(root=str(root), mode=mode, write_gis=write_gis, data_libs=data_libs)
    if read:
        sf.read()
    return sf


def build_from_steps(
    root: str | Path,
    *,
    steps: list[dict[str, dict[str, Any]]],
    data_libs: list[str | Path] | str | Path | None = None,
    mode: str = "w+",
    write: bool = True,
):
    """Build a SFINCS base model from native HydroMT-SFINCS build steps.

    ``steps`` is the list accepted by ``SfincsModel.build``: each item is a
    mapping from ``component.method`` to keyword arguments.  This deliberately
    avoids project-specific wrappers around grid, elevation, mask, roughness, and
    subgrid construction.
    """
    sf = open_model(root, mode=mode, read=False, data_libs=data_libs)
    sf.build(steps=steps, write=write)
    return sf


def update_from_steps(
    root: str | Path,
    *,
    steps: list[dict[str, dict[str, Any]]],
    model_out: str | Path | None = None,
    data_libs: list[str | Path] | str | Path | None = None,
    write: bool = True,
):
    """Update an existing model through HydroMT-SFINCS native ``update``."""
    sf = open_model(root, mode="r+", read=True, data_libs=data_libs)
    kwargs: dict[str, Any] = {"steps": steps, "write": write}
    if model_out is not None:
        kwargs["model_out"] = str(model_out)
    sf.update(**kwargs)
    return sf


def _select_infiltration_component(sf):
    if str(getattr(sf, "grid_type", "") or "").lower() == "quadtree":
        return getattr(sf, "quadtree_infiltration", None)
    return getattr(sf, "infiltration", None) or getattr(sf, "quadtree_infiltration", None)


def apply_native_infiltration(sf, cfg: dict[str, Any], *, paths: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply native HydroMT-SFINCS infiltration components.

    Supported methods map one-to-one to HydroMT-SFINCS component methods:
    ``cn_with_recovery``, ``cn``, and ``constant_lulc``.
    """
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "written": False}

    method = str(cfg.get("method", "cn_with_recovery")).lower()
    component = _select_infiltration_component(sf)
    if component is None:
        raise RuntimeError("HydroMT-SFINCS model has no infiltration component")

    def source(name: str, value, *, crs=None) -> str:
        if value in (None, ""):
            return name
        path = Path(value)
        if paths is not None and not path.is_absolute() and len(path.parts) > 1:
            path = Path(paths["location_root"]) / path
        if path.exists():
            return register_raster_or_dataset(sf, name, path, crs=crs)
        return str(value)

    if method == "cn_with_recovery":
        required = [key for key in ("lulc", "hsg", "ksat", "reclass_table", "effective") if cfg.get(key) in (None, "")]
        if required:
            raise ValueError(f"cn_with_recovery infiltration missing required keys: {required}")
        component.create_cn_with_recovery(
            lulc=source("lulc", cfg["lulc"]),
            hsg=source("hsg", cfg["hsg"]),
            ksat=source("ksat", cfg["ksat"]),
            reclass_table=str(cfg["reclass_table"]),
            effective=float(cfg["effective"]),
            factor_ksat=float(cfg.get("factor_ksat", 1.0)),
            block_size=int(cfg.get("block_size", 2000)),
        )
    elif method == "cn":
        if cfg.get("cn") in (None, ""):
            raise ValueError("cn infiltration requires cfg['cn']")
        component.create_cn(
            cn=source("curve_number", cfg["cn"]),
            antecedent_moisture=cfg.get("antecedent_moisture", "avg"),
        )
    elif method in {"constant", "constant_lulc"}:
        if cfg.get("qinf_reclass_table") in (None, ""):
            raise ValueError("constant_lulc infiltration requires cfg['qinf_reclass_table']")
        component.create_constant(
            lulc=source("lulc", cfg.get("lulc", "worldcover")),
            reclass_table=str(cfg["qinf_reclass_table"]),
        )
    else:
        raise ValueError(f"Unsupported infiltration method: {method}")

    return {"enabled": True, "method": method, "written": True}


def apply_native_structures(sf, layers: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply weirs, thin dams, and drainage structures through native components."""
    rows: list[dict[str, Any]] = []
    for layer in layers or []:
        component = str(layer.get("component", "")).strip().lower()
        locations = layer.get("locations") or layer.get("path")
        if locations in (None, ""):
            raise ValueError(f"structure layer missing locations/path: {layer}")
        merge = bool(layer.get("merge", True))
        if component == "weirs":
            kwargs = {"locations": locations, "merge": merge}
            for key in ("elevation", "par1", "dep", "buffer", "dz"):
                if key in layer and layer[key] is not None:
                    kwargs[key] = layer[key]
            sf.weirs.create(**kwargs)
        elif component == "thin_dams":
            sf.thin_dams.create(locations=locations, merge=merge)
        elif component == "drainage_structures":
            kwargs = {"locations": locations, "merge": merge}
            if layer.get("stype") not in (None, ""):
                kwargs["stype"] = layer["stype"]
            sf.drainage_structures.create(**kwargs)
        else:
            raise ValueError(f"Unsupported native SFINCS structure component: {component}")
        rows.append({"component": component, "locations": str(locations), "merge": merge, "status": "applied"})
    return pd.DataFrame(rows)
