from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
from sfincs_runs.hydrology import condition_ksat_raster

def validate_infiltration_config(infiltration_cfg, *, event_drivers):
    """Validate that rain-on-grid runs have explicit infiltration inputs."""
    infiltration_cfg = infiltration_cfg or {}
    if not bool(infiltration_cfg.get("enabled", True)):
        return
    hydrologic_drivers = set(event_drivers or []) & {"rainfall", "soil_moisture"}
    if not hydrologic_drivers:
        return
    method = str(infiltration_cfg.get("method", "cn_with_recovery")).lower()
    if method == "cn_with_recovery":
        missing = [key for key in ("hsg", "ksat", "effective") if infiltration_cfg.get(key) in (None, "")]
        if missing:
            raise RuntimeError(
                "Rainfall/soil-moisture drivers require CN-with-recovery inputs. "
                f"Missing SFINCS hydrology infiltration keys: {missing}. "
                "Add HSG and Ksat rasters plus an event/soil-moisture-derived "
                "effective soil-retention fraction before running pluvial SFINCS. "
                "The SSURGO mapunit polygons currently fetched in 01_region_setup.ipynb "
                "are geometry only; they do not contain HSG/Ksat attributes."
            )
    elif method == "cn":
        if infiltration_cfg.get("cn") in (None, ""):
            raise RuntimeError(
                "CN infiltration requires SFINCS hydrology infiltration.cn "
                "pointing to a gridded Curve Number raster with cn/cn_avg variables."
            )
    elif method == "constant_lulc":
        if infiltration_cfg.get("qinf_reclass_table") in (None, ""):
            raise RuntimeError(
                "constant_lulc infiltration requires a qinf reclass table that maps "
                "impervious/developed LULC classes to near-zero infiltration."
            )
    else:
        raise ValueError(f"Unsupported infiltration method: {method}")

def setup_hydromt_infiltration(sf, config, paths, *, datadir=None):
    hydrology_cfg = _sfincs_hydrology_config(config)
    infiltration_cfg = hydrology_cfg.get("infiltration") or {}
    hydrology_enabled = bool(infiltration_cfg.get("enabled", True))
    hydrologic_drivers = (
        set(config.get("event_drivers") or []) & {"rainfall", "soil_moisture"} if hydrology_enabled else set()
    )
    if not hydrologic_drivers:
        return {
            "enabled": hydrology_enabled,
            "drivers": sorted(hydrologic_drivers),
            "method": str(infiltration_cfg.get("method", "cn_with_recovery")).lower(),
            "written": False,
        }

    validate_infiltration_config(infiltration_cfg, event_drivers=config.get("event_drivers") or [])
    method = str(infiltration_cfg.get("method", "cn_with_recovery")).lower()
    component = _select_hydromt_infiltration_component(sf)
    if component is None:
        raise RuntimeError("HydroMT-SFINCS model has no infiltration component")
    if method == "cn_with_recovery":
        reclass_table = infiltration_cfg.get("reclass_table")
        if reclass_table in (None, "") and datadir is not None:
            reclass_table = str(Path(datadir) / "lulc" / "esa_worldcover_HSG.csv")
        ksat_source = infiltration_cfg["ksat"]
        ksat_conditioning = None
        if infiltration_cfg.get("ksat_effective") not in (None, ""):
            ksat_conditioning = condition_ksat_raster(
                _resolve_location_path(paths, infiltration_cfg["ksat"]),
                _resolve_location_path(paths, infiltration_cfg["ksat_effective"]),
                scale_factor=float(infiltration_cfg.get("ksat_scale_factor", 1.0)),
                max_mmhr=(
                    None
                    if infiltration_cfg.get("ksat_max_mmhr") in (None, "")
                    else float(infiltration_cfg.get("ksat_max_mmhr"))
                ),
            )
            ksat_source = ksat_conditioning["ksat"]
        factor_ksat = infiltration_cfg.get("factor_ksat")
        if factor_ksat in (None, ""):
            # HydroMT-SFINCS' quadtree CN-recovery path multiplies Ksat by 3.6
            # internally. A conditioned raster from this module is already mm/hr.
            factor_ksat = 1.0 / 3.6 if ksat_conditioning else 1.0
        component.create_cn_with_recovery(
            lulc=_register_raster_source_or_name(sf, paths, "lulc", infiltration_cfg.get("lulc", "worldcover")),
            hsg=_register_raster_source(sf, paths, "hsg", infiltration_cfg["hsg"]),
            ksat=_register_raster_source(sf, paths, "ksat", ksat_source),
            reclass_table=reclass_table,
            effective=float(infiltration_cfg["effective"]),
            factor_ksat=float(factor_ksat),
            block_size=int(infiltration_cfg.get("block_size", 2000)),
        )
    elif method == "cn":
        component.create_cn(
            cn=_register_raster_source(sf, paths, "curve_number", infiltration_cfg["cn"]),
            antecedent_moisture=infiltration_cfg.get("antecedent_moisture", "avg"),
        )
    elif method == "constant_lulc":
        component.create_constant(
            lulc=infiltration_cfg.get("lulc", "worldcover"),
            reclass_table=str(_resolve_location_path(paths, infiltration_cfg["qinf_reclass_table"])),
        )
    else:
        raise ValueError(f"Unsupported infiltration method: {method}")

    return {
        "enabled": hydrology_enabled,
        "drivers": sorted(hydrologic_drivers),
        "method": method,
        "written": True,
        "ksat_conditioning": ksat_conditioning if method == "cn_with_recovery" else None,
    }

def validate_physics(model_root, config=None, *, require_spatial_roughness=True):
    """Validate built SFINCS static physics expected by rain-on-grid workflows."""
    model_root = Path(model_root)
    inp = _read_sfincs_inp(model_root / "sfincs.inp")
    drivers = set((config or {}).get("event_drivers") or [])
    hydrology_cfg = _sfincs_hydrology_config(config or {})
    infiltration_cfg = hydrology_cfg.get("infiltration") or {}
    require_infiltration = bool(infiltration_cfg.get("enabled", True)) and bool(drivers & {"rainfall", "soil_moisture"})
    infiltration = _validate_sfincs_infiltration_files(
        model_root,
        inp,
        method=str(infiltration_cfg.get("method", "cn_with_recovery")).lower(),
        required=require_infiltration,
    )
    roughness = (
        _validate_sfincs_spatial_roughness(model_root, inp)
        if require_spatial_roughness
        else {"required": False, "spatially_varying": None}
    )
    return {"model_root": str(model_root), "infiltration": infiltration, "roughness": roughness}

def _validate_sfincs_infiltration_files(model_root, inp, *, method, required):
    if not required:
        return {"required": False, "status": "not_required"}
    if method == "cn_with_recovery":
        keys = ("smaxfile", "sefffile", "ksfile")
    elif method == "cn":
        keys = ("scsfile",)
    else:
        keys = ("qinffile", "scsfile", "smaxfile", "sefffile", "ksfile")
    present = [key for key in keys if inp.get(key)]
    missing = [key for key in keys if not inp.get(key)] if method in {"cn_with_recovery", "cn"} else []
    missing_files = [inp[key] for key in present if not (model_root / str(inp[key])).exists()]
    qinf = str(inp.get("qinf", "")).strip()
    qinf_inactive = method not in {"cn_with_recovery", "cn"} and qinf == "0"
    if missing or missing_files or qinf_inactive:
        raise RuntimeError(
            "SFINCS rain-on-grid model lacks active native infiltration. "
            f"method={method!r}, missing_keys={missing}, missing_files={missing_files}, qinf={qinf!r}. "
            "Rebuild the SFINCS base with HydroMT-SFINCS infiltration.create_*."
        )
    return {"required": True, "method": method, "keys": present, "status": "active"}

def _validate_sfincs_spatial_roughness(model_root, inp):
    subgrid_file = inp.get("sbgfile") or inp.get("subgridfile") or "sfincs_subgrid.nc"
    path = model_root / str(subgrid_file)
    if not path.exists():
        raise RuntimeError(f"SFINCS spatial roughness requires a subgrid file with Manning values: {path}")
    with xr.open_dataset(path) as ds:
        candidates = [name for name in ("uv_navg", "uv_nrep", "uv_n") if name in ds]
        if not candidates:
            raise RuntimeError(f"SFINCS subgrid file has no roughness variables: {path}")
        stats = {}
        varying = False
        for name in candidates:
            values = np.asarray(ds[name].values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            stats[name] = {"min": vmin, "max": vmax}
            varying = varying or not np.isclose(vmin, vmax)
    if not varying:
        raise RuntimeError(f"SFINCS roughness is not spatially varying in {path}; check subgrid roughness_list/LULC mapping.")
    return {
        "required": True,
        "subgrid_file": str(path),
        "roughness_variables": sorted(stats),
        "spatially_varying": True,
        "stats": stats,
    }

def _read_sfincs_inp(path):
    if not path.exists():
        raise FileNotFoundError(path)
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        values[key.strip().lower()] = value.strip().split()[0]
    return values

def _sfincs_hydrology_config(config):
    coastal = (config.get("coastal_wave_coupling") or {}).get("hydrology") or {}
    if coastal:
        return coastal
    inland = config.get("inland_coupling") or {}
    if not inland:
        return {}
    hydrology = {}
    if "direct_rainfall" in inland:
        hydrology["precipitation"] = inland.get("direct_rainfall") or {}
    if "infiltration" in inland:
        hydrology["infiltration"] = inland.get("infiltration") or {}
    if "soil_moisture" in inland:
        hydrology["soil_moisture"] = inland.get("soil_moisture") or {}
    return hydrology

def _select_hydromt_infiltration_component(sf):
    if str(getattr(sf, "grid_type", "") or "").lower() == "quadtree":
        # Quadtree CN recovery must write onto sf.quadtree_grid; the regular
        # component dereferences sf.grid.nmax/mmax and fails before write.
        return getattr(sf, "quadtree_infiltration", None)
    return getattr(sf, "infiltration", None) or getattr(sf, "quadtree_infiltration", None)

def _resolve_location_path(paths, value):
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(paths["location_root"]) / path

def _register_raster_source(sf, paths, name, value):
    path = _resolve_location_path(paths, value)
    if path is None:
        return name
    if not path.exists():
        raise FileNotFoundError(f"{name} raster not found: {path}")
    sf.data_catalog.from_dict({name: {"uri": str(path), "data_type": "RasterDataset", "driver": {"name": "rasterio"}}})
    return name

def _register_raster_source_or_name(sf, paths, name, value):
    if value in (None, ""):
        return name
    path = Path(str(value))
    is_path = path.is_absolute() or len(path.parts) > 1 or bool(path.suffix)
    if not is_path:
        return str(value)
    return _register_raster_source(sf, paths, name, value)
