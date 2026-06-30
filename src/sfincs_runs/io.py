from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import pandas as pd

MUTABLE_EVENT_FILES = frozenset(
    {
        "sfincs.inp",
        "sfincs.src",
        "sfincs.dis",
        "sfincs.bnd",
        "sfincs.bzs",
        "sfincs.ini",
        "sfincs.seff",
        "sfincs.precip",
        "sfincs_netampr.nc",
        "aorc_precip_for_sfincs.nc",
        "forcing_manifest.json",
        "run_metadata.json",
        "snapwave.bhs",
        "snapwave.btp",
        "snapwave.bwd",
        "snapwave.bds",
    }
)

STALE_SOLVER_OUTPUTS = frozenset(
    {
        "sfincs_map.nc",
        "sfincs_his.nc",
        "sfincs_rst.nc",
        "sfincs.nc",
        "sfincs.log",
        "sfincs_log.txt",
        "sfincs.out",
    }
)

RETAINED_OUTPUT_FILES = frozenset(
    {
        "sfincs_map.nc",
        "sfincs_his.nc",
        "sfincs.nc",
        "sfincs.log",
        "sfincs_log.txt",
        "sfincs.inp",
        "forcing_manifest.json",
        "run_metadata.json",
        "sfincs.bnd",
        "sfincs.bzs",
        "sfincs.src",
        "sfincs.dis",
        "sfincs.obs",
        "sfincs.weir",
        "sfincs.thd",
        "sfincs.drn",
        "sfincs.rug",
        "snapwave.bnd",
        "snapwave.bhs",
        "snapwave.btp",
        "snapwave.bwd",
        "snapwave.bds",
    }
)


def path_or_none(value: str | os.PathLike[str] | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)  # type: ignore[arg-type]


def resolve_path(root: Path, value: str | os.PathLike[str] | None, *, default: str | None = None) -> Path | None:
    raw = value if value not in (None, "") else default
    if raw in (None, ""):
        return None
    path = Path(raw)  # type: ignore[arg-type]
    return path if path.is_absolute() else Path(root) / path


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_sfincs_inp(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read ``sfincs.inp`` into lowercase key/value strings."""
    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().split()[0] if value.strip() else ""
    return values


def model_root_path(model: Any) -> Path:
    raw_root = getattr(model, "root", None)
    if hasattr(raw_root, "path"):
        raw_root = raw_root.path
    return Path(raw_root or ".")


def copy_base_model(base_root: Path, run_root: Path, *, force: bool = False) -> Path:
    """Create one mutable event folder from a static SFINCS base model.

    Immutable inputs are hard-linked when the filesystem allows it.  Mutable event
    forcing files are copied.  Previous solver outputs are removed.
    """
    base_root = Path(base_root)
    run_root = Path(run_root)
    if not base_root.exists():
        raise FileNotFoundError(base_root)
    if run_root.exists():
        if not force:
            raise FileExistsError(f"{run_root} exists; pass force=True to replace it")
        shutil.rmtree(run_root)

    def copy_function(src: str, dst: str) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.name in MUTABLE_EVENT_FILES:
            shutil.copy2(src_path, dst_path)
            return
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    run_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_root, run_root, copy_function=copy_function)
    remove_solver_outputs(run_root)
    return run_root


def remove_solver_outputs(root: Path, extra: Iterable[str] = ()) -> None:
    for name in set(STALE_SOLVER_OUTPUTS) | set(extra):
        (Path(root) / name).unlink(missing_ok=True)


def copy_retained_outputs(
    source_dir: Path,
    storage_dir: Path,
    *,
    overwrite: bool = True,
    retained_files: Iterable[str] | None = None,
) -> list[str]:
    copied: list[str] = []
    source_dir = Path(source_dir)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(RETAINED_OUTPUT_FILES if retained_files is None else retained_files):
        source = source_dir / name
        target = storage_dir / name
        if not source.exists() or (target.exists() and not overwrite):
            continue
        shutil.copy2(source, target)
        copied.append(name)
    return copied


def config_set(sf: Any, key: str, value: Any) -> None:
    """Set a SFINCS config key through the native config component."""
    config = sf.config
    setter = getattr(config, "set", None)
    if callable(setter):
        setter(key, value)
        return
    data = getattr(config, "data", None)
    if isinstance(data, dict):
        data[key] = value
        return
    try:
        config[key] = value
    except Exception as exc:  # pragma: no cover - HydroMT-version guard
        raise TypeError(f"Cannot set SFINCS config key {key!r}") from exc


def config_update(sf: Any, values: dict[str, Any]) -> None:
    updater = getattr(sf.config, "update", None)
    if callable(updater):
        try:
            updater(values)
            return
        except TypeError:
            pass
    for key, value in values.items():
        config_set(sf, key, value)


def set_model_time(sf: Any, start, stop) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Set ``tref``, ``tstart`` and ``tstop`` through the native config component."""
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(stop)
    config_update(sf, {"tref": t0.to_pydatetime(), "tstart": t0.to_pydatetime(), "tstop": t1.to_pydatetime()})
    return t0, t1


def register_raster_or_dataset(sf: Any, name: str, path: Path, *, crs: str | int | None = None) -> str:
    """Register a local raster/netCDF source and return its data-catalog name.

    HydroMT-SFINCS components generally accept either paths or catalog names.  We
    register explicit names to make manifests and logs auditable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    driver = "raster_xarray" if suffix in {".nc", ".zarr"} else "rasterio"
    metadata: dict[str, Any] = {}
    if crs is not None:
        metadata["crs"] = int(crs) if isinstance(crs, int) else str(crs)
    sf.data_catalog.from_dict(
        {
            name: {
                "uri": str(path),
                "data_type": "RasterDataset",
                "driver": {"name": driver},
                "metadata": metadata,
            }
        }
    )
    return name


def component_dataframe(component: Any, attr_candidates: tuple[str, ...] = ("gdf", "data")):
    for attr in attr_candidates:
        value = getattr(component, attr, None)
        if value is not None:
            return value
    return None
