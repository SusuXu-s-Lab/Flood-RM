"""Domain-set Wflow event replay → merged SFINCS discharge forcing.

The greensboro-style inland coupling builds one Wflow submodel per encompassing-HUC /
stream-boundary-crossing domain (see ``wflow.domain_set``). A single Event-Catalog event
is replayed by, for each submodel:

  1. resolving the event time window from the catalog ``event_reference_time``,
  2. writing per-event HydroMT update-forcing config + data catalog (so the
     ``event_precip`` / ``event_temp_pet`` pointers resolve to *this* event's forcing),
  3. ``hydromt update wflow_sbm <base>/<submodel> -i <update> -d <catalog> -o events/<event>/<submodel>``,
  4. running the Wflow engine on the produced run config,

then merging each submodel's ``Q`` series at its ``gauges_sfincs`` handoff points into a
single ``events/<event>/sfincs_discharge.nc`` GeoDataset keyed by ``sfincs_handoff_id``.
``SfincsModel.discharge_points.create(geodataset=...)`` reads that file to force the
coupled SFINCS run (see ``locations/<loc>/02_flood/04/run_example.ipynb``).

The single-model ``run_wflow_event_replay`` in :mod:`wflow_runs.notebook` only targets one
base; this module is the domain-set generalisation that also produces the merged handoff.

Prerequisites:
  - per-event meteo forcing ``events/<event>/precip.nc`` and ``temp_pet.nc``
    (the ``event_precip`` / ``event_temp_pet`` data-catalog contract; use
    ``build_event_meteo_forcing`` to stage it from the Event Catalog rainfall member),
  - built Wflow submodels (``wflow_sbm.toml`` + ``staticmaps.nc``),
  - a Wflow engine for ``execute=True`` (env ``WFLOW_BIN``, or ``wflow.run.command`` in config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import subprocess

import numpy as np
import pandas as pd
import yaml

from sfincs_runs.hydrology import find_aorc_event_window, prepare_aorc_precip_for_sfincs
from wflow_runs.notebook import (
    _describe_hydromt_command,
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
    resolve_location_path,
)


# ─── pure helpers (unit-tested) ───────────────────────────────────────────────


def resolve_event_window(
    reference_time,
    *,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
    timestep_seconds: int = 3600,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Wflow simulation window bracketing a catalog event's reference time.

    ``pre_event_hours`` is spin-up before the peak; ``post_event_hours`` lets the
    hydrograph recede. Both ends are snapped to the forcing timestep so the window is
    an exact number of steps.
    """
    ref = pd.Timestamp(reference_time)
    if pd.isna(ref):
        raise ValueError(f"event_reference_time is not a valid timestamp: {reference_time!r}")
    step = pd.Timedelta(seconds=int(timestep_seconds))
    start = (ref - pd.Timedelta(hours=float(pre_event_hours))).floor(step)
    end = (ref + pd.Timedelta(hours=float(post_event_hours))).ceil(step)
    return start, end


def build_event_meteo_forcing(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
    overwrite: bool = False,
) -> dict:
    """Stage the per-event Wflow forcing files consumed by the replay update.

    The rainfall field comes from the catalog-selected AORC SST storm window and is
    scaled by ``rainfall_scale_factor``. The companion ``temp_pet.nc`` supplies the
    De Bruin PET variables required by the current HydroMT-Wflow update config; radiation
    is zeroed so short event replays are driven by the scaled rainfall analog rather than
    an unavailable meteorological replay.
    """
    location_root = Path(location_root)
    wflow = config.get("wflow", {})
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    event_dir = events_root / str(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)

    row = _event_catalog_row(location_root, event_id, catalog_path)
    start, end = resolve_event_window(
        row["event_reference_time"],
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )

    precip_path = event_dir / "precip.nc"
    temp_pet_path = event_dir / "temp_pet.nc"
    write_precip = overwrite or not precip_path.exists()
    write_temp_pet = overwrite or write_precip or not temp_pet_path.exists()

    rainfall_source_nc = None
    scale_factor = _positive_float(row.get("rainfall_scale_factor"), default=1.0)
    if write_precip:
        rainfall_source_nc = _event_rainfall_source_nc(config, location_root, row)
        precip_cfg = (wflow.get("event_forcing", {}) or {}).get("precipitation", {}) or {}
        aorc_cfg = (config.get("collection", {}) or {}).get("aorc_sst", {}) or {}
        prepare_aorc_precip_for_sfincs(
            rainfall_source_nc,
            precip_path,
            t_start=start,
            t_stop=end,
            variable=str(precip_cfg.get("variable", aorc_cfg.get("variable", "APCP_surface"))),
            align_start_to_run=True,
            window_alignment=str(precip_cfg.get("window_alignment", "start")),
            precip_start=_catalog_rainfall_start(row),
            scale_factor=scale_factor,
        )

    if write_temp_pet:
        _write_neutral_temp_pet(temp_pet_path, precip_path)

    return {
        "event_id": str(event_id),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "rainfall_source_nc": str(rainfall_source_nc) if rainfall_source_nc else "",
        "rainfall_scale_factor": scale_factor,
        "precip_path": str(precip_path),
        "temp_pet_path": str(temp_pet_path),
        "precip_written": bool(write_precip),
        "temp_pet_written": bool(write_temp_pet),
    }


def build_discharge_geodataset(
    series_by_handoff: dict[str, pd.Series],
    points_by_handoff: dict[str, tuple[float, float]],
    *,
    crs,
    variable: str = "discharge",
):
    """Assemble per-handoff discharge series into a HydroMT GeoDataset.

    ``series_by_handoff`` maps ``sfincs_handoff_id`` → a discharge ``pd.Series`` on a
    DatetimeIndex; ``points_by_handoff`` maps the same id → ``(x, y)`` in ``crs``. The
    result has dims ``(index, time)`` with a *unique integer* ``index`` (required by
    ``SfincsModel.discharge_points``), the ``sfincs_handoff_id`` carried as a ``name``
    coordinate, and ``x``/``y`` point coordinates — the GeoDataset shape that
    ``discharge_points.create`` reads (pass the opened ``xr.Dataset``, not the path, so
    HydroMT uses the xarray driver rather than the vector fallback).
    """
    import xarray as xr

    handoff_ids = [hid for hid in series_by_handoff if hid in points_by_handoff]
    if not handoff_ids:
        raise ValueError("no handoff points with both a discharge series and a location")

    frame = pd.concat(
        {hid: pd.Series(series_by_handoff[hid]).astype(float) for hid in handoff_ids},
        axis=1,
    ).sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame[handoff_ids]  # stable column order matching handoff_ids

    xs = np.array([float(points_by_handoff[hid][0]) for hid in handoff_ids], dtype=float)
    ys = np.array([float(points_by_handoff[hid][1]) for hid in handoff_ids], dtype=float)

    ds = xr.Dataset(
        {variable: (("index", "time"), frame.to_numpy(dtype=float).T)},
        coords={
            "index": np.arange(1, len(handoff_ids) + 1, dtype=int),
            "time": frame.index.values,
            "name": ("index", np.array(handoff_ids, dtype=object)),
            "x": ("index", xs),
            "y": ("index", ys),
        },
    )
    ds[variable].attrs.update(units="m3 s-1", standard_name="river_water__volume_flow_rate")
    epsg = _epsg(crs)
    if epsg is not None:
        ds.attrs["crs"] = epsg
        try:  # hydromt's GeoDataset vector accessor stamps a spatial_ref the reader honours
            ds = ds.vector.set_crs(epsg)
        except Exception:
            pass
    return ds


def _epsg(crs):
    if crs is None:
        return None
    try:
        from pyproj import CRS

        return int(CRS.from_user_input(crs).to_epsg())
    except Exception:
        try:
            return int(crs)
        except (TypeError, ValueError):
            return None


def read_submodel_gauge_discharge(
    run_output_dir: Path,
    gauges_geojson: Path,
    *,
    csv_name: str | None = None,
):
    """Read one submodel's Wflow gauge discharge + handoff locations.

    Returns ``(series_by_handoff, points_by_handoff)`` in the gauges' native CRS. Wflow
    writes per-gauge columns ``Q_<index>`` (one per ``gauges_sfincs`` row) to its output
    CSV; each gauge row carries the ``sfincs_handoff_id`` and geometry we key on.
    """
    import geopandas as gpd

    gauges = gpd.read_file(gauges_geojson)
    if "sfincs_handoff_id" not in gauges or "index" not in gauges:
        raise ValueError(f"{gauges_geojson} lacks sfincs_handoff_id/index columns")

    csv_path = _resolve_wflow_output_csv(Path(run_output_dir), csv_name)
    table = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    series_by_handoff: dict[str, pd.Series] = {}
    points_by_handoff: dict[str, tuple[float, float]] = {}
    for _, gauge in gauges.iterrows():
        handoff_id = str(gauge["sfincs_handoff_id"])
        column = _match_gauge_column(table.columns, gauge["index"])
        if column is None:
            continue
        series_by_handoff[handoff_id] = table[column]
        points_by_handoff[handoff_id] = (float(gauge.geometry.x), float(gauge.geometry.y))
    if not series_by_handoff:
        raise ValueError(f"no Q_<index> gauge columns matched in {csv_path}")
    return series_by_handoff, points_by_handoff


def _resolve_wflow_output_csv(run_output_dir: Path, csv_name: str | None) -> Path:
    if csv_name:
        candidate = run_output_dir / csv_name
        if candidate.exists():
            return candidate
    candidates = sorted(run_output_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no Wflow output CSV under {run_output_dir}")
    named = [c for c in candidates if c.name in {"output.csv", "output_scalar.csv"}]
    return named[0] if named else candidates[0]


def _match_gauge_column(columns, index_value) -> str | None:
    try:
        index_text = str(int(float(index_value)))
    except (TypeError, ValueError):
        index_text = str(index_value)
    exact = {f"Q_{index_text}", f"Q_gauges_sfincs_{index_text}", f"Q_{index_value}"}
    for column in columns:
        if column in exact:
            return column
    for column in columns:  # fall back to a suffix match (Q_*_<index>)
        if str(column).startswith("Q") and str(column).endswith(f"_{index_text}"):
            return column
    return None


def merge_submodel_discharge(
    submodel_outputs: list[dict],
    *,
    model_crs,
    out_path: Path,
):
    """Merge per-submodel gauge discharge into one ``sfincs_discharge.nc`` GeoDataset.

    Each entry in ``submodel_outputs`` is ``{"run_output_dir", "gauges_geojson"}``. The
    merged series are reprojected to ``model_crs`` (the SFINCS grid CRS) so the handoff
    points land on the SFINCS ``src`` locations.
    """
    import geopandas as gpd

    series_by_handoff: dict[str, pd.Series] = {}
    native_points: dict[str, tuple[float, float]] = {}
    native_crs = None
    for entry in submodel_outputs:
        gauges_path = Path(entry["gauges_geojson"])
        native_crs = native_crs or gpd.read_file(gauges_path).crs
        series, points = read_submodel_gauge_discharge(entry["run_output_dir"], gauges_path)
        series_by_handoff.update(series)
        native_points.update(points)

    points_by_handoff = _reproject_points(native_points, native_crs, model_crs)
    ds = build_discharge_geodataset(series_by_handoff, points_by_handoff, crs=model_crs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


def _reproject_points(points_by_handoff, src_crs, dst_crs):
    if not points_by_handoff or _epsg(src_crs) == _epsg(dst_crs) or src_crs is None:
        return points_by_handoff
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = {}
    for handoff_id, (x, y) in points_by_handoff.items():
        nx, ny = transformer.transform(x, y)
        out[handoff_id] = (float(nx), float(ny))
    return out


# ─── orchestration (subprocess steps gated behind execute=True) ───────────────


@dataclass(frozen=True)
class ReplayStep:
    submodel_id: str
    update_command: str
    run_command: str
    run_output_dir: str
    gauges_geojson: str


def replay_inland_domain_set(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = False,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
) -> pd.DataFrame:
    """Replay every Wflow submodel for one event and write the merged SFINCS discharge.

    With ``execute=False`` (default) this plans the work — resolves the window, checks
    prerequisites, and returns the per-submodel HydroMT-update + Wflow-run commands —
    without spawning anything. With ``execute=True`` it runs the updates + Wflow engine
    and writes ``events/<event>/sfincs_discharge.nc``.
    """
    location_root = Path(location_root)
    wflow = config.get("wflow", {})
    base_root = resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base"))
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    update_cfg = resolve_location_path(location_root, wflow.get("update_forcing_config", "wflow_update_forcing.yml"))
    data_catalog = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    model_crs = wflow.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))

    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    start, end = resolve_event_window(reference_time, pre_event_hours=pre_event_hours, post_event_hours=post_event_hours)

    event_dir = events_root / event_id
    if execute:
        # Per-event meteo forcing is only consumed by the actual HydroMT update; a dry run
        # still plans (and writes the runnable -i/-d configs) without it.
        _require_event_forcing(event_dir)
    submodels = _domain_set_submodels(config, location_root)
    if not submodels:
        raise ValueError("wflow.domain_set has no submodels to replay")

    event_dir.mkdir(parents=True, exist_ok=True)
    per_event_catalog = _write_per_event_data_catalog(data_catalog, event_dir, event_id)
    per_event_update = _write_per_event_update_config(update_cfg, event_dir, start, end)
    run_command_template = _wflow_run_command(config)

    steps: list[ReplayStep] = []
    for submodel in submodels:
        submodel_id = str(submodel["wflow_submodel_id"])
        submodel_base = base_root / submodel_id
        if not (submodel_base / "wflow_sbm.toml").exists():
            raise FileNotFoundError(f"Wflow submodel base not built: {submodel_base}")
        out_dir = event_dir / submodel_id
        update_command = (
            f"hydromt update wflow_sbm {submodel_base} "
            f"-i {per_event_update} -d {per_event_catalog} -o {out_dir} -vvv"
        )
        run_config = out_dir / "run_event.toml"
        run_command = run_command_template.format(run_config=run_config)
        steps.append(
            ReplayStep(
                submodel_id=submodel_id,
                update_command=update_command,
                run_command=run_command,
                run_output_dir=str(out_dir / "run_event"),
                gauges_geojson=str(submodel_base / "staticgeoms" / "gauges_sfincs.geojson"),
            )
        )

    rows = []
    for step in steps:
        status = "planned"
        resolved_update_command, hydromt_runner_status, hydromt_runner_issue = _describe_hydromt_command(
            step.update_command,
            location_root,
        )
        if execute:
            _run(_resolve_hydromt_command(step.update_command, location_root), cwd=location_root)
            _run(shlex.split(step.run_command), cwd=location_root)
            status = "completed"
        rows.append(
            {
                "event_id": event_id,
                "submodel_id": step.submodel_id,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "update_command": step.update_command,
                "resolved_update_command": resolved_update_command,
                "hydromt_runner_status": hydromt_runner_status,
                "hydromt_runner_issue": hydromt_runner_issue,
                "run_command": step.run_command,
                "run_output_dir": step.run_output_dir,
                "status": status,
            }
        )

    discharge_path = event_dir / "sfincs_discharge.nc"
    if execute:
        merge_submodel_discharge(
            [{"run_output_dir": s.run_output_dir, "gauges_geojson": s.gauges_geojson} for s in steps],
            model_crs=model_crs,
            out_path=discharge_path,
        )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    return report


# ─── orchestration helpers ────────────────────────────────────────────────────


def _event_catalog_row(location_root: Path, event_id: str, catalog_path):
    catalog_path = (
        Path(catalog_path)
        if catalog_path
        else resolve_location_path(location_root, "data/event_catalog/catalog/probability_catalog.csv")
    )
    if not catalog_path.is_absolute():
        catalog_path = location_root / catalog_path
    catalog = pd.read_csv(catalog_path)
    catalog["event_id"] = catalog["event_id"].astype(str)
    match = catalog[catalog["event_id"] == str(event_id)]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not in {catalog_path}")
    if "event_reference_time" not in match:
        raise ValueError(f"{catalog_path} has no event_reference_time column")
    return match.iloc[0]


def _event_reference_time(location_root: Path, event_id: str, catalog_path):
    return _event_catalog_row(location_root, event_id, catalog_path)["event_reference_time"]


def _event_rainfall_source_nc(config: dict, location_root: Path, row: pd.Series) -> Path:
    rainfall_member_file = _required_event_value(row, "rainfall_member_file")
    rainfall_member_file = resolve_location_path(location_root, rainfall_member_file)
    precip_cfg = (
        (config.get("wflow", {}) or {})
        .get("event_forcing", {})
        .get("precipitation", {})
        or {}
    )
    event_windows_dir = precip_cfg.get("event_windows_dir") or (rainfall_member_file.parent / "event_windows")
    event_windows_dir = resolve_location_path(location_root, event_windows_dir)
    return find_aorc_event_window(
        event_windows_dir,
        member_id=str(_required_event_value(row, "rainfall_member_id")),
        storm_start=_required_event_value(row, "rainfall_member_time"),
    )


def _required_event_value(row: pd.Series, key: str):
    value = row.get(key)
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Event Catalog row is missing required Wflow forcing field: {key}")
    return value


def _catalog_rainfall_start(row: pd.Series):
    reference = row.get("event_reference_time")
    offset = row.get("rainfall_start_offset_hours")
    if reference is None or offset is None or pd.isna(reference) or pd.isna(offset):
        return None
    return pd.Timestamp(reference) + pd.Timedelta(hours=float(offset))


def _positive_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not (np.isfinite(out) and out > 0):
        return float(default)
    return out


def _write_neutral_temp_pet(
    out_path: Path,
    precip_path: Path,
    *,
    temp_c: float = 15.0,
    press_msl_hpa: float = 1013.25,
) -> Path:
    import xarray as xr

    if not Path(precip_path).exists():
        raise FileNotFoundError(f"precip.nc must be written before temp_pet.nc: {precip_path}")
    with xr.open_dataset(precip_path) as precip_ds:
        template = precip_ds["precip"].transpose("time", "y", "x")
        coords = {dim: template.coords[dim].values for dim in ("time", "y", "x")}
        shape = template.shape

    def filled(value: float):
        return np.full(shape, float(value), dtype=np.float32)

    ds = xr.Dataset(
        {
            "temp": (("time", "y", "x"), filled(temp_c)),
            "press_msl": (("time", "y", "x"), filled(press_msl_hpa)),
            "kin": (("time", "y", "x"), filled(0.0)),
            "kout": (("time", "y", "x"), filled(0.0)),
        },
        coords=coords,
        attrs={
            "crs": "EPSG:4326",
            "source": "neutral short-event Wflow PET companion for scaled AORC rainfall replay",
        },
    )
    ds["temp"].attrs.update(units="degree C")
    ds["press_msl"].attrs.update(units="hPa")
    ds["kin"].attrs.update(units="W m-2")
    ds["kout"].attrs.update(units="W m-2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


def _require_event_forcing(event_dir: Path) -> None:
    missing = [name for name in ("precip.nc", "temp_pet.nc") if not (event_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "per-event Wflow forcing is not built (the event_precip/event_temp_pet contract): "
            + ", ".join(str(event_dir / name) for name in missing)
            + ". Run build_event_meteo_forcing before replaying."
        )


def _domain_set_submodels(config: dict, location_root: Path) -> list[dict]:
    submodels = list(config.get("wflow", {}).get("domain_set", {}).get("submodels", []) or [])
    if submodels:
        return submodels
    manifest = resolve_location_path(
        location_root, config.get("wflow", {}).get("domain_set_manifest", "data/wflow/domain_set.yaml")
    )
    if manifest.exists():
        return list((yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("submodels", []) or [])
    return []


def _write_per_event_data_catalog(data_catalog: Path, event_dir: Path, event_id: str) -> Path:
    """Materialise the data catalog with ``<event_id>`` placeholders bound to this event."""
    text = Path(data_catalog).read_text(encoding="utf-8").replace("<event_id>", str(event_id))
    out = event_dir / "_replay_data_catalog.yml"
    out.write_text(text, encoding="utf-8")
    return out


def _write_per_event_update_config(update_cfg: Path, event_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Copy the update-forcing config with the event window substituted into setup_config."""
    workflow = yaml.safe_load(Path(update_cfg).read_text(encoding="utf-8")) or {}
    if "setup_config" in workflow:
        data = workflow["setup_config"].setdefault("data", {})
        data["time.starttime"] = start.strftime("%Y-%m-%dT%H:%M:%S")
        data["time.endtime"] = end.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        for step in workflow.get("steps", []):
            if "setup_config" in step:
                data = step["setup_config"].setdefault("data", {})
                data["time.starttime"] = start.strftime("%Y-%m-%dT%H:%M:%S")
                data["time.endtime"] = end.strftime("%Y-%m-%dT%H:%M:%S")
    out = event_dir / "_wflow_update_forcing.yml"
    out.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    return out


def _wflow_run_command(config: dict) -> str:
    """Resolve the Wflow engine command template (``{run_config}`` placeholder)."""
    run_cfg = config.get("wflow", {}).get("run", {}) or {}
    command = run_cfg.get("command") or os.environ.get(run_cfg.get("bin_env", "WFLOW_BIN"), "")
    if command:
        return command if "{run_config}" in command else f"{command} {{run_config}}"
    # Default to the Wflow.jl CLI convention used by the reference coupling workflow.
    return "wflow_cli {run_config}"


def _run(command_parts, *, cwd) -> None:
    if not command_parts:
        raise ValueError("empty command")
    try:
        subprocess.run(command_parts, cwd=Path(cwd), check=True, env=_hydromt_subprocess_env())
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"executable not found for replay step: {shlex.join([str(p) for p in command_parts])}"
        ) from exc
