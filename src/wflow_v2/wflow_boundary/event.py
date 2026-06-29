from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from .api import BoundaryRun, DesignEvent, Probability
from .domain import domain_submodels, model_crs, read_handoff_artifacts, read_handoff_points
from .paths import event_catalog_path, location_path, write_json
from .qa import read_acceptance, validate_event_boundary, write_acceptance
from .states import prepare_event_instate, validate_instates
from wflow_v2.wflow_boundary_compat.catalog import bind_event_catalog_template, ensure_local_catalog, write_event_update_workflow
from wflow_v2.wflow_boundary_compat.hydromt_native import update_model, workflow_steps
from wflow_v2.wflow_boundary_compat.output import gauge_discharge, match_gauge_column
from wflow_v2.wflow_boundary_compat.runner import clean_output_dir, run_solver, zero_event_forcing

CFS_TO_CMS = 0.028316846592


def event_paths(config: dict[str, Any], location_root: str | Path, event_id: str) -> dict[str, Path]:
    root = Path(location_root)
    events_root = location_path(root, (config.get("wflow", {}) or {}).get("events_root", "data/wflow/events"))
    event_root = events_root / str(event_id)
    return {
        "event_root": event_root,
        "discharge": event_root / "sfincs_discharge.nc",
        "qa_csv": event_root / "sfincs_discharge.qa.csv",
        "acceptance": event_root / "sfincs_discharge.acceptance.json",
        "amplification": event_root / "sfincs_discharge.amplification.json",
        "zero_rain_discharge": event_root / "_zero_rain" / "sfincs_discharge.nc",
    }


def event_window(reference_time, *, pre_event_hours: float = 48.0, post_event_hours: float = 72.0, timestep_seconds: int = 3600) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.Timestamp(reference_time)
    step = pd.Timedelta(seconds=int(timestep_seconds))
    return (ref - pd.Timedelta(hours=float(pre_event_hours))).floor(step), (ref + pd.Timedelta(hours=float(post_event_hours))).ceil(step)


def read_event(config: dict[str, Any], location_root: str | Path, event_id: str, *, catalog_path=None) -> DesignEvent:
    root = Path(location_root)
    catalog_file = event_catalog_path(config, root, catalog_path)
    catalog = pd.read_csv(catalog_file, dtype={"event_id": str})
    match = catalog[catalog["event_id"].astype(str).eq(str(event_id))]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not found in {catalog_file}")
    row = match.iloc[0]
    reference_time = pd.Timestamp(_first(row, ["event_reference_time", "reference_time", "time"]))
    pre, post = configured_event_window_hours(config)
    start, end = event_window(reference_time, pre_event_hours=pre, post_event_hours=post)
    paths = event_paths(config, root, event_id)
    precip = _path_from_row(root, row, ["wflow_precip_path", "event_precip", "precip_path"], paths["event_root"] / "precip.nc")
    temp_pet = _path_from_row(root, row, ["wflow_temp_pet_path", "event_temp_pet", "temp_pet_path"], paths["event_root"] / "temp_pet.nc")
    q_target_cms = _target_cms(row)
    probability = Probability(
        p_event=_float_or_none(_first(row, ["p_event", "probability", "event_probability"])),
        aep=_float_or_none(_first(row, ["aep", "annual_exceedance_probability"])),
        return_period_years=_float_or_none(_first(row, ["return_period_years", "sample_rp_years", "rp_years"])),
        weight=_float_or_none(_first(row, ["weight", "catalog_weight"])),
    )
    return DesignEvent(
        event_id=str(event_id),
        reference_time=reference_time,
        window_start=start,
        window_end=end,
        precip_path=precip,
        temp_pet_path=temp_pet,
        rainfall_member_id=_string_or_none(_first(row, ["rainfall_member_id", "member_id"])),
        rainfall_member_file=_path_from_row(root, row, ["rainfall_member_file"], None),
        rainfall_scale_factor=float(_float_or_none(_first(row, ["rainfall_scale_factor", "rainfall_scale"])) or 1.0),
        probability=probability,
        q_target_cms=q_target_cms,
        attrs={"catalog_path": str(catalog_file)},
    )


def configured_event_window_hours(config: dict[str, Any]) -> tuple[float, float]:
    cfg = ((config.get("wflow", {}) or {}).get("event_window", {}) or {})
    timing = ((config.get("scenario_build", {}) or {}).get("timing", {}) or {})
    pre = float(cfg.get("pre_event_hours", 48.0))
    post = float(cfg.get("post_event_hours", 72.0 + float(timing.get("drain_down_hours", 0.0) or 0.0)))
    return pre, post


def run_event_boundary(
    config: dict[str, Any],
    location_root: str | Path,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = True,
    force: bool = False,
    zero_rain: bool = False,
    model_cls=None,
) -> BoundaryRun:
    """Run one stochastic event through HydroMT-Wflow and write SFINCS discharge forcing."""
    root = Path(location_root)
    event = read_event(config, root, event_id, catalog_path=catalog_path)
    paths = event_paths(config, root, event_id)
    paths["event_root"].mkdir(parents=True, exist_ok=True)
    if execute:
        validate_instates(config, root, raise_on_error=True)
        _require_event_forcing(event)

    submodels = domain_submodels(config, root)
    if not submodels:
        raise RuntimeError("No Wflow domain submodels are configured or manifested")
    wflow = config.get("wflow", {}) or {}
    base_root = location_path(root, wflow.get("base_model_root", "data/wflow/base"))
    update_config = location_path(root, wflow.get("update_forcing_config", "wflow_update_forcing.yml"))
    base_catalog = ensure_local_catalog(config, root)
    event_catalog = bind_event_catalog_template(base_catalog, paths["event_root"], event.event_id)
    event_update = write_event_update_workflow(update_config, paths["event_root"], event.window_start, event.window_end)
    update_steps = workflow_steps(event_update)

    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for submodel in submodels:
        sid = str(submodel["wflow_submodel_id"])
        base_model = base_root / sid
        event_model = paths["event_root"] / sid
        if execute:
            if force and event_model.exists():
                shutil.rmtree(event_model)
            update_model(base_model, event_model, steps=update_steps, data_libs=[str(event_catalog)], model_cls=model_cls)
            prepare_event_instate(event_model, base_model, model_cls=model_cls)
            clean_output_dir(event_model / "wflow_sbm.toml")
            run_solver(config, event_model / "wflow_sbm.toml", cwd=root)
        gauges = event_model / "staticgeoms" / "gauges_sfincs.geojson"
        if not gauges.exists():
            gauges = base_model / "staticgeoms" / "gauges_sfincs.geojson"
        outputs.append({"run_model_root": event_model, "run_output_dir": event_model / "run_event", "gauges_geojson": gauges, "submodel_id": sid})
        rows.append({"event_id": event.event_id, "wflow_submodel_id": sid, "event_model_root": str(event_model), "run_output_dir": str(event_model / "run_event"), "gauges_geojson": str(gauges), "status": "completed" if execute else "planned"})

    amplification = {"K": 1.0, "status": "not_run"}
    zero_path = None
    if execute:
        merge_submodel_discharge(outputs, model_crs=model_crs(config), out_path=paths["discharge"], handoff_points=sfincs_handoff_points(config, root))
        amplification = apply_same_frequency_amplification(config, root, event, discharge_nc=paths["discharge"], submodel_outputs=outputs)
        write_json(paths["amplification"], amplification)
        if zero_rain:
            zero_path = run_zero_rain_control(config, root, event.event_id, execute=True, model_cls=model_cls)
        expected = {p.id for p in read_handoff_points(config, root, crs=model_crs(config))}
        qa = validate_event_boundary(paths["discharge"], expected_source_ids=expected, window=(event.window_start, event.window_end), zero_rain_discharge_nc=zero_path, raise_on_error=False)
        qa.to_csv(paths["qa_csv"], index=False)
        write_acceptance(paths["acceptance"], event=event, discharge_nc=paths["discharge"], qa_report=qa, amplification=amplification, metadata={"hydromt_update_workflow": str(event_update), "hydromt_data_catalog": str(event_catalog)})
        status = "accepted" if not qa["status"].isin(["failed", "review_required"]).any() else "failed"
    else:
        qa = pd.DataFrame()
        status = "planned"
    return BoundaryRun(event=event, discharge_nc=paths["discharge"], acceptance_json=paths["acceptance"], qa_csv=paths["qa_csv"], status=status, amplification=amplification, report=pd.DataFrame(rows))


def require_event_boundary(config: dict[str, Any], location_root: str | Path, event_id: str) -> pd.Series:
    paths = event_paths(config, location_root, event_id)
    payload = read_acceptance(paths["acceptance"])
    if payload.get("status") != "accepted":
        raise RuntimeError(f"Wflow event boundary is not accepted: {paths['acceptance']}")
    if not paths["discharge"].exists():
        raise FileNotFoundError(paths["discharge"])
    return pd.Series({"event_id": str(event_id), "status": "accepted", "sfincs_discharge": str(paths["discharge"]), "acceptance_json": str(paths["acceptance"])}, name="wflow_event_boundary_acceptance")


def run_zero_rain_control(config: dict[str, Any], location_root: str | Path, event_id: str, *, execute: bool = True, model_cls=None) -> Path:
    root = Path(location_root)
    paths = event_paths(config, root, event_id)
    zero_root = paths["event_root"] / "_zero_rain"
    outputs: list[dict[str, Any]] = []
    for submodel in domain_submodels(config, root):
        sid = str(submodel["wflow_submodel_id"])
        source = paths["event_root"] / sid
        target = zero_root / sid
        if execute:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            zero_event_forcing(target / "inmaps-event.nc")
            clean_output_dir(target / "wflow_sbm.toml")
            run_solver(config, target / "wflow_sbm.toml", cwd=root)
        outputs.append({"run_model_root": target, "run_output_dir": target / "run_event", "gauges_geojson": target / "staticgeoms" / "gauges_sfincs.geojson", "submodel_id": sid})
    if execute:
        merge_submodel_discharge(outputs, model_crs=model_crs(config), out_path=paths["zero_rain_discharge"], handoff_points=sfincs_handoff_points(config, root))
    return paths["zero_rain_discharge"]


def sfincs_handoff_points(config: dict[str, Any], location_root: str | Path) -> dict[str, tuple[float, float]]:
    gdf = read_handoff_artifacts(config, location_root, crs=model_crs(config))
    return {str(r["sfincs_handoff_id"]): (float(r.geometry.x), float(r.geometry.y)) for _, r in gdf.iterrows()}


def merge_submodel_discharge(submodel_outputs: list[dict[str, Any]], *, model_crs: str, out_path: str | Path, handoff_points: dict[str, tuple[float, float]] | None = None) -> Path:
    series: dict[str, pd.Series] = {}
    points: dict[str, tuple[float, float]] = {}
    crs_by_handoff: dict[str, Any] = {}
    for item in submodel_outputs:
        sub_series, sub_points, sub_crs = gauge_discharge(item["run_model_root"], item["gauges_geojson"], run_output_dir=item.get("run_output_dir"))
        series.update(sub_series)
        points.update(sub_points)
        for key in sub_points:
            crs_by_handoff[key] = sub_crs
    points = _reproject_points(points, crs_by_handoff, model_crs)
    if handoff_points:
        points.update({hid: xy for hid, xy in handoff_points.items() if hid in series})
    ds = build_discharge_dataset(series, points, crs=model_crs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def build_discharge_dataset(series_by_handoff: dict[str, pd.Series], points_by_handoff: dict[str, tuple[float, float]], *, crs: str) -> xr.Dataset:
    ids = [hid for hid in series_by_handoff if hid in points_by_handoff]
    if not ids:
        raise ValueError("no discharge series overlap handoff point coordinates")
    frame = pd.concat({hid: pd.Series(series_by_handoff[hid]).astype(float) for hid in ids}, axis=1).sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    ds = xr.Dataset(
        {"discharge": (("index", "time"), frame[ids].to_numpy(dtype=float).T)},
        coords={
            "index": np.arange(1, len(ids) + 1, dtype=int),
            "time": frame.index.values,
            "name": ("index", np.asarray(ids, dtype=object)),
            "x": ("index", np.asarray([points_by_handoff[i][0] for i in ids], dtype=float)),
            "y": ("index", np.asarray([points_by_handoff[i][1] for i in ids], dtype=float)),
        },
        attrs={"crs": str(crs), "featureType": "timeSeries"},
    )
    ds["discharge"].attrs.update(units="m3 s-1", standard_name="river_water__volume_flow_rate")
    try:
        ds.vector.set_crs(crs)
    except Exception:
        pass
    return ds


def apply_same_frequency_amplification(config: dict[str, Any], location_root: str | Path, event: DesignEvent, *, discharge_nc: str | Path, submodel_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    amp_cfg = ((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {})
    reference_gage = amp_cfg.get("primary_reference_gage") or (config.get("inland_coupling", {}) or {}).get("primary_reference_gage")
    band = amp_cfg.get("k_band", [0.25, 4.0])
    provenance = {"method": "same_frequency_amplification", "equation": "K = clip(Q*_omega(g*) / max_t Q^W_omega(g*,t), K_min, K_max)", "K": 1.0, "status": "response_based_unbiased", "reference_gage": reference_gage, "target_cms": event.q_target_cms, "wflow_peak_cms": None, "k_band": band}
    if not amp_cfg.get("enabled", True):
        provenance["status"] = "disabled"
        return provenance
    k_cal = _float_or_none(amp_cfg.get("k_calibration"))
    if k_cal and k_cal > 0 and not amp_cfg.get("prefer_per_event_target", False):
        k = _clip(k_cal, band)
        _scale_discharge(discharge_nc, k, reference_gage)
        provenance.update(K=k, status="calibration_constant")
        return provenance
    if not reference_gage or not event.q_target_cms:
        return provenance
    peak = reference_gage_peak(config, location_root, str(reference_gage), submodel_outputs)
    provenance["wflow_peak_cms"] = peak
    if not peak or peak <= 0:
        provenance["status"] = "no_wflow_peak"
        return provenance
    k = _clip(float(event.q_target_cms) / float(peak), band)
    _scale_discharge(discharge_nc, k, reference_gage)
    provenance.update(K=k, status="applied")
    return provenance


def reference_gage_peak(config: dict[str, Any], location_root: str | Path, reference_gage: str, submodel_outputs: list[dict[str, Any]]) -> float | None:
    import geopandas as gpd

    gauges_root = location_path(location_root, ((config.get("wflow", {}) or {}).get("gauges", {}) or {}).get("root", "data/wflow/domain_set_gauges"))
    for item in submodel_outputs:
        sid = str(item["submodel_id"])
        gauges_path = gauges_root / f"{sid}_observation_gauges.geojson"
        if not gauges_path.exists():
            continue
        gauges = gpd.read_file(gauges_path)
        if "site_no" not in gauges:
            continue
        match = gauges[gauges["site_no"].astype(str).str.zfill(8).eq(str(reference_gage).zfill(8))]
        if match.empty:
            continue
        csv_path = _output_csv(item["run_output_dir"])
        table = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        col = match_gauge_column(table.columns, match.iloc[0].get("index"))
        if col is None:
            continue
        peak = float(pd.to_numeric(table[col], errors="coerce").max())
        if np.isfinite(peak):
            return peak
    return None


def _reproject_points(points: dict[str, tuple[float, float]], crs_by_handoff: dict[str, Any], dst_crs: str) -> dict[str, tuple[float, float]]:
    from pyproj import CRS, Transformer

    out: dict[str, tuple[float, float]] = {}
    transformers: dict[str, Any] = {}
    for hid, (x, y) in points.items():
        src = crs_by_handoff.get(hid)
        if src is None or CRS.from_user_input(src) == CRS.from_user_input(dst_crs):
            out[hid] = (float(x), float(y))
            continue
        key = str(src)
        transformers.setdefault(key, Transformer.from_crs(src, dst_crs, always_xy=True))
        nx, ny = transformers[key].transform(x, y)
        out[hid] = (float(nx), float(ny))
    return out


def _scale_discharge(discharge_nc: str | Path, k: float, reference_gage: str | None) -> None:
    if float(k) == 1.0:
        return
    path = Path(discharge_nc)
    with xr.open_dataset(path) as opened:
        ds = opened.load()
    ds["discharge"] = ds["discharge"] * float(k)
    ds.attrs["same_frequency_amplification_K"] = float(k)
    if reference_gage:
        ds.attrs["amplification_reference_gage"] = str(reference_gage)
    tmp = path.with_suffix(".amp.tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(path)


def _output_csv(run_output_dir: str | Path) -> Path:
    root = Path(run_output_dir)
    for name in ["output.csv", "output_scalar.csv"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"no Wflow output CSV under {root}")
    return matches[0]


def _require_event_forcing(event: DesignEvent) -> None:
    missing = [str(path) for path in [event.precip_path, event.temp_pet_path] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Wflow event forcing is missing: " + "; ".join(missing))


def _target_cms(row: pd.Series) -> float | None:
    for key, factor in [("streamflow_target_cms", 1.0), ("q_target_cms", 1.0), ("streamflow_target_cfs", CFS_TO_CMS), ("target_peak_cfs", CFS_TO_CMS)]:
        value = _float_or_none(row.get(key))
        if value and value > 0:
            return float(value) * factor
    return None


def _clip(value: float, band) -> float:
    if not band:
        return float(value)
    return float(np.clip(float(value), float(band[0]), float(band[1])))


def _first(row: pd.Series, keys: list[str]):
    for key in keys:
        if key in row and not pd.isna(row.get(key)) and str(row.get(key)).strip() != "":
            return row.get(key)
    return None


def _path_from_row(root: Path, row: pd.Series, keys: list[str], default) -> Path | None:
    value = _first(row, keys)
    if value is None:
        return default
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _float_or_none(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _string_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
