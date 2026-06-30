from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from .io import config_set, read_json, register_raster_or_dataset, write_json


def build_inland_scenario_manifest(
    event: dict[str, Any],
    handoff: dict[str, Any],
    handoff_event: dict[str, Any],
    config: dict[str, Any],
    *,
    domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the notebook-visible manifest for one staged inland scenario folder."""
    coupling = config.get("inland_coupling", {}) or {}
    manifest = {
        "event_id": str(event["event_id"]),
        "forcing_mode": coupling.get("forcing_mode", handoff.get("forcing_mode", "dual_fluvial_pluvial")),
        "event_reference_time": _clean_manifest_value(event.get("event_reference_time")),
        "event_origin": _clean_manifest_value(event.get("event_origin")),
        "catalog_role": _clean_manifest_value(event.get("catalog_role")),
        "sampling_scheme": _clean_manifest_value(event.get("sampling_scheme")),
        "event_set": _clean_manifest_value(event.get("event_set")),
        "selection_role": _clean_manifest_value(event.get("selection_role")),
        "selection_reason": _clean_manifest_value(event.get("selection_reason")),
        "severity_band": _clean_manifest_value(event.get("severity_band")),
        "sample_rp_years": _clean_manifest_value(event.get("sample_rp_years")),
        "streamflow_reference_time": coupling.get("streamflow_reference_time", "dominant_streamgage_network_peak"),
        "streamflow_member_id": _clean_manifest_value(event.get("streamflow_member_id")),
        "rainfall_member_id": _clean_manifest_value(event.get("rainfall_member_id")),
        "soil_moisture_member_id": _clean_manifest_value(event.get("soil_moisture_member_id")),
        "probability_weight": _clean_manifest_value(event.get("probability_weight")),
        "wflow_event_dir": _clean_manifest_value(handoff_event.get("wflow_event_dir")),
        "wflow_discharge_forcing": _clean_manifest_value(handoff_event.get("discharge_forcing")),
        "wflow_precip_provenance": _handoff_event_artifact(handoff_event, "precip_provenance.json"),
        "wflow_temp_pet_provenance": _handoff_event_artifact(handoff_event, "temp_pet_provenance.json"),
        "wflow_source_variable": handoff.get("source_variable", "river_q"),
        "wflow_source_standard_name": handoff.get(
            "source_standard_name",
            "river_water__volume_flow_rate",
        ),
        "direct_rainfall_enabled": bool(
            coupling.get("direct_rainfall", {}).get(
                "enabled",
                handoff.get("direct_rainfall_enabled", True),
            )
        ),
        "rainfall_member_file": config.get("event_catalog", {}).get("forcing_members", {}).get("rainfall"),
        "streamflow_member_file": config.get("event_catalog", {}).get("forcing_members", {}).get("streamflow"),
        "soil_moisture_member_file": config.get("event_catalog", {}).get("forcing_members", {}).get("soil_moisture"),
    }
    if domain is not None and domain.get("sfincs_domain_id") is not None:
        manifest.update(
            {
                "sfincs_domain_id": domain["sfincs_domain_id"],
                "sfincs_domain_base_model_root": Path(domain["base_model_root"]).as_posix(),
                "sfincs_handoff_source_ids": list(domain.get("handoff_source_ids", [])),
            }
        )
    return manifest


def write_inland_scenario_reports(report: pd.DataFrame, scenarios_root: str | Path) -> dict[str, Path]:
    """Write inland scenario build and cluster catalog CSV contracts."""
    scenarios_root = Path(scenarios_root)
    scenarios_root.mkdir(parents=True, exist_ok=True)
    build_report_path = scenarios_root / "scenario_build_report.csv"
    scenario_catalog_path = scenarios_root / "scenario_catalog.csv"

    report.to_csv(build_report_path, index=False)
    catalog = report[["event_id", "run_root"]].copy()
    catalog["run_root"] = [Path(value).relative_to(scenarios_root).as_posix() for value in catalog["run_root"]]
    catalog.to_csv(scenario_catalog_path, index=False)
    return {
        "scenario_build_report": build_report_path,
        "scenario_catalog": scenario_catalog_path,
    }


def select_inland_scenario_rows(
    catalog: pd.DataFrame,
    *,
    event_ids=None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select inland scenario catalog rows using the notebook staging policy."""
    out = catalog
    if event_ids:
        wanted = {str(event_id) for event_id in event_ids}
        out = out[out["event_id"].astype(str).isin(wanted)].copy()
        missing = wanted - set(out["event_id"].astype(str))
        if missing:
            raise ValueError("Missing Event Catalog rows: " + ", ".join(sorted(missing)))
    elif limit is not None and "sample_rp_years" in out:
        out = out.copy()
        out["_rp_sort"] = pd.to_numeric(out["sample_rp_years"], errors="coerce").fillna(-1.0)
        out = out.sort_values(["_rp_sort", "event_id"], ascending=[False, True]).drop(columns=["_rp_sort"])
    if limit is not None:
        out = out.head(int(limit)).copy()
    if out.empty:
        raise ValueError("No inland coupled scenarios selected")
    return out.reset_index(drop=True)


def index_inland_handoff_events(handoff: dict[str, Any], event_ids=None) -> dict[str, dict[str, Any]]:
    """Index Wflow-SFINCS handoff events by Event Catalog ID."""
    by_event = {str(row["event_id"]): row for row in handoff.get("events", [])}
    if event_ids is not None:
        missing = sorted({str(event_id) for event_id in event_ids} - set(by_event))
        if missing:
            raise ValueError("Event Catalog has missing Wflow handoff events: " + ", ".join(missing))
    return by_event


def missing_wflow_discharge_forcing(
    location_root: str | Path,
    handoff_by_event: dict[str, dict[str, Any]],
    event_ids,
    *,
    resolve_path=None,
) -> list[str]:
    """Return selected events whose Wflow discharge forcing artifact is missing."""
    root = Path(location_root)
    resolve = resolve_path or (lambda value: root / value)
    missing = []
    for event_id in event_ids:
        handoff_event = handoff_by_event[str(event_id)]
        value = handoff_event.get("discharge_forcing")
        if value in (None, ""):
            missing.append(f"{event_id}: <missing discharge_forcing>")
            continue
        text = str(value)
        if "://" in text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = Path(resolve(text))
        if not path.exists():
            missing.append(text)
    return sorted(missing)


def inland_required_staged_files(config: dict[str, Any]) -> list[str]:
    """Return SFINCS input files required for a staged inland coupled run."""
    direct_rainfall = bool(((config.get("inland_coupling", {}) or {}).get("direct_rainfall", {}) or {}).get("enabled", False))
    initial_conditions = bool(
        ((config.get("inland_coupling", {}) or {}).get("initial_conditions", {}) or {}).get("enabled", True)
    )
    infiltration_required = bool(
        set(config.get("event_drivers") or []) & {"rainfall", "soil_moisture"}
    ) and bool(((config.get("inland_coupling", {}) or {}).get("infiltration", {}) or {}).get("enabled", True))

    required_files = ["sfincs.inp", "sfincs.src", "sfincs.dis", "forcing_manifest.json", "sfincs_subgrid.nc"]
    if direct_rainfall:
        required_files.extend(["sfincs_netampr.nc", "aorc_precip_for_sfincs.nc"])
    if initial_conditions:
        required_files.append("sfincs.ini")
    if infiltration_required:
        required_files.extend(["sfincs.smax", "sfincs.seff", "sfincs.ks"])
    return required_files


def audit_inland_staged_scenario_files(
    staged_catalog: pd.DataFrame,
    scenarios_root: str | Path,
    config: dict[str, Any],
    accepted_event_ids,
    *,
    catalog_source: str | Path,
) -> pd.DataFrame:
    """Audit staged inland SFINCS scenario folders for cluster-run inputs."""
    for column in ["event_id", "run_root"]:
        if column not in staged_catalog:
            raise ValueError(f"Scenario catalog is missing {column!r}: {catalog_source}")

    scenarios_root = Path(scenarios_root)
    staged_df = staged_catalog.copy()
    staged_df["event_id"] = staged_df["event_id"].astype(str)
    accepted_ids = [str(event_id) for event_id in accepted_event_ids]
    selected = staged_df[staged_df["event_id"].isin(accepted_ids)].copy()

    rows: list[dict[str, str]] = []
    missing_from_catalog = sorted(set(accepted_ids) - set(selected["event_id"]))
    for event_id in missing_from_catalog:
        rows.append(
            {
                "event_id": event_id,
                "sfincs_domain_id": "",
                "check": "scenario_catalog_entry",
                "status": "failed",
                "path": str(catalog_source),
                "message": "Accepted dynamic handoff is not staged in scenario_catalog.csv.",
            }
        )

    required_files = inland_required_staged_files(config)
    for staged in selected.to_dict("records"):
        event_id = str(staged["event_id"])
        run_root = Path(str(staged["run_root"]))
        if not run_root.is_absolute():
            run_root = scenarios_root / run_root
        domain_id = run_root.name if run_root.parent.name == event_id else ""
        rows.append(
            {
                "event_id": event_id,
                "sfincs_domain_id": domain_id,
                "check": "run_root",
                "status": "passed" if run_root.exists() else "failed",
                "path": str(run_root),
                "message": "",
            }
        )
        for name in required_files:
            path = run_root / name
            rows.append(
                {
                    "event_id": event_id,
                    "sfincs_domain_id": domain_id,
                    "check": f"file:{name}",
                    "status": "passed" if path.exists() and path.stat().st_size > 0 else "failed",
                    "path": str(path),
                    "message": "" if path.exists() else "missing required staged SFINCS input",
                }
            )
    return pd.DataFrame(rows)


def read_wflow_discharge(
    path: str | Path,
    *,
    variable: str = "discharge",
    index_dim: str = "index",
    name_var: str = "name",
) -> pd.DataFrame:
    """Read Wflow-produced discharge for SFINCS source points.

    Expected minimum schema:

    ``discharge(time, index)`` and ``name(index)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {path}")
        data = ds[variable]
        if "time" not in data.dims or index_dim not in data.dims:
            raise ValueError(f"{variable!r} must have dimensions ('time', {index_dim!r}); got {data.dims}")
        frame = data.transpose("time", index_dim).to_pandas()
        if name_var in ds:
            names = ds[name_var].values.astype(str)
        elif name_var in ds.coords:
            names = ds.coords[name_var].values.astype(str)
        else:
            raise KeyError(f"{name_var!r} coordinate/variable not found in {path}")
    frame.columns = names
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="time")
    return frame.sort_index()


def stage_gridded_precipitation(
    sf,
    precip_nc: str | Path,
    *,
    source_name: str = "event_precip",
    cumulative_input: bool = True,
    time_label: str = "right",
    aggregate: bool | str = False,
    buffer_m: float = 30000.0,
    dst_res: float | None = None,
) -> str:
    """Stage rain-on-grid through native ``precipitation.create``."""
    precip_nc = Path(precip_nc)
    if not precip_nc.exists():
        raise FileNotFoundError(precip_nc)
    source = register_raster_or_dataset(sf, source_name, precip_nc, crs="EPSG:4326")

    # Clear stale copied-base pointers before the native component writes them.
    config_set(sf, "precipfile", None)
    config_set(sf, "netamprfile", None)

    kwargs: dict[str, Any] = {"buffer": float(buffer_m)}
    if dst_res is not None:
        kwargs["dst_res"] = float(dst_res)
    sf.precipitation.create(
        precip=source,
        cumulative_input=bool(cumulative_input),
        time_label=str(time_label),
        aggregate=aggregate,
        **kwargs,
    )
    return "sfincs_netampr.nc" if aggregate is False else "sfincs.precip"


def stage_wflow_discharge_points(
    sf: Any,
    discharge_by_name: pd.DataFrame,
    *,
    event_id: str | None = None,
    model_root: str | Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Stage Wflow hydrographs on native HydroMT-SFINCS source points.

    Wflow writes columns keyed by SFINCS ``src.name``. HydroMT-SFINCS expects
    the time-series columns keyed by the native integer source index when
    writing ``sfincs.dis``.
    """
    if sf.discharge_points.nr_points == 0:
        root = "" if model_root is None else f"{Path(model_root)} "
        raise RuntimeError(
            f"{root}has no native SFINCS src points. "
            "Rebuild the coupled base so HydroMT-SFINCS rivers.create_river_inflow writes discharge source locations."
        )

    src = sf.discharge_points.gdf
    src_names = src["name"].astype(str).tolist()
    missing_src = sorted(set(src_names) - set(discharge_by_name.columns))
    if missing_src:
        suffix = "" if event_id is None else f" for {event_id}"
        raise RuntimeError(f"Wflow discharge forcing lacks SFINCS src IDs{suffix}: {missing_src}")

    discharge_df = discharge_by_name[src_names].copy()
    discharge_df.columns = src.index.astype(int)
    sf.discharge_points.create(timeseries=discharge_df, merge=False)
    return src_names, discharge_df


def update_inland_forcing_manifest(
    manifest_path: str | Path,
    *,
    scenario_staging: str,
    run_start,
    run_stop,
    discharge_nc: str | Path,
    source_variable: str,
    direct_rainfall_enabled: bool,
    event_precip_nc: str | Path | None = None,
    prepared_precip: str | Path | None = None,
    netamprfile: str = "",
    initial_condition: dict[str, Any] | None = None,
    dynamic_handoff_acceptance: str | Path | None = None,
) -> dict[str, Any]:
    """Update the notebook-visible manifest after inland native forcing is staged."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    direct_rainfall_enabled = bool(direct_rainfall_enabled)
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "example_notebook": scenario_staging if "c_run_example" in str(scenario_staging) else "",
            "scenario_staging": scenario_staging,
            "run_start": pd.Timestamp(run_start).strftime("%Y-%m-%d %H:%M:%S"),
            "run_stop": pd.Timestamp(run_stop).strftime("%Y-%m-%d %H:%M:%S"),
            "sfincs_run_executed": False,
            "wflow_discharge_forcing": str(discharge_nc),
            "wflow_source_variable": source_variable,
            "direct_rainfall_enabled": direct_rainfall_enabled,
            "direct_rainfall_source": str(event_precip_nc) if direct_rainfall_enabled and event_precip_nc else "",
            "prepared_precip": str(prepared_precip) if prepared_precip else "",
            "netamprfile": netamprfile,
            "sfincs_initial_condition": initial_condition or {},
            "dynamic_handoff_acceptance": "" if dynamic_handoff_acceptance is None else str(dynamic_handoff_acceptance),
            "direct_rainfall_note": (
                "SFINCS netampr rainfall staged from the same event precipitation used by Wflow."
                if direct_rainfall_enabled else "Direct SFINCS rainfall disabled by config."
            ),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def _clean_manifest_value(value):
    if pd.isna(value):
        return None
    return value


def _handoff_event_artifact(handoff_event: dict[str, Any], filename: str):
    event_dir = handoff_event.get("wflow_event_dir")
    if event_dir in (None, ""):
        return None
    return str(Path(str(event_dir)) / filename)
