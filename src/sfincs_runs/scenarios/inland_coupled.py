from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import xarray as xr
import yaml

from sfincs_runs.build_base import (
    is_built_sfincs_base,
    is_built_wflow_base,
    validate_built_sfincs_native_physics,
)
from sfincs_runs.scenarios.inland_initial_conditions import configure_hydrograph_initial_conditions
from wflow_runs.dynamic_handoff import dynamic_handoff_paths, require_accepted_dynamic_handoff
from wflow_runs.streamflow_realization import wflow_streamflow_gage_overlap


stale_sfincs_outputs = {
    "sfincs_map.nc",
    "sfincs_his.nc",
    "sfincs_rst.nc",
    "sfincs.log",
    "sfincs_log.txt",
}


@dataclass(frozen=True)
class InlandCoupledExamplePlan:
    status: str
    event_id: str | None
    event_reference_time: str | None
    catalog_path: str
    handoff_path: str
    wflow_event_dir: str | None
    wflow_discharge_forcing: str | None
    sfincs_scenario_dir: str | None
    forcing_manifest: str | None
    stage_command: str
    sfincs_dry_run_command: str
    issues: tuple[str, ...]


def stage_inland_coupled_scenarios(
    config,
    paths,
    *,
    catalog_path=None,
    event_ids=None,
    limit=None,
    force=False,
    write_reports=True,
) -> pd.DataFrame:
    """Stage inland Wflow-SFINCS scenario folders from Event Catalog rows."""
    location_root = _location_root(paths)
    catalog_path = _location_path(
        location_root,
        catalog_path or "data/event_catalog/catalog/probability_catalog.csv",
    )
    catalog = pd.read_csv(catalog_path)
    if "event_id" not in catalog:
        raise ValueError(f"Event Catalog is missing event_id: {catalog_path}")
    catalog["event_id"] = catalog["event_id"].astype(str)
    catalog = _select_rows(catalog, event_ids=event_ids, limit=limit)

    handoff = _read_handoff(config, location_root)
    handoff_by_event = {str(row["event_id"]): row for row in handoff.get("events", [])}
    missing = sorted(set(catalog["event_id"]) - set(handoff_by_event))
    if missing:
        raise ValueError("Event Catalog has missing Wflow handoff events: " + ", ".join(missing))
    missing_discharge = _missing_wflow_discharge_forcing(location_root, handoff_by_event, catalog["event_id"])
    if missing_discharge:
        raise FileNotFoundError(
            "Wflow discharge forcing is missing; run Wflow replay before staging SFINCS: "
            + ", ".join(missing_discharge)
        )
    missing_acceptance = _missing_dynamic_wflow_acceptance(config, location_root, catalog["event_id"])
    if missing_acceptance:
        raise RuntimeError(
            "Dynamic Wflow handoff is not accepted; run 04/b_prepare_wflow_dynamic_handoff.ipynb first: "
            + ", ".join(missing_acceptance)
        )

    sfincs_domains = _sfincs_domain_models(config, location_root)
    missing_bases = [domain["base_model_root"] for domain in sfincs_domains if not is_built_sfincs_base(domain["base_model_root"])]
    if missing_bases:
        raise FileNotFoundError(
            "SFINCS base model is not built: "
            + ", ".join(path.as_posix() for path in missing_bases)
        )
    require_native_physics = bool(set(config.get("event_drivers") or []) & {"rainfall", "streamflow", "soil_moisture"})
    for domain in sfincs_domains:
        validate_built_sfincs_native_physics(
            domain["base_model_root"],
            config,
            require_spatial_roughness=require_native_physics,
        )
    scenarios_root = _location_path(location_root, config.get("paths", {}).get("scenarios_root", "data/sfincs/scenarios"))
    scenarios_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, event in catalog.iterrows():
        event_id = str(event["event_id"])
        for domain in sfincs_domains:
            scenario_root = scenarios_root / event_id
            if domain["sfincs_domain_id"] is not None:
                scenario_root = scenario_root / domain["sfincs_domain_id"]
            if scenario_root.exists():
                if not force:
                    raise FileExistsError(f"{scenario_root} exists. Use force=True to replace it.")
                shutil.rmtree(scenario_root)
            shutil.copytree(domain["base_model_root"], scenario_root)
            for name in stale_sfincs_outputs:
                stale = scenario_root / name
                if stale.exists():
                    stale.unlink()
            manifest = _forcing_manifest(
                event.to_dict(),
                handoff,
                handoff_by_event[event_id],
                config,
                domain=domain,
            )
            (scenario_root / "forcing_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "event_id": event_id,
                    "sfincs_domain_id": domain["sfincs_domain_id"],
                    "scenario_status": "written",
                    "run_root": str(scenario_root),
                    "wflow_discharge_forcing": manifest["wflow_discharge_forcing"],
                    "direct_rainfall_enabled": manifest["direct_rainfall_enabled"],
                }
            )

    report = pd.DataFrame(rows)
    if write_reports:
        report.to_csv(scenarios_root / "scenario_build_report.csv", index=False)
        # The cluster array job consumes scenario_catalog.csv under a different PROJECT_ROOT,
        # so store run_root relative to scenarios_root (<event_id>/<sfincs_domain_id>); the
        # run_events --scenario-catalog reader rejoins it onto the cluster's scenarios dir.
        catalog = report[["event_id", "run_root"]].copy()
        catalog["run_root"] = [Path(value).relative_to(scenarios_root).as_posix() for value in catalog["run_root"]]
        catalog.to_csv(scenarios_root / "scenario_catalog.csv", index=False)
    return report


def dynamic_handoff_readiness_table(
    config,
    location_root,
    *,
    catalog_path=None,
    event_ids=None,
    limit=None,
) -> pd.DataFrame:
    """Return event-level readiness for dynamic Wflow-to-SFINCS handoff staging."""
    location_root = Path(location_root)
    if event_ids is None:
        catalog_path = _location_path(
            location_root,
            catalog_path or "data/event_catalog/catalog/scenario_catalog.csv",
        )
        catalog = pd.read_csv(catalog_path)
        if "event_id" not in catalog:
            raise ValueError(f"Event Catalog is missing event_id: {catalog_path}")
        catalog["event_id"] = catalog["event_id"].astype(str)
        event_ids = _select_rows(catalog, limit=limit)["event_id"].tolist()
    elif limit is not None:
        event_ids = list(event_ids)[: int(limit)]

    rows = []
    for event_id in event_ids:
        event_id = str(event_id)
        paths = dynamic_handoff_paths(config, location_root, event_id)
        try:
            accepted = require_accepted_dynamic_handoff(config, location_root, event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "status": "accepted",
                    "sfincs_discharge_forcing": accepted["sfincs_discharge_forcing"],
                    "acceptance": accepted["dynamic_handoff_acceptance"],
                    "issue": "",
                }
            )
        except Exception as exc:
            status = "blocked"
            issue = str(exc)
            compatibility = {}
            try:
                compatibility = wflow_streamflow_gage_overlap(
                    config,
                    location_root,
                    event_id,
                    catalog_path=catalog_path,
                )
                if not compatibility.get("compatible", False):
                    status = "incompatible"
                    issue = str(compatibility.get("message", issue))
            except Exception:
                compatibility = {}
            rows.append(
                {
                    "event_id": event_id,
                    "status": status,
                    "sfincs_discharge_forcing": str(paths["discharge"]),
                    "acceptance": str(paths["acceptance"]),
                    "issue": issue,
                    "streamflow_member_id": compatibility.get("member_id", ""),
                    "streamflow_member_sites": ",".join(compatibility.get("member_sites", [])),
                    "reviewed_gage_overlap": ",".join(compatibility.get("overlap_site_nos", [])),
                    "reviewed_gage_count": compatibility.get("reviewed_site_count", ""),
                }
            )
    return pd.DataFrame(rows)


def accepted_dynamic_handoff_event_ids(readiness: pd.DataFrame) -> list[str]:
    """Return accepted dynamic handoff event IDs from a readiness table."""
    if readiness.empty:
        return []
    return readiness.loc[readiness["status"].eq("accepted"), "event_id"].astype(str).tolist()


def audit_inland_coupled_batch_readiness(
    config,
    paths,
    *,
    catalog_path=None,
    event_ids=None,
    limit=None,
    staged_catalog=None,
) -> pd.DataFrame:
    """Audit whether selected inland coupled scenarios are ready for run_events.

    This is intentionally lightweight: it does not run Wflow or SFINCS. It
    verifies the accepted dynamic handoff contract and the mutable SFINCS files
    expected by the cluster runner.
    """
    location_root = _location_root(paths)
    scenarios_root = _location_path(location_root, config.get("paths", {}).get("scenarios_root", "data/sfincs/scenarios"))
    readiness = dynamic_handoff_readiness_table(
        config,
        location_root,
        catalog_path=catalog_path,
        event_ids=event_ids,
        limit=limit,
    )
    rows = []
    for record in readiness.to_dict("records"):
        event_id = str(record["event_id"])
        handoff_ok = record["status"] == "accepted"
        rows.append(
            {
                "event_id": event_id,
                "sfincs_domain_id": "",
                "check": "dynamic_handoff_acceptance",
                "status": "passed" if handoff_ok else "failed",
                "path": record.get("acceptance", ""),
                "message": "" if handoff_ok else str(record.get("issue", "")),
            }
        )
    accepted_ids = accepted_dynamic_handoff_event_ids(readiness)
    if not accepted_ids:
        return pd.DataFrame(rows)

    # The atomic per-event cluster run stages in-process and passes its scenario
    # report directly (the global scenario_catalog.csv is only written by the
    # notebook batch-staging path, and a full overwrite would race across parallel
    # array tasks). Fall back to the on-disk catalog when no report is supplied.
    if staged_catalog is not None:
        staged_df = staged_catalog.copy()
        catalog_source = "staged scenario report"
    else:
        scenario_catalog = scenarios_root / "scenario_catalog.csv"
        if not scenario_catalog.exists():
            for event_id in accepted_ids:
                rows.append(
                    {
                        "event_id": event_id,
                        "sfincs_domain_id": "",
                        "check": "scenario_catalog",
                        "status": "failed",
                        "path": str(scenario_catalog),
                        "message": "Run 05_create_scenarios.ipynb to stage accepted events.",
                    }
                )
            return pd.DataFrame(rows)
        staged_df = pd.read_csv(scenario_catalog)
        catalog_source = str(scenario_catalog)

    for column in ["event_id", "run_root"]:
        if column not in staged_df:
            raise ValueError(f"Scenario catalog is missing {column!r}: {catalog_source}")
    staged_df["event_id"] = staged_df["event_id"].astype(str)
    selected = staged_df[staged_df["event_id"].isin(accepted_ids)].copy()
    missing_from_catalog = sorted(set(accepted_ids) - set(selected["event_id"]))
    for event_id in missing_from_catalog:
        rows.append(
            {
                "event_id": event_id,
                "sfincs_domain_id": "",
                "check": "scenario_catalog_entry",
                "status": "failed",
                "path": catalog_source,
                "message": "Accepted dynamic handoff is not staged in scenario_catalog.csv.",
            }
        )

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


def stage_inland_coupled_scenario_forcing(
    config,
    paths,
    *,
    scenario_report: pd.DataFrame | None = None,
    catalog_path=None,
    event_ids=None,
    force=False,
    write_reports=True,
) -> pd.DataFrame:
    """Stage native HydroMT-SFINCS rainfall, discharge, and initial conditions.

    ``stage_inland_coupled_scenarios`` creates event/domain folders from the
    reviewed base model. This helper then uses HydroMT-SFINCS components to
    write the mutable event inputs consumed by the cluster runner:
    ``sfincs.dis``, optional ``sfincs_netampr.nc``, and ``sfincs.ini``.
    """
    # hydromt-sfincs maps environment variables onto SFINCS config fields; a
    # shell-level DEBUG=release value is valid for Python tooling but not for
    # SFINCS' integer debug flag.
    os.environ.pop("DEBUG", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/flood-rm-matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from hydromt_sfincs import SfincsModel

    location_root = _location_root(paths)
    if scenario_report is None:
        scenario_report = stage_inland_coupled_scenarios(
            config,
            paths,
            catalog_path=catalog_path,
            event_ids=event_ids,
            force=force,
        )
    if scenario_report.empty:
        raise ValueError("No SFINCS scenarios to stage forcing for.")

    events_root = _location_path(
        location_root,
        config.get("wflow", {}).get("events_root", "data/wflow/events"),
    )
    direct_rainfall_cfg = (config.get("inland_coupling", {}) or {}).get("direct_rainfall", {}) or {}
    stage_direct_rainfall = bool(direct_rainfall_cfg.get("enabled", False))
    rows = []
    for scenario in scenario_report.to_dict("records"):
        event_id = str(scenario["event_id"])
        run_dir = Path(scenario["run_root"])
        acceptance = require_accepted_dynamic_handoff(config, location_root, event_id)
        discharge_nc = Path(str(acceptance["sfincs_discharge_forcing"]))
        if not discharge_nc.exists():
            raise FileNotFoundError(discharge_nc)

        with xr.open_dataset(discharge_nc) as opened:
            discharge_ds = opened.load()
        t_start = pd.Timestamp(discharge_ds["time"].min().values)
        t_stop = pd.Timestamp(discharge_ds["time"].max().values)
        discharge_by_name = discharge_ds["discharge"].transpose("time", "index").to_pandas()
        discharge_by_name.columns = discharge_ds["name"].values.astype(str)

        sf = SfincsModel(root=str(run_dir), mode="r+")
        sf.read()
        sf.config.update(
            {
                "tref": t_start.to_pydatetime(),
                "tstart": t_start.to_pydatetime(),
                "tstop": t_stop.to_pydatetime(),
            }
        )

        prepared_precip = None
        netamprfile = ""
        event_precip_nc = events_root / event_id / "precip.nc"
        if stage_direct_rainfall:
            if not event_precip_nc.exists():
                raise FileNotFoundError(
                    f"Missing SFINCS direct rainfall source for {event_id}: {event_precip_nc}. "
                    "Run 04/b_prepare_wflow_dynamic_handoff.ipynb first."
                )
            prepared_precip = run_dir / "aorc_precip_for_sfincs.nc"
            shutil.copy2(event_precip_nc, prepared_precip)
            sf.data_catalog.from_dict(
                {
                    "event_precip": {
                        "uri": str(prepared_precip),
                        "data_type": "RasterDataset",
                        "driver": {"name": "raster_xarray"},
                        "metadata": {"crs": 4326},
                    }
                }
            )
            sf.config.set("precipfile", None)
            sf.config.set("netamprfile", None)
            sf.precipitation.create(
                precip="event_precip",
                buffer=float(direct_rainfall_cfg.get("buffer_m", 30000.0)),
                cumulative_input=bool(direct_rainfall_cfg.get("cumulative_input", True)),
                time_label=str(direct_rainfall_cfg.get("time_label", "right")),
                aggregate=False,
            )
            sf.precipitation.write()
            netamprfile = "sfincs_netampr.nc"

        if sf.discharge_points.nr_points == 0:
            raise RuntimeError(
                f"{run_dir} has no native SFINCS src points. "
                "Rebuild the coupled base so HydroMT-SFINCS rivers.create_river_inflow writes discharge source locations."
            )
        src = sf.discharge_points.gdf
        src_names = src["name"].astype(str).tolist()
        missing_src = sorted(set(src_names) - set(discharge_by_name.columns))
        if missing_src:
            raise RuntimeError(f"Wflow discharge forcing lacks SFINCS src IDs for {event_id}: {missing_src}")

        discharge_df = discharge_by_name[src_names].copy()
        discharge_df.columns = src.index.astype(int)
        sf.discharge_points.create(timeseries=discharge_df, merge=False)
        initial_condition = configure_hydrograph_initial_conditions(
            sf,
            discharge_by_name[src_names],
            config,
            run_dir=run_dir,
        )
        sf.write()

        manifest_path = run_dir / "forcing_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "scenario_staging": "05_create_scenarios.ipynb",
                "run_start": t_start.strftime("%Y-%m-%d %H:%M:%S"),
                "run_stop": t_stop.strftime("%Y-%m-%d %H:%M:%S"),
                "sfincs_run_executed": False,
                "wflow_discharge_forcing": str(discharge_nc),
                "direct_rainfall_enabled": bool(stage_direct_rainfall),
                "direct_rainfall_source": str(event_precip_nc) if stage_direct_rainfall else "",
                "prepared_precip": str(prepared_precip) if prepared_precip else "",
                "netamprfile": netamprfile,
                "sfincs_initial_condition": initial_condition,
                "dynamic_handoff_acceptance": str(acceptance["dynamic_handoff_acceptance"]),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        rows.append(
            {
                "event_id": event_id,
                "sfincs_domain_id": scenario.get("sfincs_domain_id"),
                "status": "staged",
                "run_root": str(run_dir),
                "n_src": int(sf.discharge_points.nr_points),
                "wflow_discharge_forcing": str(discharge_nc),
                "direct_rainfall_enabled": bool(stage_direct_rainfall),
                "netamprfile": netamprfile,
                "inifile": initial_condition.get("inifile", ""),
            }
        )

    report = pd.DataFrame(rows)
    if write_reports:
        scenarios_root = _location_path(location_root, config.get("paths", {}).get("scenarios_root", "data/sfincs/scenarios"))
        report.to_csv(scenarios_root / "scenario_forcing_report.csv", index=False)
    return report


def plan_inland_coupled_example(
    config,
    paths,
    *,
    catalog_path=None,
    event_id=None,
) -> InlandCoupledExamplePlan:
    """Plan the one-event Wflow-SFINCS smoke path without running models."""
    location_root = _location_root(paths)
    location_name = location_root.name
    catalog_path = _location_path(
        location_root,
        catalog_path or "data/event_catalog/catalog/probability_catalog.csv",
    )
    handoff_path = _location_path(
        location_root,
        config.get("wflow", {}).get("handoff", {}).get("manifest", "data/wflow/domain_set_handoff.yaml"),
    )
    wflow_base_model_root = _location_path(
        location_root,
        config.get("wflow", {}).get("base_model_root", "data/wflow/base"),
    )
    scenarios_root = _location_path(location_root, config.get("paths", {}).get("scenarios_root", "data/sfincs/scenarios"))
    sfincs_domains = _sfincs_domain_models(config, location_root)

    missing = [
        f"{label}: {path}"
        for label, path in {
            "Event Catalog": catalog_path,
            "Wflow-SFINCS handoff manifest": handoff_path,
        }.items()
        if not path.exists()
    ]
    missing_bases = [
        domain["base_model_root"]
        for domain in sfincs_domains
        if not is_built_sfincs_base(domain["base_model_root"])
    ]
    if missing_bases:
        missing.append("SFINCS base model is not built: " + ", ".join(path.as_posix() for path in missing_bases))
    if not is_built_wflow_base(wflow_base_model_root):
        missing.append(f"Wflow base model is not built: {wflow_base_model_root}")
    if missing:
        return _example_plan(
            location_name=location_name,
            status="missing_inputs",
            catalog_path=catalog_path,
            handoff_path=handoff_path,
            issues=missing,
        )

    catalog = pd.read_csv(catalog_path)
    if "event_id" not in catalog:
        raise ValueError(f"Event Catalog is missing event_id: {catalog_path}")
    catalog["event_id"] = catalog["event_id"].astype(str)
    selected = _select_rows(catalog, event_ids=[event_id] if event_id else None, limit=1).iloc[0].to_dict()
    selected_event_id = str(selected["event_id"])

    handoff = yaml.safe_load(handoff_path.read_text(encoding="utf-8")) or {}
    handoff_by_event = {str(row["event_id"]): row for row in handoff.get("events", [])}
    handoff_event = handoff_by_event.get(selected_event_id)
    selected_domain = sfincs_domains[0] if sfincs_domains else {"sfincs_domain_id": None}
    scenario_dir = scenarios_root / selected_event_id
    if selected_domain.get("sfincs_domain_id") is not None:
        scenario_dir = scenario_dir / str(selected_domain["sfincs_domain_id"])
    if not handoff_event:
        return _example_plan(
            location_name=location_name,
            status="missing_handoff_event",
            event_id=selected_event_id,
            event_reference_time=_clean_string(selected.get("event_reference_time")),
            catalog_path=catalog_path,
            handoff_path=handoff_path,
            sfincs_scenario_dir=scenario_dir,
            issues=(f"Wflow-SFINCS handoff manifest has no event_id={selected_event_id}",),
        )

    return _example_plan(
        location_name=location_name,
        status="ready",
        event_id=selected_event_id,
        event_reference_time=_clean_string(selected.get("event_reference_time")),
        catalog_path=catalog_path,
        handoff_path=handoff_path,
        wflow_event_dir=_clean_string(handoff_event.get("wflow_event_dir")),
        wflow_discharge_forcing=_clean_string(handoff_event.get("discharge_forcing")),
        sfincs_scenario_dir=scenario_dir,
        forcing_manifest=scenario_dir / "forcing_manifest.json",
        issues=(),
    )


def _forcing_manifest(event: dict, handoff: dict, handoff_event: dict, config: dict, *, domain: dict | None = None) -> dict:
    coupling = config.get("inland_coupling", {})
    manifest = {
        "event_id": str(event["event_id"]),
        "forcing_mode": coupling.get("forcing_mode", handoff.get("forcing_mode", "dual_fluvial_pluvial")),
        "event_reference_time": _clean(event.get("event_reference_time")),
        "event_origin": _clean(event.get("event_origin")),
        "catalog_role": _clean(event.get("catalog_role")),
        "sampling_scheme": _clean(event.get("sampling_scheme")),
        "event_set": _clean(event.get("event_set")),
        "selection_role": _clean(event.get("selection_role")),
        "selection_reason": _clean(event.get("selection_reason")),
        "severity_band": _clean(event.get("severity_band")),
        "sample_rp_years": _clean(event.get("sample_rp_years")),
        "streamflow_reference_time": coupling.get("streamflow_reference_time", "dominant_streamgage_network_peak"),
        "streamflow_member_id": _clean(event.get("streamflow_member_id")),
        "rainfall_member_id": _clean(event.get("rainfall_member_id")),
        "soil_moisture_member_id": _clean(event.get("soil_moisture_member_id")),
        "probability_weight": _clean(event.get("probability_weight")),
        "wflow_event_dir": _clean(handoff_event.get("wflow_event_dir")),
        "wflow_discharge_forcing": _clean(handoff_event.get("discharge_forcing")),
        "wflow_precip_provenance": _event_artifact(handoff_event, "precip_provenance.json"),
        "wflow_temp_pet_provenance": _event_artifact(handoff_event, "temp_pet_provenance.json"),
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
    if domain is not None and domain["sfincs_domain_id"] is not None:
        manifest.update(
            {
                "sfincs_domain_id": domain["sfincs_domain_id"],
                "sfincs_domain_base_model_root": domain["base_model_root"].as_posix(),
                "sfincs_handoff_source_ids": list(domain.get("handoff_source_ids", [])),
            }
        )
    return manifest


def _event_artifact(handoff_event: dict, filename: str):
    event_dir = handoff_event.get("wflow_event_dir")
    if event_dir in (None, ""):
        return None
    return str(Path(str(event_dir)) / filename)


def _read_handoff(config, location_root: Path) -> dict:
    handoff_path = _location_path(
        location_root,
        config.get("wflow", {}).get("handoff", {}).get("manifest", "data/wflow/domain_set_handoff.yaml"),
    )
    if not handoff_path.exists():
        raise FileNotFoundError(handoff_path)
    return yaml.safe_load(handoff_path.read_text(encoding="utf-8")) or {}


def _select_rows(catalog: pd.DataFrame, *, event_ids=None, limit=None) -> pd.DataFrame:
    out = catalog
    if event_ids:
        wanted = {str(event_id) for event_id in event_ids}
        out = out[out["event_id"].isin(wanted)].copy()
        missing = wanted - set(out["event_id"])
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


def _sfincs_domain_models(config: dict, location_root: Path) -> list[dict]:
    domain_set = config.get("sfincs_domain_set", {})
    manifest_value = domain_set.get("domain_manifest")
    manifest_path = _location_path(location_root, manifest_value) if manifest_value else None
    if bool(domain_set.get("enabled", False)) and manifest_path is not None and manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        domains = []
        for row in manifest.get("domains", []):
            domains.append(
                {
                    "sfincs_domain_id": str(row["sfincs_domain_id"]),
                    "base_model_root": _location_path(location_root, row["base_model_root"]),
                    "handoff_source_ids": tuple(row.get("handoff_source_ids", ())),
                }
            )
        if domains:
            return domains
    return [
        {
            "sfincs_domain_id": None,
            "base_model_root": _location_path(location_root, config.get("paths", {}).get("base_model_root", "data/sfincs/base")),
            "handoff_source_ids": (),
        }
    ]


def _missing_wflow_discharge_forcing(location_root: Path, handoff_by_event: dict, event_ids) -> list[str]:
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
        path = _location_path(location_root, text)
        if not path.exists():
            missing.append(text)
    return sorted(missing)


def _missing_dynamic_wflow_acceptance(config: dict, location_root: Path, event_ids) -> list[str]:
    source = str(((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}).get("source", "")).lower()
    if source != "wflow_dynamic":
        return []
    missing = []
    for event_id in event_ids:
        try:
            require_accepted_dynamic_handoff(config, location_root, str(event_id))
        except Exception as exc:
            paths = dynamic_handoff_paths(config, location_root, str(event_id))
            missing.append(f"{event_id}: {paths['acceptance']} ({exc})")
    return missing


def _location_root(paths) -> Path:
    if paths.get("location_root") is not None:
        return Path(paths["location_root"])
    repo_root = Path(paths.get("repo_root", Path.cwd()))
    location_name = paths.get("location_name")
    if location_name is None:
        raise ValueError("paths must include 'location_root' or 'location_name'")
    return repo_root / "locations" / str(location_name)


def _location_path(location_root: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("locations", location_root.name):
        return location_root.parents[1] / path
    return location_root / path


def _clean(value):
    if pd.isna(value):
        return None
    return value


def _clean_string(value) -> str | None:
    value = _clean(value)
    if value is None:
        return None
    return str(value)


def _example_plan(
    *,
    location_name: str,
    status: str,
    catalog_path: Path,
    handoff_path: Path,
    event_id: str | None = None,
    event_reference_time: str | None = None,
    wflow_event_dir: str | None = None,
    wflow_discharge_forcing: str | None = None,
    sfincs_scenario_dir: Path | None = None,
    forcing_manifest: Path | None = None,
    issues: tuple[str, ...],
) -> InlandCoupledExamplePlan:
    event_flag = f" --event-id {event_id}" if event_id else ""
    stage_command = (
        "stage_inland_coupled_scenarios(runtime_config, "
        '{"location_root": location_root}, '
        f'catalog_path="{catalog_path.as_posix()}"'
        f"{', event_ids=[' + repr(event_id) + ']' if event_id else ', limit=1'}, "
        "force=True)"
    )
    sfincs_dry_run_command = (
        "uv run python -m sfincs_runs.scenarios.run_events "
        f"--config locations/{location_name}/config.yaml"
        f"{event_flag} --dry-run"
    )
    return InlandCoupledExamplePlan(
        status=status,
        event_id=event_id,
        event_reference_time=event_reference_time,
        catalog_path=catalog_path.as_posix(),
        handoff_path=handoff_path.as_posix(),
        wflow_event_dir=wflow_event_dir,
        wflow_discharge_forcing=wflow_discharge_forcing,
        sfincs_scenario_dir=sfincs_scenario_dir.as_posix() if sfincs_scenario_dir else None,
        forcing_manifest=forcing_manifest.as_posix() if forcing_manifest else None,
        stage_command=stage_command,
        sfincs_dry_run_command=sfincs_dry_run_command,
        issues=tuple(issues),
    )
