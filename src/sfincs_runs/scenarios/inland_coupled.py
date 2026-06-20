from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from sfincs_runs.build_base import (
    is_built_sfincs_base,
    is_built_wflow_base,
    validate_built_sfincs_native_physics,
)
from wflow_runs.dynamic_handoff import dynamic_handoff_paths, require_accepted_dynamic_handoff


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
    report.to_csv(scenarios_root / "scenario_build_report.csv", index=False)
    # The cluster array job consumes scenario_catalog.csv under a different PROJECT_ROOT,
    # so store run_root relative to scenarios_root (<event_id>/<sfincs_domain_id>); the
    # run_events --scenario-catalog reader rejoins it onto the cluster's scenarios dir.
    catalog = report[["event_id", "run_root"]].copy()
    catalog["run_root"] = [Path(value).relative_to(scenarios_root).as_posix() for value in catalog["run_root"]]
    catalog.to_csv(scenarios_root / "scenario_catalog.csv", index=False)
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
