from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from design_events.utils import build_paths
from study_location import define_location
from wflow_runs.notebook import exists_table


@dataclass(frozen=True)
class EventForcingPlan:
    name: str
    member_path: Path
    pairing_policy: dict


@dataclass(frozen=True)
class EventCatalogPlan:
    study_location: str
    scenario_name: str
    event_summary_csv: Path
    event_members_nc: Path
    event_catalog_csv: Path
    audit_json: Path | None
    forcings: tuple[EventForcingPlan, ...]
    required_forcings: tuple[str, ...]
    required_source_artifacts: tuple[str, ...]
    wave_analog_policy: str

    @property
    def forcing_names(self):
        return tuple(forcing.name for forcing in self.forcings)

    def forcing(self, name):
        for forcing in self.forcings:
            if forcing.name == name:
                return forcing
        raise KeyError(f"forcing is not configured: {name}")

    def summary_rows(self):
        return [
            {"item": "study_location", "value": self.study_location},
            {"item": "scenario_name", "value": self.scenario_name},
            {"item": "event_summary_csv", "value": self.event_summary_csv.as_posix()},
            {"item": "event_catalog_csv", "value": self.event_catalog_csv.as_posix()},
            {"item": "forcings", "value": ", ".join(self.forcing_names)},
            {"item": "wave_analog_policy", "value": self.wave_analog_policy},
        ]


forcing_order = ("rainfall", "streamflow", "soil_moisture")


def build_event_catalog_plan(config, paths):
    event_cfg = config.get("event_catalog", {})
    if "forcing_members" in event_cfg:
        member_paths = event_cfg.get("forcing_members", {})
    elif "event_catalog" not in config:
        member_paths = _member_paths_from_collection(config)
    else:
        member_paths = {}
    pairing = event_cfg.get("pairing", _pairing_from_collection(config))
    forcings = tuple(
        EventForcingPlan(
            name=name,
            member_path=_repo_path(paths, member_paths[name]),
            pairing_policy=dict(pairing.get(name, {})),
        )
        for name in forcing_order
        if member_paths.get(name) is not None
    )
    wave_analog_policy = "same_historical_analog" if config.get("coastal_waves", False) else "not_required"
    required_source_artifacts = ["event_summary", "event_members"]
    required_source_artifacts.extend(f"{forcing.name}_members" for forcing in forcings)
    if config.get("coastal_waves", False):
        required_source_artifacts.append("era5_waves")
    required_forcings = ("coastal", *tuple(forcing.name for forcing in forcings))
    return EventCatalogPlan(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name")),
        scenario_name=str(paths.get("scenario", {}).get("name", "base")),
        event_summary_csv=Path(paths["event_summary_csv"]),
        event_members_nc=Path(paths["event_members_nc"]),
        event_catalog_csv=Path(paths["event_catalog_csv"]),
        audit_json=None if paths.get("event_catalog_audit_json") is None else Path(paths["event_catalog_audit_json"]),
        forcings=forcings,
        required_forcings=required_forcings,
        required_source_artifacts=tuple(required_source_artifacts),
        wave_analog_policy=wave_analog_policy,
    )


def _repo_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"} and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return Path(paths["repo_root"]) / path


def _member_paths_from_collection(config):
    collection = config.get("collection", {})
    paths = {}
    if "aorc_sst" in collection:
        paths["rainfall"] = "data/sources/aorc_sst/rainfall_members.csv"
    if "nwm" in collection:
        paths["soil_moisture"] = "data/sources/nwm/soil_moisture.csv"
    if config.get("flood_setting") == "inland" and "usgs_streamgages" in collection:
        paths["streamflow"] = "data/sources/usgs_streamgages/streamflow_members.csv"
    return paths


def _pairing_from_collection(config):
    if config.get("flood_setting") == "inland":
        return {
            "rainfall": {
                "strategy": "inland_rainfall_pairing_priority",
                "same_storm_when_available": True,
                "fallback_strategy": "seasonal_window_permutation",
                "seed": 42,
                "window_days": 45,
            },
            "streamflow": {
                "strategy": "coherent_streamgage_network_event",
                "active_records_only": True,
                "allow_multiple_frequency_basis_gages": True,
                "design_event_method": "scaled_streamgage_network_analog",
            },
            "soil_moisture": {
                "strategy": "inland_antecedent_moisture_pairing",
                "rainfall_relative_when_coherent": True,
                "fallback_reference": "dominant_streamgage_network_peak",
                "lead_time_hours": 24,
            },
        }
    return {
        "rainfall": {"strategy": "seasonal_window_permutation", "seed": 42, "window_days": 45},
        "soil_moisture": {
            "strategy": "antecedent_to_forcing",
            "reference_forcing": "rainfall",
            "lead_time_hours": 24,
        },
    }


# Notebook runtime helpers

@dataclass(frozen=True)
class EventCatalogNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    runtime_config: dict
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    runtime_paths: dict

    def resolve_location_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.location_root / path

    def ensure_parent(self, value) -> Path:
        path = self.resolve_location_path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_event_catalog_notebook_runtime(location_root) -> EventCatalogNotebookRuntime:
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    definition = define_location(location_root / "config.yaml")
    runtime_config = definition.config
    runtime_paths = build_paths(runtime_config)
    return EventCatalogNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=repo_root,
        runtime_config=runtime_config,
        config=runtime_config,
        grid_config=runtime_config,
        data_sources=runtime_config,
        sfincs_config=runtime_config,
        wflow_config={"wflow": runtime_config["wflow"]},
        runtime_paths=runtime_paths,
    )


def event_catalog_source_inventory(runtime: EventCatalogNotebookRuntime) -> pd.DataFrame:
    forcing_members = runtime.data_sources["event_catalog"]["forcing_members"]
    return exists_table(
        runtime.location_root,
        {
            "reviewed streamgage network": runtime.data_sources["collection"]["usgs_streamgages"]["reviewed_network"],
            "reviewed discharge records": runtime.data_sources["collection"]["usgs_streamgages"]["streamflow_records"]["output"],
            "streamflow members": forcing_members["streamflow"],
            "rainfall members": forcing_members["rainfall"],
            "soil moisture": forcing_members["soil_moisture"],
        },
    )


def scenario_context(runtime: EventCatalogNotebookRuntime) -> dict:
    return {
        "repo_root": runtime.repo_root,
        "location_root": runtime.location_root,
        "location_name": runtime.config["project"]["name"],
        "scenario": {"name": runtime.sfincs_config["scenario_build"]["design_scenario"]},
    }


def materialize_inland_catalog_outputs(
    *,
    runtime: EventCatalogNotebookRuntime,
    event_catalog: pd.DataFrame,
    historical_tail_catalog: pd.DataFrame,
    stress_training_catalog: pd.DataFrame,
) -> dict:
    """Write catalog/replay artifacts after visible scientific selection."""
    catalog_root = runtime.ensure_parent("data/event_catalog/catalog/probability_catalog.csv").parent
    paths = {
        "selected_design_catalog_parquet": catalog_root / "probability_catalog.parquet",
        "selected_design_catalog_csv": catalog_root / "probability_catalog.csv",
        "historical_tail_catalog_csv": catalog_root / "historical_tail_catalog.csv",
        "scenario_catalog_csv": catalog_root / "scenario_catalog.csv",
        "wflow_replay_set_parquet": catalog_root / "wflow_replay_set.parquet",
        "wflow_replay_set_csv": catalog_root / "wflow_replay_set.csv",
        "wflow_scenario_replay_set_csv": catalog_root / "wflow_scenario_replay_set.csv",
    }
    event_catalog.to_parquet(paths["selected_design_catalog_parquet"], index=False)
    event_catalog.to_csv(paths["selected_design_catalog_csv"], index=False)
    historical_tail_catalog.to_csv(paths["historical_tail_catalog_csv"], index=False)

    replay_columns = _replay_columns(event_catalog)
    wflow_replay_set = event_catalog[replay_columns].copy()
    wflow_replay_set.to_parquet(paths["wflow_replay_set_parquet"], index=False)
    wflow_replay_set.to_csv(paths["wflow_replay_set_csv"], index=False)

    stress_training_path = runtime.runtime_paths["resilience_stress_training_catalog_csv"]
    stress_training_path.parent.mkdir(parents=True, exist_ok=True)
    stress_training_catalog.to_csv(stress_training_path, index=False)

    scenario_catalog = pd.concat([stress_training_catalog, historical_tail_catalog], ignore_index=True, sort=False)
    scenario_catalog.to_csv(paths["scenario_catalog_csv"], index=False)
    scenario_catalog[_replay_columns(scenario_catalog)].copy().to_csv(paths["wflow_scenario_replay_set_csv"], index=False)

    return {
        **paths,
        "stress_training_catalog_csv": stress_training_path,
        "stress_training_catalog": stress_training_catalog,
        "scenario_catalog": scenario_catalog,
        "wflow_replay_set": wflow_replay_set,
    }


def _replay_columns(catalog: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "event_id",
            "streamflow_member_id",
            "streamflow_scale_factor",
            "rainfall_member_id",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "wflow_event_dir",
        ]
        if column in catalog.columns
    ]



# Public notebook-forward workflow surface.
from design_events.build_events.catalog import (
    attach_forcing_members,
    build_event_catalog,
    rebuild_forcing_pairing,
    validate_event_catalog,
    write_event_catalog_audit,
)
from design_events.build_events.inland import (
    InlandEventArtifacts,
    build_inland_event_artifacts,
    build_usgs_streamflow_event_members,
    write_wflow_sfincs_handoff_manifest,
)
from design_events.build_events.selection import (
    attach_antecedent_soil_moisture,
    assign_severity_bands,
    select_resilience_stress_training_set,
)
