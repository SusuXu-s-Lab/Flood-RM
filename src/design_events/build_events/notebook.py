from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from design_events.config import build_paths
from study_location import define_location
from wflow_runs.notebook import exists_table


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
