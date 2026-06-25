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


def plan(config, paths):
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
                "seed": 0,
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
        "rainfall": {"strategy": "seasonal_window_permutation", "seed": 0, "window_days": 45},
        "soil_moisture": {
            "strategy": "antecedent_to_forcing",
            "reference_forcing": "rainfall",
            "lead_time_hours": 24,
        },
    }


def configure_coastal_dependence_policy(
    config,
    paths,
    *,
    coastal_latitude: float,
    storm_centroid=None,
    ntr_target_rate_per_year: float = 5.0,
    ntr_declustering_hours: float = 120.0,
    cooccurrence_pairing_window_hours: float = 72.0,
    storm_radius_km: float = 350.0,
    min_population_events: int = 20,
) -> dict:
    """Attach the reusable coastal NTR/rainfall dependence policy to config."""
    event_cfg = config.setdefault("event_catalog", {})
    location_root = Path(paths["location_root"])
    location_name = str(paths.get("location_name") or config["project"]["name"])
    duration_hours = int(config.get("collection", {}).get("aorc_sst", {}).get("storm_duration_hours", 72))
    rainfall_stats = Path(paths["aorc_sst_root"]) / location_name / f"{duration_hours}hr-events" / "storm-stats.csv"

    policy = _deep_merge_dict(
        {
            "method": "copula_joint",
            "driver_vector": ["coastal_water_level", "rainfall"],
            "primary_driver": "coastal_water_level",
            "event_rate_per_year": float(ntr_target_rate_per_year),
            "copula_seed": 0,
            "pool_size": 100000,
            "enforce_stress_budget": True,
            "catalog_band_fractions": {
                "mild": 0.05,
                "common": 0.20,
                "significant": 0.20,
                "rare": 0.25,
                "extreme": 0.30,
            },
            "cooccurrence": {
                "target_rate_per_year": float(ntr_target_rate_per_year),
                "condition_on": ["coastal_water_level", "rainfall"],
                "decluster_window_hours": float(ntr_declustering_hours),
                "pairing_window_hours": float(cooccurrence_pairing_window_hours),
            },
            "storm_stratification": {
                "enabled": True,
                "radius_km": float(storm_radius_km),
                "days_before": 2,
                "days_after": 1,
                "cool_season_months": [10, 11, 12, 1, 2, 3, 4],
                "min_population_events": int(min_population_events),
            },
            "marginals": {"coastal_water_level": {"kind": "pot"}, "rainfall": {"kind": "pot"}},
            "driver_records": {
                "coastal_water_level": {
                    "path": _location_relative_path(paths["waterlevel_csv"], location_root),
                    "time_column": "time",
                    "value_column": "value",
                    "transform": "ntr",
                    "latitude": float(coastal_latitude),
                },
                "rainfall": {
                    "path": _location_relative_path(rainfall_stats, location_root),
                    "time_column": "storm_date",
                    "value_column": "mean",
                },
                "soil_moisture": {
                    "path": _location_relative_path(paths["nwm_soil_moisture_csv"], location_root),
                    "time_column": "time",
                    "value_column": "SOILSAT_TOP",
                    "aggregate": "mean",
                },
            },
            "member_libraries": {
                "coastal_water_level": {
                    "from": "records",
                    "index_column": "coastal_peak_m",
                    "decluster_window_hours": float(ntr_declustering_hours),
                    "target_rate_per_year": float(ntr_target_rate_per_year),
                },
                "rainfall": {"from": "member_table"},
            },
        },
        event_cfg.get("dependence", {}) or {},
    )
    policy["event_rate_per_year"] = float(ntr_target_rate_per_year)
    policy["cooccurrence"].update(
        {
            "target_rate_per_year": float(ntr_target_rate_per_year),
            "decluster_window_hours": float(ntr_declustering_hours),
            "pairing_window_hours": float(cooccurrence_pairing_window_hours),
        }
    )
    policy["storm_stratification"].update(
        {"radius_km": float(storm_radius_km), "min_population_events": int(min_population_events)}
    )
    policy["driver_records"]["coastal_water_level"].update(
        {
            "path": _location_relative_path(paths["waterlevel_csv"], location_root),
            "latitude": float(coastal_latitude),
        }
    )
    policy["driver_records"]["rainfall"]["path"] = _location_relative_path(rainfall_stats, location_root)
    policy["driver_records"]["soil_moisture"]["path"] = _location_relative_path(
        paths["nwm_soil_moisture_csv"], location_root
    )
    policy["member_libraries"]["coastal_water_level"].update(
        {
            "decluster_window_hours": float(ntr_declustering_hours),
            "target_rate_per_year": float(ntr_target_rate_per_year),
        }
    )
    if storm_centroid is not None:
        policy["storm_stratification"]["centroid"] = [float(value) for value in storm_centroid]

    event_cfg["dependence"] = policy
    return policy


def configure_coastal_design_event_policy(
    config,
    *,
    target_event_count: int = 500,
    severity_band_fractions: dict | None = None,
    benchmark_return_period_years=(10, 50, 100, 500),
) -> dict:
    """Attach compact coastal design-catalog defaults to config."""
    severity_band_fractions = dict(
        severity_band_fractions
        or {"mild": 0.05, "common": 0.20, "significant": 0.20, "rare": 0.25, "extreme": 0.30}
    )
    severity_bands = [
        {"severity_band": "mild", "rp_min_years": 0.0, "rp_max_years": 2.0},
        {"severity_band": "common", "rp_min_years": 2.0, "rp_max_years": 10.0},
        {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
        {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
        {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
        {"severity_band": "beyond_design", "rp_min_years": 500.0, "rp_max_years": None},
    ]
    event_cfg = config.setdefault("event_catalog", {})
    dependence = event_cfg.setdefault("dependence", {})
    dependence.update(
        {
            "method": "copula_joint",
            "pool_size": 100000,
            "catalog_band_fractions": severity_band_fractions,
        }
    )
    config["events"] = _deep_merge_dict(config.get("events", {}) or {}, {"target_event_count": int(target_event_count)})
    config["sampling"] = _deep_merge_dict(
        {
            "spacing": "log",
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
            "hybrid_splice_quantile": 0.95,
            "candidate_pool_count": 100000,
            "tail_sample_fraction": 0.05,
            "severity_bands": severity_bands,
        },
        config.get("sampling", {}) or {},
    )
    resilience = _deep_merge_dict(
        {
            "compound_pairing": {
                "enabled": True,
                "strategy": "operationally_severe_plausible_dependence",
                "seed": 0,
                "seasonal_window_days": 45,
                "real_event_count": 12,
                "real_event_window_hours": 72,
                "soil_moisture_lead_time_hours": 24,
                "role_fractions": {
                    "high_rainfall_cooccurrence": 0.4,
                    "rainfall_before_coastal": 0.25,
                    "rainfall_after_coastal": 0.25,
                    "wet_soil_high_rainfall": 0.1,
                },
            }
        },
        config.get("resilience_stress_training", {}) or {},
    )
    resilience.update(
        {
            "target_event_count": int(target_event_count),
            "severity_band_fractions": severity_band_fractions,
            "benchmark_return_period_years": list(benchmark_return_period_years),
        }
    )
    config["resilience_stress_training"] = resilience
    config["design_events"] = _deep_merge_dict(
        {
            "pre_event_baseline_hours": 24,
            "event_threshold_fraction": 0.1,
            "event_threshold_min_m": 0.05,
            "min_event_hours": 12,
            "max_event_hours": 168,
            "tide_resolving_half_window_hours": 72,
            "tail_morph_max_factor": 1.3,
            "tail_morph_trigger_quantile": 0.95,
        },
        config.get("design_events", {}) or {},
    )
    config["template_assignment"] = _deep_merge_dict(
        {
            "random_seed": 0,
            "nearest_pool_size": 75,
            "kernel_sigma_scale": 0.5,
            "kernel_sigma_min_m": 0.03,
            "kernel_sigma_max_m": 0.2,
            "reuse_penalty_lambda": 1.0,
            "dominant_peak_ratio_max": 0.9,
        },
        config.get("template_assignment", {}) or {},
    )
    return resilience


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def load_runtime(location_root) -> EventCatalogNotebookRuntime:
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
        wflow_config={"wflow": runtime_config.get("wflow", {})},
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
    stress_training_catalog: pd.DataFrame,
    historical_tail_catalog: pd.DataFrame | None = None,
    selected_catalog_csv: Path | None = None,
    summary_fields: dict | None = None,
) -> dict:
    """Write catalog/replay artifacts after visible scientific selection."""
    catalog_root = runtime.ensure_parent("data/event_catalog/catalog/probability_catalog.csv").parent
    paths = {
        "inland_design_event_catalog_csv": selected_catalog_csv or catalog_root / "inland_design_event_catalog.csv",
        "selected_design_catalog_parquet": catalog_root / "probability_catalog.parquet",
        "selected_design_catalog_csv": catalog_root / "probability_catalog.csv",
        "historical_tail_catalog_csv": catalog_root / "historical_tail_catalog.csv",
        "scenario_catalog_csv": catalog_root / "scenario_catalog.csv",
        "wflow_replay_set_parquet": catalog_root / "wflow_replay_set.parquet",
        "wflow_replay_set_csv": catalog_root / "wflow_replay_set.csv",
        "wflow_scenario_replay_set_csv": catalog_root / "wflow_scenario_replay_set.csv",
    }
    if historical_tail_catalog is None:
        historical_tail_catalog = (
            pd.read_csv(paths["historical_tail_catalog_csv"], dtype={"event_id": str})
            if paths["historical_tail_catalog_csv"].exists()
            else pd.DataFrame()
        )

    event_catalog = _location_relative_member_files(event_catalog, runtime.location_root)
    stress_training_catalog = _location_relative_member_files(stress_training_catalog, runtime.location_root)
    historical_tail_catalog = _location_relative_member_files(historical_tail_catalog, runtime.location_root)
    paths["inland_design_event_catalog_csv"].parent.mkdir(parents=True, exist_ok=True)
    event_catalog.to_csv(paths["inland_design_event_catalog_csv"], index=False)
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

    preview_columns = [
        column
        for column in [
            "event_id",
            "catalog_role",
            "sample_rp_years",
            "severity_band",
            "sampling_weight",
            "probability_weight",
            "rainfall_mm",
            "rainfall_member_id",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "forcing_pairing_policy",
            "event_drivers",
            "streamflow_design_role",
        ]
        if column in event_catalog.columns
    ]
    summary = {
        "event_catalog_rows": len(event_catalog),
        "design_driver": "rainfall + antecedent moisture (discharge = Wflow response)",
        "inland_design_event_catalog_csv": str(paths["inland_design_event_catalog_csv"]),
        "probability_catalog_csv": str(paths["selected_design_catalog_csv"]),
        "scenario_catalog_csv": str(paths["scenario_catalog_csv"]),
        "scenario_catalog_rows": len(scenario_catalog),
        "wflow_scenario_replay_set_csv": str(paths["wflow_scenario_replay_set_csv"]),
    }
    if "rainfall_member_id" in event_catalog:
        summary["rainfall_analog_count"] = int(event_catalog["rainfall_member_id"].nunique())
    summary.update(summary_fields or {})

    return {
        **paths,
        "event_catalog": event_catalog,
        "stress_training_catalog_csv": stress_training_path,
        "stress_training_catalog": stress_training_catalog,
        "scenario_catalog": scenario_catalog,
        "wflow_replay_set": wflow_replay_set,
        "preview": event_catalog[preview_columns].head(8).round(
            {
                "sample_rp_years": 2,
                "sampling_weight": 3,
                "probability_weight": 8,
                "rainfall_mm": 1,
                "rainfall_scale_factor": 3,
            }
        ),
        "summary": pd.Series(summary, name="event_catalog_handoff"),
    }


def _replay_columns(catalog: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "event_id",
            "streamflow_member_id",
            "streamflow_member_file",
            "streamflow_member_time",
            "streamflow_scale_factor",
            "rainfall_member_id",
            "rainfall_member_file",
            "rainfall_member_time",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "soil_moisture_member_file",
            "soil_moisture_member_time",
            "wflow_event_dir",
        ]
        if column in catalog.columns
    ]


def _location_relative_member_files(catalog: pd.DataFrame, location_root: Path) -> pd.DataFrame:
    frame = catalog.copy()
    for column in [c for c in frame.columns if c.endswith("_member_file")]:
        frame[column] = frame[column].map(lambda value: _location_relative_path(value, location_root))
    if "event_reference_time" in frame:
        frame["event_reference_time"] = pd.to_datetime(frame["event_reference_time"], errors="coerce")
    return frame


def _location_relative_path(value, location_root: Path):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NA
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(location_root).as_posix()
    except ValueError:
        return path.as_posix()



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
    write_handoff,
)
from design_events.build_events.selection import (
    attach_antecedent_soil_moisture,
    assign_severity_bands,
    select_training,
)
from design_events.build_events.probability import (
    build_tail,
    build_inland_catalog,
    build_joint_catalog,
)


# Short notebook-facing API.
build_catalog = build_inland_catalog


def build_timeseries(*args, **kwargs):
    from sfincs_runs.scenarios.coastal_realization import build_timeseries as _build_timeseries

    return _build_timeseries(*args, **kwargs)


def write_joint_handoff(*args, **kwargs):
    from sfincs_runs.scenarios.joint_handoff import write_handoff as _write_handoff

    return _write_handoff(*args, **kwargs)
