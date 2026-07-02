"""Notebook runtime and Event Catalog materialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from design_events.audit import audit_from_catalog as _audit_from_catalog
from design_events.runtime import build_paths
from study_location import deep_merge as _deep_merge_dict, define_location


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
    event_catalog = runtime_config.setdefault("event_catalog", {})
    event_catalog.setdefault("forcing_members", {})
    event_catalog["forcing_members"].setdefault("rainfall", runtime_paths["aorc_sst_rainfall_members_csv"])
    event_catalog["forcing_members"].setdefault("soil_moisture", runtime_paths["nwm_soil_moisture_csv"])
    if runtime_config.get("flood_setting") == "inland":
        event_catalog["forcing_members"].setdefault(
            "streamflow",
            location_root / "data/sources/usgs_streamgages/streamflow_members.csv",
        )
    usgs = runtime_config.get("collection", {}).get("usgs_streamgages")
    if usgs is not None and not isinstance(usgs.get("streamflow_records", {}), dict):
        usgs["streamflow_records"] = {"output": usgs["streamflow_records"]}
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
    paths = {"rainfall members": forcing_members["rainfall"], "soil moisture": forcing_members["soil_moisture"]}
    usgs = runtime.data_sources.get("collection", {}).get("usgs_streamgages")
    if usgs is not None:
        paths.update(
            {
                "reviewed streamgage network": usgs["reviewed_network"],
                "reviewed discharge records": usgs["streamflow_records"]["output"],
                "streamflow members": forcing_members["streamflow"],
            }
        )
    return pd.DataFrame(
        [
            {"artifact": name, "path": str(path), "exists": path.exists()}
            for name, value in paths.items()
            for path in [runtime.resolve_location_path(value)]
        ]
    )


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
                    "time_column": "rainfall_peak_time",
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
                    "empirical_analog_lag": 1.0,
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
        try:
            historical_tail_catalog = pd.read_csv(
                paths["historical_tail_catalog_csv"], dtype={"event_id": str}
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            historical_tail_catalog = pd.DataFrame()

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

    event_rate = (((runtime.config.get("event_catalog", {}) or {}).get("dependence", {}) or {}).get("event_rate_per_year"))
    audit = _audit_from_catalog(event_catalog, event_rate=(float(event_rate) if event_rate else None), drivers=["rainfall"])
    paths["audit_json"] = catalog_root / "audit.json"
    paths["audit_json"].write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

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


__all__ = [
    "EventCatalogNotebookRuntime",
    "configure_coastal_dependence_policy",
    "configure_coastal_design_event_policy",
    "event_catalog_source_inventory",
    "load_runtime",
    "materialize_inland_catalog_outputs",
]
