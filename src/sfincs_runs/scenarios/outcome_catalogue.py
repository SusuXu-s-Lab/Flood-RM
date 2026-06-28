"""Build event-level SFINCS flood outcome catalogues for evaluation notebooks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


front_columns = [
    "event_origin_group", "catalogue_section", "event_origin", "storm_type", "severity_band",
    "event_id", "event_set", "catalog_role", "selection_role", "selection_reason",
    "sample_rp_years", "sampling_scheme", "sampling_region", "sampling_weight", "probability_weight",
    "driver_combo", "forcing_pairing_policy", "infiltration_treatment",
    "catalog_event_reference_time", "run_event_reference_time", "run_start", "run_stop", "run_duration_hours",
    "coastal_source", "coastal_member_id", "coastal_template_peak_time", "coastal_peak_m",
    "coastal_absolute_peak_m", "coastal_analog_id", "coastal_analog_peak_time",
    "coastal_water_level_scale_factor", "snapwave_source", "snapwave_member_id",
    "snapwave_valid_start_time", "snapwave_valid_end_time", "snapwave_pairing_policy",
    "rainfall_source", "rainfall_member_id", "rainfall_member_time", "rainfall_metric_mm", "rainfall_scale_factor",
    "rainfall_pairing_policy", "rainfall_pairing_reference_time", "rainfall_pairing_lag_hours",
    "soil_moisture_included_in_run", "soil_moisture_source", "soil_moisture_member_id", "soil_moisture_member_time",
    "driver_h_magnitude", "driver_p_magnitude", "expected_has_precip", "expected_has_waves",
    "initial_soil_moisture_fraction", "expected_bzs_peak_max_m", "bzs_peak_max_m",
    "peak_incremental_land_depth_m", "peak_incremental_flooded_area_km2",
    "anytime_incremental_flooded_area_km2", "peak_newly_flooded_area_km2",
    "anytime_newly_flooded_area_km2", "longest_incremental_flood_duration_h",
    "mean_incremental_flood_duration_h", "area_incremental_flooded_ge_24h_km2",
    "cumulative_incremental_flooded_area_km2h", "returncode", "duration_min", "map_mb", "n_timesteps",
    "available_time_steps", "last_output_hour", "run_output_dir", "scenario_dir", "map_path",
    "prepared_precip", "rainfall_source_nc", "surge_dataset", "base_model_root",
    "event_catalog_csv", "classification_catalog_csv",
]


@dataclass(frozen=True)
class FloodOutcomeCatalogue:
    """Completed SFINCS runs joined to event drivers, labels, and flood outcomes."""

    catalogue: pd.DataFrame
    catalogue_path: Path
    grid_file_status: pd.DataFrame
    missing_outcome_ids: list[str]
    front_columns: list[str]
    rerun: bool

    @property
    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "catalogue_csv": str(self.catalogue_path),
                "events": int(len(self.catalogue)),
                "rerun": bool(self.rerun),
                "outcome_metrics_built": int(len(self.missing_outcome_ids)),
            },
            name="flood_outcome_catalogue",
        )

    @property
    def event_counts(self) -> pd.DataFrame:
        return (
            self.catalogue.groupby(["event_origin_group", "storm_type", "severity_band"], dropna=False)
            .size()
            .rename("event_count")
            .reset_index()
        )


def build_flood_event_outcome_catalogue(
    *,
    completed_events,
    paths: dict,
    settings: pd.Series | dict,
    health: pd.DataFrame,
    outdir: Path,
    rerun: bool,
    scenario_summary: dict,
    scenario_rows: dict,
    design_rows: dict,
    design_attrs: dict,
    ensure_grid_files: bool = True,
    progress_every: int = 50,
    workers: int | None = None,
) -> FloodOutcomeCatalogue:
    """Assemble the evaluation table used by risk, QA, and ranking plots.

    ``workers`` controls the per-event outcome-metric read parallelism (see
    ``scenario_stats.event_stats_table``): ``None`` -> auto (up to 16 / cpu count),
    ``1`` -> serial.
    """
    outdir = Path(outdir)
    catalogue_path = outdir / "flood_event_outcome_catalogue.csv"
    events = [Path(d) for d in completed_events]
    run_index = _run_index(events, paths["storage_root"])

    grid_file_status = materialize_grid_files(events, paths) if ensure_grid_files else pd.DataFrame()
    outcomes, missing_ids = _outcomes(
        events=events,
        paths=paths,
        settings=settings,
        catalogue_path=catalogue_path,
        run_index=run_index,
        rerun=rerun,
        scenario_summary=scenario_summary,
        scenario_rows=scenario_rows,
        design_rows=design_rows,
        design_attrs=design_attrs,
        progress_every=progress_every,
        workers=workers,
    )

    event_catalog = _event_catalog(paths["design_outputs_root"] / "catalog" / "event_catalog.csv")
    stress_catalog = _stress_catalog(paths["design_outputs_root"] / "catalog" / "resilience_stress_training_catalog.csv")
    manifests = _manifest_frame(events, paths["storage_root"])
    catalogue = _join_catalogue(run_index, event_catalog, stress_catalog, manifests, outcomes, health)
    catalogue = _label_catalogue(catalogue)
    catalogue = _order_catalogue(catalogue)

    catalogue_path.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(catalogue_path, index=False)
    return FloodOutcomeCatalogue(
        catalogue=catalogue,
        catalogue_path=catalogue_path,
        grid_file_status=grid_file_status,
        missing_outcome_ids=missing_ids,
        front_columns=[c for c in front_columns if c in catalogue.columns],
        rerun=rerun,
    )


def materialize_grid_files(event_dirs, paths: dict) -> pd.DataFrame:
    """Place the SFINCS grid topology file beside copied run outputs when needed."""
    stats = _scenario_stats()
    rows = []
    for event_dir in [Path(d) for d in event_dirs]:
        inp = stats.parse_sfincs_inp(event_dir / "sfincs.inp")
        grid_name = inp.get("qtrfile")
        if not grid_name:
            continue
        target = event_dir / str(grid_name)
        if target.exists():
            rows.append({"event_id": event_dir.name, "grid_file": str(target), "status": "present"})
            continue
        source = _grid_source(event_dir, str(grid_name), paths)
        if source is None:
            rows.append({"event_id": event_dir.name, "grid_file": str(target), "status": "missing_source"})
            continue
        _link_or_copy(source, target)
        rows.append(
            {
                "event_id": event_dir.name,
                "grid_file": str(target),
                "source_grid_file": str(source),
                "status": "materialized",
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty and (frame["status"] == "missing_source").any():
        missing = frame.loc[frame["status"] == "missing_source", "event_id"].head(10).tolist()
        raise FileNotFoundError(f"Could not materialize qtrfile for run outputs: {missing}")
    return frame


def _run_index(events: list[Path], storage_root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [d.name for d in events],
            "scenario_dir": [str(d) for d in events],
            "run_output_dir": [str(Path(storage_root) / d.name) for d in events],
        }
    )


def _outcomes(
    *,
    events: list[Path],
    paths: dict,
    settings,
    catalogue_path: Path,
    run_index: pd.DataFrame,
    rerun: bool,
    scenario_summary: dict,
    scenario_rows: dict,
    design_rows: dict,
    design_attrs: dict,
    progress_every: int,
    workers: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    stats = _scenario_stats()
    stats_path = Path(paths["stats_root"]) / "scenario_stats.csv"
    if rerun:
        outcomes = pd.DataFrame()
        missing_ids = run_index["event_id"].tolist()
    elif stats_path.exists():
        outcomes = pd.read_csv(stats_path)
        missing_ids = sorted(set(run_index["event_id"]) - set(outcomes["event_id"].astype(str)))
    elif catalogue_path.exists():
        outcomes = pd.read_csv(catalogue_path)
        missing_ids = sorted(set(run_index["event_id"]) - set(outcomes["event_id"].astype(str)))
    else:
        outcomes = pd.DataFrame()
        missing_ids = run_index["event_id"].tolist()

    if missing_ids:
        mode = "Rebuilding" if rerun else "Building missing"
        missing = set(missing_ids)
        selected = [d for d in events if d.name in missing]
        print(f"{mode} catalogue outcome metrics for {len(selected)} completed events...")
        fresh = stats.event_stats_table(
            selected,
            paths["storage_root"],
            land_threshold_m=settings["land_threshold_m"],
            huthresh_m=settings["huthresh_m"],
            impact_threshold_m=settings["impact_threshold_m"],
            scenario_summary=scenario_summary,
            scenario_rows=scenario_rows,
            design_rows=design_rows,
            design_attrs=design_attrs,
            workers=workers,
        )
        outcomes = pd.concat([outcomes, fresh], ignore_index=True)
        print(f"  {len(fresh)}/{len(selected)} outcome rows")

    return outcomes[outcomes["event_id"].astype(str).isin(set(run_index["event_id"]))], missing_ids


def _event_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path).rename(
        columns={
            "event_reference_time": "catalog_event_reference_time",
            "event_drivers": "driver_combo",
        }
    )
    catalog["event_catalog_csv"] = str(path)
    catalog["rainfall_metric_mm"] = _rainfall_metrics(catalog)
    catalog["driver_h_magnitude"] = pd.to_numeric(catalog.get("coastal_absolute_peak_m"), errors="coerce")
    catalog["driver_p_magnitude"] = pd.to_numeric(catalog["rainfall_metric_mm"], errors="coerce")
    return catalog


def _rainfall_metrics(catalog: pd.DataFrame) -> list[float]:
    cache = {}
    values = []
    for _, row in catalog.iterrows():
        catalog_metric = pd.to_numeric(pd.Series([row.get("rainfall_metric_mm")]), errors="coerce").iloc[0]
        if pd.notna(catalog_metric):
            values.append(float(catalog_metric))
            continue
        rainfall_file = row.get("rainfall_member_file")
        rainfall_time = pd.to_datetime(row.get("rainfall_member_time"), errors="coerce")
        metric = np.nan
        if pd.notna(rainfall_time) and isinstance(rainfall_file, str) and rainfall_file.strip():
            path = Path(rainfall_file)
            if path.exists():
                if path not in cache:
                    cache[path] = pd.read_csv(path)
                table = cache[path]
                matched = table[pd.to_datetime(table["storm_date"], errors="coerce") == rainfall_time]
                if not matched.empty and "mean" in matched:
                    metric = float(matched.iloc[0]["mean"])
        scale = pd.to_numeric(pd.Series([row.get("rainfall_scale_factor", 1.0)]), errors="coerce").iloc[0]
        if pd.notna(metric) and pd.notna(scale):
            metric *= float(scale)
        values.append(metric)
    return values


def _stress_catalog(path: Path) -> pd.DataFrame:
    columns = [
        "event_id", "event_origin", "catalog_role", "storm_type", "event_set",
        "selection_role", "selection_reason", "benchmark_return_period_years",
        "compound_pairing_policy", "compound_pairing_role", "scenario_timing_edge_case",
    ]
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "classification_catalog_csv"])
    available = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in columns if c in available]
    catalog = pd.read_csv(path, usecols=usecols)
    catalog = catalog.rename(columns={c: f"{c}_stress" for c in catalog.columns if c != "event_id"})
    catalog["classification_catalog_csv"] = str(path)
    return catalog


def _manifest_frame(events: list[Path], storage_root: Path) -> pd.DataFrame:
    rows = []
    for event_dir in events:
        manifest_path = Path(storage_root) / event_dir.name / "forcing_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows.append(
            {
                "event_id": event_dir.name,
                "run_event_reference_time": manifest.get("event_reference_time"),
                "run_start": manifest.get("run_start"),
                "run_stop": manifest.get("run_stop"),
                "run_duration_hours": manifest.get("run_duration_hours"),
                "model_t_start": manifest.get("t_start"),
                "model_t_stop": manifest.get("t_stop"),
                "timing_policy": manifest.get("timing_policy"),
                "forcing_variable": manifest.get("forcing_variable"),
                "expected_has_precip": manifest.get("expected_has_precip"),
                "expected_has_waves": manifest.get("expected_has_waves"),
                "initial_soil_moisture_fraction": manifest.get("initial_soil_moisture_fraction"),
                "run_soil_moisture_member_file": manifest.get("soil_moisture_member_file"),
                "run_soil_moisture_member_id": manifest.get("soil_moisture_member_id"),
                "run_soil_moisture_member_time": manifest.get("soil_moisture_member_time"),
                "soil_moisture_included_in_run": _has_soil(manifest),
                "driver_windows": json.dumps(manifest.get("driver_windows", [])),
                "prepared_precip": manifest.get("prepared_precip"),
                "rainfall_source_nc": manifest.get("rainfall_source_nc"),
                "surge_dataset": manifest.get("surge_dataset"),
                "base_model_root": manifest.get("base_model_root"),
            }
        )
    return pd.DataFrame(rows)


def _join_catalogue(
    run_index: pd.DataFrame,
    event_catalog: pd.DataFrame,
    stress_catalog: pd.DataFrame,
    manifests: pd.DataFrame,
    outcomes: pd.DataFrame,
    health: pd.DataFrame,
) -> pd.DataFrame:
    outcome_cols = [
        "event_id", "map_path", "design_scenario", "design_slr_offset_m",
        "expected_bzs_peak_max_m", "bzs_peak_max_m",
        "peak_incremental_land_depth_m", "peak_total_land_depth_m",
        "mean_total_land_depth_m", "flood_volume_m3",
        "peak_incremental_flooded_area_km2",
        "anytime_incremental_flooded_area_km2", "peak_newly_flooded_area_km2",
        "anytime_newly_flooded_area_km2", "longest_incremental_flood_duration_h",
        "mean_incremental_flood_duration_h", "area_incremental_flooded_ge_24h_km2",
        "cumulative_incremental_flooded_area_km2h", "available_time_steps",
        "last_output_hour", "active_cells", "land_t0_cells", "cell_area_m2",
    ]
    health_cols = [
        "event_id", "returncode", "duration_min", "map_mb", "n_timesteps",
        "zs_finite_frac", "zs_max_last_m", "open_error",
    ]
    outcome_cols = [c for c in outcome_cols if c in outcomes.columns]
    health_cols = [c for c in health_cols if c in health.columns]
    return (
        run_index.merge(event_catalog, on="event_id", how="left")
        .merge(stress_catalog, on="event_id", how="left")
        .merge(manifests, on="event_id", how="left")
        .merge(outcomes[outcome_cols], on="event_id", how="left")
        .merge(health[health_cols], on="event_id", how="left")
    )


def _label_catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    catalogue = catalogue.copy()
    label_cols = [
        "event_origin", "catalog_role", "storm_type", "event_set",
        "selection_role", "selection_reason", "benchmark_return_period_years",
        "compound_pairing_policy", "compound_pairing_role", "scenario_timing_edge_case",
    ]
    for col in label_cols:
        stress_col = f"{col}_stress"
        if col not in catalogue and stress_col in catalogue:
            catalogue[col] = catalogue[stress_col]
        elif stress_col in catalogue:
            catalogue[col] = catalogue[col].combine_first(catalogue[stress_col])

    soil_cols = ["soil_moisture_source", "soil_moisture_member_id", "soil_moisture_member_time"]
    for col in soil_cols:
        if col not in catalogue:
            catalogue[col] = ""
        catalogue[col] = catalogue[col].astype("object")
    if "soil_moisture_included_in_run" not in catalogue:
        catalogue["soil_moisture_included_in_run"] = False
    soil_absent = ~catalogue["soil_moisture_included_in_run"].fillna(False).astype(bool)
    catalogue.loc[soil_absent, soil_cols] = "not_staged"

    for col in ["event_origin", "storm_type", "severity_band", "driver_combo"]:
        if col not in catalogue:
            catalogue[col] = "unresolved"
        catalogue[col] = catalogue[col].fillna("unresolved").astype(str)

    origin_group = catalogue["event_origin"].copy()
    origin_group[origin_group.str.contains("historical", case=False, na=False)] = "historical"
    origin_group[origin_group.str.contains("synthetic", case=False, na=False)] = "synthetic"
    origin_group[origin_group.isin(["", "nan", "unresolved"])] = "unresolved"
    catalogue.insert(0, "event_origin_group", origin_group)
    catalogue.insert(
        1,
        "catalogue_section",
        catalogue["event_origin_group"] + " / " + catalogue["storm_type"] + " / " + catalogue["severity_band"],
    )
    return catalogue


def _order_catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    severity_rank = {"mild": 0, "common": 1, "significant": 2, "rare": 3, "extreme": 4}
    ordered = catalogue.copy()
    ordered["_severity_rank"] = ordered["severity_band"].map(severity_rank).fillna(99)
    ordered["_sample_rp_sort"] = pd.to_numeric(ordered.get("sample_rp_years"), errors="coerce")
    ordered["_depth_sort"] = pd.to_numeric(ordered.get("peak_incremental_land_depth_m"), errors="coerce")
    ordered = ordered.sort_values(
        ["event_origin_group", "event_origin", "storm_type", "_severity_rank", "_sample_rp_sort", "_depth_sort", "event_id"],
        ascending=[True, True, True, True, True, False, True],
    ).drop(columns=["_severity_rank", "_sample_rp_sort", "_depth_sort"])
    ordered_front = [c for c in front_columns if c in ordered.columns]
    return ordered[ordered_front + [c for c in ordered.columns if c not in ordered_front]]


def _grid_source(event_dir: Path, grid_name: str, paths: dict) -> Path | None:
    manifest = _read_json(event_dir / "forcing_manifest.json")
    metadata = _read_json(event_dir / "run_metadata.json")
    candidates = []
    for root_key in ["base_model_root", "source_scenario_dir"]:
        root = manifest.get(root_key) or metadata.get(root_key)
        if root:
            candidates.append(Path(root) / grid_name)
    candidates.append(Path(paths["scenarios_root"]) / event_dir.name / grid_name)
    return next((path for path in candidates if path.exists()), None)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _has_soil(manifest: dict) -> bool:
    keys = ["soil_moisture_member_file", "soil_moisture_member_id", "soil_moisture_member_time"]
    return manifest.get("initial_soil_moisture_fraction") is not None or any(
        str(manifest.get(key) or "").strip() for key in keys
    )


def _scenario_stats():
    from sfincs_runs.scenarios import scenario_stats

    return scenario_stats
