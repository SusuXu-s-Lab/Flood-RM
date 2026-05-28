from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SingleUseCasePlan:
    study_location: str
    event_id: str
    selection_reason: str
    design_outputs_root: Path
    base_model_root: Path
    scenarios_dir: Path
    storage_dir: Path
    run_root: Path
    stats_dir: Path
    event_catalog_csv: Path
    required_inputs: tuple[str, ...]
    build_command: list[str]
    dry_run_command: list[str]
    run_command: list[str]
    stats_command: list[str]

    def summary_rows(self):
        return [
            {"item": "study_location", "value": self.study_location},
            {"item": "event_id", "value": self.event_id},
            {"item": "selection_reason", "value": self.selection_reason},
            {"item": "base_model_root", "value": self.base_model_root.as_posix()},
            {"item": "scenarios_dir", "value": self.scenarios_dir.as_posix()},
            {"item": "storage_dir", "value": self.storage_dir.as_posix()},
            {"item": "stats_dir", "value": self.stats_dir.as_posix()},
        ]


@dataclass(frozen=True)
class EventSelection:
    event_id: str
    reason: str


def build_single_use_case_plan(
    config,
    paths,
    *,
    event_id=None,
    reference_date=None,
    base_model_root=None,
):
    event_catalog_csv = _event_catalog_csv(paths)
    selection = _selected_event(event_catalog_csv, event_id=event_id, reference_date=reference_date)
    outputs_root = Path(paths["outputs_root"])
    root = outputs_root / "single_use_case"
    design_outputs_root = Path(paths["design_outputs_root"])
    base_model_root = Path(base_model_root or paths["base_model_root"])
    scenarios_dir = root / "scenarios"
    storage_dir = root / "run_outputs"
    run_root = root / "run_stage"
    stats_dir = root / "stats"

    build_command = [
        "python",
        "-m",
        "sfincs_runs",
        "build_scenarios",
        "--config",
        str(paths.get("location_config_path", "")),
        "--design-outputs",
        str(design_outputs_root),
        "--base-dir",
        str(base_model_root),
        "--scenarios-dir",
        str(scenarios_dir),
        "--event-id",
        selection.event_id,
        "--force",
        "--limit",
        "1",
    ]
    dry_run_command = [
        "python",
        "-m",
        "sfincs_runs",
        "run_scenarios",
        "--config",
        str(paths.get("location_config_path", "")),
        "--scenarios-dir",
        str(scenarios_dir),
        "--storage-dir",
        str(storage_dir),
        "--run-root",
        str(run_root),
        "--event-id",
        selection.event_id,
        "--dry-run",
    ]
    run_command = [
        *dry_run_command[:-1],
        "--force-rerun",
    ]
    stats_command = [
        "python",
        "-m",
        "sfincs_runs",
        "stats",
        "--config",
        str(paths.get("location_config_path", "")),
        "--scenarios-dir",
        str(scenarios_dir),
        "--storage-dir",
        str(storage_dir),
        "--stats-dir",
        str(stats_dir),
        "--event-id",
        selection.event_id,
    ]
    return SingleUseCasePlan(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name")),
        event_id=selection.event_id,
        selection_reason=selection.reason,
        design_outputs_root=design_outputs_root,
        base_model_root=base_model_root,
        scenarios_dir=scenarios_dir,
        storage_dir=storage_dir,
        run_root=run_root,
        stats_dir=stats_dir,
        event_catalog_csv=event_catalog_csv,
        required_inputs=("event_catalog", "event_catalog_audit", "base_model"),
        build_command=build_command,
        dry_run_command=dry_run_command,
        run_command=run_command,
        stats_command=stats_command,
    )


def _event_catalog_csv(paths):
    if paths.get("event_catalog_csv") is not None:
        return Path(paths["event_catalog_csv"])
    return Path(paths["design_outputs_root"]) / "catalog" / "event_catalog.csv"


def _selected_event(event_catalog_csv, *, event_id=None, reference_date=None):
    if event_id is not None:
        return EventSelection(str(event_id), "explicit_event_id")
    catalog = pd.read_csv(event_catalog_csv)
    if catalog.empty:
        raise RuntimeError("Event Catalog is empty")
    reference = _reference_date(reference_date)
    cutoff = reference - pd.DateOffset(years=20)
    historical = _recent_rows(catalog, cutoff, reference)
    if not historical.empty:
        historical = historical[_historical_mask(historical)]
    if not historical.empty:
        return EventSelection(_most_extreme_event_id(historical), "recent_historical_extreme")

    proxy = _recent_rows(catalog, cutoff, reference)
    if not proxy.empty:
        return EventSelection(_most_extreme_event_id(proxy), "recent_template_extreme_proxy")
    return EventSelection(str(catalog.iloc[0]["event_id"]), "first_catalog_event")


def _reference_date(value):
    if value is not None:
        return pd.Timestamp(value).tz_localize(None).normalize()
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _event_time(catalog):
    for column in ["coastal_template_peak_time", "template_peak_time", "event_time", "event_date"]:
        if column in catalog:
            return pd.to_datetime(catalog[column], errors="coerce")
    return pd.Series([pd.NaT] * len(catalog), index=catalog.index)


def _recent_rows(catalog, cutoff, reference):
    event_time = _event_time(catalog)
    return catalog[(event_time >= cutoff) & (event_time <= reference)].copy()


def _historical_mask(catalog):
    if "event_family" not in catalog:
        return pd.Series([False] * len(catalog), index=catalog.index)
    return catalog["event_family"].astype(str).str.contains("historical", case=False, na=False)


def _most_extreme_event_id(catalog):
    ranking_columns = [
        column
        for column in ["sample_rp_years", "coastal_absolute_peak_m", "coastal_peak_m"]
        if column in catalog
    ]
    ranked = catalog.copy()
    if ranking_columns:
        for column in ranking_columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
        ranked = ranked.sort_values(ranking_columns, ascending=[False] * len(ranking_columns))
    return str(ranked.iloc[0]["event_id"])
