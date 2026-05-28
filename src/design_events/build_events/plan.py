from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    member_paths = event_cfg.get("forcing_members", {})
    pairing = event_cfg.get("pairing", {})
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
