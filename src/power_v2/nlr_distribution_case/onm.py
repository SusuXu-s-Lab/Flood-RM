"""PowerModelsONM/DynaGrid export, event bundles, and switch rendering."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .core import slug, write_json
from .profiles import build_archetype_load_profile

onm_settings_schema_version = "stage_b_onm_settings.v0.1"
load_uncertainty_schema_version = "marshfield_load_uncertainty.v0.1"
default_load_uncertainty_band_fraction = 0.20
default_fema_lifelines_horizon_hours = 72
hours_per_year = 8760
onm_events_schema_version = "marshfield_onm_events.v0.1"
_fault_study_source_impedance = "Z1=[0.25, 0.25] Z0=[0.25, 0.25]"


def render_switches_dss(controllable_switches: pd.DataFrame) -> str:
    rows = ["! Synthesized Controllable Switches"]
    for row in controllable_switches.itertuples(index=False):
        name = str(row.opendss_element).split(".", 1)[1]
        action = "Open" if row.initial_state == "open" else "Close"
        normal = "Open" if row.normal_state == "open" else "Close"
        if str(row.switch_role) == "sectionalizing" and bool(row.opens_existing_line):
            linecode = f" Linecode={row.associated_linecode}" if pd.notna(row.associated_linecode) and str(row.associated_linecode) else ""
            units = str(row.associated_units) if pd.notna(row.associated_units) else "m"
            length = max(float(row.associated_length_m) / 2.0, 0.001) if pd.notna(row.associated_length_m) else 1.0
            rows += [
                f"Disable Line.{row.associated_line_name}",
                f"New Line.{name}_seg_a Bus1={row.from_bus} Bus2={name}_a Phases={row.phases}{linecode} Length={length:.6g} units={units}",
                f"New Line.{name} Bus1={name}_a Bus2={name}_b Phases={row.phases} Switch=Yes",
                "~ R1=0.001 X1=0 R0=0.001 X0=0 C1=0 C0=0 Length=0.001 units=km",
                f"New SwtControl.{name} SwitchedObj=Line.{name} SwitchedTerm=1",
                f"~ Action={action} Normal={normal} Lock=No",
                f"New Line.{name}_seg_b Bus1={name}_b Bus2={row.to_bus} Phases={row.phases}{linecode} Length={length:.6g} units={units}",
            ]
        else:
            rows += [
                f"New Line.{name} Bus1={row.from_bus} Bus2={row.to_bus} Phases={row.phases} Switch=Yes",
                "~ R1=0.001 X1=0 R0=0.001 X0=0 C1=0 C0=0 Length=0.001 units=km",
                f"New SwtControl.{name} SwitchedObj=Line.{name} SwitchedTerm=1",
                f"~ Action={action} Normal={normal} Lock=No",
            ]
    return "\n".join(rows) + "\n"


def render_onm_settings(controllable_switches: pd.DataFrame) -> dict[str, Any]:
    switches = {}
    for row in controllable_switches.itertuples(index=False):
        key = str(row.opendss_element).split(".", 1)[1]
        switches[key] = {"dispatchable": "YES" if bool(row.dispatchable) else "NO", "status": "ENABLED" if str(row.status) == "enabled" else "DISABLED", "state": "OPEN" if str(row.normal_state) == "open" else "CLOSED", "switch_role": str(row.switch_role), "switch_id": str(row.switch_id)}
    return {"schema_version": onm_settings_schema_version, "infrastructure_step": "controllable_switch_synthesis", "settings": {"switch": switches}}


def build_load_uncertainty_bounds(nominal_windows: Sequence[Mapping[str, Any]], *, event_id: str, mc_draw: int, band_fraction: float = default_load_uncertainty_band_fraction) -> list[dict[str, Any]]:
    if band_fraction < 0:
        raise ValueError("band_fraction must be non-negative")
    rows: list[dict[str, Any]] = []
    for window in nominal_windows:
        for timestep, nominal_kw in enumerate(window["values"]):
            nominal = float(nominal_kw)
            rows.append({"event_id": event_id, "mc_draw": mc_draw, "timestep": timestep, "load_asset_id": str(window["load_asset_id"]), "cluster_id": str(window["block_id"]), "feeder_id": str(window.get("feeder_id", "")), "nominal_kw": nominal, "lower_kw": nominal * (1 - band_fraction), "upper_kw": nominal * (1 + band_fraction), "band_fraction": band_fraction, "schema_version": load_uncertainty_schema_version})
    return rows


@dataclass(frozen=True)
class EventWindow:
    values: list[float]
    start_hour_of_year: int
    end_hour_of_year: int
    horizon_hours: int
    weather_year: int
    event_start_utc: datetime
    wrapped_across_year_boundary: bool


def _coerce_utc(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)


def hour_of_year(timestamp: datetime) -> int:
    ts = _coerce_utc(timestamp)
    return int((ts - datetime(ts.year, 1, 1, tzinfo=timezone.utc)).total_seconds() // 3600)


def slice_annual_profile_to_event_window(annual_profile: Sequence[float], *, event_start_utc: datetime, weather_year: int, horizon_hours: int = default_fema_lifelines_horizon_hours) -> EventWindow:
    if len(annual_profile) != hours_per_year:
        raise ValueError(f"annual_profile must contain {hours_per_year} hours")
    start = hour_of_year(event_start_utc); end = start + horizon_hours; wrapped = end > hours_per_year
    values = list(annual_profile[start:end]) if not wrapped else list(annual_profile[start:]) + list(annual_profile[: end - hours_per_year])
    return EventWindow(values, start, end, horizon_hours, weather_year, _coerce_utc(event_start_utc), wrapped)


def _profile_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(record.get("source_provenance") or "{}"))
    except json.JSONDecodeError:
        return {}


def _load_asset_to_block_id(blocks: pd.DataFrame, *, sandbox_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if blocks.empty or "buses_json" not in blocks:
        return out
    for row in blocks.itertuples(index=False):
        for bus in json.loads(row.buses_json):
            token = slug(bus)
            out[f"{sandbox_id}:asset:load_buses:{bus}"] = row.block_id
            out[f"{sandbox_id}:asset:loads:{bus}"] = row.block_id
            out[f"{sandbox_id}:asset:load_buses:{token}"] = row.block_id
            out[f"{sandbox_id}:asset:loads:{token}"] = row.block_id
    return out


def _block_demand_summary(nodal_demand: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in nodal_demand:
        values = [float(v) for v in row["values"]]
        entry = out.setdefault(str(row["block_id"]), {"block_id": str(row["block_id"]), "load_count": 0, "peak_window_kw": 0.0, "energy_window_kwh": 0.0})
        entry["load_count"] += 1; entry["peak_window_kw"] += max(values) if values else 0.0; entry["energy_window_kwh"] += sum(values)
    return sorted(out.values(), key=lambda r: r["block_id"])


def build_event_window_bundle(*, event_start: datetime, horizon_hours: int, load_profiles: pd.DataFrame, blocks: pd.DataFrame, sandbox_id: str, uncertainty_band: float = 0.20, event_id: str = "preview_event", mc_draw: int = 0) -> dict[str, Any]:
    start = _coerce_utc(event_start); load_to_block = _load_asset_to_block_id(blocks, sandbox_id=sandbox_id)
    nodal: list[dict[str, Any]] = []; nominal: list[dict[str, Any]] = []; years: set[int] = set()
    for row in load_profiles.to_dict("records"):
        prov = _profile_provenance(row)
        archetype = {"schedule_overlay": prov.get("schedule_overlay", "business_hours"), "profile_source": row.get("profile_source"), "source_building_type": row.get("source_building_type"), "source_geography": row.get("source_geography")}
        year = int(row.get("weather_year") or start.year); years.add(year)
        window = slice_annual_profile_to_event_window(build_archetype_load_profile(archetype, peak_kw=float(row["peak_kw"])), event_start_utc=start, weather_year=year, horizon_hours=horizon_hours)
        load_asset_id = str(row["load_asset_id"]); block_id = load_to_block.get(load_asset_id, "unassigned_block")
        demand = {"load_asset_id": load_asset_id, "loadshape_id": str(row["loadshape_id"]), "block_id": block_id, "feeder_id": str(row.get("feeder_id") or ""), "customer_class": str(row.get("customer_class", "")), "peak_kw": float(row["peak_kw"]), "values": window.values, "start_hour_of_year": window.start_hour_of_year, "wrapped_across_year_boundary": window.wrapped_across_year_boundary}
        nodal.append(demand); nominal.append({"load_asset_id": load_asset_id, "block_id": block_id, "feeder_id": demand["feeder_id"], "values": window.values})
    return {"event_start": start.isoformat(), "event_end": (start + timedelta(hours=horizon_hours)).isoformat(), "horizon_hours": horizon_hours, "timestep_count": horizon_hours, "load_profile_count": len(load_profiles), "block_count": len(_block_demand_summary(nodal)), "uncertainty_band": uncertainty_band, "weather_years": sorted(years), "nodal_demand": nodal, "uncertainty_bands": build_load_uncertainty_bounds(nominal, event_id=event_id, mc_draw=mc_draw, band_fraction=uncertainty_band), "block_demand_summary": _block_demand_summary(nodal)}


@dataclass(frozen=True)
class PowerModelsOnmExport:
    output_dir: Path; network_dss_path: Path; settings_path: Path; linecodes_path: Path; transformers_path: Path; lines_path: Path; loads_path: Path; loadshapes_path: Path; ders_path: Path; switches_path: Path; buscoords_path: Path; stage_b_metadata_path: Path; manifest_path: Path


@dataclass(frozen=True)
class OnmRunBundle:
    bundle_dir: Path; events_path: Path; runtime_args_path: Path; nominal_load_window_path: Path; load_uncertainty_path: Path; block_demand_summary_path: Path; run_manifest_path: Path


def _safe_dss_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def _prefix_bus_refs(text: str, feeder_id: str) -> str:
    def token(value: str) -> str:
        return value if value.startswith(f"{feeder_id}__") or not value or value[0] in "([{\"'" else f"{feeder_id}__{value}"
    def repl(match: re.Match[str]) -> str:
        field, value = match.group(1), match.group(2)
        if value.startswith("(") and value.endswith(")"):
            inner = ", ".join(token(part.strip()) for part in value[1:-1].split(","))
            return f"{field}=({inner})"
        return f"{field}={token(value)}"
    text = re.sub(r"\b(Bus1|Bus2|Bus)=([^\s]+)", repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\bBuses=\(([^)]*)\)", lambda m: "Buses=(" + ", ".join(token(p.strip()) for p in m.group(1).split(",")) + ")", text, flags=re.IGNORECASE)
    return re.sub(r"\s+Enabled=True\b", "", text, flags=re.IGNORECASE)


def _component_text(feeder_dir: Path, filename: str, feeder_id: str) -> str:
    path = feeder_dir / filename
    return "\n".join(_prefix_bus_refs(line.strip(), feeder_id) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else ""


def _drop_named_lines(text: str, names: set[str]) -> str:
    out = []
    for line in text.splitlines():
        m = re.match(r"new\s+Line\.([^\s]+)\s+", line, flags=re.IGNORECASE)
        if m and m.group(1) in names:
            continue
        out.append(line)
    return "\n".join(out)


def _switches_text_without_controls(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("disable "):
            skip = False; continue
        if low.startswith("new swtcontrol."):
            skip = True; continue
        if skip and low.startswith("~"):
            continue
        skip = False; out.append(line)
    return "\n".join(out) + "\n"


def _loadshape_assignments(*, augmented_dir: Path, registry_dir: Path, profiles_dir: Path) -> tuple[dict[str, str], str]:
    path = augmented_dir / "load_profile_assignments.parquet"
    if not path.exists():
        return {}, "! No load_profile_assignments.parquet artifact found.\n"
    assignments = pd.read_parquet(path); assets = pd.read_parquet(augmented_dir / "assets.parquet"); loads = pd.read_csv(registry_dir / "loads.csv")
    bus_by_asset = dict(zip(assets["asset_id"].astype(str), assets["bus"].astype(str), strict=False))
    shape_by_load: dict[str, str] = {}; lines: list[str] = []; profiles_dir.mkdir(parents=True, exist_ok=True)
    for row in assignments.to_dict("records"):
        bus = bus_by_asset.get(str(row["load_asset_id"]))
        if bus is None:
            continue
        shape = _safe_dss_name(str(row["loadshape_id"])); prov = _profile_provenance(row)
        profile = build_archetype_load_profile({"profile_source": row.get("profile_source"), "source_building_type": row.get("source_building_type"), "source_geography": row.get("source_geography"), "schedule_overlay": prov.get("schedule_overlay", "business_hours")}, peak_kw=float(row["peak_kw"]))
        (profiles_dir / f"{shape}.csv").write_text("\n".join(f"{v:.6f}" for v in profile) + "\n", encoding="utf-8")
        lines.append(f"New LoadShape.{shape} npts=8760 interval=1 mult=(file=profiles/{shape}.csv) UseActual=Yes")
        for name in loads.loc[loads["bus"].astype(str).eq(bus), "load_name"].astype(str):
            shape_by_load[name] = shape
    return shape_by_load, "\n".join(lines) + ("\n" if lines else "")


def _append_yearly(line: str, shape_by_load: Mapping[str, str]) -> str:
    m = re.match(r"new\s+Load\.([^\s]+)\s+", line, flags=re.IGNORECASE)
    shape = shape_by_load.get(m.group(1)) if m else None
    return f"{line} Yearly={shape}" if shape and " yearly=" not in line.lower() else line


def _der_generators(augmented_dir: Path, feeder_ids: set[str] | None) -> tuple[str, dict[str, int], dict[str, dict]]:
    path = augmented_dir / "der_inventory.parquet"
    if not path.exists():
        return "! DER inventory missing.\n", {"provisional": 0, "reopt_sized": 0}, {}
    ders = pd.read_parquet(path)
    if feeder_ids and "bus" in ders:
        ders = ders[ders["bus"].astype(str).map(lambda b: b.split("__", 1)[0]).isin(feeder_ids)]
    rows: list[str] = []; settings: dict[str, dict] = {}; provisional = reopt = 0
    for der in ders.to_dict("records"):
        if der.get("assignment_status") != "assigned" or not bool(der.get("gfm_capable")):
            continue
        name = _safe_dss_name(str(der["der_id"])); sized = der.get("placement_rule") == "reopt_resilience_sizing"
        if not sized:
            provisional += 1; rows.append(f"! held_out_until_reopt_sized; source der_id={der['der_id']}"); continue
        reopt += 1; kw = float(der.get("genset_kw") or 0.0)
        if kw <= 0:
            rows.append(f"! reopt_sized_no_positive_genset; source der_id={der['der_id']}"); continue
        rows.append(f"New Generator.{name} Bus1={der['bus']} Phases={int(float(der.get('phases') or 3))} kV={float(der.get('nominal_voltage_kv') or 0.208):.6g} kW={kw:.6g} pf=1 Model=1")
        settings[f"Generator.{name}"] = {"inverter": "GRID_FORMING"}
    return "\n".join(rows) + ("\n" if rows else ""), {"provisional": provisional, "reopt_sized": reopt}, settings


def _buscoords_text(feeder_dir: Path, feeder_id: str) -> str:
    path = feeder_dir / "BusCoords.dss"
    if not path.exists(): return ""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].lower() == "setbusxy": rows.append(f"SetBusXY Bus={feeder_id}__{parts[1]} X={parts[2]} Y={parts[3] if len(parts)>3 else ''}".strip())
    return "\n".join(rows)


def build_powermodels_onm_export(*, feeder_opendss_dir: Path, augmented_dir: Path | None = None, smart_ds_compat_dir: Path | None = None, output_dir: Path, asset_registry_dir: Path | None = None, feeder_ids: Sequence[str] | None = None) -> PowerModelsOnmExport:
    augmented = Path(smart_ds_compat_dir or augmented_dir)  # support both old and new parameter names
    registry = Path(asset_registry_dir or feeder_opendss_dir.parent.parent / "asset_registry")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {str(f) for f in feeder_ids or []}
    feeder_dirs = [p for p in sorted(feeder_opendss_dir.iterdir()) if p.is_dir() and (not selected or p.name in selected)]
    if not feeder_dirs: raise FileNotFoundError(f"no feeder OpenDSS directories under {feeder_opendss_dir}")
    sources = pd.read_csv(registry / "sources.csv")
    if selected: sources = sources[sources["feeder_id"].astype(str).isin(selected)]
    switches = pd.read_parquet(augmented / "controllable_switches.parquet")
    if selected and "feeder_id" in switches: switches = switches[switches["feeder_id"].astype(str).isin(selected)]
    replaced = set(switches.loc[switches["opens_existing_line"].fillna(False), "associated_line_name"].dropna().astype(str))
    paths = {name: output_dir / fname for name, fname in {"linecodes": "LineCodes.dss", "transformers": "Transformers.dss", "lines": "Lines.dss", "loads": "Loads.dss", "loadshapes": "LoadShapes.dss", "ders": "DERs.dss", "switches": "Switches.dss", "buscoords": "BusCoords.dss", "settings": "settings.json", "metadata": "stage_b_onm_metadata.json", "network": "network.dss", "manifest": "manifest.json"}.items()}
    for fname, key in [("LineCodes.dss", "linecodes"), ("Transformers.dss", "transformers"), ("Lines.dss", "lines")]:
        text = "\n".join(_component_text(fd, fname, fd.name) for fd in feeder_dirs)
        if key == "lines": text = _drop_named_lines(text, replaced)
        paths[key].write_text(text + "\n", encoding="utf-8")
    shape_by_load, loadshape_text = _loadshape_assignments(augmented_dir=augmented, registry_dir=registry, profiles_dir=output_dir / "profiles")
    load_text = "\n".join("\n".join(_append_yearly(line, shape_by_load) for line in _component_text(fd, "Loads.dss", fd.name).splitlines()) for fd in feeder_dirs)
    paths["loads"].write_text(load_text + "\n", encoding="utf-8"); paths["loadshapes"].write_text(loadshape_text or "! No LoadShape assignments materialized.\n", encoding="utf-8")
    der_text, der_counts, der_settings = _der_generators(augmented, selected or None); paths["ders"].write_text(der_text or "! No assigned grid-forming DER rows.\n", encoding="utf-8")
    paths["switches"].write_text(_switches_text_without_controls(render_switches_dss(switches)), encoding="utf-8")
    paths["buscoords"].write_text("\n".join(_buscoords_text(fd, fd.name) for fd in feeder_dirs) + "\n", encoding="utf-8")
    stage_b = json.loads((augmented / "onm_settings.json").read_text(encoding="utf-8")) if (augmented / "onm_settings.json").exists() else render_onm_settings(switches)
    pmonm = {k: v for k, v in (stage_b.get("settings") or {}).items() if k in {"settings", "dss", "bus", "load", "switch", "line", "transformer", "generator", "storage", "solar", "voltage_source", "options", "solvers"}}
    if der_settings: pmonm.setdefault("dss", {}).update(der_settings)
    write_json(paths["settings"], pmonm); write_json(paths["metadata"], stage_b)
    first = sources.iloc[0]
    network = ["Clear", f"New Circuit.distribution_onm Bus1={first['bus']} BasekV={float(first['basekv'])} pu={float(first['pu'])} Angle={float(first['angle'])} Phases={int(first['phases'])} {_fault_study_source_impedance}"]
    for row in sources.iloc[1:].itertuples(index=False):
        network.append(f"New Vsource.{row.source_name} Bus1={row.bus} BasekV={float(row.basekv)} pu={float(row.pu)} Angle={float(row.angle)} Phases={int(row.phases)} {_fault_study_source_impedance}")
    network += ["redirect LineCodes.dss", "redirect Transformers.dss", "redirect Lines.dss", "redirect LoadShapes.dss", "redirect Loads.dss", "redirect DERs.dss", "redirect Switches.dss", "Set Voltagebases=[0.20784, 12.4704]", "calcv", "Solve", "redirect BusCoords.dss"]
    paths["network"].write_text("\n".join(network) + "\n", encoding="utf-8")
    man = {"schema_version": "distribution_powermodels_onm_export.v0.1", "network_dss": str(paths["network"]), "settings": str(paths["settings"]), "stage_b_onm_metadata": str(paths["metadata"]), "feeder_count": len(feeder_dirs), "source_count": len(sources), "export_scope": "pilot" if selected else "full", "selected_feeder_ids": sorted(selected), "der_export": {"reopt_sized_rows": der_counts["reopt_sized"], "not_reopt_sized_provisional_rows": der_counts["provisional"]}}
    write_json(paths["manifest"], man)
    return PowerModelsOnmExport(output_dir, paths["network"], paths["settings"], paths["linecodes"], paths["transformers"], paths["lines"], paths["loads"], paths["loadshapes"], paths["ders"], paths["switches"], paths["buscoords"], paths["metadata"], paths["manifest"])


@dataclass(frozen=True)
class AssetStateRow:
    event_id: str; mc_draw: int; timestamp: datetime; asset_id: str; state: str


@dataclass(frozen=True)
class OnmEventsResult:
    events: list[dict[str, Any]]; event_id: str; mc_draw: int; event_start_utc: datetime; schema_version: str = onm_events_schema_version; skipped_asset_ids: list[str] = field(default_factory=list)


def build_onm_events(asset_states: Iterable[AssetStateRow], *, event_id: str, mc_draw: int, asset_to_dss_element: Mapping[str, str], event_start_utc: datetime) -> OnmEventsResult:
    start = _coerce_utc(event_start_utc); by_asset: dict[str, list[AssetStateRow]] = {}
    for row in asset_states:
        if row.event_id == event_id and row.mc_draw == mc_draw:
            by_asset.setdefault(row.asset_id, []).append(row)
    events: list[dict[str, Any]] = []; skipped: list[str] = []
    for aid, rows in by_asset.items():
        trans = _first_failure_transition(sorted(rows, key=lambda r: r.timestamp))
        if trans is None: continue
        dss = asset_to_dss_element.get(aid)
        if dss is None:
            skipped.append(f"missing_dss_mapping_for_{aid}"); continue
        events.append({"timestep": max(1, int((_coerce_utc(trans.timestamp) - start).total_seconds() / 3600.0) + 1), "event_type": "switch", "affected_asset": dss, "event_data": {"dispatchable": "NO", "state": "OPEN", "status": "ENABLED"}})
    return OnmEventsResult(sorted(events, key=lambda e: (e["timestep"], e["affected_asset"])), event_id, mc_draw, start, skipped_asset_ids=skipped)


def _first_failure_transition(rows: list[AssetStateRow]) -> AssetStateRow | None:
    prev = None
    for row in rows:
        if row.state == "failed" and prev != "failed": return row
        prev = row.state
    return None


def build_asset_to_dss_element_map(*, assets: pd.DataFrame, controllable_switches: pd.DataFrame | None = None) -> dict[str, str]:
    classes = {"lines": "line", "line": "line", "switches": "line", "transformers": "transformer", "loads": "load", "load_buses": "load", "generators": "generator"}
    out: dict[str, str] = {}
    for row in assets.to_dict("records") if not assets.empty else []:
        cls = classes.get(str(row.get("source_asset_table", ""))); name = str(row.get("source_asset_name", "")); aid = str(row.get("asset_id", ""))
        if cls and name and name != "nan": out[aid] = f"{cls}.{name}"
    if controllable_switches is not None and not controllable_switches.empty:
        for row in controllable_switches.to_dict("records"):
            if row.get("switch_id") and "." in str(row.get("opendss_element")):
                cls, name = str(row["opendss_element"]).split(".", 1); out[str(row["switch_id"])] = f"{cls.lower()}.{name}"
    return out


def _asset_state_rows_from_frame(frame: pd.DataFrame) -> list[AssetStateRow]:
    return [AssetStateRow(str(r.event_id), int(r.mc_draw), _coerce_utc(pd.Timestamp(r.timestamp).to_pydatetime()), str(r.asset_id), str(r.state)) for r in frame.itertuples(index=False)]


def materialize_onm_run_bundle(*, export_dir: Path, smart_ds_compat_dir: Path, event_id: str, mc_draw: int, event_start: datetime, horizon_hours: int = default_fema_lifelines_horizon_hours, asset_states: pd.DataFrame | Iterable[AssetStateRow] | None = None, asset_to_dss_element: Mapping[str, str] | None = None, uncertainty_band: float = default_load_uncertainty_band_fraction) -> OnmRunBundle:
    bundle_dir = Path(export_dir) / "events" / event_id / f"draw_{int(mc_draw)}"; bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_event_window_bundle(event_start=event_start, horizon_hours=horizon_hours, load_profiles=pd.read_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet"), blocks=pd.read_parquet(smart_ds_compat_dir / "switch_bounded_load_blocks.parquet"), sandbox_id="case", uncertainty_band=uncertainty_band, event_id=event_id, mc_draw=mc_draw)
    rows = [] if asset_states is None else _asset_state_rows_from_frame(asset_states) if isinstance(asset_states, pd.DataFrame) else list(asset_states)
    if asset_to_dss_element is None:
        asset_to_dss_element = build_asset_to_dss_element_map(assets=pd.read_parquet(smart_ds_compat_dir / "assets.parquet"), controllable_switches=pd.read_parquet(smart_ds_compat_dir / "controllable_switches.parquet"))
    events = build_onm_events(rows, event_id=event_id, mc_draw=mc_draw, asset_to_dss_element=asset_to_dss_element, event_start_utc=_coerce_utc(event_start))
    paths = {"events": bundle_dir / "events.json", "runtime_args": bundle_dir / "runtime_args.json", "nominal": bundle_dir / "nominal_load_window.json", "uncertainty": bundle_dir / "load_uncertainty.json", "blocks": bundle_dir / "block_demand_summary.json", "manifest": bundle_dir / "run_manifest.json"}
    write_json(paths["events"], events.events)
    write_json(paths["runtime_args"], {"network": str(Path(export_dir) / "network.dss"), "settings": str(Path(export_dir) / "settings.json"), "events": str(paths["events"]), "output": str(bundle_dir / "powermodels_onm_output.json"), "nprocs": 1, "skip": ["faults", "stability"]})
    write_json(paths["nominal"], {"event_id": event_id, "mc_draw": int(mc_draw), "event_start": bundle["event_start"], "event_end": bundle["event_end"], "horizon_hours": horizon_hours, "timestep_count": horizon_hours, "units": "kW", "loads": bundle["nodal_demand"]})
    write_json(paths["uncertainty"], {"event_id": event_id, "mc_draw": int(mc_draw), "units": "kW", "bounds": bundle["uncertainty_bands"]})
    write_json(paths["blocks"], bundle["block_demand_summary"])
    manifest = {"schema_version": "distribution_onm_run_bundle.v0.1", "event_id": event_id, "mc_draw": int(mc_draw), "event_start": bundle["event_start"], "event_end": bundle["event_end"], "horizon_hours": horizon_hours, "timestep_count": horizon_hours, "network": str(Path(export_dir) / "network.dss"), "settings": str(Path(export_dir) / "settings.json"), "events": str(paths["events"]), "runtime_args": str(paths["runtime_args"]), "nominal_load_window": str(paths["nominal"]), "load_uncertainty": str(paths["uncertainty"]), "block_demand_summary": str(paths["blocks"]), "event_count": len(events.events), "skipped_asset_ids": events.skipped_asset_ids, "uncertainty_band": uncertainty_band, "load_profile_count": bundle["load_profile_count"], "block_count": bundle["block_count"], "weather_years": bundle["weather_years"]}
    write_json(paths["manifest"], manifest)
    return OnmRunBundle(bundle_dir, paths["events"], paths["runtime_args"], paths["nominal"], paths["uncertainty"], paths["blocks"], paths["manifest"])


@dataclass(frozen=True)
class JuliaToolchainResult:
    output_json: Path; command: list[str]; stdout: str; stderr: str; summary: dict[str, Any]


def run_powermodels_onm_smoke(*, network_dss: Path | str, settings_json: Path | str, output_json: Path | str, events_json: Path | str | None = None, solve_mld: bool = False, julia_executable: PathLike[str] | str = "julia", julia_channel: str = "1.10", repo_root: Path | str | None = None, project_dir: Path | str = "julia/onm", script_path: Path | str = "scripts/julia/powermodels_onm_smoke.jl") -> JuliaToolchainResult:
    return _run_julia(script_path=script_path, network_dss=network_dss, settings_json=settings_json, output_json=output_json, events_json=events_json, extra=["--mld"] if solve_mld else [], julia_executable=julia_executable, julia_channel=julia_channel, repo_root=repo_root, project_dir=project_dir)


def run_dynagrid_smoke(*, network_dss: Path | str, settings_json: Path | str, output_json: Path | str, events_json: Path | str | None = None, nrel_dynagrid_path: Path | str | None = None, julia_executable: PathLike[str] | str = "julia", julia_channel: str = "1.10", repo_root: Path | str | None = None, project_dir: Path | str = "julia/onm", script_path: Path | str = "scripts/julia/dynagrid_smoke.jl") -> JuliaToolchainResult:
    env = dict(os.environ, NRELDYNAGRID_PATH=str(nrel_dynagrid_path)) if nrel_dynagrid_path else None
    return _run_julia(script_path=script_path, network_dss=network_dss, settings_json=settings_json, output_json=output_json, events_json=events_json, extra=[], julia_executable=julia_executable, julia_channel=julia_channel, repo_root=repo_root, project_dir=project_dir, env=env)


def _run_julia(*, script_path: Path | str, network_dss: Path | str, settings_json: Path | str, output_json: Path | str, events_json: Path | str | None, extra: list[str], julia_executable: PathLike[str] | str, julia_channel: str, repo_root: Path | str | None, project_dir: Path | str, env: Mapping[str, str] | None = None) -> JuliaToolchainResult:
    root = Path(repo_root or Path.cwd()); out = Path(output_json)
    cmd = [str(julia_executable), f"+{julia_channel}", f"--project={project_dir}", str(script_path), str(network_dss), str(settings_json), str(out)]
    if events_json: cmd += ["--events", str(events_json)]
    cmd += extra
    done = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True, env=dict(env) if env else None)
    return JuliaToolchainResult(out, cmd, done.stdout, done.stderr, json.loads(out.read_text(encoding="utf-8")))
