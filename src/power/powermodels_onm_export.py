"""Artifact-first PowerModelsONM export adapter for Marshfield."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from power.load_profiles import build_archetype_load_profile
from power.onm_export import render_onm_settings, render_switches_dss


@dataclass(frozen=True)
class PowerModelsOnmExport:
    """Paths emitted by the PowerModelsONM export adapter."""

    output_dir: Path
    network_dss_path: Path
    settings_path: Path
    linecodes_path: Path
    transformers_path: Path
    lines_path: Path
    loads_path: Path
    loadshapes_path: Path
    ders_path: Path
    switches_path: Path
    buscoords_path: Path
    stage_b_metadata_path: Path
    manifest_path: Path


_BUS_FIELDS = ("Bus1", "Bus2", "Bus")
_FAULT_STUDY_SOURCE_IMPEDANCE = "Z1=[0.25, 0.25] Z0=[0.25, 0.25]"


def _prefix_bus_token(token: str, feeder_id: str) -> str:
    stripped = token.strip()
    if not stripped or stripped.startswith(f"{feeder_id}__"):
        return stripped
    if stripped[0] in "([{" or stripped[0] in "\"'":
        return stripped
    return f"{feeder_id}__{stripped}"


def _prefix_bus_value(value: str, feeder_id: str) -> str:
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1]
        return "(" + ", ".join(_prefix_bus_token(part.strip(), feeder_id) for part in inner.split(",")) + ")"
    return _prefix_bus_token(value, feeder_id)


def _prefix_bus_references(line: str, feeder_id: str) -> str:
    for field in _BUS_FIELDS:
        line = re.sub(
            rf"\b{field}=([^\s]+)",
            lambda match: f"{field}={_prefix_bus_value(match.group(1), feeder_id)}",
            line,
            flags=re.IGNORECASE,
        )
    line = re.sub(
        r"\bBuses=\(([^)]*)\)",
        lambda match: "Buses=("
        + ", ".join(_prefix_bus_token(part.strip(), feeder_id) for part in match.group(1).split(","))
        + ")",
        line,
        flags=re.IGNORECASE,
    )
    return line


def _remove_default_enabled_token(line: str) -> str:
    """Drop OpenDSS default enabled tokens that PowerModelsDistribution rejects."""

    return re.sub(r"\s+Enabled=True\b", "", line, flags=re.IGNORECASE)


def _prefixed_component_text(feeder_dir: Path, filename: str, feeder_id: str) -> str:
    path = feeder_dir / filename
    if not path.exists():
        return ""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        rows.append(_remove_default_enabled_token(_prefix_bus_references(text, feeder_id)))
    return "\n".join(rows)


def _format_dss_float(value: float) -> str:
    return f"{value:.10g}"


def _normalize_placeholder_impedance_matrix(match: re.Match[str]) -> str:
    def scale_token(token: re.Match[str]) -> str:
        value = float(token.group(0))
        if abs(value) >= 100.0:
            value /= 1_000_000.0
        return _format_dss_float(value)

    return re.sub(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", scale_token, match.group(0))


def _normalize_linecode_impedance_text(dss_text: str) -> str:
    rows = []
    for line in dss_text.splitlines():
        if re.match(r"\s*new\s+LineCode\.", line, flags=re.IGNORECASE):
            for field in ("RMatrix", "XMatrix"):
                line = re.sub(
                    rf"{field}=\([^)]*\)",
                    _normalize_placeholder_impedance_matrix,
                    line,
                    flags=re.IGNORECASE,
                )
        rows.append(line)
    return "\n".join(rows)


def _drop_named_lines(dss_text: str, line_names: set[str]) -> str:
    rows = []
    for line in dss_text.splitlines():
        match = re.match(r"new\s+Line\.([^\s]+)\s+", line, flags=re.IGNORECASE)
        if match is not None and match.group(1) in line_names:
            continue
        rows.append(line)
    return "\n".join(rows)


def _switches_text_without_disable_commands(switches_path: Path) -> str:
    rows = []
    skip_continuation = False
    for line in switches_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("disable "):
            skip_continuation = False
            continue
        if lowered.startswith("new swtcontrol."):
            skip_continuation = True
            continue
        if skip_continuation and lowered.startswith("~"):
            continue
        skip_continuation = False
        rows.append(line)
    return "\n".join(rows) + "\n"


def _safe_dss_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def _loadshape_assignments(
    *,
    smart_ds_compat_dir: Path,
    asset_registry_dir: Path,
    profiles_dir: Path,
) -> tuple[dict[str, str], str]:
    profile_assignments_path = smart_ds_compat_dir / "load_profile_assignments.parquet"
    if not profile_assignments_path.exists():
        return {}, "! No load_profile_assignments.parquet artifact found.\n"

    assignments = pd.read_parquet(profile_assignments_path)
    assets = pd.read_parquet(smart_ds_compat_dir / "assets.parquet")
    loads = pd.read_csv(asset_registry_dir / "loads.csv")
    asset_bus_by_id = dict(zip(assets["asset_id"].astype(str), assets["bus"].astype(str), strict=False))

    shape_by_load_name: dict[str, str] = {}
    loadshape_lines: list[str] = []
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for row in assignments.itertuples(index=False):
        bus = asset_bus_by_id.get(str(row.load_asset_id))
        if bus is None:
            continue
        shape_name = _safe_dss_name(str(row.loadshape_id))
        archetype = {
            "profile_source": row.profile_source,
            "source_building_type": row.source_building_type,
            "source_geography": row.source_geography,
            "schedule_overlay": json.loads(row.source_provenance).get("schedule_overlay", "business_hours"),
        }
        profile = build_archetype_load_profile(archetype, peak_kw=float(row.peak_kw))
        profile_path = profiles_dir / f"{shape_name}.csv"
        profile_path.write_text("\n".join(f"{value:.6f}" for value in profile) + "\n", encoding="utf-8")
        loadshape_lines.append(
            f"New LoadShape.{shape_name} npts=8760 interval=1 mult=(file=profiles/{shape_name}.csv) UseActual=Yes"
        )
        for load in loads.loc[loads["bus"].astype(str).eq(bus), "load_name"].astype(str):
            shape_by_load_name[load] = shape_name

    return shape_by_load_name, "\n".join(loadshape_lines) + ("\n" if loadshape_lines else "")


def _append_yearly_loadshape(line: str, loadshape_by_load_name: dict[str, str]) -> str:
    match = re.match(r"new\s+Load\.([^\s]+)\s+", line, flags=re.IGNORECASE)
    if match is None:
        return line
    loadshape = loadshape_by_load_name.get(match.group(1))
    if loadshape is None or " yearly=" in line.lower():
        return line
    return f"{line} Yearly={loadshape}"


def _der_generators_text(*, smart_ds_compat_dir: Path) -> tuple[str, dict[str, int]]:
    der_path = smart_ds_compat_dir / "der_inventory.parquet"
    profile_path = smart_ds_compat_dir / "load_profile_assignments.parquet"
    if not der_path.exists() or not profile_path.exists():
        return "! DER inventory or load-profile assignment artifact missing.\n", {"provisional": 0, "reopt_sized": 0}

    ders = pd.read_parquet(der_path)
    profiles = pd.read_parquet(profile_path)
    peak_by_load_asset_id = dict(
        zip(profiles["load_asset_id"].astype(str), profiles["peak_kw"].astype(float), strict=False)
    )

    rows: list[str] = []
    provisional = 0
    reopt_sized = 0
    for der in ders.itertuples(index=False):
        if str(der.assignment_status) != "assigned" or not bool(der.gfm_capable):
            continue
        name = _safe_dss_name(str(der.der_id))
        phases = int(float(der.phases)) if pd.notna(der.phases) else 3
        kv = float(der.nominal_voltage_kv) if pd.notna(der.nominal_voltage_kv) else 0.208
        genset_kw = float(der.genset_kw) if pd.notna(der.genset_kw) else None
        is_reopt_sized = str(getattr(der, "placement_rule", "")) == "reopt_resilience_sizing"
        if not is_reopt_sized:
            provisional += 1
            rows.append(f"! held_out_until_reopt_sized; source der_id={der.der_id}")
            continue
        reopt_sized += 1
        if genset_kw is None or genset_kw <= 0.0:
            rows.append(f"! reopt_sized_no_positive_genset; source der_id={der.der_id}")
            continue
        rows.append(f"! reopt_sized; source der_id={der.der_id}")
        rows.append(
            f"New Generator.{name} Bus1={der.bus} Phases={phases} kV={kv:.6g} "
            f"kW={genset_kw:.6g} pf=1 Model=1"
        )
    if provisional and not reopt_sized:
        rows.insert(0, "! No REopt-sized DER rows available; evidence rows held out of live OpenDSS generators.")
    return "\n".join(rows) + ("\n" if rows else ""), {
        "provisional": provisional,
        "reopt_sized": reopt_sized,
    }


def _prefixed_buscoords_text(feeder_dir: Path, feeder_id: str) -> str:
    path = feeder_dir / "BusCoords.dss"
    if not path.exists():
        return ""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text.lower().startswith("setbusxy "):
            parts = text.split(maxsplit=2)
            if len(parts) == 3:
                coordinates = parts[2].split()
                if len(coordinates) >= 2:
                    rows.append(f"{parts[0]} Bus={feeder_id}__{parts[1]} X={coordinates[0]} Y={coordinates[1]}")
                continue
        if text.lower().startswith("setkvbase "):
            continue
        rows.append(text)
    return "\n".join(rows)


def _split_pmonm_settings(stage_b_settings: dict) -> tuple[dict, dict]:
    """Separate PMONM schema-owned settings from Marshfield Stage B metadata."""

    stage_b_payload = stage_b_settings.get("settings", {}) or {}
    pmonm_settings = {}
    for section in ("settings", "load", "switch"):
        if section in stage_b_payload:
            pmonm_settings[section] = stage_b_payload[section]
    metadata = dict(stage_b_settings)
    return pmonm_settings, metadata


def build_powermodels_onm_export(
    *,
    feeder_opendss_dir: Path,
    smart_ds_compat_dir: Path,
    output_dir: Path,
    asset_registry_dir: Path | None = None,
) -> PowerModelsOnmExport:
    """Build a single OpenDSS entrypoint and companions for PowerModelsONM.

    The adapter operates from canonical artifacts instead of the notebook-only
    in-memory GDM objects. It prefixes per-feeder bus names so cross-feeder ties
    and sectionalizers attach to the same namespace used by Stage B artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    asset_registry_dir = (
        asset_registry_dir
        if asset_registry_dir is not None
        else feeder_opendss_dir.parent.parent / "asset_registry"
    )
    sources = pd.read_csv(asset_registry_dir / "sources.csv")
    feeder_dirs = sorted(path for path in feeder_opendss_dir.iterdir() if path.is_dir())
    if not feeder_dirs:
        raise FileNotFoundError(f"no feeder OpenDSS directories under {feeder_opendss_dir}")

    linecodes_path = output_dir / "LineCodes.dss"
    transformers_path = output_dir / "Transformers.dss"
    lines_path = output_dir / "Lines.dss"
    loads_path = output_dir / "Loads.dss"
    loadshapes_path = output_dir / "LoadShapes.dss"
    ders_path = output_dir / "DERs.dss"
    switches_path = output_dir / "Switches.dss"
    buscoords_path = output_dir / "BusCoords.dss"
    settings_path = output_dir / "settings.json"
    stage_b_metadata_path = output_dir / "stage_b_onm_metadata.json"
    network_dss_path = output_dir / "network.dss"
    manifest_path = output_dir / "manifest.json"

    switches = pd.read_parquet(smart_ds_compat_dir / "controllable_switches.parquet")
    replaced_line_names = set(
        switches.loc[
            switches["opens_existing_line"].fillna(False),
            "associated_line_name",
        ].dropna().astype(str)
    )
    profiles_dir = output_dir / "profiles"
    loadshape_by_load_name, loadshapes_text = _loadshape_assignments(
        smart_ds_compat_dir=smart_ds_compat_dir,
        asset_registry_dir=asset_registry_dir,
        profiles_dir=profiles_dir,
    )
    ders_text, der_export_counts = _der_generators_text(smart_ds_compat_dir=smart_ds_compat_dir)

    for filename, destination in [
        ("LineCodes.dss", linecodes_path),
        ("Transformers.dss", transformers_path),
        ("Lines.dss", lines_path),
    ]:
        chunks = [
            _prefixed_component_text(feeder_dir, filename, feeder_dir.name)
            for feeder_dir in feeder_dirs
        ]
        if filename == "LineCodes.dss":
            chunks = [_normalize_linecode_impedance_text(chunk) for chunk in chunks]
        if filename == "Lines.dss":
            chunks = [_drop_named_lines(chunk, replaced_line_names) for chunk in chunks]
        destination.write_text("\n".join(chunk for chunk in chunks if chunk) + "\n", encoding="utf-8")

    load_chunks = []
    for feeder_dir in feeder_dirs:
        load_text = _prefixed_component_text(feeder_dir, "Loads.dss", feeder_dir.name)
        load_chunks.append(
            "\n".join(
                _append_yearly_loadshape(line, loadshape_by_load_name)
                for line in load_text.splitlines()
            )
        )
    loads_path.write_text("\n".join(chunk for chunk in load_chunks if chunk) + "\n", encoding="utf-8")

    buscoords_path.write_text(
        "\n".join(
            chunk
            for chunk in (
                _prefixed_buscoords_text(feeder_dir, feeder_dir.name)
                for feeder_dir in feeder_dirs
            )
            if chunk
        )
        + "\n",
        encoding="utf-8",
    )
    loadshapes_path.write_text(
        loadshapes_text or "! No critical-load LoadShape assignments materialized.\n",
        encoding="utf-8",
    )
    ders_path.write_text(
        ders_text or "! No assigned grid-forming DER rows available for export.\n",
        encoding="utf-8",
    )
    stage_b_settings = json.loads((smart_ds_compat_dir / "onm_settings.json").read_text(encoding="utf-8"))
    pmonm_settings, stage_b_metadata = _split_pmonm_settings(stage_b_settings)
    settings_path.write_text(json.dumps(pmonm_settings, indent=2, sort_keys=True), encoding="utf-8")
    stage_b_metadata_path.write_text(json.dumps(stage_b_metadata, indent=2, sort_keys=True), encoding="utf-8")
    switches_path.write_text(
        _switches_text_without_disable_commands(smart_ds_compat_dir / "switches.dss"),
        encoding="utf-8",
    )

    first_source = sources.iloc[0]
    network_lines = [
        "Clear",
        (
            "New Circuit.marshfield_onm "
            f"Bus1={first_source['bus']} BasekV={float(first_source['basekv'])} "
            f"pu={float(first_source['pu'])} Angle={float(first_source['angle'])} "
            f"Phases={int(first_source['phases'])} {_FAULT_STUDY_SOURCE_IMPEDANCE}"
        ),
    ]
    for row in sources.iloc[1:].itertuples(index=False):
        network_lines.append(
            f"New Vsource.{row.source_name} Bus1={row.bus} BasekV={float(row.basekv)} "
            f"pu={float(row.pu)} Angle={float(row.angle)} Phases={int(row.phases)} "
            f"{_FAULT_STUDY_SOURCE_IMPEDANCE}"
        )
    network_lines.extend(
        [
            "redirect LineCodes.dss",
            "redirect Transformers.dss",
            "redirect Lines.dss",
            "redirect LoadShapes.dss",
            "redirect Loads.dss",
            "redirect DERs.dss",
            "redirect Switches.dss",
            "Set Voltagebases=[0.20784, 12.4704]",
            "calcv",
            "Solve",
            "redirect BusCoords.dss",
        ]
    )
    network_dss_path.write_text("\n".join(network_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "marshfield_powermodels_onm_export.v0.1",
        "network_dss": str(network_dss_path),
        "settings": str(settings_path),
        "stage_b_onm_metadata": str(stage_b_metadata_path),
        "feeder_count": len(feeder_dirs),
        "source_count": len(sources),
        "gdm_bridge": {
            "package": "gdm",
            "available": True,
            "role": "baseline construction context; export rebuilt from canonical OpenDSS/Parquet artifacts",
        },
        "der_export": {
            "reopt_sized_rows": der_export_counts["reopt_sized"],
            "not_reopt_sized_provisional_rows": der_export_counts["provisional"],
            "provisional_capacity_policy": "critical_load_fraction_times_profile_peak",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return PowerModelsOnmExport(
        output_dir=output_dir,
        network_dss_path=network_dss_path,
        settings_path=settings_path,
        linecodes_path=linecodes_path,
        transformers_path=transformers_path,
        lines_path=lines_path,
        loads_path=loads_path,
        loadshapes_path=loadshapes_path,
        ders_path=ders_path,
        switches_path=switches_path,
        buscoords_path=buscoords_path,
        stage_b_metadata_path=stage_b_metadata_path,
        manifest_path=manifest_path,
    )


def export_powermodels_onm(
    *,
    opendss_root: Path,
    asset_registry_dir: Path,
    blocks: pd.DataFrame,
    switches: pd.DataFrame,
    der_inventory: pd.DataFrame,
    load_profiles: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Notebook-facing PowerModelsONM export from in-memory artifact frames."""

    smart_ds_compat_dir = output_dir.parent / "augmented"
    smart_ds_compat_dir.mkdir(parents=True, exist_ok=True)

    blocks.to_parquet(smart_ds_compat_dir / "switch_bounded_load_blocks.parquet", index=False)
    switches.to_parquet(smart_ds_compat_dir / "controllable_switches.parquet", index=False)
    der_inventory.to_parquet(smart_ds_compat_dir / "der_inventory.parquet", index=False)
    load_profiles.to_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet", index=False)

    (smart_ds_compat_dir / "switches.dss").write_text(
        render_switches_dss(switches),
        encoding="utf-8",
    )
    onm_settings = render_onm_settings(switches)
    onm_settings.setdefault("settings", {})["microgrid"] = _microgrid_settings_from_blocks(blocks)
    (smart_ds_compat_dir / "onm_settings.json").write_text(
        json.dumps(onm_settings, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    export = build_powermodels_onm_export(
        feeder_opendss_dir=opendss_root,
        smart_ds_compat_dir=smart_ds_compat_dir,
        output_dir=output_dir,
        asset_registry_dir=asset_registry_dir,
    )
    return json.loads(export.manifest_path.read_text(encoding="utf-8"))


def _microgrid_settings_from_blocks(blocks: pd.DataFrame) -> dict[str, dict]:
    microgrids: dict[str, dict] = {}
    for row in blocks.itertuples(index=False):
        block_id = str(row.block_id)
        buses_json = getattr(row, "buses_json", "[]")
        try:
            buses = json.loads(buses_json)
        except TypeError:
            buses = []
        microgrids[block_id] = {
            "buses": buses,
            "load_kw": float(getattr(row, "load_kw", 0.0) or 0.0),
            "voltage_source_reachability": str(
                getattr(row, "voltage_source_reachability", "unknown")
            ),
        }
    return microgrids
