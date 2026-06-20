#!/usr/bin/env python3
"""Create a self-contained PowerModelsONM/DynaGrid run bundle zip."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from power.exports.restoration import (  # noqa: E402
    build_powermodels_onm_export,
    materialize_onm_run_bundle,
)

DEFAULT_LOCATION = "marshfield"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts"
DEFAULT_EVENT_START = "2026-01-01T00:00:00+00:00"

SMART_ARTIFACTS = (
    "assets.parquet",
    "control_units.parquet",
    "controllable_switches.parquet",
    "critical_facilities.parquet",
    "critical_load_assignments.parquet",
    "der_inventory.parquet",
    "load_profile_assignments.parquet",
    "switch_bounded_load_blocks.parquet",
    "onm_settings.json",
    "run_manifest.json",
    "validation_report.json",
)

OPTIONAL_EVENT_ARTIFACTS = (
    "asset_states.parquet",
    "telemetry_observations.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a zip with location PowerModelsONM and NRELDynaGrid assets."
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Location name under locations/. Default: marshfield",
    )
    parser.add_argument(
        "--location-dir",
        type=Path,
        default=None,
        help="Explicit location directory. Overrides --location.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip path. Default: artifacts/mfield_onm.zip for marshfield.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Optional staging directory. Defaults to <output stem>_staging beside the zip.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the staging directory after writing the zip.",
    )
    parser.add_argument(
        "--event-id",
        default="smoke_event",
        help="Event id for generated events/<event_id>/draw_<mc_draw> bundles.",
    )
    parser.add_argument(
        "--mc-draw",
        type=int,
        default=0,
        help="Monte Carlo draw id for generated run bundles.",
    )
    parser.add_argument(
        "--event-start",
        default=DEFAULT_EVENT_START,
        help="ISO timestamp for the generated event window.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=72,
        help="Event-window horizon length in hours.",
    )
    parser.add_argument(
        "--pilot-feeder",
        action="append",
        default=None,
        help="Deprecated alias for --subregion-feeder.",
    )
    parser.add_argument(
        "--subregion-feeder",
        action="append",
        default=None,
        help="Feeder id to include in the subregion export. Repeat for multiple feeders.",
    )
    parser.add_argument(
        "--export-scope",
        choices=("subregion", "pilot", "full", "both"),
        default="both",
        help="Which ONM export(s) to include. Default: both. `pilot` is accepted as an alias for subregion.",
    )
    parser.add_argument(
        "--nrel-dynagrid-path",
        type=Path,
        default=None,
        help="Optional local NRELDynaGrid source checkout to include under external/NRELDynaGrid.",
    )
    parser.add_argument(
        "--include-source-artifacts",
        action="store_true",
        help="Include source/provenance parquet artifacts under source_artifacts/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing zip/staging directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    location = str(args.location)
    location_dir = (
        args.location_dir.resolve()
        if args.location_dir is not None
        else (REPO_ROOT / "locations" / location).resolve()
    )
    output_zip = (
        args.output.resolve()
        if args.output is not None
        else (DEFAULT_OUTPUT_DIR / _default_zip_name(location)).resolve()
    )
    staging_dir = (
        args.staging_dir.resolve()
        if args.staging_dir is not None
        else output_zip.with_suffix("").with_name(f"{output_zip.stem}_staging")
    )
    event_start = _parse_event_start(args.event_start)

    _prepare_staging(staging_dir, output_zip, force=args.force)

    data_dir = location_dir / "data"
    power_grid_dir = data_dir / "power_grid"
    smart_dir = data_dir / "static" / "power_grid" / "smart_ds_compat"
    feeder_opendss_dir = power_grid_dir / "derived_opendss"
    asset_registry_dir = power_grid_dir / "asset_registry"
    full_export_src = power_grid_dir / "onm_export"

    _require_dir(smart_dir)
    _require_dir(feeder_opendss_dir)
    _require_dir(asset_registry_dir)
    _require_dir(full_export_src)

    bundle_root = staging_dir / f"{location}_power_models_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    export_scope = "subregion" if args.export_scope == "pilot" else args.export_scope
    subregion_feeders = args.subregion_feeder or args.pilot_feeder or [_default_subregion_feeder(smart_dir)]
    exports = _materialize_exports(
        export_scope=export_scope,
        bundle_root=bundle_root,
        full_export_src=full_export_src,
        feeder_opendss_dir=feeder_opendss_dir,
        smart_dir=smart_dir,
        asset_registry_dir=asset_registry_dir,
        subregion_feeders=subregion_feeders,
        event_id=args.event_id,
        mc_draw=args.mc_draw,
        event_start=event_start,
        horizon_hours=args.horizon_hours,
    )
    primary_export = exports["subregion"] if "subregion" in exports else exports["full"]

    artifact_summary: dict[str, Any] = {"included": False}
    if args.include_source_artifacts:
        artifact_summary = _copy_artifacts(
            smart_dir=smart_dir,
            data_dir=data_dir,
            destination=bundle_root / "source_artifacts",
        )
    _copy_julia_assets(bundle_root)
    dynagrid_summary = _copy_or_describe_dynagrid(args.nrel_dynagrid_path, bundle_root)
    figure_summary = _copy_location_figures(location_dir=location_dir, bundle_root=bundle_root)
    infrastructure_summary = _write_physical_infrastructure_summary(
        bundle_root=bundle_root,
        location=location,
        smart_dir=smart_dir,
        exports=exports,
        primary_export=primary_export,
        subregion_feeders=subregion_feeders,
        figure_summary=figure_summary,
    )

    manifest = _build_manifest(
        location_dir=location_dir,
        location=location,
        output_zip=output_zip,
        subregion_feeders=subregion_feeders,
        exports=exports,
        primary_export=primary_export,
        smart_dir=smart_dir,
        artifact_summary=artifact_summary,
        dynagrid_summary=dynagrid_summary,
        infrastructure_summary=infrastructure_summary,
        figure_summary=figure_summary,
        event_id=args.event_id,
        mc_draw=args.mc_draw,
        event_start=event_start,
        horizon_hours=args.horizon_hours,
    )
    _write_zip(bundle_root, output_zip)
    if not args.keep_staging:
        shutil.rmtree(staging_dir)
    print(json.dumps({"zip": str(output_zip), "manifest": manifest}, indent=2, sort_keys=True))


def _prepare_staging(staging_dir: Path, output_zip: Path, *, force: bool) -> None:
    if staging_dir.exists():
        if not force:
            raise FileExistsError(f"staging directory exists; pass --force: {staging_dir}")
        shutil.rmtree(staging_dir)
    if output_zip.exists():
        if not force:
            raise FileExistsError(f"output zip exists; pass --force: {output_zip}")
        output_zip.unlink()
    staging_dir.mkdir(parents=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)


def _parse_event_start(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _default_zip_name(location: str) -> str:
    if location == "marshfield":
        return "mfield_onm.zip"
    return f"{location}_onm.zip"


def _require_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"required directory not found: {path}")


def _default_subregion_feeder(smart_dir: Path) -> str:
    ders = pd.read_parquet(smart_dir / "der_inventory.parquet")
    live = ders[
        ders["assignment_status"].astype(str).eq("assigned")
        & ders["bus"].notna()
        & ders["genset_kw"].fillna(0).gt(0)
    ]
    if live.empty:
        raise ValueError("no assigned positive-genset DER rows found for default subregion feeder")
    bus = str(live.iloc[0]["bus"])
    if "__" not in bus:
        raise ValueError(f"cannot derive feeder id from DER bus {bus!r}")
    return bus.split("__", 1)[0]


def _materialize_exports(
    *,
    export_scope: str,
    bundle_root: Path,
    full_export_src: Path,
    feeder_opendss_dir: Path,
    smart_dir: Path,
    asset_registry_dir: Path,
    subregion_feeders: list[str],
    event_id: str,
    mc_draw: int,
    event_start: datetime,
    horizon_hours: int,
) -> dict[str, Path]:
    exports: dict[str, Path] = {}
    if export_scope in {"full", "both"}:
        full_name = "onm_export_full" if export_scope == "both" else "onm_export"
        full_export = bundle_root / full_name
        shutil.copytree(full_export_src, full_export)
        materialize_onm_run_bundle(
            export_dir=full_export,
            smart_ds_compat_dir=smart_dir,
            event_id=event_id,
            mc_draw=mc_draw,
            event_start=event_start,
            horizon_hours=horizon_hours,
        )
        exports["full"] = full_export

    if export_scope in {"subregion", "both"}:
        subregion_name = "onm_export_subregion" if export_scope == "both" else "onm_export"
        subregion_export = build_powermodels_onm_export(
            feeder_opendss_dir=feeder_opendss_dir,
            smart_ds_compat_dir=smart_dir,
            output_dir=bundle_root / subregion_name,
            asset_registry_dir=asset_registry_dir,
            feeder_ids=subregion_feeders,
        ).output_dir
        materialize_onm_run_bundle(
            export_dir=subregion_export,
            smart_ds_compat_dir=smart_dir,
            event_id=event_id,
            mc_draw=mc_draw,
            event_start=event_start,
            horizon_hours=horizon_hours,
        )
        exports["subregion"] = subregion_export
    return exports


def _copy_artifacts(*, smart_dir: Path, data_dir: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for name in SMART_ARTIFACTS:
        source = smart_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)
        else:
            missing.append(name)

    optional: list[str] = []
    for name in OPTIONAL_EVENT_ARTIFACTS:
        source = _find_first(data_dir, name)
        if source is not None:
            shutil.copy2(source, destination / name)
            optional.append(name)
    return {"copied": copied, "missing": missing, "optional_event_artifacts": optional}


def _find_first(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def _copy_julia_assets(bundle_root: Path) -> None:
    julia_project = REPO_ROOT / "julia" / "onm" / "Project.toml"
    script_target = bundle_root / "scripts" / "julia"
    script_target.mkdir(parents=True, exist_ok=True)
    if julia_project.exists():
        shutil.copy2(julia_project, script_target / "Project.toml")
    for script_name in ("powermodels_onm_smoke.jl", "dynagrid_smoke.jl"):
        source = REPO_ROOT / "scripts" / "julia" / script_name
        if source.exists():
            shutil.copy2(source, script_target / script_name)


def _copy_or_describe_dynagrid(path: Path | None, bundle_root: Path) -> dict[str, Any]:
    if path is None:
        return {
            "included": False,
            "instructions": "Clone NatLabRockies/NRELDynaGrid at commit dcdd1e8 and set NRELDYNAGRID_PATH.",
        }

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"NRELDynaGrid path not found: {resolved}")
    external = bundle_root / "external"
    external.mkdir(parents=True, exist_ok=True)
    target = external / "NRELDynaGrid"
    shutil.copytree(
        resolved,
        target,
        ignore=shutil.ignore_patterns(".git", ".julia", "Manifest.toml"),
    )
    return {"included": True, "source_path": str(resolved), "bundle_path": str(target.relative_to(bundle_root))}


def _copy_location_figures(*, location_dir: Path, bundle_root: Path) -> dict[str, Any]:
    figures_dir = location_dir / "figures"
    target_dir = bundle_root / "figures"
    copied: list[str] = []
    missing: list[str] = []
    for figure_name in (
        "distribution_net.png",
        "subregion_distribution_net.png",
        "subregion_flood_impacts.png",
    ):
        source = figures_dir / figure_name
        if not source.exists():
            missing.append(str(source))
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / figure_name
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(bundle_root)))
    return {"included": bool(copied), "paths": copied, "missing": missing}


def _build_manifest(
    *,
    location_dir: Path,
    location: str,
    output_zip: Path,
    subregion_feeders: list[str],
    exports: dict[str, Path],
    primary_export: Path,
    smart_dir: Path,
    artifact_summary: dict[str, Any],
    dynagrid_summary: dict[str, Any],
    infrastructure_summary: dict[str, Any],
    figure_summary: dict[str, Any],
    event_id: str,
    mc_draw: int,
    event_start: datetime,
    horizon_hours: int,
) -> dict[str, Any]:
    ders = pd.read_parquet(smart_dir / "der_inventory.parquet")
    dynagrid_path = (
        "external/NRELDynaGrid"
        if dynagrid_summary.get("included")
        else "/path/to/NRELDynaGrid"
    )
    export_summary: dict[str, Any] = {}
    for label, path in exports.items():
        export_manifest = _read_json(path / "manifest.json")
        export_summary[label] = {
            "path": path.name,
            "feeder_count": export_manifest.get("feeder_count"),
            "source_count": export_manifest.get("source_count"),
            "der_export": export_manifest.get("der_export"),
        }
        if label == "subregion":
            export_summary[label]["subregion_feeders"] = subregion_feeders
    primary_export_name = primary_export.name
    return {
        "schema_version": "onm_power_models_bundle.v0.1",
        "created_by": "scripts/onm_export.py",
        "location": location,
        "location_dir": str(location_dir),
        "output_zip": str(output_zip),
        "event": {
            "event_id": event_id,
            "mc_draw": mc_draw,
            "event_start": event_start.isoformat(),
            "horizon_hours": horizon_hours,
        },
        "exports": export_summary,
        "primary_export": primary_export_name,
        "der_inventory": {
            "rows": int(len(ders)),
            "reopt_sized_rows": int((ders["placement_rule"] == "reopt_resilience_sizing").sum()),
            "positive_genset_rows": int(ders["genset_kw"].fillna(0).gt(0).sum()),
            "total_genset_kw": float(ders["genset_kw"].fillna(0).sum()),
        },
        "artifact_summary": artifact_summary,
        "dynagrid": dynagrid_summary,
        "asset_summary": infrastructure_summary,
        "figures": figure_summary,
        "entrypoints": {
            "pmonm_parse_smoke": (
                "julia --project=scripts/julia scripts/julia/powermodels_onm_smoke.jl "
                f"{primary_export_name}/network.dss {primary_export_name}/settings.json powermodels_onm_smoke.json "
                f"--events {primary_export_name}/events/{event_id}/draw_{mc_draw}/events.json"
            ),
            "pmonm_mld_smoke": (
                "julia --project=scripts/julia scripts/julia/powermodels_onm_smoke.jl "
                f"{primary_export_name}/network.dss {primary_export_name}/settings.json powermodels_onm_mld_smoke.json "
                f"--events {primary_export_name}/events/{event_id}/draw_{mc_draw}/events.json --mld"
            ),
            "dynagrid_smoke": (
                f"NRELDYNAGRID_PATH={dynagrid_path} julia --project=scripts/julia "
                f"scripts/julia/dynagrid_smoke.jl {primary_export_name}/network.dss "
                f"{primary_export_name}/settings.json dynagrid_smoke.json "
                f"--events {primary_export_name}/events/{event_id}/draw_{mc_draw}/events.json"
            ),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_physical_infrastructure_summary(
    *,
    bundle_root: Path,
    location: str,
    smart_dir: Path,
    exports: dict[str, Path],
    primary_export: Path,
    subregion_feeders: list[str],
    figure_summary: dict[str, Any],
) -> dict[str, Any]:
    summary_path = bundle_root / "asset_summary.txt"
    assets = _read_parquet_if_exists(smart_dir / "assets.parquet")
    control_units = _read_parquet_if_exists(smart_dir / "control_units.parquet")
    switches = _read_parquet_if_exists(smart_dir / "controllable_switches.parquet")
    blocks = _read_parquet_if_exists(smart_dir / "switch_bounded_load_blocks.parquet")
    facilities = _read_parquet_if_exists(smart_dir / "critical_facilities.parquet")
    ders = _read_parquet_if_exists(smart_dir / "der_inventory.parquet")
    load_profiles = _read_parquet_if_exists(smart_dir / "load_profile_assignments.parquet")

    export_rows = []
    for label, export_path in exports.items():
        export_manifest = _read_json(export_path / "manifest.json")
        export_rows.append(
            {
                "label": label,
                "path": export_path.name,
                "feeder_count": export_manifest.get("feeder_count"),
                "source_count": export_manifest.get("source_count"),
                "der_rows": (export_manifest.get("der_export") or {}).get("reopt_sized_rows"),
            }
        )

    lines = [
        f"# {location.replace('_', ' ').title()} Physical Infrastructure Summary",
        "",
        "## What This Bundle Represents",
        "",
        "This bundle is a simulation-ready distribution-grid asset package for PowerModelsONM and NRELDynaGrid integration. It combines synthetic feeder topology, Marshfield critical-facility evidence, controllable-switch synthesis, critical-load profiles, and grid-forming DER sizing artifacts.",
        "",
        "Important caveat: this is a reproducible planning/simulation dataset, not a utility as-built record. Public critical-facility points are matched to synthetic service/load-bus proxies where direct utility service records are unavailable.",
        "",
        "## Network Exports",
        "",
        f"- Primary runnable export: `{primary_export.name}`",
        "- Subregion export means a small feeder subset intended for PMONM MLD and DynaGrid solver smoke runs.",
        "- Full export means the full city-scale network intended for full parse/OpenDSS checks and larger solver studies.",
        "",
        "### Included Export Folders",
        "",
    ]
    for row in export_rows:
        lines.extend(
            [
                f"- {row['label']}: `{row['path']}`",
                f"  - feeders: {row['feeder_count']}",
                f"  - voltage sources: {row['source_count']}",
                f"  - live DER rows: {row['der_rows']}",
            ]
        )
    if "subregion" in exports:
        lines.append(f"- Subregion feeders: {', '.join(subregion_feeders)}")
    if figure_summary.get("included"):
        for figure_path in figure_summary.get("paths", []):
            lines.append(f"- Included figure: `{figure_path}`")
    lines.extend(
        [
        "",
        "## Physical Asset Counts",
        "",
        f"- Asset registry rows: {_len(assets)}",
        f"- Control units/feeders: {_len(control_units)}",
        f"- Controllable switches: {_len(switches)}",
        f"- Switch-bounded load blocks: {_len(blocks)}",
        f"- Critical facilities: {_len(facilities)}",
        f"- Critical load profiles: {_len(load_profiles)}",
        f"- DER inventory rows: {_len(ders)}",
        "",
        ]
    )

    if not assets.empty and "source_asset_table" in assets.columns:
        lines.extend(["## Asset Registry By Source Table", ""])
        for name, count in assets["source_asset_table"].astype(str).value_counts().sort_index().items():
            lines.append(f"- {name}: {int(count)}")
        lines.append("")

    if not facilities.empty:
        lines.extend(["## Critical Facilities", ""])
        for column, label in (
            ("criticality_tier", "Criticality tiers"),
            ("lifeline", "FEMA/community lifelines"),
            ("backup_power_status", "Backup power status"),
        ):
            if column in facilities.columns:
                lines.append(f"### {label}")
                for name, count in facilities[column].astype(str).value_counts().sort_index().items():
                    lines.append(f"- {name}: {int(count)}")
                lines.append("")

    if not ders.empty:
        lines.extend(
            [
                "## DER / Backup Power",
                "",
                f"- REopt-sized or surrogate-sized rows: {_count_eq(ders, 'placement_rule', 'reopt_resilience_sizing')}",
                f"- Positive genset rows: {int(ders['genset_kw'].fillna(0).gt(0).sum()) if 'genset_kw' in ders.columns else 0}",
                f"- Total genset capacity: {float(ders['genset_kw'].fillna(0).sum()) if 'genset_kw' in ders.columns else 0.0:.1f} kW",
                f"- Grid-forming capable rows: {int(ders['gfm_capable'].fillna(False).sum()) if 'gfm_capable' in ders.columns else 0}",
                "",
            ]
        )
        display_cols = [
            column
            for column in ("der_id", "bus", "genset_kw", "critical_load_fraction", "outage_duration_hours")
            if column in ders.columns
        ]
        if display_cols:
            lines.append("### DER Rows")
            for row in ders[display_cols].to_dict(orient="records"):
                der_id = str(row.get("der_id", ""))
                bus = str(row.get("bus", ""))
                genset_kw = row.get("genset_kw", "")
                clf = row.get("critical_load_fraction", "")
                outage = row.get("outage_duration_hours", "")
                lines.append(f"- {der_id}: {genset_kw} kW at `{bus}`; CLF={clf}; outage={outage} h")
            lines.append("")

    if not switches.empty:
        lines.extend(["## Controllable Switches", ""])
        for column, label in (
            ("switch_role", "Switch roles"),
            ("normal_state", "Normal states"),
            ("dispatchable", "Dispatchable flags"),
        ):
            if column in switches.columns:
                lines.append(f"### {label}")
                for name, count in switches[column].astype(str).value_counts().sort_index().items():
                    lines.append(f"- {name}: {int(count)}")
                lines.append("")

    if not blocks.empty:
        load_kw = float(blocks["load_kw"].fillna(0).sum()) if "load_kw" in blocks.columns else 0.0
        lines.extend(
            [
                "## Switch-Bounded Load Blocks",
                "",
                f"- Block count: {_len(blocks)}",
                f"- Aggregate block load: {load_kw:.1f} kW",
                "",
            ]
        )
        if "voltage_source_reachability" in blocks.columns:
            lines.append("### Voltage Source Reachability")
            for name, count in blocks["voltage_source_reachability"].astype(str).value_counts().sort_index().items():
                lines.append(f"- {name}: {int(count)}")
            lines.append("")

    lines.extend(
        [
            "## Files To Run Models",
            "",
            "- Use the primary export folder named above for parser, MLD, and DynaGrid smoke runs.",
            "- The default zip contains `onm_export_full/` and `onm_export_subregion/`.",
            "- Each export folder contains `network.dss`, all redirected DSS files, `profiles/*.csv`, `settings.json`, and `events/<event_id>/draw_<mc_draw>/` runtime sidecars.",
            "- `scripts/julia/Project.toml` is only for creating a matching Julia environment for the included smoke scripts.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": summary_path.name}


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _len(frame: pd.DataFrame) -> int:
    return int(len(frame))


def _count_eq(frame: pd.DataFrame, column: str, value: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].astype(str).eq(value).sum())


def _write_zip(bundle_root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root.parent))


if __name__ == "__main__":
    main()
