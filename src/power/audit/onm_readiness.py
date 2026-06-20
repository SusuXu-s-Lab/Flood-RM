"""Readiness gates for Marshfield PowerModelsONM/DNMG artifacts."""

from __future__ import annotations

import os
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def summarize_der_export_readiness(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize whether exported DER rows are REopt-sized for DNMG studies."""

    der_export = manifest.get("der_export", {}) or {}
    reopt_sized_rows = int(der_export.get("reopt_sized_rows", 0) or 0)
    provisional_rows = int(der_export.get("not_reopt_sized_provisional_rows", 0) or 0)
    blockers = []
    if provisional_rows:
        blockers.append(
            f"{provisional_rows} DER rows are provisional critical-load proxies; "
            "run Layer 2 REopt sizing before DNMG studies."
        )

    return {
        "passed": not blockers,
        "reopt_sized_rows": reopt_sized_rows,
        "provisional_rows": provisional_rows,
        "blockers": blockers,
    }


def summarize_powermodels_onm_smoke_readiness(smoke: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize PowerModelsONM parser and strict-settings validation evidence."""

    parser_passed = smoke.get("status") == "ok" and all(
        int(smoke.get(field, 0) or 0) > 0
        for field in ("bus_count", "load_count", "switch_count")
    )
    settings_validation = smoke.get("settings_validation", {}) or {}
    strict_requested = bool(settings_validation.get("strict"))
    strict_settings_validated = bool(strict_requested and settings_validation.get("passed"))

    blockers = []
    if not parser_passed:
        blockers.append("PowerModelsONM parser smoke has not passed for this export.")
    elif strict_requested and not strict_settings_validated:
        error_summary = str(settings_validation.get("error_summary") or "unknown validation error")
        blockers.append(f"PowerModelsONM strict settings schema validation failed: {error_summary}")
    elif not strict_settings_validated:
        blockers.append(
            "PowerModelsONM parser smoke passed, but strict settings schema validation has not been recorded."
        )

    return {
        "passed": not blockers,
        "parser_passed": parser_passed,
        "strict_settings_validated": strict_settings_validated,
        "mld_solved": bool(smoke.get("mld", {}).get("solved") or smoke.get("mld_solved")),
        "bus_count": int(smoke.get("bus_count", 0) or 0),
        "load_count": int(smoke.get("load_count", 0) or 0),
        "generator_count": int(smoke.get("generator_count", 0) or 0),
        "switch_count": int(smoke.get("switch_count", 0) or 0),
        "blockers": blockers,
    }


def summarize_dynagrid_smoke_readiness(smoke: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize NRELDynaGrid online-control smoke evidence."""

    status_ok = smoke.get("status") == "ok"
    tracked_voltage_count = int(smoke.get("tracked_voltage_count", 0) or 0)
    der_setpoint_count = int(smoke.get("der_setpoint_count", 0) or 0)
    blockers = []
    if not status_ok:
        blockers.append("NRELDynaGrid smoke has not passed for the pilot export.")
    elif tracked_voltage_count <= 0:
        blockers.append("NRELDynaGrid smoke passed without tracked voltage outputs.")
    elif der_setpoint_count <= 0:
        blockers.append("NRELDynaGrid smoke passed without DER setpoint outputs.")

    return {
        "passed": not blockers,
        "status_ok": status_ok,
        "tracked_voltage_count": tracked_voltage_count,
        "der_setpoint_count": der_setpoint_count,
        "time_periods": int(smoke.get("time_periods", 0) or 0),
        "time_steps": int(smoke.get("time_steps", 0) or 0),
        "blockers": blockers,
    }


def summarize_event_bundle_readiness(export_dir: Path | str) -> dict[str, Any]:
    """Check for event-conditioned run bundles beside the ONM export."""

    root = Path(export_dir)
    manifests = sorted(root.glob("events/*/draw_*/run_manifest.json"))
    blockers = []
    bundles = []
    required_fields = (
        "events",
        "runtime_args",
        "nominal_load_window",
        "load_uncertainty",
        "block_demand_summary",
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append(f"Invalid event run manifest {manifest_path}: {exc}")
            continue
        missing_files = []
        for field in required_fields:
            raw_path = manifest.get(field)
            if not raw_path or not Path(str(raw_path)).exists():
                missing_files.append(field)
        timestep_count = int(manifest.get("timestep_count", 0) or 0)
        if missing_files:
            blockers.append(f"Event bundle {manifest_path.parent} is missing files: {missing_files}")
        if timestep_count <= 0:
            blockers.append(f"Event bundle {manifest_path.parent} has no positive timestep_count.")
        bundles.append(
            {
                "bundle_dir": str(manifest_path.parent),
                "event_id": manifest.get("event_id"),
                "mc_draw": manifest.get("mc_draw"),
                "event_count": int(manifest.get("event_count", 0) or 0),
                "timestep_count": timestep_count,
            }
        )
    if not manifests:
        blockers.append("No event-conditioned ONM run bundle found under events/<event_id>/draw_<mc_draw>.")

    return {
        "passed": not blockers,
        "bundle_count": len(manifests),
        "bundles": bundles,
        "blockers": blockers,
    }


def summarize_opendss_solve_readiness(network_dss: Path | str) -> dict[str, Any]:
    """Compile the OpenDSS network and report whether the base solve converges."""

    import opendssdirect as dss

    path = Path(network_dss)
    cwd = Path.cwd()
    blockers: list[str] = []
    try:
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{path}"')
        compiled = True
        compile_error = None
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        compiled = False
        compile_error = str(exc)
    finally:
        os.chdir(cwd)

    if not compiled:
        return {
            "passed": False,
            "compiled": False,
            "compile_error": compile_error,
            "circuit": None,
            "load_count": 0,
            "generator_count": 0,
            "line_count": 0,
            "bus_count": 0,
            "converged": False,
            "blockers": [f"OpenDSS compile failed for {path}: {compile_error}"],
        }

    converged = bool(dss.Solution.Converged())
    if not converged:
        blockers.append(
            "OpenDSS compiles but does not converge; diagnose voltage bases, source modeling, controls, "
            "and DER/load scaling before DNMG optimization."
        )

    return {
        "passed": not blockers,
        "compiled": True,
        "compile_error": None,
        "circuit": dss.Circuit.Name().lower(),
        "load_count": int(dss.Loads.Count()),
        "generator_count": int(dss.Generators.Count()),
        "line_count": int(dss.Lines.Count()),
        "bus_count": int(dss.Circuit.NumBuses()),
        "converged": converged,
        "blockers": blockers,
    }


def build_onm_readiness_report(
    export_dir: Path | str,
    *,
    smoke_filename: str = "powermodels_onm_smoke.json",
    dynagrid_smoke_filename: str = "dynagrid_smoke.json",
) -> dict[str, Any]:
    """Build a combined readiness report from a PMONM-facing export directory."""

    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    smoke_path = root / smoke_filename
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    else:
        smoke = {"status": "missing", "smoke_file": str(smoke_path)}
    dynagrid_smoke_path = root / dynagrid_smoke_filename
    if dynagrid_smoke_path.exists():
        dynagrid_smoke = json.loads(dynagrid_smoke_path.read_text(encoding="utf-8"))
    else:
        dynagrid_smoke = {"status": "missing", "smoke_file": str(dynagrid_smoke_path)}

    gates = {
        "powermodels_onm": summarize_powermodels_onm_smoke_readiness(smoke),
        "opendss": summarize_opendss_solve_readiness(root / "network.dss"),
        "der_export": summarize_der_export_readiness(manifest),
        "event_bundle": summarize_event_bundle_readiness(root),
        "dynagrid": summarize_dynagrid_smoke_readiness(dynagrid_smoke),
    }
    blockers = [
        blocker
        for gate in gates.values()
        for blocker in gate.get("blockers", [])
    ]
    recommended_next_tests = []
    if blockers:
        recommended_next_tests = [
            "Run strict PowerModelsONM settings/schema validation.",
            "Run the pilot MLD smoke with scripts/julia/powermodels_onm_smoke.jl --mld.",
            "Run the NRELDynaGrid smoke with NRELDYNAGRID_PATH set to the pinned source checkout.",
            "Materialize at least one event-conditioned ONM run bundle.",
            "Diagnose OpenDSS non-convergence and re-run the base solve gate.",
            "Run Layer 2 REopt sizing or mark the scenario explicitly as non-sizing/topology-only.",
            "Re-run the focused ONM readiness tests and the Marshfield power test subset.",
        ]

    return {
        "passed": not blockers,
        "export_dir": str(root),
        "export_scope": manifest.get("export_scope", "unknown"),
        "gates": gates,
        "blockers": blockers,
        "recommended_next_tests": recommended_next_tests,
    }
