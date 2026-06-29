from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from .io import parse_sfincs_inp, read_json, write_json
from .schema import AuditIssue, AuditReport


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "nan"
    if isinstance(value, float):
        return math.isnan(value)
    return False


def audit_run_folder(
    run_root: str | Path,
    *,
    require_discharge: bool | None = None,
    require_water_level: bool | None = None,
    require_precip: bool | None = None,
    require_snapwave: bool | None = None,
    ksat_cap_mmhr: float = 360.0,
    max_ksat_cap_fraction: float = 0.5,
    write_report: bool = True,
) -> AuditReport:
    """Stakeholder-readable pre-run audit for a SFINCS event folder."""
    run_root = Path(run_root)
    issues: list[AuditIssue] = []
    manifest_path = run_root / "forcing_manifest.json"
    if not manifest_path.exists():
        issues.append(AuditIssue("error", "missing_forcing_manifest", f"Missing {manifest_path}"))
        report = AuditReport(run_root, tuple(issues))
        if write_report:
            write_json(run_root / "forcing_audit.json", report.to_dict())
        return report

    manifest = read_json(manifest_path)
    inp = parse_sfincs_inp(run_root / "sfincs.inp")
    if _missing(manifest.get("run_start")) or _missing(manifest.get("run_stop")):
        issues.append(AuditIssue("error", "missing_run_window", "forcing_manifest.json must record run_start and run_stop."))
    if not (run_root / "sfincs.inp").exists():
        issues.append(AuditIssue("error", "missing_sfincs_inp", "Missing sfincs.inp."))

    mode = str(manifest.get("forcing_mode", ""))
    discharge_expected = require_discharge if require_discharge is not None else mode in {"inland_wflow_discharge", "compound_coastal_inland"}
    water_expected = require_water_level if require_water_level is not None else bool(manifest.get("coastal_water_level")) or mode in {"coastal_water_level", "compound_coastal_inland"}
    precip_expected = require_precip if require_precip is not None else bool(manifest.get("precipitation_nc") or manifest.get("netamprfile"))
    snapwave_expected = require_snapwave if require_snapwave is not None else bool(manifest.get("snapwave"))

    _require_file(issues, run_root, inp.get("srcfile", "sfincs.src"), "missing_srcfile", discharge_expected)
    _require_file(issues, run_root, inp.get("disfile", "sfincs.dis"), "missing_disfile", discharge_expected)
    _require_file(issues, run_root, inp.get("bndfile", "sfincs.bnd"), "missing_bndfile", water_expected)
    _require_file(issues, run_root, inp.get("bzsfile", "sfincs.bzs"), "missing_bzsfile", water_expected)
    if precip_expected:
        precip_file = inp.get("netamprfile") or inp.get("precipfile") or manifest.get("netamprfile")
        _require_file(issues, run_root, precip_file, "missing_precip_forcing", True)
    if snapwave_expected:
        for key in ("bhs", "btp", "bwd", "bds"):
            _require_file(issues, run_root, inp.get(f"{key}file", f"snapwave.{key}"), f"missing_snapwave_{key}", True)

    issues.extend(_audit_ksat_cap(run_root / "sfincs.ks", ksat_cap_mmhr=ksat_cap_mmhr, max_ksat_cap_fraction=max_ksat_cap_fraction))
    issues.extend(_audit_wave_direction_wrap(run_root / "snapwave.bwd"))

    report = AuditReport(run_root, tuple(issues))
    if write_report:
        write_json(run_root / "forcing_audit.json", report.to_dict())
    return report


def _require_file(issues: list[AuditIssue], root: Path, value, code: str, required: bool) -> None:
    if not required:
        return
    if _missing(value):
        issues.append(AuditIssue("error", code, f"Required SFINCS input key/path is missing for {code}."))
        return
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if not path.exists() or path.stat().st_size == 0:
        issues.append(AuditIssue("error", code, f"Missing or empty required SFINCS input: {path}"))


def _audit_ksat_cap(path: Path, *, ksat_cap_mmhr: float, max_ksat_cap_fraction: float) -> tuple[AuditIssue, ...]:
    if not path.exists():
        return ()
    values = np.fromfile(path, dtype="<f4")
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return (AuditIssue("warning", "ksat_empty", f"{path.name} exists but has no positive finite Ksat values."),)
    cap_fraction = float(np.isclose(finite_positive, float(ksat_cap_mmhr)).mean())
    if cap_fraction <= float(max_ksat_cap_fraction):
        return ()
    return (
        AuditIssue(
            "error",
            "ksat_cap_fraction_high",
            f"{path.name} has {cap_fraction:.1%} of positive cells at {ksat_cap_mmhr:g} mm/hr; this may suppress pluvial flooding.",
        ),
    )


def _audit_wave_direction_wrap(path: Path, *, threshold_degrees: float = 180.0) -> tuple[AuditIssue, ...]:
    if not path.exists():
        return ()
    try:
        values = np.loadtxt(path)
    except ValueError as exc:
        return (AuditIssue("warning", "wave_direction_unreadable", f"Could not read {path.name}: {exc}"),)
    if values.size == 0:
        return ()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] < 2 or values.shape[1] < 2:
        return ()
    mean_direction = np.nanmean(values[:, 1:], axis=1)
    max_jump = float(np.nanmax(np.abs(np.diff(mean_direction))))
    if max_jump <= float(threshold_degrees):
        return ()
    return (
        AuditIssue(
            "warning",
            "wave_direction_wrap_for_plotting",
            f"{path.name} has a {max_jump:.1f} degree raw direction jump; unwrap directions for plots across 0/360 degrees.",
        ),
    )
