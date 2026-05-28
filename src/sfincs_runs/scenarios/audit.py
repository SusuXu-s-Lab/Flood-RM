from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ForcingAuditIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ForcingManifestAudit:
    run_root: Path
    issues: tuple[ForcingAuditIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.error_codes

    @property
    def issue_codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    @property
    def error_codes(self) -> set[str]:
        return {issue.code for issue in self.issues if issue.severity == "error"}

    def issue(self, code: str) -> ForcingAuditIssue:
        for issue in self.issues:
            if issue.code == code:
                return issue
        raise KeyError(code)


def audit_forcing_manifest(
    run_root,
    *,
    ksat_cap_mmhr=360.0,
    max_ksat_cap_fraction=0.5,
    direction_wrap_threshold_degrees=180.0,
) -> ForcingManifestAudit:
    run_root = Path(run_root)
    issues: list[ForcingAuditIssue] = []
    manifest_path = run_root / "forcing_manifest.json"

    if not manifest_path.exists():
        return ForcingManifestAudit(
            run_root=run_root,
            issues=(
                ForcingAuditIssue(
                    "error",
                    "missing_forcing_manifest",
                    f"Missing forcing manifest: {manifest_path}",
                ),
            ),
        )

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if _missing(manifest.get("timing_policy")):
        issues.append(
            ForcingAuditIssue(
                "error",
                "missing_timing_policy",
                "forcing_manifest.json does not record the timing policy.",
            )
        )
    if _missing(manifest.get("run_start")) or _missing(manifest.get("run_stop")):
        issues.append(
            ForcingAuditIssue(
                "error",
                "missing_run_window",
                "forcing_manifest.json must record run_start and run_stop.",
            )
        )
    if _manifest_expects_precip(manifest) and _missing(manifest.get("rainfall_window_alignment")):
        issues.append(
            ForcingAuditIssue(
                "error",
                "missing_rainfall_window_alignment",
                "Precipitation is staged, but rainfall_window_alignment is missing.",
            )
        )

    issues.extend(
        _audit_ksat_cap(
            run_root / "sfincs.ks",
            ksat_cap_mmhr=ksat_cap_mmhr,
            max_ksat_cap_fraction=max_ksat_cap_fraction,
        )
    )
    issues.extend(
        _audit_wave_direction_wrap(
            run_root / "snapwave.bwd",
            direction_wrap_threshold_degrees=direction_wrap_threshold_degrees,
        )
    )

    return ForcingManifestAudit(run_root=run_root, issues=tuple(issues))


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "nan"
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _manifest_expects_precip(manifest: dict) -> bool:
    expected = manifest.get("expected_has_precip")
    if isinstance(expected, bool):
        return expected
    if isinstance(expected, str):
        lowered = expected.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return any(
        not _missing(manifest.get(key))
        for key in ("rainfall_member_id", "prepared_precip", "netamprfile", "precipfile")
    )


def _audit_ksat_cap(
    path: Path,
    *,
    ksat_cap_mmhr: float,
    max_ksat_cap_fraction: float,
) -> tuple[ForcingAuditIssue, ...]:
    if not path.exists():
        return ()

    values = np.fromfile(path, dtype="<f4")
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return (
            ForcingAuditIssue(
                "warning",
                "ksat_empty",
                f"{path.name} exists but has no positive finite Ksat values.",
            ),
        )

    cap_fraction = float(np.isclose(finite_positive, ksat_cap_mmhr).mean())
    if cap_fraction <= max_ksat_cap_fraction:
        return ()

    return (
        ForcingAuditIssue(
            "error",
            "ksat_cap_fraction_high",
            (
                f"{path.name} has {cap_fraction:.1%} of positive cells at "
                f"{ksat_cap_mmhr:g} mm/hr; this likely suppresses pluvial flooding."
            ),
        ),
    )


def _audit_wave_direction_wrap(
    path: Path,
    *,
    direction_wrap_threshold_degrees: float,
) -> tuple[ForcingAuditIssue, ...]:
    if not path.exists():
        return ()

    try:
        values = np.loadtxt(path)
    except ValueError as exc:
        return (
            ForcingAuditIssue(
                "warning",
                "wave_direction_unreadable",
                f"Could not read {path.name} as a SnapWave direction table: {exc}",
            ),
        )

    if values.size == 0:
        return ()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] < 2 or values.shape[1] < 2:
        return ()

    mean_direction = np.nanmean(values[:, 1:], axis=1)
    max_jump = float(np.nanmax(np.abs(np.diff(mean_direction))))
    if max_jump <= direction_wrap_threshold_degrees:
        return ()

    return (
        ForcingAuditIssue(
            "warning",
            "wave_direction_wrap_for_plotting",
            (
                f"{path.name} has a {max_jump:.1f} degree raw direction jump; "
                "unwrap directions before plotting across 0/360 degrees."
            ),
        ),
    )
