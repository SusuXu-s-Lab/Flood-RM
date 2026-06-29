"""Canonical reviewer-facing Event Catalog bundle for ADR-0020.

The v2 reference emits two long tables plus one audit JSON:

* ``events.csv``: one row per Event Catalog row with probability labels and weights.
* ``drivers.csv``: one row per event-driver Field-Preserving Realization.
* ``audit.json``: formulas, source/config provenance, and summary checks.

This module owns the bundle Interface only. Probability fitting and realization stay in
``probability.py``, ``records.py``, and ``realization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "event_role",
    "event_origin",
    "event_family",
    "scenario_name",
    "sample_rp_years",
    "and_joint_exceedance_prob",
    "and_joint_aep",
    "severity_band",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "event_reference_time",
    "selection_reason",
]


DRIVER_COLUMNS = [
    "event_id",
    "driver",
    "x",
    "u",
    "driver_role",
    "member_id",
    "member_file",
    "member_time",
    "template_value",
    "scale_factor",
    "lag_hours",
    "time_policy",
    "realization_policy",
    "source",
]


@dataclass(frozen=True)
class ReferenceBundle:
    """In-memory and on-disk handles for the ADR-0020 reference bundle."""

    events: pd.DataFrame
    drivers: pd.DataFrame
    audit: dict[str, Any]
    events_path: Path
    drivers_path: Path
    audit_path: Path


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with the canonical event columns first."""

    frame = events.copy()
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[EVENT_COLUMNS + [c for c in frame.columns if c not in EVENT_COLUMNS]]


def normalize_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Return ``drivers`` with the canonical long driver columns first."""

    frame = drivers.copy()
    for column in DRIVER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[DRIVER_COLUMNS + [c for c in frame.columns if c not in DRIVER_COLUMNS]]


def write_reference_bundle(
    events: pd.DataFrame,
    drivers: pd.DataFrame,
    audit: dict[str, Any],
    output_dir,
) -> ReferenceBundle:
    """Write ``events.csv``, ``drivers.csv``, and ``audit.json`` under ``output_dir``."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    events_out = normalize_events(events)
    drivers_out = normalize_drivers(drivers)
    events_path = root / "events.csv"
    drivers_path = root / "drivers.csv"
    audit_path = root / "audit.json"

    events_out.to_csv(events_path, index=False)
    drivers_out.to_csv(drivers_path, index=False)
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True), encoding="utf-8")

    return ReferenceBundle(
        events=events_out,
        drivers=drivers_out,
        audit=_jsonable(audit),
        events_path=events_path,
        drivers_path=drivers_path,
        audit_path=audit_path,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


__all__ = [
    "EVENT_COLUMNS",
    "DRIVER_COLUMNS",
    "ReferenceBundle",
    "normalize_events",
    "normalize_drivers",
    "write_reference_bundle",
]
