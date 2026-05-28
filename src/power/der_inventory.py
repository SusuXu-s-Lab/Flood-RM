"""Stage B DER inventory builders.

Layer 1 (evidence-anchored): one DER row per MHMP-anchored critical facility
with documented or planned-authorized backup power. Layer 2 (REopt) fills
capacities and resilience-sizing fields later; Layer 1 emits the
placement-and-evidence skeleton only.

Methodology: docs/power/methodology/der_placement_methodology.md
Schema: docs/power/methodology/simulated_data_protocol.md
(`stage_b_der_inventory.v0.1`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA_VERSION = "stage_b_der_inventory.v0.1"
PLACEMENT_RULE_LAYER_1 = "evidence_anchored_mhmp"
PLACEMENT_RULE_LAYER_2 = "reopt_resilience_sizing"

MHMP_BACKED_STATUSES = frozenset({"documented_present", "planned_authorized"})

DEFAULT_TIER_CLF_MAP: "dict[str, float]" = {
    "tier_0_life_safety": 1.00,
    "tier_1_response": 0.50,
    "tier_2_lifeline_support": 0.25,
}

DEFAULT_ELECTRIC_TARIFF = {
    "blended_annual_energy_rate": 0.20,
    "blended_annual_demand_rate": 0.0,
}

DEFAULT_REOPT_LOAD_YEAR = 2024

FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS = 72

HOURS_PER_YEAR = 8760


class DERAssignmentViolation(ValueError):
    """Raised when DER assignment completeness is not explicit enough."""


class ReoptSizingInputViolation(ValueError):
    """Raised when rows are not ready for live REopt resilience sizing."""


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
    except TypeError:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def validate_der_assignment_completeness(
    rows: Iterable[Mapping[str, Any]],
    *,
    valid_buses: Iterable[str] | None = None,
    valid_block_ids: Iterable[str] | None = None,
) -> None:
    """Validate DER rows are either electrically assigned or explicitly deferred.

    A DER can only be operational in ONM once it is attached to an electrical
    bus or load block. Evidence-backed rows that have not yet been matched must
    say so with ``assignment_status = unassigned`` and a machine-readable
    ``unassigned_reason``; silent nulls are not allowed through this gate.
    """

    valid_bus_set = {str(bus) for bus in valid_buses} if valid_buses is not None else None
    valid_block_set = (
        {str(block_id) for block_id in valid_block_ids} if valid_block_ids is not None else None
    )

    for index, row in enumerate(rows):
        der_id = str(row.get("der_id") or f"row[{index}]")
        bus = row.get("bus")
        block_id = row.get("block_id")
        status = row.get("assignment_status")
        reason = row.get("unassigned_reason")

        has_bus = _present(bus)
        has_block = _present(block_id)

        if has_bus or has_block:
            if status != "assigned":
                raise DERAssignmentViolation(
                    f"{der_id} has an electrical assignment but assignment_status={status!r}; "
                    "expected 'assigned'"
                )
            if _present(reason):
                raise DERAssignmentViolation(
                    f"{der_id} is assigned but still carries unassigned_reason={reason!r}"
                )
            if valid_bus_set is not None and has_bus and str(bus) not in valid_bus_set:
                raise DERAssignmentViolation(f"{der_id} references unknown bus {bus!r}")
            if valid_block_set is not None and has_block and str(block_id) not in valid_block_set:
                raise DERAssignmentViolation(f"{der_id} references unknown block_id {block_id!r}")
            continue

        if status != "unassigned":
            raise DERAssignmentViolation(
                f"{der_id} has no bus or block_id; assignment_status must be 'unassigned'"
            )
        if not _present(reason):
            raise DERAssignmentViolation(
                f"{der_id} has assignment_status='unassigned' but no unassigned_reason"
            )


def validate_reopt_sizing_inputs(
    rows: Iterable[Mapping[str, Any]],
    *,
    facility_lookup: Mapping[str, Mapping[str, Any]],
    load_profiles_kw: Mapping[str, list[float]],
    tier_clf_map: "Mapping[str, float] | None" = None,
    expected_hours: int = HOURS_PER_YEAR,
) -> dict[str, int]:
    """Validate rows that will be sent to REopt are operationally complete."""

    clf_map = dict(tier_clf_map if tier_clf_map is not None else DEFAULT_TIER_CLF_MAP)
    checked = 0
    skipped_missing_profile = 0
    skipped_tier = 0

    for row in rows:
        facility_id = str(row.get("facility_id"))
        der_id = str(row.get("der_id") or facility_id)
        facility = facility_lookup.get(facility_id)
        if facility is None:
            raise ReoptSizingInputViolation(f"{der_id}: missing facility row for REopt sizing")

        tier = facility.get("criticality_tier")
        if tier not in clf_map:
            skipped_tier += 1
            continue
        load_profile = load_profiles_kw.get(facility_id)
        if load_profile is None:
            skipped_missing_profile += 1
            continue

        if row.get("assignment_status") != "assigned":
            raise ReoptSizingInputViolation(
                f"{der_id}: assignment_status must be 'assigned' before live REopt sizing"
            )
        if not _present(row.get("bus")) and not _present(row.get("block_id")):
            raise ReoptSizingInputViolation(
                f"{der_id}: live REopt sizing requires an assigned bus or block_id"
            )
        if len(load_profile) != expected_hours:
            raise ReoptSizingInputViolation(
                f"{der_id}: REopt load profile must contain {expected_hours} hourly values; "
                f"got {len(load_profile)}"
            )
        checked += 1

    return {
        "reopt_ready_rows": checked,
        "reopt_skipped_missing_profile": skipped_missing_profile,
        "reopt_skipped_tier": skipped_tier,
    }


def der_inventory_pyarrow_schema() -> Any:
    """Return the pyarrow schema for `der_inventory.parquet` v0.1.

    All capacity, REopt-input, and bus-binding fields are nullable so that
    Layer 1 rows (which Layer 2 has not yet sized) round-trip cleanly through
    parquet. The schema mirrors the field list documented in
    `simulated_data_protocol.md`.
    """

    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sandbox_id", pa.string(), nullable=False),
            pa.field("der_id", pa.string(), nullable=False),
            pa.field("facility_id", pa.string(), nullable=True),
            pa.field("load_asset_id", pa.string(), nullable=True),
            pa.field("bus", pa.string(), nullable=True),
            pa.field("block_id", pa.string(), nullable=True),
            pa.field("assignment_status", pa.string(), nullable=False),
            pa.field("unassigned_reason", pa.string(), nullable=True),
            pa.field("phases", pa.string(), nullable=True),
            pa.field("nominal_voltage_kv", pa.float64(), nullable=True),
            pa.field("resilience_asset_type", pa.string(), nullable=False),
            pa.field("pv_kw", pa.float64(), nullable=True),
            pa.field("bess_kw", pa.float64(), nullable=True),
            pa.field("bess_kwh", pa.float64(), nullable=True),
            pa.field("genset_kw", pa.float64(), nullable=True),
            pa.field("gfm_capable", pa.bool_(), nullable=False),
            pa.field("placement_rule", pa.string(), nullable=False),
            pa.field("evidence_rank", pa.string(), nullable=False),
            pa.field("confidence", pa.string(), nullable=False),
            pa.field("outage_duration_hours", pa.int16(), nullable=True),
            pa.field("critical_load_fraction", pa.float64(), nullable=True),
            pa.field("reopt_feasible", pa.bool_(), nullable=True),
            pa.field("source_provenance", pa.string(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )


def build_synthetic_commercial_load_profile(
    *,
    peak_kw: float,
    business_start_hour: int = 8,
    business_end_hour: int = 18,
    overnight_floor_fraction: float = 0.18,
    weekend_fraction: float = 0.35,
    hours: int = HOURS_PER_YEAR,
) -> list[float]:
    """Placeholder commercial-archetype 8760 load profile.

    The shape is intentionally simple and clearly synthetic: a sinusoidal
    diurnal envelope clipped to a baseline floor, scaled down on weekends.
    Replaced by ResStock/ComStock archetype assignment in a later slice; the
    `source_provenance` of any DER inventory row sized against this profile
    must record `loadshape_id = synthetic_commercial_placeholder_v0`.
    """

    if not 0 <= business_start_hour < business_end_hour <= 24:
        raise ValueError("business hour window must lie within 0..24 with start<end")

    profile: list[float] = []
    for hour in range(hours):
        day_of_week = (hour // 24) % 7  # 0 = Monday in the synthetic year
        hour_of_day = hour % 24
        weekend = day_of_week >= 5

        if business_start_hour <= hour_of_day < business_end_hour:
            # Half-sinusoid peaking mid-business-hours.
            window = business_end_hour - business_start_hour
            phase = (hour_of_day - business_start_hour + 0.5) / window
            shape = math.sin(math.pi * phase)
            value = overnight_floor_fraction + (1.0 - overnight_floor_fraction) * shape
        else:
            value = overnight_floor_fraction

        if weekend:
            value *= weekend_fraction / overnight_floor_fraction if value > overnight_floor_fraction else 1.0
            value = min(value, weekend_fraction)

        profile.append(round(value * peak_kw, 6))

    # Force a strict global peak at peak_kw on at least one weekday hour.
    profile[business_start_hour + (business_end_hour - business_start_hour) // 2] = peak_kw
    return profile


def _facility_token(facility_id: str) -> str:
    return facility_id.rsplit(":", 1)[-1]


def _assigned_critical_load_by_facility(
    rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if rows is None:
        return {}
    assignments: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("assignment_status") != "assigned":
            continue
        facility_id = row.get("facility_id")
        if not _present(facility_id):
            continue
        assignments[str(facility_id)] = row
    return assignments


def build_layer_1_der_inventory(
    facility_rows: Iterable[Mapping[str, Any]],
    *,
    sandbox_id: str,
    critical_load_assignments: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Emit Layer 1 evidence-anchored DER rows from critical-facility rows."""

    assigned_loads = _assigned_critical_load_by_facility(critical_load_assignments)
    rows: list[dict[str, Any]] = []
    for facility in facility_rows:
        if facility.get("backup_power_status") not in MHMP_BACKED_STATUSES:
            continue
        facility_id = facility["facility_id"]
        token = _facility_token(facility_id)
        assignment = assigned_loads.get(str(facility_id))
        if assignment:
            load_asset_id = assignment.get("load_asset_id")
            bus = assignment.get("matched_bus")
            phases = assignment.get("phases")
            nominal_voltage_kv = assignment.get("nominal_voltage_kv")
            assignment_status = "assigned"
            unassigned_reason = None
        else:
            load_asset_id = facility.get("load_asset_id")
            bus = None
            phases = None
            nominal_voltage_kv = None
            assignment_status = "unassigned"
            unassigned_reason = "pending_critical_load_assignment"
        provenance = {
            "placement_rule": PLACEMENT_RULE_LAYER_1,
            "facility_token": token,
            "facility_id": facility_id,
            "facility_name": facility.get("facility_name"),
            "criticality_tier": facility.get("criticality_tier"),
            "backup_power_status": facility.get("backup_power_status"),
            "evidence_rank": facility.get("evidence_rank", "town_plan_primary"),
            "source_resilience_asset_type": facility.get("resilience_asset_type"),
            "assignment_status": assignment_status,
            "unassigned_reason": unassigned_reason,
            "critical_load_assignment": assignment.get("assignment_id") if assignment else None,
        }
        rows.append(
            {
                "sandbox_id": sandbox_id,
                "der_id": f"{sandbox_id}:asset:der:{token}:genset",
                "facility_id": facility_id,
                "load_asset_id": load_asset_id,
                "bus": bus,
                "block_id": None,
                "assignment_status": assignment_status,
                "unassigned_reason": unassigned_reason,
                "phases": phases,
                "nominal_voltage_kv": nominal_voltage_kv,
                "resilience_asset_type": "genset",
                "pv_kw": None,
                "bess_kw": None,
                "bess_kwh": None,
                "genset_kw": None,
                "gfm_capable": True,
                "placement_rule": PLACEMENT_RULE_LAYER_1,
                "evidence_rank": facility.get("evidence_rank", "town_plan_primary"),
                "confidence": facility.get("confidence", "medium"),
                "outage_duration_hours": None,
                "critical_load_fraction": None,
                "reopt_feasible": None,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return rows


def build_reopt_request_payload(
    facility: Mapping[str, Any],
    *,
    load_profile_kw: list[float],
    critical_load_fraction: float,
    outage_duration_hours: int,
    outage_start_hour: int,
    electric_tariff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a REopt v3 request payload for one facility.

    Network transport is the caller's responsibility; this builder is pure.
    """

    outage_end_time_step = outage_start_hour + outage_duration_hours - 1
    return {
        "Site": {
            "latitude": facility["lat"],
            "longitude": facility["lon"],
        },
        "ElectricLoad": {
            "loads_kw": list(load_profile_kw),
            "critical_load_fraction": critical_load_fraction,
            "year": DEFAULT_REOPT_LOAD_YEAR,
        },
        "ElectricTariff": dict(electric_tariff or DEFAULT_ELECTRIC_TARIFF),
        "ElectricUtility": {
            "outage_start_time_step": outage_start_hour,
            "outage_end_time_step": outage_end_time_step,
        },
        "PV": {},
        "ElectricStorage": {},
        "Generator": {},
    }


def _stable_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_reopt_results(results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract Layer 2 capacities and feasibility from a REopt results envelope.

    Returns a dict with `pv_kw`, `bess_kw`, `bess_kwh`, `genset_kw`, and
    `reopt_feasible`. Missing technology blocks default to zero; missing
    outage feasibility defaults to `False`.
    """

    outputs = results.get("outputs", {}) or {}
    pv = outputs.get("PV", {}) or {}
    bess = outputs.get("ElectricStorage", {}) or {}
    genset = outputs.get("Generator", {}) or {}
    outages = outputs.get("Outages", {}) or {}

    return {
        "pv_kw": float(pv.get("size_kw", 0.0)),
        "bess_kw": float(bess.get("size_kw", 0.0)),
        "bess_kwh": float(bess.get("size_kwh", 0.0)),
        "genset_kw": float(genset.get("size_kw", 0.0)),
        "reopt_feasible": bool(outages.get("critical_loads_met", False)),
    }


def apply_layer_2_reopt_sizing(
    layer_1_rows: Iterable[Mapping[str, Any]],
    *,
    facility_lookup: Mapping[str, Mapping[str, Any]],
    load_profiles_kw: Mapping[str, list[float]],
    reopt_client: Any,
    electric_tariffs: "Mapping[str, Mapping[str, Any]] | None" = None,
    electric_tariff_provenance: "Mapping[str, Mapping[str, Any]] | None" = None,
    load_profile_provenance: "Mapping[str, Mapping[str, Any]] | None" = None,
    tier_clf_map: "Mapping[str, float] | None" = None,
    outage_duration_hours: int = FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS,
    outage_start_hour: int = 4392,
) -> list[dict[str, Any]]:
    """Apply REopt resilience sizing to Layer 1 rows.

    For each Layer 1 row whose facility carries a CLF tier in ``tier_clf_map``
    (default: tier_0=1.00, tier_1=0.50, tier_2=0.25), build a REopt payload,
    invoke ``reopt_client(payload)``, parse the response, and return an
    upgraded Layer 2 row. Rows whose facility tier is absent from the CLF
    map are passed through unchanged at Layer 1 values.
    """

    layer_1_list = [dict(row) for row in layer_1_rows]
    clf_map = dict(tier_clf_map if tier_clf_map is not None else DEFAULT_TIER_CLF_MAP)
    validate_reopt_sizing_inputs(
        layer_1_list,
        facility_lookup=facility_lookup,
        load_profiles_kw=load_profiles_kw,
        tier_clf_map=clf_map,
    )

    sized_rows: list[dict[str, Any]] = []
    for row in layer_1_list:
        facility_id = row["facility_id"]
        facility = facility_lookup.get(facility_id)
        if facility is None:
            sized_rows.append(dict(row))
            continue

        tier = facility.get("criticality_tier")
        clf = clf_map.get(tier)
        load_profile = load_profiles_kw.get(facility_id)
        if clf is None or load_profile is None:
            sized_rows.append(dict(row))
            continue

        payload = build_reopt_request_payload(
            facility,
            load_profile_kw=load_profile,
            critical_load_fraction=clf,
            outage_duration_hours=outage_duration_hours,
            outage_start_hour=outage_start_hour,
            electric_tariff=(
                electric_tariffs.get(facility_id)
                if electric_tariffs is not None
                else None
            ),
        )
        response = reopt_client(payload)
        parsed = parse_reopt_results(response)

        upgraded = dict(row)
        upgraded.update(parsed)
        upgraded["placement_rule"] = PLACEMENT_RULE_LAYER_2
        upgraded["critical_load_fraction"] = clf
        upgraded["outage_duration_hours"] = outage_duration_hours

        layer_2_provenance = {
            "placement_rule": PLACEMENT_RULE_LAYER_2,
            "reopt_status": response.get("status"),
            "reopt_run_uuid": response.get("run_uuid"),
            "reopt_api_version": response.get("api_version"),
            "reopt_version": response.get("reopt_version"),
            "reopt_payload_digest": _stable_json_digest(payload),
            "reopt_cache_policy": "redacted_full_response_by_payload_digest",
            "tier": tier,
            "critical_load_fraction": clf,
            "electric_load_year": DEFAULT_REOPT_LOAD_YEAR,
            "load_profile_hours": len(load_profile),
            "load_profile_provenance": (
                dict(load_profile_provenance.get(facility_id))
                if load_profile_provenance is not None
                and facility_id in load_profile_provenance
                else None
            ),
            "electric_tariff": dict(payload["ElectricTariff"]),
            "electric_tariff_provenance": (
                dict(electric_tariff_provenance.get(facility_id))
                if electric_tariff_provenance is not None
                and facility_id in electric_tariff_provenance
                else None
            ),
            "outage_duration_hours": outage_duration_hours,
            "outage_start_hour": outage_start_hour,
            "layer_1_source_provenance": row.get("source_provenance"),
        }
        upgraded["source_provenance"] = json.dumps(layer_2_provenance, sort_keys=True)

        sized_rows.append(upgraded)

    return sized_rows
