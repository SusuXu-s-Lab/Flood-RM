"""Stage B DER inventory builders."""

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from power.artifacts import present as _present

der_version = "stage_b_der_inventory.v0.1"
placement_rule_layer_1 = "evidence_anchored_mhmp"
placement_rule_layer_2 = "reopt_resilience_sizing"

mhmp_backed_statuses = frozenset({"documented_present", "planned_authorized"})

default_tier_clf_map: "dict[str, float]" = {
    "tier_0_life_safety": 1.00,
    "tier_1_response": 0.50,
    "tier_2_lifeline_support": 0.25,
}

default_electric_tariff = {
    "blended_annual_energy_rate": 0.20,
    "blended_annual_demand_rate": 0.0,
}

default_reopt_load_year = 2024

fema_community_lifelines_outage_hours = 72

hours_per_year = 8760

class DERAssignmentViolation(ValueError):
    """Raised when DER assignment completeness is not explicit enough."""

class ReoptSizingInputViolation(ValueError):
    """Raised when rows are not ready for live REopt resilience sizing."""

def validate_der_assignments(
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
    expected_hours: int = hours_per_year,
) -> dict[str, int]:
    """Validate rows that will be sent to REopt are operationally complete."""

    clf_map = dict(tier_clf_map if tier_clf_map is not None else default_tier_clf_map)
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


def der_schema() -> Any:
    """Return the schema for `der_inventory.parquet` v0.1.

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
    hours: int = hours_per_year,
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


def _der_facility_token(facility_id: str) -> str:
    return facility_id.rsplit(":", 1)[-1]


def _load_match_by_facility(
    rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if rows is None:
        return {}
    matches: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("assignment_status") != "assigned":
            continue
        facility_id = row.get("facility_id")
        if not _present(facility_id):
            continue
        matches[str(facility_id)] = row
    return matches


def build_der_inventory(
    facility_rows: Iterable[Mapping[str, Any]],
    *,
    location_id: str,
    load_matches: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Emit Layer 1 evidence-anchored DER rows from critical-facility rows."""

    matches = _load_match_by_facility(load_matches)
    rows: list[dict[str, Any]] = []
    for facility in facility_rows:
        if facility.get("backup_power_status") not in mhmp_backed_statuses:
            continue
        facility_id = facility["facility_id"]
        token = _der_facility_token(facility_id)
        match = matches.get(str(facility_id))
        if match:
            load_asset_id = match.get("load_asset_id")
            bus = match.get("matched_bus")
            phases = match.get("phases")
            nominal_voltage_kv = match.get("nominal_voltage_kv")
            assignment_status = "assigned"
            unassigned_reason = None
        else:
            load_asset_id = facility.get("load_asset_id")
            bus = None
            phases = None
            nominal_voltage_kv = None
            assignment_status = "unassigned"
            unassigned_reason = "pending_load_match"
        provenance = {
            "placement_rule": placement_rule_layer_1,
            "facility_token": token,
            "facility_id": facility_id,
            "facility_name": facility.get("facility_name"),
            "criticality_tier": facility.get("criticality_tier"),
            "backup_power_status": facility.get("backup_power_status"),
            "evidence_rank": facility.get("evidence_rank", "town_plan_primary"),
            "source_resilience_asset_type": facility.get("resilience_asset_type"),
            "assignment_status": assignment_status,
            "unassigned_reason": unassigned_reason,
            "load_match_id": match.get("assignment_id") if match else None,
        }
        rows.append(
            {
                "sandbox_id": location_id,
                "der_id": f"{location_id}:asset:der:{token}:genset",
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
                "placement_rule": placement_rule_layer_1,
                "evidence_rank": facility.get("evidence_rank", "town_plan_primary"),
                "confidence": facility.get("confidence", "medium"),
                "outage_duration_hours": None,
                "critical_load_fraction": None,
                "reopt_feasible": None,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": der_version,
            }
        )
    return rows


def write_der_inventory(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Write DER inventory rows to the canonical parquet artifact."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = der_schema()
    row_list = list(rows)
    columns = {
        field.name: [_clean_missing(row.get(field.name)) for row in row_list]
        for field in schema
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), output_path)


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
            "year": default_reopt_load_year,
        },
        "ElectricTariff": dict(electric_tariff or default_electric_tariff),
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
    outage_duration_hours: int = fema_community_lifelines_outage_hours,
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
    clf_map = dict(tier_clf_map if tier_clf_map is not None else default_tier_clf_map)
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
        upgraded["placement_rule"] = placement_rule_layer_2
        upgraded["critical_load_fraction"] = clf
        upgraded["outage_duration_hours"] = outage_duration_hours

        layer_2_provenance = {
            "placement_rule": placement_rule_layer_2,
            "reopt_status": response.get("status"),
            "reopt_run_uuid": response.get("run_uuid"),
            "reopt_api_version": response.get("api_version"),
            "reopt_version": response.get("reopt_version"),
            "reopt_payload_digest": _stable_json_digest(payload),
            "reopt_cache_policy": "redacted_full_response_by_payload_digest",
            "tier": tier,
            "critical_load_fraction": clf,
            "electric_load_year": default_reopt_load_year,
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

# REopt client

"""HTTP client for NREL/NLR REopt v3.

The REopt v3 API is asynchronous: a job is submitted via POST, then results
are fetched by polling a results endpoint until the status reaches a terminal
value. This module factors the protocol into pure URL/payload builders plus
an orchestrator ``ReoptClient`` with injectable HTTP transport and clock,
so tests can run entirely offline.

Per the NLR domain migration (May 2026), the base URL targets
``developer.nlr.gov``; the legacy ``developer.nrel.gov`` domain is being
shut down May 29, 2026.
"""


import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


reopt_base_url = "https://developer.nlr.gov/api/reopt/stable"

# REopt v3 returns these statuses; "Optimizing..." is the in-flight token.
terminal_statuses = frozenset({"optimal", "Infeasible", "error", "Error"})


class ReoptError(RuntimeError):
    """Raised when REopt returns a terminal error status or transport fails."""


def build_job_submit_request(
    payload: Mapping[str, Any], *, api_key: str
) -> tuple[str, dict[str, Any]]:
    """Return ``(submit_url, body)`` for POSTing a REopt job."""

    url = f"{reopt_base_url}/job/?api_key={api_key}"
    return url, dict(payload)


def build_results_poll_request(run_uuid: str, *, api_key: str) -> str:
    """Return the GET URL for polling REopt job results."""

    return f"{reopt_base_url}/job/{run_uuid}/results/?api_key={api_key}"


def is_terminal_status(status: str) -> bool:
    """``True`` if the REopt status no longer requires polling."""

    return status in terminal_statuses


def load_nlr_api_key(
    *,
    env_var: str = "NLR_API_KEY",
    fallback_path: Path | None = None,
) -> str:
    """Resolve the NLR developer API key from the environment or a local file.

    The fallback file is intentionally gitignored; production callers should
    set ``NLR_API_KEY`` instead.
    """

    value = os.environ.get(env_var)
    if value:
        return value.strip()
    path = fallback_path
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise ReoptError(
        f"NLR API key not found: set {env_var} env var or provide "
        f"a fallback file at {fallback_path}"
    )


class ReoptClient:
    """Callable wrapping REopt v3 submit + poll + parse, with optional cache.

    The transport is injected via ``http_post`` and ``http_get``: each must
    accept a URL and (for POST) a JSON-serialisable body, and return the
    response decoded into a dict.
    """

    def __init__(
        self,
        http_post: Callable[[str, dict[str, Any]], dict[str, Any]],
        http_get: Callable[[str], dict[str, Any]],
        *,
        api_key: str,
        cache_dir: Path | None = None,
        poll_interval_s: float = 5.0,
        max_poll_attempts: int = 240,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._http_post = http_post
        self._http_get = http_get
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._poll_interval_s = poll_interval_s
        self._max_poll_attempts = max_poll_attempts
        self._sleep = sleep if sleep is not None else _default_sleep

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        cached = self._cache_load(payload)
        if cached is not None:
            return cached

        submit_url, body = build_job_submit_request(payload, api_key=self._api_key)
        submit_response = self._http_post(submit_url, body)
        run_uuid = submit_response.get("run_uuid")
        if not run_uuid:
            raise ReoptError(
                f"REopt submit did not return a run_uuid: {submit_response!r}"
            )

        poll_url = build_results_poll_request(run_uuid, api_key=self._api_key)
        for _ in range(self._max_poll_attempts):
            results = self._http_get(poll_url)
            status = str(results.get("status", ""))
            if is_terminal_status(status):
                if status in {"error", "Error"}:
                    raise ReoptError(
                        f"REopt job {run_uuid} returned error: "
                        f"{results.get('messages') or results}"
                    )
                self._cache_store(payload, results)
                return results
            self._sleep(self._poll_interval_s)
        raise ReoptError(
            f"REopt job {run_uuid} did not reach a terminal status after "
            f"{self._max_poll_attempts} polls"
        )

    def _cache_path(self, payload: Mapping[str, Any]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = _payload_digest(payload)
        return self._cache_dir / f"{digest}.json"

    def _cache_load(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(payload)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _cache_store(self, payload: Mapping[str, Any], results: dict[str, Any]) -> None:
        path = self._cache_path(payload)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_redact_reopt_secrets(results), sort_keys=True),
            encoding="utf-8",
        )


def _payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact_reopt_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if key == "api_key" else _redact_reopt_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_reopt_secrets(item) for item in value]
    return value


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


default_nlr_api_key_file = Path("artifacts/credentials/nlr_api.txt")


def default_reopt_client(
    *,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    poll_interval_s: float = 5.0,
    max_poll_attempts: int = 240,
    fallback_key_path: Path = default_nlr_api_key_file,
) -> ReoptClient:
    """Wire a ``ReoptClient`` against stdlib HTTP transport.

    Resolves the NLR API key in this order: explicit ``api_key`` argument,
    then ``NLR_API_KEY`` environment variable, then the gitignored
    ``docs/nlr_api.txt`` local fallback. Reads from disk only at construction
    time so the key never appears in serialised state.
    """

    resolved_key = api_key or load_nlr_api_key(fallback_path=fallback_key_path)
    return ReoptClient(
        _urllib_post,
        _urllib_get,
        api_key=resolved_key,
        cache_dir=cache_dir,
        poll_interval_s=poll_interval_s,
        max_poll_attempts=max_poll_attempts,
    )


def _urllib_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    from urllib.error import HTTPError
    from urllib import request

    encoded = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req) as resp:  # noqa: S310 -- trusted REopt API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ReoptError(_http_error_message(exc)) from exc


def _urllib_get(url: str) -> dict[str, Any]:
    from urllib.error import HTTPError
    from urllib import request

    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req) as resp:  # noqa: S310 -- trusted REopt API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ReoptError(_http_error_message(exc)) from exc


def _http_error_message(exc: Any) -> str:
    body = ""
    if exc.fp is not None:
        try:
            body = exc.fp.read().decode("utf-8")
        except Exception:  # pragma: no cover - best-effort diagnostics only
            body = ""
    if body:
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                decoded = _redact_reopt_secrets(decoded)
                diagnostic_keys = (
                    "status",
                    "messages",
                    "errors",
                    "error",
                    "run_uuid",
                    "api_version",
                )
                compact = {key: decoded[key] for key in diagnostic_keys if key in decoded}
                decoded = compact if compact else decoded
            body = json.dumps(decoded, sort_keys=True)
        except json.JSONDecodeError:
            body = body[:1000]
    return f"REopt HTTP {exc.code} {exc.reason}: {body}".strip()


# Artifact-level REopt sizing

"""Artifact-level Layer 2 REopt sizing for the DER inventory."""

from power.resilience.profiles import (
    tier_int_to_string,
    build_archetype_load_profile,
    select_eversource_south_shore_tariff,
)



@dataclass(frozen=True)
class ReoptSizingResult:
    """Summary of an artifact-level REopt sizing refresh."""

    der_inventory_path: Path
    total_rows: int
    attempted_rows: int
    reopt_sized_rows: int
    provisional_rows: int


class CachedReoptResultsClient:
    """Cache-only REopt client for replaying committed local REopt artifacts."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if not path.exists():
            raise FileNotFoundError(f"REopt cache miss for payload digest {digest}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


class OfflineReoptSurrogateClient:
    """Deterministic local resilience-sizing stand-in for REopt API results.

    This is intentionally explicit provenance, not a cost-optimization claim.
    It is useful when the Marshfield artifact pipeline needs operational DERs
    for PMONM/DynaGrid integration but no live REopt API key or cached REopt
    responses are available.
    """

    def __init__(
        self,
        *,
        reserve_margin: float = 0.15,
        capacity_step_kw: float = 5.0,
    ) -> None:
        if reserve_margin < 0.0:
            raise ValueError("reserve_margin must be non-negative")
        if capacity_step_kw <= 0.0:
            raise ValueError("capacity_step_kw must be positive")
        self.reserve_margin = reserve_margin
        self.capacity_step_kw = capacity_step_kw

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        electric_load = payload.get("ElectricLoad", {}) or {}
        utility = payload.get("ElectricUtility", {}) or {}
        loads_kw = [float(value) for value in electric_load.get("loads_kw", [])]
        critical_load_fraction = float(electric_load.get("critical_load_fraction", 1.0))
        start = int(utility.get("outage_start_time_step", 1))
        end = int(utility.get("outage_end_time_step", len(loads_kw)))
        outage_loads = _wrapped_window(loads_kw, start_time_step=start, end_time_step=end)
        peak_critical_kw = max((value * critical_load_fraction for value in outage_loads), default=0.0)
        genset_kw = _round_up_to_step(
            peak_critical_kw * (1.0 + self.reserve_margin),
            self.capacity_step_kw,
        )
        digest = _stable_json_digest(payload)
        return {
            "status": "optimal_offline_surrogate",
            "run_uuid": f"offline-surrogate-{digest[:16]}",
            "api_version": "local_offline_surrogate",
            "reopt_version": "offline_reopt_surrogate.v0.1",
            "outputs": {
                "PV": {"size_kw": 0.0},
                "ElectricStorage": {"size_kw": 0.0, "size_kwh": 0.0},
                "Generator": {"size_kw": genset_kw},
                "Outages": {
                    "critical_loads_met": genset_kw >= peak_critical_kw and peak_critical_kw > 0.0,
                    "peak_critical_kw": peak_critical_kw,
                    "reserve_margin": self.reserve_margin,
                    "capacity_step_kw": self.capacity_step_kw,
                    "sizing_method": "max_outage_window_critical_load_with_reserve_margin",
                },
            },
        }


def _wrapped_window(
    values: list[float],
    *,
    start_time_step: int,
    end_time_step: int,
) -> list[float]:
    """Return a 1-indexed inclusive timestep window, wrapping over year end."""

    if not values:
        return []
    if start_time_step <= 0 or end_time_step <= 0:
        raise ValueError("REopt outage timesteps are expected to be positive 1-indexed values")
    count = end_time_step - start_time_step + 1
    if count <= 0:
        raise ValueError("end_time_step must be greater than or equal to start_time_step")
    n = len(values)
    start_index = (start_time_step - 1) % n
    return [values[(start_index + offset) % n] for offset in range(count)]


def _round_up_to_step(value: float, step: float) -> float:
    if value <= 0.0:
        return 0.0
    return math.ceil(value / step) * step


def size_der(
    *,
    smart_ds_compat_dir: Path,
    reopt_client: Any,
    outage_duration_hours: int = fema_community_lifelines_outage_hours,
    outage_start_hour: int = 4392,
    live_limit: int | None = None,
) -> ReoptSizingResult:
    """Apply REopt Layer 2 sizing to `der_inventory.parquet` artifacts."""

    der_inventory_path = smart_ds_compat_dir / "der_inventory.parquet"
    der_rows = pd.read_parquet(der_inventory_path).to_dict(orient="records")
    if live_limit is None:
        rows_for_reopt = der_rows
    else:
        rows_for_reopt = der_rows[:live_limit]

    inputs = _load_reopt_inputs(smart_ds_compat_dir)
    sized_subset = apply_layer_2_reopt_sizing(
        rows_for_reopt,
        facility_lookup=inputs["facility_lookup"],
        load_profiles_kw=inputs["load_profiles_kw"],
        reopt_client=reopt_client,
        electric_tariffs=inputs["electric_tariffs_by_facility"],
        electric_tariff_provenance=inputs["electric_tariff_provenance_by_facility"],
        load_profile_provenance=inputs["load_profile_provenance_by_facility"],
        tier_clf_map=default_tier_clf_map,
        outage_duration_hours=outage_duration_hours,
        outage_start_hour=outage_start_hour,
    )
    sized_by_der_id = {row["der_id"]: row for row in sized_subset}
    sized_rows = [sized_by_der_id.get(row["der_id"], dict(row)) for row in der_rows]

    write_der_inventory(sized_rows, der_inventory_path)
    reopt_sized_rows = sum(
        1 for row in sized_rows if row.get("placement_rule") == "reopt_resilience_sizing"
    )
    return ReoptSizingResult(
        der_inventory_path=der_inventory_path,
        total_rows=len(sized_rows),
        attempted_rows=len(rows_for_reopt),
        reopt_sized_rows=reopt_sized_rows,
        provisional_rows=len(sized_rows) - reopt_sized_rows,
    )


def run_layer_2_offline_reopt_surrogate_sizing(
    *,
    smart_ds_compat_dir: Path,
    outage_duration_hours: int = fema_community_lifelines_outage_hours,
    outage_start_hour: int = 4392,
    reserve_margin: float = 0.15,
    capacity_step_kw: float = 5.0,
    live_limit: int | None = None,
) -> ReoptSizingResult:
    """Size Layer 1 DER rows with deterministic local outage-window logic.

    Use this when a live/cached REopt refresh is unavailable but PMONM/DynaGrid
    integration needs operational grid-forming generators. Provenance is marked
    as `offline_reopt_surrogate.v0.1` via the parsed REopt-like response.
    """

    return size_der(
        smart_ds_compat_dir=smart_ds_compat_dir,
        reopt_client=OfflineReoptSurrogateClient(
            reserve_margin=reserve_margin,
            capacity_step_kw=capacity_step_kw,
        ),
        outage_duration_hours=outage_duration_hours,
        outage_start_hour=outage_start_hour,
        live_limit=live_limit,
    )


def _load_reopt_inputs(smart_ds_compat_dir: Path) -> dict[str, Any]:
    facilities = pd.read_parquet(smart_ds_compat_dir / "critical_facilities.parquet")
    assignments = pd.read_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet")

    facility_lookup: dict[str, dict[str, Any]] = {}
    for row in facilities.to_dict(orient="records"):
        normalized = dict(row)
        raw_tier = normalized.get("criticality_tier")
        normalized["criticality_tier"] = tier_int_to_string.get(raw_tier, raw_tier)
        facility_lookup[str(normalized["facility_id"])] = normalized

    load_profiles_kw: dict[str, list[float]] = {}
    load_profile_provenance_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariffs_by_facility: dict[str, dict[str, Any]] = {}
    electric_tariff_provenance_by_facility: dict[str, dict[str, Any]] = {}

    for row in assignments.to_dict(orient="records"):
        provenance = json.loads(row["source_provenance"])
        facility_id = str(provenance["facility_id"])
        archetype = {
            "profile_source": row["profile_source"],
            "source_building_type": row["source_building_type"],
            "source_geography": row["source_geography"],
            "schedule_overlay": provenance.get("schedule_overlay", "business_hours"),
        }
        peak_kw = float(row["peak_kw"])
        load_profiles_kw[facility_id] = build_archetype_load_profile(archetype, peak_kw=peak_kw)
        load_profile_provenance_by_facility[facility_id] = {
            "profile_source": row["profile_source"],
            "profile_source_version": row.get("profile_source_version"),
            "source_building_type": row["source_building_type"],
            "source_geography": row["source_geography"],
            "schedule_overlay": archetype["schedule_overlay"],
            "loadshape_id": row.get("loadshape_id"),
            "synthetic_placeholder": bool(provenance.get("synthetic_placeholder", False)),
        }
        facility = facility_lookup[facility_id]
        tariff_selection = select_eversource_south_shore_tariff(
            facility,
            peak_kw=peak_kw,
            customer_class=str(row["customer_class"]),
        )
        electric_tariffs_by_facility[facility_id] = tariff_selection.reopt_electric_tariff()
        electric_tariff_provenance_by_facility[facility_id] = tariff_selection.provenance()

    return {
        "facility_lookup": facility_lookup,
        "load_profiles_kw": load_profiles_kw,
        "load_profile_provenance_by_facility": load_profile_provenance_by_facility,
        "electric_tariffs_by_facility": electric_tariffs_by_facility,
        "electric_tariff_provenance_by_facility": electric_tariff_provenance_by_facility,
    }


def _write_der_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    write_der_inventory(rows, path)


def _clean_missing(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
