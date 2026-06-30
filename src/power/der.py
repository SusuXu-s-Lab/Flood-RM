"""DER inventory and resilience sizing adapters.

Layer 1 records evidence-anchored DER candidates. Layer 2 upgrades rows with a
REopt-like sizing result. The offline surrogate is explicit provenance, not a
cost-optimization claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import present, write_table
from .profiles import build_archetype_load_profile, select_eversource_south_shore_tariff, tier_int_to_string

der_version = "stage_b_der_inventory.v0.1"
placement_rule_layer_1 = "evidence_anchored_mhmp"
placement_rule_layer_2 = "reopt_resilience_sizing"
mhmp_backed_statuses = frozenset({"documented_present", "planned_authorized"})
default_tier_clf_map = {"tier_0_life_safety": 1.00, "tier_1_response": 0.50, "tier_2_lifeline_support": 0.25}
default_electric_tariff = {"blended_annual_energy_rate": 0.20, "blended_annual_demand_rate": 0.0}
default_reopt_load_year = 2024
fema_community_lifelines_outage_hours = 72
hours_per_year = 8760
reopt_base_url = "https://developer.nlr.gov/api/reopt/stable"
terminal_statuses = frozenset({"optimal", "Infeasible", "error", "Error"})
default_nlr_api_key_file = Path("artifacts/credentials/nlr_api.txt")


class DERAssignmentViolation(ValueError):
    pass


class ReoptSizingInputViolation(ValueError):
    pass


class ReoptError(RuntimeError):
    pass


def validate_der_assignments(rows: Iterable[Mapping[str, Any]], *, valid_buses: Iterable[str] | None = None, valid_block_ids: Iterable[str] | None = None) -> None:
    buses = {str(b) for b in valid_buses} if valid_buses is not None else None
    blocks = {str(b) for b in valid_block_ids} if valid_block_ids is not None else None
    for i, row in enumerate(rows):
        der_id = str(row.get("der_id") or f"row[{i}]")
        has_bus, has_block = present(row.get("bus")), present(row.get("block_id"))
        if has_bus or has_block:
            if row.get("assignment_status") != "assigned":
                raise DERAssignmentViolation(f"{der_id}: electrical assignment requires assignment_status='assigned'")
            if buses is not None and has_bus and str(row.get("bus")) not in buses:
                raise DERAssignmentViolation(f"{der_id}: unknown bus {row.get('bus')!r}")
            if blocks is not None and has_block and str(row.get("block_id")) not in blocks:
                raise DERAssignmentViolation(f"{der_id}: unknown block_id {row.get('block_id')!r}")
        elif row.get("assignment_status") != "unassigned" or not present(row.get("unassigned_reason")):
            raise DERAssignmentViolation(f"{der_id}: unassigned rows must carry unassigned_reason")


def der_schema() -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field("sandbox_id", pa.string(), nullable=False), pa.field("der_id", pa.string(), nullable=False),
        pa.field("facility_id", pa.string()), pa.field("load_asset_id", pa.string()), pa.field("bus", pa.string()), pa.field("block_id", pa.string()),
        pa.field("assignment_status", pa.string(), nullable=False), pa.field("unassigned_reason", pa.string()),
        pa.field("phases", pa.string()), pa.field("nominal_voltage_kv", pa.float64()), pa.field("resilience_asset_type", pa.string(), nullable=False),
        pa.field("pv_kw", pa.float64()), pa.field("bess_kw", pa.float64()), pa.field("bess_kwh", pa.float64()), pa.field("genset_kw", pa.float64()),
        pa.field("gfm_capable", pa.bool_(), nullable=False), pa.field("placement_rule", pa.string(), nullable=False),
        pa.field("evidence_rank", pa.string(), nullable=False), pa.field("confidence", pa.string(), nullable=False),
        pa.field("outage_duration_hours", pa.int16()), pa.field("critical_load_fraction", pa.float64()), pa.field("reopt_feasible", pa.bool_()),
        pa.field("source_provenance", pa.string(), nullable=False), pa.field("schema_version", pa.string(), nullable=False),
    ])


def _facility_token(facility_id: str) -> str:
    return facility_id.rsplit(":", 1)[-1]


def _load_match_by_facility(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    return {str(r["facility_id"]): r for r in rows or [] if r.get("assignment_status") == "assigned" and present(r.get("facility_id"))}


def build_der_inventory(facility_rows: Iterable[Mapping[str, Any]], *, location_id: str, load_matches: Iterable[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    matches = _load_match_by_facility(load_matches)
    out: list[dict[str, Any]] = []
    for facility in facility_rows:
        if facility.get("backup_power_status") not in mhmp_backed_statuses:
            continue
        fid = str(facility["facility_id"])
        token = _facility_token(fid)
        match = matches.get(fid)
        if match:
            load_asset_id, bus, phases, kv, status, reason = match.get("load_asset_id"), match.get("matched_bus"), match.get("phases"), match.get("nominal_voltage_kv"), "assigned", None
        else:
            load_asset_id, bus, phases, kv, status, reason = facility.get("load_asset_id"), None, None, None, "unassigned", "pending_load_match"
        prov = {"placement_rule": placement_rule_layer_1, "facility_token": token, "facility_id": fid, "facility_name": facility.get("facility_name"), "criticality_tier": facility.get("criticality_tier"), "backup_power_status": facility.get("backup_power_status"), "assignment_status": status, "unassigned_reason": reason, "load_match_id": match.get("assignment_id") if match else None}
        out.append({
            "sandbox_id": location_id, "der_id": f"{location_id}:asset:der:{token}:genset", "facility_id": fid,
            "load_asset_id": load_asset_id, "bus": bus, "block_id": None, "assignment_status": status, "unassigned_reason": reason,
            "phases": phases, "nominal_voltage_kv": kv, "resilience_asset_type": "genset", "pv_kw": None, "bess_kw": None,
            "bess_kwh": None, "genset_kw": None, "gfm_capable": True, "placement_rule": placement_rule_layer_1,
            "evidence_rank": facility.get("evidence_rank", "town_plan_primary"), "confidence": facility.get("confidence", "medium"),
            "outage_duration_hours": None, "critical_load_fraction": None, "reopt_feasible": None,
            "source_provenance": json.dumps(prov, sort_keys=True), "schema_version": der_version,
        })
    return out


def write_der_inventory(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    write_table(output_path, list(rows), schema=der_schema())


def build_reopt_request_payload(facility: Mapping[str, Any], *, load_profile_kw: list[float], critical_load_fraction: float, outage_duration_hours: int, outage_start_hour: int, electric_tariff: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"Site": {"latitude": facility["lat"], "longitude": facility["lon"]}, "ElectricLoad": {"loads_kw": list(load_profile_kw), "critical_load_fraction": critical_load_fraction, "year": default_reopt_load_year}, "ElectricTariff": dict(electric_tariff or default_electric_tariff), "ElectricUtility": {"outage_start_time_step": outage_start_hour, "outage_end_time_step": outage_start_hour + outage_duration_hours - 1}, "PV": {}, "ElectricStorage": {}, "Generator": {}}


def parse_reopt_results(results: Mapping[str, Any]) -> dict[str, Any]:
    outputs = results.get("outputs", {}) or {}
    return {"pv_kw": float((outputs.get("PV") or {}).get("size_kw", 0.0)), "bess_kw": float((outputs.get("ElectricStorage") or {}).get("size_kw", 0.0)), "bess_kwh": float((outputs.get("ElectricStorage") or {}).get("size_kwh", 0.0)), "genset_kw": float((outputs.get("Generator") or {}).get("size_kw", 0.0)), "reopt_feasible": bool((outputs.get("Outages") or {}).get("critical_loads_met", False))}


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_job_submit_request(payload: Mapping[str, Any], *, api_key: str) -> tuple[str, dict[str, Any]]:
    return f"{reopt_base_url}/job/?api_key={api_key}", dict(payload)


def build_results_poll_request(run_uuid: str, *, api_key: str) -> str:
    return f"{reopt_base_url}/job/{run_uuid}/results/?api_key={api_key}"


def is_terminal_status(status: str) -> bool:
    return status in terminal_statuses


def load_nlr_api_key(*, env_var: str = "NLR_API_KEY", fallback_path: Path | None = None) -> str:
    value = os.environ.get(env_var)
    if value:
        return value.strip()
    if fallback_path is not None and fallback_path.exists():
        return fallback_path.read_text(encoding="utf-8").strip()
    raise ReoptError(
        f"NLR API key not found: set {env_var} env var or provide a fallback file at {fallback_path}"
    )


class ReoptClient:
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
            raise ReoptError(f"REopt submit did not return a run_uuid: {submit_response!r}")
        poll_url = build_results_poll_request(str(run_uuid), api_key=self._api_key)
        for _ in range(self._max_poll_attempts):
            results = self._http_get(poll_url)
            status = str(results.get("status", ""))
            if is_terminal_status(status):
                if status in {"error", "Error"}:
                    raise ReoptError(f"REopt job {run_uuid} returned error: {results.get('messages') or results}")
                self._cache_store(payload, results)
                return results
            self._sleep(self._poll_interval_s)
        raise ReoptError(
            f"REopt job {run_uuid} did not reach a terminal status after {self._max_poll_attempts} polls"
        )

    def _cache_path(self, payload: Mapping[str, Any]) -> Path | None:
        return None if self._cache_dir is None else self._cache_dir / f"{_digest(payload)}.json"

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
        path.write_text(json.dumps(_redact_reopt_secrets(results), sort_keys=True), encoding="utf-8")


class CachedReoptResultsClient:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        digest = _digest(payload)
        path = self.cache_dir / f"{digest}.json"
        if not path.exists():
            raise FileNotFoundError(f"REopt cache miss for payload digest {digest}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


def default_reopt_client(
    *,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    poll_interval_s: float = 5.0,
    max_poll_attempts: int = 240,
    fallback_key_path: Path = default_nlr_api_key_file,
) -> ReoptClient:
    return ReoptClient(
        _urllib_post,
        _urllib_get,
        api_key=api_key or load_nlr_api_key(fallback_path=fallback_key_path),
        cache_dir=cache_dir,
        poll_interval_s=poll_interval_s,
        max_poll_attempts=max_poll_attempts,
    )


def _redact_reopt_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: "<redacted>" if key == "api_key" else _redact_reopt_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_reopt_secrets(item) for item in value]
    return value


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _urllib_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    from urllib import request
    from urllib.error import HTTPError

    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req) as resp:  # noqa: S310 -- trusted REopt API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ReoptError(_http_error_message(exc)) from exc


def _urllib_get(url: str) -> dict[str, Any]:
    from urllib import request
    from urllib.error import HTTPError

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
        except Exception:
            body = ""
    if body:
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                decoded = _redact_reopt_secrets(decoded)
                keys = ("status", "messages", "errors", "error", "run_uuid", "api_version")
                decoded = {key: decoded[key] for key in keys if key in decoded} or decoded
            body = json.dumps(decoded, sort_keys=True)
        except json.JSONDecodeError:
            body = body[:1000]
    return f"REopt HTTP {exc.code} {exc.reason}: {body}".strip()


def validate_reopt_sizing_inputs(rows: Iterable[Mapping[str, Any]], *, facility_lookup: Mapping[str, Mapping[str, Any]], load_profiles_kw: Mapping[str, list[float]], tier_clf_map: Mapping[str, float] | None = None, expected_hours: int = hours_per_year) -> dict[str, int]:
    clf_map = dict(tier_clf_map or default_tier_clf_map)
    checked = skipped_profile = skipped_tier = 0
    for row in rows:
        fid = str(row.get("facility_id"))
        facility = facility_lookup.get(fid)
        if facility is None:
            raise ReoptSizingInputViolation(f"{row.get('der_id')}: missing facility row")
        tier = facility.get("criticality_tier")
        if tier not in clf_map:
            skipped_tier += 1; continue
        profile = load_profiles_kw.get(fid)
        if profile is None:
            skipped_profile += 1; continue
        if row.get("assignment_status") != "assigned" or not (present(row.get("bus")) or present(row.get("block_id"))):
            raise ReoptSizingInputViolation(f"{row.get('der_id')}: live sizing requires assigned bus or block")
        if len(profile) != expected_hours:
            raise ReoptSizingInputViolation(f"{row.get('der_id')}: profile must contain {expected_hours} values")
        checked += 1
    return {"reopt_ready_rows": checked, "reopt_skipped_missing_profile": skipped_profile, "reopt_skipped_tier": skipped_tier}


def apply_layer_2_reopt_sizing(layer_1_rows: Iterable[Mapping[str, Any]], *, facility_lookup: Mapping[str, Mapping[str, Any]], load_profiles_kw: Mapping[str, list[float]], reopt_client: Callable[[dict[str, Any]], dict[str, Any]], electric_tariffs: Mapping[str, Mapping[str, Any]] | None = None, electric_tariff_provenance: Mapping[str, Mapping[str, Any]] | None = None, load_profile_provenance: Mapping[str, Mapping[str, Any]] | None = None, tier_clf_map: Mapping[str, float] | None = None, outage_duration_hours: int = fema_community_lifelines_outage_hours, outage_start_hour: int = 4392) -> list[dict[str, Any]]:
    rows = [dict(r) for r in layer_1_rows]
    clf_map = dict(tier_clf_map or default_tier_clf_map)
    validate_reopt_sizing_inputs(rows, facility_lookup=facility_lookup, load_profiles_kw=load_profiles_kw, tier_clf_map=clf_map)
    out: list[dict[str, Any]] = []
    for row in rows:
        fid = str(row["facility_id"])
        facility, profile = facility_lookup.get(fid), load_profiles_kw.get(fid)
        clf = clf_map.get((facility or {}).get("criticality_tier"))
        if facility is None or profile is None or clf is None:
            out.append(row); continue
        payload = build_reopt_request_payload(facility, load_profile_kw=profile, critical_load_fraction=clf, outage_duration_hours=outage_duration_hours, outage_start_hour=outage_start_hour, electric_tariff=(electric_tariffs or {}).get(fid))
        response = reopt_client(payload)
        upgraded = {**row, **parse_reopt_results(response), "placement_rule": placement_rule_layer_2, "critical_load_fraction": clf, "outage_duration_hours": outage_duration_hours}
        upgraded["source_provenance"] = json.dumps({"placement_rule": placement_rule_layer_2, "reopt_status": response.get("status"), "reopt_run_uuid": response.get("run_uuid"), "reopt_payload_digest": _digest(payload), "tier": (facility or {}).get("criticality_tier"), "critical_load_fraction": clf, "load_profile_provenance": (load_profile_provenance or {}).get(fid), "electric_tariff_provenance": (electric_tariff_provenance or {}).get(fid), "layer_1_source_provenance": row.get("source_provenance")}, sort_keys=True)
        out.append(upgraded)
    return out


class OfflineReoptSurrogateClient:
    def __init__(self, *, reserve_margin: float = 0.15, capacity_step_kw: float = 5.0) -> None:
        if reserve_margin < 0 or capacity_step_kw <= 0:
            raise ValueError("reserve_margin must be non-negative and capacity_step_kw positive")
        self.reserve_margin = reserve_margin; self.capacity_step_kw = capacity_step_kw

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        loads = [float(v) for v in (payload.get("ElectricLoad") or {}).get("loads_kw", [])]
        clf = float((payload.get("ElectricLoad") or {}).get("critical_load_fraction", 1.0))
        util = payload.get("ElectricUtility") or {}
        window = _wrapped_window(loads, start_time_step=int(util.get("outage_start_time_step", 1)), end_time_step=int(util.get("outage_end_time_step", len(loads))))
        peak = max((v * clf for v in window), default=0.0)
        genset = _round_up_to_step(peak * (1 + self.reserve_margin), self.capacity_step_kw)
        digest = _digest(payload)
        return {"status": "optimal_offline_surrogate", "run_uuid": f"offline-surrogate-{digest[:16]}", "api_version": "local_offline_surrogate", "reopt_version": "offline_reopt_surrogate.v0.1", "outputs": {"PV": {"size_kw": 0.0}, "ElectricStorage": {"size_kw": 0.0, "size_kwh": 0.0}, "Generator": {"size_kw": genset}, "Outages": {"critical_loads_met": genset >= peak and peak > 0.0, "peak_critical_kw": peak}}}


def _wrapped_window(values: list[float], *, start_time_step: int, end_time_step: int) -> list[float]:
    if not values:
        return []
    count = end_time_step - start_time_step + 1
    if count <= 0:
        raise ValueError("end_time_step must be >= start_time_step")
    start = (start_time_step - 1) % len(values)
    return [values[(start + i) % len(values)] for i in range(count)]


def _round_up_to_step(value: float, step: float) -> float:
    return 0.0 if value <= 0 else math.ceil(value / step) * step


@dataclass(frozen=True)
class ReoptSizingResult:
    der_inventory_path: Path
    total_rows: int
    attempted_rows: int
    reopt_sized_rows: int
    provisional_rows: int


def _load_reopt_inputs(augmented_dir: Path) -> dict[str, Any]:
    facilities = pd.read_parquet(augmented_dir / "critical_facilities.parquet").to_dict("records")
    assignments = pd.read_parquet(augmented_dir / "load_profile_assignments.parquet").to_dict("records")
    lookup = {str(r["facility_id"]): {**r, "criticality_tier": tier_int_to_string.get(r.get("criticality_tier"), r.get("criticality_tier"))} for r in facilities}
    profiles: dict[str, list[float]] = {}; profile_prov: dict[str, dict[str, Any]] = {}; tariffs: dict[str, dict[str, Any]] = {}; tariff_prov: dict[str, dict[str, Any]] = {}
    for row in assignments:
        prov = json.loads(row["source_provenance"])
        fid = str(prov["facility_id"])
        archetype = {"profile_source": row["profile_source"], "source_building_type": row["source_building_type"], "source_geography": row["source_geography"], "schedule_overlay": prov.get("schedule_overlay", "business_hours")}
        peak = float(row["peak_kw"])
        profiles[fid] = build_archetype_load_profile(archetype, peak_kw=peak)
        profile_prov[fid] = {**archetype, "loadshape_id": row.get("loadshape_id"), "synthetic_placeholder": bool(prov.get("synthetic_placeholder", False))}
        tariff = select_eversource_south_shore_tariff(lookup[fid], peak_kw=peak, customer_class=str(row["customer_class"]))
        tariffs[fid] = tariff.reopt_electric_tariff(); tariff_prov[fid] = tariff.provenance()
    return {"facility_lookup": lookup, "load_profiles_kw": profiles, "load_profile_provenance_by_facility": profile_prov, "electric_tariffs_by_facility": tariffs, "electric_tariff_provenance_by_facility": tariff_prov}


def size_der(*, smart_ds_compat_dir: Path, reopt_client: Any, outage_duration_hours: int = fema_community_lifelines_outage_hours, outage_start_hour: int = 4392, live_limit: int | None = None) -> ReoptSizingResult:
    path = smart_ds_compat_dir / "der_inventory.parquet"
    rows = pd.read_parquet(path).to_dict("records")
    target = rows if live_limit is None else rows[:live_limit]
    inputs = _load_reopt_inputs(smart_ds_compat_dir)
    sized = apply_layer_2_reopt_sizing(target, facility_lookup=inputs["facility_lookup"], load_profiles_kw=inputs["load_profiles_kw"], reopt_client=reopt_client, electric_tariffs=inputs["electric_tariffs_by_facility"], electric_tariff_provenance=inputs["electric_tariff_provenance_by_facility"], load_profile_provenance=inputs["load_profile_provenance_by_facility"], outage_duration_hours=outage_duration_hours, outage_start_hour=outage_start_hour)
    by_id = {r["der_id"]: r for r in sized}
    out = [by_id.get(r["der_id"], dict(r)) for r in rows]
    write_der_inventory(out, path)
    n_sized = sum(1 for r in out if r.get("placement_rule") == placement_rule_layer_2)
    return ReoptSizingResult(path, len(out), len(target), n_sized, len(out) - n_sized)


def run_layer_2_offline_reopt_surrogate_sizing(*, smart_ds_compat_dir: Path, outage_duration_hours: int = fema_community_lifelines_outage_hours, outage_start_hour: int = 4392, reserve_margin: float = 0.15, capacity_step_kw: float = 5.0, live_limit: int | None = None) -> ReoptSizingResult:
    return size_der(smart_ds_compat_dir=smart_ds_compat_dir, reopt_client=OfflineReoptSurrogateClient(reserve_margin=reserve_margin, capacity_step_kw=capacity_step_kw), outage_duration_hours=outage_duration_hours, outage_start_hour=outage_start_hour, live_limit=live_limit)
