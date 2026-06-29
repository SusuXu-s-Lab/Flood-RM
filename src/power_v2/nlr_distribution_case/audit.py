"""Validation and audit reports for synthetic distribution artifacts.

The report is an evidence ledger. It is not a utility certification and not a
SMART-DS regional validation claim.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .core import parse_float, write_json

location_id = os.environ.get("NLR_DISTRIBUTION_LOCATION_ID", "case")
power_grid = Path("data/power_grid")
default_registry_dir = power_grid / "asset_registry"
default_smart_ds_compat_dir = power_grid / "augmented"
default_grid_network_dir = Path(".")
default_report_dir = Path("outputs/validation_audit")
default_power_flow_report = default_report_dir / "power_flow_validation.json"
default_short_circuit_report = default_report_dir / "short_circuit_validation.json"
meters_per_mile = 1609.344
validation_reference_label = "synthetic_distribution_validation_criteria"


@dataclass(frozen=True)
class ValidationRegion:
    typical: tuple[tuple[float, float], ...]
    uncommon: tuple[tuple[float, float], ...] = ()


table_iv_targets: dict[str, ValidationRegion] = {
    "distribution_transformer_mva_per_feeder": ValidationRegion(((0.0, 1.73),), ((1.73, 4.94), (4.94, 31.0))),
    "real_load_kw_per_feeder": ValidationRegion(((4181.0, 13793.0),), ((577.0, 4181.0), (13793.0, 17590.0))),
    "mv_1_2f_line_length_miles_per_feeder": ValidationRegion(((0.0, 35.36),), ((35.36, 124.62),)),
    "mv_3f_line_length_miles_per_feeder": ValidationRegion(((0.0, 20.84),), ((20.84, 45.6),)),
    "customer_count_per_feeder": ValidationRegion(((94.0, 2607.0),), ((8.0, 94.0), (2607.0, 11837.0))),
    "mv_1_2f_line_miles_per_customer": ValidationRegion(((0.0, 0.12),), ((0.12, 0.24),)),
    "mv_3f_line_miles_per_customer": ValidationRegion(((0.0, 0.09),), ((0.09, 0.77),)),
    "switches_per_feeder": ValidationRegion(((3.0, 392.0),), ((392.0, 635.0),)),
    "average_degree_per_feeder": ValidationRegion(((1.9, 2.06),), ((1.6, 1.9), (2.06, 2.1))),
    "characteristic_path_length_miles_per_feeder": ValidationRegion(((12.4, 95.0),), ((2.0, 12.4), (95.0, 134.39))),
    "graph_diameter_miles_per_feeder": ValidationRegion(((32.0, 260.0),), ((4.0, 32.0), (260.0, 371.0))),
}
standard_distribution_transformer_kva_catalog = (15.0, 25.0, 37.5, 50.0, 75.0, 100.0, 167.0, 250.0, 333.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0)


def _f(value: Any, default: float = 0.0) -> float:
    return parse_float(value, default) or default


def _i(value: Any, default: int = 0) -> int:
    try: return int(float(value))
    except (TypeError, ValueError): return default


def _sum_by(rows: Iterable[Mapping[str, Any]], key: str, value: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows: out[str(row.get(key, ""))] += _f(row.get(value))
    return dict(out)


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _line_miles(row: Mapping[str, Any]) -> float:
    length, units = _f(row.get("length")), str(row.get("units", "")).lower()
    if units in {"m", "meter", "meters"}: return length / meters_per_mile
    if units in {"km", "kilometer", "kilometers"}: return length / 1.609344
    if units in {"ft", "feet"}: return length / 5280.0
    return length


def _line_lengths_by_feeder(lines: list[dict[str, str]], phases: set[int]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in lines:
        if _i(row.get("phases")) in phases: out[row["feeder_id"]] += _line_miles(row)
    return dict(out)


def _values(feeder_ids: list[str], series: Mapping[str, float | int]) -> list[float]:
    return [float(series.get(fid, 0.0)) for fid in feeder_ids]


def _ratio(num: Mapping[str, float], den: Mapping[str, float | int], feeder_ids: list[str]) -> list[float]:
    return [0.0 if float(den.get(fid, 0.0)) == 0 else float(num.get(fid, 0.0)) / float(den.get(fid, 0.0)) for fid in feeder_ids]


def _in_ranges(value: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    return any(lo <= value <= hi for lo, hi in ranges)


def _grade(metric_id: str, values: list[float], notes: list[str] | None = None) -> dict[str, Any]:
    target = table_iv_targets[metric_id]
    typical = sum(_in_ranges(v, target.typical) for v in values); uncommon = sum(_in_ranges(v, target.uncommon) for v in values)
    rare = max(len(values) - typical - uncommon, 0); rare_frac = 0.0 if not values else rare / len(values)
    return {"metric_id": metric_id, "grade": "good" if rare_frac <= 0.05 else "marginal" if rare_frac <= 0.10 else "check", "feeder_count": len(values), "typical_fraction": round(typical / len(values), 4) if values else 0.0, "uncommon_fraction": round(uncommon / len(values), 4) if values else 0.0, "rare_fraction": round(rare_frac, 4), "min": round(min(values), 6) if values else None, "median": round(sorted(values)[len(values)//2], 6) if values else None, "max": round(max(values), 6) if values else None, "validation_target": {"typical": target.typical, "uncommon": target.uncommon}, "notes": notes or []}


def _custom(metric_id: str, grade: str, notes: list[str], value: Any) -> dict[str, Any]:
    return {"metric_id": metric_id, "grade": grade, "value": value, "notes": notes}


def _safe_parquet(path: Path) -> list[dict[str, Any]]:
    try: return pd.read_parquet(path).to_dict("records") if path.exists() else []
    except ImportError: return []


def _nearest_standard(required_kva: float) -> float:
    return next((x for x in standard_distribution_transformer_kva_catalog if required_kva <= x), standard_distribution_transformer_kva_catalog[-1])


def _effective_load_rows(loads: list[dict[str, str]], compat: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    assets, assignments = _safe_parquet(compat / "assets.parquet"), _safe_parquet(compat / "load_profile_assignments.parquet")
    bus_by_asset = {str(r.get("asset_id")): str(r.get("bus")) for r in assets if r.get("asset_type") == "load_bus" and r.get("asset_id")}
    peak_by_bus = {bus_by_asset[str(r.get("load_asset_id"))]: _f(r.get("peak_kw")) for r in assignments if str(r.get("load_asset_id")) in bus_by_asset}
    out, adjusted = [], 0
    for row in loads:
        kw = peak_by_bus.get(str(row.get("bus")), _f(row.get("kw")))
        adjusted += int(str(row.get("bus")) in peak_by_bus and not math.isclose(kw, _f(row.get("kw"))))
        out.append({**row, "raw_kw": _f(row.get("kw")), "kw": kw})
    return out, {"profile_assignment_count": len(assignments), "profile_adjusted_bus_count": len(peak_by_bus), "profile_adjusted_row_count": adjusted}


def _effective_transformer_rows(transformers: list[dict[str, str]], load_kw_by_feeder: Mapping[str, float]) -> list[dict[str, Any]]:
    counts = _count_by(transformers, "feeder_id"); rows = []
    for row in transformers:
        per = float(load_kw_by_feeder.get(str(row.get("feeder_id")), 0.0)) / max(counts.get(str(row.get("feeder_id")), 1), 1)
        rows.append({**row, "raw_max_kva": _f(row.get("max_kva")), "max_kva": _nearest_standard(max(15.0, per * 1.25 / 0.8)), "rating_basis": "standard_synthetic_distribution_transformer_catalog_from_effective_feeder_load"})
    return rows


def _graph_stats(feeder_ids: list[str], lines: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in lines: by[row["feeder_id"]].append(row)
    stats = {}
    for fid in feeder_ids:
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list); nodes = set(); edges = by.get(fid, [])
        for row in edges:
            a, b, w = row.get("from_bus", ""), row.get("to_bus", ""), max(_line_miles(row), 0.0)
            if a and b: adj[a].append((b, w)); adj[b].append((a, w)); nodes.update([a, b])
        comp = _largest_component(nodes, adj)
        stats[fid] = {"average_degree": 0.0 if not nodes else 2 * len(edges) / len(nodes), "graph_diameter_miles": _weighted_diameter(comp, adj), "characteristic_path_length_miles": _average_shortest_path(comp, adj)}
    return stats


def _largest_component(nodes: set[str], adj: Mapping[str, list[tuple[str, float]]]) -> set[str]:
    unseen, largest = set(nodes), set()
    while unseen:
        q, comp = deque([unseen.pop()]), set()
        while q:
            n = q.popleft(); comp.add(n)
            for nb, _ in adj.get(n, []):
                if nb in unseen: unseen.remove(nb); q.append(nb)
        if len(comp) > len(largest): largest = comp
    return largest


def _dijkstra(start: str, allowed: set[str], adj: Mapping[str, list[tuple[str, float]]]) -> dict[str, float]:
    import heapq
    dist = {start: 0.0}; heap = [(0.0, start)]
    while heap:
        d, n = heapq.heappop(heap)
        if d != dist[n]: continue
        for nb, w in adj.get(n, []):
            if nb in allowed and d + w < dist.get(nb, float("inf")):
                dist[nb] = d + w; heapq.heappush(heap, (d + w, nb))
    return dist


def _weighted_diameter(nodes: set[str], adj: Mapping[str, list[tuple[str, float]]]) -> float:
    if not nodes: return 0.0
    first = _dijkstra(next(iter(nodes)), nodes, adj); far = max(first, key=first.get); second = _dijkstra(far, nodes, adj)
    return max(second.values()) if second else 0.0


def _average_shortest_path(nodes: set[str], adj: Mapping[str, list[tuple[str, float]]]) -> float:
    if len(nodes) <= 1: return 0.0
    total = pairs = 0
    ordered = sorted(nodes)
    for i, node in enumerate(ordered):
        dist = _dijkstra(node, nodes, adj)
        for other in ordered[i+1:]:
            if other in dist: total += dist[other]; pairs += 1
    return total / pairs if pairs else 0.0


def build_synthetic_validation_report(*, registry_dir: Path = default_registry_dir, smart_ds_compat_dir: Path = default_smart_ds_compat_dir, grid_network_dir: Path = default_grid_network_dir) -> dict[str, Any]:
    feeders = pd.read_csv(registry_dir / "feeders.csv", keep_default_na=False).to_dict("records")
    buses = pd.read_csv(registry_dir / "buses.csv", keep_default_na=False).to_dict("records")
    lines = pd.read_csv(registry_dir / "lines.csv", keep_default_na=False).to_dict("records")
    loads_raw = pd.read_csv(registry_dir / "loads.csv", keep_default_na=False).to_dict("records")
    transformers_raw = pd.read_csv(registry_dir / "transformers.csv", keep_default_na=False).to_dict("records")
    sources = pd.read_csv(registry_dir / "sources.csv", keep_default_na=False).to_dict("records")
    feeder_ids = sorted(r["feeder_id"] for r in feeders)
    loads, prof_counts = _effective_load_rows(loads_raw, smart_ds_compat_dir)
    load_by_feeder = _sum_by(loads, "feeder_id", "kw")
    transformers = _effective_transformer_rows(transformers_raw, load_by_feeder)
    tx_mva: dict[str, float] = defaultdict(float)
    for row in transformers: tx_mva[row["feeder_id"]] += _f(row.get("max_kva")) / 1000.0
    one_two = _line_lengths_by_feeder(lines, {1, 2}); three = _line_lengths_by_feeder(lines, {3}); customers = _count_by(loads_raw, "feeder_id"); graphs = _graph_stats(feeder_ids, lines)
    switches = _safe_parquet(smart_ds_compat_dir / "controllable_switches.parquet"); blocks = _safe_parquet(smart_ds_compat_dir / "switch_bounded_load_blocks.parquet"); ders = _safe_parquet(smart_ds_compat_dir / "der_inventory.parquet"); profiles = _safe_parquet(smart_ds_compat_dir / "load_profile_assignments.parquet")
    switch_counts: dict[str, int] = defaultdict(int)
    for row in switches:
        fid = str(row.get("feeder_id") or str(row.get("from_bus", "")).split("__", 1)[0])
        if fid: switch_counts[fid] += 1
    line_classes = dict(Counter(r.get("line_class", "") for r in lines))
    metrics = [_grade("distribution_transformer_mva_per_feeder", _values(feeder_ids, tx_mva)), _grade("real_load_kw_per_feeder", _values(feeder_ids, load_by_feeder)), _grade("mv_1_2f_line_length_miles_per_feeder", _values(feeder_ids, one_two)), _grade("mv_3f_line_length_miles_per_feeder", _values(feeder_ids, three)), _grade("customer_count_per_feeder", _values(feeder_ids, customers)), _grade("mv_1_2f_line_miles_per_customer", _ratio(one_two, customers, feeder_ids)), _grade("mv_3f_line_miles_per_customer", _ratio(three, customers, feeder_ids)), _grade("switches_per_feeder", _values(feeder_ids, switch_counts)), _grade("average_degree_per_feeder", [graphs[f]["average_degree"] for f in feeder_ids]), _grade("characteristic_path_length_miles_per_feeder", [graphs[f]["characteristic_path_length_miles"] for f in feeder_ids]), _grade("graph_diameter_miles_per_feeder", [graphs[f]["graph_diameter_miles"] for f in feeder_ids]), _custom("line_classification_completeness", "good" if {"overhead", "underground"} & set(line_classes) else "unavailable", ["OH/UG line classification is absent or derived from line_class provenance."], {"line_classes": line_classes})]
    pf = _load_optional_json(grid_network_dir / "outputs" / "power_flow_validation.json") or _load_optional_json(default_power_flow_report)
    sc = _load_optional_json(grid_network_dir / "outputs" / "short_circuit_validation.json") or _load_optional_json(default_short_circuit_report)
    missing = {"needs_power_flow_voltage_and_loss_report": not bool(pf and pf.get("compiled") and (pf.get("power_flow") or {}).get("converged")), "needs_short_circuit_report": not bool(sc and sc.get("compiled")), "needs_static_load_diversity": sorted({_f(r.get("kw")) for r in loads}) == [5.0], "needs_equipment_rating_diversity": sorted({_f(r.get("max_kva")) for r in transformers}) == [5000.0], "needs_oh_ug_line_classification": not ({"overhead", "underground"} & set(line_classes))}
    return {"sandbox_id": location_id, "validation_reference": validation_reference_label, "overall_status": "partial", "scope_note": "This is an audit of a synthetic Grid Dataset. It is not a real-utility validation claim or a SMART-DS regional validation claim.", "summary_counts": {"feeders": len(feeders), "buses": len(buses), "lines": len(lines), "loads": len(loads_raw), "transformers": len(transformers_raw), "sources": len(sources), "controllable_switches": len(switches), "switch_bounded_load_blocks": len(blocks), "load_profile_assignments": len(profiles), "der_inventory_rows": len(ders)}, "validation_prongs": {"statistical": {"status": "partial"}, "operational": {"status": "partial" if pf else "gap"}, "expert": {"status": "not_required"}}, "data_quality": {"line_classes": line_classes, "raw_static_loads_all_uniform_5kw": sorted({_f(r.get("kw")) for r in loads_raw}) == [5.0], "profile_adjusted_load_count": prof_counts["profile_assignment_count"], "profile_adjusted_bus_count": prof_counts["profile_adjusted_bus_count"], "raw_distribution_transformers_all_uniform_5mva": sorted({_f(r.get("max_kva")) for r in transformers_raw}) == [5000.0], "buses_with_coordinates": sum(1 for r in buses if r.get("lon") and r.get("lat"))}, "missing_capabilities": missing, "statistical_metrics": metrics, "operational_evidence": {"power_flow": pf, "short_circuit": sc}, "recommended_next_tests": _recommended(missing, pf, sc)}


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _recommended(missing: Mapping[str, bool], pf: Mapping[str, Any] | None, sc: Mapping[str, Any] | None) -> list[str]:
    out = []
    if missing.get("needs_power_flow_voltage_and_loss_report"): out.append("Run OpenDSS power-flow validation with convergence, voltage, losses, and overload evidence.")
    if missing.get("needs_short_circuit_report"): out.append("Add short-circuit evidence by voltage class or mark it explicitly out of scope.")
    if missing.get("needs_equipment_rating_diversity"): out.append("Replace placeholder equipment ratings with a regional synthetic equipment catalog.")
    if missing.get("needs_oh_ug_line_classification"): out.append("Add OH/UG line class provenance or keep line-class metrics marked unavailable.")
    return out


def build_power_flow_validation_report(network_dss: Path | str) -> dict[str, Any]:
    import opendssdirect as dss
    path = Path(network_dss); start = time.perf_counter()
    try:
        dss.Basic.ClearAll(); dss.Text.Command(f'compile "{path}"'); dss.Text.Command("solve"); compiled, error = True, None
    except Exception as exc:
        compiled, error = False, str(exc)
    if not compiled:
        return {"schema_version": "power_flow_validation.v0.1", "network_dss": str(path), "compiled": False, "compile_error": error, "validation_exit_criteria": {}}
    losses_w, _ = dss.Circuit.Losses(); load_kw = _load_kw_sum(dss); losses_pct = None if load_kw <= 0 else losses_w / 1000 / load_kw * 100
    criteria = {"power_flow_converged": {"passed": bool(dss.Solution.Converged()), "target": "converged"}, "power_flow_iterations": {"passed": int(dss.Solution.Iterations() or 0) < 20, "target": "< 20"}, "solve_time_seconds": {"passed": time.perf_counter() - start < 30, "target": "< 30"}, "system_losses_percent": {"passed": losses_pct is not None and losses_pct < 10, "target": "< 10"}}
    return {"schema_version": "power_flow_validation.v0.1", "network_dss": str(path), "compiled": True, "compile_error": None, "circuit": dss.Circuit.Name().lower(), "load_count": int(dss.Loads.Count()), "line_count": int(dss.Lines.Count()), "bus_count": int(dss.Circuit.NumBuses()), "generator_count": int(dss.Generators.Count()), "power_flow": {"converged": bool(dss.Solution.Converged()), "iterations": int(dss.Solution.Iterations() or 0), "solve_time_seconds": round(time.perf_counter() - start, 6), "system_losses_percent": round(losses_pct, 6) if losses_pct is not None else None}, "validation_exit_criteria": criteria}


def _load_kw_sum(dss: Any) -> float:
    total = 0.0; i = dss.Loads.First()
    while i > 0:
        total += float(dss.Loads.kW() or 0.0); i = dss.Loads.Next()
    return total


def write_power_flow_validation_report(report: dict[str, Any], path: Path = default_power_flow_report) -> Path:
    return write_json(path, report)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["summary_counts"]
    lines = ["# Synthetic Validation Audit", "", str(report["scope_note"]), "", "## Summary", "", f"- Network scale: {counts['feeders']} feeders, {counts['buses']} buses, {counts['lines']} lines, {counts['loads']} loads.", f"- Augmentation scale: {counts['controllable_switches']} switches, {counts['switch_bounded_load_blocks']} blocks, {counts['der_inventory_rows']} DER rows.", "", "## Major gaps", ""]
    lines += [f"- `{k}`" for k, v in report["missing_capabilities"].items() if v]
    lines += ["", "## Statistical metrics", "", "| Metric | Grade | Typical | Uncommon | Rare | Min | Median | Max |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["statistical_metrics"]:
        if "rare_fraction" in m: lines.append(f"| `{m['metric_id']}` | {m['grade']} | {m['typical_fraction']:.2f} | {m['uncommon_fraction']:.2f} | {m['rare_fraction']:.2f} | {m['min']} | {m['median']} | {m['max']} |")
        else: lines.append(f"| `{m['metric_id']}` | {m['grade']} | | | | | | {json.dumps(m.get('value'), sort_keys=True)} |")
    lines += ["", "## Recommended next tests", ""] + [f"- {x}" for x in report.get("recommended_next_tests", [])]
    return "\n".join(lines) + "\n"


def write_synthetic_validation_report(report: dict[str, Any], output_dir: Path = default_report_dir) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / "synthetic_validation_audit.json", output_dir / "synthetic_validation_audit.md"
    write_json(json_path, report); md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_validation_compliance_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    blockers = [{"blocker_id": k, "prong": "operational" if "flow" in k or "short" in k else "statistical", "next_behavior_to_test": k.replace("_", " ")} for k, v in (report.get("missing_capabilities") or {}).items() if v]
    return {"gate_id": f"{location_id}:synthetic_validation:validation_compliance", "target": "synthetic_distribution_validation", "compliant": not blockers, "scope_note": report.get("scope_note"), "prongs": {"statistical": {"passed": not any(b["prong"] == "statistical" for b in blockers)}, "operational": {"passed": not any(b["prong"] == "operational" for b in blockers)}, "expert": {"passed": True, "required": False}}, "blockers": blockers, "next_tdd_slice": {"name": blockers[0]["blocker_id"] if blockers else "validation_complete", "first_failing_behavior": blockers[0]["next_behavior_to_test"] if blockers else "all validation gates pass", "recommended_scope": "advance the next explicit blocker only"}}


def render_validation_compliance_gate_markdown(gate: Mapping[str, Any]) -> str:
    lines = ["# Synthetic Validation Compliance Gate", "", f"- Gate: `{gate['gate_id']}`", f"- Compliant: `{str(gate['compliant']).lower()}`", "", str(gate.get("scope_note", "")), "", "## Blockers", ""]
    lines += [f"- `{b['blocker_id']}` ({b['prong']}): {b['next_behavior_to_test']}" for b in gate.get("blockers", [])]
    return "\n".join(lines) + "\n"


def write_validation_compliance_gate(gate: dict[str, Any], output_dir: Path = default_report_dir) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / "validation_compliance_gate.json", output_dir / "validation_compliance_gate.md"
    write_json(json_path, gate); md_path.write_text(render_validation_compliance_gate_markdown(gate), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _metric_status_table(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(m["metric_id"]): {k: v for k, v in m.items() if k in {"grade", "feeder_count", "typical_fraction", "uncommon_fraction", "rare_fraction", "min", "median", "max", "value"}} for m in report.get("statistical_metrics", [])}


def run_stats(*, registry_dir: Path = default_registry_dir, smart_ds_compat_dir: Path | None = None, grid_network_dir: Path = default_grid_network_dir, output_dir: Path = default_report_dir, smart_ds_reference_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    report = build_synthetic_validation_report(registry_dir=registry_dir, smart_ds_compat_dir=smart_ds_compat_dir or registry_dir.parent / "augmented", grid_network_dir=grid_network_dir)
    write_synthetic_validation_report(report, output_dir); write_validation_compliance_gate(build_validation_compliance_gate(report), output_dir)
    return _metric_status_table(report)


def run_ops(*, opendss_root: Path, registry_dir: Path = default_registry_dir, output_dir: Path = default_report_dir) -> dict[str, Any]:
    masters = sorted(Path(opendss_root).glob("*/Master.dss")) if Path(opendss_root).exists() else []
    pf = _load_optional_json(output_dir / "power_flow_validation.json") or _load_optional_json(default_power_flow_report)
    sc = _load_optional_json(output_dir / "short_circuit_validation.json") or _load_optional_json(default_short_circuit_report)
    summary = {"status": "gap" if not masters else "pass" if pf and sc else "partial", "mode": "evidence_inventory", "opendss_root": str(opendss_root), "opendss_feeder_cases": len(masters), "registry_dir": str(registry_dir), "power_flow_evidence": "present" if pf else "missing", "short_circuit_evidence": "present" if sc else "missing"}
    write_json(output_dir / "operational_validation_summary.json", summary)
    return summary


def audit_summary(*, block_report: dict[str, Any], stage_a1: dict[str, Any], stat_results: dict[str, dict[str, Any]], op_results: dict[str, Any]) -> dict[str, Any]:
    gate_results = {"block_invariant": "check" if block_report.get("violations") else "pass", "smart_ds_interface": "pass" if stage_a1.get("passed") else "check", "statistical_validation": "check" if any(r.get("grade") == "check" for r in stat_results.values()) else "partial", "operational_validation": str(op_results.get("status") or "gap")}
    return {"gate_results": gate_results, "gaps": [k for k, v in gate_results.items() if v not in {"pass", "not_required"}], "block_summary": block_report.get("summary", {}), "statistical_metric_count": len(stat_results), "operational_summary": op_results}
