# Resilience switch placement and block validation
"""Exact Sectionalizing Switch Allocation Problem (SSAP) and switch artifacts.

Each subtree DP state is ``(C, L, E, S)``:
``C`` is cost of already sealed zones, ``L`` and ``E`` are the residual load
and exposure still connected upward, and ``S`` is the chosen switch set.
For a child edge with exposure ``x``:

    no switch: (Ca + Cb, La + Lb, Ea + Eb + x)
    switch:    (Ca + Cb + Lb * (Eb + x), La, Ea)

At the root, objective = ``C + L * E``.  
The rest of the file: candidate gates, topology construction, and switch artifact rows.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

placement_rule_ssap = "ssap_radial_sectionalizing_switch_allocation"
ssap_subrule = "exact_tree_dynamic_programming_ssap"
controllable_switches_schema_version = "stage_b_controllable_switches.v0.1"

@dataclass(frozen=True)
class SsapEdge:
    upstream: str
    downstream: str
    exposure: float

@dataclass(frozen=True)
class SsapZone:
    nodes: tuple[str, ...]
    edges: tuple[SsapEdge, ...]
    load_kw: float
    exposure: float

@dataclass(frozen=True)
class RootedFeeder:
    root: str
    loads_kw: Mapping[str, float]
    edges: tuple[SsapEdge, ...]

    @property
    def children(self) -> Mapping[str, tuple[SsapEdge, ...]]:
        return _children_by_parent(self.edges)

@dataclass(frozen=True)
class SsapSolution:
    switch_edges: tuple[SsapEdge, ...]
    objective_value: float
    zones: tuple[SsapZone, ...]

@dataclass(frozen=True)
class SsapCandidatePolicy:
    min_edge_exposure: float = 0.0
    min_block_bus_count: int = 0
    min_block_load_kw: float = 0.0
    max_candidate_edges: int | None = None

@dataclass(frozen=True)
class SsapFrontierPoint:
    k_switches: int
    solution: SsapSolution
    marginal_benefit: float

    @property
    def objective_value(self) -> float:
        return self.solution.objective_value

@dataclass(frozen=True)
class SsapGlobalMarginalSelection:
    selected_points: Mapping[str, SsapFrontierPoint]
    selected_switch_count: int
    total_marginal_benefit: float
    total_objective_value: float
    marginal_benefit_floor: float

@dataclass(frozen=True)
class ComponentMeta:
    component_id: str
    feeder_id: str
    root_bus: str
    bus_count: int
    edge_count: int

@dataclass(frozen=True)
class _DPState:
    committed_cost: float
    residual_load: float
    residual_exposure: float
    switch_set: frozenset[SsapEdge]

@dataclass(frozen=True)
class _TreeView:
    children: Mapping[str, tuple[SsapEdge, ...]]
    post_order: tuple[str, ...]

    @classmethod
    def from_feeder(cls, feeder: RootedFeeder) -> _TreeView:
        children = _children_by_parent(feeder.edges)
        return cls(children, tuple(_post_order_from_children(feeder.root, children)))

# ---------------------------------------------------------------------------
# Exact SSAP solver

def solve_switches(
    feeder: RootedFeeder, *, k_switches: int, algorithm: str = "tree_dp",
    eligible_edges: Iterable[SsapEdge] | None = None,
) -> SsapSolution:
    switchable_edges = frozenset(feeder.edges) if eligible_edges is None else frozenset(eligible_edges)
    if k_switches < 0:
        raise ValueError("k_switches must be non-negative")
    if not switchable_edges.issubset(feeder.edges):
        raise ValueError("eligible_edges must be drawn from feeder.edges")
    if k_switches > len(switchable_edges):
        edge_label = "available" if eligible_edges is None else "eligible"
        raise ValueError(f"k_switches={k_switches} exceeds {edge_label} edges={len(switchable_edges)}")
    if algorithm == "tree_dp":
        return _solve_ssap_tree_dp(feeder, k_switches=k_switches, switchable_edges=switchable_edges)
    if algorithm == "brute_force":
        return _solve_ssap_brute_force(feeder, k_switches=k_switches, switchable_edges=switchable_edges)
    raise ValueError(f"unknown algorithm {algorithm!r}; expected 'tree_dp' or 'brute_force'")

def _solve_ssap_brute_force(
    feeder: RootedFeeder, *, k_switches: int, switchable_edges: frozenset[SsapEdge]
) -> SsapSolution:
    best: SsapSolution | None = None
    for switch_combo in combinations(feeder.edges, k_switches):
        if not set(switch_combo).issubset(switchable_edges):
            continue
        zones = tuple(_compute_zones(feeder, switch_set=frozenset(switch_combo)))
        candidate = SsapSolution(tuple(switch_combo), sum(z.load_kw * z.exposure for z in zones), zones)
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    assert best is not None
    return best

def _solve_ssap_tree_dp(
    feeder: RootedFeeder, *, k_switches: int, switchable_edges: frozenset[SsapEdge]
) -> SsapSolution:
    tree = _TreeView.from_feeder(feeder)
    states_by_node: dict[str, dict[int, list[_DPState]]] = {}
    for node in tree.post_order:
        accumulator = {0: [_DPState(0.0, float(feeder.loads_kw.get(node, 0.0)), 0.0, frozenset())]}
        for edge in tree.children.get(node, ()):  # fold children one at a time
            accumulator = _merge_child_buckets(
                accumulator, states_by_node[edge.downstream], edge,
                k_switches=k_switches, switchable_edges=switchable_edges,
            )
        states_by_node[node] = accumulator

    root_states = states_by_node[feeder.root]
    if k_switches not in root_states:
        raise ValueError(f"SSAP infeasible: no DP state at root with exactly {k_switches} switches")
    best = min(root_states[k_switches], key=_root_cost)
    switch_edges = tuple(best.switch_set)
    return SsapSolution(switch_edges, _root_cost(best), tuple(_compute_zones(feeder, switch_set=frozenset(switch_edges))))

def _merge_child_buckets(
    accumulator: dict[int, list[_DPState]], child_states: dict[int, list[_DPState]], edge: SsapEdge, *,
    k_switches: int, switchable_edges: frozenset[SsapEdge],
) -> dict[int, list[_DPState]]:
    merged: dict[int, list[_DPState]] = {}
    can_switch = edge in switchable_edges
    for k_acc, acc_states in accumulator.items():
        for k_child, child_bucket in child_states.items():
            k = k_acc + k_child
            for acc in acc_states:
                for child in child_bucket:
                    if k <= k_switches:
                        merged.setdefault(k, []).append(_merge_residual(acc, child, edge))
                    if can_switch and k + 1 <= k_switches:
                        merged.setdefault(k + 1, []).append(_seal_child(acc, child, edge))
    return {k: _pareto_prune(states) for k, states in merged.items()}

def _merge_residual(parent: _DPState, child: _DPState, edge: SsapEdge) -> _DPState:
    return _DPState(
        parent.committed_cost + child.committed_cost,
        parent.residual_load + child.residual_load,
        parent.residual_exposure + child.residual_exposure + edge.exposure,
        parent.switch_set | child.switch_set,
    )

def _seal_child(parent: _DPState, child: _DPState, edge: SsapEdge) -> _DPState:
    return _DPState(
        parent.committed_cost + child.committed_cost + child.residual_load * (child.residual_exposure + edge.exposure),
        parent.residual_load, parent.residual_exposure, parent.switch_set | child.switch_set | {edge},
    )

def _root_cost(state: _DPState) -> float:
    return state.committed_cost + state.residual_load * state.residual_exposure

def _pareto_prune(states: list[_DPState]) -> list[_DPState]:
    """Keep only nondominated ``(committed_cost, residual_load, residual_exposure)`` states."""
    if not states:
        return []
    states_sorted = sorted(states, key=lambda s: (s.committed_cost, s.residual_load, s.residual_exposure))
    load_values = sorted({s.residual_load for s in states_sorted})
    min_exposure = _PrefixMinTree(len(load_values))
    result: list[_DPState] = []
    seen_dimensions: set[tuple[float, float, float]] = set()
    for state in states_sorted:
        dimensions = (state.committed_cost, state.residual_load, state.residual_exposure)
        if dimensions in seen_dimensions:
            continue
        load_index = bisect_right(load_values, state.residual_load)
        if min_exposure.query(load_index) <= state.residual_exposure:
            continue
        result.append(state)
        seen_dimensions.add(dimensions)
        min_exposure.update(load_index, state.residual_exposure)
    return result

class _PrefixMinTree:
    def __init__(self, size: int) -> None:
        self._size = size
        self._tree = [float("inf")] * (size + 1)

    def update(self, index: int, value: float) -> None:
        while index <= self._size:
            self._tree[index] = min(self._tree[index], value)
            index += index & -index

    def query(self, index: int) -> float:
        best = float("inf")
        while index > 0:
            best = min(best, self._tree[index])
            index -= index & -index
        return best

# ---------------------------------------------------------------------------
# Candidate policy, frontiers, and global budget selection

def solve_rpop_ready_ssap(
    feeder: RootedFeeder, *, k_switches: int, policy: SsapCandidatePolicy | None = None,
    algorithm: str = "tree_dp", eligible_edges: Iterable[SsapEdge] | None = None,
) -> SsapSolution:
    policy = policy or SsapCandidatePolicy()
    candidates = tuple(feeder.edges) if eligible_edges is None else tuple(eligible_edges)
    return solve_switches(
        feeder, k_switches=k_switches, algorithm=algorithm,
        eligible_edges=select_policy_candidate_edges(feeder, candidates, policy),
    )

def solve_rpop_ready_ssap_frontier(
    feeder: RootedFeeder, *, max_switches: int, policy: SsapCandidatePolicy | None = None,
    algorithm: str = "tree_dp", eligible_edges: Iterable[SsapEdge] | None = None,
) -> tuple[SsapFrontierPoint, ...]:
    policy = policy or SsapCandidatePolicy()
    candidates = tuple(feeder.edges) if eligible_edges is None else tuple(eligible_edges)
    switchable = select_policy_candidate_edges(feeder, candidates, policy)
    points: list[SsapFrontierPoint] = []
    previous_objective: float | None = None
    for k_switches in range(min(max(0, max_switches), len(switchable)) + 1):
        solution = solve_switches(feeder, k_switches=k_switches, algorithm=algorithm, eligible_edges=switchable)
        benefit = 0.0 if previous_objective is None else max(0.0, previous_objective - solution.objective_value)
        points.append(SsapFrontierPoint(k_switches, solution, benefit))
        previous_objective = solution.objective_value
    return tuple(points)

def select_global_marginal_benefit_budget(
    frontiers: Mapping[str, tuple[SsapFrontierPoint, ...]], *, min_marginal_benefit: float,
    max_total_switches: int | None = None, min_per_feeder: int = 0,
) -> SsapGlobalMarginalSelection:
    if min_marginal_benefit < 0:
        raise ValueError("min_marginal_benefit must be non-negative")
    if min_per_feeder < 0:
        raise ValueError("min_per_feeder must be non-negative")
    selected_index = _initial_frontier_indices(frontiers)
    total_switches = 0
    total_benefit = 0.0

    def under_budget() -> bool:
        return max_total_switches is None or total_switches < max_total_switches
    def advance(component_id: str, point: SsapFrontierPoint) -> None:
        nonlocal total_switches, total_benefit
        selected_index[component_id] += 1
        total_switches += 1
        total_benefit += point.marginal_benefit

    while min_per_feeder > 0 and under_budget():
        pick = _best_next_frontier_point(frontiers, selected_index, max_index=lambda f: min(min_per_feeder, len(f) - 1))
        if pick is None:
            break
        advance(*pick)
    while under_budget():
        pick = _best_next_frontier_point(frontiers, selected_index, max_index=lambda f: len(f) - 1)
        if pick is None or pick[1].marginal_benefit < min_marginal_benefit:
            break
        advance(*pick)

    selected_points = {component_id: frontiers[component_id][idx] for component_id, idx in selected_index.items()}
    return SsapGlobalMarginalSelection(
        selected_points, total_switches, total_benefit,
        sum(point.objective_value for point in selected_points.values()), min_marginal_benefit,
    )

def _initial_frontier_indices(frontiers: Mapping[str, tuple[SsapFrontierPoint, ...]]) -> dict[str, int]:
    selected_index: dict[str, int] = {}
    for component_id, frontier in frontiers.items():
        if not frontier or frontier[0].k_switches != 0:
            raise ValueError("each frontier must start with k_switches=0")
        selected_index[component_id] = 0
    return selected_index

def _best_next_frontier_point(
    frontiers: Mapping[str, tuple[SsapFrontierPoint, ...]], selected_index: Mapping[str, int], *,
    max_index: Callable[[tuple[SsapFrontierPoint, ...]], int],
) -> tuple[str, SsapFrontierPoint] | None:
    best_component: str | None = None
    best_point: SsapFrontierPoint | None = None
    for component_id in sorted(frontiers):
        frontier = frontiers[component_id]
        next_index = selected_index[component_id] + 1
        if next_index > max_index(frontier):
            continue
        candidate = frontier[next_index]
        if best_point is None or (candidate.marginal_benefit, component_id) > (best_point.marginal_benefit, best_component or ""):
            best_component, best_point = component_id, candidate
    return None if best_component is None or best_point is None else (best_component, best_point)

def classify_two_tier_switches(rows: list[dict], *, automated_count: int) -> list[dict]:
    if automated_count < 0:
        raise ValueError("automated_count must be non-negative")
    automated_ids = {id(r) for r in sorted(rows, key=lambda r: float(r.get("marginal_benefit", 0.0)), reverse=True)[:automated_count]}
    out: list[dict] = []
    for row in rows:
        new_row = dict(row)
        is_automated = id(row) in automated_ids
        new_row["switch_role_class"] = "automated_sectionalizing" if is_automated else "manual_sectionalizing"
        new_row["dispatchable"] = is_automated
        out.append(new_row)
    return out

# ---------------------------------------------------------------------------
# Load/exposure conditioning and policy metrics

def build_customer_weighted_loads(
    *, loads_kw: Mapping[str, float], customers_by_bus: Mapping[str, float],
    kw_weight: float = 1.0, customer_weight: float = 0.0,
) -> dict[str, float]:
    if kw_weight < 0 or customer_weight < 0:
        raise ValueError("kw_weight and customer_weight must be non-negative")
    out = {bus: float(kw) * kw_weight + float(customers_by_bus.get(bus, 0.0)) * customer_weight for bus, kw in loads_kw.items()}
    out.update({bus: float(customers) * customer_weight for bus, customers in customers_by_bus.items() if bus not in out})
    return out

def homogenize_feeder_exposure(feeder: RootedFeeder, *, exposure: float = 1.0) -> RootedFeeder:
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    return RootedFeeder(feeder.root, feeder.loads_kw, tuple(SsapEdge(e.upstream, e.downstream, float(exposure)) for e in feeder.edges))

def select_policy_candidate_edges(
    feeder: RootedFeeder, candidate_edges: Iterable[SsapEdge], policy: SsapCandidatePolicy,
) -> tuple[SsapEdge, ...]:
    if policy.max_candidate_edges is not None and policy.max_candidate_edges < 0:
        raise ValueError("max_candidate_edges must be non-negative or None")
    metrics = _single_switch_metrics(feeder)
    switchable = tuple(edge for edge in candidate_edges if _edge_satisfies_candidate_metrics(edge, metrics[edge], policy))
    if policy.max_candidate_edges is None or len(switchable) <= policy.max_candidate_edges:
        return switchable
    return tuple(sorted(switchable, key=lambda e: (metrics[e]["benefit"], e.upstream, e.downstream), reverse=True)[: policy.max_candidate_edges])

def _edge_satisfies_candidate_metrics(edge: SsapEdge, metrics: Mapping[str, float], policy: SsapCandidatePolicy) -> bool:
    return not (
        edge.exposure < policy.min_edge_exposure
        or metrics["downstream_nodes"] < policy.min_block_bus_count
        or metrics["upstream_nodes"] < policy.min_block_bus_count
        or metrics["downstream_load"] < policy.min_block_load_kw
        or metrics["upstream_load"] < policy.min_block_load_kw
    )

def _single_switch_metrics(feeder: RootedFeeder) -> dict[SsapEdge, dict[str, float]]:
    tree = _TreeView.from_feeder(feeder)
    subtree_nodes, subtree_load, subtree_exposure = _subtree_stats(feeder, tree)
    total_nodes = subtree_nodes.get(feeder.root, 0)
    total_load = subtree_load.get(feeder.root, 0.0)
    total_exposure = subtree_exposure.get(feeder.root, 0.0)
    base_objective = total_load * total_exposure
    metrics: dict[SsapEdge, dict[str, float]] = {}
    for edge in feeder.edges:
        child = edge.downstream
        d_nodes = subtree_nodes[child]
        d_load = subtree_load[child]
        d_exposure = subtree_exposure[child] + edge.exposure
        u_nodes = total_nodes - d_nodes
        u_load = total_load - d_load
        u_exposure = total_exposure - d_exposure
        metrics[edge] = {
            "downstream_nodes": float(d_nodes), "upstream_nodes": float(u_nodes),
            "downstream_load": d_load, "upstream_load": u_load,
            "benefit": base_objective - d_load * d_exposure - u_load * u_exposure,
        }
    return metrics

def _subtree_stats(feeder: RootedFeeder, tree: _TreeView) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    subtree_nodes: dict[str, int] = {}
    subtree_load: dict[str, float] = {}
    subtree_exposure: dict[str, float] = {}
    for node in tree.post_order:
        node_count = 1
        load = float(feeder.loads_kw.get(node, 0.0))
        exposure = 0.0
        for edge in tree.children.get(node, ()):
            child = edge.downstream
            node_count += subtree_nodes[child]
            load += subtree_load[child]
            exposure += subtree_exposure[child] + edge.exposure
        subtree_nodes[node] = node_count
        subtree_load[node] = load
        subtree_exposure[node] = exposure
    return subtree_nodes, subtree_load, subtree_exposure

# ---------------------------------------------------------------------------
# Zones and base switch rows

def derive_switch_bounded_blocks(feeder: RootedFeeder, switch_edges: Iterable[SsapEdge]) -> tuple[SsapZone, ...]:
    return tuple(_compute_zones(feeder, switch_set=frozenset(switch_edges)))

def _compute_zones(feeder: RootedFeeder, *, switch_set: frozenset[SsapEdge]) -> list[SsapZone]:
    tree = _TreeView.from_feeder(feeder)
    zone_roots = [feeder.root] + [edge.downstream for edge in feeder.edges if edge in switch_set]
    zones: list[SsapZone] = []
    for root_bus in zone_roots:
        nodes: list[str] = []
        zone_edges: list[SsapEdge] = []
        stack = [root_bus]
        while stack:
            current = stack.pop()
            nodes.append(current)
            for edge in tree.children.get(current, ()):
                if edge in switch_set:
                    continue
                zone_edges.append(edge)
                stack.append(edge.downstream)
        if root_bus != feeder.root:
            zone_edges.append(next(edge for edge in switch_set if edge.downstream == root_bus))
        zones.append(SsapZone(tuple(nodes), tuple(zone_edges), sum(feeder.loads_kw.get(n, 0.0) for n in nodes), sum(e.exposure for e in zone_edges)))
    return zones

def emit_ssap_switch_rows(solution: SsapSolution, *, feeder_id: str, location_id: str) -> list[dict]:
    return [_base_switch_row(edge, solution, feeder_id=feeder_id, location_id=location_id) for edge in solution.switch_edges]

def _base_switch_row(edge: SsapEdge, solution: SsapSolution, *, feeder_id: str, location_id: str) -> dict[str, Any]:
    bus_slug = f"{_safe_token(edge.upstream)}__{_safe_token(edge.downstream)}"
    provenance = {
        "placement_rule": placement_rule_ssap, "sub_rule": ssap_subrule, "feeder_id": feeder_id,
        "k_switches": len(solution.switch_edges), "objective_value": solution.objective_value,
        "edge_exposure": edge.exposure,
    }
    return {
        "sandbox_id": location_id,
        "switch_id": f"{location_id}:asset:controllable_switches:{_safe_token(feeder_id)}:{bus_slug}",
        "from_bus": edge.upstream, "to_bus": edge.downstream, "switch_role": "sectionalizing",
        "normal_state": "closed", "initial_state": "closed", "dispatchable": True, "status": "enabled",
        "placement_rule": placement_rule_ssap, "source_provenance": json.dumps(provenance, sort_keys=True),
        "schema_version": controllable_switches_schema_version,
    }

# ---------------------------------------------------------------------------
# Table -> rooted feeder topology

def build_rooted_feeder_from_tables(
    buses: Iterable[Mapping[str, Any]], lines: Iterable[Mapping[str, Any]], *, source_bus: str,
    length_field: str = "length_m", default_exposure: float = 1.0,
) -> RootedFeeder:
    loads_kw = _loads_from_buses(buses)
    if source_bus not in loads_kw:
        raise ValueError(f"source_bus {source_bus!r} not found in buses table")
    neighbours = _neighbours_from_lines(lines, length_field=length_field, default_exposure=default_exposure)
    if source_bus not in neighbours:
        raise ValueError(f"source_bus {source_bus!r} is isolated -- no lines touch it")
    return RootedFeeder(source_bus, loads_kw, _orient_tree_edges(source_bus, neighbours))

def build_rooted_feeders_from_tables(
    buses: Iterable[Mapping[str, Any]], lines: Iterable[Mapping[str, Any]], *, source_buses: Iterable[str] = (),
    length_field: str = "length_m", default_exposure: float = 1.0,
) -> tuple[RootedFeeder, ...]:
    bus_rows = list(buses)
    line_rows = list(lines)
    loads_kw = _loads_from_buses(bus_rows)
    neighbours = _neighbours_from_lines(line_rows, length_field=length_field, default_exposure=default_exposure, nodes=loads_kw)
    for line in line_rows:
        loads_kw.setdefault(line["from_bus"], 0.0)
        loads_kw.setdefault(line["to_bus"], 0.0)
    source_set = set(source_buses)
    seen: set[str] = set()
    feeders: list[RootedFeeder] = []
    for start in sorted(neighbours):
        if start in seen or not neighbours[start]:
            continue
        component = _component_nodes(start, neighbours, seen)
        root = _choose_component_root(component, source_set, neighbours, loads_kw)
        loads = {node: loads_kw.get(node, 0.0) for node in sorted(component)}
        feeders.append(RootedFeeder(root, loads, _orient_tree_edges(root, neighbours, component=component)))
    return tuple(sorted(feeders, key=lambda feeder: feeder.root))

def _loads_from_buses(buses: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    loads: dict[str, float] = {}
    for bus in buses:
        loads[bus["name"]] = float(bus.get("load_kw", 0.0) or 0.0)
    return loads

def _neighbours_from_lines(
    lines: Iterable[Mapping[str, Any]], *, length_field: str, default_exposure: float, nodes: Iterable[str] = (),
) -> dict[str, list[tuple[str, float]]]:
    neighbours: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for line in lines:
        u, v = line["from_bus"], line["to_bus"]
        raw = line.get(length_field)
        exposure = float(raw) if raw not in (None, "") else default_exposure
        neighbours.setdefault(u, []).append((v, exposure))
        neighbours.setdefault(v, []).append((u, exposure))
    return neighbours

def _component_nodes(start: str, neighbours: Mapping[str, list[tuple[str, float]]], seen: set[str]) -> set[str]:
    component: set[str] = set()
    stack = [start]
    seen.add(start)
    while stack:
        node = stack.pop()
        component.add(node)
        for child, _ in neighbours.get(node, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return component

def _choose_component_root(
    component: set[str], source_set: set[str], neighbours: Mapping[str, list[tuple[str, float]]], loads_kw: Mapping[str, float],
) -> str:
    component_sources = sorted(component & source_set)
    if component_sources:
        return component_sources[0]
    return sorted(component, key=lambda node: (-len(neighbours.get(node, ())), -loads_kw.get(node, 0.0), node))[0]

def _orient_tree_edges(
    root: str, neighbours: Mapping[str, list[tuple[str, float]]], *, component: set[str] | None = None,
) -> tuple[SsapEdge, ...]:
    oriented: list[SsapEdge] = []
    visited: set[str] = {root}
    queue: deque[str] = deque([root])
    while queue:
        parent = queue.popleft()
        for child, exposure in neighbours.get(parent, ()):
            if child in visited or (component is not None and child not in component):
                continue
            visited.add(child)
            oriented.append(SsapEdge(parent, child, exposure))
            queue.append(child)
    return tuple(oriented)

# ---------------------------------------------------------------------------
# Switch synthesis workflow helpers

switch_artifact_columns: tuple[str, ...] = (
    "sandbox_id", "switch_id", "component_id", "feeder_id", "opendss_element", "from_bus", "to_bus", "phases", "lon", "lat",
    "switch_role", "normal_state", "initial_state", "dispatchable", "status", "opens_existing_line", "associated_line_name",
    "associated_linecode", "associated_units", "associated_length_m", "marginal_benefit", "placement_rule", "source_provenance", "schema_version",
)

def physical_lines_only(lines: pd.DataFrame) -> pd.DataFrame:
    return lines[lines["line_class"].fillna("line").eq("line")].copy()

def derive_fuses(lines: pd.DataFrame) -> pd.DataFrame:
    max_phases_at_bus: dict[str, int] = defaultdict(int)
    for row in lines.itertuples(index=False):
        phases = int(row.phases)
        max_phases_at_bus[row.from_bus] = max(max_phases_at_bus[row.from_bus], phases)
        max_phases_at_bus[row.to_bus] = max(max_phases_at_bus[row.to_bus], phases)
    rows: list[dict[str, object]] = []
    for line in lines.itertuples(index=False):
        phases = int(line.phases)
        for endpoint in (line.from_bus, line.to_bus):
            parent_phases = max_phases_at_bus[endpoint]
            if parent_phases > phases:
                rows.append({"fuse_id": f"fuse_{line.line_name}_{endpoint}", "feeder_id": line.feeder_id, "line_name": line.line_name, "head_bus": endpoint, "child_phases": phases, "parent_phases": parent_phases})
                break
    return pd.DataFrame(rows, columns=["fuse_id", "feeder_id", "line_name", "head_bus", "child_phases", "parent_phases"])

def physical_switch_candidate_edges(feeder: RootedFeeder, physical_lines: pd.DataFrame) -> tuple[SsapEdge, ...]:
    physical_keys = {frozenset((str(row.from_bus), str(row.to_bus))) for row in physical_lines.itertuples(index=False)}
    return tuple(edge for edge in feeder.edges if frozenset((edge.upstream, edge.downstream)) in physical_keys)

def switch_inputs(
    buses: pd.DataFrame, physical_lines: pd.DataFrame, sources: pd.DataFrame, *, exposure_mode: str,
    transformers: pd.DataFrame | None = None,
) -> tuple[dict[str, RootedFeeder], list[ComponentMeta]]:
    if exposure_mode not in {"homogeneous", "length_weighted"}:
        raise ValueError("exposure_mode must be 'homogeneous' or 'length_weighted'")
    component_feeders: dict[str, RootedFeeder] = {}
    component_meta: list[ComponentMeta] = []
    transformers = transformers if transformers is not None else pd.DataFrame()
    for feeder_id in sorted(buses["feeder_id"].dropna().unique()):
        feeder_buses = buses[buses["feeder_id"].eq(feeder_id)].copy()
        feeder_lines = physical_lines[physical_lines["feeder_id"].eq(feeder_id)].copy()
        bridge_rows = _transformer_bridge_rows(transformers, str(feeder_id))
        feeder_buses = _include_bridge_buses(buses, feeder_buses, bridge_rows)
        feeder_sources = _source_buses_for_feeder(sources, feeder_buses, feeder_id)
        if feeder_buses.empty or feeder_lines.empty:
            continue
        rooted_components = build_rooted_feeders_from_tables(
            feeder_buses.rename(columns={"bus": "name"}).to_dict("records"),
            feeder_lines.to_dict("records") + bridge_rows, source_buses=feeder_sources, length_field="length", default_exposure=1.0,
        )
        for component_index, feeder in enumerate(rooted_components, start=1):
            component_id = feeder_id if len(rooted_components) == 1 else f"{feeder_id}:component:{component_index:02d}"
            conditioned = homogenize_feeder_exposure(feeder, exposure=1.0) if exposure_mode == "homogeneous" else feeder
            component_feeders[component_id] = conditioned
            component_meta.append(ComponentMeta(component_id, feeder_id, feeder.root, len(feeder.loads_kw), len(feeder.edges)))
    if not component_feeders:
        raise RuntimeError("No rooted feeder components were available for SSAP.")
    return component_feeders, component_meta

def _include_bridge_buses(buses: pd.DataFrame, feeder_buses: pd.DataFrame, bridge_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not bridge_rows:
        return feeder_buses
    bridge_buses = {bus for row in bridge_rows for bus in (row["from_bus"], row["to_bus"])}
    return buses[buses["bus"].isin(set(feeder_buses["bus"].astype(str)) | bridge_buses)].copy()

def _source_buses_for_feeder(sources: pd.DataFrame, feeder_buses: pd.DataFrame, feeder_id: str) -> list[str]:
    feeder_sources = sources.loc[sources["feeder_id"].eq(feeder_id), "bus"].dropna().astype(str).tolist()
    if not feeder_sources and "source_count" in feeder_buses.columns:
        feeder_sources = feeder_buses.loc[feeder_buses["source_count"].fillna(0).gt(0), "bus"].astype(str).tolist()
    return feeder_sources

def _transformer_bridge_rows(transformers: pd.DataFrame, feeder_id: str) -> list[dict[str, Any]]:
    if transformers.empty or "winding_buses" not in transformers.columns:
        return []
    rows: list[dict[str, Any]] = []
    feeder_transformers = transformers[transformers["feeder_id"].astype(str).eq(feeder_id)]
    for tx in feeder_transformers.itertuples(index=False):
        windings: list[str] = []
        for bus in str(tx.winding_buses).split(","):
            bus = bus.strip()
            if bus and bus not in windings:
                windings.append(bus)
        for downstream in windings[1:]:
            rows.append({"from_bus": windings[0], "to_bus": downstream, "length": 0.0, "line_class": "transformer_bridge"})
    return rows

def write_switches(
    selection: SsapGlobalMarginalSelection, frontiers: dict[str, tuple[SsapFrontierPoint, ...]], component_meta: list[ComponentMeta],
    physical_lines: pd.DataFrame, *, location_id: str, exposure_mode: str, candidate_policy: SsapCandidatePolicy, ssap_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    line_by_edge = _line_lookup(physical_lines)
    meta_by_component = {meta.component_id: meta for meta in component_meta}
    switch_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for component_id, point in selection.selected_points.items():
        meta = meta_by_component[component_id]
        rows = emit_ssap_switch_rows(point.solution, feeder_id=component_id, location_id=location_id)
        for row_index, (row, edge) in enumerate(zip(rows, point.solution.switch_edges), start=1):
            switch_rows.append(_enrich_switch_row(
                row, _matching_line(line_by_edge, meta.feeder_id, edge), meta, point,
                row_index=row_index, exposure_mode=exposure_mode, candidate_policy=candidate_policy, ssap_budget=ssap_budget,
            ))
        diagnostic_rows.append(_diagnostic_row(meta, point, frontiers, exposure_mode, ssap_budget))
    switches = pd.DataFrame(switch_rows, columns=list(switch_artifact_columns))
    diagnostics = pd.DataFrame(diagnostic_rows)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(["selected_switches", "marginal_benefit", "component_id"], ascending=[False, False, True])
    return switches, diagnostics

def _line_lookup(physical_lines: pd.DataFrame) -> dict[tuple[str, tuple[str, str]], pd.Series]:
    return {(str(line.feeder_id), tuple(sorted((str(line.from_bus), str(line.to_bus))))): pd.Series(line._asdict()) for line in physical_lines.itertuples(index=False)}

def _matching_line(line_by_edge: Mapping[tuple[str, tuple[str, str]], pd.Series], feeder_id: str, edge: SsapEdge) -> pd.Series:
    key = (str(feeder_id), tuple(sorted((str(edge.upstream), str(edge.downstream)))))
    if key not in line_by_edge:
        raise KeyError(f"SSAP edge {edge.upstream} -> {edge.downstream} has no matching line row")
    return line_by_edge[key]

def _enrich_switch_row(
    row: dict[str, Any], line: pd.Series, meta: ComponentMeta, point: SsapFrontierPoint, *, row_index: int,
    exposure_mode: str, candidate_policy: SsapCandidatePolicy, ssap_budget: int,
) -> dict[str, Any]:
    line_name = str(line["line_name"])
    lon, lat = _line_midpoint(line)
    provenance = json.loads(row["source_provenance"])
    provenance.update({
        "component_id": meta.component_id, "feeder_id": meta.feeder_id, "exposure_mode": exposure_mode,
        "candidate_policy": {
            "min_block_bus_count": candidate_policy.min_block_bus_count,
            "min_block_load_kw": candidate_policy.min_block_load_kw,
            "max_candidate_edges_per_component": candidate_policy.max_candidate_edges,
        },
        "associated_line_name": line_name, "global_budget": ssap_budget,
        "component_selected_switches": point.k_switches, "component_marginal_benefit": point.marginal_benefit,
    })
    row.update({
        "component_id": meta.component_id, "feeder_id": meta.feeder_id,
        "opendss_element": _opendss_switch_name(meta.feeder_id, line_name, row_index),
        "phases": _phases(line), "lon": lon, "lat": lat, "opens_existing_line": True,
        "associated_line_name": line_name, "associated_linecode": line.get("linecode"),
        "associated_units": line.get("units", "m"), "associated_length_m": _line_length_m(line),
        "marginal_benefit": point.marginal_benefit / max(point.k_switches, 1),
        "source_provenance": json.dumps(provenance, sort_keys=True), "schema_version": controllable_switches_schema_version,
    })
    return row

def _opendss_switch_name(feeder_id: str, line_name: str, row_index: int) -> str:
    return f"Line.sw_sectionalizing_{_safe_token(feeder_id)}_{row_index:04d}_{_safe_token(line_name)}"

def _diagnostic_row(
    meta: ComponentMeta, point: SsapFrontierPoint, frontiers: Mapping[str, tuple[SsapFrontierPoint, ...]], exposure_mode: str, ssap_budget: int,
) -> dict[str, Any]:
    return {
        "component_id": meta.component_id, "feeder_id": meta.feeder_id, "root_bus": meta.root_bus,
        "bus_count": meta.bus_count, "edge_count": meta.edge_count, "budget": ssap_budget,
        "exposure_mode": exposure_mode, "frontier_points": len(frontiers.get(meta.component_id, ())),
        "selected_switches": point.k_switches, "objective_value": point.objective_value,
        "marginal_benefit": point.marginal_benefit, "placement_rule": placement_rule_ssap,
    }

# ---------------------------------------------------------------------------
# Small graph and row utilities

def _children_by_parent(edges: Iterable[SsapEdge]) -> dict[str, tuple[SsapEdge, ...]]:
    children: dict[str, list[SsapEdge]] = {}
    for edge in edges:
        children.setdefault(edge.upstream, []).append(edge)
    return {parent: tuple(edge_list) for parent, edge_list in children.items()}

def _post_order(feeder: RootedFeeder) -> list[str]:
    return list(_post_order_from_children(feeder.root, _children_by_parent(feeder.edges)))

def _post_order_from_children(root: str, children_by_parent: Mapping[str, tuple[SsapEdge, ...]]) -> list[str]:
    order: list[str] = []
    stack: list[tuple[str, Iterable[SsapEdge]]] = [(root, iter(children_by_parent.get(root, ())))]
    while stack:
        node, child_iter = stack[-1]
        try:
            child = next(child_iter).downstream
            stack.append((child, iter(children_by_parent.get(child, ()))))
        except StopIteration:
            stack.pop()
            order.append(node)
    return order

def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")

def _line_length_m(line: pd.Series) -> float:
    raw_length = line.get("length")
    if pd.isna(raw_length):
        return 1.0
    units = str(line.get("units", "m")).lower()
    return float(raw_length) * (1000.0 if units in {"km", "kilometer", "kilometers"} else 1.0)

def _line_midpoint(line: pd.Series) -> tuple[float | None, float | None]:
    coords = [line.get("from_lon"), line.get("from_lat"), line.get("to_lon"), line.get("to_lat")]
    if any(pd.isna(value) for value in coords):
        return None, None
    return ((float(line["from_lon"]) + float(line["to_lon"])) / 2.0, (float(line["from_lat"]) + float(line["to_lat"])) / 2.0)

def _phases(line: pd.Series) -> int:
    raw = line.get("phases", 3)
    return int(raw) if pd.notna(raw) else 3
