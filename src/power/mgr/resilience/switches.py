

# Resilience switch placement and block validation

"""Sectionalizing Switch Allocation Problem (SSAP), switches, and load blocks."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
import hashlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

import pandas as pd


placement_rule_ssap = "ssap_radial_sectionalizing_switch_allocation"
ssap_subrule = "exact_tree_dynamic_programming_ssap"
controllable_switches_schema_version = "stage_b_controllable_switches.v0.1"


@dataclass(frozen=True)
class SsapEdge:
    """A directed edge from the rooted feeder, oriented parent -> child."""

    upstream: str
    downstream: str
    exposure: float


@dataclass(frozen=True)
class SsapZone:
    """A connected component of the feeder after switch edges are removed.

    ``edges`` includes the upstream-switch edge bounding the zone from above
    (if any) because faulting on that edge isolates this zone.
    """

    nodes: tuple[str, ...]
    edges: tuple[SsapEdge, ...]
    load_kw: float
    exposure: float


@dataclass(frozen=True)
class RootedFeeder:
    """Rooted radial feeder consumed by SSAP.

    The caller is responsible for orienting edges parent -> child away from
    ``root`` and supplying loads for every node.
    """

    root: str
    loads_kw: Mapping[str, float]
    edges: tuple[SsapEdge, ...]

    @property
    def children(self) -> Mapping[str, tuple[SsapEdge, ...]]:
        adj: dict[str, list[SsapEdge]] = {}
        for edge in self.edges:
            adj.setdefault(edge.upstream, []).append(edge)
        return {parent: tuple(es) for parent, es in adj.items()}


@dataclass(frozen=True)
class SsapSolution:
    switch_edges: tuple[SsapEdge, ...]
    objective_value: float
    zones: tuple[SsapZone, ...]


@dataclass(frozen=True)
class SsapCandidatePolicy:
    """Plug-in eligibility gates for RPOP-ready sectionalizing candidates."""

    min_edge_exposure: float = 0.0
    min_block_bus_count: int = 0
    min_block_load_kw: float = 0.0
    # Optional tractability screen for large synthetic feeders. The exact SSAP
    # solve still runs on the resulting eligible set; provenance must record
    # this cap when a caller uses it.
    max_candidate_edges: int | None = None


@dataclass(frozen=True)
class SsapFrontierPoint:
    """One exact SSAP solution on a budget frontier."""

    k_switches: int
    solution: SsapSolution
    marginal_benefit: float

    @property
    def objective_value(self) -> float:
        return self.solution.objective_value


@dataclass(frozen=True)
class SsapGlobalMarginalSelection:
    """Selected component frontier points from a global marginal-benefit rule."""

    selected_points: Mapping[str, SsapFrontierPoint]
    selected_switch_count: int
    total_marginal_benefit: float
    total_objective_value: float
    marginal_benefit_floor: float


def solve_switches(
    feeder: RootedFeeder,
    *,
    k_switches: int,
    algorithm: str = "tree_dp",
    eligible_edges: Iterable[SsapEdge] | None = None,
) -> SsapSolution:
    """Return the optimal SSAP solution placing exactly ``k_switches`` switches.

    ``algorithm`` selects between the polynomial-time tree DP (default) and
    the brute-force enumerator (parity oracle for small instances).
    """

    switchable_edges = (
        frozenset(feeder.edges) if eligible_edges is None else frozenset(eligible_edges)
    )
    if k_switches < 0:
        raise ValueError("k_switches must be non-negative")
    if not switchable_edges.issubset(feeder.edges):
        raise ValueError("eligible_edges must be drawn from feeder.edges")
    if k_switches > len(switchable_edges):
        edge_label = "available" if eligible_edges is None else "eligible"
        raise ValueError(
            f"k_switches={k_switches} exceeds {edge_label} edges={len(switchable_edges)}"
        )
    if algorithm == "tree_dp":
        return _solve_ssap_tree_dp(
            feeder, k_switches=k_switches, switchable_edges=switchable_edges
        )
    if algorithm == "brute_force":
        return _solve_ssap_brute_force(
            feeder, k_switches=k_switches, switchable_edges=switchable_edges
        )
    raise ValueError(
        f"unknown algorithm {algorithm!r}; expected 'tree_dp' or 'brute_force'"
    )


def _solve_ssap_brute_force(
    feeder: RootedFeeder,
    *,
    k_switches: int,
    switchable_edges: frozenset[SsapEdge],
) -> SsapSolution:
    """Exhaustive enumeration over K-subsets of edges. Parity oracle only."""

    best: Optional[SsapSolution] = None
    for switch_combo in combinations(feeder.edges, k_switches):
        if not set(switch_combo).issubset(switchable_edges):
            continue
        zones = _compute_zones(feeder, switch_set=frozenset(switch_combo))
        objective = sum(zone.load_kw * zone.exposure for zone in zones)
        candidate = SsapSolution(
            switch_edges=tuple(switch_combo),
            objective_value=objective,
            zones=tuple(zones),
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    assert best is not None
    return best


@dataclass(frozen=True)
class _DPState:
    """Bottom-up DP state for the exact SSAP tree DP.

    ``committed_cost`` is the sum of ``L(z) * E(z)`` for zones fully sealed
    off inside the current subtree by switches placed below; ``residual_load``
    and ``residual_exposure`` track the part of the subtree still attached to
    the current node (which will merge into the parent's zone unless the
    parent-edge gets a switch).
    """

    committed_cost: float
    residual_load: float
    residual_exposure: float
    switch_set: frozenset


def _pareto_prune(states: list[_DPState]) -> list[_DPState]:
    """Drop states dominated by another along all three minimization axes.

    A state ``s`` is dominated when some ``s'`` has ``committed_cost'<=s.cost
    AND residual_load'<=s.rL AND residual_exposure'<=s.rE`` and at least one
    inequality is strict. Anywhere in any future merge the dominator yields a
    final root cost no worse than the dominated state, so dominated states
    can be discarded.
    """

    if not states:
        return []
    # Sort by cost ascending then load then exposure: when we see s, no later
    # state can have lower cost, so we only need to check against the result
    # set for domination of (rL, rE) at equal or lower cost.
    states_sorted = sorted(
        states,
        key=lambda s: (s.committed_cost, s.residual_load, s.residual_exposure),
    )
    load_values = sorted({s.residual_load for s in states_sorted})
    min_exposure = _PrefixMinTree(len(load_values))
    result: list[_DPState] = []
    seen_dimensions: set[tuple[float, float, float]] = set()
    for s in states_sorted:
        dimensions = (s.committed_cost, s.residual_load, s.residual_exposure)
        if dimensions in seen_dimensions:
            continue
        load_index = bisect_right(load_values, s.residual_load)
        if min_exposure.query(load_index) <= s.residual_exposure:
            continue
        result.append(s)
        seen_dimensions.add(dimensions)
        min_exposure.update(load_index, s.residual_exposure)
    return result


class _PrefixMinTree:
    """Fenwick tree for exact prefix-min dominance queries."""

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


def _post_order(feeder: RootedFeeder) -> list[str]:
    """Iterative post-order traversal of the rooted arborescence."""

    children_by_parent = feeder.children
    order: list[str] = []
    stack: list[tuple[str, Iterable[SsapEdge]]] = [
        (feeder.root, iter(children_by_parent.get(feeder.root, ())))
    ]
    while stack:
        node, child_iter = stack[-1]
        try:
            next_edge = next(child_iter)
            child = next_edge.downstream
            stack.append((child, iter(children_by_parent.get(child, ()))))
        except StopIteration:
            stack.pop()
            order.append(node)
    return order


def _solve_ssap_tree_dp(
    feeder: RootedFeeder,
    *,
    k_switches: int,
    switchable_edges: frozenset[SsapEdge],
) -> SsapSolution:
    """Polynomial-time exact SSAP via tree dynamic programming.

    Each subtree carries Pareto-pruned ``(committed_cost, residual_load,
    residual_exposure, switch_set)`` states keyed by switches-used. At every
    node we merge the accumulator with the next child's states, branching on
    whether the parent-edge gets a switch. The root's exact-K states are
    scored as ``committed_cost + residual_load * residual_exposure`` to pick
    up the root zone, and the minimum-cost state is returned.
    """

    children_by_parent = feeder.children
    state_by_node: dict[str, dict[int, list[_DPState]]] = {}

    for node in _post_order(feeder):
        # Initial accumulator: this node alone forms the residual zone.
        accumulator: dict[int, list[_DPState]] = {
            0: [
                _DPState(
                    committed_cost=0.0,
                    residual_load=float(feeder.loads_kw.get(node, 0.0)),
                    residual_exposure=0.0,
                    switch_set=frozenset(),
                )
            ]
        }

        for child_edge in children_by_parent.get(node, ()):
            child_states = state_by_node[child_edge.downstream]
            new_accumulator: dict[int, list[_DPState]] = {}
            for k_acc, acc_states in accumulator.items():
                for k_child, c_states in child_states.items():
                    for acc in acc_states:
                        for c in c_states:
                            # Branch 1: no switch on the parent edge -> child's
                            # residual merges with the accumulator's residual,
                            # and the parent edge's exposure joins the residual.
                            if k_acc + k_child <= k_switches:
                                new_accumulator.setdefault(
                                    k_acc + k_child, []
                                ).append(
                                    _DPState(
                                        committed_cost=acc.committed_cost
                                        + c.committed_cost,
                                        residual_load=acc.residual_load
                                        + c.residual_load,
                                        residual_exposure=acc.residual_exposure
                                        + c.residual_exposure
                                        + child_edge.exposure,
                                        switch_set=acc.switch_set
                                        | c.switch_set,
                                    )
                                )
                            # Branch 2: switch on the parent edge -> child's
                            # residual zone is sealed off, paying its load
                            # times its exposure (including the switch edge).
                            if k_acc + k_child + 1 <= k_switches:
                                if child_edge in switchable_edges:
                                    new_accumulator.setdefault(
                                        k_acc + k_child + 1, []
                                    ).append(
                                        _DPState(
                                            committed_cost=acc.committed_cost
                                            + c.committed_cost
                                            + c.residual_load
                                            * (
                                                c.residual_exposure
                                                + child_edge.exposure
                                            ),
                                            residual_load=acc.residual_load,
                                            residual_exposure=acc.residual_exposure,
                                            switch_set=acc.switch_set
                                            | c.switch_set
                                            | {child_edge},
                                        )
                                    )
            accumulator = {
                k: _pareto_prune(states) for k, states in new_accumulator.items()
            }

        state_by_node[node] = accumulator

    root_states = state_by_node[feeder.root]
    if k_switches not in root_states:
        raise ValueError(
            f"SSAP infeasible: no DP state at root with exactly {k_switches} switches"
        )

    best = min(
        root_states[k_switches],
        key=lambda s: s.committed_cost + s.residual_load * s.residual_exposure,
    )
    objective = best.committed_cost + best.residual_load * best.residual_exposure
    switch_edges = tuple(best.switch_set)
    zones = _compute_zones(feeder, switch_set=frozenset(switch_edges))
    return SsapSolution(
        switch_edges=switch_edges,
        objective_value=objective,
        zones=tuple(zones),
    )


def solve_rpop_ready_ssap(
    feeder: RootedFeeder,
    *,
    k_switches: int,
    policy: SsapCandidatePolicy | None = None,
    algorithm: str = "tree_dp",
    eligible_edges: Iterable[SsapEdge] | None = None,
) -> SsapSolution:
    """Solve SSAP after applying RPOP/DNMG candidate-admissibility gates."""

    policy = policy or SsapCandidatePolicy()
    candidate_edges = tuple(feeder.edges) if eligible_edges is None else tuple(eligible_edges)
    switchable_edges = select_policy_candidate_edges(feeder, candidate_edges, policy)
    return solve_switches(
        feeder,
        k_switches=k_switches,
        algorithm=algorithm,
        eligible_edges=switchable_edges,
    )


def solve_rpop_ready_ssap_frontier(
    feeder: RootedFeeder,
    *,
    max_switches: int,
    policy: SsapCandidatePolicy | None = None,
    algorithm: str = "tree_dp",
    eligible_edges: Iterable[SsapEdge] | None = None,
) -> tuple[SsapFrontierPoint, ...]:
    """Solve exact RPOP-ready SSAP for every budget from zero to ``max_switches``.

    The frontier exposes that input rather than hiding it behind a fixed
    density, so callers can choose a study budget from marginal ENS benefit.
    """

    policy = policy or SsapCandidatePolicy()
    candidate_edges = tuple(feeder.edges) if eligible_edges is None else tuple(eligible_edges)
    switchable_edges = select_policy_candidate_edges(feeder, candidate_edges, policy)
    feasible_max = min(max(0, max_switches), len(switchable_edges))
    points: list[SsapFrontierPoint] = []
    previous_objective: float | None = None
    for k_switches in range(feasible_max + 1):
        solution = solve_switches(
            feeder,
            k_switches=k_switches,
            algorithm=algorithm,
            eligible_edges=switchable_edges,
        )
        marginal_benefit = (
            0.0
            if previous_objective is None
            else max(0.0, previous_objective - solution.objective_value)
        )
        points.append(
            SsapFrontierPoint(
                k_switches=k_switches,
                solution=solution,
                marginal_benefit=marginal_benefit,
            )
        )
        previous_objective = solution.objective_value
    return tuple(points)


def select_global_marginal_benefit_budget(
    frontiers: Mapping[str, tuple[SsapFrontierPoint, ...]],
    *,
    min_marginal_benefit: float,
    max_total_switches: int | None = None,
    min_per_feeder: int = 0,
) -> SsapGlobalMarginalSelection:
    """Select component budgets by repeatedly taking the best next benefit.

    Each component point remains an exact SSAP solution for its own switch
    budget. The global rule only decides how many of those exact-budget
    switches to accept without adding utility cost, failure-rate, or flood data.

    ``min_per_feeder`` enforces an equity floor over the caller-supplied
    frontier components: each component should
    receive at least that many switches before any component can be advanced
    past the floor, provided the component's frontier has enough points to
    support the request. If the global budget cannot cover every floor
    increment, scarce floor switches are assigned by next marginal benefit
    rather than component-id order. The floor is consumed before the global
    marginal-benefit rule resumes.
    """

    if min_marginal_benefit < 0:
        raise ValueError("min_marginal_benefit must be non-negative")
    if min_per_feeder < 0:
        raise ValueError("min_per_feeder must be non-negative")

    selected_index: dict[str, int] = {}
    for component_id, frontier in frontiers.items():
        if not frontier or frontier[0].k_switches != 0:
            raise ValueError("each frontier must start with k_switches=0")
        selected_index[component_id] = 0

    total_switches = 0
    total_benefit = 0.0
    # Phase 1: per-component equity floor. When the global budget cannot cover
    # every floor increment, choose the largest next benefit rather than
    # exhausting the budget in lexicographic component-id order.
    if min_per_feeder > 0:
        while True:
            if max_total_switches is not None and total_switches >= max_total_switches:
                break
            best_component: str | None = None
            best_point: SsapFrontierPoint | None = None
            for component_id in sorted(frontiers):
                frontier = frontiers[component_id]
                next_index = selected_index[component_id] + 1
                target = min(min_per_feeder, len(frontier) - 1)
                if next_index > target:
                    continue
                candidate = frontier[next_index]
                if best_point is None or (
                    candidate.marginal_benefit,
                    component_id,
                ) > (
                    best_point.marginal_benefit,
                    best_component or "",
                ):
                    best_component = component_id
                    best_point = candidate
            if best_component is None or best_point is None:
                break
            selected_index[best_component] += 1
            total_benefit += best_point.marginal_benefit
            total_switches += 1
    while max_total_switches is None or total_switches < max_total_switches:
        best_component: str | None = None
        best_point: SsapFrontierPoint | None = None
        for component_id in sorted(frontiers):
            next_index = selected_index[component_id] + 1
            frontier = frontiers[component_id]
            if next_index >= len(frontier):
                continue
            candidate = frontier[next_index]
            if best_point is None or (
                candidate.marginal_benefit,
                component_id,
            ) > (
                best_point.marginal_benefit,
                best_component or "",
            ):
                best_component = component_id
                best_point = candidate
        if best_component is None or best_point is None:
            break
        if best_point.marginal_benefit < min_marginal_benefit:
            break
        selected_index[best_component] += 1
        total_switches += 1
        total_benefit += best_point.marginal_benefit

    selected_points = {
        component_id: frontiers[component_id][point_index]
        for component_id, point_index in selected_index.items()
    }
    total_objective = sum(
        point.objective_value for point in selected_points.values()
    )
    return SsapGlobalMarginalSelection(
        selected_points=selected_points,
        selected_switch_count=total_switches,
        total_marginal_benefit=total_benefit,
        total_objective_value=total_objective,
        marginal_benefit_floor=min_marginal_benefit,
    )


def classify_two_tier_switches(
    rows: list[dict],
    *,
    automated_count: int,
) -> list[dict]:
    """Promote the top ``automated_count`` switches by marginal benefit
    to ``automated_sectionalizing`` (``dispatchable=True``); the rest are
    ``manual_sectionalizing`` (``dispatchable=False``).

    The two-tier split keeps the highest-benefit switches dispatchable and
    leaves the rest as manually operable devices. Both tiers still
    export as OpenDSS ``Line ... Switch=Yes`` so PowerModelsDistribution
    recognises them as native switches; only PowerModelsONM's RPOP
    optimisation uses the ``dispatchable`` flag.
    """
    if automated_count < 0:
        raise ValueError("automated_count must be non-negative")
    ranked = sorted(
        rows, key=lambda r: float(r.get("marginal_benefit", 0.0)), reverse=True
    )
    automated_ids = {id(r) for r in ranked[:automated_count]}
    classified: list[dict] = []
    for row in rows:
        new_row = dict(row)
        is_automated = id(row) in automated_ids
        new_row["switch_role_class"] = (
            "automated_sectionalizing" if is_automated else "manual_sectionalizing"
        )
        new_row["dispatchable"] = is_automated
        classified.append(new_row)
    return classified


def build_customer_weighted_loads(
    *,
    loads_kw: Mapping[str, float],
    customers_by_bus: Mapping[str, float],
    kw_weight: float = 1.0,
    customer_weight: float = 0.0,
) -> dict[str, float]:
    """Blend static kW with a per-bus customer-count proxy for the SSAP load term.

    Supports customer-weighted outage cost so per-zone outage cost can be
    proportional to customer count, not just connected kW. Marshfield exposes
    the same blend via

        L(bus) = kw_weight * load_kw(bus) + customer_weight * n_customers(bus)

    Pure kW recovers when ``customer_weight=0``; pure SAIDI recovers when
    ``kw_weight=0``. The blend is a caller-supplied input to the existing
    SSAP solver, not a change to its objective.

    Coastal Marshfield circuits (Ocean Bluff, Brant Rock, Green Harbor,
    Rexhame) have dense small-lot service drops with low kW per customer;
    raising ``customer_weight`` increases their effective load and biases
    global marginal-benefit selection toward more switches in those
    feeders without introducing flood, FEMA, SFINCS, or ERAD signals.

    """
    if kw_weight < 0 or customer_weight < 0:
        raise ValueError("kw_weight and customer_weight must be non-negative")
    out: dict[str, float] = {}
    for bus, kw in loads_kw.items():
        customers = float(customers_by_bus.get(bus, 0.0))
        out[bus] = float(kw) * kw_weight + customers * customer_weight
    for bus, customers in customers_by_bus.items():
        if bus not in out:
            out[bus] = float(customers) * customer_weight
    return out


def homogenize_feeder_exposure(
    feeder: RootedFeeder, *, exposure: float = 1.0
) -> RootedFeeder:
    """Return the same rooted feeder with every edge assigned equal exposure."""

    if exposure <= 0:
        raise ValueError("exposure must be positive")
    return RootedFeeder(
        root=feeder.root,
        loads_kw=feeder.loads_kw,
        edges=tuple(
            SsapEdge(
                upstream=edge.upstream,
                downstream=edge.downstream,
                exposure=float(exposure),
            )
            for edge in feeder.edges
        ),
    )


def select_policy_candidate_edges(
    feeder: RootedFeeder,
    candidate_edges: Iterable[SsapEdge],
    policy: SsapCandidatePolicy,
) -> tuple[SsapEdge, ...]:
    """Apply RPOP candidate gates and optional exact-solve tractability cap."""

    if policy.max_candidate_edges is not None and policy.max_candidate_edges < 0:
        raise ValueError("max_candidate_edges must be non-negative or None")

    metrics = _single_switch_metrics(feeder)
    switchable_edges = tuple(
        edge
        for edge in candidate_edges
        if _edge_satisfies_candidate_metrics(edge, metrics[edge], policy)
    )
    if (
        policy.max_candidate_edges is None
        or len(switchable_edges) <= policy.max_candidate_edges
    ):
        return switchable_edges

    def one_switch_benefit(edge: SsapEdge) -> tuple[float, str, str]:
        return (metrics[edge]["benefit"], edge.upstream, edge.downstream)

    return tuple(
        sorted(switchable_edges, key=one_switch_benefit, reverse=True)[
            : policy.max_candidate_edges
        ]
    )


def _edge_satisfies_candidate_metrics(
    edge: SsapEdge,
    metrics: Mapping[str, float],
    policy: SsapCandidatePolicy,
) -> bool:
    if edge.exposure < policy.min_edge_exposure:
        return False
    if metrics["downstream_nodes"] < policy.min_block_bus_count:
        return False
    if metrics["upstream_nodes"] < policy.min_block_bus_count:
        return False
    if metrics["downstream_load"] < policy.min_block_load_kw:
        return False
    if metrics["upstream_load"] < policy.min_block_load_kw:
        return False
    return True


def _single_switch_metrics(feeder: RootedFeeder) -> dict[SsapEdge, dict[str, float]]:
    """Return subtree metrics for each possible single-switch boundary."""

    children_by_parent = feeder.children
    subtree_nodes: dict[str, int] = {}
    subtree_load: dict[str, float] = {}
    subtree_exposure: dict[str, float] = {}
    for node in _post_order(feeder):
        node_count = 1
        load = float(feeder.loads_kw.get(node, 0.0))
        exposure = 0.0
        for edge in children_by_parent.get(node, ()):
            child = edge.downstream
            node_count += subtree_nodes[child]
            load += subtree_load[child]
            exposure += subtree_exposure[child] + edge.exposure
        subtree_nodes[node] = node_count
        subtree_load[node] = load
        subtree_exposure[node] = exposure

    total_nodes = subtree_nodes.get(feeder.root, 0)
    total_load = subtree_load.get(feeder.root, 0.0)
    total_exposure = subtree_exposure.get(feeder.root, 0.0)
    base_objective = total_load * total_exposure
    metrics: dict[SsapEdge, dict[str, float]] = {}
    for edge in feeder.edges:
        child = edge.downstream
        downstream_nodes = subtree_nodes[child]
        downstream_load = subtree_load[child]
        downstream_exposure = subtree_exposure[child] + edge.exposure
        upstream_nodes = total_nodes - downstream_nodes
        upstream_load = total_load - downstream_load
        upstream_exposure = total_exposure - downstream_exposure
        objective = (
            downstream_load * downstream_exposure
            + upstream_load * upstream_exposure
        )
        metrics[edge] = {
            "downstream_nodes": float(downstream_nodes),
            "upstream_nodes": float(upstream_nodes),
            "downstream_load": downstream_load,
            "upstream_load": upstream_load,
            "benefit": base_objective - objective,
        }
    return metrics


def derive_switch_bounded_blocks(
    feeder: RootedFeeder,
    switch_edges: Iterable[SsapEdge],
) -> tuple[SsapZone, ...]:
    """Compute static Switch-Bounded Load Blocks.

    Equivalent to opening every switch in ``switch_edges`` and computing the
    connected components of the remaining graph; each switch edge is
    attributed to the exposure of its downstream block.
    """

    return tuple(_compute_zones(feeder, switch_set=frozenset(switch_edges)))


def _compute_zones(
    feeder: RootedFeeder, *, switch_set: frozenset[SsapEdge]
) -> list[SsapZone]:
    """Partition the rooted feeder into zones by removing ``switch_set`` edges.

    Each switch edge is attributed to its downstream zone for exposure
    accounting.
    """

    children_by_parent = feeder.children
    zone_roots = [feeder.root]
    for edge in feeder.edges:  # preserve deterministic edge order
        if edge in switch_set:
            zone_roots.append(edge.downstream)

    zones: list[SsapZone] = []
    for root_bus in zone_roots:
        nodes: list[str] = []
        non_switch_edges: list[SsapEdge] = []
        stack: list[str] = [root_bus]
        while stack:
            current = stack.pop()
            nodes.append(current)
            for child_edge in children_by_parent.get(current, ()):
                if child_edge in switch_set:
                    continue  # downstream subtree is its own zone
                non_switch_edges.append(child_edge)
                stack.append(child_edge.downstream)

        zone_edges: tuple[SsapEdge, ...] = tuple(non_switch_edges)
        if root_bus != feeder.root:
            entering = next(
                edge for edge in switch_set if edge.downstream == root_bus
            )
            zone_edges = zone_edges + (entering,)

        load_kw = sum(feeder.loads_kw.get(node, 0.0) for node in nodes)
        exposure = sum(edge.exposure for edge in zone_edges)
        zones.append(
            SsapZone(
                nodes=tuple(nodes),
                edges=zone_edges,
                load_kw=load_kw,
                exposure=exposure,
            )
        )
    return zones


def emit_ssap_switch_rows(
    solution: SsapSolution,
    *,
    feeder_id: str,
    location_id: str,
) -> list[dict]:
    """Convert an SSAP solution into ``controllable_switches.parquet`` rows.

    SSAP-specific fields are populated here: ``switch_id``, ``switch_role``,
    ``placement_rule``, ``source_provenance``, ``schema_version``, and the
    bus endpoints. Electrical metadata (phases, voltage, ratings, lon/lat)
    is left to the caller to merge from the SHIFT bus/line tables so this
    module stays decoupled from the asset registry layout.
    """

    feeder_token = _safe_token(feeder_id)
    rows: list[dict] = []
    for switch_edge in solution.switch_edges:
        bus_slug = f"{_safe_token(switch_edge.upstream)}__{_safe_token(switch_edge.downstream)}"
        switch_id = f"{location_id}:asset:controllable_switches:{feeder_token}:{bus_slug}"
        provenance = {
            "placement_rule": placement_rule_ssap,
            "sub_rule": ssap_subrule,
            "feeder_id": feeder_id,
            "k_switches": len(solution.switch_edges),
            "objective_value": solution.objective_value,
            "edge_exposure": switch_edge.exposure,
        }
        rows.append(
            {
                "sandbox_id": location_id,
                "switch_id": switch_id,
                "from_bus": switch_edge.upstream,
                "to_bus": switch_edge.downstream,
                "switch_role": "sectionalizing",
                "normal_state": "closed",
                "initial_state": "closed",
                "dispatchable": True,
                "status": "enabled",
                "placement_rule": placement_rule_ssap,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": controllable_switches_schema_version,
            }
        )
    return rows


def _safe_token(value: str) -> str:
    """Slugify a value for use inside a stable artifact ID."""

    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def build_rooted_feeder_from_tables(
    buses: Iterable[Mapping[str, Any]],
    lines: Iterable[Mapping[str, Any]],
    *,
    source_bus: str,
    length_field: str = "length_m",
    default_exposure: float = 1.0,
) -> RootedFeeder:
    """Convert SHIFT-style bus/line row dicts into a ``RootedFeeder``.

    Edges are oriented parent -> child away from ``source_bus`` via BFS; the
    raw ``from_bus`` / ``to_bus`` column ordering is ignored. Missing or
    null line lengths fall back to ``default_exposure`` (1.0).

    Raises ``ValueError`` if ``source_bus`` is not present in ``buses`` or is
    isolated from the line graph.
    """

    loads_kw: dict[str, float] = {}
    for bus in buses:
        name = bus["name"]
        loads_kw[name] = float(bus.get("load_kw", 0.0) or 0.0)

    if source_bus not in loads_kw:
        raise ValueError(
            f"source_bus {source_bus!r} not found in buses table"
        )

    # Undirected adjacency built from the line table.
    neighbours: dict[str, list[tuple[str, float]]] = {}
    for line in lines:
        u = line["from_bus"]
        v = line["to_bus"]
        raw_len = line.get(length_field)
        exposure = float(raw_len) if raw_len not in (None, "") else default_exposure
        neighbours.setdefault(u, []).append((v, exposure))
        neighbours.setdefault(v, []).append((u, exposure))

    if source_bus not in neighbours:
        raise ValueError(
            f"source_bus {source_bus!r} is isolated -- no lines touch it"
        )

    # BFS from source_bus, orienting each edge parent -> child on first visit.
    oriented: list[SsapEdge] = []
    visited: set[str] = {source_bus}
    queue: deque[str] = deque([source_bus])
    while queue:
        parent = queue.popleft()
        for child, exposure in neighbours.get(parent, ()):
            if child in visited:
                continue
            visited.add(child)
            oriented.append(
                SsapEdge(upstream=parent, downstream=child, exposure=exposure)
            )
            queue.append(child)

    return RootedFeeder(
        root=source_bus,
        loads_kw=loads_kw,
        edges=tuple(oriented),
    )


def build_rooted_feeders_from_tables(
    buses: Iterable[Mapping[str, Any]],
    lines: Iterable[Mapping[str, Any]],
    *,
    source_buses: Iterable[str] = (),
    length_field: str = "length_m",
    default_exposure: float = 1.0,
) -> tuple[RootedFeeder, ...]:
    """Convert possibly disconnected bus/line tables into rooted components."""

    bus_rows = list(buses)
    line_rows = list(lines)
    loads_kw: dict[str, float] = {}
    for bus in bus_rows:
        name = bus["name"]
        loads_kw[name] = float(bus.get("load_kw", 0.0) or 0.0)

    source_set = set(source_buses)
    neighbours: dict[str, list[tuple[str, float]]] = {name: [] for name in loads_kw}
    for line in line_rows:
        u = line["from_bus"]
        v = line["to_bus"]
        raw_len = line.get(length_field)
        exposure = float(raw_len) if raw_len not in (None, "") else default_exposure
        neighbours.setdefault(u, []).append((v, exposure))
        neighbours.setdefault(v, []).append((u, exposure))
        loads_kw.setdefault(u, 0.0)
        loads_kw.setdefault(v, 0.0)

    seen: set[str] = set()
    feeders: list[RootedFeeder] = []
    for start in sorted(neighbours):
        if start in seen or not neighbours[start]:
            continue
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

        component_sources = sorted(component & source_set)
        if component_sources:
            root = component_sources[0]
        else:
            root = sorted(
                component,
                key=lambda node: (-len(neighbours.get(node, ())), -loads_kw.get(node, 0.0), node),
            )[0]

        oriented: list[SsapEdge] = []
        visited = {root}
        queue: deque[str] = deque([root])
        while queue:
            parent = queue.popleft()
            for child, exposure in neighbours.get(parent, ()):
                if child not in component or child in visited:
                    continue
                visited.add(child)
                oriented.append(
                    SsapEdge(upstream=parent, downstream=child, exposure=exposure)
                )
                queue.append(child)

        feeders.append(
            RootedFeeder(
                root=root,
                loads_kw={node: loads_kw.get(node, 0.0) for node in sorted(component)},
                edges=tuple(oriented),
            )
        )

    return tuple(sorted(feeders, key=lambda feeder: feeder.root))


# Switch synthesis workflow helpers

switch_artifact_columns: tuple[str, ...] = (
    "sandbox_id",
    "switch_id",
    "component_id",
    "feeder_id",
    "opendss_element",
    "from_bus",
    "to_bus",
    "phases",
    "lon",
    "lat",
    "switch_role",
    "normal_state",
    "initial_state",
    "dispatchable",
    "status",
    "opens_existing_line",
    "associated_line_name",
    "associated_linecode",
    "associated_units",
    "associated_length_m",
    "marginal_benefit",
    "placement_rule",
    "source_provenance",
    "schema_version",
)


@dataclass(frozen=True)
class ComponentMeta:
    """Bookkeeping for one rooted feeder component."""

    component_id: str
    feeder_id: str
    root_bus: str
    bus_count: int
    edge_count: int


def physical_lines_only(lines: pd.DataFrame) -> pd.DataFrame:
    """Return ordinary physical line rows used for switch candidacy."""

    return lines[lines["line_class"].fillna("line").eq("line")].copy()


def derive_fuses(lines: pd.DataFrame) -> pd.DataFrame:
    """Derive Fuse Proxy rows where lower-phase laterals leave a higher-phase bus."""
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
                rows.append(
                    {
                        "fuse_id": f"fuse_{line.line_name}_{endpoint}",
                        "feeder_id": line.feeder_id,
                        "line_name": line.line_name,
                        "head_bus": endpoint,
                        "child_phases": phases,
                        "parent_phases": parent_phases,
                    }
                )
                break

    return pd.DataFrame(
        rows,
        columns=["fuse_id", "feeder_id", "line_name", "head_bus", "child_phases", "parent_phases"],
    )


def physical_switch_candidate_edges(
    feeder: RootedFeeder,
    physical_lines: pd.DataFrame,
) -> tuple[SsapEdge, ...]:
    """Return feeder edges that correspond to ordinary physical line rows.

    Transformer bridge edges may be present in ``feeder`` for topology/load
    accounting, but they are not switch candidates.
    """

    physical_keys = {
        frozenset((str(row.from_bus), str(row.to_bus)))
        for row in physical_lines.itertuples(index=False)
    }
    return tuple(
        edge
        for edge in feeder.edges
        if frozenset((edge.upstream, edge.downstream)) in physical_keys
    )


def switch_inputs(
    buses: pd.DataFrame,
    physical_lines: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    exposure_mode: str,
    transformers: pd.DataFrame | None = None,
) -> tuple[dict[str, RootedFeeder], list[ComponentMeta]]:
    """Build one rooted feeder per electrical island, applying exposure mode.

    ``exposure_mode='homogeneous'`` collapses all edge exposures to 1.0 so the
    SSAP risk proxy becomes a customer-count proxy; ``'length_weighted'``
    keeps raw line length as the per-edge exposure. Transformer winding bridges
    keep service components connected for block accounting, but only ordinary
    physical lines should be passed to SSAP as switch candidates.
    """

    if exposure_mode not in {"homogeneous", "length_weighted"}:
        raise ValueError("exposure_mode must be 'homogeneous' or 'length_weighted'")

    component_feeders: dict[str, RootedFeeder] = {}
    component_meta: list[ComponentMeta] = []
    transformers = transformers if transformers is not None else pd.DataFrame()

    for feeder_id in sorted(buses["feeder_id"].dropna().unique()):
        feeder_buses = buses[buses["feeder_id"].eq(feeder_id)].copy()
        feeder_lines = physical_lines[physical_lines["feeder_id"].eq(feeder_id)].copy()
        bridge_rows = _transformer_bridge_rows(transformers, str(feeder_id))
        if bridge_rows:
            bridge_buses = {
                bus
                for row in bridge_rows
                for bus in (row["from_bus"], row["to_bus"])
            }
            feeder_buses = buses[
                buses["bus"].isin(set(feeder_buses["bus"].astype(str)) | bridge_buses)
            ].copy()
        feeder_sources = (
            sources.loc[sources["feeder_id"].eq(feeder_id), "bus"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not feeder_sources and "source_count" in feeder_buses.columns:
            feeder_sources = (
                feeder_buses.loc[feeder_buses["source_count"].fillna(0).gt(0), "bus"]
                .astype(str)
                .tolist()
            )
        if feeder_buses.empty or feeder_lines.empty:
            continue

        rooted_components = build_rooted_feeders_from_tables(
            feeder_buses.rename(columns={"bus": "name"}).to_dict("records"),
            feeder_lines.to_dict("records") + bridge_rows,
            source_buses=feeder_sources,
            length_field="length",
            default_exposure=1.0,
        )
        for component_index, feeder in enumerate(rooted_components, start=1):
            component_id = (
                feeder_id
                if len(rooted_components) == 1
                else f"{feeder_id}:component:{component_index:02d}"
            )
            conditioned = (
                homogenize_feeder_exposure(feeder, exposure=1.0)
                if exposure_mode == "homogeneous"
                else feeder
            )
            component_feeders[component_id] = conditioned
            component_meta.append(
                ComponentMeta(
                    component_id=component_id,
                    feeder_id=feeder_id,
                    root_bus=feeder.root,
                    bus_count=len(feeder.loads_kw),
                    edge_count=len(feeder.edges),
                )
            )

    if not component_feeders:
        raise RuntimeError("No rooted feeder components were available for SSAP.")
    return component_feeders, component_meta


def _transformer_bridge_rows(
    transformers: pd.DataFrame,
    feeder_id: str,
) -> list[dict[str, Any]]:
    """Represent transformer windings as non-switchable topology bridges."""

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
        if len(windings) < 2:
            continue
        for downstream in windings[1:]:
            rows.append(
                {
                    "from_bus": windings[0],
                    "to_bus": downstream,
                    "length": 0.0,
                    "line_class": "transformer_bridge",
                }
            )
    return rows


def write_switches(
    selection: SsapGlobalMarginalSelection,
    frontiers: dict[str, tuple[SsapFrontierPoint, ...]],
    component_meta: list[ComponentMeta],
    physical_lines: pd.DataFrame,
    *,
    location_id: str,
    exposure_mode: str,
    candidate_policy: SsapCandidatePolicy,
    ssap_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize an SSAP selection into OpenDSS-ready switch + diagnostic frames.

    Maps each selected SSAP edge back to its physical line, mints a stable
    OpenDSS element name, and folds the per-component selection context into
    each row's ``source_provenance`` JSON.
    """

    line_by_edge: dict[tuple[str, tuple[str, str]], pd.Series] = {}
    for line in physical_lines.itertuples(index=False):
        key = (
            str(line.feeder_id),
            tuple(sorted((str(line.from_bus), str(line.to_bus)))),
        )
        line_by_edge[key] = pd.Series(line._asdict())

    meta_by_component = {meta.component_id: meta for meta in component_meta}

    switch_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for component_id, point in selection.selected_points.items():
        meta = meta_by_component[component_id]
        feeder_id = meta.feeder_id
        solution = point.solution
        rows = emit_ssap_switch_rows(
            solution, feeder_id=component_id, location_id=location_id
        )

        for row_index, (row, edge) in enumerate(
            zip(rows, solution.switch_edges), start=1
        ):
            lookup_key = (
                feeder_id,
                tuple(sorted((edge.upstream, edge.downstream))),
            )
            if lookup_key not in line_by_edge:
                raise KeyError(
                    f"SSAP edge {edge.upstream} -> {edge.downstream} has no matching line row"
                )
            line = line_by_edge[lookup_key]
            line_name = str(line["line_name"])
            opendss_name = (
                f"sw_sectionalizing_{_safe_token(feeder_id)}_{row_index:04d}"
                f"_{_safe_token(line_name)}"
            )
            lon, lat = _line_midpoint(line)

            provenance = json.loads(row["source_provenance"])
            provenance.update(
                {
                    "component_id": component_id,
                    "feeder_id": feeder_id,
                    "exposure_mode": exposure_mode,
                    "candidate_policy": {
                        "min_block_bus_count": candidate_policy.min_block_bus_count,
                        "min_block_load_kw": candidate_policy.min_block_load_kw,
                        "max_candidate_edges_per_component": candidate_policy.max_candidate_edges,
                    },
                    "associated_line_name": line_name,
                    "global_budget": ssap_budget,
                    "component_selected_switches": point.k_switches,
                    "component_marginal_benefit": point.marginal_benefit,
                }
            )

            row.update(
                {
                    "component_id": component_id,
                    "feeder_id": feeder_id,
                    "opendss_element": f"Line.{opendss_name}",
                    "phases": _phases(line),
                    "lon": lon,
                    "lat": lat,
                    "opens_existing_line": True,
                    "associated_line_name": line_name,
                    "associated_linecode": line.get("linecode"),
                    "associated_units": line.get("units", "m"),
                    "associated_length_m": _line_length_m(line),
                    "marginal_benefit": point.marginal_benefit / max(point.k_switches, 1),
                    "source_provenance": json.dumps(provenance, sort_keys=True),
                    "schema_version": controllable_switches_schema_version,
                }
            )
            switch_rows.append(row)

        diagnostic_rows.append(
            {
                "component_id": meta.component_id,
                "feeder_id": meta.feeder_id,
                "root_bus": meta.root_bus,
                "bus_count": meta.bus_count,
                "edge_count": meta.edge_count,
                "budget": ssap_budget,
                "exposure_mode": exposure_mode,
                "frontier_points": len(frontiers.get(component_id, ())),
                "selected_switches": point.k_switches,
                "objective_value": point.objective_value,
                "marginal_benefit": point.marginal_benefit,
                "placement_rule": placement_rule_ssap,
            }
        )

    switches = pd.DataFrame(switch_rows, columns=list(switch_artifact_columns))
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["selected_switches", "marginal_benefit", "component_id"],
        ascending=[False, False, True],
    )
    return switches, diagnostics


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def _line_length_m(line: pd.Series) -> float:
    raw_length = line.get("length")
    if pd.isna(raw_length):
        return 1.0
    units = str(line.get("units", "m")).lower()
    multiplier = 1000.0 if units in {"km", "kilometer", "kilometers"} else 1.0
    return float(raw_length) * multiplier


def _line_midpoint(line: pd.Series) -> tuple[float | None, float | None]:
    coords = [line.get("from_lon"), line.get("from_lat"), line.get("to_lon"), line.get("to_lat")]
    if any(pd.isna(value) for value in coords):
        return None, None
    lon = (float(line["from_lon"]) + float(line["to_lon"])) / 2.0
    lat = (float(line["from_lat"]) + float(line["to_lat"])) / 2.0
    return lon, lat


def _phases(line: pd.Series) -> int:
    raw = line.get("phases", 3)
    return int(raw) if pd.notna(raw) else 3


# Switch-bounded load blocks
switch_bounded_load_blocks_schema_version = "stage_b_switch_bounded_load_blocks.v0.1"


class BlockInvariantViolation(ValueError):
    """Raised when a derived Switch-Bounded Load Block fails B/C/D/E/F."""


@dataclass(frozen=True)
class Block:
    """Switch-Bounded Load Block record passing the Block Invariant Contract."""

    buses: frozenset
    load_kw: float
    voltage_class_kv: float = 0.0
    phi_max: int = 0
    phase_set: frozenset = frozenset()
    has_substation_pcc: bool = False
    gfm_eligible_der_ids: tuple[str, ...] = ()
    voltage_source_reachability: str = "none"


def derive_validated_blocks(
    *,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    transformers: pd.DataFrame,
    sources: pd.DataFrame,
    loads: pd.DataFrame,
    controllable_switches: pd.DataFrame,
    der_inventory: pd.DataFrame | None = None,
) -> tuple[Block, ...]:
    """Derive blocks from the static line graph minus opened switch lines.

    Returns a tuple of ``Block`` objects, each having passed invariant B
    (and later C/D/E/F). G is recorded as a string field once added.
    """

    opened_lines = _opened_line_names(controllable_switches)
    components = _connected_components(buses, lines, opened_lines, transformers)
    loads_by_bus = _loads_by_bus(buses)
    voltage_class = label_bus_voltage_classes(
        sources=sources, lines=lines, transformers=transformers, loads=loads
    )
    bus_phase_count = _bus_phase_counts(lines, opened_lines)
    edges_per_component = _block_edge_counts(lines, opened_lines, components, transformers)
    line_voltage_violations = _line_voltage_transitions(lines, opened_lines, voltage_class)
    pcc_buses = set(sources["bus"].astype(str)) if not sources.empty else set()
    gfm_by_bus = _gfm_eligible_by_bus(der_inventory)
    # DER bus assignment pending until at least one GFM-capable DER row has a
    # non-null bus (DER inventory ships before the critical-load assignment
    # layer in Marshfield Stage B).
    der_assignment_pending = der_inventory is None or not gfm_by_bus

    blocks: list[Block] = []
    for component_idx, bus_set in enumerate(components):
        block_load_kw = sum(loads_by_bus.get(bus, 0.0) for bus in bus_set)
        has_pcc = bool(bus_set & pcc_buses)
        # Invariant B: a block must have load or a voltage source. A load-less
        # block carrying a substation PCC is a valid trunk; a block with neither
        # load nor PCC is a dead spur.
        if block_load_kw <= 0.0 and not has_pcc:
            raise BlockInvariantViolation(
                f"invariant_b: block {component_idx} has no load and no substation PCC "
                f"(buses={sorted(bus_set)})"
            )
        # Invariant C: multi-voltage blocks are allowed when bridged by
        # transformer windings, but a voltage transition across a LINE edge
        # between bands (LV/MV/HV) is a data error.
        block_line_violations = [
            (u, v) for (u, v) in line_voltage_violations if u in bus_set and v in bus_set
        ]
        if block_line_violations:
            u, v = block_line_violations[0]
            raise BlockInvariantViolation(
                f"invariant_c: block {component_idx} has line edge crossing voltage "
                f"bands without a transformer: ({u} {voltage_band(voltage_class.get(u))}) -> "
                f"({v} {voltage_band(voltage_class.get(v))})"
            )
        labeled = {voltage_class[bus] for bus in bus_set if bus in voltage_class}
        # Block voltage class = MV class when present (max), else LV.
        block_voltage = max(labeled) if labeled else 0.0
        phase_counts = {bus_phase_count.get(bus, 0) for bus in bus_set if bus_phase_count.get(bus, 0) > 0}
        phi_max = max(phase_counts) if phase_counts else 0
        edge_count = edges_per_component[component_idx]
        if edge_count > len(bus_set) - 1:
            raise BlockInvariantViolation(
                f"invariant_f: block {component_idx} is cyclic "
                f"(buses={len(bus_set)}, edges={edge_count}, spanning_tree_requires={len(bus_set) - 1})"
            )
        gfm_ids = tuple(sorted({did for bus in bus_set for did in gfm_by_bus.get(bus, ())}))
        if has_pcc and gfm_ids:
            reachability = "both"
        elif has_pcc:
            reachability = "substation_pcc"
        elif gfm_ids:
            reachability = "gfm_eligible_der"
        elif der_assignment_pending:
            reachability = "pending_der_inventory"
        else:
            reachability = "none"
        blocks.append(
            Block(
                buses=frozenset(bus_set),
                load_kw=block_load_kw,
                voltage_class_kv=block_voltage,
                phi_max=phi_max,
                phase_set=frozenset(phase_counts),
                has_substation_pcc=has_pcc,
                gfm_eligible_der_ids=gfm_ids,
                voltage_source_reachability=reachability,
            )
        )
    _raise_if_source_less_blocks_lack_switch_reachable_source_path(
        blocks, controllable_switches
    )
    return tuple(blocks)


def label_bus_voltage_classes(
    *,
    sources: pd.DataFrame,
    lines: pd.DataFrame,
    transformers: pd.DataFrame,
    loads: pd.DataFrame,
) -> dict[str, float]:
    """Label each bus with a kV voltage class via graph-walk propagation.

    Propagation rules:
      - source buses seeded from ``sources.basekv``
      - propagation continues across ``lines`` (no kv crossing on a line)
      - transformer windings terminate propagation; the opposite winding
        seeds a new voltage class from ``transformers.max_kv``/``min_kv``
      - loads supply a corroborating seed for leaf buses
    """
    adj: dict[str, list[str]] = {}
    for row in lines.itertuples(index=False):
        u, v = str(row.from_bus), str(row.to_bus)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    transformer_buses: set[str] = set()
    seeds: dict[str, float] = {}
    for row in sources.itertuples(index=False):
        seeds[str(row.bus)] = float(row.basekv)

    for row in transformers.itertuples(index=False):
        windings = [b.strip() for b in str(row.winding_buses).split(",") if b.strip()]
        max_kv = float(row.max_kv) if pd.notna(row.max_kv) else None
        min_kv = float(row.min_kv) if pd.notna(row.min_kv) else None
        for bus in windings:
            transformer_buses.add(bus)
        if len(windings) >= 2 and max_kv is not None and min_kv is not None:
            seeds.setdefault(windings[0], max_kv)
            for bus in windings[1:]:
                seeds.setdefault(bus, min_kv)

    for row in loads.itertuples(index=False):
        bus = str(row.bus)
        if bus not in seeds and pd.notna(row.kv):
            seeds[bus] = float(row.kv)

    voltage_class: dict[str, float] = dict(seeds)
    queue: deque[str] = deque(seeds.keys())
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, ()):
            if nbr in voltage_class:
                continue
            # Transformer winding buses retain their own seeded class
            # and do not propagate the upstream class outward.
            if node in transformer_buses and nbr in transformer_buses:
                continue
            voltage_class[nbr] = voltage_class[node]
            queue.append(nbr)
    return voltage_class


# --- helpers --------------------------------------------------------------


def _opened_line_names(controllable_switches: pd.DataFrame) -> set[str]:
    if controllable_switches.empty:
        return set()
    opens = controllable_switches.get("opens_existing_line")
    if opens is None:
        return set()
    opened = controllable_switches[opens.fillna(False).astype(bool)]
    return set(opened["associated_line_name"].dropna().astype(str))


def _connected_components(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    opened_lines: set[str],
    transformers: pd.DataFrame | None = None,
) -> list[set[str]]:
    """Connected components of (lines minus opened) plus transformer-winding
    bridges.

    Transformer edges connect the network across voltage classes but do
    NOT partition blocks; only Controllable Switches partition blocks.
    """
    adj: dict[str, list[str]] = {str(bus): [] for bus in buses["bus"]}
    for row in lines.itertuples(index=False):
        if str(row.line_name) in opened_lines:
            continue
        u, v = str(row.from_bus), str(row.to_bus)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    if transformers is not None and not transformers.empty and "winding_buses" in transformers.columns:
        for row in transformers.itertuples(index=False):
            windings = [b.strip() for b in str(row.winding_buses).split(",") if b.strip()]
            # Deduplicate (SHIFT split-phase emits 3 windings but only 2
            # distinct buses; un-deduped star edges would create spurious
            # parallel edges). Preserve first occurrence as the hub.
            unique: list[str] = []
            seen_w: set[str] = set()
            for bus in windings:
                if bus not in seen_w:
                    seen_w.add(bus)
                    unique.append(bus)
            # Star pattern (hub = first unique winding) avoids spurious
            # cycles from 3+ winding transformers.
            if len(unique) >= 2:
                hub = unique[0]
                for other in unique[1:]:
                    adj.setdefault(hub, []).append(other)
                    adj.setdefault(other, []).append(hub)

    seen: set[str] = set()
    components: list[set[str]] = []
    for start in adj:
        if start in seen:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            for nbr in adj.get(node, ()):
                if nbr in seen:
                    continue
                seen.add(nbr)
                queue.append(nbr)
        components.append(component)
    return components


def build_block_artifact_rows(
    blocks: tuple[Block, ...],
    *,
    location_id: str,
    buses: pd.DataFrame | None = None,
    controllable_switches: pd.DataFrame | None = None,
) -> list[dict]:
    """Convert validated blocks to ``switch_bounded_load_blocks.parquet`` rows.

    Each row carries a stable ``block_id`` derived from the sorted bus set,
    plus the per-block invariant fields and the bounding controllable
    switches.
    """
    feeder_by_bus = _feeder_by_bus(buses) if buses is not None else {}
    bounding_switches = _bounding_switches_by_block_buses(blocks, controllable_switches)

    rows: list[dict] = []
    for block in blocks:
        feeder_id = _modal_feeder(block.buses, feeder_by_bus)
        block_id = _stable_block_id(location_id=location_id, feeder_id=feeder_id, buses=block.buses)
        rows.append(
            {
                "sandbox_id": location_id,
                "block_id": block_id,
                "feeder_id": feeder_id,
                "bus_count": len(block.buses),
                "buses_json": json.dumps(sorted(block.buses)),
                "load_kw": block.load_kw,
                "voltage_class_kv": block.voltage_class_kv,
                "phi_max": block.phi_max,
                "phase_set_json": json.dumps(sorted(block.phase_set)),
                "has_substation_pcc": block.has_substation_pcc,
                "gfm_eligible_der_ids_json": json.dumps(list(block.gfm_eligible_der_ids)),
                "voltage_source_reachability": block.voltage_source_reachability,
                "bounding_switch_ids_json": json.dumps(sorted(bounding_switches.get(frozenset(block.buses), set()))),
                "schema_version": switch_bounded_load_blocks_schema_version,
            }
        )
    return rows


def inject_block_ids_into_onm_settings(
    onm_settings: dict,
    *,
    blocks: tuple[Block, ...],
    buses: pd.DataFrame | None = None,
    loads: pd.DataFrame | None = None,
    location_id: str,
) -> dict:
    """Add per-bus and per-load ``block_id`` and per-block ``microgrid``
    sections to the PowerModelsONM settings sidecar.

    RPOP requires block_id propagation so its block-state variable is
    consistent with the planning-time partition.
    """
    feeder_by_bus = _feeder_by_bus(buses) if buses is not None else {}
    bus_to_block_id: dict[str, str] = {}
    microgrid_section: dict[str, dict] = {}
    for block in blocks:
        feeder_id = _modal_feeder(block.buses, feeder_by_bus)
        block_id = _stable_block_id(location_id=location_id, feeder_id=feeder_id, buses=block.buses)
        microgrid_section[block_id] = {
            "buses": sorted(block.buses),
            "load_kw": block.load_kw,
            "voltage_class_kv": block.voltage_class_kv,
            "phi_max": block.phi_max,
            "phase_set": sorted(block.phase_set),
            "has_substation_pcc": block.has_substation_pcc,
            "gfm_eligible_der_ids": list(block.gfm_eligible_der_ids),
            "voltage_source_reachability": block.voltage_source_reachability,
        }
        for bus in block.buses:
            bus_to_block_id[bus] = block_id

    settings = onm_settings.setdefault("settings", {})
    settings["microgrid"] = microgrid_section

    if loads is not None and not loads.empty:
        load_section = settings.setdefault("load", {})
        for row in loads.itertuples(index=False):
            block_id = bus_to_block_id.get(str(row.bus))
            if not block_id:
                continue
            load_key = str(row.load_name)
            entry = load_section.setdefault(load_key, {})
            entry["block_id"] = block_id

    return onm_settings


def build_blocks(
    *,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    loads: pd.DataFrame,
    sources: pd.DataFrame,
    switches: pd.DataFrame,
    der_inventory: pd.DataFrame | None = None,
    transformers: pd.DataFrame | None = None,
    location_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Notebook-facing builder for validated switch-bounded load blocks.

    This keeps the notebook workflow readable while preserving the stricter
    module boundary: derive blocks, convert them to artifact rows, and return
    a short validation report that can be written next to the Parquet output.
    """

    transformer_rows = (
        transformers
        if transformers is not None
        else pd.DataFrame(columns=["transformer_name", "winding_buses"])
    )
    block_objects = derive_validated_blocks(
        buses=buses,
        lines=lines,
        transformers=transformer_rows,
        sources=sources,
        loads=loads,
        controllable_switches=switches,
        der_inventory=der_inventory,
    )
    rows = build_block_artifact_rows(
        block_objects,
        location_id=location_id,
        buses=buses,
        controllable_switches=switches,
    )
    blocks = pd.DataFrame(rows)
    summary = {
        "block_count": int(len(blocks)),
        "total_load_kw": float(blocks["load_kw"].sum()) if not blocks.empty else 0.0,
        "max_bus_count": int(blocks["bus_count"].max()) if not blocks.empty else 0,
        "voltage_source_reachability": (
            blocks["voltage_source_reachability"].value_counts().sort_index().to_dict()
            if not blocks.empty
            else {}
        ),
    }
    report = {
        "schema_version": switch_bounded_load_blocks_schema_version,
        "violations": [],
        "summary": summary,
    }
    return blocks, report


# --- internal helpers for artifact emission ---


def _feeder_by_bus(buses: pd.DataFrame) -> dict[str, str]:
    if buses is None or buses.empty or "feeder_id" not in buses.columns:
        return {}
    return {str(row.bus): str(row.feeder_id) for row in buses.itertuples(index=False)}


def _modal_feeder(bus_set, feeder_by_bus: dict[str, str]) -> str:
    if not feeder_by_bus:
        return "unknown"
    counts: dict[str, int] = {}
    for bus in bus_set:
        feeder = feeder_by_bus.get(bus)
        if feeder is None:
            continue
        counts[feeder] = counts.get(feeder, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _stable_block_id(*, location_id: str, feeder_id: str, buses) -> str:
    digest = hashlib.sha1("|".join(sorted(buses)).encode("utf-8")).hexdigest()[:12]
    feeder_token = re.sub(r"[^A-Za-z0-9_]+", "_", feeder_id).strip("_")
    return f"{location_id}:block:{feeder_token}:{digest}"


def _bounding_switches_by_block_buses(
    blocks, controllable_switches: pd.DataFrame | None
) -> dict[frozenset, set[str]]:
    if controllable_switches is None or controllable_switches.empty:
        return {}
    out: dict[frozenset, set[str]] = {frozenset(b.buses): set() for b in blocks}
    bus_to_block: dict[str, frozenset] = {}
    for block in blocks:
        key = frozenset(block.buses)
        for bus in block.buses:
            bus_to_block[bus] = key
    for row in controllable_switches.itertuples(index=False):
        for bus in (str(row.from_bus), str(row.to_bus)):
            key = bus_to_block.get(bus)
            if key is None:
                continue
            out[key].add(str(row.switch_id))
    return out


def _raise_if_source_less_blocks_lack_switch_reachable_source_path(
    blocks: list[Block],
    controllable_switches: pd.DataFrame | None,
) -> None:
    """Hard gate: source-less blocks must be switch-reachable to a source."""

    source_indices = {
        idx
        for idx, block in enumerate(blocks)
        if block.has_substation_pcc or block.gfm_eligible_der_ids
    }
    source_less_indices = set(range(len(blocks))) - source_indices
    if not source_less_indices:
        return
    if not source_indices:
        raise BlockInvariantViolation(
            "invariant_g_switch_reachable: no source-hosting block exists"
        )

    bus_to_block_index: dict[str, int] = {}
    for idx, block in enumerate(blocks):
        for bus in block.buses:
            bus_to_block_index[str(bus)] = idx

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(blocks))}
    if controllable_switches is not None and not controllable_switches.empty:
        for row in controllable_switches.itertuples(index=False):
            left = bus_to_block_index.get(str(row.from_bus))
            right = bus_to_block_index.get(str(row.to_bus))
            if left is None or right is None or left == right:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)

    reachable = set(source_indices)
    queue: deque[int] = deque(source_indices)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in reachable:
                continue
            reachable.add(neighbor)
            queue.append(neighbor)

    unreachable = sorted(source_less_indices - reachable)
    if unreachable:
        sample = unreachable[0]
        raise BlockInvariantViolation(
            "invariant_g_switch_reachable: source-less block has no "
            "Controllable Switch path to a source-hosting block "
            f"(block_index={sample}, buses={sorted(blocks[sample].buses)})"
        )


_gfm_flag_columns = ("gfm_capable", "grid_forming_eligible")


def _gfm_eligible_by_bus(der_inventory: pd.DataFrame | None) -> dict[str, list[str]]:
    """Group GFM-eligible DER IDs by host bus. Empty when DER inventory absent.

    Accepts either ``gfm_capable`` (Marshfield Stage B schema) or
    ``grid_forming_eligible`` (generic) as the flag column.
    """
    if der_inventory is None or der_inventory.empty:
        return {}
    flag_col = next((c for c in _gfm_flag_columns if c in der_inventory.columns), None)
    if flag_col is None:
        return {}
    gfm = der_inventory[der_inventory[flag_col].fillna(False).astype(bool)]
    out: dict[str, list[str]] = {}
    for row in gfm.itertuples(index=False):
        if pd.isna(row.bus):
            continue
        out.setdefault(str(row.bus), []).append(str(row.der_id))
    return out


def _block_edge_counts(
    lines: pd.DataFrame,
    opened_lines: set[str],
    components: list[set[str]],
    transformers: pd.DataFrame | None = None,
) -> list[int]:
    """Count edges fully contained within each connected component.

    Transformer winding-bus pairs are counted as edges because they are
    network elements and contribute to the spanning-tree check (invariant F).
    """
    bus_to_component: dict[str, int] = {}
    for idx, component in enumerate(components):
        for bus in component:
            bus_to_component[bus] = idx
    counts = [0] * len(components)
    for row in lines.itertuples(index=False):
        if str(row.line_name) in opened_lines:
            continue
        c1 = bus_to_component.get(str(row.from_bus))
        c2 = bus_to_component.get(str(row.to_bus))
        if c1 is not None and c1 == c2:
            counts[c1] += 1
    if transformers is not None and not transformers.empty and "winding_buses" in transformers.columns:
        for row in transformers.itertuples(index=False):
            windings = [b.strip() for b in str(row.winding_buses).split(",") if b.strip()]
            unique: list[str] = []
            seen_w: set[str] = set()
            for bus in windings:
                if bus not in seen_w:
                    seen_w.add(bus)
                    unique.append(bus)
            if len(unique) >= 2:
                hub = unique[0]
                for other in unique[1:]:
                    c1 = bus_to_component.get(hub)
                    c2 = bus_to_component.get(other)
                    if c1 is not None and c1 == c2:
                        counts[c1] += 1
    return counts


def voltage_band(kv: float | None) -> str:
    """Bucket a kV value into a coarse band for invariant C.

    SHIFT split-phase secondaries are commonly represented as 0.12 kV per
    winding, with loads connected line-to-line at 0.2078 kV; both are part
    of the same physical LV secondary network. Bucketing into LV / MV / HV
    keeps invariant C robust to that representation choice while still
    catching genuine line edges that cross between MV primary and LV
    secondary outside of a transformer.
    """
    if kv is None:
        return "unknown"
    if kv < 1.0:
        return "LV"
    if kv < 69.0:
        return "MV"
    return "HV"


def _line_voltage_transitions(
    lines: pd.DataFrame, opened_lines: set[str], voltage_class: dict[str, float]
) -> list[tuple[str, str]]:
    """Return line edges (excluding opened) whose endpoints disagree in
    voltage class -- a data error per invariant C.

    Cross-band edges (LV-MV, MV-HV, LV-HV) are violations. Intra-band
    variation (e.g. 12.47 L-L vs 7.2 L-N on the same primary feeder; 0.12
    winding vs 0.2078 line-to-line on the same secondary) is allowed
    because both representations describe the same physical system.
    """
    violations: list[tuple[str, str]] = []
    for row in lines.itertuples(index=False):
        if str(row.line_name) in opened_lines:
            continue
        u, v = str(row.from_bus), str(row.to_bus)
        bu = voltage_band(voltage_class.get(u))
        bv = voltage_band(voltage_class.get(v))
        if bu == "unknown" or bv == "unknown":
            continue
        if bu != bv:
            violations.append((u, v))
    return violations


def _bus_phase_counts(lines: pd.DataFrame, opened_lines: set[str]) -> dict[str, int]:
    """Per-bus phase count: max phases over lines incident to the bus.

    Lines opened by Controllable Switches are excluded because the block
    partition assumes those switches are open.
    """
    counts: dict[str, int] = {}
    for row in lines.itertuples(index=False):
        if str(row.line_name) in opened_lines:
            continue
        try:
            phases = int(row.phases)
        except (TypeError, ValueError):
            phases = 0
        for bus in (str(row.from_bus), str(row.to_bus)):
            if phases > counts.get(bus, 0):
                counts[bus] = phases
    return counts


def _loads_by_bus(buses: pd.DataFrame) -> dict[str, float]:
    if "load_kw" not in buses.columns:
        return {}
    return {
        str(row.bus): float(row.load_kw or 0.0)
        for row in buses.itertuples(index=False)
    }
