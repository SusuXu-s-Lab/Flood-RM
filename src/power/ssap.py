"""Sectionalizing Switch Allocation Problem (SSAP) solver."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional


PLACEMENT_RULE_SSAP = "ssap_radial_sectionalizing_switch_allocation"
SSAP_SUBRULE = "galias_usberti_ssap"
CONTROLLABLE_SWITCHES_SCHEMA_VERSION = "stage_b_controllable_switches.v0.1"

SSAP_CITATIONS: tuple[str, ...] = (
    "Levitin, Mazal-Tov, Elmakis 1995, DOI 10.1016/0378-7796(95)01002-5",
    "Billinton, Jonnavithula 1996, DOI 10.1109/61.517529",
    "Galias 2019, DOI 10.1109/TPWRS.2019.2909836",
    "Usberti et al. 2025, DOI 10.1016/j.epsr.2025.112016",
)

RESILIENCE_SWITCH_PLACEMENT_CITATIONS: tuple[str, ...] = (
    "Fayyazi, Azad-Farsani, Haghighi 2024, DOI 10.1016/j.ress.2023.109919",
)

TWO_TIER_SWITCH_CITATIONS: tuple[str, ...] = (
    "IEEE Std 1366-2022 (Distribution Reliability Indices)",
    "IEEE Std 1547-2018 (DER Interconnection)",
    "Brown 2017, Electric Power Distribution Reliability, 3rd ed. Ch. 5-6",
    "EPRI 1024101, Distribution Automation Reliability Practices",
)


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


def solve_ssap_per_feeder(
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
    """Bottom-up DP state for the Usberti 2025 / Galias 2019 SSAP tree DP.

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
    """Polynomial-time exact SSAP via Usberti 2025 / Galias 2019 tree DP.

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
    """Solve SSAP after applying RPOP/DNMG candidate-admissibility gates.

    Usberti et al. 2025 provides the exact fixed-budget SSAP optimizer; the
    policy gates are caller-supplied inputs that keep candidates aligned with
    Moring-style switch-bounded blocks before the exact solve runs. They are
    not flood-risk terms, geographic spacing rules, or changes to the SSAP
    objective.
    """

    policy = policy or SsapCandidatePolicy()
    candidate_edges = tuple(feeder.edges) if eligible_edges is None else tuple(eligible_edges)
    switchable_edges = select_policy_candidate_edges(feeder, candidate_edges, policy)
    return solve_ssap_per_feeder(
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

    SAP_2025/Usberti takes the number of available switches as an input.
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
        solution = solve_ssap_per_feeder(
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
    frontier components (Abiri-Jahromi et al. 2012): each component should
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

    Marshfield utility-realistic switch density per IEEE 1366 / EPRI
    1024101 / Brown 2017: ~20-60 automated (SCADA) plus 150-400+ manual
    operable devices on a town of Marshfield's scale. Both tiers still
    export as OpenDSS ``Line ... Switch=Yes`` so PowerModelsDistribution
    recognises them as native switches; only PowerModelsONM's RPOP
    optimisation uses the ``dispatchable`` flag.

    Citations are added to row provenance as ``two_tier_citations``.
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
        new_row["two_tier_citations"] = list(TWO_TIER_SWITCH_CITATIONS)
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

    Implements the SAIDI-style weighting introduced by Billinton &
    Jonnavithula (1996): per-zone outage cost is proportional to customer
    count, not just connected kW. Marshfield exposes the same blend via

        L(bus) = kw_weight * load_kw(bus) + customer_weight * n_customers(bus)

    Pure kW recovers when ``customer_weight=0``; pure SAIDI recovers when
    ``kw_weight=0``. The blend is a caller-supplied input to the existing
    SSAP solver, not a change to its objective.

    Coastal Marshfield circuits (Ocean Bluff, Brant Rock, Green Harbor,
    Rexhame) have dense small-lot service drops with low kW per customer;
    raising ``customer_weight`` increases their effective load and biases
    global marginal-benefit selection toward more switches in those
    feeders without introducing flood, FEMA, SFINCS, or ERAD signals.

    Citation: Billinton, Jonnavithula 1996, DOI 10.1109/61.517529.
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


def _edge_satisfies_candidate_policy(
    feeder: RootedFeeder, edge: SsapEdge, policy: SsapCandidatePolicy
) -> bool:
    metrics = _single_switch_metrics(feeder)
    return _edge_satisfies_candidate_metrics(edge, metrics[edge], policy)


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
    """Compute the Moring (2025) static Switch-Bounded Load Blocks.

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
    sandbox_id: str,
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
        switch_id = f"{sandbox_id}:asset:controllable_switches:{feeder_token}:{bus_slug}"
        provenance = {
            "placement_rule": PLACEMENT_RULE_SSAP,
            "sub_rule": SSAP_SUBRULE,
            "feeder_id": feeder_id,
            "k_switches": len(solution.switch_edges),
            "objective_value": solution.objective_value,
            "edge_exposure": switch_edge.exposure,
            "citations": list(SSAP_CITATIONS)
            + list(RESILIENCE_SWITCH_PLACEMENT_CITATIONS),
        }
        rows.append(
            {
                "sandbox_id": sandbox_id,
                "switch_id": switch_id,
                "from_bus": switch_edge.upstream,
                "to_bus": switch_edge.downstream,
                "switch_role": "sectionalizing",
                "normal_state": "closed",
                "initial_state": "closed",
                "dispatchable": True,
                "status": "enabled",
                "placement_rule": PLACEMENT_RULE_SSAP,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": CONTROLLABLE_SWITCHES_SCHEMA_VERSION,
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
