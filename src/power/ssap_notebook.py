"""Notebook orchestration helpers for the SSAP switch-synthesis cell.

Keeps data-frame wrangling, OpenDSS naming, and provenance JSON out of the
notebook so the math cell can read like an optimization-problem spec.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from power.ssap import (
    CONTROLLABLE_SWITCHES_SCHEMA_VERSION,
    PLACEMENT_RULE_SSAP,
    RootedFeeder,
    SsapEdge,
    SsapCandidatePolicy,
    SsapGlobalMarginalSelection,
    SsapFrontierPoint,
    build_rooted_feeders_from_tables,
    emit_ssap_switch_rows,
    homogenize_feeder_exposure,
)


SWITCH_ARTIFACT_COLUMNS: tuple[str, ...] = (
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


def derive_lateral_fuses(lines: pd.DataFrame) -> pd.DataFrame:
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


def build_ssap_components(
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


def assemble_switch_artifact(
    selection: SsapGlobalMarginalSelection,
    frontiers: dict[str, tuple[SsapFrontierPoint, ...]],
    component_meta: list[ComponentMeta],
    physical_lines: pd.DataFrame,
    *,
    sandbox_id: str,
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
            solution, feeder_id=component_id, sandbox_id=sandbox_id
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
                    "schema_version": CONTROLLABLE_SWITCHES_SCHEMA_VERSION,
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
                "placement_rule": PLACEMENT_RULE_SSAP,
            }
        )

    switches = pd.DataFrame(switch_rows, columns=list(SWITCH_ARTIFACT_COLUMNS))
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
