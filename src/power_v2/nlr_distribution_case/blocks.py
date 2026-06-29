"""Switch-bounded load-block derivation and invariant checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import pandas as pd



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
