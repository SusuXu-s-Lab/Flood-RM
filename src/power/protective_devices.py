"""Lateral-tap fuse placement for synthesized distribution feeders.

A fuse is placed at the head of any line whose endpoint touches at least one
other line with strictly more phases (a phase drop at a bus indicates a lateral
tap off a higher-phase parent). One fuse per phase-drop endpoint per line.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def derive_lateral_fuses(lines: pd.DataFrame) -> pd.DataFrame:
    max_phases_at_bus: dict[str, int] = defaultdict(int)
    for row in lines.itertuples(index=False):
        phases = int(row.phases)
        if phases > max_phases_at_bus[row.from_bus]:
            max_phases_at_bus[row.from_bus] = phases
        if phases > max_phases_at_bus[row.to_bus]:
            max_phases_at_bus[row.to_bus] = phases

    fuses: list[dict[str, object]] = []
    for row in lines.itertuples(index=False):
        phases = int(row.phases)
        for endpoint in (row.from_bus, row.to_bus):
            parent_phases = max_phases_at_bus[endpoint]
            if parent_phases > phases:
                fuses.append(
                    {
                        "fuse_id": f"fuse_{row.line_name}_{endpoint}",
                        "feeder_id": row.feeder_id,
                        "line_name": row.line_name,
                        "head_bus": endpoint,
                        "child_phases": phases,
                        "parent_phases": parent_phases,
                    }
                )
                break

    return pd.DataFrame(
        fuses,
        columns=["fuse_id", "feeder_id", "line_name", "head_bus", "child_phases", "parent_phases"],
    )
