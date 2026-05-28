"""Switch-review figures for the configured grid dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from power.paths import POWER_GRID


DEFAULT_REGISTRY_DIR = POWER_GRID / "asset_registry"
DEFAULT_SMART_DS_COMPAT_DIR = POWER_GRID / "augmented"
DEFAULT_FIGURE_PATH = POWER_GRID / "figures" / "switch_line_overlay.png"

# Tie switches (normally-open, cross-feeder) are kept distinct from
# sectionalizers (normally-closed, intra-feeder) because DNMG reconfiguration
# treats them differently: sectionalizers isolate faulted sections, ties
# reroute load between feeders during restoration.
SECTIONALIZING_SWITCH_COLOR = "#f97316"
TIE_SWITCH_COLOR = "#dc2626"


def build_switch_line_overlay(
    *,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    smart_ds_compat_dir: Path = DEFAULT_SMART_DS_COMPAT_DIR,
    output_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    """Render SFO-style switch line notation for qualitative switch-placement review."""

    buses = pd.read_csv(registry_dir / "buses.csv")
    lines = pd.read_csv(registry_dir / "lines.csv")
    switches = pd.read_parquet(smart_ds_compat_dir / "controllable_switches.parquet")
    switch_role_counts = switches["switch_role"].value_counts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bus_xy = {
        row.bus: (float(row.lon), float(row.lat))
        for row in buses.dropna(subset=["lon", "lat"]).itertuples(index=False)
    }
    line_xy = {
        row.line_name: ((float(row.from_lon), float(row.from_lat)), (float(row.to_lon), float(row.to_lat)))
        for row in lines.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"]).itertuples(index=False)
    }

    base_segments = [
        ((float(row.from_lon), float(row.from_lat)), (float(row.to_lon), float(row.to_lat)))
        for row in lines.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"]).itertuples(index=False)
    ]
    sectionalizing_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    tie_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    zero_length_ties = 0

    for row in switches.itertuples(index=False):
        if row.switch_role == "sectionalizing" and bool(row.opens_existing_line):
            segment = line_xy.get(str(row.associated_line_name))
            if segment is None:
                segment = _bus_segment(bus_xy, str(row.from_bus), str(row.to_bus))
            if segment is not None:
                sectionalizing_segments.append(segment)
        elif row.switch_role == "tie":
            segment = _bus_segment(bus_xy, str(row.from_bus), str(row.to_bus))
            if segment is None:
                continue
            if segment[0] == segment[1]:
                zero_length_ties += 1
                segment = _tiny_tick(segment[0])
            tie_segments.append(segment)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(LineCollection(base_segments, colors="#486b7d", linewidths=0.35, alpha=0.40))
    if sectionalizing_segments:
        ax.add_collection(LineCollection(sectionalizing_segments, colors=SECTIONALIZING_SWITCH_COLOR, linewidths=1.2, alpha=0.95))
    if tie_segments:
        ax.add_collection(LineCollection(tie_segments, colors=TIE_SWITCH_COLOR, linewidths=1.6, alpha=0.95))
    ax.scatter(buses["lon"], buses["lat"], s=0.45, c="#1f2933", alpha=0.28, linewidths=0)
    ax.set_title(
        "Grid Dataset: SFO-style switch line overlay\n"
        f"{len(lines):,} plotted lines, {len(buses):,} buses, {len(switches):,} controllable switches"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#486b7d", lw=1.4, label="Line"),
            Line2D([0], [0], color=SECTIONALIZING_SWITCH_COLOR, lw=1.8, label="Sectionalizing switch (NC)"),
            Line2D([0], [0], color=TIE_SWITCH_COLOR, lw=1.8, label="Tie switch (NO)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f2933", markersize=4, label="Bus"),
        ],
        loc="lower left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "output_path": str(output_path),
        "line_count": len(lines),
        "bus_count": len(buses),
        "switch_count": len(switches),
        "sectionalizing_switch_count": int(switch_role_counts.get("sectionalizing", 0)),
        "tie_switch_count": int(switch_role_counts.get("tie", 0)),
        "sectionalizing_switch_segments_plotted": len(sectionalizing_segments),
        "tie_switch_segments_plotted": len(tie_segments),
        "zero_length_tie_ticks": zero_length_ties,
    }


def _bus_segment(
    bus_xy: dict[str, tuple[float, float]],
    from_bus: str,
    to_bus: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if from_bus not in bus_xy or to_bus not in bus_xy:
        return None
    return bus_xy[from_bus], bus_xy[to_bus]


def _tiny_tick(point: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    lon, lat = point
    delta = 0.00018
    return (lon - delta, lat - delta), (lon + delta, lat + delta)
