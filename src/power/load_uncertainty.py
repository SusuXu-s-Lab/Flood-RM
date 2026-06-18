"""Per-load uncertainty bounds for robust restoration optimization.

Implements the per-(load, timestep) uncertainty set consumed by the two-stage
Robust Partitioning and Operation Problem (RPOP). The uncertainty model defines

    s_d^{phi} = s_d^{0,phi} + u_d^{phi},     underline{s_d^{phi}} <= s_d^{phi} <= overline{s_d^{phi}}

i.e. each load's realized demand lies in a per-load band around a nominal
demand. The implementation supports spatially clustered load uncertainty:
loads in the same cluster respond simultaneously to the worst-case realization.
This reduces the cardinality of the inner max problem from one realization per
load/phase to one realization per cluster.

This module assembles the bound table the master/subproblem consumes:

* ``nominal_kw`` comes from the event-windowed load profile produced by
  ``dft.power.event_window.slice_annual_profile_to_event_window`` after
  diversity-preserving nodal assignment in
  ``dft.power.nodal_load_profiles.assign_nodal_load_profiles``.
* The default symmetric band is ±20%, matching the current stress-test policy.
* The cluster id defaults to the Switch-Bounded Load Block id
  (``simulated_data_protocol.md`` §"switch_bounded_load_blocks"), so loads
  inside the same block share a cluster and avoid inventing a new spatial
  grouping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LOAD_UNCERTAINTY_SCHEMA_VERSION = "marshfield_load_uncertainty.v0.1"
DEFAULT_LOAD_UNCERTAINTY_BAND_FRACTION = 0.20


def build_load_uncertainty_bounds(
    nominal_windows: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    mc_draw: int,
    band_fraction: float = DEFAULT_LOAD_UNCERTAINTY_BAND_FRACTION,
) -> list[dict[str, Any]]:
    """Build per-(load, timestep) uncertainty bounds for the RPOP solver.

    Each ``nominal_windows`` entry must carry ``load_asset_id``, ``block_id``,
    ``feeder_id``, and ``values`` (a sequence of nominal kW values, one per
    optimization timestep). Output rows expand to one record per
    (load_asset_id, timestep).
    """

    if band_fraction < 0.0:
        raise ValueError("band_fraction must be non-negative")

    rows: list[dict[str, Any]] = []
    for window in nominal_windows:
        load_asset_id = str(window["load_asset_id"])
        cluster_id = str(window["block_id"])
        feeder_id = str(window.get("feeder_id", ""))
        for timestep, nominal_kw in enumerate(window["values"]):
            nominal = float(nominal_kw)
            rows.append(
                {
                    "event_id": event_id,
                    "mc_draw": mc_draw,
                    "timestep": timestep,
                    "load_asset_id": load_asset_id,
                    "cluster_id": cluster_id,
                    "feeder_id": feeder_id,
                    "nominal_kw": nominal,
                    "lower_kw": nominal * (1.0 - band_fraction),
                    "upper_kw": nominal * (1.0 + band_fraction),
                    "band_fraction": band_fraction,
                    "schema_version": LOAD_UNCERTAINTY_SCHEMA_VERSION,
                }
            )
    return rows
