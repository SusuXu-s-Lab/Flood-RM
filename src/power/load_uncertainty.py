"""Per-load uncertainty bounds for robust restoration optimization.

Implements the per-(load, timestep) uncertainty set consumed by the two-stage
Robust Partitioning and Operation Problem (RPOP) of Moring et al. (2025),
"Reconfiguration and Real-Time Operation of Networked Microgrids Under Load
Uncertainty," arXiv:2504.15084v2. The paper's eqn (1) defines

    s_d^{phi} = s_d^{0,phi} + u_d^{phi},     underline{s_d^{phi}} <= s_d^{phi} <= overline{s_d^{phi}}

i.e. each load's realized demand lies in a per-load band around a nominal
demand. Improvement (3) in Section I of the paper is *spatial clustering of
load uncertainty* (cluster index gamma in Γ, eqn 12): loads in the same
cluster respond simultaneously to the worst-case realization. This reduces
the cardinality of the inner max problem from 2^{|D × Φ|} to 2^{|Γ|}.

This module assembles the bound table the master/subproblem consumes:

* ``nominal_kw`` comes from the event-windowed load profile produced by
  ``dft.power.event_window.slice_annual_profile_to_event_window`` after
  diversity-preserving nodal assignment in
  ``dft.power.nodal_load_profiles.assign_nodal_load_profiles``.
* The default symmetric band is ±20%, matching the Moring et al. Section V
  case study (Fig. 5 caption, "20% load uncertainty"; T=1..60 at 90%, T=61..120
  at 100%, T=121..180 at 120% of nominal).
* The cluster id defaults to the Switch-Bounded Load Block id
  (``simulated_data_protocol.md`` §"switch_bounded_load_blocks"), so loads
  inside the same block share a cluster — this aligns the paper's eqn (12)
  cluster set Γ with this codebase's existing Stage B Candidate Control
  Units and avoids inventing a new spatial grouping.

Citation:
    Moring, H., Poolla, B. K., Nagarajan, H., Mathieu, J. L., Bernstein, A.,
    and Fobes, D. M. "Reconfiguration and Real-Time Operation of Networked
    Microgrids Under Load Uncertainty." arXiv:2504.15084v2, 2025. Local copy
    at ``docs/Reconfiguration and Real-Time Operation of Networked
    Microgrids Under Load Uncertainty - 2504.15084v2.pdf``.
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
