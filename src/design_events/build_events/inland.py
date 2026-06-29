"""Compatibility shim — inland streamgage-network path + Wflow handoff now live in design_events_v2.inland.

ADR-0021 convergence: the ADR-0017 external-boundary fluvial streamgage event builder,
the USGS POT/decluster member construction, and the Wflow handoff manifest writer moved to
``design_events_v2.inland``. Re-exported here so notebook and builder imports keep resolving.
"""
from __future__ import annotations

from design_events_v2.inland import (
    InlandEventArtifacts,
    build_inland_event_artifacts,
    build_usgs_streamflow_event_members,
    decluster_streamflow_network_events,
    load_usgs_streamflow_records,
    streamflow_event_members_path,
    streamflow_pot_candidate_peaks,
    streamflow_records_path,
    write_handoff,
    write_streamflow_event_members,
)

__all__ = [
    "InlandEventArtifacts",
    "build_inland_event_artifacts",
    "build_usgs_streamflow_event_members",
    "decluster_streamflow_network_events",
    "load_usgs_streamflow_records",
    "streamflow_event_members_path",
    "streamflow_pot_candidate_peaks",
    "streamflow_records_path",
    "write_handoff",
    "write_streamflow_event_members",
]
