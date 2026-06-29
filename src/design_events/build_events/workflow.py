"""Compatibility shim — Event Catalog workflow surface now lives in design_events_v2.

ADR-0021 convergence: path/plan resolution and config policies moved to
``design_events_v2.runtime``; the notebook runtime + catalog/replay materialization moved
to ``design_events_v2.workflow``; catalog assembly to ``design_events_v2.catalog``. This
module re-exports the notebook-forward surface and keeps the cross-package SFINCS bridges
(``build_timeseries``, ``write_joint_handoff``) that hand off to ``sfincs_runs``.
"""
from __future__ import annotations

from design_events_v2.runtime import (
    EventCatalogPlan,
    EventForcingPlan,
    build_paths,
    plan,
)
from design_events_v2.workflow import (
    EventCatalogNotebookRuntime,
    configure_coastal_dependence_policy,
    configure_coastal_design_event_policy,
    event_catalog_source_inventory,
    load_runtime,
    materialize_inland_catalog_outputs,
    scenario_context,
    _replay_columns,
)
from design_events.build_events.catalog import (
    attach_forcing_members,
    build_event_catalog,
    rebuild_forcing_pairing,
    validate_event_catalog,
    write_event_catalog_audit,
)
from design_events.build_events.inland import (
    InlandEventArtifacts,
    build_inland_event_artifacts,
    build_usgs_streamflow_event_members,
    write_handoff,
)
from design_events.build_events.selection import (
    assign_severity_bands,
    attach_antecedent_soil_moisture,
    select_training,
)
from design_events.build_events.probability import (
    build_inland_catalog,
    build_joint_catalog,
    build_tail,
)


# Short notebook-facing API.
build_catalog = build_inland_catalog


def build_timeseries(*args, **kwargs):
    from sfincs_runs.scenarios.coastal_realization import build_timeseries as _build_timeseries

    return _build_timeseries(*args, **kwargs)


def write_joint_handoff(*args, **kwargs):
    from sfincs_runs.scenarios.joint_handoff import write_handoff as _write_handoff

    return _write_handoff(*args, **kwargs)
