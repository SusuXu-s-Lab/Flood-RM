# Separate SMART-DS-Compatible Interface and ONM Export Stages

## Status

Accepted.

## Context

The `src/power` architecture review reorganized the package into the five Grid
Dataset Stages that were previously declared in the stage-aligned layout. That
raised the question of whether the two export stages —
`stage_a` (**SMART-DS-Compatible Interface**: `assets`, `control_units`,
`asset_states`, `telemetry_observations` parquet) and
`onm` (**ONM/RPOP-ready export**: PowerModelsONM network/settings, the
event-window bundle, `events.json`) — should collapse into one "export" module,
since together they form the export tail of the Grid Dataset pipeline.

## Decision

Keep them as **two separate Grid Dataset Stages**.

They are **artifact-coupled, not code-coupled**: the `onm` modules consume
`stage_a`'s parquet outputs as *data files* at runtime and import none of
`stage_a`'s code (and vice versa). The only intra-stage code edge is
`export_stage_a2` importing `DEFAULT_OUTPUT_DIR` from `export_stage_a1` — same
stage.

Merging would relocate files without concentrating any shared complexity (it
fails the deletion test) and would blur two distinct domain stages that have
distinct outputs, schemas, and downstream consumers.

## Consequences

- Historical note: `stage_a` and `onm` were kept distinct packages under the
  earlier stage-letter layout.
- The `onm` stage has a single live export path:
  `event_window.build_event_window_bundle` + `powermodels_onm_export`
  (+ `onm_export` rendering). The superseded `onm_run` orchestrator, which had
  no callers, was removed.
- `onm_events` (PMONM `events.json` / DNMG contingency events) remains available
  but is not yet wired into the live bundle; its module docstring records the
  gap. Wiring it in does not require merging the stages.
- A future reviewer should not re-propose merging `stage_a` and `onm`; the
  decoupling is intentional.
