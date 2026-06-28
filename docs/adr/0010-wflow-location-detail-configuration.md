# Wflow Location Detail Configuration

## Status

Accepted.

## Context

Austin and Greensboro need Wflow-coupled inland/fluvial truth sets while the
existing Marshfield reference workflow keeps SFINCS-specific build and scenario
settings in `sfincs.yaml`. Wflow has its own HydroMT build/update YAML,
HydroMT data catalog needs, model roots, gauge/output handoff settings, and
runner settings, so hiding it inside `sfincs.yaml` would blur the boundary
between hydrologic routing and hydrodynamic staging.

## Decision

Use `wflow.yaml` as a separate **Location Detail Configuration** for
Wflow-coupled Study Locations.

`sfincs.yaml` continues to own SFINCS model construction, scenario staging, and
SFINCS runtime settings. `wflow.yaml` owns HydroMT-Wflow model setup/update
paths, Wflow roots, event-forcing update settings, outlet/gauge output
configuration, and Wflow runner settings. The Event Catalog remains the source
of streamflow frequency provenance; Wflow remains the hydrologic bridge that
produces routed discharge forcing for SFINCS.

## Consequences

- Austin and Greensboro Location Workspaces should include `includes.wflow:
  wflow.yaml` when they become configured Study Locations.
- The Location Definition Interface must learn to load optional `wflow.yaml`
  without requiring it for coastal reference locations such as Marshfield.
- Notebook text and reusable modules should call this a Wflow-coupled build,
  not a SFINCS hydrology option.
